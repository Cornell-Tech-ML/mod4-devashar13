# type: ignore
# Currently pyright doesn't support numba.cuda

from typing import Callable, Optional, TypeVar, Any

import numba
from numba import cuda
from numba.cuda import jit as _jit
from .tensor import Tensor
from .tensor_data import (
    MAX_DIMS,
    Shape,
    Storage,
    Strides,
    TensorData,
    broadcast_index,
    index_to_position,
    shape_broadcast,
    to_index,
)
from .tensor_ops import MapProto, TensorOps

FakeCUDAKernel = Any

# This code will CUDA compile fast versions of your tensor_data functions.
# If you get an error, read the docs for NUMBA to understand what is allowed
# in these functions.

Fn = TypeVar("Fn")


def device_jit(fn: Fn, **kwargs: Any) -> Fn:
    """Compiles a Python function for device (GPU) execution using Numba's JIT compiler.

    Args:
    ----
        fn (Fn): The function to compile.
        **kwargs (Any): Additional keyword arguments for the JIT compiler.

    Returns:
    -------
        Fn: The compiled device function.

    """
    return _jit(device=True, **kwargs)(fn)  # type: ignore


def jit(fn: Fn, **kwargs: Any) -> FakeCUDAKernel:
    """Compiles a Python function for device (GPU) execution using Numba's JIT compiler.

    Args:
    ----
        fn (Fn): The function to compile.
        **kwargs (Any): Additional keyword arguments for the JIT compiler.

    Returns:
    -------
        FakeCUDAKernel: The compiled CUDA kernel.

    """
    return _jit(**kwargs)(fn)  # type: ignore


# JIT compile the utility functions for device usage
to_index = device_jit(to_index)
index_to_position = device_jit(index_to_position)
broadcast_index = device_jit(broadcast_index)

# Define the number of threads per block for CUDA kernels
THREADS_PER_BLOCK = 32


class CudaOps(TensorOps):
    """CUDA-accelerated implementations of tensor operations using Numba's CUDA JIT compiler."""

    cuda = True

    @staticmethod
    def map(fn: Callable[[float], float]) -> MapProto:
        """Creates a CUDA-accelerated map function that applies a given unary function to each element of a tensor.

        Args:
        ----
            fn (Callable[[float], float]): A function that maps a single float to another float.

        Returns:
        -------
            MapProto: A function that applies `fn` to each element of a tensor.

        """
        cufn: Callable[[float], float] = device_jit(fn)
        f = tensor_map(cufn)

        def ret(a: Tensor, out: Optional[Tensor] = None) -> Tensor:
            """Applies the mapped function to tensor `a` and stores the result in `out`.

            Args:
            ----
                a (Tensor): The input tensor.
                out (Optional[Tensor], optional): The output tensor. If not provided, a new tensor with the same shape as `a` is created.

            Returns:
            -------
                Tensor: The output tensor with the function applied.

            """
            if out is None:
                out = a.zeros(a.shape)

            # Calculate grid dimensions to cover all elements
            threadsperblock = THREADS_PER_BLOCK
            blockspergrid = (out.size + THREADS_PER_BLOCK - 1) // THREADS_PER_BLOCK

            # Launch the CUDA kernel
            f[blockspergrid, threadsperblock](*out.tuple(), out.size, *a.tuple())  # type: ignore
            return out

        return ret

    @staticmethod
    def zip(fn: Callable[[float, float], float]) -> Callable[[Tensor, Tensor], Tensor]:
        """Creates a CUDA-accelerated zip function that applies a given binary function to pairs of elements from two tensors.

        Args:
        ----
            fn (Callable[[float, float], float]): A function that maps two floats to a single float.

        Returns:
        -------
            Callable[[Tensor, Tensor], Tensor]: A function that applies `fn` to pairs of elements from two tensors.

        """
        cufn: Callable[[float, float], float] = device_jit(fn)
        f = tensor_zip(cufn)

        def ret(a: Tensor, b: Tensor) -> Tensor:
            """Applies the zipped function to tensors `a` and `b`, storing the result in a new tensor.

            Args:
            ----
                a (Tensor): The first input tensor.
                b (Tensor): The second input tensor.

            Returns:
            -------
                Tensor: The output tensor with the function applied to each pair of elements.

            """
            c_shape = shape_broadcast(a.shape, b.shape)
            out = a.zeros(c_shape)
            threadsperblock = THREADS_PER_BLOCK
            blockspergrid = (out.size + (threadsperblock - 1)) // threadsperblock

            # Launch the CUDA kernel
            f[blockspergrid, threadsperblock](  # type: ignore
                *out.tuple(), out.size, *a.tuple(), *b.tuple()
            )
            return out

        return ret

    @staticmethod
    def reduce(
        fn: Callable[[float, float], float], start: float = 0.0
    ) -> Callable[[Tensor, int], Tensor]:
        """Creates a CUDA-accelerated reduce function that reduces a tensor along a specified dimension using a given binary function.

        Args:
        ----
            fn (Callable[[float, float], float]): A reduction function that maps two floats to a single float.
            start (float, optional): The initial value for the reduction. Defaults to 0.0.

        Returns:
        -------
            Callable[[Tensor, int], Tensor]: A function that reduces a tensor along a specified dimension.

        """
        cufn: Callable[[float, float], float] = device_jit(fn)
        f = tensor_reduce(cufn)

        def ret(a: Tensor, dim: int) -> Tensor:
            """Reduces tensor `a` along dimension `dim` using the reduction function `fn`.

            Args:
            ----
                a (Tensor): The input tensor.
                dim (int): The dimension along which to reduce.

            Returns:
            -------
                Tensor: The reduced tensor.

            """
            out_shape = list(a.shape)
            out_shape[dim] = (a.shape[dim] - 1) // 1024 + 1
            out_a = a.zeros(tuple(out_shape))

            # Define CUDA kernel launch parameters
            threadsperblock = 1024
            blockspergrid = out_a.size

            # Launch the CUDA kernel
            f[blockspergrid, threadsperblock](  # type: ignore
                *out_a.tuple(), out_a.size, *a.tuple(), dim, start
            )

            return out_a

        return ret

    @staticmethod
    def matrix_multiply(a: Tensor, b: Tensor) -> Tensor:
        """Performs a CUDA-accelerated batched matrix multiplication between tensors `a` and `b`.

        Args:
        ----
            a (Tensor): The first input tensor.
            b (Tensor): The second input tensor.

        Returns:
        -------
            Tensor: The result of the matrix multiplication.

        """
        # Ensure tensors are 3-dimensional for batched operations
        both_2d = 0
        if len(a.shape) == 2:
            a = a.contiguous().view(1, a.shape[0], a.shape[1])
            both_2d += 1
        if len(b.shape) == 2:
            b = b.contiguous().view(1, b.shape[0], b.shape[1])
            both_2d += 1
        both_2d = both_2d == 2

        # Broadcast the batch dimensions
        ls = list(shape_broadcast(a.shape[:-2], b.shape[:-2]))
        ls.append(a.shape[-2])
        ls.append(b.shape[-1])
        assert (
            a.shape[-1] == b.shape[-2]
        ), "Inner dimensions must match for matrix multiplication."
        out = a.zeros(tuple(ls))

        # Define CUDA kernel launch parameters
        blockspergrid = (
            (out.shape[-2] + (THREADS_PER_BLOCK - 1)) // THREADS_PER_BLOCK,
            (out.shape[-1] + (THREADS_PER_BLOCK - 1)) // THREADS_PER_BLOCK,
            out.shape[0],
        )
        threadsperblock = (THREADS_PER_BLOCK, THREADS_PER_BLOCK)

        # Launch the CUDA kernel
        tensor_matrix_multiply[blockspergrid, threadsperblock](
            *out.tuple(), out.size, *a.tuple(), *b.tuple()
        )

        # If the original tensors were 2D, revert the output tensor back to 2D
        if both_2d:
            out = out.view(out.shape[-2], out.shape[-1])
        return out


def tensor_map(
    fn: Callable[[float], float],
) -> Callable[[Storage, Shape, Strides, int, Storage, Shape, Strides], None]:
    """Creates a CUDA-accelerated tensor map function that applies a unary function to each element of a tensor.

    Args:
    ----
        fn (Callable[[float], float]): A function that maps a single float to another float.

    Returns:
    -------
        Callable[[Storage, Shape, Strides, int, Storage, Shape, Strides], None]: The CUDA kernel function.

    """

    @cuda.jit
    def _map(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        out_size: int,
        in_storage: Storage,
        in_shape: Shape,
        in_strides: Strides,
    ) -> None:
        """CUDA kernel for element-wise tensor mapping.

        Args:
        ----
            out (Storage): Output tensor storage.
            out_shape (Shape): Shape of the output tensor.
            out_strides (Strides): Strides of the output tensor.
            out_size (int): Total number of elements in the output tensor.
            in_storage (Storage): Input tensor storage.
            in_shape (Shape): Shape of the input tensor.
            in_strides (Strides): Strides of the input tensor.

        """
        out_index = cuda.local.array(MAX_DIMS, numba.int32)
        in_index = cuda.local.array(MAX_DIMS, numba.int32)
        i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
        if i < out_size:
            # Convert linear index to multi-dimensional index
            to_index(i, out_shape, out_index)
            # Broadcast input index based on tensor shapes
            broadcast_index(out_index, out_shape, in_shape, in_index)
            # Convert multi-dimensional index back to linear position
            in_idx = index_to_position(in_index, in_strides)
            out_pos = index_to_position(out_index, out_strides)
            # Apply the mapping function
            out[out_pos] = fn(in_storage[in_idx])

    return _map


def tensor_zip(
    fn: Callable[[float, float], float],
) -> Callable[
    [Storage, Shape, Strides, Storage, Shape, Strides, Storage, Shape, Strides], None
]:
    """Creates a CUDA-accelerated tensor zip function that applies a binary function to pairs of elements from two tensors.

    Args:
    ----
        fn (Callable[[float, float], float]): A function that maps two floats to a single float.

    Returns:
    -------
        Callable[[Storage, Shape, Strides, Storage, Shape, Strides, Storage, Shape, Strides], None]: The CUDA kernel function.

    """

    @cuda.jit
    def _zip(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        out_size: int,
        a_storage: Storage,
        a_shape: Shape,
        a_strides: Strides,
        b_storage: Storage,
        b_shape: Shape,
        b_strides: Strides,
    ) -> None:
        """CUDA kernel for element-wise tensor zipping.

        Args:
        ----
            out (Storage): Output tensor storage.
            out_shape (Shape): Shape of the output tensor.
            out_strides (Strides): Strides of the output tensor.
            out_size (int): Total number of elements in the output tensor.
            a_storage (Storage): First input tensor storage.
            a_shape (Shape): Shape of the first input tensor.
            a_strides (Strides): Strides of the first input tensor.
            b_storage (Storage): Second input tensor storage.
            b_shape (Shape): Shape of the second input tensor.
            b_strides (Strides): Strides of the second input tensor.

        """
        out_index = cuda.local.array(MAX_DIMS, numba.int32)
        a_index = cuda.local.array(MAX_DIMS, numba.int32)
        b_index = cuda.local.array(MAX_DIMS, numba.int32)
        i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
        if i < out_size:
            # Convert linear index to multi-dimensional index
            to_index(i, out_shape, out_index)
            # Broadcast input indices based on tensor shapes
            broadcast_index(out_index, out_shape, a_shape, a_index)
            broadcast_index(out_index, out_shape, b_shape, b_index)
            # Convert multi-dimensional indices back to linear positions
            a_idx = index_to_position(a_index, a_strides)
            b_idx = index_to_position(b_index, b_strides)
            out_pos = index_to_position(out_index, out_strides)
            # Apply the zip function
            out[out_pos] = fn(a_storage[a_idx], b_storage[b_idx])

    return _zip


def _sum_practice(out: Storage, a: Storage, size: int) -> None:
    """CUDA kernel to perform a block-wise sum of elements in tensor `a`.

    Given an array `a` of length `size`, this kernel sums up `BLOCK_DIM` elements per block and stores the result in `out`.

    Args:
    ----
        out (Storage): Output tensor storage.
        a (Storage): Input tensor storage.
        size (int): Length of the input tensor.

    """
    BLOCK_DIM = 32

    # Define shared memory as float32
    cache = cuda.shared.array(BLOCK_DIM, numba.float32)
    i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    pos = cuda.threadIdx.x

    if i < size:
        cache[pos] = a[i]
    else:
        cache[pos] = 0.0
    cuda.syncthreads()

    offset = 1
    while offset < BLOCK_DIM:
        if pos % (2 * offset) == 0 and pos + offset < BLOCK_DIM:
            cache[pos] += cache[pos + offset]
        offset *= 2
        cuda.syncthreads()

    if pos == 0:
        out[cuda.blockIdx.x] = cache[0]


jit_sum_practice = cuda.jit(_sum_practice)


def sum_practice(a: Tensor) -> TensorData:
    """Performs a sum reduction on tensor `a` using the `_sum_practice` CUDA kernel.

    Args:
    ----
        a (Tensor): The input tensor to be summed.

    Returns:
    -------
        TensorData: The output tensor containing the sum of elements.

    """
    (size,) = a.shape
    threadsperblock = THREADS_PER_BLOCK
    blockspergrid = (size + THREADS_PER_BLOCK - 1) // THREADS_PER_BLOCK
    out_size = (size + THREADS_PER_BLOCK - 1) // THREADS_PER_BLOCK
    out = TensorData([0.0 for _ in range(out_size)], (out_size,))
    out.to_cuda_()
    jit_sum_practice[blockspergrid, threadsperblock](
        out.tuple()[0], a._tensor._storage, size
    )
    return out


def tensor_reduce(
    fn: Callable[[float, float], float],
) -> Callable[
    [Storage, Shape, Strides, int, Storage, Shape, Strides, int, float], None
]:
    """Creates a CUDA-accelerated tensor reduce function that reduces a tensor along a specified dimension using a given binary function.

    Args:
    ----
        fn (Callable[[float, float], float]): A reduction function that maps two floats to a single float.

    Returns:
    -------
        Callable[[Storage, Shape, Strides, int, Storage, Shape, Strides, int, float], None]: The CUDA kernel function.

    """

    @cuda.jit
    def _reduce(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        out_size: int,
        a_storage: Storage,
        a_shape: Shape,
        a_strides: Strides,
        reduce_dim: int,
        reduce_value: float,
    ) -> None:
        """CUDA kernel for tensor reduction.

        Args:
        ----
            out (Storage): Output tensor storage.
            out_shape (Shape): Shape of the output tensor.
            out_strides (Strides): Strides of the output tensor.
            out_size (int): Total number of elements in the output tensor.
            a_storage (Storage): Input tensor storage.
            a_shape (Shape): Shape of the input tensor.
            a_strides (Strides): Strides of the input tensor.
            reduce_dim (int): The dimension along which to reduce.
            reduce_value (float): The initial value for the reduction.

        """
        BLOCK_DIM = 1024
        # Define shared memory as float32
        cache = cuda.shared.array(BLOCK_DIM, numba.float32)
        out_index = cuda.local.array(MAX_DIMS, numba.int32)
        out_pos = cuda.blockIdx.x
        pos = cuda.threadIdx.x

        # Initialize shared memory with reduction value
        cache[pos] = reduce_value

        if out_pos < out_size:
            # Convert linear index to multi-dimensional index
            to_index(out_pos, out_shape, out_index)
            # Calculate the position in the input tensor
            relative_a_index = index_to_position(out_index, a_strides)
            # Calculate thread offset
            thread_offset = a_strides[reduce_dim] * pos
            # Load element into shared memory
            if pos < a_shape[reduce_dim]:
                cache[pos] = a_storage[relative_a_index + thread_offset]
            else:
                cache[pos] = reduce_value
        cuda.syncthreads()

        # Perform parallel reduction
        offset = 1
        while offset < BLOCK_DIM:
            if pos % (2 * offset) == 0 and pos + offset < BLOCK_DIM:
                cache[pos] = fn(cache[pos], cache[pos + offset])
            offset *= 2
            cuda.syncthreads()

        # Write the result to the output tensor
        if pos == 0 and out_pos < out_size:
            out[out_pos] = cache[0]

    return _reduce


def _mm_practice(out: Storage, a: Storage, b: Storage, size: int) -> None:
    """CUDA kernel to perform matrix multiplication between two square matrices `a` and `b`.

    Given that both matrices are of shape `[size, size]` with strides `[size, 1]`, this kernel computes the product and stores it in `out`.

    Args:
    ----
        out (Storage): Output tensor storage for the result matrix.
        a (Storage): Input tensor storage for matrix `a`.
        b (Storage): Input tensor storage for matrix `b`.
        size (int): The size of the square matrices.

    """
    BLOCK_DIM = 32

    # Define shared memory as float32
    a_shared = cuda.shared.array((BLOCK_DIM, BLOCK_DIM), numba.float32)
    b_shared = cuda.shared.array((BLOCK_DIM, BLOCK_DIM), numba.float32)
    tx, ty = cuda.threadIdx.x, cuda.threadIdx.y

    # Load elements into shared memory
    if tx < size and ty < size:
        a_shared[tx, ty] = a[tx * size + ty]
        b_shared[tx, ty] = b[tx * size + ty]
    else:
        a_shared[tx, ty] = 0.0
        b_shared[tx, ty] = 0.0
    cuda.syncthreads()

    if tx < size and ty < size:
        # Compute the dot product
        temp = numba.float32(0.0)
        for k in range(size):
            temp += a_shared[tx, k] * b_shared[k, ty]
        # Store the result
        out[tx * size + ty] = temp


jit_mm_practice = cuda.jit(_mm_practice)


def mm_practice(a: Tensor, b: Tensor) -> TensorData:
    """Performs a matrix multiplication between two square matrices using the `_mm_practice` CUDA kernel.

    Args:
    ----
        a (Tensor): The first input tensor (matrix).
        b (Tensor): The second input tensor (matrix).

    Returns:
    -------
        TensorData: The resulting tensor after matrix multiplication.

    """
    (size, _) = a.shape
    threadsperblock = (THREADS_PER_BLOCK, THREADS_PER_BLOCK)
    blockspergrid = 1
    out = TensorData([0.0 for _ in range(size * size)], (size, size))
    out.to_cuda_()
    jit_mm_practice[blockspergrid, threadsperblock](
        out.tuple()[0], a._tensor._storage, b._tensor._storage, size
    )
    return out


def _tensor_matrix_multiply(
    out: Storage,
    out_shape: Shape,
    out_strides: Strides,
    out_size: int,
    a_storage: Storage,
    a_shape: Shape,
    a_strides: Strides,
    b_storage: Storage,
    b_shape: Shape,
    b_strides: Strides,
) -> None:
    """CUDA kernel to perform tensor matrix multiplication with support for broadcasting.

    This kernel multiplies two tensors `a` and `b` along their last two dimensions and stores the result in `out`. It supports broadcasting on batch dimensions.

    Args:
    ----
        out (Storage): Output tensor storage for the result.
        out_shape (Shape): Shape of the output tensor.
        out_strides (Strides): Strides of the output tensor.
        out_size (int): Total number of elements in the output tensor.
        a_storage (Storage): Input tensor storage for tensor `a`.
        a_shape (Shape): Shape of tensor `a`.
        a_strides (Strides): Strides of tensor `a`.
        b_storage (Storage): Input tensor storage for tensor `b`.
        b_shape (Shape): Shape of tensor `b`.
        b_strides (Strides): Strides of tensor `b`.

    """
    a_batch_stride = a_strides[0] if a_shape[0] > 1 else 0
    b_batch_stride = b_strides[0] if b_shape[0] > 1 else 0
    # Batch dimension - fixed
    batch = cuda.blockIdx.z

    BLOCK_DIM = 32
    # Define shared memory as float32
    a_shared = cuda.shared.array((BLOCK_DIM, BLOCK_DIM), numba.float32)
    b_shared = cuda.shared.array((BLOCK_DIM, BLOCK_DIM), numba.float32)

    # Calculate global row and column indices
    i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    j = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y

    K = a_shape[-1]

    # Local thread indices
    pi = cuda.threadIdx.x
    pj = cuda.threadIdx.y

    # Initialize accumulator for the dot product
    temp = numba.float32(0.0)

    # Loop over all blocks in the K dimension
    for block_idx in range(0, K, BLOCK_DIM):
        # Load a block of A into shared memory
        a_col = block_idx + pj
        if a_col < a_shape[-1] and i < a_shape[-2]:
            a_idx = batch * a_batch_stride + i * a_strides[-2] + a_col * a_strides[-1]
            a_shared[pi, pj] = a_storage[a_idx]
        else:
            a_shared[pi, pj] = 0.0

        # Load a block of B into shared memory
        b_row = block_idx + pi
        if b_row < b_shape[-2] and j < b_shape[-1]:
            b_idx = batch * b_batch_stride + b_row * b_strides[-2] + j * b_strides[-1]
            b_shared[pi, pj] = b_storage[b_idx]
        else:
            b_shared[pi, pj] = 0.0

        # Synchronize to ensure all data is loaded
        cuda.syncthreads()

        # Compute the partial dot product
        if i < a_shape[-2] and j < b_shape[-1]:
            for k in range(BLOCK_DIM):
                if (block_idx + k) < K:
                    temp += a_shared[pi, k] * b_shared[k, pj]

        # Synchronize before loading the next block
        cuda.syncthreads()

    # Write the accumulated result to the output tensor
    if i < out_shape[-2] and j < out_shape[-1]:
        out_idx = batch * out_strides[0] + i * out_strides[-2] + j * out_strides[-1]
        out[out_idx] = temp


# Compile the matrix multiply CUDA kernel
tensor_matrix_multiply = jit(_tensor_matrix_multiply)
