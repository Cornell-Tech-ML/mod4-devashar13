from __future__ import annotations

from typing import TYPE_CHECKING, TypeVar, Any
import numpy as np
from numba import prange
from numba import njit as _njit


from .tensor_data import (
    MAX_DIMS,
    broadcast_index,
    index_to_position,
    shape_broadcast,
    to_index,
)
from .tensor_ops import MapProto, TensorOps

if TYPE_CHECKING:
    from typing import Callable, Optional

    from .tensor import Tensor
    from .tensor_data import Shape, Storage, Strides

# TIP: Use `NUMBA_DISABLE_JIT=1 pytest tests/ -m task3_1` to run these tests without JIT.

# This code will JIT compile fast versions your tensor_data functions.
# If you get an error, read the docs for NUMBA as to what is allowed
# in these functions.
Fn = TypeVar("Fn")


def njit(fn: Fn, **kwargs: Any) -> Fn:
    """Compiles a Python function for CPU execution using Numba's JIT compiler with optimization.

    Args:
    ----
        fn (Fn): The function to compile.
        **kwargs (Any): Additional keyword arguments for the JIT compiler.

    Returns:
    -------
        Fn: The compiled CPU function.

    """
    return _njit(inline="always", **kwargs)(fn)  # type: ignore


to_index = njit(to_index)
index_to_position = njit(index_to_position)
broadcast_index = njit(broadcast_index)


class FastOps(TensorOps):
    @staticmethod
    def map(fn: Callable[[float], float]) -> MapProto:
        """See `tensor_ops.py`"""
        # This line JIT compiles your tensor_map
        f = tensor_map(njit(fn))

        def ret(a: Tensor, out: Optional[Tensor] = None) -> Tensor:
            if out is None:
                out = a.zeros(a.shape)
            f(*out.tuple(), *a.tuple())
            return out

        return ret

    @staticmethod
    def zip(fn: Callable[[float, float], float]) -> Callable[[Tensor, Tensor], Tensor]:
        """See `tensor_ops.py`"""
        f = tensor_zip(njit(fn))

        def ret(a: Tensor, b: Tensor) -> Tensor:
            c_shape = shape_broadcast(a.shape, b.shape)
            out = a.zeros(c_shape)
            f(*out.tuple(), *a.tuple(), *b.tuple())
            return out

        return ret

    @staticmethod
    def reduce(
        fn: Callable[[float, float], float], start: float = 0.0
    ) -> Callable[[Tensor, int], Tensor]:
        """See `tensor_ops.py`"""
        f = tensor_reduce(njit(fn))

        def ret(a: Tensor, dim: int) -> Tensor:
            out_shape = list(a.shape)
            out_shape[dim] = 1

            # Other values when not sum.
            out = a.zeros(tuple(out_shape))
            out._tensor._storage[:] = start

            f(*out.tuple(), *a.tuple(), dim)
            return out

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
        # Make these always be a 3 dimensional multiply
        both_2d = 0
        if len(a.shape) == 2:
            a = a.contiguous().view(1, a.shape[0], a.shape[1])
            both_2d += 1
        if len(b.shape) == 2:
            b = b.contiguous().view(1, b.shape[0], b.shape[1])
            both_2d += 1
        both_2d = both_2d == 2

        ls = list(shape_broadcast(a.shape[:-2], b.shape[:-2]))
        ls.append(a.shape[-2])
        ls.append(b.shape[-1])
        assert a.shape[-1] == b.shape[-2]
        out = a.zeros(tuple(ls))

        tensor_matrix_multiply(*out.tuple(), *a.tuple(), *b.tuple())

        # Undo 3d if we added it.
        if both_2d:
            out = out.view(out.shape[1], out.shape[2])
        return out


# Implementations


def tensor_map(
    fn: Callable[[float], float],
) -> Callable[[Storage, Shape, Strides, Storage, Shape, Strides], None]:
    """NUMBA low_level tensor_map function. See `tensor_ops.py` for description.

    Optimizations:

    * Main loop in parallel
    * All indices use numpy buffers
    * When `out` and `in` are stride-aligned, avoid indexing

    Args:
    ----
        fn: function mappings floats-to-floats to apply.

    Returns:
    -------
        Tensor map function.

    """

    def _map(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        in_storage: Storage,
        in_shape: Shape,
        in_strides: Strides,
    ) -> None:
        # TODO: Implement for Task 3.1.
        # raise NotImplementedError("Need to implement for Task 3.1")
        for out_idx in prange(len(out)):
            # Local buffers for thread safety
            out_midx = np.zeros(MAX_DIMS, np.int32)
            in_midx = np.zeros(MAX_DIMS, np.int32)

            # Compute indices
            to_index(out_idx, out_shape, out_midx)
            broadcast_index(out_midx, out_shape, in_shape, in_midx)

            # Map function
            in_idx = index_to_position(in_midx, in_strides)
            out_pos = index_to_position(out_midx, out_strides)
            out[out_pos] = fn(in_storage[in_idx])

    return njit(_map, parallel=True)  # Ensure thread safety with local buffers


def tensor_zip(
    fn: Callable[[float, float], float],
) -> Callable[
    [Storage, Shape, Strides, Storage, Shape, Strides, Storage, Shape, Strides], None
]:
    """NUMBA higher-order tensor zip function. See `tensor_ops.py` for description.

    Optimizations:

    * Main loop in parallel
    * All indices use numpy buffers
    * When `out`, `a`, `b` are stride-aligned, avoid indexing

    Args:
    ----
        fn: function maps two floats to float to apply.

    Returns:
    -------
        Tensor zip function.

    """

    def _zip(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        a_storage: Storage,
        a_shape: Shape,
        a_strides: Strides,
        b_storage: Storage,
        b_shape: Shape,
        b_strides: Strides,
    ) -> None:
        for out_idx in prange(len(out)):
            # Local buffers for thread safety
            out_midx = np.zeros(MAX_DIMS, np.int32)
            a_midx = np.zeros(MAX_DIMS, np.int32)
            b_midx = np.zeros(MAX_DIMS, np.int32)

            # Compute indices
            to_index(out_idx, out_shape, out_midx)
            broadcast_index(out_midx, out_shape, a_shape, a_midx)
            broadcast_index(out_midx, out_shape, b_shape, b_midx)

            # Zip function
            a_idx = index_to_position(a_midx, a_strides)
            b_idx = index_to_position(b_midx, b_strides)
            out_pos = index_to_position(out_midx, out_strides)
            out[out_pos] = fn(a_storage[a_idx], b_storage[b_idx])

    return njit(_zip, parallel=True)  # Thread-safe implementation


def tensor_reduce(
    fn: Callable[[float, float], float],
) -> Callable[[Storage, Shape, Strides, Storage, Shape, Strides, int], None]:
    """NUMBA higher-order tensor reduce function. See `tensor_ops.py` for description.

    Optimizations:

    * Main loop in parallel
    * All indices use numpy buffers
    * Inner-loop should not call any functions or write non-local variables

    Args:
    ----
        fn: reduction function mapping two floats to float.

    Returns:
    -------
        Tensor reduce function

    """

    def _reduce(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        a_storage: Storage,
        a_shape: Shape,
        a_strides: Strides,
        reduce_dim: int,
    ) -> None:
        """Optimized reduction function."""
        # Total number of elements in the output tensor
        out_size = int(np.prod(out_shape))

        # Parallel loop over the output tensor
        for ordinal in prange(out_size):
            # Local index buffers
            out_index = np.zeros_like(out_shape, dtype=np.int32)
            a_index = np.zeros_like(a_shape, dtype=np.int32)

            # Convert ordinal to multi-dimensional index
            to_index(ordinal, out_shape, out_index)

            # Compute the base position in the input tensor
            result = 0.0
            to_index(ordinal, out_shape, a_index)
            a_index[reduce_dim] = 0
            base_pos = index_to_position(a_index, a_strides)

            # Reduce along the specified dimension
            result = a_storage[base_pos]
            for r in range(1, a_shape[reduce_dim]):
                a_index[reduce_dim] = r
                pos = index_to_position(a_index, a_strides)
                result = fn(result, a_storage[pos])

            # Write the reduced value to the output tensor
            out_pos = index_to_position(out_index, out_strides)
            out[out_pos] = result

    return njit(_reduce, parallel=True)


def _tensor_matrix_multiply(
    out: Storage,
    out_shape: Shape,
    out_strides: Strides,
    a_storage: Storage,
    a_shape: Shape,
    a_strides: Strides,
    b_storage: Storage,
    b_shape: Shape,
    b_strides: Strides,
) -> None:
    """NUMBA tensor matrix multiply function.

    Should work for any tensor shapes that broadcast as long as

    ```
    assert a_shape[-1] == b_shape[-2]
    ```

    Optimizations:

    * Outer loop in parallel
    * No index buffers or function calls
    * Inner loop should have no global writes, 1 multiply.


    Args:
    ----
        out (Storage): storage for `out` tensor
        out_shape (Shape): shape for `out` tensor
        out_strides (Strides): strides for `out` tensor
        a_storage (Storage): storage for `a` tensor
        a_shape (Shape): shape for `a` tensor
        a_strides (Strides): strides for `a` tensor
        b_storage (Storage): storage for `b` tensor
        b_shape (Shape): shape for `b` tensor
        b_strides (Strides): strides for `b` tensor

    Returns:
    -------
        None : Fills in `out`

    """
    batch_size = out_shape[0]
    m, n = out_shape[1], out_shape[2]
    k = a_shape[-1]

    a_batch_stride = a_strides[0] if a_shape[0] > 1 else 0
    b_batch_stride = b_strides[0] if b_shape[0] > 1 else 0

    for batch in prange(batch_size):
        for i in range(m):
            for j in range(n):
                result = 0.0
                for p in range(k):
                    a_index = (
                        batch * a_batch_stride + i * a_strides[1] + p * a_strides[2]
                    )
                    b_index = (
                        batch * b_batch_stride + p * b_strides[1] + j * b_strides[2]
                    )
                    result += a_storage[a_index] * b_storage[b_index]
                out_index = (
                    batch * out_strides[0] + i * out_strides[1] + j * out_strides[2]
                )
                out[out_index] = result


tensor_matrix_multiply = njit(_tensor_matrix_multiply, parallel=True)
assert tensor_matrix_multiply is not None
