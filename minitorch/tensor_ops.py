from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Optional, Type

import numpy as np
from typing_extensions import Protocol

from . import operators
from .tensor_data import (
    broadcast_index,
    index_to_position,
    shape_broadcast,
    to_index,
)

if TYPE_CHECKING:
    from .tensor import Tensor
    from .tensor_data import Shape, Storage, Strides


class MapProto(Protocol):
    def __call__(self, x: Tensor, out: Optional[Tensor] = ..., /) -> Tensor:
        """Call a map function"""
        ...


class TensorOps:
    @staticmethod
    def map(fn: Callable[[float], float]) -> MapProto:
        """Map placeholder"""
        ...

    @staticmethod
    def zip(
        fn: Callable[[float, float], float],
    ) -> Callable[[Tensor, Tensor], Tensor]:
        """Zip placeholder"""
        ...

    @staticmethod
    def reduce(
        fn: Callable[[float, float], float], start: float = 0.0
    ) -> Callable[[Tensor, int], Tensor]:
        """Creates a reduction function that applies a given binary function
        to reduce a tensor along a specified dimension.

        Args:
        ----
        fn (Callable[[float, float], float]): A binary function that takes two floats
            and returns a float. It defines how the reduction is performed.
        start (float, optional): The starting value for the reduction. Defaults to 0.0.

        Returns:
        -------
        Callable[[Tensor, int], Tensor]: A function that reduces a given tensor along
        the specified dimension using the provided binary function.

        """
        ...

    @staticmethod
    def matrix_multiply(a: Tensor, b: Tensor) -> Tensor:
        """Matrix multiply"""
        raise NotImplementedError("Not implemented in this assignment")

    cuda = False


class TensorBackend:
    def __init__(self, ops: Type[TensorOps]):
        """Dynamically construct a tensor backend based on a `tensor_ops` object
        that implements map, zip, and reduce higher-order functions.

        Args:
        ----
            ops : tensor operations object see `tensor_ops.py`


        Returns:
        -------
            A collection of tensor functions

        """
        # Maps
        self.neg_map = ops.map(operators.neg)
        self.sigmoid_map = ops.map(operators.sigmoid)
        self.relu_map = ops.map(operators.relu)
        self.log_map = ops.map(operators.log)
        self.exp_map = ops.map(operators.exp)
        self.id_map = ops.map(operators.id)
        self.inv_map = ops.map(operators.inv)

        # Zips
        self.add_zip = ops.zip(operators.add)
        self.mul_zip = ops.zip(operators.mul)
        self.lt_zip = ops.zip(operators.lt)
        self.eq_zip = ops.zip(operators.eq)
        self.is_close_zip = ops.zip(operators.is_close)
        self.relu_back_zip = ops.zip(operators.relu_back)
        self.log_back_zip = ops.zip(operators.log_back)
        self.inv_back_zip = ops.zip(operators.inv_back)

        # Reduce
        self.add_reduce = ops.reduce(operators.add, 0.0)
        self.mul_reduce = ops.reduce(operators.mul, 1.0)
        self.matrix_multiply = ops.matrix_multiply
        self.cuda = ops.cuda


class SimpleOps(TensorOps):
    @staticmethod
    def map(fn: Callable[[float], float]) -> MapProto:
        """Higher-order tensor map function ::

          fn_map = map(fn)
          fn_map(a, out)
          out

        Simple version::

            for i:
                for j:
                    out[i, j] = fn(a[i, j])

        Broadcasted version (`a` might be smaller than `out`) ::

            for i:
                for j:
                    out[i, j] = fn(a[i, 0])

        Args:
        ----
            fn: function from float-to-float to apply.
            a (:class:`TensorData`): tensor to map over
            out (:class:`TensorData`): optional, tensor data to fill in,
                   should broadcast with `a`

        Returns:
        -------
            new tensor data

        """
        f = tensor_map(fn)

        def ret(a: Tensor, out: Optional[Tensor] = None) -> Tensor:
            if out is None:
                out = a.zeros(a.shape)
            f(*out.tuple(), *a.tuple())
            return out

        return ret

    @staticmethod
    def zip(
        fn: Callable[[float, float], float],
    ) -> Callable[["Tensor", "Tensor"], "Tensor"]:
        """Higher-order tensor zip function ::

          fn_zip = zip(fn)
          out = fn_zip(a, b)

        Simple version ::

            for i:
                for j:
                    out[i, j] = fn(a[i, j], b[i, j])

        Broadcasted version (`a` and `b` might be smaller than `out`) ::

            for i:
                for j:
                    out[i, j] = fn(a[i, 0], b[0, j])


        Args:
        ----
            fn: function from two floats-to-float to apply
            a (:class:`TensorData`): tensor to zip over
            b (:class:`TensorData`): tensor to zip over

        Returns:
        -------
            :class:`TensorData` : new tensor data

        """
        f = tensor_zip(fn)

        def ret(a: "Tensor", b: "Tensor") -> "Tensor":
            if a.shape != b.shape:
                c_shape = shape_broadcast(a.shape, b.shape)
            else:
                c_shape = a.shape
            out = a.zeros(c_shape)
            f(*out.tuple(), *a.tuple(), *b.tuple())
            return out

        return ret

    @staticmethod
    def reduce(
        fn: Callable[[float, float], float], start: float = 0.0
    ) -> Callable[["Tensor", int], "Tensor"]:
        """Creates a reduction function that applies a given binary function
        to reduce a tensor along a specified dimension.

        Args:
        ----
            fn (Callable[[float, float], float]): A binary function that takes two floats
                and returns a float. It defines how the reduction is performed.
            start (float, optional): The initial value to start the reduction from.
                Defaults to 0.0.

        Returns:
        -------
            Callable[[Tensor, int], Tensor]: A function that reduces a given tensor along
            the specified dimension using the provided binary function.

        """
        f = tensor_reduce(fn)

        def ret(a: "Tensor", dim: int) -> "Tensor":
            out_shape = list(a.shape)
            out_shape[dim] = 1

            # Other values when not sum.
            out = a.zeros(tuple(out_shape))
            out._tensor._storage[:] = start

            f(*out.tuple(), *a.tuple(), dim)
            return out

        return ret

    @staticmethod
    def matrix_multiply(a: "Tensor", b: "Tensor") -> "Tensor":
        """Matrix multiplication"""
        raise NotImplementedError("Not implemented in this assignment")

    is_cuda = False


# Implementations.


def tensor_map(
    fn: Callable[[float], float],
) -> Callable[[Storage, Shape, Strides, Storage, Shape, Strides], None]:
    """Low-level implementation of tensor map between
    tensors with *possibly different strides*.

    Simple version:

    * Fill in the `out` array by applying `fn` to each
      value of `in_storage` assuming `out_shape` and `in_shape`
      are the same size.

    Broadcasted version:

    * Fill in the `out` array by applying `fn` to each
      value of `in_storage` assuming `out_shape` and `in_shape`
      broadcast. (`in_shape` must be smaller than `out_shape`).

    Args:
    ----
        fn: function from float-to-float to apply

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
        """Apply `fn` to each element of `in_storage` and store the result in `out`."""
        # Calculate the size of the output tensor
        out_size = int(operators.prod(out_shape))
        # Initialize indices for input and output tensors
        in_index = np.zeros_like(in_shape)
        out_index = np.zeros_like(out_shape)

        # Iterate through each element in the output tensor
        for ordinal in range(out_size):
            # Convert ordinal to multi-dimensional index for output tensor
            to_index(ordinal, out_shape, out_index)

            # Use broadcasting rules to calculate the corresponding input index
            broadcast_index(out_index, out_shape, in_shape, in_index)

            # Compute the positions in the flattened storage arrays
            in_pos = index_to_position(in_index, in_strides)
            out_pos = index_to_position(out_index, out_strides)

            # Apply the function and store the result
            out[out_pos] = fn(in_storage[in_pos])

    return _map


def tensor_zip(
    fn: Callable[[float, float], float],
) -> Callable[
    [Storage, Shape, Strides, Storage, Shape, Strides, Storage, Shape, Strides], None
]:
    """Low-level implementation of tensor zip between tensors with possibly different strides."""

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
        """Apply `fn` to elements of `a_storage` and `b_storage` and store results in `out`."""
        # Calculate the total number of elements in the output tensor
        out_size = int(operators.prod(out_shape))

        # Initialize indices for input and output tensors
        a_index = np.zeros_like(a_shape)
        b_index = np.zeros_like(b_shape)
        out_index = np.zeros_like(out_shape)

        # Iterate through each element in the output tensor
        for ordinal in range(out_size):
            # Convert ordinal to multi-dimensional index for output tensor
            to_index(ordinal, out_shape, out_index)

            # Use broadcasting rules to calculate corresponding input indices
            broadcast_index(out_index, out_shape, a_shape, a_index)
            broadcast_index(out_index, out_shape, b_shape, b_index)

            # Compute the positions in the flattened storage arrays
            a_pos = index_to_position(a_index, a_strides)
            b_pos = index_to_position(b_index, b_strides)
            out_pos = index_to_position(out_index, out_strides)

            # Apply the function to the input elements and store the result in output
            out[out_pos] = fn(a_storage[a_pos], b_storage[b_pos])

    return _zip


def tensor_reduce(
    fn: Callable[[float, float], float],
) -> Callable[[Storage, Shape, Strides, Storage, Shape, Strides, int], None]:
    """Low-level implementation of tensor reduce."""

    def _reduce(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        a_storage: Storage,
        a_shape: Shape,
        a_strides: Strides,
        reduce_dim: int,
    ) -> None:
        """Apply reduction function along the specified dimension."""
        # Calculate the total number of elements in the output tensor
        out_size = int(operators.prod(out_shape))

        # Initialize indices for input and output tensors
        a_index = np.zeros_like(a_shape)
        out_index = np.zeros_like(out_shape)

        # Iterate through each element in the output tensor
        for ordinal in range(out_size):
            # Convert ordinal to multi-dimensional index for output tensor
            to_index(ordinal, out_shape, out_index)

            # Map output index to input index, setting reduce_dim to 0 initially
            for i in range(len(a_index)):
                a_index[i] = out_index[i]
            a_index[reduce_dim] = 0

            # Compute the initial position in the flattened storage arrays
            out_pos = index_to_position(out_index, out_strides)
            a_pos = index_to_position(a_index, a_strides)

            # Initialize the reduction with the first element in the reduce dimension
            out[out_pos] = a_storage[a_pos]

            # Iterate through the reduce dimension
            for r in range(1, a_shape[reduce_dim]):
                # Update the index for the reduce dimension
                a_index[reduce_dim] = r

                # Compute the position in the flattened storage array
                a_pos = index_to_position(a_index, a_strides)

                # Apply the reduction function
                out[out_pos] = fn(out[out_pos], a_storage[a_pos])

    return _reduce


SimpleBackend = TensorBackend(SimpleOps)
