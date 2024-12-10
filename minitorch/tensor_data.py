from __future__ import annotations
import random
from typing import Iterable, Optional, Sequence, Tuple, Union
import numba
import numba.cuda
import numpy as np
import numpy.typing as npt
from numpy import array, float64
from typing_extensions import TypeAlias

from . import operators

MAX_DIMS = 32


class IndexingError(RuntimeError):
    """Exception raised for indexing errors."""

    pass


# Type aliases for various tensor data types
Storage: TypeAlias = npt.NDArray[np.float64]
OutIndex: TypeAlias = npt.NDArray[np.int32]
Index: TypeAlias = npt.NDArray[np.int32]
Shape: TypeAlias = npt.NDArray[np.int32]
Strides: TypeAlias = npt.NDArray[np.int32]

UserIndex: TypeAlias = Sequence[int]
UserShape: TypeAlias = Sequence[int]
UserStrides: TypeAlias = Sequence[int]


def index_to_position(index: Index, strides: Strides) -> int:
    """Converts a multidimensional tensor `index` into a single-dimensional position in
    storage based on strides.

    Args:
    ----
        index (Index): The index tuple of ints representing the multidimensional index.
        strides (Strides): The tensor strides.

    Returns:
    -------
        int: Position in storage.

    """
    pos = 0
    for i, stride in zip(index, strides):
        pos += i * stride
    return pos


def to_index(ordinal: int, shape: Shape, out_index: OutIndex) -> None:
    """Converts an `ordinal` to an index in the `shape`.

    Args:
    ----
        ordinal (int): The ordinal position to convert.
        shape (Shape): The tensor shape.
        out_index (OutIndex): The output index corresponding to position.

    """
    ordinal = ordinal + 0
    for dim in range(len(shape) - 1, -1, -1):
        out_index[dim] = ordinal % shape[dim]
        ordinal = ordinal // shape[dim]


def broadcast_index(
    big_index: Index, big_shape: Shape, shape: Shape, out_index: OutIndex
) -> None:
    """Converts a `big_index` from a `big_shape` to a smaller `out_index`
    following broadcasting rules.

    Args:
    ----
        big_index (Index): The multidimensional index of the bigger tensor.
        big_shape (Shape): The shape of the bigger tensor.
        shape (Shape): The shape of the smaller tensor.
        out_index (OutIndex): The multidimensional index of the smaller tensor.

    """
    offset = len(big_shape) - len(shape)
    for i in range(len(shape)):
        if shape[i] == 1:
            out_index[i] = 0
        else:
            out_index[i] = big_index[i + offset]


def shape_broadcast(shape1: UserShape, shape2: UserShape) -> UserShape:
    """Broadcasts two shapes to create a new union shape.

    Args:
    ----
        shape1 (UserShape): The first shape.
        shape2 (UserShape): The second shape.

    Returns:
    -------
        UserShape: The broadcasted shape.

    Raises:
    ------
        IndexingError: If the shapes cannot be broadcast together.

    """
    len1, len2 = len(shape1), len(shape2)
    if len1 < len2:
        shape1 = (1,) * (len2 - len1) + tuple(shape1)
    elif len2 < len1:
        shape2 = (1,) * (len1 - len2) + tuple(shape2)

    broadcast_shape = []
    for dim1, dim2 in zip(shape1, shape2):
        if dim1 == dim2:
            broadcast_shape.append(dim1)
        elif dim1 == 1:
            broadcast_shape.append(dim2)
        elif dim2 == 1:
            broadcast_shape.append(dim1)
        else:
            raise IndexingError(
                f"Shapes {shape1} and {shape2} cannot be broadcast together"
            )

    return tuple(broadcast_shape)


def strides_from_shape(shape: UserShape) -> UserStrides:
    """Returns a contiguous stride for a given shape.

    Args:
    ----
        shape (UserShape): The shape of the tensor.

    Returns:
    -------
        UserStrides: The computed strides for the given shape.

    """
    layout = [1]
    offset = 1
    for s in reversed(shape):
        layout.append(s * offset)
        offset = s * offset
    return tuple(reversed(layout[:-1]))


class TensorData:
    """Manages tensor data, including storage, shape, and strides.

    Attributes
    ----------
        _storage (Storage): The underlying storage for the tensor.
        _strides (Strides): The strides of the tensor.
        _shape (Shape): The shape of the tensor.
        strides (UserStrides): User-facing strides.
        shape (UserShape): User-facing shape.
        dims (int): Number of dimensions in the tensor.

    """

    _storage: Storage
    _strides: Strides
    _shape: Shape
    strides: UserStrides
    shape: UserShape
    dims: int

    def __init__(
        self,
        storage: Union[Sequence[float], Storage],
        shape: UserShape,
        strides: Optional[UserStrides] = None,
    ):
        """Initializes the TensorData object.

        Args:
        ----
            storage (Union[Sequence[float], Storage]): The tensor storage.
            shape (UserShape): The shape of the tensor.
            strides (Optional[UserStrides], optional): The strides of the tensor. Defaults to None.

        Raises:
        ------
            IndexingError: If the strides do not match the shape.

        """
        if isinstance(storage, np.ndarray):
            self._storage = storage
        else:
            self._storage = array(storage, dtype=float64)

        if strides is None:
            strides = strides_from_shape(shape)

        assert isinstance(strides, tuple), "Strides must be a tuple"
        assert isinstance(shape, tuple), "Shape must be a tuple"
        if len(strides) != len(shape):
            raise IndexingError(f"Length of strides {strides} must match {shape}.")
        self._strides = array(strides)
        self._shape = array(shape)
        self.strides = strides
        self.dims = len(strides)
        self.size = int(operators.prod(shape))
        self.shape = shape
        assert len(self._storage) == self.size

    def to_cuda_(self) -> None:  # pragma: no cover
        """Converts the storage to CUDA.

        Converts the tensor's storage to CUDA for GPU acceleration.
        """
        if not numba.cuda.is_cuda_array(self._storage):
            self._storage = numba.cuda.to_device(self._storage)

    def is_contiguous(self) -> bool:
        """Checks if the layout is contiguous, i.e., outer dimensions have bigger
        strides than inner dimensions.

        Returns
        -------
            bool: True if the tensor is contiguous, False otherwise.

        """
        last = 1e9
        for stride in self._strides:
            if stride > last:
                return False
            last = stride
        return True

    @staticmethod
    def shape_broadcast(shape_a: UserShape, shape_b: UserShape) -> UserShape:
        """Computes the broadcasted shape of two shapes.

        Args:
        ----
            shape_a (UserShape): The first shape.
            shape_b (UserShape): The second shape.

        Returns:
        -------
            UserShape: The broadcasted shape.

        """
        return shape_broadcast(shape_a, shape_b)

    def index(self, index: Union[int, UserIndex]) -> int:
        """Converts a multidimensional index to a single-dimensional storage index.

        Args:
        ----
            index (Union[int, UserIndex]): The input index.

        Returns:
        -------
            int: The single-dimensional index in storage.

        Raises:
        ------
            IndexingError: If the index is out of bounds or has negative values.

        """
        if isinstance(index, int):
            aindex: Index = array([index])
        else:
            aindex = array(index)

        shape = self.shape
        if len(shape) == 0 and len(aindex) != 0:
            shape = (1,)

        if aindex.shape[0] != len(self.shape):
            raise IndexingError(f"Index {aindex} must be size of {self.shape}.")
        for i, ind in enumerate(aindex):
            if ind >= self.shape[i]:
                raise IndexingError(f"Index {aindex} out of range {self.shape}.")
            if ind < 0:
                raise IndexingError(f"Negative indexing for {aindex} not supported.")

        return index_to_position(array(index), self._strides)

    def indices(self) -> Iterable[UserIndex]:
        """Generates all possible indices for the tensor.

        Yields
        ------
            UserIndex: The next index in the tensor.

        """
        lshape: Shape = array(self.shape)
        out_index: Index = array(self.shape)
        for i in range(self.size):
            to_index(i, lshape, out_index)
            yield tuple(out_index)

    def sample(self) -> UserIndex:
        """Generates a random valid index within the tensor.

        Returns
        -------
            UserIndex: A random index within the tensor.

        """
        return tuple((random.randint(0, s - 1) for s in self.shape))

    def get(self, key: UserIndex) -> float:
        """Retrieves the value at the specified index.

        Args:
        ----
            key (UserIndex): The index to access.

        Returns:
        -------
            float: The value at the specified index.

        """
        x: float = self._storage[self.index(key)]
        return x

    def set(self, key: UserIndex, val: float) -> None:
        """Sets the value at the specified index.

        Args:
        ----
            key (UserIndex): The index to modify.
            val (float): The value to set.

        """
        self._storage[self.index(key)] = val

    def tuple(self) -> Tuple[Storage, Shape, Strides]:
        """Returns the core tensor data as a tuple.

        Returns
        -------
            Tuple[Storage, Shape, Strides]: The tensor's storage, shape, and strides.

        """
        return (self._storage, self._shape, self._strides)

    def permute(self, *order: int) -> TensorData:
        """Permutes the dimensions of the tensor.

        Args:
        ----
            *order (int): The new order of dimensions.

        Returns:
        -------
            TensorData: A new `TensorData` object with permuted dimensions.

        Raises:
        ------
            AssertionError: If the order does not match the dimensions.

        """
        assert list(sorted(order)) == list(
            range(len(self.shape))
        ), f"Must give a position to each dimension. Shape: {self.shape} Order: {order}"

        new_shape = tuple(self.shape[i] for i in order)
        new_strides = tuple(self.strides[i] for i in order)
        return TensorData(self._storage, new_shape, new_strides)

    def to_string(self) -> str:
        """Converts the tensor to a formatted string.

        Returns
        -------
            str: The string representation of the tensor.

        """
        s = ""
        for index in self.indices():
            l = ""
            for i in range(len(index) - 1, -1, -1):
                if index[i] == 0:
                    l = "\n%s[" % ("\t" * i) + l
                else:
                    break
            s += l
            v = self.get(index)
            s += f"{v:3.2f}"
            l = ""
            for i in range(len(index) - 1, -1, -1):
                if index[i] == self.shape[i] - 1:
                    l += "]"
                else:
                    break
            if l:
                s += l
            else:
                s += " "
        return s
