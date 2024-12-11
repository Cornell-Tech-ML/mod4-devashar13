"""Implementation of the core Tensor object for autodifferentiation."""

from __future__ import annotations
from dataclasses import dataclass
from typing import TYPE_CHECKING
import numpy as np
from . import operators
from .autodiff import Context, Variable, backpropagate
from .tensor_data import TensorData

# Comment these out if not yet implemented
from .tensor_functions import (
    EQ,
    LT,
    Add,
    All,
    Copy,
    Exp,
    Inv,
    IsClose,
    Log,
    MatMul,
    Mul,
    Neg,
    Permute,
    ReLU,
    Sigmoid,
    Sum,
    View,
)

if TYPE_CHECKING:
    from typing import Any, Iterable, List, Optional, Sequence, Tuple, Type, Union
    import numpy.typing as npt
    from .tensor_data import Shape, Storage, Strides, UserIndex, UserShape, UserStrides
    from .tensor_functions import Function
    from .tensor_ops import TensorBackend

    TensorLike = Union[float, int, "Tensor"]


@dataclass
class History:
    """Stores the history of `Function` operations that were used to construct
    the current Variable.

    Attributes
    ----------
        last_fn (Optional[Type[Function]]): The last function applied to this variable.
        ctx (Optional[Context]): The context of the last function call.
        inputs (Sequence[Tensor]): The input tensors to the last function call.

    """

    last_fn: Optional[Type[Function]] = None
    ctx: Optional[Context] = None
    inputs: Sequence[Tensor] = ()


_tensor_count = 0


class Tensor:
    """A generalization of Scalar that handles multidimensional arrays.

    Attributes
    ----------
        backend (TensorBackend): The backend used for tensor operations.
        history (Optional[History]): The history of operations applied to the tensor.
        grad (Optional[Tensor]): The gradient of the tensor.
        _tensor (TensorData): The tensor data.
        unique_id (int): A unique identifier for the tensor.
        name (str): The name of the tensor.

    """

    def __init__(
        self,
        v: TensorData,
        back: Optional[History] = None,
        name: Optional[str] = None,
        backend: Optional[TensorBackend] = None,
    ):
        """Initializes the Tensor object.

        Args:
        ----
            v (TensorData): The tensor data.
            back (Optional[History], optional): The history of operations. Defaults to None.
            name (Optional[str], optional): The name of the tensor. Defaults to None.
            backend (Optional[TensorBackend], optional): The backend used for tensor operations.
                Must be provided. Defaults to None.

        """
        global _tensor_count
        _tensor_count += 1
        self.unique_id = _tensor_count
        assert isinstance(v, TensorData)
        assert backend is not None
        self._tensor = v
        self.history = back
        self.backend = backend
        self.grad = None
        self.name = name if name else str(self.unique_id)
        self.f = backend

    def requires_grad_(self, x: bool) -> None:
        """Sets the tensor to require gradients.

        Args:
        ----
            x (bool): Whether gradients are required.

        """
        self.history = History()

    def requires_grad(self) -> bool:
        """Checks if the tensor requires gradients.

        Returns
        -------
            bool: True if the tensor requires gradients, False otherwise.

        """
        return self.history is not None

    def to_numpy(self) -> npt.NDArray[np.float64]:
        """Converts the tensor to a numpy array.

        Returns
        -------
            npt.NDArray[np.float64]: The tensor as a numpy array.

        """
        return self.contiguous()._tensor._storage.reshape(self.shape)

    def _ensure_tensor(self, b: TensorLike) -> Tensor:
        """Converts a Python number into a tensor with the same backend.

        Args:
        ----
            b (TensorLike): The input to be converted.

        Returns:
        -------
            Tensor: The converted tensor.

        """
        if isinstance(b, (int, float)):
            c = Tensor.make([b], (1,), backend=self.backend)
        else:
            b._type_(self.backend)
            c = b
        return c

    def item(self) -> float:
        """Converts a 1-element tensor to a float.

        Returns
        -------
            float: The tensor value as a float.

        Raises
        ------
            AssertionError: If the tensor size is not 1.

        """
        assert self.size == 1
        x: float = self._tensor._storage[0]
        return x

    def contiguous(self) -> Tensor:
        """Returns a contiguous tensor with the same data.

        Returns
        -------
            Tensor: A contiguous tensor.

        """
        return Copy.apply(self)

    def __repr__(self) -> str:
        """Returns a string representation of the tensor.

        Returns
        -------
            str: The string representation of the tensor.

        """
        return self._tensor.to_string()

    def __getitem__(self, key: Union[int, UserIndex]) -> float:
        """Gets the value at the specified index.

        Args:
        ----
            key (Union[int, UserIndex]): The index to access.

        Returns:
        -------
            float: The value at the specified index.

        """
        key2 = (key,) if isinstance(key, int) else key
        return self._tensor.get(key2)

    def __setitem__(self, key: Union[int, UserIndex], val: float) -> None:
        """Sets the value at the specified index.

        Args:
        ----
            key (Union[int, UserIndex]): The index to modify.
            val (float): The value to set.

        """
        key2 = (key,) if isinstance(key, int) else key
        self._tensor.set(key2, val)

    def _type_(self, backend: TensorBackend) -> None:
        """Sets the backend of the tensor.

        Args:
        ----
            backend (TensorBackend): The new backend.

        """
        self.backend = backend
        if backend.cuda:  # pragma: no cover
            self._tensor.to_cuda_()

    def _new(self, tensor_data: TensorData) -> Tensor:
        """Creates a new tensor with the given data.

        Args:
        ----
            tensor_data (TensorData): The tensor data.

        Returns:
        -------
            Tensor: The new tensor.

        """
        return Tensor(tensor_data, backend=self.backend)

    @staticmethod
    def make(
        storage: Union[Storage, List[float]],
        shape: UserShape,
        strides: Optional[UserStrides] = None,
        backend: Optional[TensorBackend] = None,
    ) -> Tensor:
        """Creates a new tensor from data.

        Args:
        ----
            storage (Union[Storage, List[float]]): The storage for the tensor.
            shape (UserShape): The shape of the tensor.
            strides (Optional[UserStrides], optional): The strides of the tensor. Defaults to None.
            backend (Optional[TensorBackend], optional): The backend used for tensor operations.
                Defaults to None.

        Returns:
        -------
            Tensor: The new tensor.

        """
        return Tensor(TensorData(storage, shape, strides), backend=backend)

    def expand(self, other: Tensor) -> Tensor:
        """Expands the tensor for backprop over broadcasting.

        Args:
        ----
            other (Tensor): The backward tensor (must broadcast with self).

        Returns:
        -------
            Tensor: The expanded version of `other` with the right derivatives.

        """
        # Case 1: Both the same shape.
        if self.shape == other.shape:
            return other

        # Case 2: Backward is smaller than self. Broadcast up.
        true_shape = TensorData.shape_broadcast(self.shape, other.shape)
        buf = self.zeros(true_shape)
        self.backend.id_map(other, buf)
        if self.shape == true_shape:
            return buf

        # Case 3: Still different, reduce extra dims.
        out = buf
        orig_shape = [1] * (len(out.shape) - len(self.shape)) + list(self.shape)
        for dim, shape in enumerate(out.shape):
            if orig_shape[dim] == 1 and shape != 1:
                out = self.backend.add_reduce(out, dim)
        assert out.size == self.size, f"{out.shape} {self.shape}"
        return Tensor.make(out._tensor._storage, self.shape, backend=self.backend)

    def zeros(self, shape: Optional[UserShape] = None) -> Tensor:
        """Creates a tensor filled with zeros.

        Args:
        ----
            shape (Optional[UserShape], optional): The shape of the tensor.
                If None, the shape of the current tensor is used. Defaults to None.

        Returns:
        -------
            Tensor: The tensor filled with zeros.

        """

        def zero(shape: UserShape) -> Tensor:
            return Tensor.make(
                [0.0] * int(operators.prod(shape)), shape, backend=self.backend
            )

        if shape is None:
            out = zero(self.shape)
        else:
            out = zero(shape)
        out._type_(self.backend)
        return out

    def tuple(self) -> Tuple[Storage, Shape, Strides]:
        """Gets the tensor data info as a tuple.

        Returns
        -------
            Tuple[Storage, Shape, Strides]: The tensor data tuple.

        """
        return self._tensor.tuple()

    def detach(self) -> Tensor:
        """Detaches the tensor from backpropagation.

        Returns
        -------
            Tensor: The detached tensor.

        """
        return Tensor(self._tensor, backend=self.backend)

    def accumulate_derivative(self, x: Any) -> None:
        """Adds `x` to the derivative accumulated on this variable.
        Should only be called during autodifferentiation on leaf variables.

        Args:
        ----
            x (Any): The value to be accumulated.

        Raises:
        ------
            AssertionError: If the variable is not a leaf.

        """
        assert self.is_leaf(), "Only leaf variables can have derivatives."
        if self.grad is None:
            self.grad = Tensor.make(
                [0.0] * int(operators.prod(self.shape)),
                self.shape,
                backend=self.backend,
            )
        self.grad += x

    def is_leaf(self) -> bool:
        """Checks if this variable is a leaf.

        Returns
        -------
            bool: True if this is a leaf variable, False otherwise.

        """
        return self.history is not None and self.history.last_fn is None

    def is_constant(self) -> bool:
        """Checks if the tensor is constant.

        Returns
        -------
            bool: True if the tensor is constant, False otherwise.

        """
        return self.history is None

    @property
    def parents(self) -> Iterable[Variable]:
        """Gets the parent variables.

        Returns
        -------
            Iterable[Variable]: The parent variables.

        """
        assert self.history is not None
        return self.history.inputs

    def chain_rule(self, d_output: Any) -> Iterable[Tuple[Variable, Any]]:
        """Applies the chain rule for backpropagation.

        Args:
        ----
            d_output (Any): The gradient of the output.

        Returns:
        -------
            Iterable[Tuple[Variable, Any]]: The gradients of the inputs.

        """
        h = self.history
        if h is None:
            return []
        assert h.last_fn is not None
        assert h.ctx is not None

        x = h.last_fn._backward(h.ctx, d_output)
        assert len(x) == len(h.inputs), f"Bug in function {h.last_fn}"
        return [
            (inp, inp.expand(self._ensure_tensor(d_in)))
            for inp, d_in in zip(h.inputs, x)
        ]

    def backward(self, grad_output: Optional[Tensor] = None) -> None:
        """Performs backpropagation on the tensor.

        Args:
        ----
            grad_output (Optional[Tensor], optional): The gradient of the output.
                If None, a gradient of 1 is assumed for scalars. Defaults to None.

        """
        if grad_output is None:
            assert self.shape == (1,), "Must provide grad_output if non-scalar"
            grad_output = Tensor.make([1.0], (1,), backend=self.backend)
        backpropagate(self, grad_output)

    def __truediv__(self, b: TensorLike) -> Tensor:
        """Element-wise division of the tensor.

        Args:
        ----
            b (TensorLike): The divisor.

        Returns:
        -------
            Tensor: The result of division.

        """
        return Mul.apply(self, Inv.apply(self._ensure_tensor(b)))

    def __rtruediv__(self, b: TensorLike) -> Tensor:
        """Element-wise division, with the tensor as the divisor.

        Args:
        ----
            b (TensorLike): The dividend.

        Returns:
        -------
            Tensor: The result of division.

        """
        return Mul.apply(self._ensure_tensor(b), Inv.apply(self))

    def __matmul__(self, b: Tensor) -> Tensor:
        """Performs matrix multiplication with another tensor.

        Args:
        ----
            b (Tensor): The tensor to multiply with.

        Returns:
        -------
            Tensor: The result of matrix multiplication.

        """
        return MatMul.apply(self, b)

    @property
    def shape(self) -> UserShape:
        """Gets the shape of the tensor.

        Returns
        -------
            UserShape: The shape of the tensor.

        """
        return self._tensor.shape

    def __add__(self, other: TensorLike) -> Tensor:
        """Element-wise addition of tensors.

        Args:
        ----
            other (TensorLike): The tensor to add.

        Returns:
        -------
            Tensor: The sum of the tensors.

        """
        other = self._ensure_tensor(other)
        return Add.apply(self, other)

    def __radd__(self, other: TensorLike) -> Tensor:
        """Right-side element-wise addition of tensors.

        Args:
        ----
            other (TensorLike): The tensor to add.

        Returns:
        -------
            Tensor: The sum of the tensors.

        """
        return self.__add__(other)

    def __sub__(self, other: TensorLike) -> Tensor:
        """Element-wise subtraction of tensors.

        Args:
        ----
            other (TensorLike): The tensor to subtract.

        Returns:
        -------
            Tensor: The result of subtraction.

        """
        other = self._ensure_tensor(other)
        return Add.apply(self, other.neg())

    def __rsub__(self, other: TensorLike) -> Tensor:
        """Right-side element-wise subtraction of tensors.

        Args:
        ----
            other (TensorLike): The tensor to subtract from.

        Returns:
        -------
            Tensor: The result of subtraction.

        """
        return (-self).__add__(other)

    def __mul__(self, other: TensorLike) -> Tensor:
        """Element-wise multiplication of tensors.

        Args:
        ----
            other (TensorLike): The tensor to multiply.

        Returns:
        -------
            Tensor: The product of the tensors.

        """
        other = self._ensure_tensor(other)
        return Mul.apply(self, other)

    def __rmul__(self, other: TensorLike) -> Tensor:
        """Right-side element-wise multiplication of tensors.

        Args:
        ----
            other (TensorLike): The tensor to multiply.

        Returns:
        -------
            Tensor: The product of the tensors.

        """
        return self.__mul__(other)

    def __neg__(self) -> Tensor:
        """Negates the tensor.

        Returns
        -------
            Tensor: The negated tensor.

        """
        return Neg.apply(self)

    def __gt__(self, other: TensorLike) -> Tensor:
        """Element-wise greater-than comparison.

        Args:
        ----
            other (TensorLike): The tensor to compare with.

        Returns:
        -------
            Tensor: The result of the comparison.

        """
        other = self._ensure_tensor(other)
        return LT.apply(other, self)

    def __lt__(self, other: TensorLike) -> Tensor:
        """Element-wise less-than comparison.

        Args:
        ----
            other (TensorLike): The tensor to compare with.

        Returns:
        -------
            Tensor: The result of the comparison.

        """
        other = self._ensure_tensor(other)
        return LT.apply(self, other)

    def __eq__(self, other: TensorLike) -> Tensor:
        """Element-wise equality comparison.

        Args:
        ----
            other (TensorLike): The tensor to compare with.

        Returns:
        -------
            Tensor: The result of the comparison.

        """
        return EQ.apply(self, self._ensure_tensor(other))

    def relu(self) -> Tensor:
        """Applies the ReLU function element-wise.

        Returns
        -------
            Tensor: The result after applying ReLU.

        """
        return ReLU.apply(self)

    def sigmoid(self) -> Tensor:
        """Applies the Sigmoid function element-wise.

        Returns
        -------
            Tensor: The result after applying Sigmoid.

        """
        return Sigmoid.apply(self)

    def exp(self) -> Tensor:
        """Computes the element-wise exponential of the tensor.

        Returns
        -------
            Tensor: The result of the exponential operation.

        """
        return Exp.apply(self)

    def is_close(self, other: TensorLike, tol: float = 1e-5) -> Tensor:
        """Checks if the tensor is element-wise close to another tensor.

        Args:
        ----
            other (TensorLike): The tensor to compare with.
            tol (float, optional): The tolerance for closeness. Defaults to 1e-5.

        Returns:
        -------
            Tensor: A tensor indicating element-wise closeness.

        """
        other = self._ensure_tensor(other)
        result = IsClose.apply(self, other, self._ensure_tensor(tol))
        return result

    def log(self) -> Tensor:
        """Computes the element-wise natural logarithm of the tensor.

        Returns
        -------
            Tensor: The result of the logarithm operation.

        """
        return Log.apply(self)

    def neg(self) -> Tensor:
        """Negates the tensor element-wise.

        Returns
        -------
            Tensor: The negated tensor.

        """
        return Neg.apply(self)

    def sum(self, dim: Optional[int] = None) -> Tensor:
        """Sums the tensor along a specified dimension.

        Args:
        ----
            dim (Optional[int], optional): The dimension to sum over.
                If None, sums over all dimensions. Defaults to None.

        Returns:
        -------
            Tensor: The summed tensor.

        """
        if dim is None:
            result = self
            for d in range(len(self.shape)):
                result = Sum.apply(result, Tensor.make([d], (1,), backend=self.backend))
            return result.view((1,))
        else:
            result = Sum.apply(self, Tensor.make([dim], (1,), backend=self.backend))
            new_shape = list(self.shape)
            new_shape[dim] = 1
            return result.view(tuple(new_shape))

    def mean(self, dim: Optional[int] = None) -> Tensor:
        """Computes the mean of the tensor along a specified dimension.

        Args:
        ----
            dim (Optional[int], optional): The dimension to average over.
                If None, averages over all dimensions. Defaults to None.

        Returns:
        -------
            Tensor: The mean of the tensor.

        """
        summed = self.sum(dim=dim)
        size = self.shape[dim] if dim is not None else self.size
        return summed / size

    def all(self, dim: Optional[int] = None) -> Tensor:
        """Checks if all elements of the tensor are True along a specified dimension.

        Args:
        ----
            dim (Optional[int], optional): The dimension to check.
                If None, checks all elements. Defaults to None.

        Returns:
        -------
            Tensor: A tensor indicating if all elements are True.

        """
        if dim is None:
            return All.apply(self, Tensor.make([0], (1,), backend=self.backend))
        return All.apply(self, Tensor.make([dim], (1,), backend=self.backend))

    def permute(self, *order: Union[int, Tuple[int, ...]]) -> Tensor:
        """Permutes the dimensions of the tensor.

        Args:
        ----
            order (Union[int, Tuple[int, ...]]): The new order of dimensions.

        Returns:
        -------
            Tensor: The permuted tensor.

        """
        # If a single tuple or list is provided, unpack it
        if len(order) == 1 and isinstance(order[0], (tuple, list)):
            order = order[0]

        # Initialize an empty list to store the flattened order
        flat_order: List[float] = []

        # Iterate through each item in order
        for item in order:
            if isinstance(item, int):
                # Convert integer to float and append
                flat_order.append(float(item))
            elif isinstance(item, tuple):
                # Iterate through the tuple and convert each integer to float
                for dim in item:
                    if not isinstance(dim, int):
                        raise TypeError("All permutation indices must be integers.")
                    flat_order.append(float(dim))
            else:
                # Raise an error if the item is neither int nor tuple
                raise TypeError("Order must be an integer or a tuple of integers.")

        # Create a Tensor representing the permutation order
        order_tensor = Tensor.make(flat_order, (len(flat_order),), backend=self.backend)

        # Apply the permutation using the Permute function
        return Permute.apply(self, order_tensor)

    def view(self, *shape: Union[int, Tuple[int, ...]]) -> Tensor:
        """Reshapes the tensor to the specified shape.

        Args:
        ----
            shape (Union[int, Tuple[int, ...]]): The new shape.

        Returns:
        -------
            Tensor: The reshaped tensor.

        Raises:
        ------
            ValueError: If the total number of elements changes.
            TypeError: If non-integer dimensions are provided.

        """
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = shape[0]

        flat_shape = []
        for dim in shape:
            if isinstance(dim, int):
                flat_shape.append(dim)
            elif isinstance(dim, tuple):
                flat_shape.extend(dim)
            else:
                raise TypeError(
                    "All dimensions must be integers or tuples of integers."
                )

        flat_shape = tuple(flat_shape)

        if operators.prod(flat_shape) != self.size:
            raise ValueError("Total number of elements must remain unchanged.")

        shape_storage = [float(dim) for dim in flat_shape]
        shape_tensor = Tensor.make(
            shape_storage, (len(flat_shape),), backend=self.backend
        )
        return View.apply(self, shape_tensor)

    def zero_grad_(self) -> None:
        """Resets the gradient of the tensor to None."""
        self.grad = None

    @property
    def size(self) -> int:
        """Gets the total number of elements in the tensor.

        Returns
        -------
            int: The total number of elements.

        """
        return int(operators.prod(self.shape))
