from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Optional, Sequence, Tuple, Type, Union

import numpy as np

from dataclasses import field
from .autodiff import Context, Variable, backpropagate, central_difference
from .scalar_functions import (
    EQ,
    LT,
    Add,
    Exp,
    Inv,
    Log,
    Mul,
    Neg,
    ReLU,
    ScalarFunction,
    Sigmoid,
)

ScalarLike = Union[float, int, "Scalar"]


@dataclass
class ScalarHistory:
    """`ScalarHistory` stores the history of `Function` operations that were
    used to construct the current Variable.

    Attributes
    ----------
        last_fn (Optional[Type[ScalarFunction]]): The last Function that was called.
        ctx (Optional[Context]): The context for that Function.
        inputs (Sequence[Scalar]): The inputs that were given when `last_fn.forward` was called.

    """

    last_fn: Optional[Type[ScalarFunction]] = None
    ctx: Optional[Context] = None
    inputs: Sequence[Scalar] = ()


_var_count = 0


@dataclass
class Scalar:
    """A reimplementation of scalar values for autodifferentiation
    tracking. Scalar Variables behave as close as possible to standard
    Python numbers while also tracking the operations that led to the
    number's creation. They can only be manipulated by `ScalarFunction`.

    Attributes
    ----------
        data (float): The scalar value.
        history (Optional[ScalarHistory]): The history of operations leading to the value.
        derivative (Optional[float]): The derivative of the scalar.
        name (str): A unique name for the scalar.
        unique_id (int): A unique identifier for each scalar.

    """

    data: float
    history: Optional[ScalarHistory] = field(default_factory=ScalarHistory)
    derivative: Optional[float] = None
    name: str = field(default="")
    unique_id: int = field(default=0)

    def __post_init__(self):
        """Post-initialization method to set unique ID and name for Scalar."""
        global _var_count
        _var_count += 1
        object.__setattr__(self, "unique_id", _var_count)
        object.__setattr__(self, "name", str(self.unique_id))
        object.__setattr__(self, "data", float(self.data))

    def __lt__(self, b: ScalarLike) -> Scalar:
        """Less-than comparison for Scalar values.

        Args:
        ----
            b (ScalarLike): A scalar-like value to compare.

        Returns:
        -------
            Scalar: Result of the less-than comparison.

        """
        return LT.apply(self, b)

    def __eq__(self, b: ScalarLike) -> Scalar:
        """Equality comparison for Scalar values.

        Args:
        ----
            b (ScalarLike): A scalar-like value to compare.

        Returns:
        -------
            Scalar: Result of the equality comparison.

        """
        return EQ.apply(self, b)

    def __gt__(self, b: ScalarLike) -> Scalar:
        """Greater-than comparison for Scalar values.

        Args:
        ----
            b (ScalarLike): A scalar-like value to compare.

        Returns:
        -------
            Scalar: Result of the greater-than comparison.

        """
        return LT.apply(b, self)

    def __sub__(self, b: ScalarLike) -> Scalar:
        """Subtraction for Scalar values.

        Args:
        ----
            b (ScalarLike): A scalar-like value to subtract.

        Returns:
        -------
            Scalar: Result of the subtraction.

        """
        return Add.apply(self, Neg.apply(b))

    def __neg__(self) -> "Scalar":
        """Negation of the Scalar value.

        Returns
        -------
            Scalar: Result of the negation.

        """
        return Neg.apply(self)

    def __add__(self, b: ScalarLike) -> Scalar:
        """Addition for Scalar values.

        Args:
        ----
            b (ScalarLike): A scalar-like value to add.

        Returns:
        -------
            Scalar: Result of the addition.

        """
        return Add.apply(self, b)

    def __repr__(self) -> str:
        """Returns a string representation of the Scalar.

        Returns
        -------
            str: String representation of the Scalar.

        """
        return f"Scalar({self.data})"

    def __mul__(self, b: ScalarLike) -> Scalar:
        """Multiplication for Scalar values.

        Args:
        ----
            b (ScalarLike): A scalar-like value to multiply.

        Returns:
        -------
            Scalar: Result of the multiplication.

        """
        return Mul.apply(self, b)

    def __truediv__(self, b: ScalarLike) -> Scalar:
        """True division for Scalar values.

        Args:
        ----
            b (ScalarLike): A scalar-like value to divide by.

        Returns:
        -------
            Scalar: Result of the division.

        """
        return Mul.apply(self, Inv.apply(b))

    def __rtruediv__(self, b: ScalarLike) -> Scalar:
        """Reverse true division for Scalar values.

        Args:
        ----
            b (ScalarLike): A scalar-like value to divide by.

        Returns:
        -------
            Scalar: Result of the division.

        """
        return Mul.apply(b, Inv.apply(self))

    def __bool__(self) -> bool:
        """Boolean conversion of the Scalar value.

        Returns
        -------
            bool: True if the scalar value is non-zero, False otherwise.

        """
        return bool(self.data)

    def __radd__(self, b: ScalarLike) -> Scalar:
        """Reverse addition for Scalar values.

        Args:
        ----
            b (ScalarLike): A scalar-like value to add.

        Returns:
        -------
            Scalar: Result of the addition.

        """
        return self + b

    def __rmul__(self, b: ScalarLike) -> Scalar:
        """Reverse multiplication for Scalar values.

        Args:
        ----
            b (ScalarLike): A scalar-like value to multiply.

        Returns:
        -------
            Scalar: Result of the multiplication.

        """
        return self * b

    def log(self) -> "Scalar":
        """Logarithm of the Scalar value.

        Returns
        -------
            Scalar: The result of applying the logarithm function.

        """
        return Log.apply(self)

    def exp(self) -> Scalar:
        """Exponential of the Scalar value.

        Returns
        -------
            Scalar: The result of applying the exponential function.

        """
        return Exp.apply(self)

    def sigmoid(self) -> Scalar:
        """Sigmoid activation function applied to the Scalar value.

        Returns
        -------
            Scalar: The result of applying the sigmoid function.

        """
        return Sigmoid.apply(self)

    def relu(self) -> Scalar:
        """ReLU activation function applied to the Scalar value.

        Returns
        -------
            Scalar: The result of applying the ReLU function.

        """
        return ReLU.apply(self)

    def accumulate_derivative(self, x: Any) -> None:
        """Accumulate the derivative for leaf variables.

        Args:
        ----
            x (Any): Value to accumulate for the derivative.

        """
        assert self.is_leaf(), "Only leaf variables can have derivatives."
        if self.derivative is None:
            self.__setattr__("derivative", 0.0)
        self.__setattr__("derivative", self.derivative + x)

    def is_leaf(self) -> bool:
        """Check if the variable is a leaf variable.

        Returns
        -------
            bool: True if the variable is created by the user, False otherwise.

        """
        return self.history is not None and self.history.last_fn is None

    def is_constant(self) -> bool:
        """Check if the variable is constant.

        Returns
        -------
            bool: True if the variable has no history, False otherwise.

        """
        return self.history is None

    @property
    def parents(self) -> Iterable[Variable]:
        """Get the parent variables of the current variable.

        Returns
        -------
            Iterable[Variable]: The parent variables.

        """
        assert self.history is not None
        return self.history.inputs

    def chain_rule(self, d_output: Any) -> Iterable[Tuple[Variable, Any]]:
        """Apply the chain rule for backpropagation.

        Args:
        ----
            d_output (Any): The derivative with respect to the output.

        Returns:
        -------
            Iterable[Tuple[Variable, Any]]: The local gradients for each input.

        """
        h = self.history
        assert h is not None
        assert h.last_fn is not None
        assert h.ctx is not None

        local_gradients = h.last_fn._backward(h.ctx, d_output)
        gradients = []

        for input_var, local_grad in zip(h.inputs, local_gradients):
            gradients.append((input_var, local_grad))

        return gradients

    def backward(self, d_output: Optional[float] = None) -> None:
        """Perform backpropagation to compute derivatives.

        Args:
        ----
            d_output (Optional[float]): Starting derivative (default is 1.0).

        """
        if d_output is None:
            d_output = 1.0
        backpropagate(self, d_output)


def derivative_check(f: Any, *scalars: Scalar) -> None:
    """Check that autodiff works on a python function by comparing the computed
    derivative with the central difference approximation.

    Args:
    ----
        f (Any): A function from n-scalars to 1-scalar.
        *scalars (Scalar): Input scalar values.

    Raises:
    ------
        AssertionError: If the derivative check fails.

    """
    out = f(*scalars)
    out.backward()

    err_msg = """
    Derivative check at arguments f(%s) and received derivative f'=%f for argument %d,
    but was expecting derivative f'=%f from central difference."""
    for i, x in enumerate(scalars):
        check = central_difference(f, *scalars, arg=i)
        assert x.derivative is not None
        np.testing.assert_allclose(
            x.derivative,
            check.data,
            1e-2,
            1e-2,
            err_msg=err_msg
            % (str([x.data for x in scalars]), x.derivative, i, check.data),
        )
