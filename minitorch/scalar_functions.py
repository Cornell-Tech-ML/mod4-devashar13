from __future__ import annotations

from typing import TYPE_CHECKING
import minitorch

from . import operators
from .autodiff import Context

if TYPE_CHECKING:
    from typing import Tuple

    from .scalar import Scalar, ScalarLike


def wrap_tuple(x: float | Tuple[float, ...]) -> Tuple[float, ...]:
    """Converts a value into a tuple if it's not already a tuple.

    Args:
    ----
        x (float | Tuple[float, ...]): A float or a tuple of floats.

    Returns:
    -------
        Tuple[float, ...]: A tuple containing the input value(s).

    """
    if isinstance(x, tuple):
        return x
    return (x,)


class ScalarFunction:
    """A wrapper for a mathematical function that processes and produces Scalar variables.

    This is a static class and is never instantiated. We use `class` here to group together the
    `forward` and `backward` code.
    """

    @classmethod
    def _backward(cls, ctx: Context, d_out: float) -> Tuple[float, ...]:
        """Wraps the `backward` function of the scalar function.

        Args:
        ----
            ctx (Context): Context object for storing intermediate results.
            d_out (float): Derivative of the output with respect to some variable.

        Returns:
        -------
            Tuple[float, ...]: The gradient with respect to each input.

        """
        return wrap_tuple(cls.backward(ctx, d_out))  # type: ignore

    @classmethod
    def _forward(cls, ctx: Context, *inps: float) -> float:
        """Wraps the `forward` function of the scalar function.

        Args:
        ----
            ctx (Context): Context object for storing intermediate results.
            *inps (float): Input values to the forward function.

        Returns:
        -------
            float: The result of applying the forward function.

        """
        return cls.forward(ctx, *inps)  # type: ignore

    @classmethod
    def apply(cls, *vals: ScalarLike) -> Scalar:
        """Applies the forward function and sets up the backward propagation.

        Args:
        ----
            *vals (ScalarLike): Scalar-like input values.

        Returns:
        -------
            Scalar: The result of the forward computation wrapped in a Scalar.

        """
        raw_vals = []
        scalars = []
        for v in vals:
            if isinstance(v, minitorch.scalar.Scalar):
                scalars.append(v)
                raw_vals.append(v.data)
            else:
                scalars.append(minitorch.scalar.Scalar(v))
                raw_vals.append(float(v))

        # Create the context.
        ctx = Context(False)

        # Call forward with the variables.
        c = cls._forward(ctx, *raw_vals)
        assert isinstance(c, float), "Expected return type float, got %s" % (type(c))

        # Create a new variable from the result with a new history.
        back = minitorch.scalar.ScalarHistory(cls, ctx, scalars)
        return minitorch.scalar.Scalar(c, back)


# Examples
class Add(ScalarFunction):
    """Addition function f(x, y) = x + y."""

    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        """Computes the forward pass of addition.

        Args:
        ----
            ctx (Context): Context object for saving intermediate values for the backward pass.
            a (float): First input.
            b (float): Second input.

        Returns:
        -------
            float: The result of adding a and b.

        """
        return a + b

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, ...]:
        """Computes the backward pass of addition.

        Args:
        ----
            ctx (Context): Context object with saved values from the forward pass.
            d_output (float): The gradient of the output with respect to the final output.

        Returns:
        -------
            Tuple[float, ...]: The gradients with respect to both inputs (a, b).

        """
        return d_output, d_output


class Log(ScalarFunction):
    """Log function f(x) = log(x)."""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Computes the forward pass of the logarithm function.

        Args:
        ----
            ctx (Context): Context object for saving intermediate values for the backward pass.
            a (float): Input value.

        Returns:
        -------
            float: The result of log(a).

        """
        ctx.save_for_backward(a)
        return operators.log(a)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Computes the backward pass of the logarithm function.

        Args:
        ----
            ctx (Context): Context object with saved values from the forward pass.
            d_output (float): The gradient of the output with respect to the final output.

        Returns:
        -------
            float: The gradient with respect to the input a.

        """
        (a,) = ctx.saved_values
        return operators.log_back(a, d_output)


class Mul(ScalarFunction):
    """Multiplication function f(x, y) = x * y."""

    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        """Computes the forward pass of multiplication.

        Args:
        ----
            ctx (Context): Context object for saving intermediate values for the backward pass.
            a (float): First input.
            b (float): Second input.

        Returns:
        -------
            float: The result of multiplying a and b.

        """
        ctx.save_for_backward(a, b)
        return operators.mul(a, b)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, float]:
        """Computes the backward pass of multiplication.

        Args:
        ----
            ctx (Context): Context object with saved values from the forward pass.
            d_output (float): The gradient of the output with respect to the final output.

        Returns:
        -------
            Tuple[float, float]: The gradients with respect to both inputs (a, b).

        """
        a, b = ctx.saved_values
        return d_output * b, d_output * a


class Inv(ScalarFunction):
    """Inverse function f(x) = 1 / x."""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Computes the forward pass of inverse.

        Args:
        ----
            ctx (Context): Context object for saving intermediate values for the backward pass.
            a (float): Input value.

        Returns:
        -------
            float: The result of 1 / a.

        """
        ctx.save_for_backward(a)
        return operators.inv(a)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Computes the backward pass of inverse.

        Args:
        ----
            ctx (Context): Context object with saved values from the forward pass.
            d_output (float): The gradient of the output with respect to the final output.

        Returns:
        -------
            float: The gradient with respect to the input a.

        """
        (a,) = ctx.saved_values
        return d_output * (-1.0 / (a**2))


class Neg(ScalarFunction):
    """Negation function f(x) = -x."""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Computes the forward pass of negation.

        Args:
        ----
            ctx (Context): Context object for saving intermediate values for the backward pass.
            a (float): Input value.

        Returns:
        -------
            float: The result of -a.

        """
        return operators.neg(a)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Computes the backward pass of negation.

        Args:
        ----
            ctx (Context): Context object with saved values from the forward pass.
            d_output (float): The gradient of the output with respect to the final output.

        Returns:
        -------
            float: The gradient with respect to the input a.

        """
        return -d_output


class Sigmoid(ScalarFunction):
    """Sigmoid function f(x) = 1 / (1 + exp(-x))."""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Computes the forward pass of sigmoid.

        Args:
        ----
            ctx (Context): Context object for saving intermediate values for the backward pass.
            a (float): Input value.

        Returns:
        -------
            float: The result of sigmoid(a).

        """
        sig = operators.sigmoid(a)
        ctx.save_for_backward(sig)
        return sig

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Computes the backward pass of sigmoid.

        Args:
        ----
            ctx (Context): Context object with saved values from the forward pass.
            d_output (float): The gradient of the output with respect to the final output.

        Returns:
        -------
            float: The gradient with respect to the input a.

        """
        (sig,) = ctx.saved_values
        return d_output * sig * (1 - sig)


class ReLU(ScalarFunction):
    """ReLU function f(x) = max(0, x)."""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Computes the forward pass of ReLU.

        Args:
        ----
            ctx (Context): Context object for saving intermediate values for the backward pass.
            a (float): Input value.

        Returns:
        -------
            float: The result of max(0, a).

        """
        ctx.save_for_backward(a)
        return operators.relu(a)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Computes the backward pass of ReLU.

        Args:
        ----
            ctx (Context): Context object with saved values from the forward pass.
            d_output (float): The gradient of the output with respect to the final output.

        Returns:
        -------
            float: The gradient with respect to the input a (0 if a <= 0).

        """
        (a,) = ctx.saved_values
        return operators.relu_back(a, d_output)


class Exp(ScalarFunction):
    """Exponential function f(x) = exp(x)."""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Computes the forward pass of the exponential function.

        Args:
        ----
            ctx (Context): Context object for saving intermediate values for the backward pass.
            a (float): Input value.

        Returns:
        -------
            float: The result of exp(a).

        """
        result = operators.exp(a)
        ctx.save_for_backward(result)
        return result

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Computes the backward pass of the exponential function.

        Args:
        ----
            ctx (Context): Context object with saved values from the forward pass.
            d_output (float): The gradient of the output with respect to the final output.

        Returns:
        -------
            float: The gradient with respect to the input a.

        """
        (result,) = ctx.saved_values
        return d_output * result


class LT(ScalarFunction):
    """Less than comparison function f(x, y) = x < y."""

    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        """Computes the forward pass of less than.

        Args:
        ----
            ctx (Context): Context object for saving intermediate values for the backward pass.
            a (float): First input.
            b (float): Second input.

        Returns:
        -------
            float: 1.0 if a < b, else 0.0.

        """
        return operators.lt(a, b)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, float]:
        """Computes the backward pass of less than. Less-than comparison has no gradients.

        Args:
        ----
            ctx (Context): Context object with saved values from the forward pass.
            d_output (float): The gradient of the output with respect to the final output.

        Returns:
        -------
            Tuple[float, float]: 0.0 for both inputs.

        """
        return 0.0, 0.0


class EQ(ScalarFunction):
    """Equality comparison function f(x, y) = x == y."""

    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        """Computes the forward pass of equality.

        Args:
        ----
            ctx (Context): Context object for saving intermediate values for the backward pass.
            a (float): First input.
            b (float): Second input.

        Returns:
        -------
            float: 1.0 if a == b, else 0.0.

        """
        return operators.eq(a, b)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, float]:
        """Computes the backward pass of equality. Equality comparison has no gradients.

        Args:
        ----
            ctx (Context): Context object with saved values from the forward pass.
            d_output (float): The gradient of the output with respect to the final output.

        Returns:
        -------
            Tuple[float, float]: 0.0 for both inputs.

        """
        return 0.0, 0.0
