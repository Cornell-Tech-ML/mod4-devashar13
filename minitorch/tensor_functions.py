"""Implementation of the autodifferentiation Functions for Tensor."""

from __future__ import annotations
import random
from typing import TYPE_CHECKING
import numpy as np
import minitorch
from . import operators
from .autodiff import Context
from .tensor_ops import SimpleBackend, TensorBackend

if TYPE_CHECKING:
    from typing import Any, List, Tuple
    from .tensor import Tensor
    from .tensor_data import UserIndex, UserShape


def wrap_tuple(x: Any) -> tuple:
    """Converts a value into a tuple if it is not already a tuple.

    Args:
    ----
        x (Any): The input value.

    Returns:
    -------
        tuple: The input value wrapped in a tuple.

    """
    if isinstance(x, tuple):
        return x
    return (x,)


class Function:
    """Base class for all functions that can be used in autodifferentiation.

    Methods
    -------
        apply: Applies the function to input tensors and tracks history.
        _forward: Internal forward pass method.
        _backward: Internal backward pass method.

    """

    @classmethod
    def _backward(cls, ctx: Context, grad_out: Tensor) -> Tuple[Tensor, ...]:
        """Internal method to perform the backward pass.

        Args:
        ----
            ctx (Context): The context for saved values.
            grad_out (Tensor): The gradient output from the previous layer.

        Returns:
        -------
            Tuple[Tensor, ...]: Gradients with respect to the input tensors.

        """
        return wrap_tuple(cls.backward(ctx, grad_out))  # type: ignore

    @classmethod
    def _forward(cls, ctx: Context, *inps: Tensor) -> Tensor:
        """Internal method to perform the forward pass.

        Args:
        ----
            ctx (Context): The context for saved values.
            inps (Tensor): The input tensors.

        Returns:
        -------
            Tensor: The result of the forward pass.

        """
        return cls.forward(ctx, *inps)  # type: ignore

    @classmethod
    def apply(cls, *vals: Tensor) -> Tensor:
        """Applies the function to input tensors and tracks history for gradients.

        Args:
        ----
            vals (Tensor): The input tensors.

        Returns:
        -------
            Tensor: The result of the function application.

        """
        raw_vals = []
        need_grad = False
        for v in vals:
            if v is None:
                print("Received 'None' as a tensor input in Function.apply.")
            if v.requires_grad():
                need_grad = True
            raw_vals.append(v.detach())

        # Create the context
        ctx = Context(not need_grad)

        # Call forward with the variables
        c = cls._forward(ctx, *raw_vals)

        if c is None or not isinstance(c, minitorch.Tensor):
            print("Forward pass did not return a valid Tensor.")

        # Create a new variable from the result with a new history
        back = None
        if need_grad:
            back = minitorch.History(cls, ctx, vals)
        return minitorch.Tensor(c._tensor, back, backend=c.backend)


class Neg(Function):
    """Implements element-wise negation of a tensor."""

    @staticmethod
    def forward(ctx: Context, t1: Tensor) -> Tensor:
        """Forward pass for negation.

        Args:
        ----
            ctx (Context): The context for saved values.
            t1 (Tensor): The input tensor.

        Returns:
        -------
            Tensor: The negated tensor.

        """
        return t1.f.neg_map(t1)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        """Backward pass for negation.

        Args:
        ----
            ctx (Context): The context for saved values.
            grad_output (Tensor): The gradient output from the next layer.

        Returns:
        -------
            Tensor: The gradient with respect to the input tensor.

        """
        return grad_output.f.neg_map(grad_output)


class Inv(Function):
    """Implements element-wise inversion of a tensor."""

    @staticmethod
    def forward(ctx: Context, t1: Tensor) -> Tensor:
        """Forward pass for inversion.

        Args:
        ----
            ctx (Context): The context for saved values.
            t1 (Tensor): The input tensor.

        Returns:
        -------
            Tensor: The inverted tensor.

        """
        ctx.save_for_backward(t1)
        return t1.f.inv_map(t1)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        """Backward pass for inversion.

        Args:
        ----
            ctx (Context): The context for saved values.
            grad_output (Tensor): The gradient output from the next layer.

        Returns:
        -------
            Tensor: The gradient with respect to the input tensor.

        """
        (t1,) = ctx.saved_values
        return grad_output.f.inv_back_zip(t1, grad_output)


class Add(Function):
    """Implements element-wise addition of two tensors."""

    @staticmethod
    def forward(ctx: Context, t1: Tensor, t2: Tensor) -> Tensor:
        """Forward pass for addition.

        Args:
        ----
            ctx (Context): The context for saved values.
            t1 (Tensor): The first input tensor.
            t2 (Tensor): The second input tensor.

        Returns:
        -------
            Tensor: The sum of the input tensors.

        """
        return t1.f.add_zip(t1, t2)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        """Backward pass for addition.

        Args:
        ----
            ctx (Context): The context for saved values.
            grad_output (Tensor): The gradient output from the next layer.

        Returns:
        -------
            Tuple[Tensor, Tensor]: The gradients with respect to the input tensors.

        """
        return grad_output, grad_output


class All(Function):
    """Implements an operation that returns 1 if all elements are true."""

    @staticmethod
    def forward(ctx: Context, a: Tensor, dim: Tensor) -> Tensor:
        """Forward pass for checking if all elements are true.

        Args:
        ----
            ctx (Context): The context for saved values.
            a (Tensor): The input tensor.
            dim (Tensor): The dimension to reduce along.

        Returns:
        -------
            Tensor: 1 if all elements are true, else 0.

        """
        if dim is not None:
            result = a.f.mul_reduce(a, int(dim.item()))
        else:
            result = a.f.mul_reduce(
                a.contiguous().view(int(operators.prod(a.shape))), 0
            )

        if result.size > 1:
            return result.f.mul_reduce(result.contiguous().view(result.size), 0)
        return result


class Mul(Function):
    """Implements element-wise multiplication of two tensors."""

    @staticmethod
    def forward(ctx: Context, t1: Tensor, t2: Tensor) -> Tensor:
        """Forward pass for multiplication.

        Args:
        ----
            ctx (Context): The context for saved values.
            t1 (Tensor): The first input tensor.
            t2 (Tensor): The second input tensor.

        Returns:
        -------
            Tensor: The product of the input tensors.

        """
        ctx.save_for_backward(t1, t2)
        return t1.f.mul_zip(t1, t2)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        """Backward pass for multiplication.

        Args:
        ----
            ctx (Context): The context for saved values.
            grad_output (Tensor): The gradient output from the next layer.

        Returns:
        -------
            Tuple[Tensor, Tensor]: The gradients with respect to the input tensors.

        """
        t1, t2 = ctx.saved_values
        grad_t1 = grad_output.f.mul_zip(grad_output, t2)
        grad_t2 = grad_output.f.mul_zip(grad_output, t1)
        return grad_t1, grad_t2


class Sigmoid(Function):
    """Implements the sigmoid activation function."""

    @staticmethod
    def forward(ctx: Context, t1: Tensor) -> Tensor:
        """Forward pass for the sigmoid function.

        Args:
        ----
            ctx (Context): The context for saved values.
            t1 (Tensor): The input tensor.

        Returns:
        -------
            Tensor: The result after applying the sigmoid function.

        """
        sig = t1.f.sigmoid_map(t1)
        ctx.save_for_backward(sig)
        return sig

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        """Backward pass for the sigmoid function.

        Args:
        ----
            ctx (Context): The context for saved values.
            grad_output (Tensor): The gradient output from the next layer.

        Returns:
        -------
            Tensor: The gradient with respect to the input tensor.

        """
        (sig,) = ctx.saved_values
        return sig * (tensor(1) - sig) * grad_output


class ReLU(Function):
    """Implements the ReLU activation function."""

    @staticmethod
    def forward(ctx: Context, t1: Tensor) -> Tensor:
        """Forward pass for the ReLU function.

        Args:
        ----
            ctx (Context): The context for saved values.
            t1 (Tensor): The input tensor.

        Returns:
        -------
            Tensor: The result after applying the ReLU function.

        """
        ctx.save_for_backward(t1)
        return t1.f.relu_map(t1)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        """Backward pass for the ReLU function.

        Args:
        ----
            ctx (Context): The context for saved values.
            grad_output (Tensor): The gradient output from the next layer.

        Returns:
        -------
            Tensor: The gradient with respect to the input tensor.

        """
        (t1,) = ctx.saved_values
        return grad_output.f.relu_back_zip(t1, grad_output)


class Log(Function):
    """Implements the natural logarithm function."""

    @staticmethod
    def forward(ctx: Context, t1: Tensor) -> Tensor:
        """Forward pass for the logarithm function.

        Args:
        ----
            ctx (Context): The context for saved values.
            t1 (Tensor): The input tensor.

        Returns:
        -------
            Tensor: The result after applying the logarithm function.

        """
        ctx.save_for_backward(t1)
        return t1.f.log_map(t1)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        """Backward pass for the logarithm function.

        Args:
        ----
            ctx (Context): The context for saved values.
            grad_output (Tensor): The gradient output from the next layer.

        Returns:
        -------
            Tensor: The gradient with respect to the input tensor.

        """
        t1 = ctx.saved_values[0]
        grad_input = grad_output.f.mul_zip(grad_output, t1.f.inv_map(t1))
        return grad_input


class Exp(Function):
    """Implements the exponential function."""

    @staticmethod
    def forward(ctx: Context, t1: Tensor) -> Tensor:
        """Forward pass for the exponential function.

        Args:
        ----
            ctx (Context): The context for saved values.
            t1 (Tensor): The input tensor.

        Returns:
        -------
            Tensor: The result after applying the exponential function.

        """
        result = t1.f.exp_map(t1)
        ctx.save_for_backward(result)
        return result

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        """Backward pass for the exponential function.

        Args:
        ----
            ctx (Context): The context for saved values.
            grad_output (Tensor): The gradient output from the next layer.

        Returns:
        -------
            Tensor: The gradient with respect to the input tensor.

        """
        result = ctx.saved_values[0]
        grad_input = grad_output.f.mul_zip(grad_output, result)
        return grad_input


class Sum(Function):
    """Implements the sum operation along a specified dimension."""

    @staticmethod
    def forward(ctx: Context, a: Tensor, dim: Tensor) -> Tensor:
        """Forward pass for summing elements along a specified dimension.

        Args:
        ----
            ctx (Context): The context for saved values.
            a (Tensor): The input tensor.
            dim (Tensor): The dimension to sum along.

        Returns:
        -------
            Tensor: The summed tensor.

        """
        ctx.save_for_backward(a.shape, dim)
        return a.f.add_reduce(a, int(dim.item()))

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, float]:
        """Backward pass for the sum operation.

        Args:
        ----
            ctx (Context): The context for saved values.
            grad_output (Tensor): The gradient output from the next layer.

        Returns:
        -------
            Tuple[Tensor, float]: The gradient with respect to the input tensor.

        """
        a_shape, dim = ctx.saved_values
        return grad_output, 0.0


class LT(Function):
    """Implements element-wise less-than comparison of two tensors."""

    @staticmethod
    def forward(ctx: Context, t1: Tensor, t2: Tensor) -> Tensor:
        """Forward pass for less-than comparison.

        Args:
        ----
            ctx (Context): The context for saved values.
            t1 (Tensor): The first input tensor.
            t2 (Tensor): The second input tensor.

        Returns:
        -------
            Tensor: The result of the comparison.

        """
        return t1.f.lt_zip(t1, t2)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        """Backward pass for less-than comparison.

        Args:
        ----
            ctx (Context): The context for saved values.
            grad_output (Tensor): The gradient output from the next layer.

        Returns:
        -------
            Tuple[Tensor, Tensor]: Zero gradients for the input tensors.

        """
        return grad_output.zeros(), grad_output.zeros()


class EQ(Function):
    """Implements element-wise equality comparison of two tensors."""

    @staticmethod
    def forward(ctx: Context, t1: Tensor, t2: Tensor) -> Tensor:
        """Forward pass for equality comparison.

        Args:
        ----
            ctx (Context): The context for saved values.
            t1 (Tensor): The first input tensor.
            t2 (Tensor): The second input tensor.

        Returns:
        -------
            Tensor: The result of the comparison.

        """
        return t1.f.eq_zip(t1, t2)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        """Backward pass for equality comparison.

        Args:
        ----
            ctx (Context): The context for saved values.
            grad_output (Tensor): The gradient output from the next layer.

        Returns:
        -------
            Tuple[Tensor, Tensor]: Zero gradients for the input tensors.

        """
        return grad_output.zeros(), grad_output.zeros()


class IsClose(Function):
    """Implements element-wise closeness comparison of two tensors."""

    @staticmethod
    def forward(ctx: Context, t1: Tensor, t2: Tensor, tol: float = 1e-5) -> Tensor:
        """Forward pass for closeness comparison.

        Args:
        ----
            ctx (Context): The context for saved values.
            t1 (Tensor): The first input tensor.
            t2 (Tensor): The second input tensor.
            tol (float, optional): The tolerance for closeness. Defaults to 1e-5.

        Returns:
        -------
            Tensor: The result of the closeness comparison.

        """
        return t1.f.is_close_zip(t1, t2)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        """Backward pass for closeness comparison.

        Args:
        ----
            ctx (Context): The context for saved values.
            grad_output (Tensor): The gradient output from the next layer.

        Returns:
        -------
            Tuple[Tensor, Tensor]: Zero gradients for the input tensors.

        """
        return grad_output.zeros(), grad_output.zeros()


class Permute(Function):
    """Implements permutation of the tensor dimensions."""

    @staticmethod
    def forward(ctx: Context, t1: Tensor, order: Tensor) -> Tensor:
        """Forward pass for permuting tensor dimensions.

        Args:
        ----
            ctx (Context): The context for saved values.
            t1 (Tensor): The input tensor.
            order (Tensor): The new order of dimensions.

        Returns:
        -------
            Tensor: The permuted tensor.

        """
        perm_order = tuple(int(order[i]) for i in range(order.size))
        ctx.save_for_backward(perm_order)
        permuted_data = t1._tensor.permute(*perm_order)
        return t1._new(permuted_data)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, float]:
        """Backward pass for permutation.

        Args:
        ----
            ctx (Context): The context for saved values.
            grad_output (Tensor): The gradient output from the next layer.

        Returns:
        -------
        -------ty
            Tuple[Tensor, float]: The gradient with respect to the input tensor.

        """
        perm_order = ctx.saved_values[0]
        inv_perm_order = [perm_order.index(i) for i in range(len(perm_order))]
        grad_input = grad_output._tensor.permute(*inv_perm_order)
        return grad_output._new(grad_input), 0.0


class View(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor, shape: Tensor) -> Tensor:
        """Returns a new tensor with the same data but a different shape."""
        ctx.save_for_backward(a.shape)
        assert a._tensor.is_contiguous(), "Must be contiguous to view"
        shape2 = [int(shape[i]) for i in range(shape.size)]
        return minitorch.Tensor.make(
            a._tensor._storage, tuple(shape2), backend=a.backend
        )

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, float]:
        """Compute the gradient of the view function."""
        (original,) = ctx.saved_values
        return (
            minitorch.Tensor.make(
                grad_output._tensor._storage, original, backend=grad_output.backend
            ),
            0.0,
        )


class Copy(Function):
    """Implements the identity operation that creates a contiguous copy of the tensor."""

    @staticmethod
    def forward(ctx: Context, a: Tensor) -> Tensor:
        """Forward pass for creating a contiguous copy.

        Args:
        ----
            ctx (Context): The context for saved values.
            a (Tensor): The input tensor.

        Returns:
        -------
            Tensor: A contiguous copy of the input tensor.

        """
        return a.f.id_map(a)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        """Backward pass for the identity operation.

        Args:
        ----
            ctx (Context): The context for saved values.
            grad_output (Tensor): The gradient output from the next layer.

        Returns:
        -------
            Tensor: The gradient with respect to the input tensor.

        """
        return grad_output


class MatMul(Function):
    """Implements matrix multiplication of two tensors."""

    @staticmethod
    def forward(ctx: Context, t1: Tensor, t2: Tensor) -> Tensor:
        """Forward pass for matrix multiplication.

        Args:
        ----
            ctx (Context): The context for saved values.
            t1 (Tensor): The first input tensor.
            t2 (Tensor): The second input tensor.

        Returns:
        -------
            Tensor: The result of matrix multiplication.

        """
        ctx.save_for_backward(t1, t2)
        return t1.f.matrix_multiply(t1, t2)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        """Backward pass for matrix multiplication.

        Args:
        ----
            ctx (Context): The context for saved values.
            grad_output (Tensor): The gradient output from the next layer.

        Returns:
        -------
            Tuple[Tensor, Tensor]: The gradients with respect to the input tensors.

        """

        def transpose(a: Tensor) -> Tensor:
            """Transposes a matrix.

            Args:
            ----
                a (Tensor): The input matrix.

            Returns:
            -------
                Tensor: The transposed matrix.

            """
            order = list(range(len(a.shape)))
            order[-2], order[-1] = order[-1], order[-2]
            return a._new(a._tensor.permute(*order))

        t1, t2 = ctx.saved_values
        return (
            grad_output.f.matrix_multiply(grad_output, transpose(t2)),
            grad_output.f.matrix_multiply(transpose(t1), grad_output),
        )


def zeros(shape: UserShape, backend: TensorBackend = SimpleBackend) -> Tensor:
    """Produces a zero tensor of the specified shape.

    Args:
    ----
        shape (UserShape): The shape of the tensor.
        backend (TensorBackend, optional): The tensor backend to use. Defaults to SimpleBackend.

    Returns:
    -------
        Tensor: A tensor filled with zeros.

    """
    return minitorch.Tensor.make(
        [0.0] * int(operators.prod([float(d) for d in shape])),
        tuple(shape),
        backend=backend,
    )


def rand(
    shape: UserShape,
    backend: TensorBackend = SimpleBackend,
    requires_grad: bool = False,
) -> Tensor:
    """Produces a random tensor of the specified shape.

    Args:
    ----
        shape (UserShape): The shape of the tensor.
        backend (TensorBackend, optional): The tensor backend to use. Defaults to SimpleBackend.
        requires_grad (bool, optional): Whether to enable gradient tracking. Defaults to False.

    Returns:
    -------
        Tensor: A random tensor.

    """
    vals = [
        random.random() for _ in range(int(operators.prod([float(d) for d in shape])))
    ]
    tensor = minitorch.Tensor.make(vals, shape, backend=backend)
    tensor.requires_grad_(requires_grad)
    return tensor


def _tensor(
    ls: Any,
    shape: UserShape,
    backend: TensorBackend = SimpleBackend,
    requires_grad: bool = False,
) -> Tensor:
    """Produces a tensor with the specified data and shape.

    Args:
    ----
        ls (Any): The data for the tensor.
        shape (UserShape): The shape of the tensor.
        backend (TensorBackend, optional): The tensor backend to use. Defaults to SimpleBackend.
        requires_grad (bool, optional): Whether to enable gradient tracking. Defaults to False.

    Returns:
    -------
        Tensor: The created tensor.

    """
    tensor = minitorch.Tensor.make(ls, shape, backend=backend)
    tensor.requires_grad_(requires_grad)
    return tensor


def tensor(
    ls: Any, backend: TensorBackend = SimpleBackend, requires_grad: bool = False
) -> Tensor:
    """Produces a tensor with data and shape derived from the input.

    Args:
    ----
        ls (Any): The data for the tensor.
        backend (TensorBackend, optional): The tensor backend to use. Defaults to SimpleBackend.
        requires_grad (bool, optional): Whether to enable gradient tracking. Defaults to False.

    Returns:
    -------
        Tensor: The created tensor.

    """

    def shape(ls: Any) -> List[int]:
        if isinstance(ls, (list, tuple)):
            return [len(ls)] + shape(ls[0])
        else:
            return []

    def flatten(ls: Any) -> List[float]:
        if isinstance(ls, (list, tuple)):
            return [y for x in ls for y in flatten(x)]
        else:
            return [ls]

    cur = flatten(ls)
    shape2 = shape(ls)
    return _tensor(cur, tuple(shape2), backend=backend, requires_grad=requires_grad)


def grad_central_difference(
    f: Any, *vals: Tensor, arg: int = 0, epsilon: float = 1e-6, ind: UserIndex
) -> float:
    """Computes the gradient using central difference approximation.

    Args:
    ----
        f (Any): The function to differentiate.
        vals (Tensor): The input tensors.
        arg (int, optional): The argument index to differentiate. Defaults to 0.
        epsilon (float, optional): The epsilon for central difference. Defaults to 1e-6.
        ind (UserIndex): The index to sample.

    Returns:
    -------
        float: The approximated gradient.

    """
    x = vals[arg]
    up = zeros(x.shape)
    up[ind] = epsilon
    vals1 = [x if j != arg else x + up for j, x in enumerate(vals)]
    vals2 = [x if j != arg else x - up for j, x in enumerate(vals)]
    delta: Tensor = f(*vals1).sum() - f(*vals2).sum()

    return delta[0] / (2.0 * epsilon)


def grad_check(f: Any, *vals: Tensor) -> None:
    """Checks whether the autodiff gradient matches the central difference approximation.

    Args:
    ----
        f (Any): The function to differentiate.
        vals (Tensor): The input tensors.

    Raises:
    ------
        AssertionError: If the gradients do not match within the tolerance.

    """
    for x in vals:
        x.requires_grad_(True)
        x.zero_grad_()
    random.seed(10)
    out = f(*vals)
    out.sum().backward()
    err_msg = """

Gradient check error for function %s.

Input %s

Received derivative %f for argument %d and index %s,
but was expecting derivative %f from central difference.

"""

    for i, x in enumerate(vals):
        ind = x._tensor.sample()
        check = grad_central_difference(f, *vals, arg=i, ind=ind)
        assert x.grad is not None
        np.testing.assert_allclose(
            x.grad[ind],
            check,
            1e-2,
            1e-2,
            err_msg=err_msg % (f, vals, x.grad[ind], i, ind, check),
        )
