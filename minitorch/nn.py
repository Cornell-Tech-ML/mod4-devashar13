from typing import Tuple

from . import operators
from .autodiff import Context
from .fast_ops import FastOps
from .tensor import Tensor
from .tensor_functions import Function, rand


# List of functions in this file:
# - avgpool2d: Tiled average pooling 2D
# - argmax: Compute the argmax as a 1-hot tensor
# - Max: New Function for max operator
# - max: Apply max reduction
# - softmax: Compute the softmax as a tensor
# - logsoftmax: Compute the log of the softmax as a tensor - See https://en.wikipedia.org/wiki/LogSumExp#log-sum-exp_trick_for_log-domain_calculations
# - maxpool2d: Tiled max pooling 2D
# - dropout: Dropout positions based on random noise, include an argument to turn off


def tile(input: Tensor, kernel: Tuple[int, int]) -> Tuple[Tensor, int, int]:
    """Reshape an image tensor for 2D pooling

    Args:
    ----
        input: batch x channel x height x width
        kernel: height x width of pooling

    Returns:
    -------
        Tensor of size batch x channel x new_height x new_width x (kernel_height * kernel_width) as well as the new_height and new_width value.

    """
    batch, channel, height, width = input.shape
    kh, kw = kernel
    assert height % kh == 0
    assert width % kw == 0

    new_height = int(height / kh)
    new_width = int(width / kw)

    input = (
        input.contiguous()
        .view(batch, channel, height, new_width, kw)
        .permute(0, 1, 3, 2, 4)
        .contiguous()
        .view(batch, channel, new_width, new_height, kh * kw)
    )

    return (input, new_height, new_width)


max_reduce = FastOps.reduce(operators.max, -1e9)


def avgpool2d(input: Tensor, kernel: Tuple[int, int]) -> Tensor:
    """Tiled average pooling 2D

    Args:
    ----
        input (:class:`Tensor`): batch x channel x height x width
        kernel ( pair of ints ): height x width of pooling

    Returns:
    -------
        :class:`Tensor` : pooled tensor

    """
    output, new_height, new_width = tile(input, kernel)
    # Take mean over the last dimension (tile_size)
    pooled = output.mean(4)
    return pooled.contiguous().view(
        output.shape[0], output.shape[1], new_height, new_width
    )


def argmax(input: Tensor, dim: int) -> Tensor:
    """Compute the argmax as a 1-hot tensor.

    Args:
    ----
        input (:class:`Tensor`): input tensor
        dim (int): dimension to apply argmax


    Returns:
    -------
        :class:`Tensor` : tensor with 1 on highest cell in dim, 0 otherwise

    """
    out = max_reduce(input, dim)
    return out == input


class Max(Function):
    @staticmethod
    def forward(ctx: Context, input: Tensor, dim: Tensor) -> Tensor:
        """Forward of max should be max reduction"""
        out = max_reduce(input, int(dim.item()))
        mask = input == out
        ctx.save_for_backward(mask)
        return out

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, float]:
        """Backward of max should be argmax (see above)"""
        (mask,) = ctx.saved_values
        return mask * grad_output, 0.0


def max(input: Tensor, dim: int) -> Tensor:
    """Compute the max along a dimension.

    Args:
    ----
        input: Tensor
        dim: dimension to compute max

    Returns:
    -------
        Tensor with dimension dim reduced to 1

    """
    if dim is None:
        return Max.apply(input.contiguous().view(input.size), input._ensure_tensor(0))
    else:
        return Max.apply(input, input._ensure_tensor(dim))


def softmax(input: Tensor, dim: int) -> Tensor:
    """Compute the softmax as a tensor

    Args:
    ----
        input: Tensor
        dim: int

    Returns:
    -------
        Tensor: softmax tensor

    """
    out = input.exp()
    return out / (out.sum(dim))


def logsoftmax(input: Tensor, dim: int) -> Tensor:
    """Compute the log of the softmax as a tensor

    Args:
    ----
        input: Tensor
        dim: int

    Returns:
    -------
        Tensor: log of the softmax tensor

    """
    t = input.exp()
    t = t.sum(dim)
    t = t.log()
    return input - t


def maxpool2d(input: Tensor, kernel: Tuple[int, int]) -> Tensor:
    """Tiled max pooling 2D

    Args:
    ----
        input (:class:`Tensor`): batch x channel x height x width
        kernel ( pair of ints ): height x width of pooling

    Returns:
    -------
        :class:`Tensor` : pooled tensor

    """
    output, new_height, new_width = tile(input, kernel)
    pooled = max(output, 4)
    return pooled.contiguous().view(
        output.shape[0], output.shape[1], new_height, new_width
    )


def dropout(input: Tensor, rate: float, ignore: bool = False) -> Tensor:
    """Dropout positions based on random noise, include an argument to turn off

    Args:
    ----
        input: Tensor
        p: float
        ignore: bool

    Returns:
    -------
        Tensor: with dropout applied

    """
    if not ignore:
        rand_tensor = rand(input.shape)
        rand_drop = rand_tensor > rate
        return input * rand_drop
    else:
        return input
