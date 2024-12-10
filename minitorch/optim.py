from typing import Sequence
from .module import Parameter
from .scalar import Scalar


class Optimizer:
    """Base class for all optimizers.

    Attributes
    ----------
        parameters (Sequence[Parameter]): A sequence of parameters to be optimized.

    """

    def __init__(self, parameters: Sequence[Parameter]):
        """Initializes the optimizer with a sequence of parameters.

        Args:
        ----
            parameters (Sequence[Parameter]): The parameters that need to be optimized.

        """
        self.parameters = parameters


class SGD(Optimizer):
    """Stochastic Gradient Descent (SGD) optimizer.

    Attributes
    ----------
        parameters (Sequence[Parameter]): A sequence of parameters to be optimized.
        lr (float): Learning rate for the optimizer.

    """

    def __init__(self, parameters: Sequence[Parameter], lr: float = 1.0):
        """Initializes the SGD optimizer with parameters and learning rate.

        Args:
        ----
            parameters (Sequence[Parameter]): The parameters that need to be optimized.
            lr (float, optional): Learning rate for the optimizer. Defaults to 1.0.

        """
        super().__init__(parameters)
        self.lr = lr

    def zero_grad(self) -> None:
        """Sets the gradients of all parameters to zero.

        If the parameter has attributes 'derivative' or 'grad', this method will
        reset them to None.
        """
        for p in self.parameters:
            if p.value is None:
                continue
            if hasattr(p.value, "derivative"):
                if p.value.derivative is not None:
                    p.value.derivative = None
            if hasattr(p.value, "grad"):
                if p.value.grad is not None:
                    p.value.grad = None

    def step(self) -> None:
        """Performs a single optimization step.

        Updates the parameters based on their gradients and the learning rate.
        """
        for p in self.parameters:
            if p.value is None:
                continue
            if hasattr(p.value, "derivative"):
                if p.value.derivative is not None:
                    p.update(Scalar(p.value.data - self.lr * p.value.derivative))
            elif hasattr(p.value, "grad"):
                if p.value.grad is not None:
                    p.update(p.value - self.lr * p.value.grad)
