from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Iterable, Tuple, Protocol


def central_difference(f: Any, *vals: Any, arg: int = 0, epsilon: float = 1e-6) -> Any:
    """Computes an approximation to the derivative of `f` with respect to one arg.

    Args:
    ----
        f (Any): Arbitrary function from n-scalar args to one value.
        *vals (Any): n-float values x_0 ... x_{n-1}.
        arg (int, optional): The number i of the arg to compute the derivative. Defaults to 0.
        epsilon (float, optional): A small constant. Defaults to 1e-6.

    Returns:
    -------
        Any: An approximation of f'_i(x_0, ..., x_{n-1}).

    """
    vals_plus = list(vals)
    vals_minus = list(vals)
    vals_plus[arg] += epsilon
    vals_minus[arg] -= epsilon
    f_plus = f(*vals_plus)
    f_minus = f(*vals_minus)
    derivative = (f_plus - f_minus) / (2 * epsilon)
    return derivative


class Variable(Protocol):
    """Protocol defining the interface for a variable in the computation graph."""

    def accumulate_derivative(self, x: Any) -> None:
        """Accumulates the derivative for this variable.

        Args:
        ----
            x (Any): The derivative to be accumulated.

        """
        ...

    @property
    def unique_id(self) -> int:
        """Returns a unique identifier for this variable.

        Returns
        -------
            int: A unique integer identifier for the variable.

        """
        ...

    def is_leaf(self) -> bool:
        """Checks if this variable is a leaf node in the computation graph.

        Returns
        -------
            bool: True if the variable is a leaf node, False otherwise.

        """
        ...

    def is_constant(self) -> bool:
        """Checks if this variable is a constant.

        Returns
        -------
            bool: True if the variable is a constant, False otherwise.

        """
        ...

    @property
    def parents(self) -> Iterable[Variable]:
        """Returns the parent variables in the computation graph.

        Returns
        -------
            Iterable[Variable]: An iterable of parent Variable objects.

        """
        ...

    def chain_rule(self, d_output: Any) -> Iterable[Tuple[Variable, Any]]:
        """Applies the chain rule to compute gradients for this variable's parents.

        Args:
        ----
            d_output (Any): The gradient of the final output with respect to this variable.

        Returns:
        -------
            Iterable[Tuple[Variable, Any]]: An iterable of tuples, each containing a parent Variable
            and its corresponding gradient.

        """
        ...


def topological_sort(variable: Variable) -> Iterable[Variable]:
    """Computes the topological order of the computation graph.

    Args:
    ----
        variable (Variable): The right-most variable.

    Returns:
    -------
        Iterable[Variable]: Non-constant Variables in topological order starting from the right.

    """
    visited = set()
    sorted_vars = []

    def dfs(v: Variable) -> None:
        """Depth-first search helper function for topological sorting.

        Args:
        ----
            v (Variable): The current variable being processed.

        """
        if v.unique_id in visited:
            return
        visited.add(v.unique_id)

        if not v.is_constant():
            for parent in v.parents:
                dfs(parent)

        sorted_vars.append(v)

    dfs(variable)

    return reversed(sorted_vars)


def backpropagate(variable: Variable, deriv: Any) -> None:
    """Runs backpropagation on the computation graph to compute derivatives for the leaf nodes.

    Args:
    ----
        variable (Variable): The right-most variable (final output of the computational graph).
        deriv (Any): The derivative of the output with respect to itself (usually 1 for a scalar).

    """
    sorted_vars = list(topological_sort(variable))
    derivatives = {variable.unique_id: deriv}

    for var in sorted_vars:
        d_output = derivatives.get(var.unique_id, 0)
        if var.is_leaf():
            var.accumulate_derivative(d_output)
            continue

        for parent_var, local_grad in var.chain_rule(d_output):
            if parent_var.unique_id in derivatives:
                derivatives[parent_var.unique_id] += local_grad
            else:
                derivatives[parent_var.unique_id] = local_grad


@dataclass
class Context:
    """Context class used by `Function` to store information during the forward pass.

    Attributes
    ----------
        no_grad (bool): If True, gradients are not computed. Defaults to False.
        saved_values (Tuple[Any, ...]): Values saved for use in backpropagation. Defaults to an empty tuple.

    """

    no_grad: bool = False
    saved_values: Tuple[Any, ...] = ()

    def save_for_backward(self, *values: Any) -> None:
        """Store the given `values` if they need to be used during backpropagation.

        Args:
        ----
            *values (Any): Values to be saved for backpropagation.

        """
        if self.no_grad:
            return
        self.saved_values = values

    @property
    def saved_tensors(self) -> Tuple[Any, ...]:
        """Returns the saved values for use in backpropagation.

        Returns
        -------
            Tuple[Any, ...]: The saved values.

        """
        return self.saved_values
