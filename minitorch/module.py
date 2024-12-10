from __future__ import annotations

from typing import Any, Optional, Sequence, Tuple


class Module:
    """Modules form a tree that store parameters and other
    submodules. They make up the basis of neural network stacks.

    Attributes
    ----------
        _modules : Storage of the child modules
        _parameters : Storage of the module's parameters
        training : Whether the module is in training mode or evaluation mode

    """

    def __init__(self) -> None:
        self._modules = {}
        self._parameters = {}
        self.training = True

    def modules(self) -> Sequence[Module]:
        """Return the direct child modules of this module."""
        m = self.__dict__["_modules"]
        return list(m.values())

    def train(self) -> None:
        """Set the mode of this module and all its descendent modules to `train`."""
        self.training = True
        for module in self.modules():
            module.train()

    def eval(self) -> None:
        """Set the mode of this module and all its descendent modules to `eval`."""
        self.training = False
        for module in self.modules():
            module.eval()

    def named_parameters(self) -> Sequence[Tuple[str, Parameter]]:
        """Collect all the parameters of this module and its descendents.

        Returns
        -------
            list of pairs: Contains the name and `Parameter` of each ancestor parameter.

        """
        results = []
        for name, param in self._parameters.items():
            results.append((name, param))
        for name, module in self._modules.items():
            for subname, param in module.named_parameters():
                results.append((f"{name}.{subname}", param))
        return results

    def parameters(self) -> Sequence[Parameter]:
        """Enumerate over all the parameters of this module and its descendents."""
        # TODO: Implement for Task 0.4.
        param = []
        for i in self._parameters.values():
            param.append(i)

        for m in self._modules.values():
            m1 = m.parameters()
            for j in m1:
                param.append(j)
        return param

    def add_parameter(self, k: str, v: Any) -> Parameter:
        """Manually add a parameter. Useful helper for scalar parameters.

        Args:
        ----
            k: Local name of the parameter.
            v: Value for the parameter.

        Returns:
        -------
            Newly created parameter.

        """
        val = Parameter(v, k)
        self._parameters[k] = val
        return val

    def __setattr__(self, key: str, val: Parameter) -> None:
        if isinstance(val, Parameter):
            self._parameters[key] = val
        elif isinstance(val, Module):
            self._modules[key] = val
        else:
            super().__setattr__(key, val)

    def __getattr__(self, key: str) -> Any:
        if key in self._parameters:
            return self._parameters[key]

        if key in self._modules:
            return self._modules[key]
        return super().__getattribute__(key)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Call the module's forward method.

        Args:
        ----
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
        -------
            Any: The result of the forward method.

        """
        return self.forward(*args, **kwargs)

    def __repr__(self) -> str:
        def _addindent(s_: str, numSpaces: int) -> str:
            s = s_.split("\n")
            if len(s) == 1:
                return s_
            first = s.pop(0)
            s = [(numSpaces * " ") + line for line in s]
            s = "\n".join(s)
            s = first + "\n" + s
            return s

        child_lines = []

        for key, module in self._modules.items():
            mod_str = repr(module)
            mod_str = _addindent(mod_str, 2)
            child_lines.append("(" + key + "): " + mod_str)
        lines = child_lines

        main_str = self.__class__.__name__ + "("
        if lines:
            # simple one-liner info, which most builtin Modules will use
            main_str += "\n  " + "\n  ".join(lines) + "\n"

        main_str += ")"
        return main_str


class Parameter:
    """A Parameter is a special container stored in a `Module`.

    It is designed to hold a `Variable`, but we allow it to hold
    any value for testing.
    """

    def __init__(self, x: Any, name: Optional[str] = None) -> None:
        self.value = x
        self.name = name
        if hasattr(x, "requires_grad_"):
            self.value.requires_grad_(True)
            if self.name:
                self.value.name = self.name

    def update(self, x: Any) -> None:
        """Update the parameter value."""
        self.value = x
        if hasattr(x, "requires_grad_"):
            self.value.requires_grad_(True)
            if self.name:
                self.value.name = self.name

    def __repr__(self) -> str:
        return repr(self.value)

    def __str__(self) -> str:
        return str(self.value)
