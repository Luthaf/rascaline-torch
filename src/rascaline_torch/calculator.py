import ctypes
from typing import List, Union

import torch
from equistore import TensorMap
from equistore._c_api import eqs_tensormap_t
from rascaline.calculators import CalculatorBase

from .system import System


class Calculator(torch.nn.Module):
    """Small wrapper around a rascaline calculator, integrating it with PyTorch"""

    def __init__(
        self,
        calculator,
        *,
        keep_forward_grad=False,
    ):
        super().__init__()

        if not isinstance(calculator, CalculatorBase):
            raise ValueError(
                "the calculator must be one of rascaline calculator, "
                f"got a value of type {calculator.__class__}"
            )

        self.keep_forward_grad = keep_forward_grad

        self.calculator = torch.classes.rascaline.Calculator(
            calculator.c_name, calculator.parameters
        )

    def forward(
        self,
        systems: Union[System, List[System]],
        *,
        keep_forward_grad=None,
    ):
        """TODO"""

        if keep_forward_grad is None:
            keep_forward_grad = self.keep_forward_grad

        if hasattr(systems, "species"):  # we can't use isinstance(systems, System)
            systems = [systems]

        options = {
            "keep_forward_grad": torch.tensor(bool(keep_forward_grad)),
        }

        ptr = torch.ops.rascaline.rascaline_autograd(self.calculator, systems, options)
        ptr = ctypes.cast(ptr, ctypes.POINTER(eqs_tensormap_t))
        return TensorMap._from_ptr(ptr)
