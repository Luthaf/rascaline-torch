import os
import torch
import ctypes
import json

import rascaline
from rascaline.calculators import CalculatorBase

_HERE = os.path.dirname(__file__)

# force loading of rascaline library before trying to load _rascaline_torch.so
rascaline.clib._get_library()

torch.ops.load_library(os.path.join(_HERE, "_rascaline_torch.so"))

TorchCalculator = torch.classes.rascaline.TorchCalculator


def _calculator_to_torch(calculator):
    # safety: we guarantee that _rascaline_torch.so and the rascaline python
    # module use the same librascaline.so, so here we can pass a pointer from
    # the rascaline module to _rascaline_torch.so
    ptr = ctypes.cast(calculator._as_parameter_, ctypes.c_void_p).value
    return TorchCalculator(calculator.c_name, ptr)


def compute(calculator, positions, species, cell):
    if not isinstance(calculator, (CalculatorBase, TorchCalculator)):
        raise Exception("expected a rascaline calculator as the first parameter")

    if isinstance(calculator, CalculatorBase):
        calculator = _calculator_to_torch(calculator)

    return torch.ops.rascaline.compute(calculator, positions, species, cell)


class RascalineModule(torch.nn.Module):
    def __init__(self, calculator):
        super().__init__()
        assert isinstance(calculator, CalculatorBase)
        self._calculator = calculator
        self._torch_calculator = _calculator_to_torch(self._calculator)
        self._do_gradient = json.loads(self._calculator.parameters)["gradients"]
        self._positions = None

    def forward(self, positions, species, cell):
        if positions.requires_grad:
            self.do_gradient = True
        else:
            self.do_gradient = False

        return torch.ops.rascaline.compute(
            self._torch_calculator, positions, species, cell
        )

    def features_count(self):
        return self._calculator.features_count()

    @property
    def do_gradient(self):
        return self._do_gradient

    @do_gradient.setter
    def do_gradient(self, toggle):
        if self._do_gradient == toggle:
            return

        self._do_gradient = toggle
        hypers = json.loads(self._calculator.parameters)
        hypers["gradients"] = self._do_gradient
        # recreate a calculator with the new hypers
        self._calculator = CalculatorBase(
            self._calculator.c_name,
            hypers,
        )
        self._torch_calculator = _calculator_to_torch(self._calculator)
