import os

import rascaline
import torch

# force loading of rascaline library before trying to load _rascaline_torch.so
rascaline._c_lib._get_library()
# load the C++ operators and custom classes
torch.ops.load_library(os.path.join(os.path.dirname(__file__), "_rascaline_torch.so"))

from .calculator import Calculator
from .data import _register_cxx_torch_tensor_with_equistore
from .system import System, as_torch_system

_register_cxx_torch_tensor_with_equistore()

__all__ = [
    "System",
    "as_torch_system",
    "Calculator",
]
