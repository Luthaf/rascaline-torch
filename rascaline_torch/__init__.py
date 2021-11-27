import os

import torch
import rascaline

# force loading of rascaline library before trying to load _rascaline_torch.so
rascaline.clib._get_library()
# load the C++ operators and custom classes
torch.ops.load_library(os.path.join(os.path.dirname(__file__), "_rascaline_torch.so"))

from .system import System, as_torch_system
from .calculator import IndexesTensor, Calculator, Descriptor

__all__ = [
    "System",
    "as_torch_system",
    "IndexesTensor",
    "Calculator",
    "Descriptor",
]
