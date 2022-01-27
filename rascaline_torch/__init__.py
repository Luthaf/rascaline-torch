import os

import torch
import rascaline

# force loading of rascaline library before trying to load _rascaline_torch.so
rascaline.clib._get_library()
# load the C++ operators and custom classes
torch.ops.load_library(os.path.join(os.path.dirname(__file__), "_rascaline_torch.so"))

from .system import System, as_torch_system  # noqa
from .calculator import IndexesTensor, Calculator, Descriptor  # noqa
from .lammps import save_lammps_model  # noqa
