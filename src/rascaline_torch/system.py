from typing import TYPE_CHECKING

import rascaline
import torch
from torch import Tensor

if TYPE_CHECKING:
    # define a dummy class with the same interface to make mypy happy
    class System:
        def __init__(self, species: Tensor, position: Tensor, cell: Tensor):
            pass

        @property
        def species(self) -> Tensor:
            pass

        @property
        def positions(self) -> Tensor:
            pass

        @property
        def cell(self) -> Tensor:
            pass

else:
    System = torch.classes.rascaline.System


def as_torch_system(frame, positions_requires_grad=False, cell_requires_grad=False):
    system = rascaline.systems.wrap_system(frame)

    return System(
        species=torch.tensor(
            system.species(),
            requires_grad=False,
            dtype=torch.int32,
        ),
        positions=torch.tensor(
            system.positions(),
            requires_grad=positions_requires_grad,
            dtype=torch.double,
        ),
        cell=torch.tensor(
            system.cell(),
            requires_grad=cell_requires_grad,
            dtype=torch.double,
        ),
    )
