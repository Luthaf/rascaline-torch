from __future__ import annotations

import os
import json
from collections import namedtuple
from typing import Dict, List, Any, Union, NamedTuple, Optional, TYPE_CHECKING

import torch
import numpy as np

import rascaline
from rascaline.calculators import CalculatorBase

_HERE = os.path.dirname(__file__)

# force loading of rascaline library before trying to load _rascaline_torch.so
rascaline.clib._get_library()

torch.ops.load_library(os.path.join(_HERE, "_rascaline_torch.so"))

_get_interned_string = torch.ops.rascaline.get_interned_string


class IndexesTensor(object):
    def __init__(self, data: torch.Tensor, names: List[str]):
        self.data = data
        self.names = names

    def __repr__(self):
        return f"IndexesTensor(\n    names={self.names},\n    tensor={self.data}\n)"

    def index(self, name: str):
        assert torch.jit.isinstance(name, str)
        return self.data[:, self.names.index(name)]


if TYPE_CHECKING:
    # define a dummy class with the same interface to make mypy happy
    class System:
        def __init__(
            self,
            species: torch.Tensor,
            position: torch.Tensor,
            cell: torch.Tensor,
        ):
            pass

        @property
        def species(self) -> torch.Tensor:
            pass

        @property
        def positions(self) -> torch.Tensor:
            pass

        @property
        def cell(self) -> torch.Tensor:
            pass


else:
    System = torch.classes.rascaline.System


Descriptor = NamedTuple(
    "Descriptor",
    [
        ("values", torch.Tensor),
        ("samples", IndexesTensor),
        ("features", IndexesTensor),
    ],
)


def as_torch_system(frame, requires_grad=False):
    system = rascaline.systems.wrap_system(frame)

    return System(
        torch.tensor(
            system.species(),
            requires_grad=False,
            dtype=torch.int,
        ),
        torch.tensor(
            system.positions(),
            requires_grad=requires_grad,
            dtype=torch.double,
        ),
        torch.tensor(
            system.cell(),
            requires_grad=False,
            dtype=torch.double,
        ),
    )


class Calculator(torch.nn.Module):
    def __init__(self, calculator, species):
        """TODO"""
        super().__init__()

        if not isinstance(calculator, CalculatorBase):
            raise ValueError(
                "the calculator must be one of rascaline calculator, "
                f"got a value of type {calculator.__class__}"
            )

        self.register_buffer(
            "cutoff", torch.tensor(json.loads(calculator.parameters)["cutoff"])
        )

        self.calculator = torch.classes.rascaline.Calculator(
            calculator.c_name, calculator.parameters
        )

        if not np.can_cast(np.asarray(species), np.int32, casting="same_kind"):
            raise ValueError("species must be provided as an array of integers")

        species = np.unique(np.array(species, dtype=np.int32))
        if calculator.c_name == "spherical_expansion":
            all_species = [[s] for s in species]
        elif calculator.c_name == "soap_power_spectrum":
            all_species = []
            for s1 in species:
                for s2 in species:
                    if s1 <= s2:
                        all_species.append((s1, s2))
        else:
            raise Exception("unknown calculator, please edit this file")

        self.densify_species = torch.tensor(all_species)
        self.n_features = calculator.features_count() * len(all_species)

    def forward(
        self,
        system: System,
        options: Optional[Dict[str, torch.Tensor]] = None,
    ):
        """TODO"""
        if options is None:
            options = {}
        options["densify_species"] = self.densify_species

        result = torch.ops.rascaline.compute(self.calculator, system, options)

        return Descriptor(
            values=result["values"],
            samples=IndexesTensor(
                data=result["samples"],
                names=[_get_interned_string(i) for i in result["samples_names"]],
            ),
            features=IndexesTensor(
                data=result["features"],
                names=[_get_interned_string(i) for i in result["features_names"]],
            ),
        )
