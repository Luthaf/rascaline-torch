import os
from collections import namedtuple
from typing import Dict

import torch
import numpy as np

import rascaline
from rascaline.calculators import CalculatorBase

_HERE = os.path.dirname(__file__)

# force loading of rascaline library before trying to load _rascaline_torch.so
rascaline.clib._get_library()

torch.ops.load_library(os.path.join(_HERE, "_rascaline_torch.so"))


System = namedtuple("System", ["species", "positions", "cell"])
Descriptor = namedtuple("Descriptor", ["values", "samples", "features"])


def as_torch_system(frame, requires_grad=False):
    system = rascaline.systems.wrap_system(frame)

    return System(
        species=torch.tensor(
            system.species(),
            requires_grad=False,
            dtype=torch.int,
        ),
        positions=torch.tensor(
            system.positions(),
            requires_grad=requires_grad,
            dtype=torch.double,
        ),
        cell=torch.tensor(
            system.cell(),
            requires_grad=False,
            dtype=torch.double,
        ),
    )


class Calculator(torch.nn.Module):
    def __init__(self, calculator, species):
        super().__init__()

        if not isinstance(calculator, CalculatorBase):
            raise ValueError(
                "the calculator must be one of rascaline calculator, "
                f"got a value of type {calculator.__class__}"
            )

        self.calculator = torch.classes.rascaline.Calculator(
            calculator.c_name, calculator.parameters
        )

        if not np.can_cast(np.asarray(species), np.int32, casting="same_kind"):
            raise ValueError("species must be provided as an array of integers")

        species = np.unique(np.array(species, dtype=np.int32))
        if calculator.c_name == "spherical_expansion":
            all_species = list(species)
        elif calculator.c_name == "soap_power_spectrum":
            all_species = []
            for s1 in species:
                for s2 in species:
                    if s1 <= s2:
                        all_species.append((s1, s2))
        else:
            raise Exception("unknown calculator, please edit this file")

        self.options = {"TODO": torch.tensor(all_species)}
        self.n_features = calculator.features_count() * len(all_species)

    def forward(self, system: System):
        # C++ code uses dictionaries since there are no named tuples there, and
        # pytorch can not track calculations on tensor members of custom
        # classes. Python uses named tuples to limit what the user can stick in
        # the "system"; and for a nicer access API (`system.positions` vs
        # system["positions"]). So here, we need to convert between the two
        system_dict = {
            "species": system.species,
            "positions": system.positions,
            "cell": system.cell,
        }

        result = torch.ops.rascaline.compute(self.calculator, system_dict, self.options)

        return Descriptor(
            values=result["values"],
            samples=result["samples"],
            features=result["features"],
        )
