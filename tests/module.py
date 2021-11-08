import os
import copy
import tempfile
import unittest

import torch
import rascaline
from rascaline_torch import Calculator, System


class TestModule(unittest.TestCase):
    def test_save_module(self):
        hypers = {
            "cutoff": 3,
            "max_radial": 6,
            "max_angular": 6,
            "atomic_gaussian_width": 0.3,
            "gradients": False,
            "cutoff_function": {"ShiftedCosine": {"width": 0.5}},
            "radial_basis": {"Gto": {}},
        }

        model = Calculator(rascaline.SphericalExpansion(**hypers), [1, 6])
        jit_model = torch.jit.script(model)

        with tempfile.TemporaryDirectory() as directory:
            os.chdir(directory)
            jit_model.save("model.pt")

            loaded_model = torch.jit.load("model.pt")

        system = System(
            species=torch.tensor([1, 1], dtype=torch.int32),
            positions=torch.tensor([[0, 0, 0], [1, 1, 1]], dtype=torch.float64),
            cell=torch.zeros((3, 3), dtype=torch.float64),
        )

        result = loaded_model(system)
        expected = model(system)

        self.assertTrue(torch.all(result.values == expected.values))
