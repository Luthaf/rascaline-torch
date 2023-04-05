import os
import tempfile
import unittest

import rascaline
import torch

from rascaline_torch import Calculator, System

HYPERS = {
    "cutoff": 3,
    "max_radial": 6,
    "max_angular": 6,
    "atomic_gaussian_width": 0.3,
    "center_atom_weight": 1.0,
    "cutoff_function": {"ShiftedCosine": {"width": 0.5}},
    "radial_basis": {"Gto": {}},
}


class TestCalculator(unittest.TestCase):
    def setUp(self):
        self.system = System(
            species=torch.tensor([1, 1], dtype=torch.int32),
            positions=torch.tensor([[0, 0, 0], [1, 1, 1]], dtype=torch.float64),
            cell=torch.zeros((3, 3), dtype=torch.float64),
        )

    def test_save_calculator(self):
        # TODO: re-enable this, it might require registering TensorMap and
        # all corresponding functions with PyTorch
        pass
        # model = Calculator(rascaline.SphericalExpansion(**HYPERS))
        # jit_model = torch.jit.script(model)

        # with tempfile.TemporaryDirectory() as directory:
        #     os.chdir(directory)
        #     jit_model.save("model.pt")

        #     loaded_model = torch.jit.load("model.pt")

        # result = loaded_model(self.system)
        # expected = model(self.system)

        # self.assertTrue(torch.all(result.values == expected.values))


if __name__ == "__main__":
    unittest.main()
