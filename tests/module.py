import unittest

import torch
import rascaline

from rascaline_torch import RascalineModule

HYPERS = {
    "cutoff": 3,
    "max_radial": 6,
    "max_angular": 6,
    "atomic_gaussian_width": 0.3,
    "gradients": False,
    "cutoff_function": {"ShiftedCosine": {"width": 0.5}},
    "radial_basis": {"Gto": {}},
}


class TestNNModule(unittest.TestCase):
    def test_automatic_do_gradient(self):
        """
        Check that the module adds the "gradients" parameter to the hypers if
        it needs it.
        """
        calculator = rascaline.SphericalExpansion(**HYPERS)

        module = RascalineModule(calculator)
        self.assertFalse(module.do_gradient)

        positions = torch.tensor(
            [
                [0.8012, 1.9272, 1.5049],
                [1.9359, 2.1641, 2.0911],
                [-0.0613, 1.0889, 2.8169],
                [2.1420, 3.2371, 2.0217],
                [1.6448, 0.7276, 1.5679],
                [1.4386, 2.1708, 2.1019],
            ],
            requires_grad=True,
        )
        species = torch.tensor([1, 1, 1, 6, 6, 1])
        cell = torch.tensor([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
        values, _, _ = module(positions, species, cell)

        self.assertTrue(module.do_gradient)

        # mimic a linear regression
        weights = torch.randn((588, 1), dtype=torch.float64)
        results = values @ weights

        # this should work fine now
        results.squeeze().backward(torch.tensor([1.0 for _ in range(6)]))
