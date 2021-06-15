import unittest

import torch
import rascaline

import rascaline_torch

torch.manual_seed(0)


def gradcheck_rascaline(calculator, n_atoms, cell):
    # random positions & species in a small cell
    species = torch.randint(3, (n_atoms,))
    positions = cell / 2 * torch.randn((n_atoms, 3), dtype=torch.float64)
    positions += cell / 2
    cell = torch.tensor(
        [
            [cell, 0, 0],
            [0, cell, 0],
            [0, 0, cell],
        ]
    )

    def compute(calculator, positions, cell, species):
        values, _, _ = rascaline_torch.compute(calculator, positions, species, cell)
        return values

    positions.requires_grad = True

    return torch.autograd.gradcheck(
        compute,
        (calculator, positions, cell, species),
        eps=1e-6,
        atol=1e-6,
        fast_mode=True,
    )


class TestFiniteDifferences(unittest.TestCase):
    def test_spherical_expansion(self):
        hypers = {
            "cutoff": 3,
            "max_radial": 6,
            "max_angular": 6,
            "atomic_gaussian_width": 0.3,
            "gradients": True,
            "cutoff_function": {"ShiftedCosine": {"width": 0.5}},
            "radial_basis": {"Gto": {}},
        }

        self.assertTrue(
            gradcheck_rascaline(
                rascaline.SphericalExpansion(**hypers),
                n_atoms=50,
                cell=5,
            )
        )
