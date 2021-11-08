import unittest

import torch
import rascaline

import rascaline_torch

torch.manual_seed(0)


def gradcheck_rascaline(calculator, n_atoms, cell):
    # random positions & species in a small cell
    species = torch.randint(3, (n_atoms,), dtype=torch.int)

    positions = cell / 2 * torch.randn((n_atoms, 3), dtype=torch.float64)
    positions += cell / 2
    positions.requires_grad = True

    cell = torch.tensor(
        [
            [cell, 0, 0],
            [0, cell, 0],
            [0, 0, cell],
        ],
        dtype=torch.float64,
    )

    model = rascaline_torch.Calculator(calculator, species)

    def compute(positions, species, cell):
        system = rascaline_torch.System(
            positions=positions,
            species=species,
            cell=cell,
        )
        return model(system).values

    return torch.autograd.gradcheck(
        compute,
        (positions, species, cell),
        eps=1e-12,
        atol=1e-2,
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
