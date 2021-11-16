import unittest

import torch
import rascaline

import rascaline_torch

torch.manual_seed(0)

HYPERS = {
    "cutoff": 3,
    "max_radial": 6,
    "max_angular": 6,
    "atomic_gaussian_width": 0.3,
    "gradients": True,
    "cutoff_function": {"ShiftedCosine": {"width": 0.5}},
    "radial_basis": {"Gto": {}},
}


class TestFiniteDifferences(unittest.TestCase):
    def _create_random_system(self, n_atoms, cell_size):
        species = torch.randint(3, (n_atoms,), dtype=torch.int)

        positions = cell_size / 2 * torch.randn((n_atoms, 3), dtype=torch.float64)
        positions += cell_size / 2

        cell = torch.tensor(
            [
                [cell_size, 0, 0],
                [0, cell_size, 0],
                [0, 0, cell_size],
            ],
            dtype=torch.float64,
        )

        return species, positions, cell

    def test_spherical_expansion(self):
        species, positions, cell = self._create_random_system(n_atoms=75, cell_size=5.0)
        positions.requires_grad = True

        model = rascaline_torch.Calculator(
            rascaline.SphericalExpansion(**HYPERS),
            species=[0, 1, 2],
        )

        def compute(positions, species, cell):
            system = rascaline_torch.System(
                positions=positions,
                species=species,
                cell=cell,
            )
            return model(system).values

        self.assertTrue(
            torch.autograd.gradcheck(
                compute,
                (positions, species, cell),
                eps=1e-12,
                atol=1e-2,
                fast_mode=True,
            )
        )

    def test_power_spectrum(self):
        species, positions, cell = self._create_random_system(n_atoms=75, cell_size=5.0)
        positions.requires_grad = True

        model = rascaline_torch.Calculator(
            rascaline.SoapPowerSpectrum(**HYPERS),
            species=[0, 1, 2],
        )

        def compute(positions, species, cell):
            system = rascaline_torch.System(
                positions=positions,
                species=species,
                cell=cell,
            )
            return model(system).values

        self.assertTrue(
            torch.autograd.gradcheck(
                compute,
                (positions, species, cell),
                eps=1e-12,
                atol=1e-2,
                fast_mode=True,
            )
        )

    def test_forces_gradients(self):
        species, positions, cell = self._create_random_system(n_atoms=75, cell_size=5.0)
        positions.requires_grad = True
        system = rascaline_torch.System(
            positions=positions,
            species=species,
            cell=cell,
        )

        model = rascaline_torch.Calculator(
            rascaline.SphericalExpansion(**HYPERS),
            species=[0, 1, 2],
        )

        def compute(weights):
            energy = model(system).values.sum(dim=0) @ weights.T
            (forces,) = torch.autograd.grad(
                energy,
                system.positions,
                grad_outputs=-torch.ones_like(energy),
                create_graph=True,
                retain_graph=True,
            )
            return forces

        weights = torch.rand((1, 882), dtype=torch.float64, requires_grad=True)
        # check that we can compute gradients of forces w.r.t. weights of a model
        self.assertTrue(
            torch.autograd.gradcheck(
                compute,
                (weights),
                eps=1e-12,
                atol=1e-2,
                fast_mode=True,
            )
        )
