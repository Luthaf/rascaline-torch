import unittest

import ase
import rascaline
import torch

import rascaline_torch

torch.manual_seed(0)

HYPERS = {
    "cutoff": 3,
    "max_radial": 2,
    "max_angular": 0,
    "atomic_gaussian_width": 0.3,
    "center_atom_weight": 1.0,
    "cutoff_function": {"ShiftedCosine": {"width": 0.5}},
    "radial_basis": {"Gto": {}},
}


class TestFiniteDifferences(unittest.TestCase):
    def _create_random_system(self, n_atoms, cell_size):
        species = torch.randint(3, (n_atoms,), dtype=torch.int)

        cell = ase.cell.Cell.new(
            [cell_size, 1.4 * cell_size, 0.8 * cell_size, 90, 80, 110]
        )
        cell = torch.tensor(cell[:], dtype=torch.float64)

        positions = torch.rand((n_atoms, 3), dtype=torch.float64) @ cell

        return species, positions, cell

    def test_spherical_expansion_positions_grad(self):
        species, positions, cell = self._create_random_system(n_atoms=75, cell_size=5.0)
        positions.requires_grad = True

        calculator = rascaline_torch.Calculator(rascaline.SphericalExpansion(**HYPERS))

        def compute(positions, species, cell):
            system = rascaline_torch.System(
                positions=positions,
                species=species,
                cell=cell,
            )
            descriptor = calculator(system)
            descriptor = descriptor.components_to_properties("spherical_harmonics_m")
            descriptor = descriptor.keys_to_properties("spherical_harmonics_l")

            descriptor = descriptor.keys_to_samples("species_center")
            descriptor = descriptor.keys_to_properties("species_neighbor")

            return descriptor.block().values

        self.assertTrue(
            torch.autograd.gradcheck(
                compute,
                (positions, species, cell),
                fast_mode=True,
            )
        )

    def test_spherical_expansion_cell_grad(self):
        species, positions, cell = self._create_random_system(n_atoms=75, cell_size=5.0)

        original_cell = cell.clone()
        cell.requires_grad = True

        calculator = rascaline_torch.Calculator(rascaline.SphericalExpansion(**HYPERS))

        def compute(positions, species, cell):
            # modifying the cell for numerical gradients should also displace
            # the atoms
            fractional = positions @ torch.linalg.inv(original_cell)
            positions = fractional @ cell.detach()

            system = rascaline_torch.System(
                positions=positions,
                species=species,
                cell=cell,
            )
            descriptor = calculator(system)
            descriptor = descriptor.components_to_properties("spherical_harmonics_m")
            descriptor = descriptor.keys_to_properties("spherical_harmonics_l")

            descriptor = descriptor.keys_to_samples("species_center")
            descriptor = descriptor.keys_to_properties("species_neighbor")

            return descriptor.block().values

        self.assertTrue(
            torch.autograd.gradcheck(
                compute,
                (positions, species, cell),
                fast_mode=True,
            )
        )

    def test_power_spectrum_positions_grad(self):
        species, positions, cell = self._create_random_system(n_atoms=75, cell_size=5.0)
        positions.requires_grad = True

        calculator = rascaline_torch.Calculator(rascaline.SoapPowerSpectrum(**HYPERS))

        def compute(positions, species, cell):
            system = rascaline_torch.System(
                positions=positions,
                species=species,
                cell=cell,
            )
            descriptor = calculator(system)

            descriptor = descriptor.keys_to_samples("species_center")
            descriptor = descriptor.keys_to_properties(
                ["species_neighbor_1", "species_neighbor_2"]
            )

            return descriptor.block().values

        self.assertTrue(
            torch.autograd.gradcheck(
                compute,
                (positions, species, cell),
                fast_mode=True,
            )
        )

    def test_power_spectrum_cell_grad(self):
        species, positions, cell = self._create_random_system(n_atoms=75, cell_size=5.0)

        original_cell = cell.clone()
        cell.requires_grad = True

        calculator = rascaline_torch.Calculator(rascaline.SoapPowerSpectrum(**HYPERS))

        def compute(positions, species, cell):
            # modifying the cell for numerical gradients should also displace
            # the atoms
            fractional = positions @ torch.linalg.inv(original_cell)
            positions = fractional @ cell.detach()

            system = rascaline_torch.System(
                positions=positions,
                species=species,
                cell=cell,
            )
            descriptor = calculator(system)

            descriptor = descriptor.keys_to_samples("species_center")
            descriptor = descriptor.keys_to_properties(
                ["species_neighbor_1", "species_neighbor_2"]
            )

            return descriptor.block().values

        self.assertTrue(
            torch.autograd.gradcheck(
                compute,
                (positions, species, cell),
                fast_mode=True,
            )
        )


if __name__ == "__main__":
    unittest.main()
