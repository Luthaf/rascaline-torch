import unittest

import torch
import rascaline
from rascaline_torch import Calculator, System

HYPERS = {
    "cutoff": 3,
    "max_radial": 6,
    "max_angular": 6,
    "atomic_gaussian_width": 0.3,
    "gradients": False,
    "cutoff_function": {"ShiftedCosine": {"width": 0.5}},
    "radial_basis": {"Gto": {}},
}


class TestErrors(unittest.TestCase):
    def setUp(self):
        self.calculator = Calculator(rascaline.SphericalExpansion(**HYPERS), [1, 2])

        self.system = System(
            positions=torch.tensor([[0, 0, 0], [1, 1, 1]], dtype=torch.float64),
            cell=torch.zeros((3, 3), dtype=torch.float64),
            species=torch.tensor([1, 2], dtype=torch.int32),
        )

    def test_bad_calculator(self):
        with self.assertRaises(ValueError) as cm:
            Calculator(3, [])

        self.assertEqual(
            str(cm.exception),
            "the calculator must be one of rascaline calculator, "
            "got a value of type <class 'int'>",
        )

        with self.assertRaises(ValueError) as cm:
            Calculator(rascaline.SphericalExpansion(**HYPERS), [1, "3"])

        self.assertEqual(
            str(cm.exception),
            "species must be provided as an array of integers",
        )

    def test_system_species_dtype(self):
        with self.assertRaises(RuntimeError) as cm:
            _ = System(
                species=self.system.species.clone().to(dtype=torch.float64),
                positions=self.system.positions,
                cell=self.system.cell,
            )

        self.assertEqual(
            str(cm.exception),
            "atomic species must be stored as 32-bit integers",
        )

    def test_system_species_shape(self):
        with self.assertRaises(RuntimeError) as cm:
            _ = System(
                species=self.system.species.reshape((-1, 1)),
                positions=self.system.positions,
                cell=self.system.cell,
            )

        self.assertEqual(
            str(cm.exception),
            "atomic species tensor must be a 1D tensor",
        )

    def test_system_species_contiguous(self):
        with self.assertRaises(RuntimeError) as cm:
            _ = System(
                species=torch.tensor([1, 2, 3, 4], dtype=torch.int32)[::2],
                positions=self.system.positions,
                cell=self.system.cell,
            )

        self.assertEqual(
            str(cm.exception),
            "atomic species must be stored as a contiguous tensor on CPU",
        )

    def test_system_species_positive(self):
        with self.assertRaises(RuntimeError) as cm:
            _ = System(
                species=torch.tensor([1, -2], dtype=torch.int32),
                positions=self.system.positions,
                cell=self.system.cell,
            )

        self.assertEqual(
            str(cm.exception),
            "all atomic species must be positive integers",
        )

    def test_system_positions_dtype(self):
        with self.assertRaises(RuntimeError) as cm:
            _ = System(
                species=self.system.species,
                positions=self.system.positions.clone().to(dtype=torch.float32),
                cell=self.system.cell,
            )

        self.assertEqual(
            str(cm.exception),
            "atomic positions must be stored as 64-bit floating point values",
        )

    def test_system_positions_shape(self):
        with self.assertRaises(RuntimeError) as cm:
            _ = System(
                species=self.system.species,
                positions=self.system.positions.reshape((-1, 3, 1)),
                cell=self.system.cell,
            )

        self.assertEqual(
            str(cm.exception),
            "the positions tensor must be a (n_atoms x 3) tensor",
        )

        with self.assertRaises(RuntimeError) as cm:
            _ = System(
                species=self.system.species,
                positions=torch.tensor(
                    [
                        [0, 0, 0],
                        [1, 1, 1],
                        [2, 2, 2],
                    ],
                    dtype=torch.float64,
                ),
                cell=self.system.cell,
            )

        self.assertEqual(
            str(cm.exception),
            "the positions tensor must be a (n_atoms x 3) tensor",
        )

    def test_system_positions_contiguous(self):
        positions = torch.tensor(
            [
                [0, 0, 0],
                [1, 1, 1],
                [2, 2, 2],
                [3, 3, 3],
            ],
            dtype=torch.float64,
        )[::2]

        with self.assertRaises(RuntimeError) as cm:
            _ = System(
                species=self.system.species,
                positions=positions,
                cell=self.system.cell,
            )

        self.assertEqual(
            str(cm.exception),
            "atomic positions must be stored as a contiguous tensor on CPU",
        )

    def test_system_cell_dtype(self):
        with self.assertRaises(RuntimeError) as cm:
            _ = System(
                species=self.system.species,
                positions=self.system.positions,
                cell=self.system.cell.clone().to(dtype=torch.float32),
            )

        self.assertEqual(
            str(cm.exception),
            "unit cell must be stored as 64-bit floating point values",
        )

    def test_system_cell_shape(self):
        with self.assertRaises(RuntimeError) as cm:
            _ = System(
                species=self.system.species,
                positions=self.system.positions,
                cell=self.system.cell.reshape((3, 3, 1)),
            )

        self.assertEqual(
            str(cm.exception),
            "the cell tensor must be a (3 x 3) tensor",
        )

        with self.assertRaises(RuntimeError) as cm:
            _ = System(
                species=self.system.species,
                positions=self.system.positions,
                cell=torch.zeros((3, 4), dtype=torch.float64),
            )

        self.assertEqual(
            str(cm.exception),
            "the cell tensor must be a (3 x 3) tensor",
        )

    def test_system_cell_contiguous(self):
        with self.assertRaises(RuntimeError) as cm:
            _ = System(
                species=self.system.species,
                positions=self.system.positions,
                cell=torch.zeros((6, 3), dtype=torch.float64)[::2],
            )

        self.assertEqual(
            str(cm.exception),
            "unit cell must be stored as a contiguous tensor on CPU",
        )

    def test_system_cell_grad(self):
        with self.assertRaises(RuntimeError) as cm:
            _ = System(
                species=self.system.species,
                positions=self.system.positions,
                cell=torch.zeros((3, 3), dtype=torch.float64, requires_grad=True),
            )

        self.assertEqual(
            str(cm.exception),
            "we can not track the gradient with respect to the cell yet",
        )


if __name__ == "__main__":
    unittest.main()
