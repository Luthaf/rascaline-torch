import os
import copy
import tempfile
import unittest

import torch
import rascaline
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


# class TestCalculator(unittest.TestCase):
#     def setUp(self):
#         self.system = System(
#             species=torch.tensor([1, 1], dtype=torch.int32),
#             positions=torch.tensor([[0, 0, 0], [1, 1, 1]], dtype=torch.float64),
#             cell=torch.zeros((3, 3), dtype=torch.float64),
#         )

#     def test_save_calculator(self):
#         model = Calculator(rascaline.SphericalExpansion(**HYPERS), [1, 6])
#         jit_model = torch.jit.script(model)

#         with tempfile.TemporaryDirectory() as directory:
#             os.chdir(directory)
#             jit_model.save("model.pt")

#             loaded_model = torch.jit.load("model.pt")

#         result = loaded_model(self.system)
#         expected = model(self.system)

#         self.assertTrue(torch.all(result.values == expected.values))

#     def test_invalid_options(self):
#         model = Calculator(rascaline.SphericalExpansion(**HYPERS), [1, 6])

#         with self.assertRaises(RuntimeError) as cm:
#             model(self.system, {"unknown-option": torch.zeros(3, 3)})
#         self.assertEqual(
#             str(cm.exception),
#             "got unknown option in rascaline calculator: 'unknown-option'",
#         )

#     def test_densified_species(self):
#         model = Calculator(rascaline.SphericalExpansion(**HYPERS), [1, 6])
#         descriptor_1_6 = model(self.system)

#         model = Calculator(rascaline.SphericalExpansion(**HYPERS), [1])
#         descriptor_1 = model(self.system)

#         self.assertEqual(descriptor_1_6.values.shape, (2, 588))
#         self.assertEqual(descriptor_1.values.shape, (2, 294))
