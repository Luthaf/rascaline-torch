# Integration between rascaline & pytorch

## Installation


```bash
pip install git+https://github.com/Luthaf/rascaline-torch
```

## Usage

```py
import torch
import rascaline

from rascaline_torch import RascalineModule

HYPER_PARAMETERS = {
    "cutoff": 3,
    "max_radial": 6,
    "max_angular": 6,
    "atomic_gaussian_width": 0.3,
    "gradients": True,
    "cutoff_function": {"ShiftedCosine": {"width": 0.5}},
    "radial_basis": {"Gto": {}},
}

# wrap a rascaline calculator in a torch.nn.Module
calculator = rascaline.SphericalExpansion(**HYPER_PARAMETERS)
calculator = RascalineModule(calculator)

# compute spherical expansion
n_atoms = ...
positions = torch.tensor(..., shape=(n_atoms, 3), dtype=torch.float64)
positions.requires_grad = True
cell = torch.tensor(..., shape=(3, 3), dtype=torch.float64)
species = torch.tensor(..., shape=(n_atoms,), dtype=torch.int32)

# you can usually ignore samples & features metadata here
spherical_expansion, samples, features = calculator(positions, species, cell)

# compute the property of interest with your model of choice
my_model = ...
energy = my_model(spherical_expansion)

# backward propagate to extract forces
energy.backward()
forces = - positions.grad
```
