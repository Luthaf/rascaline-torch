# Integration between rascaline & pytorch

## Installation


```bash
pip install git+https://github.com/Luthaf/rascaline-torch
```

## Usage

```py
import torch
import rascaline

import rascaline_torch

HYPER_PARAMETERS = {
    "cutoff": 3,
    "max_radial": 6,
    "max_angular": 6,
    "atomic_gaussian_width": 0.3,
    "gradients": True,
    "cutoff_function": {"ShiftedCosine": {"width": 0.5}},
    "radial_basis": {"Gto": {}},
}

# wrap a rascaline calculator inside a torch.nn.Module (rascaline_torch.Calculator)
# you need to specify which neighboring species will be taken into account by
# the model
calculator = rascaline_torch.Calculator(
    rascaline.SphericalExpansion(**HYPER_PARAMETERS),
    species=[1, 6, 8]
)

# compute spherical expansion
frames = ase.io.read(...)
system = rascaline_torch.as_torch_system(frames[0], requires_grad=True)

# descriptor has three attributes: values, samples and features
descriptor = calculator(system)

# compute the property of interest with your model of choice
my_model = ...
energy = my_model(descriptor.values)

# backward propagate to extract forces
energy.backward()
forces = - positions.grad
```
