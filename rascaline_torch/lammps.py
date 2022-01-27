from typing import Dict, Optional
import json

import torch
from torch import Tensor

from .system import System
from .calculator import Calculator


def save_lammps_model(module, path, freeze=True):
    _validate_types(module.forward)
    cutoff = _get_cutoff(module)

    module.train(False)
    script = torch.jit.script(module)
    if freeze:
        script = torch.jit.freeze(script)

    script.save(path, _extra_files={"cutoff": str(cutoff)})


def _validate_types(forward):
    error_prefix = "this model can not be exported to LAMMPS"

    if "return" not in forward.__annotations__:
        raise TypeError(f"{error_prefix}: missing return type for forward")
    if forward.__annotations__["return"] != Tensor:
        raise TypeError(f"{error_prefix}: return type must be torch.Tensor")

    input_types = [
        type_ for name, type_ in forward.__annotations__.items() if name != "return"
    ]

    if len(input_types) != 2:
        raise TypeError(
            f"{error_prefix}: expected 2 arguments in the forward function, "
            f"got {len(input_types)}"
        )

    if input_types[0] is not System:
        raise TypeError(
            f"{error_prefix}: the first argument of forward must be "
            "a `rascaline_torch.System`"
        )

    if input_types[1] is not Optional[Dict[str, Tensor]]:
        raise TypeError(
            f"{error_prefix}:  the second argument of forward must be "
            "an `Optional[Dict[str, torch.Tensor]]`"
        )


def _extract_potential_cutoffs(module):
    """
    Recursively extract the cutoff requested by each sub module of ``module``
    which are actually :py:class:`rascaline_torch.Calculator`
    """
    potential_cutoffs = []

    if isinstance(module, Calculator):
        hyperparameters = json.loads(module._rascaline_parameters)
        potential_cutoffs.append(float(hyperparameters["cutoff"]))
    else:
        for name, children in module.named_children():
            potential_cutoffs += _extract_potential_cutoffs(children)

    return potential_cutoffs


def _get_cutoff(module):
    potential_cutoffs = set(_extract_potential_cutoffs(module))
    if len(potential_cutoffs) == 0:
        raise TypeError(
            "could not find a rascaline cutoff in this module, is there a "
            "`rascaline_torch.Calculator` somewhere in the computational graph?"
        )
    elif len(potential_cutoffs) > 1:
        raise TypeError(
            f"found multiple different cutoff ({potential_cutoffs}) in this "
            "module, this is not supported yet"
        )

    return potential_cutoffs.pop()
