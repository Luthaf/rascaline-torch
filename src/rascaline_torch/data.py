import ctypes

import equistore.core.data
import torch
from equistore.core._c_api import eqs_array_t

_eqs_array_to_tensor = torch.ops.rascaline.eqs_array_to_tensor


# small wrapper around a torch.Tensor that keeps it's parent alive
# (see `equistore.data.register_external_data_wrapper`)
class TorchDataArray(torch.Tensor):
    def __new__(cls, eqs_array, parent):
        eqs_array_ptr = ctypes.POINTER(eqs_array_t)(eqs_array)
        ptr_as_int = ctypes.cast(eqs_array_ptr, ctypes.c_void_p).value
        tensor = _eqs_array_to_tensor(ptr_as_int)

        obj = tensor.as_subclass(TorchDataArray)
        obj._parent = parent

        return obj


def _register_cxx_torch_tensor_with_equistore():
    equistore.core.data.register_external_data_wrapper(
        "equistore_torch::TorchDataArray",
        TorchDataArray,
    )
