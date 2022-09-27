#include <torch/torch.h>

#include "equistore_torch.hpp"
#include "rascaline_torch.hpp"
using namespace rascaline;
using namespace equistore;

// we are casting pointers to 64-bit integers, check that this is fine
static_assert(sizeof(int64_t) == sizeof(void*), "only 64-bit targets are supported");

int64_t rascaline_autograd_py(
    torch::intrusive_ptr<TorchCalculator> calculator,
    std::vector<torch::intrusive_ptr<TorchSystem>> systems,
    torch::Dict<std::string, torch::Tensor> options
) {
    eqs_tensormap_t* tensor_map = rascaline::rascaline_autograd(
        std::move(calculator),
        std::move(systems),
        std::move(options)
    );

    // cast the pointer to an integer to be able to pass it to Python
    return reinterpret_cast<int64_t>(tensor_map);
}

/// Convert a pointer to `eqs_array_t` containing a `torch::Tensor` (inside an
/// equistore::TorchDataArray) into said `torch::Tensor`.
static torch::Tensor eqs_array_to_tensor(int64_t eqs_array_ptr) {
    // HERE BE DRAGONS: we hope that eqs_array_ptr is actually a pointer to
    // `eqs_array_t`, but we have no way to check. Get it right on the Python
    // side!
    eqs_array_t* array = reinterpret_cast<eqs_array_t*>(eqs_array_ptr);

    eqs_data_origin_t origin = 0;
    auto status = array->origin(array->ptr, &origin);
    if (status != EQS_SUCCESS) {
        throw equistore::Error("failed to get data origin for this array");
    }

    if (TORCH_DATA_ORIGIN != 0 && origin == TORCH_DATA_ORIGIN) {
        // we have registered torch tensors with equistore, and this eqs_array
        // contains a torch tensor. Let's extract it!
        auto* ptr = reinterpret_cast<TorchDataArray*>(array->ptr);
        return ptr->tensor();
    }

    throw equistore::Error("this array does not contain a C++ torch Tensor");
}

TORCH_LIBRARY(rascaline, m) {
    m.class_<TorchSystem>("System")
        .def(torch::init<torch::Tensor, torch::Tensor, torch::Tensor>(),
            "", /* TODO: docstrings */
            {torch::arg("species"), torch::arg("positions"), torch::arg("cell")}
        )
        .def_property("species", &TorchSystem::get_species)
        .def_property("positions", &TorchSystem::get_positions)
        .def_property("cell", &TorchSystem::get_cell)
        ;


    m.class_<TorchCalculator>("Calculator")
        .def(torch::init<std::string, std::string>())
        .def_pickle(
            // __getstate__
            [](const c10::intrusive_ptr<TorchCalculator>& self) -> std::vector<std::string> {
                return {self->name(), self->parameters()};
            },
            // __setstate__
            [](std::vector<std::string> state) -> c10::intrusive_ptr<TorchCalculator> {
                return c10::make_intrusive<TorchCalculator>(
                    state[0], state[1]
                );
            })
        ;

    m.def("rascaline_autograd", rascaline_autograd_py);
    m.def("eqs_array_to_tensor", eqs_array_to_tensor);
}
