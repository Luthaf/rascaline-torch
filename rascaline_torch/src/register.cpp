#include <torch/torch.h>

#include "rascaline_torch.hpp"
using namespace rascaline;

/// Small wrapper around `RascalineAutograd` with a slightly more ergonomic API
/// (using named parameters inside of the tensors instead of relying on a
/// specific order for all the tensors input & outputs).
torch::Dict<std::string, torch::Tensor> compute(
    c10::intrusive_ptr<TorchCalculator> calculator,
    c10::intrusive_ptr<TorchSystem> system,
    torch::Dict<std::string, torch::Tensor> options
) {
    auto outputs = RascalineAutograd::apply(
        calculator,
        options,
        system,
        system->get_positions(),
        system->get_cell()
    );

    auto descriptor = torch::Dict<std::string, torch::Tensor>();
    descriptor.insert("values", std::move(outputs[0]));
    descriptor.insert("samples", std::move(outputs[1]));
    descriptor.insert("samples_names", std::move(outputs[2]));
    descriptor.insert("features", std::move(outputs[3]));
    descriptor.insert("features_names", std::move(outputs[4]));

    return descriptor;
}


std::string get_interned_string(int64_t id) {
    return StringInterner::get(id);
}

TORCH_LIBRARY(rascaline, m) {
    // only register this class to be able to create c10::intrusive_ptr with it
    // and store it in the custom autograd functions
    m.class_<DescriptorHolder>("__DescriptorHolder");

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

    m.def("compute", compute);
    m.def("get_interned_string", get_interned_string);
}
