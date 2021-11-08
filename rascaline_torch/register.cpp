#include <torch/torch.h>

#include "rascaline_torch.hpp"
using namespace rascaline;

torch::Dict<std::string, torch::Tensor> compute(
    c10::intrusive_ptr<TorchCalculator> calculator,
    torch::Dict<std::string, torch::Tensor> system,
    torch::Dict<std::string, torch::Tensor> options
) {
    auto species = system.at("species");
    auto positions = system.at("positions");
    auto cell = system.at("cell");

    auto result = RascalineAutograd::apply(
        calculator,
        options,
        species,
        positions,
        cell
    );

    auto descriptor = torch::Dict<std::string, torch::Tensor>();
    descriptor.insert("values", std::move(result[0]));
    descriptor.insert("samples", std::move(result[1]));
    descriptor.insert("features", std::move(result[2]));

    return descriptor;
}

TORCH_LIBRARY(rascaline, m) {
    // only register this class to be able to create c10::intrusive_ptr with it
    // and store it in the custom autograd functions
    m.class_<DescriptorHolder>("__DescriptorHolder");


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
}
