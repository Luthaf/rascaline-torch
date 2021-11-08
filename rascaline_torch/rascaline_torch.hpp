#ifndef RASCALINE_TORCH_HPP
#define RASCALINE_TORCH_HPP

#include <string>
#include <vector>

#include <torch/torch.h>
#include <rascaline.hpp>

namespace rascaline {

class DescriptorHolder: public torch::CustomClassHolder {
public:
    DescriptorHolder(rascaline::Descriptor descriptor): data(std::move(descriptor)) {}

    at::Tensor values_as_tensor();
    at::Tensor samples_as_tensor();
    at::Tensor features_as_tensor();

    at::Tensor gradients_as_tensor();

    // hold the descriptor memory around
    rascaline::Descriptor data;
};


class TensorSystem: public rascaline::System {
public:
    TensorSystem(torch::Tensor species, torch::Tensor positions, torch::Tensor cell);

    virtual ~TensorSystem() {}

    uintptr_t size() const override {
        return species_.sizes()[0];
    }

    const int32_t* species() const override {
        return species_.data_ptr<int32_t>();
    }

    const double* positions() const override {
        return positions_.data_ptr<double>();
    }

    CellMatrix cell() const override {
        auto data = cell_.data_ptr<double>();
        return CellMatrix{{
            {{data[0], data[1], data[2]}},
            {{data[3], data[4], data[5]}},
            {{data[6], data[7], data[8]}},
        }};
    }

    void compute_neighbors(double cutoff) override {
        throw RascalError("this system only support 'use_native_systems=true'");
    }

    const std::vector<rascal_pair_t>& pairs() const override {
        throw RascalError("this system only support 'use_native_systems=true'");
    }

    const std::vector<rascal_pair_t>& pairs_containing(uintptr_t center) const override {
        throw RascalError("this system only support 'use_native_systems=true'");
    }

private:
    at::Tensor species_;
    at::Tensor positions_;
    at::Tensor cell_;
};

class TorchCalculator: public torch::CustomClassHolder {
public:
    TorchCalculator(std::string name, std::string parameters):
        name_(std::move(name)),
        instance_(name_, std::move(parameters))
    {}

    rascal_calculator_t* as_rascal_calculator_t() {
        return instance_.as_rascal_calculator_t();
    }

    const std::string& name() const {
        return name_;
    }

    std::string parameters() const {
        return instance_.parameters();
    }

private:
    std::string name_;
    rascaline::Calculator instance_;
};


class RascalineAutograd: public torch::autograd::Function<RascalineAutograd> {
public:
    static torch::autograd::variable_list forward(
        torch::autograd::AutogradContext *ctx,
        c10::intrusive_ptr<TorchCalculator> calculator,
        torch::Dict<std::string, torch::Tensor> options,
        torch::Tensor species,
        torch::Tensor positions,
        torch::Tensor cell
    );

    static torch::autograd::variable_list backward(
        torch::autograd::AutogradContext *ctx,
        torch::autograd::variable_list grad_outputs
    );
};

} // namespace rascaline

#endif
