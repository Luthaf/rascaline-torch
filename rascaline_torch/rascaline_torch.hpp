#ifndef RASCALINE_TORCH_HPP
#define RASCALINE_TORCH_HPP

#include <string>
#include <vector>
#include <mutex>

#include <torch/torch.h>
#include <rascaline.hpp>

namespace rascaline {

/// Basic string interner to pass strings from rascaline to the Python side of
/// rascaline_torch. Since PyTorch does not support string tensors, we pass
/// tensor of integers, where each integer correspond to a string stored by this
/// class.
class StringInterner {
public:
    /// Get the string corresponding to a given id
    static const std::string& get(size_t i);
    /// Add a new string in the global store, and get the corresponding integer
    /// id. If the string is already in the store, the corresponding id is
    /// returned and no new string is added.
    static size_t add(const std::string& value);

private:
    static std::mutex MUTEX_;
    static std::vector<std::string> STRINGS_;
};

/// Custom class holder used to store gradients data inside a
/// `rascaline::Descriptor` in a `torch::autograd::AutogradContext`.
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


/// Implementation of `rascaline::System` using torch tensors as backing memory
/// for all the data
class TensorSystem: public rascaline::System {
public:
    /// Try to construct a `TorchSystem` with the given tensors. This function
    /// will validate that the tensor have the right shape, dtype, and that they
    /// live on CPU and are contiguous.
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

/// Custom class holder to store, serialize and load rascaline calculators
/// inside Torch(Script) modules.
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


/// Custom torch::autograd::Function integrating rascaline with torch autorgrad.
class RascalineAutograd: public torch::autograd::Function<RascalineAutograd> {
public:
    /// Compute the representation of the system formed with `species`,
    /// `positions` and `cell` using the given `calculator` and corresponding
    /// `options`.
    ///
    /// The descriptor returned by rascaline will automatically be **densified**
    /// along `neighbor_species`. `options["densify_species"]` can contain a
    /// tensor that will be passed to `rascaline::Descriptor::densify` as the
    /// list of requested features. See the corresponding documentation for more
    /// information.
    ///
    /// @returns {values, samples, samples_names, features, features_names}
    /// where `values` is a tensor containing the **densified** representation
    /// returned by the `calculator`; `samples` contains the indexes used to
    /// describe the samples and `samples_names` contains interned strings id
    /// (to be used with `StringIndexer::get`) corresponding to the columns of
    /// `samples`; `features` contains the indexes used to describe the features
    /// and `features_names` contains interned strings id (to be used with
    /// `StringIndexer::get`) corresponding to the columns of `features`;
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
