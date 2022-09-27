#ifndef RASCALINE_TORCH_HPP
#define RASCALINE_TORCH_HPP

#include <string>
#include <vector>
#include <mutex>

#include <torch/torch.h>
#include <rascaline.hpp>

namespace rascaline {

/// Implementation of `rascaline::System` using torch tensors as backing memory
/// for all the data.
class TorchSystem final: public rascaline::System, public torch::CustomClassHolder {
public:
    /// Try to construct a `TorchSystem` with the given tensors. This function
    /// will validate that the tensor have the right shape, dtype, that they
    /// live on CPU, and are contiguous.
    TorchSystem(torch::Tensor species, torch::Tensor positions, torch::Tensor cell);

    TorchSystem(const TorchSystem&) = delete;
    TorchSystem& operator=(const TorchSystem&) = delete;

    TorchSystem(TorchSystem&&) = default;
    TorchSystem& operator=(TorchSystem&&) = default;

    virtual ~TorchSystem() {}

    /*========================================================================*/
    /*            Functions to implement rascaline::System                    */
    /*========================================================================*/

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

    void compute_neighbors(double cutoff) override;

    const std::vector<rascal_pair_t>& pairs() const override;

    const std::vector<rascal_pair_t>& pairs_containing(uintptr_t center) const override;

    /*========================================================================*/
    /*                 Functions to re-use pre-computed pairs                 */
    /*========================================================================*/

    /// Should we copy data to rascaline internal data structure and compute the
    /// neighbor list there?
    bool use_native_system() const {
        return !has_precomputed_pairs_;
    }

    /// set the list of pre-computed pairs to `pairs` (following the convention
    /// required by `rascaline::System::pairs`), and store the `cutoff` used to
    /// compute the pairs.
    void set_precomputed_pairs(double cutoff, std::vector<rascal_pair_t> pairs);

    /*========================================================================*/
    /*                 Functions for the Python interface                     */
    /*========================================================================*/

    torch::Tensor get_species() {
        return species_;
    }

    torch::Tensor get_positions() {
        return positions_;
    }

    torch::Tensor get_cell() {
        return cell_;
    }


private:
    torch::Tensor species_;
    torch::Tensor positions_;
    torch::Tensor cell_;

    double cutoff_ = 0.0;
    bool has_precomputed_pairs_ = false;
    std::vector<rascal_pair_t> pairs_;
    std::vector<std::vector<rascal_pair_t>> pairs_containing_;
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


/// Custom torch::autograd::Function integrating rascaline with torch autograd.
///
/// This is a bit more complex than your typical autograd because there is some
/// impedance mismatch between rascaline and torch. Most of it should be taken
/// care of by the `compute` function below.
class RascalineAutograd: public torch::autograd::Function<RascalineAutograd> {
public:
    /// Compute the representation of the `systems` using the `calculator` and
    /// corresponding `options`.
    ///
    /// `all_positions` and `all_cell` are only used to make sure torch
    /// registers nodes in the calculation graph. They must be the same as
    /// `torch::vstack([s->get_positions() for s in systems])` and
    /// `torch::vstack([s->get_cell() for s in systems])` respectively.
    ///
    /// This function "returns" an equistore TensorMap in it's last parameter,
    /// which should then be passed on to C++/Python code.
    ///
    /// This function also actually returns a list of torch::Tensor containing
    /// the values for each block in the TensorMap. This should be left unused,
    /// and is only there to make sure torch registers a `grad_fn` for the
    /// tensors stored inside the TensorMap.
    static torch::autograd::variable_list forward(
        torch::autograd::AutogradContext *ctx,
        c10::intrusive_ptr<TorchCalculator> calculator,
        torch::Dict<std::string, torch::Tensor> options,
        std::vector<c10::intrusive_ptr<TorchSystem>> systems,
        torch::Tensor all_positions,
        torch::Tensor all_cells,
        eqs_tensormap_t** tensor_map
    );

    static torch::autograd::variable_list backward(
        torch::autograd::AutogradContext *ctx,
        torch::autograd::variable_list grad_outputs
    );
};

/// User-facing version of `RascalineAutograd`, taking a single `calculator`,
/// and running it on a bunch of `systems` with the given `options`.
///
/// This function returns a pointer to an equistore TensorMap, which should be
/// freed by the caller when done with it.
eqs_tensormap_t* rascaline_autograd(
    torch::intrusive_ptr<TorchCalculator> calculator,
    std::vector<torch::intrusive_ptr<TorchSystem>> systems,
    torch::Dict<std::string, torch::Tensor> options
);

} // namespace rascaline

#endif
