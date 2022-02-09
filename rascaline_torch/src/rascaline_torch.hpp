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
    DescriptorHolder(rascaline::Descriptor descriptor):
        data_(std::move(descriptor)), densified_positions_(nullptr, 0) {}

    torch::Tensor values_as_tensor();
    torch::Tensor samples_as_tensor();
    torch::Tensor features_as_tensor();
    torch::Tensor gradients_as_tensor();

    rascaline::Indexes raw_samples() const {
        return data_.samples();
    }

    rascaline::Indexes raw_features() const {
        return data_.features();
    }

    rascaline::Indexes raw_gradients_samples() const {
        return data_.gradients_samples();
    }

    void densify_values(std::vector<std::string> variables, const rascaline::ArrayView<int32_t>& requested) {
        densified_positions_ = data_.densify_values(std::move(variables), requested);
    }

    const rascaline::MallocArray<rascal_densified_position_t>& densified_positions() const {
        return densified_positions_;
    }

private:
    // keep the descriptor around
    rascaline::Descriptor data_;

    rascaline::MallocArray<rascal_densified_position_t> densified_positions_;
};

/// Implementation of `rascaline::System` using torch tensors as backing memory
/// for all the data.
class TorchSystem: public rascaline::System, public torch::CustomClassHolder {
public:
    /// Try to construct a `TorchSystem` with the given tensors. This function
    /// will validate that the tensor have the right shape, dtype, and that they
    /// live on CPU and are contiguous.
    TorchSystem(
        torch::Tensor species,
        torch::Tensor positions,
        torch::Tensor cell
    );

    TorchSystem(const TorchSystem&) = delete;
    TorchSystem& operator=(const TorchSystem&) = delete;

    TorchSystem(TorchSystem&&) = default;
    TorchSystem& operator=(TorchSystem&&) = default;

    virtual ~TorchSystem() {}

    /*========================================================================*/
    /*            Functions to implement rascaline::System                    */
    /*========================================================================*/

    uintptr_t size() const override final {
        return species_.sizes()[0];
    }

    const int32_t* species() const override final {
        return species_.data_ptr<int32_t>();
    }

    const double* positions() const override final {
        return positions_.data_ptr<double>();
    }

    CellMatrix cell() const override final {
        auto data = cell_.data_ptr<double>();
        return CellMatrix{{
            {{data[0], data[1], data[2]}},
            {{data[3], data[4], data[5]}},
            {{data[6], data[7], data[8]}},
        }};
    }

    void compute_neighbors(double cutoff) override final;

    const std::vector<rascal_pair_t>& pairs() const override final;

    const std::vector<rascal_pair_t>& pairs_containing(uintptr_t center) const override final;

    /*========================================================================*/
    /*                 Functions to re-use pre-computed pairs                 */
    /*========================================================================*/

    /// Should we copy data to rascaline internal data structure and compute the
    /// neighbor list there?
    bool use_native_system() const {
        return !this->has_precomputed_pairs_;
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
    std::vector<rascal_pair_t> pairs_;
    std::vector<std::vector<rascal_pair_t>> pairs_containing_;
    bool has_precomputed_pairs_ = false;
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
    /// Compute the representation of the `system` using the `calculator` and
    /// corresponding `options`. `positions` and `cell` are dummy parameters
    /// that must be the same as `system->positions()` and `system->cell()`.
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
        c10::intrusive_ptr<TorchSystem> system,
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
