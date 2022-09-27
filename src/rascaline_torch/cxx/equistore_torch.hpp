#ifndef EQUISTORE_TORCH_HPP
#define EQUISTORE_TORCH_HPP

#include <torch/torch.h>
#include <equistore.hpp>

namespace equistore {

/// Equistore data origin for torch arrays. This is either 0 if no torch::Tensor
/// has been registered with equistore yet, or the origin for torch::Tensor.
extern eqs_data_origin_t TORCH_DATA_ORIGIN;

/// An equistore::DataArray using torch::Tensor to store the data
class TorchDataArray final: public DataArrayBase {
public:
    TorchDataArray(torch::Tensor tensor): tensor_(std::move(tensor)) {
        this->update_shape();
    }

    torch::Tensor tensor() const {
        return tensor_;
    }

    /*========================================================================*/
    /*          Functions to implement equistore::DataArrayBase               */
    /*========================================================================*/

    eqs_data_origin_t origin() const override;

    std::unique_ptr<DataArrayBase> copy() const override;

    std::unique_ptr<DataArrayBase> create(std::vector<uintptr_t> shape) const override;

    const double* data() const override;

    const std::vector<uintptr_t>& shape() const override;

    void reshape(std::vector<uintptr_t> shape) override;

    void swap_axes(uintptr_t axis_1, uintptr_t axis_2) override;

    void move_samples_from(
        const DataArrayBase& input,
        std::vector<eqs_sample_mapping_t> samples,
        uintptr_t property_start,
        uintptr_t property_end
    ) override;

private:
    torch::Tensor tensor_;

    // cache the array shape as a vector of unsigned integers (as expected by
    // equistore) instead of signed integer (as stored in torch::Tensor::sizes)
    std::vector<uintptr_t> shape_;

    void update_shape() {
        shape_.clear();
        for (auto size: this->tensor_.sizes()) {
            shape_.push_back(static_cast<uintptr_t>(size));
        }
    }
};

}

#endif
