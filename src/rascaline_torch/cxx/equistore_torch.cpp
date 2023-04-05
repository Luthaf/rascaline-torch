#include <fstream>

#include <equistore.hpp>
#include "equistore_torch.hpp"

using namespace equistore;

eqs_data_origin_t equistore::TORCH_DATA_ORIGIN = 0;

// eqs_data_origin registration in a thread-safe way through C++11 static
// initialization of a class with a constructor
struct EqsOriginRegistration {
    EqsOriginRegistration(const char* name) {
        auto status = eqs_register_data_origin(name, &TORCH_DATA_ORIGIN);
        if (status != EQS_SUCCESS) {
            throw equistore::Error("failed to register torch data origin");
        }
    }
};

eqs_data_origin_t TorchDataArray::origin() const {
    static EqsOriginRegistration REGISTRATION = EqsOriginRegistration("equistore_torch::TorchDataArray");
    return TORCH_DATA_ORIGIN;
}

std::unique_ptr<DataArrayBase> TorchDataArray::copy() const {
    return std::unique_ptr<DataArrayBase>(new TorchDataArray(this->tensor().clone()));
}

std::unique_ptr<DataArrayBase> TorchDataArray::create(std::vector<uintptr_t> shape) const {
    auto sizes = std::vector<int64_t>();
    for (auto size: shape) {
        sizes.push_back(static_cast<int64_t>(size));
    }

    return std::unique_ptr<DataArrayBase>(new TorchDataArray(
        torch::zeros(
            sizes,
            torch::TensorOptions()
                .dtype(this->tensor().dtype())
                .device(this->tensor().device())
        )
    ));
}

double* TorchDataArray::data() {
    if (!this->tensor_.device().is_cpu()) {
        throw equistore::Error("can not access the data of a tensor not on CPU");
    }

    if (this->tensor_.dtype() != torch::kF64) {
        throw equistore::Error("TODO");
    }

    if (!this->tensor_.is_contiguous()) {
        throw equistore::Error("TODO");
    }

    return static_cast<double*>(this->tensor_.data_ptr());
}

const std::vector<uintptr_t>& TorchDataArray::shape() const {
    return shape_;
}

void TorchDataArray::reshape(std::vector<uintptr_t> shape) {
    auto sizes = std::vector<int64_t>();
    for (auto size: shape) {
        sizes.push_back(static_cast<int64_t>(size));
    }

    this->tensor_ = this->tensor().reshape(sizes).contiguous();

    this->update_shape();
}

void TorchDataArray::swap_axes(uintptr_t axis_1, uintptr_t axis_2) {
    this->tensor_ = this->tensor().swapaxes(axis_1, axis_2).contiguous();

    this->update_shape();
}

void TorchDataArray::move_samples_from(
    const DataArrayBase& raw_input,
    std::vector<eqs_sample_mapping_t> samples,
    uintptr_t property_start,
    uintptr_t property_end
) {
    const auto& input = dynamic_cast<const TorchDataArray&>(raw_input);
    auto input_tensor = input.tensor();

    auto input_samples = std::vector<int64_t>();
    input_samples.reserve(samples.size());
    auto output_samples = std::vector<int64_t>();
    output_samples.reserve(samples.size());

    for (const auto& sample: samples) {
        input_samples.push_back(static_cast<int64_t>(sample.input));
        output_samples.push_back(static_cast<int64_t>(sample.output));
    }

    using torch::indexing::Slice;
    using torch::indexing::Ellipsis;
    auto output_tensor = this->tensor();

    // output[output_samples, ..., properties] = input[input_samples, ..., :]
    output_tensor.index_put_(
        {torch::tensor(std::move(output_samples)), Ellipsis, Slice(property_start, property_end)},
        input_tensor.index({torch::tensor(std::move(input_samples)), Ellipsis, Slice()})
    );
}
