#include "rascaline_torch.hpp"

using namespace rascaline;


// get the c10::ScalarType corresponding to the template parameter T
template<typename T>
c10::ScalarType torch_dtype();

template<> inline c10::ScalarType torch_dtype<double>() {
    return c10::kDouble;
}

template<> inline c10::ScalarType torch_dtype<int32_t>() {
    return c10::kInt;
}

template<typename T>
torch::Tensor array_to_tensor(const rascaline::ArrayView<T>& array) {
    int64_t shape[2] = {
        static_cast<int64_t>(array.shape()[0]),
        static_cast<int64_t>(array.shape()[1]),
    };

    auto options = torch::TensorOptions().dtype(torch_dtype<T>());

    if (shape[0] * shape[1] == 0) {
        return torch::empty(shape, options);
    } else {
        // cast away the const in Indexes array since pytorch can not deal with
        // const tensor.
        // TODO: remove the const at rascaline level
        return torch::from_blob(const_cast<T*>(array.data()), shape, options);
    }
}


at::Tensor DescriptorHolder::values_as_tensor() {
    return array_to_tensor(data_.values());
}

at::Tensor DescriptorHolder::samples_as_tensor() {
    return array_to_tensor(data_.samples());
}

at::Tensor DescriptorHolder::features_as_tensor() {
    return array_to_tensor(data_.features());
}

at::Tensor DescriptorHolder::gradients_as_tensor() {
    return array_to_tensor(data_.gradients());
}
