#include <string>
#include <vector>
#include <stdexcept>

#include <torch/torch.h>
#include <rascaline.hpp>

using rascaline::RascalError;

static int64_t find_position(const std::vector<std::string>& names, const char* name);

// get the c10::ScalarType corresponding to the template parameter T
template<typename T>
c10::ScalarType torch_dtype();

template<> c10::ScalarType torch_dtype<double>() {
    return torch::kFloat64;
}

template<> c10::ScalarType torch_dtype<int32_t>() {
    return torch::kInt32;
}

template<typename T>
torch::Tensor array_to_tensor(const rascaline::ArrayView<T>& array) {
    int64_t shape[2] = {
        static_cast<int64_t>(array.shape()[0]),
        static_cast<int64_t>(array.shape()[1]),
    };

    auto tensor = torch::from_blob(
        // TODO: torch does not support read-only tensors, there is a tracking
        // issue at https://github.com/pytorch/pytorch/issues/44027. Until then,
        // we cast away the const and try not to write to the data.
        const_cast<T*>(array.data()),
        shape,
        torch::TensorOptions().dtype(torch_dtype<T>())
    );

    // TODO: it looks like the data in the underlying Descriptor gets garbage
    // collected too early, and the `from_blob` tensor then points to random
    // memory. For now, the easiest solution is to clone the tensor, but it
    // would be good to figure out a way to re-use the descriptor memory.
    return tensor.clone();
}


class TorchSystem: public rascaline::System {
public:
    TorchSystem(
        const torch::Tensor& positions,
        const torch::Tensor& species,
        const torch::Tensor& cell
    ) {
        auto cell_sizes = cell.sizes();
        if (cell_sizes.size() != 2 || cell_sizes[0] != 3 || cell_sizes[1] != 3) {
            throw RascalError("the cell tensor must be a (3 x 3) tensor");
        }

        auto species_sizes = species.sizes();
        if (species_sizes.size() != 1) {
            throw RascalError("the species tensor must be a 1D-tensor");
        }
        auto n_atoms = species_sizes[0];

        auto positions_sizes = positions.sizes();
        if (positions_sizes.size() != 2 || positions_sizes[0] != n_atoms || positions_sizes[1] != 3) {
            throw RascalError("the positions tensor must be a (n_atoms x 3) tensor");
        }

        this->positions_ = positions.cpu().contiguous().to(torch::kDouble);
        this->species_.reserve(n_atoms);
        for (size_t i=0; i<n_atoms; i++) {
            auto s = species[i].item<double>();
            if (s < 0.0 || std::fmod(s, 1.0) != 0.0) {
                throw RascalError("all atomic species must be positive integers");
            }
            this->species_.push_back(static_cast<uintptr_t>(s));
        }
        this->cell_ = cell.cpu().contiguous().to(torch::kDouble);
    }

    virtual ~TorchSystem() {}

    uintptr_t size() const override {
        return species_.size();
    }

    const uintptr_t* species() const override {
        return species_.data();
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
    torch::Tensor positions_;
    std::vector<uintptr_t> species_;
    torch::Tensor cell_;
};

class TorchDescriptor: public torch::CustomClassHolder {
public:
    TorchDescriptor(rascaline::Descriptor descriptor):
        descriptor_(std::move(descriptor)) {}

    rascaline::ArrayView<double> values() const {
        return this->descriptor_.values();
    }

    rascaline::ArrayView<double> gradients() const {
        return this->descriptor_.gradients();
    }

    rascaline::Indexes samples() const {
        return this->descriptor_.samples();
    }

    rascaline::Indexes features() const {
        return this->descriptor_.features();
    }

    rascaline::Indexes gradients_samples() const {
        return this->descriptor_.gradients_samples();
    }

private:
    rascaline::Descriptor descriptor_;
};

class TorchCalculator: public torch::CustomClassHolder {
public:
    /// Constructor taking the pointer to `rascal_calculator_t` as an integer,
    /// to be used in Python
    TorchCalculator(std::string name, int64_t calculator_ptr):
        TorchCalculator(name, reinterpret_cast<rascal_calculator_t*>(calculator_ptr)) {}

    /// Create a new TorchCalculator using the given calculator. This class does
    /// not take ownership of the calculator, which still needs to be freed when
    /// no longer useful.
    TorchCalculator(std::string name, rascal_calculator_t* calculator): calculator_(calculator) {
        if (name == "spherical_expansion") {
            densify_variables_ = {"species_neighbor"};
        } else if (name == "soap_power_spectrum") {
            densify_variables_ = {"species_neighbor_1", "species_neighbor_2"};
        } else {
            throw RascalError("unknown calculator, please edit this file");
        }
    }

    c10::intrusive_ptr<TorchDescriptor> compute(TorchSystem& system) {
        auto rascal_system = system.as_rascal_system_t();

        auto options = rascaline::CalculationOptions();
        options.use_native_system = true;

        auto descriptor = rascaline::Descriptor();
        auto status = rascal_calculator_compute(
            this->calculator_,
            descriptor.as_rascal_descriptor_t(),
            &rascal_system,
            1,
            options.as_rascal_calculation_options_t()
        );
        if (status != RASCAL_SUCCESS) {
            throw RascalError(rascal_last_error());
        }

        // TODO: there might be a more efficient way to do this, especially if
        // gradients are required.
        descriptor.densify(densify_variables_);

        return c10::make_intrusive<TorchDescriptor>(std::move(descriptor));
    }

private:
    rascal_calculator_t* calculator_;
    std::vector<std::string> densify_variables_;
};


class RascalineFunction: public torch::autograd::Function<RascalineFunction> {
public:
    static torch::autograd::variable_list forward(
        torch::autograd::AutogradContext *ctx,
        c10::intrusive_ptr<TorchCalculator> calculator,
        torch::Tensor positions,
        torch::Tensor species,
        torch::Tensor cell
    ) {
        auto system = TorchSystem(positions, species, cell);

        auto descriptor = calculator->compute(system);
        ctx->saved_data["descriptor"] = descriptor;
        ctx->saved_data["requires_grad"] = c10::List<bool>{
            positions.requires_grad(),
            species.requires_grad(),
            cell.requires_grad()
        };

        return {
            array_to_tensor(descriptor->values()),
            array_to_tensor(descriptor->samples()),
            array_to_tensor(descriptor->features()),
        };
    }

    static torch::autograd::tensor_list backward(
        torch::autograd::AutogradContext *ctx,
        torch::autograd::tensor_list grad_outputs
    ) {
        auto grad_calculator = torch::Tensor();
        auto grad_positions = torch::Tensor();
        auto grad_species = torch::Tensor();
        auto grad_cell = torch::Tensor();

        auto input_requires_grad = ctx->saved_data["requires_grad"].toBoolList();

        if (input_requires_grad[0]) {
            const auto& features_grad = grad_outputs[0];
            // we don't care about gradient w.r.t. samples/features, i.e.
            // whatever is in grad_outputs[1] and grad_outputs[2]

            auto descriptor = ctx->saved_data["descriptor"].toCustomClass<TorchDescriptor>();
            const auto& gradients_samples = descriptor->gradients_samples();
            auto gradients = array_to_tensor(descriptor->gradients());

            auto n_atoms = features_grad.sizes()[0];
            auto n_features = features_grad.sizes()[1];
            grad_positions = torch::zeros(
                {n_atoms, 3},
                torch::TensorOptions().dtype(torch::kFloat64)
            );

            auto grad_positions_accessor = grad_positions.accessor<double, 2>();

            auto n_samples = gradients_samples.shape()[0];
            assert(gradients.sizes()[0] == n_samples);
            if (gradients.sizes()[1] != n_features) {
                if (gradients.sizes()[1] == 0) {
                    throw RascalError("missing gradients in call to backward. Did you set the correct hyper-parameters?");
                } else {
                    throw RascalError("size mismatch between gradients and values, something is very wrong");
                }
            }

            const auto& names = gradients_samples.names();
            // TODO: this will only work for per-atom representation
            auto center_position = find_position(names, "center");
            auto neighbor_position = find_position(names, "neighbor");
            auto spatial_position = find_position(names, "spatial");

            // compute the Vector-Jacobian product
            for (int64_t sample_i=0; sample_i<n_samples; sample_i++) {
                auto center_i = gradients_samples(sample_i, center_position);
                auto neighbor_i = gradients_samples(sample_i, neighbor_position);
                auto spatial_i = gradients_samples(sample_i, spatial_position);

                auto feature_row = features_grad.index({center_i, torch::indexing::Slice()});
                auto gradient_row = gradients.index({sample_i, torch::indexing::Slice()});

                auto dot = feature_row.dot(gradient_row);
                grad_positions_accessor[neighbor_i][spatial_i] += dot.item<double>();
            }
        }

        if (input_requires_grad[1]) {
            throw RascalError("can not get gradient w.r.t. species");
        }

        if (input_requires_grad[2]) {
            throw RascalError("gradient w.r.t. cell are not yet implemented");
        }

        return {
            grad_calculator,
            grad_positions,
            grad_species,
            grad_cell,
        };
    }
};

torch::autograd::variable_list compute(
    c10::intrusive_ptr<TorchCalculator> calculator,
    const torch::Tensor& positions,
    const torch::Tensor& species,
    const torch::Tensor& cell
) {
    return RascalineFunction::apply(calculator, positions, species, cell);
}


TORCH_LIBRARY(rascaline, m) {
    m.class_<TorchCalculator>("TorchCalculator")
        .def(torch::init<std::string, int64_t>());

    // only register the class to be able to store it in RascalineFunction
    // but do not expose any method
    m.class_<TorchDescriptor>("TorchDescriptor");

    m.def("compute(__torch__.torch.classes.rascaline.TorchCalculator calculator, Tensor positions, Tensor species, Tensor cell) -> Tensor[]", compute);
}

/* helper functions */

int64_t find_position(const std::vector<std::string>& names, const char* name) {
    auto it = std::find(names.begin(), names.end(), name);
    if (it == names.end()) {
        throw RascalError("can not find " + std::string(name) + " in the samples");
    }
    return it - names.begin();
}
