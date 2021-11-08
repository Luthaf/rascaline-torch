#include "rascaline_torch.hpp"

using namespace rascaline;

static size_t find_position(const std::vector<std::string>& names, const char* name) {
    auto it = std::find(names.begin(), names.end(), name);
    if (it == names.end()) {
        throw RascalError("can not find " + std::string(name) + " in the samples");
    }
    return it - names.begin();
}

static std::vector<std::string> get_densify_for_calculator(const std::string& name) {
    if (name == "spherical_expansion") {
        return {"species_neighbor"};
    } else if (name == "soap_power_spectrum") {
        return {"species_neighbor_1", "species_neighbor_2"};
    } else {
        throw RascalError("unknown calculator, please edit this file");
    }
}

torch::autograd::variable_list RascalineAutograd::forward(
    torch::autograd::AutogradContext *ctx,
    c10::intrusive_ptr<TorchCalculator> calculator,
    torch::Dict<std::string, torch::Tensor> options_dict,
    torch::Tensor species,
    torch::Tensor positions,
    torch::Tensor cell
) {
    auto system = TensorSystem(species, positions, cell);
    auto rascal_system = system.as_rascal_system_t();

    auto options = rascaline::CalculationOptions();
    options.use_native_system = true;

    auto descriptor = rascaline::Descriptor();
    auto status = rascal_calculator_compute(
        calculator->as_rascal_calculator_t(),
        descriptor.as_rascal_descriptor_t(),
        &rascal_system,
        1,
        options.as_rascal_calculation_options_t()
    );
    if (status != RASCAL_SUCCESS) {
        throw RascalError(rascal_last_error());
    }

    auto densify_variables = get_densify_for_calculator(
        calculator->name()
    );
    auto densified_gradients_indexes = descriptor.densify_values(densify_variables);

    auto descriptor_holder = c10::make_intrusive<DescriptorHolder>(std::move(descriptor));
    auto values = descriptor_holder->values_as_tensor();
    auto samples = descriptor_holder->samples_as_tensor();
    auto features = descriptor_holder->features_as_tensor();

    ctx->saved_data["descriptor"] = descriptor_holder;

    // TODO: use fixed size integers in rascaline API instead of uintptr_t
    static_assert(sizeof(int64_t) == sizeof(uintptr_t), "this code only works on 32-bit platform");
    ctx->saved_data["densified_gradients_indexes"] = torch::from_blob(
        densified_gradients_indexes.data(),
        {static_cast<int64_t>(densified_gradients_indexes.size()), 3},
        torch::TensorOptions().dtype(torch::kInt64)
    ).clone(); // we need to copy since densified_gradients_indexes will be freed

    ctx->saved_data["positions_requires_grad"] = positions.requires_grad();
    ctx->saved_data["cell_requires_grad"] = cell.requires_grad();

    if (!positions.requires_grad()) {
        // the descriptor holder will only be kept around if positions requires
        // a gradient. If this is not the case, we need to copy the values to
        // prevent using freed memory
        values = values.clone();
        samples = samples.clone();
        features = features.clone();
    }

    // TODO: sample names & features names

    return {values, samples, features};
}

torch::autograd::variable_list RascalineAutograd::backward(
    torch::autograd::AutogradContext *ctx,
    torch::autograd::variable_list grad_outputs
) {
    auto grad_calculator = torch::Tensor();
    auto grad_options = torch::Tensor();
    auto grad_species = torch::Tensor();
    auto grad_positions = torch::Tensor();
    auto grad_cell = torch::Tensor();

    if (ctx->saved_data["positions_requires_grad"].toBool()) {
        // we don't care about gradient w.r.t. samples/features, i.e.
        // whatever is in grad_outputs[1] and grad_outputs[2]
        const auto& values_grad = grad_outputs[0];

        auto densified_gradients_indexes = ctx->saved_data["densified_gradients_indexes"].toTensor();
        auto n_features_blocks = torch::max(
            densified_gradients_indexes.index({torch::indexing::Slice(), 2})
        ).item<int64_t>() + 1;

        auto descriptor = ctx->saved_data["descriptor"].toCustomClass<DescriptorHolder>();
        const auto& gradients_samples = descriptor->data.gradients_samples();
        auto gradients = descriptor->gradients_as_tensor();

        auto n_atoms = values_grad.sizes()[0];
        auto n_features = values_grad.sizes()[1];
        grad_positions = torch::zeros(
            {n_atoms, 3},
            torch::TensorOptions().dtype(torch::kFloat64)
        );
        auto grad_positions_accessor = grad_positions.accessor<double, 2>();

        auto n_samples = gradients_samples.shape()[0];
        auto grad_feature_size = gradients.sizes()[1];
        assert(gradients.sizes()[0] == n_samples);
        if (grad_feature_size != n_features / n_features_blocks) {
            if (grad_feature_size == 0) {
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

        // compute the Vector-Jacobian product, dealing with densified species
        for (int64_t i=0; i<densified_gradients_indexes.sizes()[0]; i++) {
            auto old_sample = densified_gradients_indexes.index({i, 0}).item<int64_t>();
            auto feature_block = densified_gradients_indexes.index({i, 2}).item<int64_t>();

            auto start = grad_feature_size * feature_block;
            auto stop = grad_feature_size * (feature_block + 1);

            auto center_i = gradients_samples(old_sample, center_position);
            auto neighbor_i = gradients_samples(old_sample, neighbor_position);
            auto spatial_i = gradients_samples(old_sample, spatial_position);

            auto feature_row = values_grad.index({center_i, torch::indexing::Slice(start, stop)});
            auto gradient_row = gradients.index({old_sample, torch::indexing::Slice()});

            auto dot = feature_row.dot(gradient_row);
            grad_positions_accessor[neighbor_i][spatial_i] += dot.item<double>();
        }
    }

    if (ctx->saved_data["cell_requires_grad"].toBool()) {
        throw RascalError("gradient w.r.t. cell are not yet implemented");
    }

    return {grad_calculator, grad_options, grad_species, grad_positions, grad_cell};
}
