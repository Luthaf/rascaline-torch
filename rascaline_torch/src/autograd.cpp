#include <unordered_set>

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

static std::unordered_set<std::string> KNOWN_OPTIONS = std::unordered_set<std::string>{
    // densify options
    "densify_species",
    // calculator options
    "selected_centers",
};


/// Custom function allowing to take gradients of the gradients w.r.t. positions
/// computed with RascalineAutograd
class RascalinePositionsGrad: public torch::autograd::Function<RascalinePositionsGrad> {
public:
    static torch::autograd::variable_list forward(
        torch::autograd::AutogradContext *ctx,
        // metadata
        c10::intrusive_ptr<rascaline::DescriptorHolder> descriptor,
        torch::Tensor selected_centers,
        int n_feature_blocks,
        // actual function input
        torch::Tensor values_grad,
        // along for the ride so that torch is able to include them in the graph
        torch::Tensor positions,
        torch::Tensor cell
    );

    static torch::autograd::variable_list backward(
        torch::autograd::AutogradContext *ctx,
        torch::autograd::variable_list grad_outputs
    );
};

torch::autograd::variable_list RascalineAutograd::forward(
    torch::autograd::AutogradContext *ctx,
    c10::intrusive_ptr<TorchCalculator> calculator,
    torch::Dict<std::string, torch::Tensor> options_dict,
    c10::intrusive_ptr<TorchSystem> system,
    // we need to take positions & cell as parameters for the registration of
    // `backward_fn` to work
    torch::Tensor positions,
    torch::Tensor cell
) {
    for (const auto& entry: options_dict) {
        if (KNOWN_OPTIONS.find(entry.key()) == KNOWN_OPTIONS.end()) {
            throw RascalError("got unknown option in rascaline calculator: '" + entry.key() + "'");
        }
    }

    auto rascal_system = system->as_rascal_system_t();
    auto options = rascaline::CalculationOptions();
    options.use_native_system = system->use_native_system();
    // TODO: better handling of selected_samples and selected_features & their
    // interaction with densify
    if (options_dict.contains("selected_centers")) {
        auto selected_centers = options_dict.at("selected_centers");
        auto selected_centers_sizes = selected_centers.sizes();
        if (selected_centers.dtype() != torch::kInt64 || selected_centers_sizes.size() != 1) {
            throw RascalError("selected_centers must be a 1D tensor containing 64-bit integers");
        }

        if (!selected_centers.is_contiguous() || !selected_centers.device().is_cpu()) {
            throw RascalError("selected_centers must be stored as a contiguous tensor on CPU");
        }

        options.selected_samples = rascaline::SelectedIndexes({"center"});
        for (size_t i=0; i<selected_centers_sizes[0]; i++) {
            options.selected_samples.add({
                static_cast<int32_t>(selected_centers[i].item<int64_t>())
            });
        }

        ctx->saved_data["selected_centers"] = selected_centers;
    }

    auto raw_descriptor = rascaline::Descriptor();
    auto status = rascal_calculator_compute(
        calculator->as_rascal_calculator_t(),
        raw_descriptor.as_rascal_descriptor_t(),
        &rascal_system,
        1,
        options.as_rascal_calculation_options_t()
    );
    if (status != RASCAL_SUCCESS) {
        throw RascalError(rascal_last_error());
    }

    auto descriptor = c10::make_intrusive<DescriptorHolder>(std::move(raw_descriptor));

    /**************************************************************************/
    // densify the descriptor along the default variable for this calculator
    auto densify_variables = get_densify_for_calculator(
        calculator->name()
    );

    auto requested = rascaline::ArrayView<int32_t>(static_cast<const int32_t*>(nullptr), {0, 0});
    auto n_feature_blocks = 1;
    if (options_dict.contains("densify_species")) {
        auto densify_species = options_dict.at("densify_species");
        auto densify_species_sizes = densify_species.sizes();
        if (densify_species.dtype() != torch::kInt || densify_species_sizes.size() != 2) {
            throw RascalError("densify_species must be a 2D tensor containing 32-bit integers");
        }

        if (!densify_species.is_contiguous() || !densify_species.device().is_cpu()) {
            throw RascalError("densify_species must be stored as a contiguous tensor on CPU");
        }

        requested = rascaline::ArrayView<int32_t>(
            densify_species.data_ptr<int32_t>(),
            {static_cast<size_t>(densify_species_sizes[0]), static_cast<size_t>(densify_species_sizes[1])}
        );

        n_feature_blocks = densify_species_sizes[0];
    }

    descriptor->densify_values(densify_variables, requested);

    /**************************************************************************/

    auto values = descriptor->values_as_tensor();
    auto samples = descriptor->samples_as_tensor();
    auto features = descriptor->features_as_tensor();

    ctx->saved_data["descriptor"] = descriptor;
    ctx->saved_data["n_feature_blocks"] = n_feature_blocks;

    ctx->save_for_backward({positions, cell});

    if (!(positions.requires_grad() || cell.requires_grad())) {
        // the descriptor holder will only be kept around if one of the inputs
        // requires a gradient. If this is not the case, we need to copy the
        // values to prevent using freed memory from inside the descriptor
        values = values.clone();
        samples = samples.clone();
        features = features.clone();
    }

    // pass sample names & feature names as integer id. The global StringInterner
    // deals with converting strings to id and id back to strings
    auto raw_samples = descriptor->raw_samples();
    auto raw_samples_names = raw_samples.names();
    auto samples_names = torch::zeros({static_cast<int64_t>(raw_samples_names.size())}, torch::kInt64);
    for (size_t i=0; i<raw_samples_names.size(); i++) {
        samples_names[i] = static_cast<int64_t>(
            StringInterner::add(raw_samples_names[i])
        );
    }

    auto raw_features = descriptor->raw_features();
    auto raw_features_names = raw_features.names();
    auto features_names = torch::zeros({static_cast<int64_t>(raw_features_names.size())}, torch::kInt64);
    for (size_t i=0; i<raw_features_names.size(); i++) {
        features_names[i] = static_cast<int64_t>(
            StringInterner::add(raw_features_names[i])
        );
    }

    return {values, samples, samples_names, features, features_names};
}

torch::autograd::variable_list RascalineAutograd::backward(
    torch::autograd::AutogradContext *ctx,
    torch::autograd::variable_list grad_outputs
) {
    // only gradients related to the values of the representation make sense,
    // so we ignore the gradients w.r.t. samples/features/xxx_names
    auto values_grad = grad_outputs[0];

    // initialize output as empty tensors (corresponding to None in Python)
    auto grad_positions = torch::Tensor();
    auto grad_cell = torch::Tensor();

    auto saved_variables = ctx->get_saved_variables();
    auto positions = saved_variables[0];
    auto cell = saved_variables[1];

    if (positions.requires_grad()) {
        torch::Tensor selected_centers = torch::empty({});
        auto it = ctx->saved_data.find("selected_centers");
        if (it != ctx->saved_data.end()) {
            selected_centers = it->second.toTensor();
        }

        // implement backward for RascalineAutograd as forward of RascalineGradGrad
        // to allow taking gradients of the gradients w.r.t. positions (i.e.
        // forces) w.r.t. other parameters in the model.
        grad_positions = RascalinePositionsGrad::apply(
            ctx->saved_data["descriptor"].toCustomClass<DescriptorHolder>(),
            selected_centers,
            ctx->saved_data["n_feature_blocks"].toInt(),
            values_grad,
            positions,
            cell
        )[0];
    }

    if (cell.requires_grad()) {
        throw RascalError("gradient w.r.t. cell are not yet implemented");
    }

    return {torch::Tensor(), torch::Tensor(), torch::Tensor(), grad_positions, grad_cell};
}

torch::autograd::variable_list RascalinePositionsGrad::forward(
    torch::autograd::AutogradContext *ctx,
    c10::intrusive_ptr<rascaline::DescriptorHolder> descriptor,
    torch::Tensor selected_centers,
    int n_feature_blocks,
    torch::Tensor values_grad,
    torch::Tensor positions,
    torch::Tensor cell
) {
    const auto& densified_positions = descriptor->densified_positions();
    if (densified_positions.size() == 0) {
        throw RascalError("missing gradients in call to backward. Did you set the correct hyper-parameters?");
    }

    const auto& gradients_samples = descriptor->raw_gradients_samples();
    auto gradients = descriptor->gradients_as_tensor();

    auto n_atoms = positions.sizes()[0];
    auto n_features = values_grad.sizes()[1];
    auto n_samples = gradients_samples.shape()[0];
    auto grad_feature_size = gradients.sizes()[1];
    assert(gradients.sizes()[0] == n_samples);
    if (grad_feature_size != n_features / n_feature_blocks) {
        if (grad_feature_size == 0) {
            throw RascalError("missing gradients in call to backward. Did you set the correct hyper-parameters?");
        } else {
            throw RascalError("size mismatch between gradients and values, something is very wrong");
        }
    }

    const auto& gradients_samples_names = gradients_samples.names();
    assert(gradients_samples_names[0] == "sample");
    assert(gradients_samples_names[1] == "atom");
    assert(gradients_samples_names[2] == "spatial");

    auto grad_positions = torch::zeros_like(positions);
    auto grad_positions_accessor = grad_positions.accessor<double, 2>();

    // compute the Vector-Jacobian product, dealing with the sparse species storage
    for (int64_t grad_sample_i=0; grad_sample_i<gradients_samples.shape()[0]; grad_sample_i++) {
        auto sample_i = gradients_samples(grad_sample_i, 0);
        auto atom_i = gradients_samples(grad_sample_i, 1);
        auto spatial_i = gradients_samples(grad_sample_i, 2);

        auto& position = densified_positions[sample_i];
        if (!position.used) {
            continue;
        }

        auto start = grad_feature_size * position.feature_block;
        auto stop = grad_feature_size * (position.feature_block + 1);
        auto feature_row = values_grad.index({
            static_cast<int64_t>(position.new_sample), torch::indexing::Slice(start, stop)
        });
        auto gradient_row = gradients.index({
            grad_sample_i, torch::indexing::Slice()
        });

        auto dot = feature_row.dot(gradient_row);
        grad_positions_accessor[atom_i][spatial_i] += dot.item<double>();
    }

    ctx->save_for_backward({values_grad, positions, cell});
    ctx->saved_data["descriptor"] = descriptor;
    ctx->saved_data["selected_centers"] = selected_centers;

    return {grad_positions};
}

torch::autograd::variable_list RascalinePositionsGrad::backward(
    torch::autograd::AutogradContext *ctx,
    torch::autograd::variable_list grad_outputs
) {
    auto grad_output = grad_outputs[0];

    // initialize output as empty tensors (corresponding to None in Python)
    auto grad_grad_values = torch::Tensor();

    auto saved_variables = ctx->get_saved_variables();
    auto values_grad = saved_variables[0];
    auto positions = saved_variables[1];
    auto cell = saved_variables[2];

    auto selected_centers = ctx->saved_data["selected_centers"].toTensor();
    auto descriptor = ctx->saved_data["descriptor"].toCustomClass<DescriptorHolder>();
    auto& densified_positions = descriptor->densified_positions();

    const auto& gradients_samples = descriptor->raw_gradients_samples();
    auto gradients = descriptor->gradients_as_tensor();
    auto grad_feature_size = gradients.sizes()[1];

    const auto& gradients_samples_names = gradients_samples.names();
    assert(gradients_samples_names[0] == "sample");
    assert(gradients_samples_names[1] == "atom");
    assert(gradients_samples_names[2] == "spatial");

    if (values_grad.requires_grad()) {
        grad_grad_values = torch::zeros_like(values_grad);

        // compute the Vector-Jacobian product, dealing with the sparse species storage
        for (int64_t grad_sample_i=0; grad_sample_i<gradients_samples.shape()[0]; grad_sample_i++) {
            auto sample_i = gradients_samples(grad_sample_i, 0);
            auto atom_i = gradients_samples(grad_sample_i, 1);
            auto spatial_i = gradients_samples(grad_sample_i, 2);

            auto& position = densified_positions[sample_i];
            if (!position.used) {
                continue;
            }

            auto start = grad_feature_size * position.feature_block;
            auto stop = grad_feature_size * (position.feature_block + 1);

            auto gradient_row = gradients.index({grad_sample_i, torch::indexing::Slice()});
            auto slice = grad_grad_values.index({
                static_cast<int64_t>(position.new_sample), torch::indexing::Slice(start, stop)
            });
            slice += grad_output.index({atom_i, spatial_i}) * gradient_row;
        }
    }

    if (positions.requires_grad()) {
        // TODO: warn that second derivatives are not implemented?
        // or fill the corresponding gradient with NaN
    }

    if (cell.requires_grad()) {
        throw RascalError("gradient w.r.t. cell are not yet implemented");
    }

    return {
        torch::Tensor(),
        torch::Tensor(),
        torch::Tensor(),
        grad_grad_values,
        torch::Tensor(),
        torch::Tensor(),
    };
}
