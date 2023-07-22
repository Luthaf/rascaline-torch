#include <unordered_set>

#include <equistore.hpp>
#include <rascaline.hpp>

#include "equistore_torch.hpp"
#include "rascaline_torch.hpp"

#define stringify(str) #str
#define always_assert(condition)                                               \
    do {                                                                       \
        if (!(condition)) {                                                    \
            throw rascaline::RascalineError(                                   \
                std::string("assert failed: ") + stringify(condition)          \
            );                                                                 \
        }                                                                      \
    } while (false)

using namespace rascaline;

/// move a descriptor from Rust-owned arrays to torch-owned arrays on CPU
static equistore::TensorMap descriptor_to_torch(
    equistore::TensorMap descriptor,
    // should we keep the gradients in the descriptor
    bool keep_forward_grad,
    // all values in the descriptor (one per block)
    std::vector<torch::Tensor>& values,
    // all values samples in the descriptor (one per block)
    std::vector<torch::Tensor>& torch_samples,
    // all gradients w.r.t. positions in the descriptor (one per block)
    std::vector<torch::Tensor>& positions_grad,
    // samples for all gradients w.r.t. positions in the descriptor (one per block)
    std::vector<torch::Tensor>& positions_grad_samples,
    // all gradients w.r.t. cell in the descriptor (one per block)
    std::vector<torch::Tensor>& cell_grad,
    // samples for all gradients w.r.t. cell in the descriptor (one per block)
    std::vector<torch::Tensor>& cell_grad_samples
);


static bool all_systems_use_native(const std::vector<torch::intrusive_ptr<TorchSystem>>& systems) {
    auto result = systems[0]->use_native_system();
    for (const auto& system: systems) {
        if (system->use_native_system() != result) {
            throw rascaline::RascalineError("TODO");
        }
    }
    return result;
}


struct AutogradOptions {
    bool keep_forward_grad = false;
};


static AutogradOptions extract_options(const torch::Dict<std::string, torch::Tensor>& options_dict) {
    auto options = AutogradOptions();
    for (const auto& entry: options_dict) {
        if (entry.key() == "keep_forward_grad") {
            options.keep_forward_grad = entry.value().item<bool>();
        } else {
            throw rascaline::RascalineError("unknown option '" + entry.key() + "' given to RascalineAutograd");
        }
    }
    return options;
}


torch::autograd::variable_list RascalineAutograd::forward(
    torch::autograd::AutogradContext *ctx,
    c10::intrusive_ptr<TorchCalculator> calculator,
    torch::Dict<std::string, torch::Tensor> options_dict,
    std::vector<torch::intrusive_ptr<TorchSystem>> systems,
    torch::Tensor all_positions,
    torch::Tensor all_cells,
    eqs_tensormap_t** tensor_map
) {
    /* extract metadata and prepare the calculation */
    auto options = extract_options(options_dict);
    if (systems.empty()) {
        throw RascalineError("can not run a calculation on an empty list of systems");
    }

    auto calculation_options = CalculationOptions();
    if (all_positions.requires_grad()) {
        calculation_options.gradients.push_back("positions");
    }

    if (all_cells.requires_grad()) {
        calculation_options.gradients.push_back("cell");
    }

    calculation_options.use_native_system = all_systems_use_native(systems);
    // calculation_options.selected_properties = "TODO";
    // calculation_options.selected_samples = "TODO";

    auto rascal_system = std::vector<rascal_system_t>();
    auto structures_start = std::vector<int64_t>();
    int64_t current_start = 0;
    for (auto& system: systems) {
        rascal_system.push_back(system->as_rascal_system_t());
        structures_start.push_back(current_start);
        current_start += system->size();
    }

    /* run the actual calculation */
    eqs_tensormap_t* c_descriptor = nullptr;
    auto status = rascal_calculator_compute(
        calculator->as_rascal_calculator_t(),
        &c_descriptor,
        rascal_system.data(),
        rascal_system.size(),
        calculation_options.as_rascal_calculation_options_t()
    );

    if (status != RASCAL_SUCCESS) {
        throw RascalineError(rascal_last_error());
    }

    /* convert the resulting data to torch & extract the data for autograd */
    auto values = std::vector<torch::Tensor>();
    auto samples = std::vector<torch::Tensor>();
    auto positions_grad = std::vector<torch::Tensor>();
    auto positions_grad_samples = std::vector<torch::Tensor>();
    auto cell_grad = std::vector<torch::Tensor>();
    auto cell_grad_samples = std::vector<torch::Tensor>();

    auto torch_descriptor = descriptor_to_torch(
        equistore::TensorMap(c_descriptor),
        options.keep_forward_grad,
        values,
        samples,
        positions_grad,
        positions_grad_samples,
        cell_grad,
        cell_grad_samples
    );

    // we explicitly leak the descriptor, ownership of it is passed to the caller
    *tensor_map = torch_descriptor.release();

    // save the required data for backward pass
    ctx->save_for_backward({all_positions, all_cells});
    ctx->saved_data["samples"] = samples;
    ctx->saved_data["positions_grad"] = positions_grad;
    ctx->saved_data["positions_grad_samples"] = positions_grad_samples;
    ctx->saved_data["cell_grad"] = cell_grad;
    ctx->saved_data["cell_grad_samples"] = cell_grad_samples;
    ctx->saved_data["structures_start"] = structures_start;

    return values;
}

torch::autograd::variable_list RascalineAutograd::backward(
    torch::autograd::AutogradContext *ctx,
    torch::autograd::variable_list grad_outputs
) {
    /* get the saved data from the forward pass */
    auto saved_variables = ctx->get_saved_variables();
    auto all_positions = saved_variables[0];
    auto all_cells = saved_variables[1];

    auto samples_per_block = ctx->saved_data["samples"].toTensorVector();
    auto positions_grad_per_block = ctx->saved_data["positions_grad"].toTensorVector();
    auto positions_grad_samples = ctx->saved_data["positions_grad_samples"].toTensorVector();
    auto cell_grad_per_block = ctx->saved_data["cell_grad"].toTensorVector();
    auto cell_grad_samples = ctx->saved_data["cell_grad_samples"].toTensorVector();
    auto structures_start = ctx->saved_data["structures_start"].toIntVector();

    auto n_blocks = grad_outputs.size();

    /* gradient w.r.t. positions */
    auto positions_grad = torch::Tensor();
    if (all_positions.requires_grad()) {
        always_assert(positions_grad_per_block.size() == n_blocks);
        always_assert(positions_grad_samples.size() == n_blocks);

        positions_grad = torch::zeros_like(all_positions);
        auto positions_grad_accessor = positions_grad.accessor<double, 2>();
        for (size_t block_i=0; block_i<n_blocks; block_i++) {
            
            auto tmp_grad_values = grad_outputs[block_i];
            auto grad_values = tmp_grad_values.contiguous();
            always_assert(grad_values.is_contiguous() && grad_values.is_cpu());

            auto grad_values_ptr = grad_values.data_ptr<double>();
            // total size of component + property dimension
            size_t dot_dimensions = 1;
            for (int i=1; i<grad_values.sizes().size(); i++) {
                dot_dimensions *= grad_values.size(i);
            }

            auto tmp_forward_grad = positions_grad_per_block[block_i];
            auto forward_grad = tmp_forward_grad.contiguous();

            always_assert(forward_grad.is_contiguous() && forward_grad.is_cpu());
            auto forward_grad_ptr = forward_grad.data_ptr<double>();

            auto grad_samples = positions_grad_samples[block_i].accessor<int32_t, 2>();

            for (int64_t grad_sample_i=0; grad_sample_i<grad_samples.sizes()[0]; grad_sample_i++) {
                auto sample = grad_samples[grad_sample_i][0];
                auto structure = grad_samples[grad_sample_i][1];
                auto atom = grad_samples[grad_sample_i][2];

                auto global_atom = structures_start[structure] + atom;

                for (int64_t direction=0; direction<3; direction++) {
                    auto dot = 0.0;
                    for (int64_t i=0; i<dot_dimensions; i++) {
                        auto dX_dr = forward_grad_ptr[(grad_sample_i * 3 + direction) * dot_dimensions + i];
                        auto dl_dX = grad_values_ptr[sample * dot_dimensions + i];
                        dot += dX_dr * dl_dX;
                    }
                    positions_grad_accessor[global_atom][direction] += dot;
                }
            }
        }
    }

    /* gradient w.r.t. cell */
    auto cell_grad = torch::Tensor();
    if (all_cells.requires_grad()) {
        always_assert(cell_grad_per_block.size() == n_blocks);
        always_assert(cell_grad_samples.size() == n_blocks);

        cell_grad = torch::zeros_like(all_cells);
        auto cell_grad_accessor = cell_grad.accessor<double, 2>();
        for (size_t block_i=0; block_i<n_blocks; block_i++) {
            
            auto tmp_grad_values = grad_outputs[block_i];
            auto grad_values = tmp_grad_values.contiguous();
            
            always_assert(grad_values.is_contiguous() && grad_values.is_cpu());

            auto grad_values_ptr = grad_values.data_ptr<double>();
            // total size of component + property dimension
            size_t dot_dimensions = 1;
            for (int i=1; i<grad_values.sizes().size(); i++) {
                dot_dimensions *= grad_values.size(i);
            }

            auto tmp_forward_grad = cell_grad_per_block[block_i];
            auto forward_grad = tmp_forward_grad.contiguous();

            always_assert(forward_grad.is_contiguous() && forward_grad.is_cpu());
            auto forward_grad_ptr = forward_grad.data_ptr<double>();

            auto values_samples = samples_per_block[block_i].accessor<int32_t, 2>();
            auto grad_samples = cell_grad_samples[block_i].accessor<int32_t, 2>();

            for (int64_t grad_sample_i=0; grad_sample_i<grad_samples.sizes()[0]; grad_sample_i++) {
                auto sample = grad_samples[grad_sample_i][0];
                // FIXME: this assumes single center representation
                auto structure = values_samples[sample][0];

                for (int64_t direction_1=0; direction_1<3; direction_1++) {
                    for (int64_t direction_2=0; direction_2<3; direction_2++) {
                        // auto grad_value = grad_values.index({sample, "..."}).reshape({-1});
                        // // TODO: figure out why we need a transpose here?
                        // auto forward_grad = forward_grads.index({grad_sample_i, direction_2, direction_1, "..."}).reshape({-1});

                        // cell_grad_accessor[3 * structure + direction_1][direction_2] += grad_value.dot(forward_grad).item<double>();


                        auto dot = 0.0;
                        for (int64_t i=0; i<dot_dimensions; i++) {
                            // TODO: figure out why we need a transpose here?
                            auto id = (grad_sample_i * 3 + direction_2) * 3 + direction_1;
                            auto dX_dr = forward_grad_ptr[id * dot_dimensions + i];
                            auto dl_dX = grad_values_ptr[sample * dot_dimensions + i];
                            dot += dX_dr * dl_dX;
                        }
                        cell_grad_accessor[3 * structure + direction_1][direction_2] += dot;
                    }
                }
            }
        }
    }

    return {
        torch::Tensor(),
        torch::Tensor(),
        torch::Tensor(),
        positions_grad,
        cell_grad,
        torch::Tensor(),
    };
}

eqs_tensormap_t* rascaline::rascaline_autograd(
    torch::intrusive_ptr<TorchCalculator> calculator,
    std::vector<torch::intrusive_ptr<TorchSystem>> systems,
    torch::Dict<std::string, torch::Tensor> options
) {

    auto all_positions = std::vector<torch::Tensor>();
    auto all_cells = std::vector<torch::Tensor>();

    for (const auto& system: systems) {
        all_positions.push_back(system->get_positions());
        all_cells.push_back(system->get_cell());
    }

    eqs_tensormap_t* tensor_map = nullptr;
    auto outputs = RascalineAutograd::apply(
        calculator,
        options,
        std::move(systems),
        torch::vstack(all_positions),
        torch::vstack(all_cells),
        &tensor_map
    );

    return tensor_map;
}

/******************************************************************************/

static torch::Tensor ndarray_to_tensor(const equistore::NDArray<double>& array) {
    auto sizes = std::vector<int64_t>();
    for (auto size: array.shape()) {
        sizes.push_back(static_cast<int64_t>(size));
    }

    return torch::from_blob(
        // TODO: explain why is const_cast fine here
        const_cast<double*>(array.data()),
        sizes,
        torch::TensorOptions().dtype(torch::kF64)
    ).clone();
}

static torch::Tensor labels_to_tensor(const equistore::Labels& array) {
    return torch::from_blob(
        // TODO: explain why is const_cast fine here
        const_cast<int32_t*>(array.data()),
        {static_cast<int64_t>(array.count()), static_cast<int64_t>(array.size())},
        torch::TensorOptions().dtype(torch::kI32)
    ).clone();
}

equistore::TensorMap descriptor_to_torch(
    equistore::TensorMap descriptor,
    bool keep_forward_grad,
    std::vector<torch::Tensor>& torch_values,
    std::vector<torch::Tensor>& torch_samples,
    std::vector<torch::Tensor>& positions_grad,
    std::vector<torch::Tensor>& positions_grad_samples,
    std::vector<torch::Tensor>& cell_grad,
    std::vector<torch::Tensor>& cell_grad_samples
) {
    auto n_blocks = descriptor.keys().count();

    auto new_blocks = std::vector<equistore::TensorBlock>();
    for (size_t i=0; i<n_blocks; i++) {
        auto block = descriptor.block_by_id(i);

        // FIXME: this is only valid for per-center representation
        always_assert(block.samples().names().size() == 2);
        always_assert(block.samples().names()[0] == std::string("structure"));
        always_assert(block.samples().names()[1] == std::string("center"));
        torch_samples.push_back(labels_to_tensor(block.samples()));

        auto values_tensor = ndarray_to_tensor(block.values());
        // Since torch::Tensor is reference-counted internally, this makes it so
        // the same tensor is placed inside the block below & returned from
        // RascalineAutograd::forward. Returning the tensor from forward
        // registers it with the autograd framework, while the tensor is also
        // inside a TensorMap for convenient manipulations.
        torch_values.push_back(values_tensor);

        auto new_block = equistore::TensorBlock(
            std::unique_ptr<equistore::DataArrayBase>(new equistore::TorchDataArray(values_tensor)),
            block.samples(),
            block.components(),
            block.properties()
        );

        for (const auto& parameter: block.gradients_list()) {
            if (parameter == "positions") {
                auto gradient = block.gradient("positions");
                auto gradient_tensor = ndarray_to_tensor(gradient.values());

                always_assert(gradient.components()[0].names().size() == 1);
                always_assert(gradient.components()[0].names()[0] == std::string("direction"));

                always_assert(gradient.samples().names().size() == 3);
                always_assert(gradient.samples().names()[0] == std::string("sample"));
                always_assert(gradient.samples().names()[1] == std::string("structure"));
                always_assert(gradient.samples().names()[2] == std::string("atom"));

                positions_grad.push_back(gradient_tensor);
                positions_grad_samples.push_back(labels_to_tensor(gradient.samples()));

                if (keep_forward_grad) {
                    auto gradient_block =  equistore::TensorBlock(
                        std::unique_ptr<equistore::DataArrayBase>(new equistore::TorchDataArray(gradient_tensor)),
                        gradient.samples(),
                        gradient.components(),
                        gradient.properties());

                    new_block.add_gradient(
                        "positions",
                        std::move(gradient_block));
                }
            } else if (parameter == "cell") {
                auto gradient = block.gradient("cell");
                auto gradient_tensor = ndarray_to_tensor(gradient.values());

                always_assert(gradient.components().at(0).names().size() == 1);
                always_assert(gradient.components().at(0).names()[0] == std::string("direction_1"));
                always_assert(gradient.components().at(1).names().size() == 1);
                always_assert(gradient.components().at(1).names()[0] == std::string("direction_2"));

                always_assert(gradient.samples().names().size() == 1);
                always_assert(gradient.samples().names()[0] == std::string("sample"));

                cell_grad.push_back(gradient_tensor);
                cell_grad_samples.push_back(labels_to_tensor(gradient.samples()));

                if (keep_forward_grad) {
                    
                    auto gradient_block = equistore::TensorBlock(
                        std::unique_ptr<equistore::DataArrayBase>(new equistore::TorchDataArray(gradient_tensor)),
                        gradient.samples(),
                        gradient.components(),
                        gradient.properties());

                    new_block.add_gradient(
                        "cell",
                        std::move(gradient_block)
                    );
                }
            } else {
                TORCH_WARN("got an unexpected gradient parameter in TensorMap: '" + parameter + "'");
            }
        }

        new_blocks.push_back(std::move(new_block));
    }

    return equistore::TensorMap(descriptor.keys(), std::move(new_blocks));
}
