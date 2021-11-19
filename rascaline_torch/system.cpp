#include "rascaline_torch.hpp"

using namespace rascaline;

TorchSystem::TorchSystem(torch::Tensor species, torch::Tensor positions, torch::Tensor cell) {
    auto species_sizes = species.sizes();
    if (species_sizes.size() != 1) {
        throw RascalError("atomic species tensor must be a 1D tensor");
    }
    auto n_atoms = species_sizes[0];

    this->species_ = species;
    if (this->species_.dtype() != torch::kInt) {
        throw RascalError("atomic species must be stored as 32-bit integers");
    }

    if (!this->species_.is_contiguous() || !this->species_.device().is_cpu()) {
        throw RascalError("atomic species must be stored as a contiguous tensor on CPU");
    }

    for (size_t i=0; i<n_atoms; i++) {
        if (this->species_[i].item<int32_t>() < 0) {
            throw RascalError("all atomic species must be positive integers");
        }
    }

    if (this->species_.requires_grad()) {
        throw RascalError("species can not have requires_grad=True");
    }

    /**************************************************************************/
    auto positions_sizes = positions.sizes();
    if (positions_sizes.size() != 2 || positions_sizes[0] != n_atoms || positions_sizes[1] != 3) {
        throw RascalError("the positions tensor must be a (n_atoms x 3) tensor");
    }

    this->positions_ = positions;
    if (this->positions_.dtype() != torch::kDouble) {
        throw RascalError("atomic positions must be stored as 64-bit floating point values");
    }

    if (!this->positions_.is_contiguous() || !this->species_.device().is_cpu()) {
        throw RascalError("atomic positions must be stored as a contiguous tensor on CPU");
    }

    /**************************************************************************/
    auto cell_sizes = cell.sizes();
    if (cell_sizes.size() != 2 || cell_sizes[0] != 3 || cell_sizes[1] != 3) {
        throw RascalError("the cell tensor must be a (3 x 3) tensor");
    }

    this->cell_ = cell;
    if (this->cell_.dtype() != torch::kDouble) {
        throw RascalError("unit cell must be stored as 64-bit floating point values");
    }

    if (!this->cell_.is_contiguous() || !this->species_.device().is_cpu()) {
        throw RascalError("unit cell must be stored as a contiguous tensor on CPU");
    }

    if (this->cell_.requires_grad()) {
        throw RascalError("we can not track the gradient with respect to the cell yet");
    }
}
