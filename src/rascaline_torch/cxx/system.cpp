#include "rascaline_torch.hpp"

using namespace rascaline;

TorchSystem::TorchSystem(torch::Tensor species, torch::Tensor positions, torch::Tensor cell) {
    auto species_sizes = species.sizes();
    if (species_sizes.size() != 1) {
        throw RascalineError("atomic species tensor must be a 1D tensor");
    }
    auto n_atoms = species_sizes[0];

    this->species_ = species;
    if (this->species_.dtype() != torch::kInt) {
        throw RascalineError("atomic species must be stored as 32-bit integers");
    }

    if (!this->species_.is_contiguous() || !this->species_.device().is_cpu()) {
        throw RascalineError("atomic species must be stored as a contiguous tensor on CPU");
    }

    for (size_t i=0; i<n_atoms; i++) {
        if (this->species_[i].item<int32_t>() < 0) {
            throw RascalineError("all atomic species must be positive integers");
        }
    }

    if (this->species_.requires_grad()) {
        throw RascalineError("species can not have requires_grad=True");
    }

    /**************************************************************************/
    auto positions_sizes = positions.sizes();
    if (positions_sizes.size() != 2 || positions_sizes[0] != n_atoms || positions_sizes[1] != 3) {
        throw RascalineError("the positions tensor must be a (n_atoms x 3) tensor");
    }

    this->positions_ = positions;
    if (this->positions_.dtype() != torch::kDouble) {
        throw RascalineError("atomic positions must be stored as 64-bit floating point values");
    }

    if (!this->positions_.is_contiguous() || !this->species_.device().is_cpu()) {
        throw RascalineError("atomic positions must be stored as a contiguous tensor on CPU");
    }

    /**************************************************************************/
    auto cell_sizes = cell.sizes();
    if (cell_sizes.size() != 2 || cell_sizes[0] != 3 || cell_sizes[1] != 3) {
        throw RascalineError("the cell tensor must be a (3 x 3) tensor");
    }

    this->cell_ = cell;
    if (this->cell_.dtype() != torch::kDouble) {
        throw RascalineError("unit cell must be stored as 64-bit floating point values");
    }

    if (!this->cell_.is_contiguous() || !this->species_.device().is_cpu()) {
        throw RascalineError("unit cell must be stored as a contiguous tensor on CPU");
    }
}

void TorchSystem::compute_neighbors(double cutoff) {
    if (!this->has_precomputed_pairs_) {
        throw RascalineError("this system only support 'use_native_systems=true'");
    }

    if (cutoff != this->cutoff_) {
        throw RascalineError("trying to get neighbor list with a different cutoff than the pre-computed neighbor list");
    }
}

const std::vector<rascal_pair_t>& TorchSystem::pairs() const {
    if (!this->has_precomputed_pairs_) {
        throw RascalineError("this system only support 'use_native_systems=true'");
    }
    return pairs_;
}

const std::vector<rascal_pair_t>& TorchSystem::pairs_containing(uintptr_t center) const {
    if (!this->has_precomputed_pairs_) {
        throw RascalineError("this system only support 'use_native_systems=true'");
    }
    return pairs_containing_[center];
}

void TorchSystem::set_precomputed_pairs(double cutoff, std::vector<rascal_pair_t> pairs) {
    has_precomputed_pairs_ = true;

    this->cutoff_ = cutoff;
    this->pairs_ = pairs;

    pairs_containing_.clear();
    pairs_containing_.resize(this->size());
    for (const auto& pair: this->pairs_) {
        pairs_containing_[pair.first].push_back(pair);
        pairs_containing_[pair.second].push_back(pair);
    }
}
