#include "rascaline_torch.hpp"

using namespace rascaline;

const std::string& StringInterner::get(size_t i) {
    std::lock_guard<std::mutex> lock(MUTEX_);
    assert(i < STRINGS_.size());
    return STRINGS_[i];
}

size_t StringInterner::add(const std::string& value) {
    std::lock_guard<std::mutex> lock(MUTEX_);

    for (size_t i=0; i<STRINGS_.size(); i++) {
        if (STRINGS_[i] == value) {
            return i;
        }
    }

    STRINGS_.push_back(value);
    return STRINGS_.size() - 1;
}

std::mutex StringInterner::MUTEX_;
std::vector<std::string> StringInterner::STRINGS_;
