#pragma once

#include <string>

namespace cfd::core {
bool cuda_available();

// Converts user input to a normalized backend string and validates availability.
std::string normalize_backend(std::string requested_backend);
}  // namespace cfd::core