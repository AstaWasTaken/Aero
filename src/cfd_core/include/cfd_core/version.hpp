#pragma once

#include <string>
#include <string_view>

#ifndef CFD_CORE_VERSION_STR
#define CFD_CORE_VERSION_STR "0.0.0-dev"
#endif

namespace cfd::core {
inline constexpr std::string_view kVersion = CFD_CORE_VERSION_STR;

inline std::string version() {
  return std::string(kVersion);
}
}  // namespace cfd::core