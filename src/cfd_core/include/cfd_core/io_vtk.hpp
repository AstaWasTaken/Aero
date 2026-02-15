#pragma once

#include <filesystem>

namespace cfd::core {
// Writes a minimal VTU file that can be loaded in ParaView.
bool write_placeholder_vtu(const std::filesystem::path& output_path);
}  // namespace cfd::core