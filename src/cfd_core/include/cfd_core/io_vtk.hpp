#pragma once

#include "cfd_core/mesh.hpp"

#include <filesystem>
#include <vector>

namespace cfd::core {
// Writes a minimal cell-based VTU for scalar advection diagnostics.
bool write_scalar_cell_vtu(const std::filesystem::path& output_path, const UnstructuredMesh& mesh,
                           const std::vector<float>& phi, const std::vector<float>& residual);
}  // namespace cfd::core
