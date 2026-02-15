#pragma once

#include "cfd_core/mesh.hpp"

#include <filesystem>
#include <vector>

namespace cfd::core {
// Writes a minimal cell-based VTU for scalar advection diagnostics.
bool write_scalar_cell_vtu(const std::filesystem::path& output_path, const UnstructuredMesh& mesh,
                           const std::vector<float>& phi, const std::vector<float>& residual);
bool write_euler_cell_vtu(const std::filesystem::path& output_path, const UnstructuredMesh& mesh,
                          const std::vector<float>& rho, const std::vector<float>& u,
                          const std::vector<float>& v, const std::vector<float>& p,
                          const std::vector<float>& mach,
                          const std::vector<float>& residual_magnitude);
}  // namespace cfd::core
