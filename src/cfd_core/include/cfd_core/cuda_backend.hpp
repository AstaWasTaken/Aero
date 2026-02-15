#pragma once

#include "cfd_core/mesh.hpp"

#include <array>
#include <string>
#include <vector>

namespace cfd::cuda_backend {
bool compute_scalar_residual_cuda(const cfd::core::UnstructuredMesh& mesh,
                                  const std::vector<float>& phi,
                                  const std::array<float, 3>& u_inf, float inflow_phi,
                                  std::vector<float>* residual, std::string* error_message);
}  // namespace cfd::cuda_backend
