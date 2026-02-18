#include "cfd_core/backend.hpp"
#include "cfd_core/mesh.hpp"
#include "cfd_core/solver.hpp"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <stdexcept>

int main() {
  using cfd::core::ScalarCaseConfig;

  const cfd::core::UnstructuredMesh mesh = cfd::core::make_demo_tri_mesh_2x2();

  ScalarCaseConfig config;
  config.u_inf = {1.0f, -0.35f, 0.0f};
  config.inflow_phi = 0.75f;
  config.phi.resize(static_cast<std::size_t>(mesh.num_cells), 0.0f);
  for (int c = 0; c < mesh.num_cells; ++c) {
    config.phi[c] = 0.5f + 0.2f * static_cast<float>(c);
  }

  config.output_dir = std::filesystem::path("out/tests/scalar_parity_cpu");
  const cfd::core::ScalarRunResult cpu_run = cfd::core::run_scalar_case(mesh, config, "cpu");

  if (!cfd::core::cuda_available()) {
    std::cout << "CUDA unavailable; skipping CPU/CUDA parity assertion.\n";
    return 0;
  }

  config.output_dir = std::filesystem::path("out/tests/scalar_parity_cuda");
  const cfd::core::ScalarRunResult cuda_run = cfd::core::run_scalar_case(mesh, config, "cuda");

  if (cpu_run.residual.size() != cuda_run.residual.size()) {
    std::cerr << "Residual vector size mismatch.\n";
    return 2;
  }

  float max_abs_diff = 0.0f;
  for (std::size_t i = 0; i < cpu_run.residual.size(); ++i) {
    max_abs_diff = std::max(max_abs_diff, std::abs(cpu_run.residual[i] - cuda_run.residual[i]));
  }

  if (max_abs_diff > 1.0e-6f) {
    std::cerr << "CPU/CUDA residual mismatch too large: " << max_abs_diff << "\n";
    return 3;
  }

  return 0;
}
