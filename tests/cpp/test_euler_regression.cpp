#include "cfd_core/solvers/euler_solver.hpp"

#include <cmath>
#include <filesystem>
#include <iostream>

int main() {
  cfd::core::EulerAirfoilCaseConfig config;
  config.output_dir = std::filesystem::path("euler_regression");
  config.mesh.naca_code = "0012";
  config.mesh.num_circumferential = 72;
  config.mesh.num_radial = 18;
  config.mesh.farfield_radius = 12.0f;
  config.mesh.radial_stretch = 1.35f;
  config.iterations = 80;
  config.min_iterations = 20;
  config.cfl_start = 0.15f;
  config.cfl_max = 0.7f;
  config.cfl_ramp_iters = 80;
  config.residual_reduction_target = 1.0e-2f;
  config.force_stability_tol = 5.0e-4f;
  config.mach = 0.15f;
  config.aoa_deg = 2.0f;

  const cfd::core::EulerRunResult result = cfd::core::run_euler_airfoil_case_cpu(config);
  if (result.history.size() < 2) {
    std::cerr << "Euler history too short.\n";
    return 2;
  }

  const float initial_l2 = result.history.front().residual_l2;
  const float final_l2 = result.history.back().residual_l2;
  if (!std::isfinite(final_l2) || final_l2 <= 0.0f || final_l2 > initial_l2 * 1.2f) {
    std::cerr << "Euler residual did not remain bounded. initial=" << initial_l2
              << " final=" << final_l2 << "\n";
    return 3;
  }

  if (!(result.forces.cl > 0.0f)) {
    std::cerr << "Expected positive lift at AoA=2deg. Cl=" << result.forces.cl << "\n";
    return 4;
  }

  if (!std::filesystem::exists(result.residuals_csv_path) ||
      !std::filesystem::exists(result.forces_csv_path) || !std::filesystem::exists(result.cp_csv_path) ||
      !std::filesystem::exists(result.vtu_path)) {
    std::cerr << "Missing expected Euler output files.\n";
    return 5;
  }

  return 0;
}
