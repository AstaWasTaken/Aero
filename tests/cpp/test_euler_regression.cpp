#include "cfd_core/solvers/euler_solver.hpp"

#include <cmath>
#include <filesystem>
#include <iostream>

int main() {
  cfd::core::EulerAirfoilCaseConfig config;
  config.output_dir = std::filesystem::path("out/tests/euler_regression");
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
  config.force_stability_window = 6;
  config.flux_scheme = cfd::core::EulerFluxScheme::kRusanov;
  config.limiter = cfd::core::LimiterType::kMinmod;
  config.precond_on = false;
  config.precond_mach_ref = 0.15f;
  config.precond_mach_min = 1.0e-3f;
  config.startup_first_order_iters = 25;
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

  if (result.forces.cl < 0.16f || result.forces.cl > 0.30f) {
    std::cerr << "AoA=2 lift is out of expected range. Cl=" << result.forces.cl << "\n";
    return 4;
  }
  if (std::abs(result.forces.cd) > 0.25f) {
    std::cerr << "AoA=2 drag is too large. Cd=" << result.forces.cd << "\n";
    return 8;
  }

  if (!std::isfinite(result.forces.cl) || !std::isfinite(result.forces.cd) ||
      !std::isfinite(result.forces.cm)) {
    std::cerr << "Force coefficients contain non-finite values.\n";
    return 5;
  }

  if (!std::isfinite(result.max_wall_mass_flux) || result.max_wall_mass_flux > 1.0e-4f) {
    std::cerr << "Wall mass flux diagnostic too large: " << result.max_wall_mass_flux << "\n";
    return 6;
  }

  if (!std::filesystem::exists(result.residuals_csv_path) ||
      !std::filesystem::exists(result.forces_csv_path) ||
      !std::filesystem::exists(result.cp_csv_path) ||
      !std::filesystem::exists(result.wall_flux_csv_path) ||
      !std::filesystem::exists(result.vtu_path)) {
    std::cerr << "Missing expected Euler output files.\n";
    return 7;
  }

  return 0;
}
