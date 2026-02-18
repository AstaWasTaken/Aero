#include "cfd_core/solvers/euler_solver.hpp"
#include "cfd_core/post/forces.hpp"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <iostream>

int main() {
  cfd::core::EulerAirfoilCaseConfig config;
  config.output_dir = std::filesystem::path("out/tests/euler_aoa0_regression");
  config.mesh.naca_code = "0012";
  config.mesh.num_circumferential = 64;
  config.mesh.num_radial = 14;
  config.mesh.farfield_radius = 12.0f;
  config.mesh.radial_stretch = 1.35f;
  config.iterations = 12;
  config.min_iterations = 12;
  config.cfl_start = 0.15f;
  config.cfl_max = 0.7f;
  config.cfl_ramp_iters = 12;
  config.residual_reduction_target = 1.0e-2f;
  config.force_stability_tol = 5.0e-4f;
  config.force_stability_window = 6;
  config.flux_scheme = cfd::core::EulerFluxScheme::kRusanov;
  config.limiter = cfd::core::LimiterType::kMinmod;
  config.precond_on = false;
  config.startup_first_order_iters = 25;
  config.mach = 0.15f;
  config.aoa_deg = 0.0f;

  const cfd::core::EulerRunResult result = cfd::core::run_euler_airfoil_case_cpu(config);
  float rho_inf = config.rho_inf;
  if (rho_inf <= 0.0f) {
    rho_inf = config.p_inf / (config.gas_constant * config.t_inf);
  }
  const float a_inf = std::sqrt(config.gamma * config.p_inf / std::max(rho_inf, 1.0e-12f));
  const float speed_inf = std::max(config.mach, 0.0f) * a_inf;
  const cfd::core::FreestreamReference reference = {
    rho_inf,
    config.p_inf,
    config.aoa_deg,
    speed_inf,
    config.mesh.chord,
    config.x_ref,
    config.y_ref,
  };
  const cfd::core::PressureForceDiagnostics pressure_diag =
    cfd::core::compute_pressure_force_diagnostics(result.mesh, result.p, reference, "wall");
  const cfd::core::ForceCoefficients abs_forces =
    cfd::core::integrate_pressure_forces(result.mesh, result.p, reference, "wall", false);
  const cfd::core::ForceCoefficients gauge_forces =
    cfd::core::integrate_pressure_forces(result.mesh, result.p, reference, "wall", true);
  const auto print_diagnostics = [&]() {
    const float sum_nA_mag = std::sqrt(pressure_diag.sum_nA_x * pressure_diag.sum_nA_x +
                                       pressure_diag.sum_nA_y * pressure_diag.sum_nA_y);
    std::cerr << "wall_faces_integrated=" << pressure_diag.integrated_face_count << "\n";
    std::cerr << "sum_nA_x=" << pressure_diag.sum_nA_x << " sum_nA_y=" << pressure_diag.sum_nA_y
              << " sum_nA_mag=" << sum_nA_mag << "\n";
    std::cerr << "F_abs=(" << pressure_diag.fx_abs << "," << pressure_diag.fy_abs
              << ") F_gauge=(" << pressure_diag.fx_gauge << "," << pressure_diag.fy_gauge << ")\n";
    std::cerr << "Cl_abs=" << abs_forces.cl << " Cd_abs=" << abs_forces.cd
              << " Cl_gauge=" << gauge_forces.cl << " Cd_gauge=" << gauge_forces.cd << "\n";
  };

  if (result.history.empty()) {
    std::cerr << "Euler history is empty.\n";
    print_diagnostics();
    return 2;
  }
  if (!std::isfinite(result.forces.cl) || !std::isfinite(result.forces.cd) ||
      !std::isfinite(result.forces.cm)) {
    std::cerr << "Non-finite force coefficients at AoA=0.\n";
    print_diagnostics();
    return 3;
  }

  if (std::abs(result.forces.cl) >= 0.05f) {
    std::cerr << "AoA=0 lift is too large. Cl=" << result.forces.cl << "\n";
    print_diagnostics();
    return 4;
  }
  if (std::abs(result.forces.cd) >= 0.06f) {
    std::cerr << "AoA=0 drag is too large. Cd=" << result.forces.cd << "\n";
    print_diagnostics();
    return 5;
  }
  if (!std::isfinite(result.max_wall_mass_flux) || result.max_wall_mass_flux > 1.0e-8f) {
    std::cerr << "AoA=0 wall mass flux too large: " << result.max_wall_mass_flux << "\n";
    print_diagnostics();
    return 9;
  }

  return 0;
}
