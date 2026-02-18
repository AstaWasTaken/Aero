#include "cfd_core/solvers/euler_solver.hpp"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

int main() {
  cfd::core::EulerAirfoilCaseConfig config;
  config.output_dir = std::filesystem::path("out/tests/euler_wall_sanity");
  config.mesh.naca_code = "0012";
  config.mesh.num_circumferential = 40;
  config.mesh.num_radial = 8;
  config.mesh.farfield_radius = 10.0f;
  config.mesh.radial_stretch = 1.25f;
  config.iterations = 1;
  config.min_iterations = 1;
  config.cfl_start = 0.0f;
  config.cfl_max = 0.0f;
  config.cfl_ramp_iters = 1;
  config.residual_reduction_target = 0.0f;
  config.force_stability_tol = 0.0f;
  config.flux_scheme = cfd::core::EulerFluxScheme::kHllc;
  config.limiter = cfd::core::LimiterType::kVenkat;
  config.precond_on = true;
  config.precond_mach_ref = 0.15f;
  config.precond_mach_min = 1.0e-3f;
  config.precond_beta_min = 0.2f;
  config.precond_beta_max = 1.0f;
  config.mach = 0.2f;
  config.aoa_deg = 0.0f;

  const cfd::core::EulerRunResult result = cfd::core::run_euler_airfoil_case_cpu(config);
  if (!std::isfinite(result.max_wall_mass_flux) || result.max_wall_mass_flux > 1.0e-4f) {
    std::cerr << "Wall flux diagnostic too large: " << result.max_wall_mass_flux << "\n";
    return 2;
  }

  std::ifstream wall_flux_csv(result.wall_flux_csv_path);
  if (!wall_flux_csv) {
    std::cerr << "Missing wall_flux.csv output.\n";
    return 3;
  }

  std::string line;
  std::getline(wall_flux_csv, line);
  int count = 0;
  float max_abs_flux = 0.0f;
  while (std::getline(wall_flux_csv, line)) {
    if (line.empty()) {
      continue;
    }
    std::stringstream row(line);
    std::string token;
    std::getline(row, token, ',');
    std::getline(row, token, ',');
    std::getline(row, token, ',');
    if (!std::getline(row, token, ',')) {
      continue;
    }
    const float mass_flux_n = std::stof(token);
    if (!std::isfinite(mass_flux_n)) {
      std::cerr << "Non-finite wall mass flux entry.\n";
      return 4;
    }
    max_abs_flux = std::max(max_abs_flux, std::abs(mass_flux_n));
    ++count;
  }

  if (count == 0) {
    std::cerr << "No wall faces found in wall_flux.csv.\n";
    return 5;
  }
  if (max_abs_flux > 1.0e-4f) {
    std::cerr << "wall_flux.csv max wall mass flux too large: " << max_abs_flux << "\n";
    return 6;
  }

  return 0;
}
