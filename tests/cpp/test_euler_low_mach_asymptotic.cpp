#include "cfd_core/solvers/euler_solver.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace {
struct SweepPoint {
  float mach = 0.0f;
  float cl = 0.0f;
  float cd = 0.0f;
  float cm = 0.0f;
  float drag = 0.0f;
  float cd_times_mach = 0.0f;
  float p_fluct_nd = 0.0f;
  bool force_window_stable = false;
  bool force_physical_converged = false;
};

struct MechanismThresholds {
  float mach = 0.01f;
  int iterations = 1200;
  int min_iterations = 300;
  float all_speed_f_min = 0.05f;
  int all_speed_ramp_start_iter = 100;
  int all_speed_ramp_iters = 200;
  float cd_times_mach_improvement_factor = 0.95f;
};

std::string trim_copy(const std::string& value) {
  std::size_t first = 0;
  while (first < value.size() && std::isspace(static_cast<unsigned char>(value[first])) != 0) {
    ++first;
  }
  std::size_t last = value.size();
  while (last > first && std::isspace(static_cast<unsigned char>(value[last - 1])) != 0) {
    --last;
  }
  return value.substr(first, last - first);
}

std::filesystem::path baseline_dir() {
  const std::filesystem::path test_src(__FILE__);
  return test_src.parent_path().parent_path() / "data/baselines/low_mach_euler";
}

MechanismThresholds load_mechanism_thresholds() {
  MechanismThresholds thresholds;
  const std::filesystem::path expected_path = baseline_dir() / "expected.yaml";
  std::ifstream stream(expected_path);
  if (!stream) {
    std::cerr << "Missing baseline file: " << expected_path.string() << "\n";
    return thresholds;
  }

  std::string line;
  while (std::getline(stream, line)) {
    const std::string trimmed = trim_copy(line);
    if (trimmed.empty() || trimmed.front() == '#') {
      continue;
    }
    const std::size_t colon = trimmed.find(':');
    if (colon == std::string::npos) {
      continue;
    }
    const std::string key = trim_copy(trimmed.substr(0, colon));
    const std::string value = trim_copy(trimmed.substr(colon + 1));
    if (value.empty()) {
      continue;
    }

    try {
      if (key == "mach") {
        thresholds.mach = std::stof(value);
      } else if (key == "iterations") {
        thresholds.iterations = std::stoi(value);
      } else if (key == "min_iterations") {
        thresholds.min_iterations = std::stoi(value);
      } else if (key == "all_speed_f_min") {
        thresholds.all_speed_f_min = std::stof(value);
      } else if (key == "all_speed_ramp_start_iter") {
        thresholds.all_speed_ramp_start_iter = std::stoi(value);
      } else if (key == "all_speed_ramp_iters") {
        thresholds.all_speed_ramp_iters = std::stoi(value);
      } else if (key == "cd_times_mach_improvement_factor") {
        thresholds.cd_times_mach_improvement_factor = std::stof(value);
      }
    } catch (...) {
      std::cerr << "Ignoring malformed threshold entry: " << trimmed << "\n";
    }
  }

  return thresholds;
}

bool env_flag_on(const char* name) {
  const char* value = std::getenv(name);
  if (value == nullptr) {
    return false;
  }
  const std::string token(value);
  return token == "1" || token == "true" || token == "TRUE" || token == "on" || token == "ON";
}

int env_int(const char* name, const int fallback) {
  const char* value = std::getenv(name);
  if (value == nullptr) {
    return fallback;
  }
  try {
    return std::stoi(value);
  } catch (...) {
    return fallback;
  }
}

float env_float(const char* name, const float fallback) {
  const char* value = std::getenv(name);
  if (value == nullptr) {
    return fallback;
  }
  try {
    return std::stof(value);
  } catch (...) {
    return fallback;
  }
}

cfd::core::EulerAirfoilCaseConfig make_config(const float mach, const bool all_speed_flux_fix,
                                              const int iterations, const int min_iterations,
                                              const float all_speed_f_min,
                                              const int all_speed_ramp_start_iter,
                                              const int all_speed_ramp_iters,
                                              const bool all_speed_staged_controller,
                                              const int all_speed_stage_a_min_iters,
                                              const bool precond_on,
                                              const std::filesystem::path& output_dir) {
  cfd::core::EulerAirfoilCaseConfig config;
  config.output_dir = output_dir;
  config.mesh.naca_code = "0012";
  config.mesh.num_circumferential = 72;
  config.mesh.num_radial = 18;
  config.mesh.farfield_radius = 12.0f;
  config.mesh.radial_stretch = 1.35f;
  config.iterations = iterations;
  config.min_iterations = min_iterations;
  config.cfl_start = 0.10f;
  config.cfl_max = 0.6f;
  config.cfl_ramp_iters = 180;
  config.residual_reduction_target = 1.0e-3f;
  config.force_stability_tol = 3.0e-3f;
  config.force_stability_window = 6;
  config.force_mean_drift_tol = 3.0e-3f;
  config.startup_first_order_iters = 30;
  config.rk_stages = 3;
  config.local_time_stepping = true;
  config.precond_on = precond_on;
  config.precond_mach_ref = mach;
  config.precond_mach_min = 1.0e-3f;
  config.precond_beta_min = 1.0e-4f;
  config.precond_beta_max = 1.0f;
  config.precond_farfield_bc = false;
  config.stabilization_mach_floor_k_start = 5.0f;
  config.stabilization_mach_floor_k_target = 1.0f;
  config.stabilization_ramp_iters = 700;
  config.flux_scheme = cfd::core::EulerFluxScheme::kHllc;
  config.limiter = cfd::core::LimiterType::kMinmod;
  config.low_mach_fix = false;
  config.mach_cutoff = 0.3f;
  config.all_speed_flux_fix = all_speed_flux_fix;
  config.all_speed_mach_cutoff = 0.25f;
  config.all_speed_f_min = all_speed_f_min;
  config.all_speed_ramp_start_iter = all_speed_ramp_start_iter;
  config.all_speed_ramp_iters = all_speed_ramp_iters;
  config.all_speed_staged_controller = all_speed_staged_controller;
  config.all_speed_stage_a_min_iters = all_speed_stage_a_min_iters;
  config.all_speed_stage_a_max_iters = 0;
  config.all_speed_stage_f_min_high = 0.2f;
  config.all_speed_stage_f_min_mid = 0.1f;
  config.all_speed_stage_f_min_low = 0.05f;
  config.all_speed_pjump_wall_target = 0.0f;
  config.all_speed_pjump_spike_factor = 1.5f;
  config.all_speed_pjump_hold_iters = 160;
  config.all_speed_freeze_iters = 160;
  config.all_speed_cfl_drop_factor = 0.7f;
  config.all_speed_cfl_restore_factor = 1.02f;
  config.all_speed_cfl_min_scale = 0.05f;
  config.mach = mach;
  config.aoa_deg = 0.0f;
  return config;
}

SweepPoint run_point(const float mach, const bool all_speed_flux_fix, const int iterations,
                    const int min_iterations, const float all_speed_f_min,
                    const int all_speed_ramp_start_iter, const int all_speed_ramp_iters,
                    const bool all_speed_staged_controller,
                    const int all_speed_stage_a_min_iters,
                    const bool precond_on,
                    const std::string& tag) {
  const cfd::core::EulerAirfoilCaseConfig config =
    make_config(mach, all_speed_flux_fix, iterations, min_iterations, all_speed_f_min,
                all_speed_ramp_start_iter, all_speed_ramp_iters, all_speed_staged_controller,
                all_speed_stage_a_min_iters, precond_on,
                std::filesystem::path("out/tests/euler_low_mach_asymptotic") / tag);
  const cfd::core::EulerRunResult result = cfd::core::run_euler_airfoil_case_cpu(config);

  SweepPoint point;
  point.mach = mach;
  point.cl = result.forces.cl;
  point.cd = result.forces.cd;
  point.cm = result.forces.cm;
  point.drag = result.forces.drag;
  point.cd_times_mach = result.cd_times_mach_final;
  point.p_fluct_nd = result.pressure_fluctuation_nd_max_final;
  point.force_window_stable = result.force_window_stable_final;
  point.force_physical_converged = result.force_physical_converged_final;
  return point;
}

bool finite_point(const SweepPoint& point) {
  return std::isfinite(point.cl) && std::isfinite(point.cd) && std::isfinite(point.cm) &&
         std::isfinite(point.drag) &&
         std::isfinite(point.cd_times_mach) && std::isfinite(point.p_fluct_nd);
}

int run_long_sweep_mode() {
  const int iterations = std::max(env_int("AERO_LM_ITERATIONS", 5000), 1);
  const int min_iterations = std::max(env_int("AERO_LM_MIN_ITERATIONS", 1200), 1);
  const float all_speed_f_min = std::clamp(env_float("AERO_LM_FMIN", 0.05f), 1.0e-6f, 1.0f);
  const int all_speed_ramp_start_iter = std::max(env_int("AERO_LM_RAMP_START", 0), 0);
  const int all_speed_ramp_iters = std::max(env_int("AERO_LM_RAMP_ITERS", 2500), 0);
  const bool all_speed_on = !env_flag_on("AERO_LM_ALL_SPEED_OFF");
  const bool all_speed_staged_controller = !env_flag_on("AERO_LM_STAGED_CONTROLLER_OFF");
  const int all_speed_stage_a_min_iters = std::max(env_int("AERO_LM_STAGE_A_MIN_ITERS", 1200), 0);
  const bool precond_requested = env_flag_on("AERO_LM_PRECOND_ON") && !env_flag_on("AERO_LM_PRECOND_OFF");
  const bool precond_on = precond_requested && env_flag_on("AERO_ENABLE_EXPERIMENTAL_PRECOND");
  if (precond_requested && !precond_on) {
    std::cout << "preconditioning forced off (experimental)\n";
  }

  std::cout << "long_sweep_settings "
            << "iterations=" << iterations << " min_iterations=" << min_iterations
            << " all_speed_flux_fix=" << (all_speed_on ? 1 : 0)
            << " all_speed_staged_controller=" << (all_speed_staged_controller ? 1 : 0)
            << " all_speed_stage_a_min_iters=" << all_speed_stage_a_min_iters
            << " precond_on=" << (precond_on ? 1 : 0)
            << " all_speed_f_min=" << all_speed_f_min
            << " all_speed_ramp_start_iter=" << all_speed_ramp_start_iter
            << " all_speed_ramp_iters=" << all_speed_ramp_iters << "\n";

  const std::vector<float> mach_values = {0.01f, 0.05f, 0.15f};
  for (const float mach : mach_values) {
    std::string tag = std::string(all_speed_on ? "long_on_m" : "long_off_m") +
                      std::to_string(static_cast<int>(std::round(1000.0f * mach)));
    const SweepPoint point =
      run_point(mach, all_speed_on, iterations, min_iterations, all_speed_f_min,
                all_speed_ramp_start_iter, all_speed_ramp_iters, all_speed_staged_controller,
                all_speed_stage_a_min_iters, precond_on, tag);
    if (!finite_point(point)) {
      std::cerr << "non_finite_point mach=" << mach << "\n";
      return 2;
    }
    if (!point.force_physical_converged) {
      std::cout << "mach=" << mach << " status=NOT_CONVERGED coefficients_unreliable=1\n";
    } else {
      std::cout << "mach=" << mach << " status=CONVERGED cd=" << point.cd << " cl=" << point.cl
                << " cm=" << point.cm << "\n";
    }
  }
  return 0;
}
}  // namespace

int main() {
  if (env_flag_on("AERO_LM_LONG_SWEEP")) {
    return run_long_sweep_mode();
  }

  if (!std::filesystem::exists(baseline_dir() / "resolved_case.yaml") ||
      !std::filesystem::exists(baseline_dir() / "expected.yaml")) {
    std::cerr << "Missing baseline dataset in " << baseline_dir().string() << "\n";
    return 7;
  }

  const MechanismThresholds thresholds = load_mechanism_thresholds();

  // CI mechanism test: short, robust check of fix-on vs fix-off at fixed iteration count.
  // Thresholds are sourced from tests/data/baselines/low_mach_euler/expected.yaml.
  const int iterations = thresholds.iterations;
  const int min_iterations = thresholds.min_iterations;
  const float all_speed_f_min = thresholds.all_speed_f_min;
  const int all_speed_ramp_start_iter = thresholds.all_speed_ramp_start_iter;
  const int all_speed_ramp_iters = thresholds.all_speed_ramp_iters;
  const bool precond_on = false;
  const SweepPoint baseline_off =
    run_point(thresholds.mach, false, iterations, min_iterations, all_speed_f_min,
              all_speed_ramp_start_iter, all_speed_ramp_iters, false, 0, precond_on,
              "m001_all_speed_off_short");
  const SweepPoint m001 =
    run_point(thresholds.mach, true, iterations, min_iterations, all_speed_f_min,
              all_speed_ramp_start_iter, all_speed_ramp_iters, false, 0, precond_on,
              "m001_all_speed_on_short");

  if (!finite_point(baseline_off) || !finite_point(m001)) {
    std::cerr << "Non-finite low-Mach asymptotic metric at M=" << thresholds.mach << "\n";
    return 2;
  }

  if (!(std::abs(m001.cd_times_mach) <
        thresholds.cd_times_mach_improvement_factor * std::abs(baseline_off.cd_times_mach))) {
    std::cerr << "all_speed_flux_fix did not reduce Cd*M enough at M=" << thresholds.mach << "."
              << " off=" << baseline_off.cd_times_mach << " on=" << m001.cd_times_mach << "\n";
    return 3;
  }

  std::cout << "low_mach_ci_result"
            << " off_cd_times_mach=" << baseline_off.cd_times_mach
            << " on_cd_times_mach=" << m001.cd_times_mach
            << " off_p_fluct=" << baseline_off.p_fluct_nd
            << " on_p_fluct=" << m001.p_fluct_nd << "\n";

  return 0;
}
