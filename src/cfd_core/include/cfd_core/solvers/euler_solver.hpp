#pragma once

#include "cfd_core/mesh.hpp"
#include "cfd_core/mesh/airfoil_mesh.hpp"
#include "cfd_core/numerics/euler_flux.hpp"
#include "cfd_core/numerics/reconstruction.hpp"
#include "cfd_core/post/forces.hpp"

#include <filesystem>
#include <string>
#include <vector>

namespace cfd::core {
struct EulerAirfoilCaseConfig {
  AirfoilMeshConfig mesh;
  int iterations = 400;
  int min_iterations = 40;
  int cfl_ramp_iters = 120;
  float gamma = 1.4f;
  float gas_constant = 287.05f;
  float mach = 0.15f;
  float aoa_deg = 2.0f;
  float p_inf = 101325.0f;
  float t_inf = 288.15f;
  float rho_inf = 0.0f;
  float cfl_start = 0.2f;
  float cfl_max = 1.2f;
  float residual_reduction_target = 1.0e-3f;
  float force_stability_tol = 2.0e-5f;
  int force_stability_window = 6;
  float force_mean_drift_tol = 2.0e-5f;
  int startup_first_order_iters = 25;
  int rk_stages = 3;
  bool local_time_stepping = true;
  bool precond_on = false;
  float precond_mach_ref = 0.15f;
  float precond_mach_min = 1.0e-3f;
  float precond_beta_min = 0.2f;
  float precond_beta_max = 1.0f;
  bool precond_farfield_bc = false;
  float stabilization_mach_floor_start = 0.1f;
  float stabilization_mach_floor_target = 0.02f;
  float stabilization_mach_floor_k_start = -1.0f;
  float stabilization_mach_floor_k_target = -1.0f;
  int stabilization_ramp_iters = 400;
  float x_ref = 0.25f;
  float y_ref = 0.0f;
  EulerFluxScheme flux_scheme = EulerFluxScheme::kRusanov;
  LimiterType limiter = LimiterType::kMinmod;
  // Deprecated knobs kept for config compatibility.
  bool low_mach_fix = false;
  float mach_cutoff = 0.3f;
  bool all_speed_flux_fix = false;
  float all_speed_mach_cutoff = 0.25f;
  float all_speed_f_min = 0.05f;
  int all_speed_ramp_start_iter = 0;
  int all_speed_ramp_iters = 0;
  bool all_speed_staged_controller = false;
  int all_speed_stage_a_min_iters = 600;
  int all_speed_stage_a_max_iters = 0;
  float all_speed_stage_f_min_high = 0.2f;
  float all_speed_stage_f_min_mid = 0.1f;
  float all_speed_stage_f_min_low = 0.05f;
  float all_speed_pjump_wall_target = 0.0f;
  float all_speed_pjump_spike_factor = 1.5f;
  int all_speed_pjump_hold_iters = 120;
  int all_speed_freeze_iters = 120;
  float all_speed_cfl_drop_factor = 0.8f;
  float all_speed_cfl_restore_factor = 1.02f;
  float all_speed_cfl_min_scale = 0.1f;
  bool aoa0_symmetry_enforce = false;
  int aoa0_symmetry_enforce_interval = 0;
  std::filesystem::path output_dir = ".";
};

struct EulerIterationRecord {
  int iter = 0;
  float residual_l1 = 0.0f;
  float residual_l2 = 0.0f;
  float residual_linf = 0.0f;
  float cl = 0.0f;
  float cd = 0.0f;
  float cm = 0.0f;
};

struct EulerRunResult {
  UnstructuredMesh mesh;
  std::vector<float> conserved;  // [cell][rho,rhou,rhov,rhow,rhoE]
  std::vector<float> last_residual;       // [cell][rho,rhou,rhov,rhow,rhoE]
  std::vector<float> last_spectral_radius;
  std::vector<float> residual_magnitude;
  std::vector<float> rho;
  std::vector<float> u;
  std::vector<float> v;
  std::vector<float> w;
  std::vector<float> p;
  std::vector<float> mach;
  std::vector<EulerIterationRecord> history;
  std::vector<WallCpSample> wall_cp;
  ForceCoefficients forces;
  float max_wall_mass_flux = 0.0f;
  float beta_min = 1.0f;
  float beta_max = 1.0f;
  float beta_mean = 1.0f;
  float mach_local_min = 0.0f;
  float mach_local_max = 0.0f;
  float mach_local_mean = 0.0f;
  float acoustic_scale_min = 1.0f;
  float acoustic_scale_max = 1.0f;
  float acoustic_scale_mean = 1.0f;
  float acoustic_scale_final_min = 1.0f;
  float acoustic_scale_final_max = 1.0f;
  float acoustic_scale_final_mean = 1.0f;
  float wall_adjacent_acoustic_scale_min = 1.0f;
  float wall_adjacent_acoustic_scale_p01 = 1.0f;
  float wall_adjacent_acoustic_scale_p50 = 1.0f;
  float wall_adjacent_stabilization_scale_min = 1.0f;
  float wall_adjacent_stabilization_scale_p01 = 1.0f;
  float wall_adjacent_stabilization_scale_p50 = 1.0f;
  float wall_adjacent_acoustic_scale_final_min = 1.0f;
  float wall_adjacent_acoustic_scale_final_p01 = 1.0f;
  float wall_adjacent_acoustic_scale_final_p50 = 1.0f;
  bool force_window_stable_final = false;
  bool force_mean_drift_stable_final = false;
  bool force_physical_converged_final = false;
  float force_mean_dcl_final = 0.0f;
  float force_mean_dcd_final = 0.0f;
  float force_mean_dcm_final = 0.0f;
  float pressure_fluctuation_nd_max_final = 0.0f;
  float pressure_fluctuation_nd_wall_final = 0.0f;
  float pressure_fluctuation_nd_farfield_final = 0.0f;
  float cd_times_mach_final = 0.0f;
  float precond_mach_ref_effective = 0.15f;
  float precond_beta_min_effective = 0.2f;
  float precond_beta_max_effective = 1.0f;
  float stabilization_mach_floor_start_effective = 0.1f;
  float stabilization_mach_floor_target_effective = 0.02f;
  float stabilization_mach_floor_k_start_effective = 0.0f;
  float stabilization_mach_floor_k_target_effective = 0.0f;
  int stabilization_ramp_iters_effective = 400;
  float stabilization_mach_floor_final = 0.02f;
  std::filesystem::path residuals_csv_path;
  std::filesystem::path forces_csv_path;
  std::filesystem::path cp_csv_path;
  std::filesystem::path wall_flux_csv_path;
  std::filesystem::path dissipation_debug_csv_path;
  std::filesystem::path vtu_path;
};

EulerRunResult run_euler_airfoil_case(const EulerAirfoilCaseConfig& config,
                                      const std::string& backend = "cpu");
EulerRunResult run_euler_airfoil_case_cpu(const EulerAirfoilCaseConfig& config);
}  // namespace cfd::core
