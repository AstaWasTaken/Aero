#include "cfd_core/solvers/euler_solver.hpp"

#include "cfd_core/backend.hpp"
#include "cfd_core/io_vtk.hpp"
#include "cfd_core/numerics/reconstruction.hpp"
#include "cfd_core/post/forces.hpp"
#include "cfd_core/solvers/euler/boundary_conditions.hpp"
#include "cfd_core/solvers/euler/convergence_monitor.hpp"
#include "cfd_core/solvers/euler/mesh_geometry.hpp"
#include "cfd_core/solvers/euler/preconditioning.hpp"
#include "cfd_core/solvers/euler/residual_assembly.hpp"
#include "cfd_core/solvers/euler/time_integrator.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#if CFD_HAS_CUDA
#include "cfd_core/cuda_backend.hpp"
#endif

namespace cfd::core {
namespace {
constexpr float kPi = 3.14159265358979323846f;
constexpr float kStabilizationSmoothMaxEps = 1.0e-6f;
constexpr float kAutoPjumpCoeff = 1.6e-2f;
constexpr float kAutoPjumpTargetMin = 1.0e-9f;
constexpr float kAutoPjumpTargetMax = 2.0e-6f;

bool env_flag_on(const char* name) {
  const char* value = std::getenv(name);
  if (value == nullptr) {
    return false;
  }
  const std::string token(value);
  return token == "1" || token == "true" || token == "TRUE" || token == "on" || token == "ON";
}

ConservativeState load_conservative_state(const std::vector<float>& conserved, const int cell) {
  return {
    conserved[5 * cell + 0],
    conserved[5 * cell + 1],
    conserved[5 * cell + 2],
    conserved[5 * cell + 3],
    conserved[5 * cell + 4],
  };
}

void store_conservative_state(std::vector<float>* conserved, const int cell,
                              const ConservativeState& state) {
  if (conserved == nullptr) {
    return;
  }
  (*conserved)[5 * cell + 0] = state.rho;
  (*conserved)[5 * cell + 1] = state.rhou;
  (*conserved)[5 * cell + 2] = state.rhov;
  (*conserved)[5 * cell + 3] = state.rhow;
  (*conserved)[5 * cell + 4] = state.rhoE;
}

float compute_cfl(const EulerAirfoilCaseConfig& config, const int iter, const float scale) {
  const float base_cfl = [&]() {
    if (config.cfl_ramp_iters <= 0) {
      return config.cfl_max;
    }
    const float t = std::min(1.0f, static_cast<float>(iter + 1) /
                                     static_cast<float>(std::max(config.cfl_ramp_iters, 1)));
    return config.cfl_start + t * (config.cfl_max - config.cfl_start);
  }();
  return base_cfl * scale;
}

float clamp_mach_scale(const float value) {
  return std::clamp(value, 0.0f, 1.0f);
}

float clamp_nonnegative(const float value) {
  return std::max(value, 0.0f);
}

float smooth_max(const float a, const float b) {
  const float diff = a - b;
  return 0.5f * (a + b + std::sqrt(diff * diff +
                                   kStabilizationSmoothMaxEps * kStabilizationSmoothMaxEps));
}

float compute_stabilization_mach_floor(const float start, const float target, const int ramp_iters,
                                       const int iter) {
  const float m0_start = clamp_mach_scale(start);
  const float m0_target = clamp_mach_scale(target);
  if (ramp_iters <= 0) {
    return m0_target;
  }
  const float t = std::clamp(static_cast<float>(std::max(iter, 0)) /
                               static_cast<float>(std::max(ramp_iters, 1)),
                             0.0f, 1.0f);
  return m0_start + t * (m0_target - m0_start);
}

float compute_all_speed_ramp_weight(const int iter_on, const int ramp_len, const int iter) {
  if (ramp_len <= 0) {
    return (iter >= iter_on) ? 1.0f : 0.0f;
  }
  const float x = static_cast<float>(iter - iter_on) / static_cast<float>(std::max(ramp_len, 1));
  return std::clamp(x, 0.0f, 1.0f);
}

enum class AllSpeedControlStage {
  kSingle = 0,
  kStageA = 1,
  kStageB = 2,
};

const char* all_speed_stage_name(const AllSpeedControlStage stage) {
  switch (stage) {
    case AllSpeedControlStage::kStageA:
      return "A";
    case AllSpeedControlStage::kStageB:
      return "B";
    case AllSpeedControlStage::kSingle:
    default:
      return "single";
  }
}

struct StabilizationFloorSchedule {
  float mach_floor_start = 0.0f;
  float mach_floor_target = 0.0f;
  float k_start = 0.0f;
  float k_target = 0.0f;
};

StabilizationFloorSchedule resolve_stabilization_floor_schedule(const EulerAirfoilCaseConfig& config) {
  const float mach_inf = std::max(config.mach, 1.0e-4f);
  const bool has_k_start = config.stabilization_mach_floor_k_start >= 0.0f;
  const bool has_k_target = config.stabilization_mach_floor_k_target >= 0.0f;
  const float k_start = has_k_start ? config.stabilization_mach_floor_k_start
                                    : (config.stabilization_mach_floor_start / mach_inf);
  const float k_target = has_k_target ? config.stabilization_mach_floor_k_target
                                      : (config.stabilization_mach_floor_target / mach_inf);
  const float k_start_clamped = clamp_nonnegative(k_start);
  const float k_target_clamped = clamp_nonnegative(k_target);
  return {
    clamp_mach_scale(k_start_clamped * mach_inf),
    clamp_mach_scale(k_target_clamped * mach_inf),
    k_start_clamped,
    k_target_clamped,
  };
}

PrimitiveGradients make_zero_gradients(const UnstructuredMesh& mesh) {
  PrimitiveGradients gradients;
  gradients.num_components = std::clamp(mesh.dimension, 2, 3);
  gradients.values.assign(static_cast<std::size_t>(mesh.num_cells) * 5 * gradients.num_components,
                          0.0f);
  return gradients;
}

std::vector<int> to_cuda_boundary_types(
  const std::vector<EulerBoundaryConditionType>& boundary_type) {
  std::vector<int> labels(boundary_type.size(), 0);
  for (std::size_t i = 0; i < boundary_type.size(); ++i) {
    if (boundary_type[i] == EulerBoundaryConditionType::kSlipWall) {
      labels[i] = 2;
    } else if (boundary_type[i] == EulerBoundaryConditionType::kFarfield) {
      labels[i] = 1;
    } else {
      labels[i] = 0;
    }
  }
  return labels;
}

struct FaceSetGeometryStats {
  int count = 0;
  int first_face = -1;
  int last_face = -1;
  float x_min = 0.0f;
  float x_max = 0.0f;
  float y_min = 0.0f;
  float y_max = 0.0f;
  float sum_ds = 0.0f;
  float sum_nx_ds = 0.0f;
  float sum_ny_ds = 0.0f;
};

FaceSetGeometryStats summarize_face_geometry(const UnstructuredMesh& mesh,
                                             const std::vector<int>& faces) {
  FaceSetGeometryStats stats;
  stats.count = static_cast<int>(faces.size());
  if (faces.empty()) {
    return stats;
  }

  stats.first_face = faces.front();
  stats.last_face = faces.front();
  stats.x_min = std::numeric_limits<float>::max();
  stats.x_max = std::numeric_limits<float>::lowest();
  stats.y_min = std::numeric_limits<float>::max();
  stats.y_max = std::numeric_limits<float>::lowest();

  for (const int face : faces) {
    stats.first_face = std::min(stats.first_face, face);
    stats.last_face = std::max(stats.last_face, face);
    const float x = mesh.face_center[3 * face + 0];
    const float y = mesh.face_center[3 * face + 1];
    const float nx = mesh.face_normal[3 * face + 0];
    const float ny = mesh.face_normal[3 * face + 1];
    const float ds = mesh.face_area[face];
    stats.x_min = std::min(stats.x_min, x);
    stats.x_max = std::max(stats.x_max, x);
    stats.y_min = std::min(stats.y_min, y);
    stats.y_max = std::max(stats.y_max, y);
    stats.sum_ds += ds;
    stats.sum_nx_ds += nx * ds;
    stats.sum_ny_ds += ny * ds;
  }
  return stats;
}

int symmetric_difference_size(const int num_faces, const std::vector<int>& lhs,
                              const std::vector<int>& rhs) {
  std::vector<unsigned char> lhs_mask(static_cast<std::size_t>(num_faces), 0u);
  std::vector<unsigned char> rhs_mask(static_cast<std::size_t>(num_faces), 0u);
  for (const int face : lhs) {
    if (face >= 0 && face < num_faces) {
      lhs_mask[static_cast<std::size_t>(face)] = 1u;
    }
  }
  for (const int face : rhs) {
    if (face >= 0 && face < num_faces) {
      rhs_mask[static_cast<std::size_t>(face)] = 1u;
    }
  }
  int mismatch_count = 0;
  for (int face = 0; face < num_faces; ++face) {
    if (lhs_mask[static_cast<std::size_t>(face)] != rhs_mask[static_cast<std::size_t>(face)]) {
      ++mismatch_count;
    }
  }
  return mismatch_count;
}

void write_boundary_face_debug(
  const std::filesystem::path& output_dir, const UnstructuredMesh& mesh,
  const std::vector<int>& wall_patch_faces, const std::vector<int>& wall_bc_faces,
  const std::vector<int>& farfield_patch_faces, const std::vector<int>& farfield_bc_faces,
  const int wall_patch_bc_mismatch, const int farfield_patch_bc_mismatch,
  const FaceNormalQuality& normal_quality) {
  const FaceSetGeometryStats wall_patch_stats = summarize_face_geometry(mesh, wall_patch_faces);
  const FaceSetGeometryStats wall_bc_stats = summarize_face_geometry(mesh, wall_bc_faces);
  const FaceSetGeometryStats farfield_patch_stats =
    summarize_face_geometry(mesh, farfield_patch_faces);
  const FaceSetGeometryStats farfield_bc_stats = summarize_face_geometry(mesh, farfield_bc_faces);

  std::ofstream out(output_dir / "boundary_faces_debug.txt", std::ios::trunc);
  if (!out) {
    throw std::runtime_error("Failed to write boundary_faces_debug.txt.");
  }

  out << "boundary_patch_id,name,start_face,face_count\n";
  for (std::size_t i = 0; i < mesh.boundary_patches.size(); ++i) {
    const auto& patch = mesh.boundary_patches[i];
    out << i << "," << patch.name << "," << patch.start_face << "," << patch.face_count << "\n";
  }
  out << "\n";
  out << "set,count,first_face,last_face,x_min,x_max,y_min,y_max,sum_ds,sum_nx_ds,sum_ny_ds\n";
  out << "wall_patch," << wall_patch_stats.count << "," << wall_patch_stats.first_face << ","
      << wall_patch_stats.last_face << "," << wall_patch_stats.x_min << "," << wall_patch_stats.x_max
      << "," << wall_patch_stats.y_min << "," << wall_patch_stats.y_max << ","
      << wall_patch_stats.sum_ds << "," << wall_patch_stats.sum_nx_ds << ","
      << wall_patch_stats.sum_ny_ds << "\n";
  out << "wall_bc_type," << wall_bc_stats.count << "," << wall_bc_stats.first_face << ","
      << wall_bc_stats.last_face << "," << wall_bc_stats.x_min << "," << wall_bc_stats.x_max << ","
      << wall_bc_stats.y_min << "," << wall_bc_stats.y_max << "," << wall_bc_stats.sum_ds << ","
      << wall_bc_stats.sum_nx_ds << "," << wall_bc_stats.sum_ny_ds << "\n";
  out << "farfield_patch," << farfield_patch_stats.count << "," << farfield_patch_stats.first_face
      << "," << farfield_patch_stats.last_face << "," << farfield_patch_stats.x_min << ","
      << farfield_patch_stats.x_max << "," << farfield_patch_stats.y_min << ","
      << farfield_patch_stats.y_max << "," << farfield_patch_stats.sum_ds << ","
      << farfield_patch_stats.sum_nx_ds << "," << farfield_patch_stats.sum_ny_ds << "\n";
  out << "farfield_bc_type," << farfield_bc_stats.count << "," << farfield_bc_stats.first_face << ","
      << farfield_bc_stats.last_face << "," << farfield_bc_stats.x_min << ","
      << farfield_bc_stats.x_max << "," << farfield_bc_stats.y_min << ","
      << farfield_bc_stats.y_max << "," << farfield_bc_stats.sum_ds << ","
      << farfield_bc_stats.sum_nx_ds << "," << farfield_bc_stats.sum_ny_ds << "\n";
  out << "\n";
  out << "wall_patch_vs_bc_mismatch_faces=" << wall_patch_bc_mismatch << "\n";
  out << "farfield_patch_vs_bc_mismatch_faces=" << farfield_patch_bc_mismatch << "\n";
  out << "normal_normalized_faces=" << normal_quality.normalized_count << "\n";
  out << "normal_invalid_faces=" << normal_quality.invalid_count << "\n";
  out << "normal_max_norm_deviation=" << normal_quality.max_norm_deviation << "\n";
}

struct ResidualNorms {
  float l1 = 0.0f;
  float l2 = 0.0f;
  float linf = 0.0f;
};

ResidualNorms compute_residual_norms(const UnstructuredMesh& mesh, const std::vector<float>& residual,
                                     std::vector<float>* residual_magnitude) {
  if (residual_magnitude != nullptr) {
    residual_magnitude->assign(static_cast<std::size_t>(mesh.num_cells), 0.0f);
  }

  double residual_l1 = 0.0;
  double residual_l2_sum = 0.0;
  double residual_linf = 0.0;
  for (int cell = 0; cell < mesh.num_cells; ++cell) {
    const float inv_vol = 1.0f / read_cell_volume(mesh, cell);
    double cell_mag2 = 0.0;
    for (int k = 0; k < 5; ++k) {
      const float value = residual[static_cast<std::size_t>(5 * cell + k)] * inv_vol;
      const double mag = std::abs(static_cast<double>(value));
      residual_l1 += mag;
      residual_l2_sum += mag * mag;
      residual_linf = std::max(residual_linf, mag);
      cell_mag2 += static_cast<double>(value) * static_cast<double>(value);
    }
    if (residual_magnitude != nullptr) {
      (*residual_magnitude)[static_cast<std::size_t>(cell)] = static_cast<float>(std::sqrt(cell_mag2));
    }
  }

  ResidualNorms norms;
  norms.l1 = static_cast<float>(residual_l1);
  norms.l2 = static_cast<float>(std::sqrt(residual_l2_sum / std::max(1, mesh.num_cells * 5)));
  norms.linf = static_cast<float>(residual_linf);
  return norms;
}

LowMachPreconditionConfig build_precondition_config(const EulerAirfoilCaseConfig& config) {
  LowMachPreconditionConfig precondition;
  precondition.enabled = config.precond_on;
  const float mach_inf = std::max(config.mach, 1.0e-4f);
  const float auto_mach_ref = mach_inf;
  const float auto_beta_min = std::max(1.0e-4f, std::min(1.0e-3f, 0.1f * mach_inf));
  precondition.mach_ref =
    std::max(config.precond_mach_ref > 0.0f ? config.precond_mach_ref : auto_mach_ref, 1.0e-4f);
  precondition.mach_min = std::max(config.precond_mach_min > 0.0f ? config.precond_mach_min : 1.0e-6f,
                                   1.0e-6f);
  precondition.beta_min =
    std::max(config.precond_beta_min > 0.0f ? config.precond_beta_min : auto_beta_min, 1.0e-4f);
  precondition.beta_max = std::max(config.precond_beta_max, precondition.beta_min);
  return precondition;
}

struct AcousticScalingInfo {
  float beta = 1.0f;
  float face_mach = 1.0f;
  float low_mach_scale = 1.0f;
  float acoustic_scale_base = 1.0f;
  float stabilization_mach_floor = 0.0f;
  float stabilization_scale = 1.0f;
  float acoustic_scale = 1.0f;
};

float preconditioned_spectral_radius(const float un, const float a, const float acoustic_scale) {
  const float beta = std::clamp(acoustic_scale, 1.0e-4f, 1.0f);
  const float beta2 = beta * beta;
  const float one_minus_beta2 = 1.0f - beta2;
  const float u_tilde = 0.5f * (1.0f + beta2) * un;
  const float discriminant = one_minus_beta2 * one_minus_beta2 * un * un +
                             4.0f * beta2 * a * a;
  const float c_tilde = 0.5f * std::sqrt(std::max(discriminant, 0.0f));
  const float lambda_minus = u_tilde - c_tilde;
  const float lambda_plus = u_tilde + c_tilde;
  return std::max({std::abs(un), std::abs(lambda_minus), std::abs(lambda_plus), 1.0e-8f});
}

float linear_quantile_from_sorted(const std::vector<float>& sorted_values, const float q) {
  if (sorted_values.empty()) {
    return 1.0f;
  }
  const float qc = std::clamp(q, 0.0f, 1.0f);
  const float idx = qc * static_cast<float>(sorted_values.size() - 1);
  const std::size_t i0 = static_cast<std::size_t>(std::floor(idx));
  const std::size_t i1 = std::min(i0 + 1, sorted_values.size() - 1);
  const float t = idx - static_cast<float>(i0);
  return (1.0f - t) * sorted_values[i0] + t * sorted_values[i1];
}

AcousticScalingInfo compute_acoustic_scaling_info(const ConservativeState& left,
                                                  const ConservativeState& right,
                                                  const float gamma,
                                                  const LowMachPreconditionConfig& precondition,
                                                  const bool low_mach_fix,
                                                  const float mach_cutoff,
                                                  const float stabilization_mach_floor) {
  AcousticScalingInfo info;
  const PrimitiveState pl = conservative_to_primitive(left, gamma);
  const PrimitiveState pr = conservative_to_primitive(right, gamma);
  info.face_mach = 0.5f * (compute_local_mach(pl, gamma) + compute_local_mach(pr, gamma));
  info.beta = compute_face_precondition_beta(left, right, gamma, precondition);
  info.acoustic_scale_base = precondition.enabled ? std::clamp(info.beta, 1.0e-4f, 1.0f) : 1.0f;
  info.low_mach_scale = low_mach_fix
                          ? compute_low_mach_dissipation_scale(left, right, gamma, true,
                                                               mach_cutoff, precondition.mach_ref)
                          : 1.0f;
  info.acoustic_scale_base =
    std::clamp(std::min(info.acoustic_scale_base, info.low_mach_scale), 1.0e-4f, 1.0f);
  info.stabilization_mach_floor = clamp_mach_scale(stabilization_mach_floor);
  const float meff = smooth_max(info.face_mach, info.stabilization_mach_floor);
  info.stabilization_scale = std::clamp(meff, 1.0e-4f, 1.0f);
  info.acoustic_scale = std::max(info.acoustic_scale_base, info.stabilization_scale);
  return info;
}

void enforce_aoa0_symmetry_state(const float aoa_deg, const bool enable,
                                 const int enforce_interval, const int iter, const float gamma,
                                 std::vector<float>* conserved) {
  if (!enable || conserved == nullptr || enforce_interval <= 0) {
    return;
  }
  if (std::abs(aoa_deg) > 1.0e-12f) {
    return;
  }
  if (((iter + 1) % enforce_interval) != 0) {
    return;
  }

  const int num_cells = static_cast<int>(conserved->size() / 5);
  for (int cell = 0; cell < num_cells; ++cell) {
    const ConservativeState state = {
      (*conserved)[5 * cell + 0],
      (*conserved)[5 * cell + 1],
      (*conserved)[5 * cell + 2],
      (*conserved)[5 * cell + 3],
      (*conserved)[5 * cell + 4],
    };
    PrimitiveState prim = conservative_to_primitive(state, gamma);
    prim.v = 0.0f;
    prim.w = 0.0f;
    const ConservativeState sym = primitive_to_conservative(prim, gamma);
    (*conserved)[5 * cell + 0] = sym.rho;
    (*conserved)[5 * cell + 1] = sym.rhou;
    (*conserved)[5 * cell + 2] = sym.rhov;
    (*conserved)[5 * cell + 3] = sym.rhow;
    (*conserved)[5 * cell + 4] = sym.rhoE;
  }
}

#if CFD_HAS_CUDA
struct EulerCudaBuffersGuard {
  cfd::cuda_backend::EulerDeviceBuffers buffers;

  ~EulerCudaBuffersGuard() {
    cfd::cuda_backend::free_euler_device_buffers(&buffers);
  }
};
#endif
}  // namespace

EulerRunResult run_euler_airfoil_case(const EulerAirfoilCaseConfig& config,
                                      const std::string& backend) {
  if (config.iterations <= 0) {
    throw std::invalid_argument("Euler iterations must be positive.");
  }
  if (config.rk_stages != 1 && config.rk_stages != 3) {
    throw std::invalid_argument("Euler pseudo-time rk_stages must be 1 or 3.");
  }

  const std::string resolved_backend = normalize_backend(backend);
  EulerRunResult result;
  result.mesh = make_airfoil_ogrid_mesh(config.mesh);
  const FaceNormalQuality normal_quality = normalize_face_normals(&result.mesh);
  if (normal_quality.invalid_count > 0) {
    throw std::runtime_error("Mesh contains invalid face normals.");
  }

  const int num_cells = result.mesh.num_cells;
  const float gamma = config.gamma;
  const bool precond_requested = config.precond_on;
  const bool precond_experimental_enabled = env_flag_on("AERO_ENABLE_EXPERIMENTAL_PRECOND");
  const bool precond_forced_off = precond_requested && !precond_experimental_enabled;
  EulerAirfoilCaseConfig effective_config = config;
  if (precond_forced_off) {
    effective_config.precond_on = false;
    std::cerr << "preconditioning forced off (experimental)\n";
  }
  const LowMachPreconditionConfig precondition = build_precondition_config(effective_config);
  const StabilizationFloorSchedule stabilization_floor_schedule =
    resolve_stabilization_floor_schedule(config);
  const float stabilization_mach_floor_start = stabilization_floor_schedule.mach_floor_start;
  const float stabilization_mach_floor_target = stabilization_floor_schedule.mach_floor_target;
  const int stabilization_ramp_iters = std::max(config.stabilization_ramp_iters, 0);
  const float all_speed_f_min = std::clamp(config.all_speed_f_min, 1.0e-6f, 1.0f);
  const int all_speed_ramp_start_iter = std::max(config.all_speed_ramp_start_iter, 0);
  const int all_speed_ramp_iters = std::max(config.all_speed_ramp_iters, 0);
  const bool all_speed_staged_controller = config.all_speed_staged_controller;
  const int all_speed_stage_a_min_iters = std::max(config.all_speed_stage_a_min_iters, 0);
  const int all_speed_stage_a_max_iters = std::max(config.all_speed_stage_a_max_iters, 0);
  const float all_speed_stage_f_min_high =
    std::clamp(config.all_speed_stage_f_min_high, 1.0e-6f, 1.0f);
  const float all_speed_stage_f_min_mid = std::clamp(
    std::min(config.all_speed_stage_f_min_mid, all_speed_stage_f_min_high), 1.0e-6f, 1.0f);
  const float all_speed_stage_f_min_low = std::clamp(
    std::min(config.all_speed_stage_f_min_low, all_speed_stage_f_min_mid), 1.0e-6f, 1.0f);
  // Auto-scale p-jump target: dp ~ ½ρM²a² → dp²/(ρa²)² ~ M⁴/4
  const float mach_inf = std::max(config.mach, 1.0e-4f);
  const float auto_pjump = std::clamp(
    kAutoPjumpCoeff * mach_inf * mach_inf * mach_inf * mach_inf, kAutoPjumpTargetMin,
    kAutoPjumpTargetMax);
  const bool use_auto_pjump_wall_target = config.all_speed_pjump_wall_target <= 0.0f;
  const float all_speed_pjump_wall_target =
    use_auto_pjump_wall_target ? auto_pjump : config.all_speed_pjump_wall_target;
  const float all_speed_pjump_spike_factor = std::max(config.all_speed_pjump_spike_factor, 1.0f);
  const int all_speed_pjump_hold_iters = std::max(config.all_speed_pjump_hold_iters, 1);
  const int all_speed_freeze_iters = std::max(config.all_speed_freeze_iters, 0);
  const float all_speed_cfl_drop_factor =
    std::clamp(config.all_speed_cfl_drop_factor, 0.05f, 1.0f);
  const float all_speed_cfl_restore_factor =
    std::clamp(config.all_speed_cfl_restore_factor, 1.0f, 1.2f);
  const float all_speed_cfl_min_scale =
    std::clamp(config.all_speed_cfl_min_scale, 0.01f, 1.0f);
  result.precond_mach_ref_effective = precondition.mach_ref;
  result.precond_beta_min_effective = precondition.beta_min;
  result.precond_beta_max_effective = precondition.beta_max;
  result.stabilization_mach_floor_start_effective = stabilization_mach_floor_start;
  result.stabilization_mach_floor_target_effective = stabilization_mach_floor_target;
  result.stabilization_mach_floor_k_start_effective = stabilization_floor_schedule.k_start;
  result.stabilization_mach_floor_k_target_effective = stabilization_floor_schedule.k_target;
  result.stabilization_ramp_iters_effective = stabilization_ramp_iters;

  if (resolved_backend == "cuda" &&
      (result.mesh.dimension != 2 || config.flux_scheme != EulerFluxScheme::kRusanov ||
       config.limiter != LimiterType::kMinmod || precondition.enabled || config.low_mach_fix ||
       config.all_speed_flux_fix)) {
    throw std::invalid_argument(
      "CUDA Euler residual currently supports only 2D + flux=rusanov + limiter=minmod + "
      "precond=off + low_mach_fix=off + all_speed_flux_fix=off.");
  }

  float rho_inf = config.rho_inf;
  if (rho_inf <= 0.0f) {
    rho_inf = config.p_inf / (config.gas_constant * config.t_inf);
  }
  if (rho_inf <= 0.0f) {
    throw std::invalid_argument("Invalid freestream density. Set rho_inf or p_inf/T_inf.");
  }

  const float alpha = config.aoa_deg * (kPi / 180.0f);
  const float a_inf = std::sqrt(gamma * config.p_inf / rho_inf);
  const float v_inf = std::max(config.mach, 0.0f) * a_inf;
  const PrimitiveState primitive_inf = {
    rho_inf,
    v_inf * std::cos(alpha),
    v_inf * std::sin(alpha),
    0.0f,
    config.p_inf,
  };
  const ConservativeState conservative_inf = primitive_to_conservative(primitive_inf, gamma);

  result.conserved.assign(static_cast<std::size_t>(num_cells) * 5, 0.0f);
  for (int cell = 0; cell < num_cells; ++cell) {
    store_conservative_state(&result.conserved, cell, conservative_inf);
  }

  std::vector<PrimitiveState> primitive(static_cast<std::size_t>(num_cells));
  std::vector<float> residual(static_cast<std::size_t>(num_cells) * 5, 0.0f);
  std::vector<float> residual_preconditioned(static_cast<std::size_t>(num_cells) * 5, 0.0f);
  std::vector<float> spectral_radius(static_cast<std::size_t>(num_cells), 0.0f);
  std::vector<float> dt_over_volume(static_cast<std::size_t>(num_cells), 0.0f);
  std::vector<float> state_n(static_cast<std::size_t>(num_cells) * 5, 0.0f);
  std::vector<float> state_stage(static_cast<std::size_t>(num_cells) * 5, 0.0f);
  std::vector<float> state_next(static_cast<std::size_t>(num_cells) * 5, 0.0f);
  std::vector<float> cell_pressure(static_cast<std::size_t>(num_cells), config.p_inf);
  result.last_residual.assign(static_cast<std::size_t>(num_cells) * 5, 0.0f);
  result.last_spectral_radius.assign(static_cast<std::size_t>(num_cells), 0.0f);
  result.residual_magnitude.assign(static_cast<std::size_t>(num_cells), 0.0f);

  const std::vector<EulerBoundaryConditionType> face_boundary_type =
    build_euler_boundary_types(result.mesh);
  const std::vector<int> wall_patch_faces = find_patch_faces(result.mesh, "wall");
  std::vector<int> farfield_patch_faces = find_patch_faces(result.mesh, "farfield");
  if (farfield_patch_faces.empty()) {
    farfield_patch_faces = find_patch_faces(result.mesh, "boundary");
  }
  const std::vector<int> wall_bc_faces =
    collect_boundary_faces(result.mesh, face_boundary_type, EulerBoundaryConditionType::kSlipWall);
  const std::vector<int> farfield_bc_faces = collect_boundary_faces(
    result.mesh, face_boundary_type, EulerBoundaryConditionType::kFarfield);
  const int wall_patch_face_count = static_cast<int>(wall_patch_faces.size());
  const int wall_patch_bc_mismatch =
    symmetric_difference_size(result.mesh.num_faces, wall_patch_faces, wall_bc_faces);
  const int farfield_patch_bc_mismatch =
    symmetric_difference_size(result.mesh.num_faces, farfield_patch_faces, farfield_bc_faces);
  if (wall_patch_face_count <= 0) {
    throw std::runtime_error("Wall patch face set is empty.");
  }
  if (wall_patch_bc_mismatch != 0) {
    throw std::runtime_error("Wall patch and wall boundary-type face sets are inconsistent.");
  }

  const std::filesystem::path output_dir = config.output_dir.empty() ? "." : config.output_dir;
  std::filesystem::create_directories(output_dir);
  write_boundary_face_debug(output_dir, result.mesh, wall_patch_faces, wall_bc_faces,
                            farfield_patch_faces, farfield_bc_faces, wall_patch_bc_mismatch,
                            farfield_patch_bc_mismatch, normal_quality);

  result.residuals_csv_path = output_dir / "residuals.csv";
  result.forces_csv_path = output_dir / "forces.csv";
  result.cp_csv_path = output_dir / "cp_wall.csv";
  result.wall_flux_csv_path = output_dir / "wall_flux.csv";
  result.dissipation_debug_csv_path = output_dir / "dissipation_debug.csv";
  result.vtu_path = output_dir / "field_0000.vtu";
  const std::filesystem::path diagnostics_csv_path = output_dir / "euler_diagnostics.csv";

  const FreestreamReference reference = {
    rho_inf,
    config.p_inf,
    config.aoa_deg,
    v_inf,
    config.mesh.chord,
    config.x_ref,
    config.y_ref,
  };
  const float chord_ref = std::max(reference.chord, 1.0e-8f);
  const float s_ref = chord_ref * 1.0f;
  const float q_inf_ref =
    0.5f * reference.rho_inf * reference.speed_inf * reference.speed_inf + 1.0e-12f;
  const float pressure_ref = std::max(reference.rho_inf * a_inf * a_inf, 1.0e-12f);
  const FaceSetGeometryStats wall_force_face_stats =
    summarize_face_geometry(result.mesh, wall_patch_faces);
  std::vector<unsigned char> wall_adjacent_cell_mask(static_cast<std::size_t>(num_cells), 0u);
  for (const int face : wall_bc_faces) {
    const int owner = result.mesh.face_owner[face];
    if (owner >= 0 && owner < num_cells) {
      wall_adjacent_cell_mask[static_cast<std::size_t>(owner)] = 1u;
    }
  }
  {
    const float all_speed_f_min_effective_initial =
      all_speed_staged_controller ? all_speed_stage_f_min_high : all_speed_f_min;
    const bool all_speed_flux_fix_effective_initial =
      all_speed_staged_controller ? false : config.all_speed_flux_fix;
    const float all_speed_ramp_weight_initial =
      all_speed_staged_controller
        ? 0.0f
        : compute_all_speed_ramp_weight(all_speed_ramp_start_iter, all_speed_ramp_iters, 0);
    std::ofstream snapshot_csv(output_dir / "effective_config_snapshot.txt", std::ios::trunc);
    if (!snapshot_csv) {
      throw std::runtime_error("Failed to write effective_config_snapshot.txt.");
    }
    snapshot_csv << "iterations=" << config.iterations << "\n";
    snapshot_csv << "min_iterations=" << config.min_iterations << "\n";
    snapshot_csv << "aoa_deg=" << config.aoa_deg << "\n";
    snapshot_csv << "mach_inf=" << config.mach << "\n";
    snapshot_csv << "gamma=" << gamma << "\n";
    snapshot_csv << "u_inf=" << primitive_inf.u << "\n";
    snapshot_csv << "v_inf=" << primitive_inf.v << "\n";
    snapshot_csv << "rho_inf=" << rho_inf << "\n";
    snapshot_csv << "a_inf=" << a_inf << "\n";
    snapshot_csv << "q_inf=" << q_inf_ref << "\n";
    snapshot_csv << "cfl_start=" << config.cfl_start << "\n";
    snapshot_csv << "cfl_max=" << config.cfl_max << "\n";
    snapshot_csv << "cfl_ramp_iters=" << config.cfl_ramp_iters << "\n";
    snapshot_csv << "all_speed_flux_fix=" << (config.all_speed_flux_fix ? 1 : 0) << "\n";
    snapshot_csv << "all_speed_flux_fix_effective_initial="
                 << (all_speed_flux_fix_effective_initial ? 1 : 0) << "\n";
    snapshot_csv << "all_speed_mach_cutoff=" << config.all_speed_mach_cutoff << "\n";
    snapshot_csv << "all_speed_f_min=" << all_speed_f_min << "\n";
    snapshot_csv << "all_speed_ramp_start_iter=" << all_speed_ramp_start_iter << "\n";
    snapshot_csv << "all_speed_ramp_iters=" << all_speed_ramp_iters << "\n";
    snapshot_csv << "all_speed_staged_controller=" << (all_speed_staged_controller ? 1 : 0) << "\n";
    snapshot_csv << "all_speed_stage_initial="
                 << (all_speed_staged_controller ? "A" : "single") << "\n";
    snapshot_csv << "all_speed_stage_a_min_iters=" << all_speed_stage_a_min_iters << "\n";
    snapshot_csv << "all_speed_stage_a_max_iters=" << all_speed_stage_a_max_iters << "\n";
    snapshot_csv << "all_speed_stage_f_min_high=" << all_speed_stage_f_min_high << "\n";
    snapshot_csv << "all_speed_stage_f_min_mid=" << all_speed_stage_f_min_mid << "\n";
    snapshot_csv << "all_speed_stage_f_min_low=" << all_speed_stage_f_min_low << "\n";
    snapshot_csv << "all_speed_pjump_wall_target_requested=" << config.all_speed_pjump_wall_target
                 << "\n";
    snapshot_csv << "all_speed_pjump_wall_target_auto_enabled=" << (use_auto_pjump_wall_target ? 1 : 0)
                 << "\n";
    snapshot_csv << "all_speed_pjump_wall_target_auto_value=" << auto_pjump << "\n";
    snapshot_csv << "all_speed_pjump_wall_target_effective=" << all_speed_pjump_wall_target << "\n";
    snapshot_csv << "all_speed_pjump_spike_factor=" << all_speed_pjump_spike_factor << "\n";
    snapshot_csv << "all_speed_pjump_hold_iters=" << all_speed_pjump_hold_iters << "\n";
    snapshot_csv << "all_speed_freeze_iters=" << all_speed_freeze_iters << "\n";
    snapshot_csv << "all_speed_cfl_drop_factor=" << all_speed_cfl_drop_factor << "\n";
    snapshot_csv << "all_speed_cfl_restore_factor=" << all_speed_cfl_restore_factor << "\n";
    snapshot_csv << "all_speed_cfl_min_scale=" << all_speed_cfl_min_scale << "\n";
    snapshot_csv << "all_speed_f_min_effective_initial=" << all_speed_f_min_effective_initial << "\n";
    snapshot_csv << "all_speed_ramp_weight_initial=" << all_speed_ramp_weight_initial << "\n";
    snapshot_csv << "force_stability_tol=" << config.force_stability_tol << "\n";
    snapshot_csv << "force_stability_window=" << config.force_stability_window << "\n";
    snapshot_csv << "force_mean_drift_tol=" << config.force_mean_drift_tol << "\n";
    snapshot_csv << "residual_reduction_target=" << config.residual_reduction_target << "\n";
    snapshot_csv << "precond_requested=" << (precond_requested ? 1 : 0) << "\n";
    snapshot_csv << "precond_experimental_env_enabled=" << (precond_experimental_enabled ? 1 : 0)
                 << "\n";
    snapshot_csv << "precond_forced_off=" << (precond_forced_off ? 1 : 0) << "\n";
    snapshot_csv << "precond_effective=" << (precondition.enabled ? 1 : 0) << "\n";
    snapshot_csv << "precond_mach_ref_effective=" << precondition.mach_ref << "\n";
    snapshot_csv << "precond_mach_min=" << precondition.mach_min << "\n";
    snapshot_csv << "precond_beta_min_effective=" << precondition.beta_min << "\n";
    snapshot_csv << "precond_beta_max_effective=" << precondition.beta_max << "\n";
    snapshot_csv << "precond_farfield_bc=" << (config.precond_farfield_bc ? 1 : 0) << "\n";
    snapshot_csv << "stabilization_mach_floor_start_effective=" << stabilization_mach_floor_start
                 << "\n";
    snapshot_csv << "stabilization_mach_floor_target_effective=" << stabilization_mach_floor_target
                 << "\n";
    snapshot_csv << "stabilization_ramp_iters_effective=" << stabilization_ramp_iters << "\n";
  }

  std::ofstream residual_csv(result.residuals_csv_path, std::ios::trunc);
  std::ofstream forces_csv(result.forces_csv_path, std::ios::trunc);
  std::ofstream diagnostics_csv(diagnostics_csv_path, std::ios::trunc);
  if (!residual_csv || !forces_csv || !diagnostics_csv) {
    throw std::runtime_error("Failed to open Euler CSV outputs.");
  }
  residual_csv << "iter,residual_l1,residual_l2,residual_linf,residual_ratio\n";
  forces_csv << "iter,cl,cd,cm,lift,drag,moment,dcl,dcd,dcm,mean_dcl,mean_dcd,mean_dcm,"
                "force_window_stable,force_mean_drift_stable,force_physical_converged\n";
  diagnostics_csv
    << "iter,residual_ratio,force_window_stable,integrated_wall_faces,wall_patch_faces,"
       "wall_patch_bc_mismatch,sum_wall_ds,sum_nA_x,sum_nA_y,sum_nA_mag,max_wall_un,"
       "max_wall_mass_flux_n,max_farfield_abs_dp,max_farfield_abs_drho,max_farfield_abs_du,"
       "max_farfield_abs_dv,max_farfield_abs_dw,wall_cp_min,wall_cp_max,wall_cp_nonfinite,"
       "q_inf,rho_inf,V_inf,S_ref,chord,Fx_gauge,Fy_gauge,Fx_abs,Fy_abs,L,D,Cl,Cd,Cm,"
       "cl_abs,cd_abs,cm_abs,dcl,dcd,dcm,force_mean_drift_stable,force_physical_converged,"
       "mean_dcl,mean_dcd,mean_dcm,cd_sign_flip,p_jump_energy_nd,p_jump_energy_wall_nd,"
       "all_speed_stage,all_speed_flux_fix_effective,all_speed_f_min_effective,"
       "all_speed_ramp_weight_effective,all_speed_freeze_counter,all_speed_controller_action,"
       "cfl_scale_effective,cd_times_mach,p_fluct_nd_max,p_fluct_nd_wall,p_fluct_nd_farfield,"
       "num_faces_first_order_fallback,num_faces_diffusive_fallback,first_failure_face,"
       "first_failure_owner,first_failure_neighbor\n";

  ConvergenceMonitor monitor({config.residual_reduction_target, config.force_stability_tol,
                              config.force_stability_window, config.min_iterations,
                              config.force_mean_drift_tol});

  float cfl_scale = 1.0f;
  float previous_residual_l2 = -1.0f;
  float stabilization_mach_floor_current = stabilization_mach_floor_target;
  bool all_speed_flux_fix_effective = config.all_speed_flux_fix;
  float all_speed_f_min_effective = all_speed_f_min;
  float all_speed_ramp_weight_current = 1.0f;
  AllSpeedControlStage all_speed_stage =
    all_speed_staged_controller ? AllSpeedControlStage::kStageA : AllSpeedControlStage::kSingle;
  float all_speed_stage_b_f_min_current = all_speed_stage_f_min_high;
  int all_speed_stage_b_progress_iters = 0;
  int all_speed_pjump_below_counter = 0;
  int all_speed_freeze_counter = 0;
  bool first_face_failure_logged = false;
  bool non_finite_tripped = false;

#if CFD_HAS_CUDA
  EulerCudaBuffersGuard cuda_buffers_guard;
  if (resolved_backend == "cuda") {
    std::string error_message;
    const std::vector<int> cuda_boundary_type = to_cuda_boundary_types(face_boundary_type);
    if (!cfd::cuda_backend::init_euler_device_buffers(
          result.mesh, cuda_boundary_type, &cuda_buffers_guard.buffers, &error_message)) {
      throw std::runtime_error("CUDA Euler buffer initialization failed: " + error_message);
    }
  }
#endif

  for (int iter = 0; iter < config.iterations; ++iter) {
    const bool use_second_order = iter >= std::max(config.startup_first_order_iters, 0);
    const float cfl = compute_cfl(config, iter, cfl_scale);
    stabilization_mach_floor_current = compute_stabilization_mach_floor(
      stabilization_mach_floor_start, stabilization_mach_floor_target, stabilization_ramp_iters, iter);
    if (!all_speed_staged_controller) {
      all_speed_flux_fix_effective = config.all_speed_flux_fix;
      all_speed_f_min_effective = all_speed_f_min;
      all_speed_ramp_weight_current =
        compute_all_speed_ramp_weight(all_speed_ramp_start_iter, all_speed_ramp_iters, iter);
    } else if (all_speed_stage == AllSpeedControlStage::kStageA) {
      all_speed_flux_fix_effective = false;
      all_speed_f_min_effective = all_speed_stage_f_min_high;
      all_speed_ramp_weight_current = 0.0f;
    } else {
      all_speed_flux_fix_effective = config.all_speed_flux_fix;
      all_speed_f_min_effective =
        std::clamp(all_speed_stage_b_f_min_current, all_speed_stage_f_min_low, all_speed_stage_f_min_high);
      if (all_speed_ramp_iters <= 0) {
        all_speed_ramp_weight_current = 1.0f;
      } else {
        all_speed_ramp_weight_current = std::clamp(
          static_cast<float>(all_speed_stage_b_progress_iters) /
            static_cast<float>(std::max(all_speed_ramp_iters, 1)),
          0.0f, 1.0f);
      }
    }
    state_n = result.conserved;
    state_stage = state_n;
    EulerResidualAssemblyDiagnostics iter_residual_diagnostics;

    for (int stage = 0; stage < config.rk_stages; ++stage) {
      for (int cell = 0; cell < num_cells; ++cell) {
        primitive[static_cast<std::size_t>(cell)] =
          conservative_to_primitive(load_conservative_state(state_stage, cell), gamma);
      }

      PrimitiveGradients gradients =
        use_second_order ? compute_green_gauss_gradients(result.mesh, primitive)
                         : make_zero_gradients(result.mesh);
      std::fill(residual.begin(), residual.end(), 0.0f);
      std::fill(spectral_radius.begin(), spectral_radius.end(), 0.0f);

      if (resolved_backend == "cpu") {
        const EulerResidualAssemblyConfig assembly_config = {
          &result.mesh,
          &primitive,
          &gradients,
          &face_boundary_type,
          conservative_inf,
          gamma,
          config.flux_scheme,
          config.limiter,
          use_second_order,
          config.low_mach_fix,
          config.mach_cutoff,
          stabilization_mach_floor_current,
          config.precond_farfield_bc,
          precondition,
          all_speed_flux_fix_effective,
          config.all_speed_mach_cutoff,
          all_speed_f_min_effective,
          all_speed_ramp_weight_current,
        };
        EulerResidualAssemblyDiagnostics stage_residual_diagnostics;
        assemble_euler_residual_cpu(assembly_config, &residual, &spectral_radius,
                                    &stage_residual_diagnostics);
        iter_residual_diagnostics.num_faces_first_order_fallback +=
          stage_residual_diagnostics.num_faces_first_order_fallback;
        iter_residual_diagnostics.num_faces_diffusive_fallback +=
          stage_residual_diagnostics.num_faces_diffusive_fallback;
        if (iter_residual_diagnostics.first_failure_face < 0 &&
            stage_residual_diagnostics.first_failure_face >= 0) {
          iter_residual_diagnostics.first_failure_face =
            stage_residual_diagnostics.first_failure_face;
          iter_residual_diagnostics.first_failure_owner =
            stage_residual_diagnostics.first_failure_owner;
          iter_residual_diagnostics.first_failure_neighbor =
            stage_residual_diagnostics.first_failure_neighbor;
          iter_residual_diagnostics.first_failure_left =
            stage_residual_diagnostics.first_failure_left;
          iter_residual_diagnostics.first_failure_right =
            stage_residual_diagnostics.first_failure_right;
        }
        if (!first_face_failure_logged && stage_residual_diagnostics.first_failure_face >= 0) {
          std::cerr << "first_reconstruction_failure face="
                    << stage_residual_diagnostics.first_failure_face
                    << " owner=" << stage_residual_diagnostics.first_failure_owner
                    << " neighbor=" << stage_residual_diagnostics.first_failure_neighbor
                    << " left=(rho=" << stage_residual_diagnostics.first_failure_left.rho
                    << ",u=" << stage_residual_diagnostics.first_failure_left.u
                    << ",v=" << stage_residual_diagnostics.first_failure_left.v
                    << ",p=" << stage_residual_diagnostics.first_failure_left.p << ")"
                    << " right=(rho=" << stage_residual_diagnostics.first_failure_right.rho
                    << ",u=" << stage_residual_diagnostics.first_failure_right.u
                    << ",v=" << stage_residual_diagnostics.first_failure_right.v
                    << ",p=" << stage_residual_diagnostics.first_failure_right.p << ")\n";
          first_face_failure_logged = true;
        }
      } else {
#if CFD_HAS_CUDA
        if (use_second_order && gradients.num_components != 2) {
          throw std::runtime_error("CUDA Euler residual supports only 2D gradient storage.");
        }
        cfd::cuda_backend::EulerResidualConfig cuda_config;
        cuda_config.gamma = gamma;
        cuda_config.use_second_order = use_second_order;
        cuda_config.farfield_state = conservative_inf;

        std::string error_message;
        const float* gradient_ptr =
          (use_second_order && !gradients.values.empty()) ? gradients.values.data() : nullptr;
        if (!cfd::cuda_backend::euler_residual_cuda(
              result.mesh, cuda_config, &cuda_buffers_guard.buffers, state_stage.data(),
              gradient_ptr, residual.data(), spectral_radius.data(), &error_message)) {
          throw std::runtime_error("CUDA Euler residual failed: " + error_message);
        }
#else
        throw std::runtime_error("CUDA backend requested but unavailable.");
#endif
      }

      residual_preconditioned = residual;
      apply_low_mach_preconditioned_residual(primitive, gamma, precondition, &residual_preconditioned);
      dt_over_volume =
        compute_pseudo_time_step_over_volume(spectral_radius, cfl, config.local_time_stepping);
      apply_rk3_stage_update(stage, state_n, state_stage, residual_preconditioned, dt_over_volume,
                             gamma, &state_next);
      state_stage.swap(state_next);
    }

    result.conserved = state_stage;
    enforce_aoa0_symmetry_state(config.aoa_deg, config.aoa0_symmetry_enforce,
                                config.aoa0_symmetry_enforce_interval, iter, gamma,
                                &result.conserved);
    result.last_residual = residual;
    result.last_spectral_radius = spectral_radius;
    for (std::size_t idx = 0; idx < result.conserved.size(); ++idx) {
      const float value = result.conserved[idx];
      if (!std::isfinite(value)) {
        const int cell = static_cast<int>(idx / 5);
        const int component = static_cast<int>(idx % 5);
        std::cerr << "non_finite_trap iter=" << iter << " location=conserved cell=" << cell
                  << " component=" << component << " value=" << value << "\n";
        non_finite_tripped = true;
        break;
      }
    }
    if (non_finite_tripped) {
      break;
    }

    float max_abs_dp_domain = 0.0f;
    float max_abs_dp_wall_adjacent = 0.0f;
    for (int cell = 0; cell < num_cells; ++cell) {
      const PrimitiveState prim =
        conservative_to_primitive(load_conservative_state(result.conserved, cell), gamma);
      if (!std::isfinite(prim.rho) || !std::isfinite(prim.u) || !std::isfinite(prim.v) ||
          !std::isfinite(prim.w) || !std::isfinite(prim.p)) {
        std::cerr << "non_finite_trap iter=" << iter << " location=primitive cell=" << cell
                  << " rho=" << prim.rho << " u=" << prim.u << " v=" << prim.v
                  << " w=" << prim.w << " p=" << prim.p << "\n";
        non_finite_tripped = true;
        break;
      }
      primitive[static_cast<std::size_t>(cell)] = prim;
      cell_pressure[static_cast<std::size_t>(cell)] = prim.p;
      const float abs_dp = std::abs(prim.p - config.p_inf);
      max_abs_dp_domain = std::max(max_abs_dp_domain, abs_dp);
      if (wall_adjacent_cell_mask[static_cast<std::size_t>(cell)] != 0u) {
        max_abs_dp_wall_adjacent = std::max(max_abs_dp_wall_adjacent, abs_dp);
      }
    }
    if (non_finite_tripped) {
      break;
    }

    const ResidualNorms residual_norms =
      compute_residual_norms(result.mesh, residual, &result.residual_magnitude);
    const float residual_ratio = monitor.update_residual(residual_norms.l2);
    if (previous_residual_l2 > 0.0f) {
      if (residual_norms.l2 > previous_residual_l2 * 1.05f) {
        cfl_scale = std::max(cfl_scale * 0.65f, 0.05f);
      } else if (residual_norms.l2 < previous_residual_l2 * 0.90f) {
        cfl_scale = std::min(cfl_scale * 1.03f, 1.0f);
      }
    }
    previous_residual_l2 = residual_norms.l2;

    const PressureForceDiagnostics pressure_force_diag =
      compute_pressure_force_diagnostics(result.mesh, cell_pressure, reference, "wall");
    if (pressure_force_diag.integrated_face_count != wall_patch_face_count) {
      throw std::runtime_error("Integrated wall force face count does not match wall patch face count.");
    }
    const ForceCoefficients forces_abs =
      integrate_pressure_forces(result.mesh, cell_pressure, reference, "wall", false);
    const ForceCoefficients forces_gauge =
      integrate_pressure_forces(result.mesh, cell_pressure, reference, "wall", true);
    result.forces = forces_gauge;
    const ForceDrift force_drift = monitor.update_forces(result.forces);
    const bool force_window_stable = monitor.forces_stable();
    const ForceMeanDrift force_mean_drift = monitor.force_mean_drift();
    const bool force_mean_drift_stable = monitor.forces_mean_drift_stable();
    const bool force_physical_converged = monitor.forces_physically_converged();

    float max_wall_un = 0.0f;
    float max_wall_mass_flux = 0.0f;
    float wall_cp_min = std::numeric_limits<float>::max();
    float wall_cp_max = std::numeric_limits<float>::lowest();
    int wall_cp_nonfinite = 0;
    for (const int face : wall_bc_faces) {
      const int owner = result.mesh.face_owner[face];
      const ConservativeState interior = load_conservative_state(result.conserved, owner);
      const PrimitiveState prim = primitive[static_cast<std::size_t>(owner)];
      const std::array<float, 3> normal = read_face_unit_normal(result.mesh, face);
      const float un = prim.u * normal[0] + prim.v * normal[1] + prim.w * normal[2];
      max_wall_un = std::max(max_wall_un, std::abs(un));

      const AcousticScalingInfo scaling = compute_acoustic_scaling_info(
        interior, interior, gamma, precondition, config.low_mach_fix, config.mach_cutoff,
        stabilization_mach_floor_current);
      const BoundaryFluxResult wall_flux =
        compute_boundary_flux(EulerBoundaryConditionType::kSlipWall, interior, conservative_inf,
                              normal, config.flux_scheme, gamma, scaling.acoustic_scale, false,
                              all_speed_flux_fix_effective, config.all_speed_mach_cutoff,
                              all_speed_f_min_effective, all_speed_ramp_weight_current);
      max_wall_mass_flux = std::max(max_wall_mass_flux, std::abs(wall_flux.flux[0]));

      const float cp = (prim.p - config.p_inf) / q_inf_ref;
      if (std::isfinite(cp)) {
        wall_cp_min = std::min(wall_cp_min, cp);
        wall_cp_max = std::max(wall_cp_max, cp);
      } else {
        ++wall_cp_nonfinite;
      }
    }
    if (wall_bc_faces.empty()) {
      wall_cp_min = 0.0f;
      wall_cp_max = 0.0f;
    }
    result.max_wall_mass_flux = max_wall_mass_flux;

    float max_farfield_abs_dp = 0.0f;
    float max_farfield_abs_drho = 0.0f;
    float max_farfield_abs_du = 0.0f;
    float max_farfield_abs_dv = 0.0f;
    float max_farfield_abs_dw = 0.0f;
    for (const int face : farfield_bc_faces) {
      const int owner = result.mesh.face_owner[face];
      const PrimitiveState prim = primitive[static_cast<std::size_t>(owner)];
      max_farfield_abs_dp = std::max(max_farfield_abs_dp, std::abs(prim.p - config.p_inf));
      max_farfield_abs_drho = std::max(max_farfield_abs_drho, std::abs(prim.rho - rho_inf));
      max_farfield_abs_du = std::max(max_farfield_abs_du, std::abs(prim.u - primitive_inf.u));
      max_farfield_abs_dv = std::max(max_farfield_abs_dv, std::abs(prim.v - primitive_inf.v));
      max_farfield_abs_dw = std::max(max_farfield_abs_dw, std::abs(prim.w - primitive_inf.w));
    }
    double pressure_jump_energy_sum = 0.0;
    double pressure_jump_energy_wall_sum = 0.0;
    int pressure_jump_count = 0;
    int pressure_jump_wall_count = 0;
    for (int face = 0; face < result.mesh.num_faces; ++face) {
      const int owner = result.mesh.face_owner[face];
      const int neighbor = result.mesh.face_neighbor[face];
      if (neighbor < 0) {
        continue;
      }
      const float dp = primitive[static_cast<std::size_t>(owner)].p -
                       primitive[static_cast<std::size_t>(neighbor)].p;
      const double energy = static_cast<double>(dp) * static_cast<double>(dp);
      pressure_jump_energy_sum += energy;
      ++pressure_jump_count;
      if (wall_adjacent_cell_mask[static_cast<std::size_t>(owner)] != 0u ||
          wall_adjacent_cell_mask[static_cast<std::size_t>(neighbor)] != 0u) {
        pressure_jump_energy_wall_sum += energy;
        ++pressure_jump_wall_count;
      }
    }
    const double pressure_ref2 =
      static_cast<double>(pressure_ref) * static_cast<double>(pressure_ref) + 1.0e-24;
    const float p_jump_energy_nd =
      pressure_jump_count > 0
        ? static_cast<float>((pressure_jump_energy_sum / static_cast<double>(pressure_jump_count)) /
                             pressure_ref2)
        : 0.0f;
    const float p_jump_energy_wall_nd =
      pressure_jump_wall_count > 0
        ? static_cast<float>((pressure_jump_energy_wall_sum /
                              static_cast<double>(pressure_jump_wall_count)) /
                             pressure_ref2)
        : 0.0f;
    const float p_fluct_nd_max = max_abs_dp_domain / pressure_ref;
    const float p_fluct_nd_wall = max_abs_dp_wall_adjacent / pressure_ref;
    const float p_fluct_nd_farfield = max_farfield_abs_dp / pressure_ref;
    const float cd_times_mach = result.forces.cd * std::max(config.mach, 0.0f);
    const bool cd_sign_flip =
      !result.history.empty() &&
      ((result.history.back().cd >= 0.0f) != (result.forces.cd >= 0.0f));
    int all_speed_controller_action = 0;
    if (all_speed_staged_controller) {
      if (all_speed_stage == AllSpeedControlStage::kStageA) {
        const bool stage_a_ready =
          (iter + 1 >= all_speed_stage_a_min_iters) && force_window_stable;
        const bool stage_a_timeout =
          (all_speed_stage_a_max_iters > 0) && (iter + 1 >= all_speed_stage_a_max_iters);
        if ((stage_a_ready || stage_a_timeout) && config.all_speed_flux_fix) {
          all_speed_stage = AllSpeedControlStage::kStageB;
          all_speed_stage_b_f_min_current = all_speed_stage_f_min_high;
          all_speed_stage_b_progress_iters = 0;
          all_speed_pjump_below_counter = 0;
          all_speed_freeze_counter = 0;
          all_speed_controller_action = stage_a_ready ? 10 : 11;
        }
      } else if (all_speed_stage == AllSpeedControlStage::kStageB) {
        const float spike_threshold =
          (all_speed_pjump_wall_target > 0.0f)
            ? all_speed_pjump_wall_target * all_speed_pjump_spike_factor
            : std::numeric_limits<float>::max();
        const bool pjump_spike = p_jump_energy_wall_nd > spike_threshold;
        const bool pjump_good = (all_speed_pjump_wall_target <= 0.0f) ||
                                (p_jump_energy_wall_nd <= all_speed_pjump_wall_target);
        if (pjump_spike) {
          all_speed_freeze_counter = all_speed_freeze_iters;
          all_speed_pjump_below_counter = 0;
          all_speed_stage_b_f_min_current = std::min(
            all_speed_stage_f_min_high, all_speed_stage_b_f_min_current + 0.05f);
          cfl_scale = std::max(cfl_scale * all_speed_cfl_drop_factor, all_speed_cfl_min_scale);
          all_speed_controller_action = 20;
        } else {
          if (all_speed_freeze_counter > 0) {
            --all_speed_freeze_counter;
          }
          if (pjump_good) {
            ++all_speed_pjump_below_counter;
          } else {
            all_speed_pjump_below_counter = 0;
          }

          if (all_speed_freeze_counter == 0) {
            if (all_speed_stage_b_f_min_current > all_speed_stage_f_min_mid + 1.0e-6f &&
                all_speed_pjump_below_counter >= all_speed_pjump_hold_iters) {
              all_speed_stage_b_f_min_current = all_speed_stage_f_min_mid;
              all_speed_pjump_below_counter = 0;
              all_speed_controller_action = 31;
            } else if (all_speed_stage_b_f_min_current > all_speed_stage_f_min_low + 1.0e-6f &&
                       all_speed_pjump_below_counter >= all_speed_pjump_hold_iters) {
              all_speed_stage_b_f_min_current = all_speed_stage_f_min_low;
              all_speed_pjump_below_counter = 0;
              all_speed_controller_action = 32;
            } else if (all_speed_stage_b_f_min_current <= all_speed_stage_f_min_low + 1.0e-6f &&
                       pjump_good) {
              ++all_speed_stage_b_progress_iters;
              if (all_speed_controller_action == 0) {
                all_speed_controller_action = 33;
              }
            }

            if (pjump_good) {
              cfl_scale = std::min(cfl_scale * all_speed_cfl_restore_factor, 1.0f);
            } else if (all_speed_pjump_wall_target > 0.0f &&
                       p_jump_energy_wall_nd > all_speed_pjump_wall_target) {
              cfl_scale = std::max(cfl_scale * all_speed_cfl_drop_factor, all_speed_cfl_min_scale);
              if (all_speed_controller_action == 0) {
                all_speed_controller_action = 21;
              }
            }
          }
        }
      }
    }
    result.force_window_stable_final = force_window_stable;
    result.force_mean_drift_stable_final = force_mean_drift_stable;
    result.force_physical_converged_final = force_physical_converged;
    result.force_mean_dcl_final = force_mean_drift.dcl;
    result.force_mean_dcd_final = force_mean_drift.dcd;
    result.force_mean_dcm_final = force_mean_drift.dcm;
    result.pressure_fluctuation_nd_max_final = p_fluct_nd_max;
    result.pressure_fluctuation_nd_wall_final = p_fluct_nd_wall;
    result.pressure_fluctuation_nd_farfield_final = p_fluct_nd_farfield;
    result.cd_times_mach_final = cd_times_mach;

    EulerIterationRecord record;
    record.iter = iter;
    record.residual_l1 = residual_norms.l1;
    record.residual_l2 = residual_norms.l2;
    record.residual_linf = residual_norms.linf;
    record.cl = result.forces.cl;
    record.cd = result.forces.cd;
    record.cm = result.forces.cm;
    result.history.push_back(record);

    residual_csv << iter << "," << residual_norms.l1 << "," << residual_norms.l2 << ","
                 << residual_norms.linf << "," << residual_ratio << "\n";
    forces_csv << iter << "," << result.forces.cl << "," << result.forces.cd << ","
               << result.forces.cm << "," << result.forces.lift << "," << result.forces.drag
               << "," << result.forces.moment << "," << force_drift.dcl << "," << force_drift.dcd
               << "," << force_drift.dcm << "," << force_mean_drift.dcl << ","
               << force_mean_drift.dcd << "," << force_mean_drift.dcm << ","
               << (force_window_stable ? 1 : 0) << "," << (force_mean_drift_stable ? 1 : 0)
               << "," << (force_physical_converged ? 1 : 0) << "\n";

    diagnostics_csv << iter << "," << residual_ratio << "," << (force_window_stable ? 1 : 0) << ","
                    << pressure_force_diag.integrated_face_count << "," << wall_patch_face_count
                    << "," << wall_patch_bc_mismatch << "," << wall_force_face_stats.sum_ds << ","
                    << pressure_force_diag.sum_nA_x << "," << pressure_force_diag.sum_nA_y << ","
                    << std::sqrt(pressure_force_diag.sum_nA_x * pressure_force_diag.sum_nA_x +
                                 pressure_force_diag.sum_nA_y * pressure_force_diag.sum_nA_y)
                    << "," << max_wall_un << "," << max_wall_mass_flux << ","
                    << max_farfield_abs_dp << "," << max_farfield_abs_drho << ","
                    << max_farfield_abs_du << "," << max_farfield_abs_dv << ","
                    << max_farfield_abs_dw << "," << wall_cp_min << "," << wall_cp_max << ","
                    << wall_cp_nonfinite << "," << q_inf_ref << "," << reference.rho_inf << ","
                    << reference.speed_inf << "," << s_ref << "," << chord_ref << ","
                    << pressure_force_diag.fx_gauge << "," << pressure_force_diag.fy_gauge << ","
                    << pressure_force_diag.fx_abs << "," << pressure_force_diag.fy_abs << ","
                    << result.forces.lift << "," << result.forces.drag << "," << result.forces.cl
                    << "," << result.forces.cd << "," << result.forces.cm << "," << forces_abs.cl
                    << "," << forces_abs.cd << "," << forces_abs.cm << "," << force_drift.dcl << ","
                    << force_drift.dcd << "," << force_drift.dcm << ","
                    << (force_mean_drift_stable ? 1 : 0) << ","
                    << (force_physical_converged ? 1 : 0) << "," << force_mean_drift.dcl << ","
                    << force_mean_drift.dcd << "," << force_mean_drift.dcm << ","
                    << (cd_sign_flip ? 1 : 0) << "," << p_jump_energy_nd << ","
                    << p_jump_energy_wall_nd << "," << all_speed_stage_name(all_speed_stage) << ","
                    << (all_speed_flux_fix_effective ? 1 : 0) << "," << all_speed_f_min_effective
                    << "," << all_speed_ramp_weight_current << "," << all_speed_freeze_counter
                    << "," << all_speed_controller_action << "," << cfl_scale << ","
                    << cd_times_mach << "," << p_fluct_nd_max << "," << p_fluct_nd_wall << ","
                    << p_fluct_nd_farfield << ","
                    << iter_residual_diagnostics.num_faces_first_order_fallback << ","
                    << iter_residual_diagnostics.num_faces_diffusive_fallback << ","
                    << iter_residual_diagnostics.first_failure_face << ","
                    << iter_residual_diagnostics.first_failure_owner << ","
                    << iter_residual_diagnostics.first_failure_neighbor << "\n";

    if (monitor.converged(iter, residual_ratio)) {
      break;
    }
  }

  if (non_finite_tripped) {
    result.force_window_stable_final = false;
    result.force_mean_drift_stable_final = false;
    result.force_physical_converged_final = false;
  }
  result.stabilization_mach_floor_final = stabilization_mach_floor_current;

  result.rho.assign(static_cast<std::size_t>(num_cells), 0.0f);
  result.u.assign(static_cast<std::size_t>(num_cells), 0.0f);
  result.v.assign(static_cast<std::size_t>(num_cells), 0.0f);
  result.w.assign(static_cast<std::size_t>(num_cells), 0.0f);
  result.p.assign(static_cast<std::size_t>(num_cells), 0.0f);
  result.mach.assign(static_cast<std::size_t>(num_cells), 0.0f);
  result.beta_min = std::numeric_limits<float>::max();
  result.beta_max = std::numeric_limits<float>::lowest();
  result.mach_local_min = std::numeric_limits<float>::max();
  result.mach_local_max = std::numeric_limits<float>::lowest();
  double beta_sum = 0.0;
  double mach_sum = 0.0;
  for (int cell = 0; cell < num_cells; ++cell) {
    const PrimitiveState prim =
      conservative_to_primitive(load_conservative_state(result.conserved, cell), gamma);
    result.rho[static_cast<std::size_t>(cell)] = prim.rho;
    result.u[static_cast<std::size_t>(cell)] = prim.u;
    result.v[static_cast<std::size_t>(cell)] = prim.v;
    result.w[static_cast<std::size_t>(cell)] = prim.w;
    result.p[static_cast<std::size_t>(cell)] = prim.p;
    const float a = speed_of_sound(prim, gamma);
    const float velocity_mag = std::sqrt(prim.u * prim.u + prim.v * prim.v + prim.w * prim.w);
    const float local_mach = velocity_mag / std::max(a, 1.0e-8f);
    const float beta = compute_cell_precondition_beta(prim, gamma, precondition);
    result.mach[static_cast<std::size_t>(cell)] = local_mach;
    result.beta_min = std::min(result.beta_min, beta);
    result.beta_max = std::max(result.beta_max, beta);
    result.mach_local_min = std::min(result.mach_local_min, local_mach);
    result.mach_local_max = std::max(result.mach_local_max, local_mach);
    beta_sum += beta;
    mach_sum += local_mach;
    cell_pressure[static_cast<std::size_t>(cell)] = prim.p;
  }
  if (num_cells > 0) {
    result.beta_mean = static_cast<float>(beta_sum / static_cast<double>(num_cells));
    result.mach_local_mean = static_cast<float>(mach_sum / static_cast<double>(num_cells));
  } else {
    result.beta_min = 1.0f;
    result.beta_max = 1.0f;
    result.beta_mean = 1.0f;
    result.mach_local_min = 0.0f;
    result.mach_local_max = 0.0f;
    result.mach_local_mean = 0.0f;
  }

  result.acoustic_scale_min = std::numeric_limits<float>::max();
  result.acoustic_scale_max = std::numeric_limits<float>::lowest();
  result.acoustic_scale_final_min = std::numeric_limits<float>::max();
  result.acoustic_scale_final_max = std::numeric_limits<float>::lowest();
  double acoustic_scale_sum = 0.0;
  double acoustic_scale_final_sum = 0.0;
  int acoustic_scale_count = 0;
  for (int face = 0; face < result.mesh.num_faces; ++face) {
    const int owner = result.mesh.face_owner[face];
    const int neighbor = result.mesh.face_neighbor[face];
    const ConservativeState left = load_conservative_state(result.conserved, owner);
    ConservativeState right = conservative_inf;
    if (neighbor >= 0) {
      right = load_conservative_state(result.conserved, neighbor);
    } else if (face_boundary_type[static_cast<std::size_t>(face)] == EulerBoundaryConditionType::kSlipWall) {
      right = left;
    }
    const AcousticScalingInfo scaling = compute_acoustic_scaling_info(
      left, right, gamma, precondition, config.low_mach_fix, config.mach_cutoff,
      result.stabilization_mach_floor_final);
    result.acoustic_scale_min = std::min(result.acoustic_scale_min, scaling.acoustic_scale_base);
    result.acoustic_scale_max = std::max(result.acoustic_scale_max, scaling.acoustic_scale_base);
    result.acoustic_scale_final_min = std::min(result.acoustic_scale_final_min, scaling.acoustic_scale);
    result.acoustic_scale_final_max = std::max(result.acoustic_scale_final_max, scaling.acoustic_scale);
    acoustic_scale_sum += scaling.acoustic_scale_base;
    acoustic_scale_final_sum += scaling.acoustic_scale;
    ++acoustic_scale_count;
  }
  if (acoustic_scale_count > 0) {
    result.acoustic_scale_mean =
      static_cast<float>(acoustic_scale_sum / static_cast<double>(acoustic_scale_count));
    result.acoustic_scale_final_mean =
      static_cast<float>(acoustic_scale_final_sum / static_cast<double>(acoustic_scale_count));
  } else {
    result.acoustic_scale_min = 1.0f;
    result.acoustic_scale_max = 1.0f;
    result.acoustic_scale_mean = 1.0f;
    result.acoustic_scale_final_min = 1.0f;
    result.acoustic_scale_final_max = 1.0f;
    result.acoustic_scale_final_mean = 1.0f;
  }

  {
    std::ofstream wall_flux_csv(result.wall_flux_csv_path, std::ios::trunc);
    if (!wall_flux_csv) {
      throw std::runtime_error("Failed to write wall_flux.csv.");
    }
    wall_flux_csv << "face,x,y,mass_flux_n\n";

    float final_max_wall_mass_flux = 0.0f;
    for (const int face : wall_bc_faces) {
      const int owner = result.mesh.face_owner[face];
      const ConservativeState interior = load_conservative_state(result.conserved, owner);
      const std::array<float, 3> normal = read_face_unit_normal(result.mesh, face);
      const AcousticScalingInfo scaling = compute_acoustic_scaling_info(
        interior, interior, gamma, precondition, config.low_mach_fix, config.mach_cutoff,
        result.stabilization_mach_floor_final);
      const BoundaryFluxResult wall_flux =
        compute_boundary_flux(EulerBoundaryConditionType::kSlipWall, interior, conservative_inf,
                              normal, config.flux_scheme, gamma, scaling.acoustic_scale, false,
                              all_speed_flux_fix_effective, config.all_speed_mach_cutoff,
                              all_speed_f_min_effective, all_speed_ramp_weight_current);
      const float mass_flux_n = wall_flux.flux[0];
      final_max_wall_mass_flux = std::max(final_max_wall_mass_flux, std::abs(mass_flux_n));

      wall_flux_csv << face << "," << result.mesh.face_center[3 * face + 0] << ","
                    << result.mesh.face_center[3 * face + 1] << "," << mass_flux_n << "\n";
    }
    result.max_wall_mass_flux = final_max_wall_mass_flux;
  }

  {
    std::vector<unsigned char> wall_cell_mask(static_cast<std::size_t>(num_cells), 0u);
    for (const int face : wall_bc_faces) {
      const int owner = result.mesh.face_owner[face];
      if (owner >= 0 && owner < num_cells) {
        wall_cell_mask[static_cast<std::size_t>(owner)] = 1u;
      }
    }

    std::vector<int> wall_adjacent_interior_faces;
    wall_adjacent_interior_faces.reserve(static_cast<std::size_t>(result.mesh.num_faces / 8 + 1));
    for (int face = 0; face < result.mesh.num_faces; ++face) {
      const int owner = result.mesh.face_owner[face];
      const int neighbor = result.mesh.face_neighbor[face];
      if (neighbor < 0) {
        continue;
      }
      if (wall_cell_mask[static_cast<std::size_t>(owner)] == 0u &&
          wall_cell_mask[static_cast<std::size_t>(neighbor)] == 0u) {
        continue;
      }
      wall_adjacent_interior_faces.push_back(face);
    }

    std::vector<float> wall_adjacent_acoustic_scales;
    std::vector<float> wall_adjacent_stabilization_scales;
    std::vector<float> wall_adjacent_acoustic_scales_final;
    wall_adjacent_acoustic_scales.reserve(wall_adjacent_interior_faces.size());
    wall_adjacent_stabilization_scales.reserve(wall_adjacent_interior_faces.size());
    wall_adjacent_acoustic_scales_final.reserve(wall_adjacent_interior_faces.size());
    for (const int face : wall_adjacent_interior_faces) {
      const int owner = result.mesh.face_owner[face];
      const int neighbor = result.mesh.face_neighbor[face];
      const ConservativeState left = load_conservative_state(result.conserved, owner);
      const ConservativeState right = load_conservative_state(result.conserved, neighbor);
      const AcousticScalingInfo scaling = compute_acoustic_scaling_info(
        left, right, gamma, precondition, config.low_mach_fix, config.mach_cutoff,
        result.stabilization_mach_floor_final);
      wall_adjacent_acoustic_scales.push_back(scaling.acoustic_scale_base);
      wall_adjacent_stabilization_scales.push_back(scaling.stabilization_scale);
      wall_adjacent_acoustic_scales_final.push_back(scaling.acoustic_scale);
    }
    if (!wall_adjacent_acoustic_scales.empty()) {
      std::sort(wall_adjacent_acoustic_scales.begin(), wall_adjacent_acoustic_scales.end());
      std::sort(wall_adjacent_stabilization_scales.begin(), wall_adjacent_stabilization_scales.end());
      std::sort(wall_adjacent_acoustic_scales_final.begin(), wall_adjacent_acoustic_scales_final.end());
      result.wall_adjacent_acoustic_scale_min = wall_adjacent_acoustic_scales.front();
      result.wall_adjacent_acoustic_scale_p01 =
        linear_quantile_from_sorted(wall_adjacent_acoustic_scales, 0.01f);
      result.wall_adjacent_acoustic_scale_p50 =
        linear_quantile_from_sorted(wall_adjacent_acoustic_scales, 0.50f);
      result.wall_adjacent_stabilization_scale_min = wall_adjacent_stabilization_scales.front();
      result.wall_adjacent_stabilization_scale_p01 =
        linear_quantile_from_sorted(wall_adjacent_stabilization_scales, 0.01f);
      result.wall_adjacent_stabilization_scale_p50 =
        linear_quantile_from_sorted(wall_adjacent_stabilization_scales, 0.50f);
      result.wall_adjacent_acoustic_scale_final_min = wall_adjacent_acoustic_scales_final.front();
      result.wall_adjacent_acoustic_scale_final_p01 =
        linear_quantile_from_sorted(wall_adjacent_acoustic_scales_final, 0.01f);
      result.wall_adjacent_acoustic_scale_final_p50 =
        linear_quantile_from_sorted(wall_adjacent_acoustic_scales_final, 0.50f);
    } else {
      result.wall_adjacent_acoustic_scale_min = 1.0f;
      result.wall_adjacent_acoustic_scale_p01 = 1.0f;
      result.wall_adjacent_acoustic_scale_p50 = 1.0f;
      result.wall_adjacent_stabilization_scale_min = 1.0f;
      result.wall_adjacent_stabilization_scale_p01 = 1.0f;
      result.wall_adjacent_stabilization_scale_p50 = 1.0f;
      result.wall_adjacent_acoustic_scale_final_min = 1.0f;
      result.wall_adjacent_acoustic_scale_final_p01 = 1.0f;
      result.wall_adjacent_acoustic_scale_final_p50 = 1.0f;
    }

    std::ofstream dissipation_csv(result.dissipation_debug_csv_path, std::ios::trunc);
    if (!dissipation_csv) {
      throw std::runtime_error("Failed to write dissipation_debug.csv.");
    }
    dissipation_csv
      << "face,region,x,y,area,mach_local,un_abs,a,beta,low_mach_scale,acoustic_scale,"
         "s_stab,acoustic_scale_final,stabilization_m0,all_speed_pressure_scale,"
         "smax_physical,smax_dissipation\n";

    const auto write_face_row = [&](const int face, const char* region, const ConservativeState& left,
                                    const ConservativeState& right) {
      const std::array<float, 3> normal = read_face_unit_normal(result.mesh, face);
      const PrimitiveState pl = conservative_to_primitive(left, gamma);
      const PrimitiveState pr = conservative_to_primitive(right, gamma);
      const float un_l = pl.u * normal[0] + pl.v * normal[1] + pl.w * normal[2];
      const float un_r = pr.u * normal[0] + pr.v * normal[1] + pr.w * normal[2];
      const float a_l = speed_of_sound(pl, gamma);
      const float a_r = speed_of_sound(pr, gamma);
      const float vmag_l = std::sqrt(pl.u * pl.u + pl.v * pl.v + pl.w * pl.w);
      const float vmag_r = std::sqrt(pr.u * pr.u + pr.v * pr.v + pr.w * pr.w);
      const float mach_l = vmag_l / std::max(a_l, 1.0e-8f);
      const float mach_r = vmag_r / std::max(a_r, 1.0e-8f);
      const AcousticScalingInfo scaling = compute_acoustic_scaling_info(
        left, right, gamma, precondition, config.low_mach_fix, config.mach_cutoff,
        result.stabilization_mach_floor_final);
      const float all_speed_pressure_scale = compute_all_speed_hllc_pressure_scale(
        left, right, normal, gamma, all_speed_flux_fix_effective, config.all_speed_mach_cutoff,
        all_speed_f_min_effective, all_speed_ramp_weight_current);
      const float smax_physical = std::max(std::abs(un_l) + a_l, std::abs(un_r) + a_r);
      const float smax_dissipation =
        std::max(preconditioned_spectral_radius(un_l, a_l, scaling.acoustic_scale),
                 preconditioned_spectral_radius(un_r, a_r, scaling.acoustic_scale));

      dissipation_csv << face << "," << region << "," << result.mesh.face_center[3 * face + 0] << ","
                      << result.mesh.face_center[3 * face + 1] << "," << read_face_measure(result.mesh, face)
                      << "," << 0.5f * (mach_l + mach_r) << ","
                      << 0.5f * (std::abs(un_l) + std::abs(un_r)) << "," << 0.5f * (a_l + a_r) << ","
                      << scaling.beta << "," << scaling.low_mach_scale << ","
                      << scaling.acoustic_scale_base << "," << scaling.stabilization_scale << ","
                      << scaling.acoustic_scale << "," << scaling.stabilization_mach_floor << ","
                      << all_speed_pressure_scale << ","
                      << smax_physical << "," << smax_dissipation << "\n";
    };

    for (std::size_t i = 0; i < wall_adjacent_interior_faces.size() && i < 10; ++i) {
      const int face = wall_adjacent_interior_faces[i];
      const int owner = result.mesh.face_owner[face];
      const int neighbor = result.mesh.face_neighbor[face];
      const ConservativeState left = load_conservative_state(result.conserved, owner);
      const ConservativeState right = load_conservative_state(result.conserved, neighbor);
      write_face_row(face, "wall_adjacent_interior", left, right);
    }

    for (int i = 0; i < static_cast<int>(farfield_bc_faces.size()) && i < 10; ++i) {
      const int face = farfield_bc_faces[static_cast<std::size_t>(i)];
      const int owner = result.mesh.face_owner[face];
      const ConservativeState left = load_conservative_state(result.conserved, owner);
      write_face_row(face, "farfield_boundary", left, conservative_inf);
    }
  }

  result.wall_cp = extract_wall_cp(result.mesh, cell_pressure, reference, "wall");
  {
    std::ofstream cp_csv(result.cp_csv_path, std::ios::trunc);
    if (!cp_csv) {
      throw std::runtime_error("Failed to write cp_wall.csv.");
    }
    cp_csv << "s,x,y,Cp\n";
    for (const auto& sample : result.wall_cp) {
      cp_csv << sample.s << "," << sample.x << "," << sample.y << "," << sample.cp << "\n";
    }
  }

  const bool vtu_ok =
    write_euler_cell_vtu(result.vtu_path, result.mesh, result.rho, result.u, result.v, result.p,
                         result.mach, result.residual_magnitude);
  if (!vtu_ok) {
    throw std::runtime_error("Failed to write Euler VTU output.");
  }

  return result;
}

EulerRunResult run_euler_airfoil_case_cpu(const EulerAirfoilCaseConfig& config) {
  return run_euler_airfoil_case(config, "cpu");
}
}  // namespace cfd::core
