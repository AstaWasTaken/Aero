#include "cfd_core/solvers/euler/residual_assembly.hpp"

#include "cfd_core/solvers/euler/boundary_conditions.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>

namespace cfd::core {
namespace {
constexpr float kStabilizationSmoothMaxEps = 1.0e-6f;
constexpr float kRhoFloor = 1.0e-8f;
constexpr float kPressureFloor = 1.0e-8f;

float smooth_max(const float a, const float b) {
  const float diff = a - b;
  return 0.5f * (a + b + std::sqrt(diff * diff +
                                   kStabilizationSmoothMaxEps * kStabilizationSmoothMaxEps));
}

float compute_face_mach(const ConservativeState& left, const ConservativeState& right,
                        const float gamma) {
  const PrimitiveState pl = conservative_to_primitive(left, gamma);
  const PrimitiveState pr = conservative_to_primitive(right, gamma);
  const float mach_l = compute_local_mach(pl, gamma);
  const float mach_r = compute_local_mach(pr, gamma);
  return 0.5f * (mach_l + mach_r);
}

float compute_face_acoustic_scale(const EulerResidualAssemblyConfig& config,
                                  const ConservativeState& left,
                                  const ConservativeState& right, const float face_beta) {
  float acoustic_scale = 1.0f;
  if (config.precondition.enabled) {
    acoustic_scale = std::clamp(face_beta, 1.0e-4f, 1.0f);
  }
  if (config.low_mach_dissipation_fix) {
    const float low_mach_scale = compute_low_mach_dissipation_scale(
      left, right, config.gamma, true, config.low_mach_dissipation_cutoff,
      config.precondition.mach_ref);
    acoustic_scale = std::min(acoustic_scale, low_mach_scale);
  }
  acoustic_scale = std::clamp(acoustic_scale, 1.0e-4f, 1.0f);

  const float face_mach = compute_face_mach(left, right, config.gamma);
  const float stabilization_mach_floor = std::clamp(config.stabilization_mach_floor, 0.0f, 1.0f);
  const float meff = smooth_max(face_mach, stabilization_mach_floor);
  const float stabilization_scale = std::clamp(meff, 1.0e-4f, 1.0f);
  return std::max(acoustic_scale, stabilization_scale);
}

bool is_valid_primitive_state(const PrimitiveState& primitive) {
  return std::isfinite(primitive.rho) && std::isfinite(primitive.u) && std::isfinite(primitive.v) &&
         std::isfinite(primitive.w) && std::isfinite(primitive.p) && primitive.rho > kRhoFloor &&
         primitive.p > kPressureFloor;
}

void record_first_failure(EulerResidualAssemblyDiagnostics* diagnostics, const int face, const int owner,
                          const int neighbor, const PrimitiveState& left,
                          const PrimitiveState& right) {
  if (diagnostics == nullptr || diagnostics->first_failure_face >= 0) {
    return;
  }
  diagnostics->first_failure_face = face;
  diagnostics->first_failure_owner = owner;
  diagnostics->first_failure_neighbor = neighbor;
  diagnostics->first_failure_left = left;
  diagnostics->first_failure_right = right;
}
}  // namespace

void assemble_euler_residual_cpu(const EulerResidualAssemblyConfig& config,
                                 std::vector<float>* residual,
                                 std::vector<float>* spectral_radius,
                                 EulerResidualAssemblyDiagnostics* diagnostics) {
  if (config.mesh == nullptr || config.primitive == nullptr || config.gradients == nullptr ||
      config.boundary_type == nullptr || residual == nullptr || spectral_radius == nullptr) {
    throw std::invalid_argument("Euler residual assembly received null input.");
  }

  const UnstructuredMesh& mesh = *config.mesh;
  const std::vector<PrimitiveState>& primitive = *config.primitive;
  const PrimitiveGradients& gradients = *config.gradients;
  const std::vector<EulerBoundaryConditionType>& boundary_type = *config.boundary_type;
  if (primitive.size() != static_cast<std::size_t>(mesh.num_cells)) {
    throw std::invalid_argument("Primitive size must match mesh.num_cells.");
  }
  if (residual->size() != static_cast<std::size_t>(mesh.num_cells) * 5 ||
      spectral_radius->size() != static_cast<std::size_t>(mesh.num_cells)) {
    throw std::invalid_argument("Residual arrays do not match mesh dimensions.");
  }
  if (diagnostics != nullptr) {
    *diagnostics = EulerResidualAssemblyDiagnostics{};
  }

  for (int face = 0; face < mesh.num_faces; ++face) {
    const int owner = mesh.face_owner[face];
    const int neighbor = mesh.face_neighbor[face];
    const std::array<float, 3> normal = read_face_unit_normal(mesh, face);
    const float area = read_face_measure(mesh, face);

    std::array<float, 5> flux = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    float max_wave_speed = 0.0f;
    float face_beta = 1.0f;

    if (neighbor >= 0) {
      PrimitiveState left_primitive = primitive[static_cast<std::size_t>(owner)];
      PrimitiveState right_primitive = primitive[static_cast<std::size_t>(neighbor)];
      if (config.use_second_order) {
        reconstruct_interior_face_states(mesh, primitive, gradients, face, config.limiter,
                                         &left_primitive, &right_primitive);
        if (!is_valid_primitive_state(left_primitive) || !is_valid_primitive_state(right_primitive)) {
          record_first_failure(diagnostics, face, owner, neighbor, left_primitive, right_primitive);
          left_primitive = primitive[static_cast<std::size_t>(owner)];
          right_primitive = primitive[static_cast<std::size_t>(neighbor)];
          if (diagnostics != nullptr) {
            ++diagnostics->num_faces_first_order_fallback;
          }
        }
      }

      const ConservativeState left_state = primitive_to_conservative(left_primitive, config.gamma);
      const ConservativeState right_state = primitive_to_conservative(right_primitive, config.gamma);
      face_beta =
        compute_face_precondition_beta(left_state, right_state, config.gamma, config.precondition);
      const float acoustic_scale =
        compute_face_acoustic_scale(config, left_state, right_state, face_beta);
      FluxDebug flux_debug;
      flux = compute_euler_face_flux(left_state, right_state, normal, area, config.gamma,
                                     config.flux_scheme, &max_wave_speed, acoustic_scale,
                                     config.all_speed_flux_fix, config.all_speed_mach_cutoff,
                                     config.all_speed_f_min, config.all_speed_ramp_weight,
                                     &flux_debug);
      if (diagnostics != nullptr && flux_debug.used_diffusive_fallback) {
        ++diagnostics->num_faces_diffusive_fallback;
      }
    } else {
      const ConservativeState interior =
        primitive_to_conservative(primitive[static_cast<std::size_t>(owner)], config.gamma);
      const EulerBoundaryConditionType bc = boundary_type[static_cast<std::size_t>(face)];
      ConservativeState exterior = config.farfield_state;
      if (bc == EulerBoundaryConditionType::kSlipWall) {
        face_beta =
          compute_face_precondition_beta(interior, interior, config.gamma, config.precondition);
        exterior = interior;
      } else {
        face_beta = compute_face_precondition_beta(interior, config.farfield_state, config.gamma,
                                                   config.precondition);
      }
      const float acoustic_scale =
        compute_face_acoustic_scale(config, interior, exterior, face_beta);

      FluxDebug flux_debug;
      const BoundaryFluxResult boundary_flux =
        compute_boundary_flux(bc, interior, config.farfield_state, normal, config.flux_scheme,
                              config.gamma, acoustic_scale,
                              config.preconditioned_farfield_characteristic,
                              config.all_speed_flux_fix, config.all_speed_mach_cutoff,
                              config.all_speed_f_min, config.all_speed_ramp_weight, &flux_debug);
      flux = boundary_flux.flux;
      max_wave_speed = boundary_flux.max_wave_speed;
      if (diagnostics != nullptr && flux_debug.used_diffusive_fallback) {
        ++diagnostics->num_faces_diffusive_fallback;
      }
    }

    const float wave = max_wave_speed * area;
    (*spectral_radius)[static_cast<std::size_t>(owner)] += wave;

    for (int k = 0; k < 5; ++k) {
      const float flux_area = flux[k] * area;
      (*residual)[static_cast<std::size_t>(5 * owner + k)] += flux_area;
      if (neighbor >= 0) {
        (*residual)[static_cast<std::size_t>(5 * neighbor + k)] -= flux_area;
      }
    }
    if (neighbor >= 0) {
      (*spectral_radius)[static_cast<std::size_t>(neighbor)] += wave;
    }
  }
}
}  // namespace cfd::core
