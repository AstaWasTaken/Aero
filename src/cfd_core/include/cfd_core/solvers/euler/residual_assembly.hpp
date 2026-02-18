#pragma once

#include "cfd_core/mesh.hpp"
#include "cfd_core/numerics/euler_flux.hpp"
#include "cfd_core/numerics/reconstruction.hpp"
#include "cfd_core/solvers/euler/mesh_geometry.hpp"
#include "cfd_core/solvers/euler/preconditioning.hpp"

#include <vector>

namespace cfd::core {
struct EulerResidualAssemblyConfig {
  const UnstructuredMesh* mesh = nullptr;
  const std::vector<PrimitiveState>* primitive = nullptr;
  const PrimitiveGradients* gradients = nullptr;
  const std::vector<EulerBoundaryConditionType>* boundary_type = nullptr;
  ConservativeState farfield_state;
  float gamma = 1.4f;
  EulerFluxScheme flux_scheme = EulerFluxScheme::kHllc;
  LimiterType limiter = LimiterType::kMinmod;
  bool use_second_order = true;
  bool low_mach_dissipation_fix = false;
  float low_mach_dissipation_cutoff = 0.3f;
  float stabilization_mach_floor = 0.0f;
  bool preconditioned_farfield_characteristic = false;
  LowMachPreconditionConfig precondition;
  bool all_speed_flux_fix = false;
  float all_speed_mach_cutoff = 0.25f;
  float all_speed_f_min = 0.05f;
  float all_speed_ramp_weight = 1.0f;
};

struct EulerResidualAssemblyDiagnostics {
  int num_faces_first_order_fallback = 0;
  int num_faces_diffusive_fallback = 0;
  int first_failure_face = -1;
  int first_failure_owner = -1;
  int first_failure_neighbor = -1;
  PrimitiveState first_failure_left{};
  PrimitiveState first_failure_right{};
};

void assemble_euler_residual_cpu(const EulerResidualAssemblyConfig& config,
                                 std::vector<float>* residual,
                                 std::vector<float>* spectral_radius,
                                 EulerResidualAssemblyDiagnostics* diagnostics = nullptr);
}  // namespace cfd::core
