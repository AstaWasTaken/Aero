#pragma once

#include "cfd_core/mesh.hpp"
#include "cfd_core/numerics/euler_flux.hpp"

#include <vector>

namespace cfd::core {
struct LowMachPreconditionConfig {
  bool enabled = true;
  float mach_ref = 0.15f;
  float mach_min = 1.0e-3f;
  float beta_min = 0.2f;
  float beta_max = 1.0f;
};

float compute_local_mach(const PrimitiveState& primitive, float gamma);
float compute_cell_precondition_beta(const PrimitiveState& primitive, float gamma,
                                     const LowMachPreconditionConfig& config);
float compute_face_precondition_beta(const ConservativeState& left, const ConservativeState& right,
                                     float gamma, const LowMachPreconditionConfig& config);

void apply_low_mach_preconditioned_residual(const std::vector<PrimitiveState>& primitive,
                                            float gamma, const LowMachPreconditionConfig& config,
                                            std::vector<float>* residual);
}  // namespace cfd::core
