#include "cfd_core/solvers/euler/preconditioning.hpp"

#include <algorithm>
#include <cmath>

namespace cfd::core {
namespace {
constexpr float kMachFloor = 1.0e-8f;

float clamp_precondition_beta(const float beta, const LowMachPreconditionConfig& config) {
  const float beta_min = std::max(config.beta_min, 1.0e-4f);
  const float beta_max = std::max(beta_min, config.beta_max);
  return std::clamp(beta, beta_min, beta_max);
}
}  // namespace

float compute_local_mach(const PrimitiveState& primitive, const float gamma) {
  const float a = std::max(speed_of_sound(primitive, gamma), kMachFloor);
  const float velocity_mag = std::sqrt(primitive.u * primitive.u + primitive.v * primitive.v +
                                       primitive.w * primitive.w);
  return velocity_mag / a;
}

float compute_cell_precondition_beta(const PrimitiveState& primitive, const float gamma,
                                     const LowMachPreconditionConfig& config) {
  if (!config.enabled) {
    return 1.0f;
  }
  const float local_mach = compute_local_mach(primitive, gamma);
  const float target_mach = std::max({local_mach, config.mach_ref, config.mach_min, kMachFloor});
  return clamp_precondition_beta(target_mach, config);
}

float compute_face_precondition_beta(const ConservativeState& left, const ConservativeState& right,
                                     const float gamma, const LowMachPreconditionConfig& config) {
  if (!config.enabled) {
    return 1.0f;
  }
  const PrimitiveState pl = conservative_to_primitive(left, gamma);
  const PrimitiveState pr = conservative_to_primitive(right, gamma);
  const PrimitiveState avg = {
    0.5f * (pl.rho + pr.rho),
    0.5f * (pl.u + pr.u),
    0.5f * (pl.v + pr.v),
    0.5f * (pl.w + pr.w),
    0.5f * (pl.p + pr.p),
  };
  return compute_cell_precondition_beta(avg, gamma, config);
}

void apply_low_mach_preconditioned_residual(const std::vector<PrimitiveState>& primitive,
                                            const float gamma,
                                            const LowMachPreconditionConfig& config,
                                            std::vector<float>* residual) {
  if (residual == nullptr || !config.enabled) {
    return;
  }
  const auto num_cells = static_cast<int>(primitive.size());
  if (residual->size() < static_cast<std::size_t>(num_cells) * 5) {
    return;
  }
  for (int cell = 0; cell < num_cells; ++cell) {
    const PrimitiveState& p = primitive[static_cast<std::size_t>(cell)];
    const float beta = compute_cell_precondition_beta(p, gamma, config);
    const float beta2 = beta * beta;
    const float rho = std::max(p.rho, 1.0e-8f);
    const float inv_rho = 1.0f / rho;
    const std::size_t base = static_cast<std::size_t>(cell) * 5;
    // Simplified Turkel Γ⁻¹: R_E' = β²·R_E + (β²−1)·(u⃗·R_mom)/ρ
    const float u_dot_Rmom = p.u * (*residual)[base + 1] +
                             p.v * (*residual)[base + 2] +
                             p.w * (*residual)[base + 3];
    (*residual)[base + 4] = beta2 * (*residual)[base + 4] +
                            (beta2 - 1.0f) * inv_rho * u_dot_Rmom;
  }
}
}  // namespace cfd::core
