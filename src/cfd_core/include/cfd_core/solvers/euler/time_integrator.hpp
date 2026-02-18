#pragma once

#include "cfd_core/numerics/euler_flux.hpp"

#include <vector>

namespace cfd::core {
std::vector<float> compute_pseudo_time_step_over_volume(const std::vector<float>& spectral_radius,
                                                        float cfl, bool local_time_stepping);

void apply_rk3_stage_update(int stage, const std::vector<float>& state_n,
                            const std::vector<float>& state_stage,
                            const std::vector<float>& residual_preconditioned,
                            const std::vector<float>& delta_tau_over_volume, float gamma,
                            std::vector<float>* state_next);
}  // namespace cfd::core
