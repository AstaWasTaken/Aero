#pragma once

#include "cfd_core/numerics/euler_flux.hpp"
#include "cfd_core/solvers/euler/mesh_geometry.hpp"

#include <array>

namespace cfd::core {
struct BoundaryFluxResult {
  std::array<float, 5> flux = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
  float max_wave_speed = 0.0f;
};

BoundaryFluxResult compute_boundary_flux(EulerBoundaryConditionType boundary_type,
                                         const ConservativeState& interior,
                                         const ConservativeState& farfield_state,
                                         const std::array<float, 3>& unit_normal,
                                         EulerFluxScheme flux_scheme, float gamma,
                                         float acoustic_scale,
                                         bool preconditioned_farfield_characteristic = false,
                                         bool all_speed_flux_fix = false,
                                         float all_speed_mach_cutoff = 0.25f,
                                         float all_speed_f_min = 0.05f,
                                         float all_speed_ramp_weight = 1.0f,
                                         FluxDebug* debug = nullptr);
}  // namespace cfd::core
