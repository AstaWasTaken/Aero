#include "cfd_core/solvers/euler/time_integrator.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace cfd::core {
namespace {
ConservativeState load_cell_state(const std::vector<float>& conserved, const int cell) {
  return {
    conserved[5 * cell + 0],
    conserved[5 * cell + 1],
    conserved[5 * cell + 2],
    conserved[5 * cell + 3],
    conserved[5 * cell + 4],
  };
}

void store_cell_state(std::vector<float>* conserved, const int cell, const ConservativeState& state) {
  if (conserved == nullptr) {
    return;
  }
  (*conserved)[5 * cell + 0] = state.rho;
  (*conserved)[5 * cell + 1] = state.rhou;
  (*conserved)[5 * cell + 2] = state.rhov;
  (*conserved)[5 * cell + 3] = state.rhow;
  (*conserved)[5 * cell + 4] = state.rhoE;
}
}  // namespace

std::vector<float> compute_pseudo_time_step_over_volume(const std::vector<float>& spectral_radius,
                                                        const float cfl,
                                                        const bool local_time_stepping) {
  constexpr float kWaveFloor = 1.0e-8f;
  const std::size_t num_cells = spectral_radius.size();
  std::vector<float> dt_over_volume(num_cells, 0.0f);
  if (num_cells == 0) {
    return dt_over_volume;
  }

  if (local_time_stepping) {
    for (std::size_t cell = 0; cell < num_cells; ++cell) {
      const float denom = std::max(spectral_radius[cell], kWaveFloor);
      dt_over_volume[cell] = cfl / denom;
    }
    return dt_over_volume;
  }

  float global_dt_over_volume = std::numeric_limits<float>::max();
  for (const float wave : spectral_radius) {
    const float denom = std::max(wave, kWaveFloor);
    global_dt_over_volume = std::min(global_dt_over_volume, cfl / denom);
  }
  if (!std::isfinite(global_dt_over_volume)) {
    global_dt_over_volume = 0.0f;
  }
  std::fill(dt_over_volume.begin(), dt_over_volume.end(), global_dt_over_volume);
  return dt_over_volume;
}

void apply_rk3_stage_update(const int stage, const std::vector<float>& state_n,
                            const std::vector<float>& state_stage,
                            const std::vector<float>& residual_preconditioned,
                            const std::vector<float>& delta_tau_over_volume, const float gamma,
                            std::vector<float>* state_next) {
  if (state_next == nullptr) {
    return;
  }
  const std::size_t num_cells = delta_tau_over_volume.size();
  state_next->assign(num_cells * 5, 0.0f);
  if (state_n.size() != num_cells * 5 || state_stage.size() != num_cells * 5 ||
      residual_preconditioned.size() != num_cells * 5) {
    return;
  }

  for (std::size_t cell = 0; cell < num_cells; ++cell) {
    ConservativeState state = load_cell_state(state_stage, static_cast<int>(cell));
    const float dt_over_volume = delta_tau_over_volume[cell];
    state.rho -= dt_over_volume * residual_preconditioned[5 * cell + 0];
    state.rhou -= dt_over_volume * residual_preconditioned[5 * cell + 1];
    state.rhov -= dt_over_volume * residual_preconditioned[5 * cell + 2];
    state.rhow -= dt_over_volume * residual_preconditioned[5 * cell + 3];
    state.rhoE -= dt_over_volume * residual_preconditioned[5 * cell + 4];

    if (stage == 1) {
      const ConservativeState base = load_cell_state(state_n, static_cast<int>(cell));
      state.rho = 0.75f * base.rho + 0.25f * state.rho;
      state.rhou = 0.75f * base.rhou + 0.25f * state.rhou;
      state.rhov = 0.75f * base.rhov + 0.25f * state.rhov;
      state.rhow = 0.75f * base.rhow + 0.25f * state.rhow;
      state.rhoE = 0.75f * base.rhoE + 0.25f * state.rhoE;
    } else if (stage >= 2) {
      const ConservativeState base = load_cell_state(state_n, static_cast<int>(cell));
      state.rho = (1.0f / 3.0f) * base.rho + (2.0f / 3.0f) * state.rho;
      state.rhou = (1.0f / 3.0f) * base.rhou + (2.0f / 3.0f) * state.rhou;
      state.rhov = (1.0f / 3.0f) * base.rhov + (2.0f / 3.0f) * state.rhov;
      state.rhow = (1.0f / 3.0f) * base.rhow + (2.0f / 3.0f) * state.rhow;
      state.rhoE = (1.0f / 3.0f) * base.rhoE + (2.0f / 3.0f) * state.rhoE;
    }

    enforce_physical_state(&state, gamma);
    store_cell_state(state_next, static_cast<int>(cell), state);
  }
}
}  // namespace cfd::core
