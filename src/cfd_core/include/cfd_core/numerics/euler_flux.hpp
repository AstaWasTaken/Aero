#pragma once

#include <array>

namespace cfd::core {
struct PrimitiveState {
  float rho = 1.0f;
  float u = 0.0f;
  float v = 0.0f;
  float w = 0.0f;
  float p = 1.0f;
};

struct ConservativeState {
  float rho = 1.0f;
  float rhou = 0.0f;
  float rhov = 0.0f;
  float rhow = 0.0f;
  float rhoE = 1.0f;
};

enum class EulerFluxScheme {
  kRusanov = 0,
  kHllc = 1,
};

struct FluxDebug {
  bool used_diffusive_fallback = false;
};

ConservativeState primitive_to_conservative(const PrimitiveState& primitive, float gamma);
PrimitiveState conservative_to_primitive(const ConservativeState& conservative, float gamma);
void enforce_physical_state(ConservativeState* state, float gamma);
float pressure_from_conservative(const ConservativeState& conservative, float gamma);
float speed_of_sound(const PrimitiveState& primitive, float gamma);
float compute_low_mach_dissipation_scale(const ConservativeState& left,
                                         const ConservativeState& right, float gamma,
                                         bool enable_low_mach_fix, float mach_cutoff,
                                         float mach_ref);
float compute_all_speed_hllc_pressure_scale(const ConservativeState& left,
                                            const ConservativeState& right,
                                            const std::array<float, 3>& unit_normal,
                                            float gamma,
                                            bool enable_all_speed_fix,
                                            float mach_cutoff,
                                            float f_min = 0.05f,
                                            float ramp_weight = 1.0f);

std::array<float, 5> rusanov_flux(const ConservativeState& left, const ConservativeState& right,
                                  const std::array<float, 3>& unit_normal, float gamma,
                                  float* max_wave_speed, float acoustic_scale = 1.0f);
std::array<float, 5> euler_flux_hllc(const ConservativeState& left,
                                     const ConservativeState& right,
                                     const std::array<float, 3>& unit_normal, float area,
                                     float gamma, float* max_wave_speed,
                                     float acoustic_scale = 1.0f,
                                     bool all_speed_flux_fix = false,
                                     float all_speed_mach_cutoff = 0.25f,
                                     float all_speed_f_min = 0.05f,
                                     float all_speed_ramp_weight = 1.0f,
                                     FluxDebug* debug = nullptr);
std::array<float, 5> compute_euler_face_flux(const ConservativeState& left,
                                             const ConservativeState& right,
                                             const std::array<float, 3>& unit_normal,
                                             float area, float gamma,
                                             EulerFluxScheme flux_scheme,
                                             float* max_wave_speed,
                                             float acoustic_scale = 1.0f,
                                             bool all_speed_flux_fix = false,
                                             float all_speed_mach_cutoff = 0.25f,
                                             float all_speed_f_min = 0.05f,
                                             float all_speed_ramp_weight = 1.0f,
                                             FluxDebug* debug = nullptr);
ConservativeState reflect_slip_wall(const ConservativeState& interior,
                                    const std::array<float, 3>& unit_normal, float gamma);
}  // namespace cfd::core
