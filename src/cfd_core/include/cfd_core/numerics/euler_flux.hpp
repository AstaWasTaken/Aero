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

ConservativeState primitive_to_conservative(const PrimitiveState& primitive, float gamma);
PrimitiveState conservative_to_primitive(const ConservativeState& conservative, float gamma);
float pressure_from_conservative(const ConservativeState& conservative, float gamma);
float speed_of_sound(const PrimitiveState& primitive, float gamma);

std::array<float, 5> rusanov_flux(const ConservativeState& left, const ConservativeState& right,
                                  const std::array<float, 3>& unit_normal, float gamma,
                                  float* max_wave_speed);
ConservativeState reflect_slip_wall(const ConservativeState& interior,
                                    const std::array<float, 3>& unit_normal, float gamma);
}  // namespace cfd::core
