#include "cfd_core/numerics/euler_flux.hpp"

#include <algorithm>
#include <array>
#include <cmath>

namespace cfd::core {
namespace {
constexpr float kRhoFloor = 1.0e-8f;
constexpr float kPressureFloor = 1.0e-8f;
}  // namespace

ConservativeState primitive_to_conservative(const PrimitiveState& primitive, const float gamma) {
  ConservativeState conservative;
  conservative.rho = std::max(primitive.rho, kRhoFloor);
  conservative.rhou = conservative.rho * primitive.u;
  conservative.rhov = conservative.rho * primitive.v;
  conservative.rhow = conservative.rho * primitive.w;
  const float kinetic = 0.5f * conservative.rho *
                        (primitive.u * primitive.u + primitive.v * primitive.v +
                         primitive.w * primitive.w);
  const float pressure = std::max(primitive.p, kPressureFloor);
  conservative.rhoE = pressure / (gamma - 1.0f) + kinetic;
  return conservative;
}

PrimitiveState conservative_to_primitive(const ConservativeState& conservative, const float gamma) {
  PrimitiveState primitive;
  primitive.rho = std::max(conservative.rho, kRhoFloor);
  const float inv_rho = 1.0f / primitive.rho;
  primitive.u = conservative.rhou * inv_rho;
  primitive.v = conservative.rhov * inv_rho;
  primitive.w = conservative.rhow * inv_rho;
  const float kinetic =
    0.5f * primitive.rho *
    (primitive.u * primitive.u + primitive.v * primitive.v + primitive.w * primitive.w);
  primitive.p = std::max((gamma - 1.0f) * (conservative.rhoE - kinetic), kPressureFloor);
  return primitive;
}

float pressure_from_conservative(const ConservativeState& conservative, const float gamma) {
  return conservative_to_primitive(conservative, gamma).p;
}

float speed_of_sound(const PrimitiveState& primitive, const float gamma) {
  return std::sqrt(std::max(gamma * primitive.p / std::max(primitive.rho, kRhoFloor), 0.0f));
}

std::array<float, 5> rusanov_flux(const ConservativeState& left, const ConservativeState& right,
                                  const std::array<float, 3>& unit_normal, const float gamma,
                                  float* max_wave_speed) {
  const PrimitiveState pl = conservative_to_primitive(left, gamma);
  const PrimitiveState pr = conservative_to_primitive(right, gamma);

  const float un_l = pl.u * unit_normal[0] + pl.v * unit_normal[1] + pl.w * unit_normal[2];
  const float un_r = pr.u * unit_normal[0] + pr.v * unit_normal[1] + pr.w * unit_normal[2];

  const float a_l = speed_of_sound(pl, gamma);
  const float a_r = speed_of_sound(pr, gamma);
  const float smax = std::max(std::abs(un_l) + a_l, std::abs(un_r) + a_r);
  if (max_wave_speed != nullptr) {
    *max_wave_speed = smax;
  }

  std::array<float, 5> f_l = {
    left.rho * un_l,
    left.rhou * un_l + unit_normal[0] * pl.p,
    left.rhov * un_l + unit_normal[1] * pl.p,
    left.rhow * un_l + unit_normal[2] * pl.p,
    (left.rhoE + pl.p) * un_l,
  };
  std::array<float, 5> f_r = {
    right.rho * un_r,
    right.rhou * un_r + unit_normal[0] * pr.p,
    right.rhov * un_r + unit_normal[1] * pr.p,
    right.rhow * un_r + unit_normal[2] * pr.p,
    (right.rhoE + pr.p) * un_r,
  };

  std::array<float, 5> flux = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
  const std::array<float, 5> du = {
    right.rho - left.rho,
    right.rhou - left.rhou,
    right.rhov - left.rhov,
    right.rhow - left.rhow,
    right.rhoE - left.rhoE,
  };
  for (int k = 0; k < 5; ++k) {
    flux[k] = 0.5f * (f_l[k] + f_r[k]) - 0.5f * smax * du[k];
  }
  return flux;
}

ConservativeState reflect_slip_wall(const ConservativeState& interior,
                                    const std::array<float, 3>& unit_normal, const float gamma) {
  const PrimitiveState pi = conservative_to_primitive(interior, gamma);
  const float un = pi.u * unit_normal[0] + pi.v * unit_normal[1] + pi.w * unit_normal[2];
  PrimitiveState ghost = pi;
  ghost.u = pi.u - 2.0f * un * unit_normal[0];
  ghost.v = pi.v - 2.0f * un * unit_normal[1];
  ghost.w = pi.w - 2.0f * un * unit_normal[2];
  return primitive_to_conservative(ghost, gamma);
}
}  // namespace cfd::core
