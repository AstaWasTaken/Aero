#include "cfd_core/solvers/euler/boundary_conditions.hpp"

#include <algorithm>
#include <cmath>

namespace cfd::core {
namespace {
constexpr float kStateFloor = 1.0e-8f;

float clamp_acoustic_scale(const float acoustic_scale) {
  return std::clamp(acoustic_scale, 1.0e-4f, 1.0f);
}

float preconditioned_spectral_radius(const float un, const float a, const float acoustic_scale) {
  const float beta = clamp_acoustic_scale(acoustic_scale);
  const float beta2 = beta * beta;
  const float one_minus_beta2 = 1.0f - beta2;
  const float u_tilde = 0.5f * (1.0f + beta2) * un;
  const float discriminant = one_minus_beta2 * one_minus_beta2 * un * un +
                             4.0f * beta2 * a * a;
  const float c_tilde = 0.5f * std::sqrt(std::max(discriminant, 0.0f));
  const float lambda_minus = u_tilde - c_tilde;
  const float lambda_plus = u_tilde + c_tilde;
  return std::max({std::abs(un), std::abs(lambda_minus), std::abs(lambda_plus), 1.0e-8f});
}

ConservativeState make_preconditioned_farfield_state(const ConservativeState& interior,
                                                     const ConservativeState& farfield_state,
                                                     const std::array<float, 3>& unit_normal,
                                                     const float gamma,
                                                     const float acoustic_scale) {
  const PrimitiveState pi = conservative_to_primitive(interior, gamma);
  const PrimitiveState pf = conservative_to_primitive(farfield_state, gamma);
  const float un_i =
    pi.u * unit_normal[0] + pi.v * unit_normal[1] + pi.w * unit_normal[2];
  const float un_f =
    pf.u * unit_normal[0] + pf.v * unit_normal[1] + pf.w * unit_normal[2];
  const float a_i = std::max(speed_of_sound(pi, gamma), kStateFloor);
  const float a_f = std::max(speed_of_sound(pf, gamma), kStateFloor);
  const float rho_ref = std::max(0.5f * (pi.rho + pf.rho), kStateFloor);
  const float a_ref = std::max(0.5f * (a_i + a_f), kStateFloor);
  const float a_tilde = std::max(clamp_acoustic_scale(acoustic_scale) * a_ref, 1.0e-6f);

  if (std::abs(un_i) >= a_tilde) {
    return (un_i >= 0.0f) ? interior : farfield_state;
  }

  const float r_plus_interior = un_i + pi.p / (rho_ref * a_tilde);
  const float r_minus_farfield = un_f - pf.p / (rho_ref * a_tilde);
  const float un_b = 0.5f * (r_plus_interior + r_minus_farfield);
  const float p_b =
    std::max(0.5f * rho_ref * a_tilde * (r_plus_interior - r_minus_farfield), kStateFloor);

  const std::array<float, 3> tangential_i = {
    pi.u - un_i * unit_normal[0],
    pi.v - un_i * unit_normal[1],
    pi.w - un_i * unit_normal[2],
  };
  const std::array<float, 3> tangential_f = {
    pf.u - un_f * unit_normal[0],
    pf.v - un_f * unit_normal[1],
    pf.w - un_f * unit_normal[2],
  };
  const bool inflow = un_b < 0.0f;
  const std::array<float, 3>& tangential = inflow ? tangential_f : tangential_i;
  const float rho_base = inflow ? pf.rho : pi.rho;
  const float p_base = inflow ? pf.p : pi.p;
  const float a_base = inflow ? a_f : a_i;

  PrimitiveState boundary;
  boundary.rho = std::max(
    rho_base + (p_b - p_base) / std::max(a_base * a_base, kStateFloor), kStateFloor);
  boundary.u = un_b * unit_normal[0] + tangential[0];
  boundary.v = un_b * unit_normal[1] + tangential[1];
  boundary.w = un_b * unit_normal[2] + tangential[2];
  boundary.p = p_b;
  return primitive_to_conservative(boundary, gamma);
}
}  // namespace

BoundaryFluxResult compute_boundary_flux(const EulerBoundaryConditionType boundary_type,
                                         const ConservativeState& interior,
                                         const ConservativeState& farfield_state,
                                         const std::array<float, 3>& unit_normal,
                                         const EulerFluxScheme flux_scheme, const float gamma,
                                         const float acoustic_scale,
                                         const bool preconditioned_farfield_characteristic,
                                         const bool all_speed_flux_fix,
                                         const float all_speed_mach_cutoff,
                                         const float all_speed_f_min,
                                         const float all_speed_ramp_weight,
                                         FluxDebug* debug) {
  BoundaryFluxResult result;
  if (debug != nullptr) {
    debug->used_diffusive_fallback = false;
  }
  if (boundary_type == EulerBoundaryConditionType::kSlipWall) {
    const PrimitiveState primitive = conservative_to_primitive(interior, gamma);
    const float un = primitive.u * unit_normal[0] + primitive.v * unit_normal[1] +
                     primitive.w * unit_normal[2];
    const float a = speed_of_sound(primitive, gamma);
    result.flux = {
      0.0f,
      primitive.p * unit_normal[0],
      primitive.p * unit_normal[1],
      primitive.p * unit_normal[2],
      0.0f,
    };
    result.max_wave_speed = preconditioned_spectral_radius(un, a, acoustic_scale);
    return result;
  }

  const ConservativeState boundary_state =
    preconditioned_farfield_characteristic
      ? make_preconditioned_farfield_state(
          interior, farfield_state, unit_normal, gamma, acoustic_scale)
      : farfield_state;
  result.flux = compute_euler_face_flux(interior, boundary_state, unit_normal, 1.0f, gamma,
                                        flux_scheme, &result.max_wave_speed, acoustic_scale,
                                        all_speed_flux_fix, all_speed_mach_cutoff, all_speed_f_min,
                                        all_speed_ramp_weight, debug);
  return result;
}
}  // namespace cfd::core
