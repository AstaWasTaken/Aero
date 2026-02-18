#include "cfd_core/numerics/euler_flux.hpp"

#include <algorithm>
#include <array>
#include <cmath>

namespace cfd::core {
namespace {
constexpr float kRhoFloor = 1.0e-8f;
constexpr float kPressureFloor = 1.0e-8f;
constexpr float kWaveSpeedFloor = 1.0e-8f;
constexpr float kLowMachMinScale = 1.0e-4f;
constexpr float kAllSpeedMachFloor = 1.0e-6f;

std::array<float, 5> conservative_to_array(const ConservativeState& state) {
  return {state.rho, state.rhou, state.rhov, state.rhow, state.rhoE};
}

bool is_finite_vector(const std::array<float, 5>& values) {
  for (const float value : values) {
    if (!std::isfinite(value)) {
      return false;
    }
  }
  return true;
}

std::array<float, 5> physical_flux(const ConservativeState& state, const PrimitiveState& primitive,
                                   const std::array<float, 3>& unit_normal) {
  const float un = primitive.u * unit_normal[0] + primitive.v * unit_normal[1] +
                   primitive.w * unit_normal[2];
  return {
    state.rho * un,
    state.rhou * un + unit_normal[0] * primitive.p,
    state.rhov * un + unit_normal[1] * primitive.p,
    state.rhow * un + unit_normal[2] * primitive.p,
    (state.rhoE + primitive.p) * un,
  };
}

float clamp_acoustic_scale(const float acoustic_scale) {
  return std::clamp(acoustic_scale, kLowMachMinScale, 1.0f);
}

float with_sign_floor(const float value) {
  if (std::abs(value) >= kWaveSpeedFloor) {
    return value;
  }
  return (value >= 0.0f) ? kWaveSpeedFloor : -kWaveSpeedFloor;
}

struct PreconditionedNormalSpeeds {
  float lambda_minus = 0.0f;
  float lambda_plus = 0.0f;
  float spectral_radius = 0.0f;
};

PreconditionedNormalSpeeds compute_preconditioned_normal_speeds(const float un, const float a,
                                                                const float acoustic_scale) {
  const float beta = clamp_acoustic_scale(acoustic_scale);
  const float beta2 = beta * beta;
  const float one_minus_beta2 = 1.0f - beta2;
  const float u_tilde = 0.5f * (1.0f + beta2) * un;
  const float discriminant = one_minus_beta2 * one_minus_beta2 * un * un +
                             4.0f * beta2 * a * a;
  const float c_tilde = 0.5f * std::sqrt(std::max(discriminant, 0.0f));

  PreconditionedNormalSpeeds speeds;
  speeds.lambda_minus = u_tilde - c_tilde;
  speeds.lambda_plus = u_tilde + c_tilde;
  speeds.spectral_radius = std::max(
    {std::abs(un), std::abs(speeds.lambda_minus), std::abs(speeds.lambda_plus), kWaveSpeedFloor});
  return speeds;
}

struct AllSpeedMachSensors {
  float mach = 1.0f;
  float mach_normal = 1.0f;
};

AllSpeedMachSensors compute_face_mach_sensors(const PrimitiveState& pl, const PrimitiveState& pr,
                                              const std::array<float, 3>& unit_normal,
                                              const float gamma) {
  const float un_l = pl.u * unit_normal[0] + pl.v * unit_normal[1] + pl.w * unit_normal[2];
  const float un_r = pr.u * unit_normal[0] + pr.v * unit_normal[1] + pr.w * unit_normal[2];
  const float a_l = std::max(speed_of_sound(pl, gamma), kWaveSpeedFloor);
  const float a_r = std::max(speed_of_sound(pr, gamma), kWaveSpeedFloor);
  const float vmag_l = std::sqrt(pl.u * pl.u + pl.v * pl.v + pl.w * pl.w);
  const float vmag_r = std::sqrt(pr.u * pr.u + pr.v * pr.v + pr.w * pr.w);

  AllSpeedMachSensors sensors;
  sensors.mach = std::max(0.5f * (vmag_l / a_l + vmag_r / a_r), kAllSpeedMachFloor);
  sensors.mach_normal =
    std::max(0.5f * (std::abs(un_l) / a_l + std::abs(un_r) / a_r), kAllSpeedMachFloor);
  sensors.mach_normal = std::min(sensors.mach_normal, sensors.mach);
  sensors.mach_normal = std::max(sensors.mach_normal, kAllSpeedMachFloor);
  return sensors;
}
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

void enforce_physical_state(ConservativeState* state, const float gamma) {
  if (state == nullptr) {
    return;
  }
  state->rho = std::max(state->rho, kRhoFloor);
  const float inv_rho = 1.0f / state->rho;
  const float u = state->rhou * inv_rho;
  const float v = state->rhov * inv_rho;
  const float w = state->rhow * inv_rho;
  const float kinetic = 0.5f * state->rho * (u * u + v * v + w * w);
  state->rhoE = std::max(state->rhoE, kinetic + kPressureFloor / (gamma - 1.0f));
  const float p = (gamma - 1.0f) * (state->rhoE - kinetic);
  if (p < kPressureFloor) {
    state->rhoE = kinetic + kPressureFloor / (gamma - 1.0f);
  }
}

float pressure_from_conservative(const ConservativeState& conservative, const float gamma) {
  return conservative_to_primitive(conservative, gamma).p;
}

float speed_of_sound(const PrimitiveState& primitive, const float gamma) {
  return std::sqrt(std::max(gamma * primitive.p / std::max(primitive.rho, kRhoFloor), 0.0f));
}

float compute_low_mach_dissipation_scale(const ConservativeState& left,
                                         const ConservativeState& right, const float gamma,
                                         const bool enable_low_mach_fix,
                                         const float mach_cutoff,
                                         const float mach_ref) {
  if (!enable_low_mach_fix || mach_cutoff <= 0.0f) {
    return 1.0f;
  }

  const PrimitiveState pl = conservative_to_primitive(left, gamma);
  const PrimitiveState pr = conservative_to_primitive(right, gamma);
  const float al = std::max(speed_of_sound(pl, gamma), kWaveSpeedFloor);
  const float ar = std::max(speed_of_sound(pr, gamma), kWaveSpeedFloor);
  const float vmag_l = std::sqrt(pl.u * pl.u + pl.v * pl.v + pl.w * pl.w);
  const float vmag_r = std::sqrt(pr.u * pr.u + pr.v * pr.v + pr.w * pr.w);
  const float mach_l = vmag_l / al;
  const float mach_r = vmag_r / ar;
  const float face_mach = 0.5f * (mach_l + mach_r);
  const float mach_star = std::max(face_mach, std::max(mach_ref, kWaveSpeedFloor));
  const float ratio = mach_star / std::max(mach_cutoff, kWaveSpeedFloor);
  return std::clamp(ratio, kLowMachMinScale, 1.0f);
}

float compute_all_speed_hllc_pressure_scale(const ConservativeState& left,
                                            const ConservativeState& right,
                                            const std::array<float, 3>& unit_normal,
                                            const float gamma,
                                            const bool enable_all_speed_fix,
                                            const float mach_cutoff,
                                            const float f_min,
                                            const float ramp_weight) {
  if (!enable_all_speed_fix) {
    return 1.0f;
  }

  const PrimitiveState pl = conservative_to_primitive(left, gamma);
  const PrimitiveState pr = conservative_to_primitive(right, gamma);
  const AllSpeedMachSensors sensors = compute_face_mach_sensors(pl, pr, unit_normal, gamma);
  const float cutoff = std::max(mach_cutoff, kAllSpeedMachFloor);
  const float theta_floor = std::clamp(f_min, kAllSpeedMachFloor, 1.0f);
  const float theta_n = std::clamp(sensors.mach_normal / cutoff, theta_floor, 1.0f);
  const float mach_ratio = std::clamp(sensors.mach / cutoff, 0.0f, 1.0f);
  const float mach_gate = mach_ratio * mach_ratio;
  const float theta_corr = (1.0f - mach_gate) * theta_n + mach_gate;
  const float w = std::clamp(ramp_weight, 0.0f, 1.0f);
  const float theta_eff = (1.0f - w) + w * theta_corr;
  return std::clamp(theta_eff, theta_floor, 1.0f);
}

std::array<float, 5> rusanov_flux(const ConservativeState& left, const ConservativeState& right,
                                  const std::array<float, 3>& unit_normal, const float gamma,
                                  float* max_wave_speed, const float acoustic_scale) {
  const PrimitiveState pl = conservative_to_primitive(left, gamma);
  const PrimitiveState pr = conservative_to_primitive(right, gamma);

  const float un_l = pl.u * unit_normal[0] + pl.v * unit_normal[1] + pl.w * unit_normal[2];
  const float un_r = pr.u * unit_normal[0] + pr.v * unit_normal[1] + pr.w * unit_normal[2];

  const float a_l = speed_of_sound(pl, gamma);
  const float a_r = speed_of_sound(pr, gamma);
  const float smax_physical = std::max(std::abs(un_l) + a_l, std::abs(un_r) + a_r);
  const PreconditionedNormalSpeeds speeds_l =
    compute_preconditioned_normal_speeds(un_l, a_l, acoustic_scale);
  const PreconditionedNormalSpeeds speeds_r =
    compute_preconditioned_normal_speeds(un_r, a_r, acoustic_scale);
  const float smax_dissipation = std::max(speeds_l.spectral_radius, speeds_r.spectral_radius);
  if (max_wave_speed != nullptr) {
    *max_wave_speed = std::max(smax_physical, kWaveSpeedFloor);
  }

  const std::array<float, 5> f_l = physical_flux(left, pl, unit_normal);
  const std::array<float, 5> f_r = physical_flux(right, pr, unit_normal);

  std::array<float, 5> flux = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
  const std::array<float, 5> du = {
    right.rho - left.rho,
    right.rhou - left.rhou,
    right.rhov - left.rhov,
    right.rhow - left.rhow,
    right.rhoE - left.rhoE,
  };
  for (int k = 0; k < 5; ++k) {
    flux[k] = 0.5f * (f_l[k] + f_r[k]) - 0.5f * smax_dissipation * du[k];
  }
  return flux;
}

std::array<float, 5> euler_flux_hllc(const ConservativeState& left,
                                     const ConservativeState& right,
                                     const std::array<float, 3>& unit_normal, const float area,
                                     const float gamma, float* max_wave_speed,
                                     const float acoustic_scale,
                                     const bool all_speed_flux_fix,
                                     const float all_speed_mach_cutoff,
                                     const float all_speed_f_min,
                                     const float all_speed_ramp_weight,
                                     FluxDebug* debug) {
  (void)area;
  if (debug != nullptr) {
    debug->used_diffusive_fallback = false;
  }

  const PrimitiveState pl = conservative_to_primitive(left, gamma);
  const PrimitiveState pr = conservative_to_primitive(right, gamma);
  const float un_l = pl.u * unit_normal[0] + pl.v * unit_normal[1] + pl.w * unit_normal[2];
  const float un_r = pr.u * unit_normal[0] + pr.v * unit_normal[1] + pr.w * unit_normal[2];
  const float a_l = speed_of_sound(pl, gamma);
  const float a_r = speed_of_sound(pr, gamma);
  const float pressure_scale = compute_all_speed_hllc_pressure_scale(
    left, right, unit_normal, gamma, all_speed_flux_fix, all_speed_mach_cutoff, all_speed_f_min,
    all_speed_ramp_weight);
  const float p_avg = 0.5f * (pl.p + pr.p);
  const float p_l_corr = p_avg + pressure_scale * (pl.p - p_avg);
  const float p_r_corr = p_avg + pressure_scale * (pr.p - p_avg);
  const float acoustic_scale_l = a_l;
  const float acoustic_scale_r = a_r;

  const PreconditionedNormalSpeeds speeds_l =
    compute_preconditioned_normal_speeds(un_l, acoustic_scale_l, acoustic_scale);
  const PreconditionedNormalSpeeds speeds_r =
    compute_preconditioned_normal_speeds(un_r, acoustic_scale_r, acoustic_scale);
  const float s_l = std::min(speeds_l.lambda_minus, speeds_r.lambda_minus);
  const float s_r = std::max(speeds_l.lambda_plus, speeds_r.lambda_plus);
  const float s_l_phys = std::min(un_l - a_l, un_r - a_r);
  const float s_r_phys = std::max(un_l + a_l, un_r + a_r);
  if (max_wave_speed != nullptr) {
    *max_wave_speed = std::max(std::max(std::abs(s_l_phys), std::abs(s_r_phys)), kWaveSpeedFloor);
  }

  if (std::abs(s_r - s_l) < kWaveSpeedFloor) {
    if (debug != nullptr) {
      debug->used_diffusive_fallback = true;
    }
    return rusanov_flux(left, right, unit_normal, gamma, max_wave_speed, acoustic_scale);
  }

  const std::array<float, 5> u_l = conservative_to_array(left);
  const std::array<float, 5> u_r = conservative_to_array(right);
  const std::array<float, 5> f_l = physical_flux(left, pl, unit_normal);
  const std::array<float, 5> f_r = physical_flux(right, pr, unit_normal);

  if (s_l >= 0.0f) {
    return f_l;
  }
  if (s_r <= 0.0f) {
    return f_r;
  }

  const float rho_l = std::max(pl.rho, kRhoFloor);
  const float rho_r = std::max(pr.rho, kRhoFloor);
  const float denom = rho_l * (s_l - un_l) - rho_r * (s_r - un_r);
  if (std::abs(denom) < kWaveSpeedFloor) {
    if (debug != nullptr) {
      debug->used_diffusive_fallback = true;
    }
    return rusanov_flux(left, right, unit_normal, gamma, max_wave_speed, acoustic_scale);
  }
  const float s_m =
    (p_r_corr - p_l_corr + rho_l * un_l * (s_l - un_l) - rho_r * un_r * (s_r - un_r)) / denom;

  const auto build_star_state = [&](const ConservativeState& state, const PrimitiveState& prim,
                                    const float pressure_corr, const float s_k,
                                    const float un_k) {
    std::array<float, 5> star = conservative_to_array(state);
    const float rho_k = std::max(prim.rho, kRhoFloor);
    const float sk_minus_un = s_k - un_k;
    const float sk_minus_sm = s_k - s_m;
    if (std::abs(sk_minus_un) < kWaveSpeedFloor || std::abs(sk_minus_sm) < kWaveSpeedFloor) {
      return star;
    }

    const float rho_star = rho_k * sk_minus_un / sk_minus_sm;
    const float velocity_shift = s_m - un_k;
    star[0] = rho_star;
    star[1] = rho_star * (prim.u + velocity_shift * unit_normal[0]);
    star[2] = rho_star * (prim.v + velocity_shift * unit_normal[1]);
    star[3] = rho_star * (prim.w + velocity_shift * unit_normal[2]);
    const float e_total = state.rhoE / rho_k;
    const float p_term = pressure_corr / with_sign_floor(rho_k * sk_minus_un);
    star[4] = rho_star * (e_total + velocity_shift * (s_m + p_term));
    return star;
  };

  const std::array<float, 5> u_star_l = build_star_state(left, pl, p_l_corr, s_l, un_l);
  const std::array<float, 5> u_star_r = build_star_state(right, pr, p_r_corr, s_r, un_r);
  if (!is_finite_vector(u_star_l) || !is_finite_vector(u_star_r) || u_star_l[0] <= 0.0f ||
      u_star_r[0] <= 0.0f) {
    if (debug != nullptr) {
      debug->used_diffusive_fallback = true;
    }
    return rusanov_flux(left, right, unit_normal, gamma, max_wave_speed, acoustic_scale);
  }

  std::array<float, 5> flux = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
  if (s_m >= 0.0f) {
    for (int k = 0; k < 5; ++k) {
      flux[k] = f_l[k] + s_l * (u_star_l[k] - u_l[k]);
    }
  } else {
    for (int k = 0; k < 5; ++k) {
      flux[k] = f_r[k] + s_r * (u_star_r[k] - u_r[k]);
    }
  }
  if (!is_finite_vector(flux)) {
    if (debug != nullptr) {
      debug->used_diffusive_fallback = true;
    }
    return rusanov_flux(left, right, unit_normal, gamma, max_wave_speed, acoustic_scale);
  }
  return flux;
}

std::array<float, 5> compute_euler_face_flux(const ConservativeState& left,
                                             const ConservativeState& right,
                                             const std::array<float, 3>& unit_normal,
                                             const float area, const float gamma,
                                             const EulerFluxScheme flux_scheme,
                                             float* max_wave_speed,
                                             const float acoustic_scale,
                                             const bool all_speed_flux_fix,
                                             const float all_speed_mach_cutoff,
                                             const float all_speed_f_min,
                                             const float all_speed_ramp_weight,
                                             FluxDebug* debug) {
  switch (flux_scheme) {
    case EulerFluxScheme::kHllc:
      return euler_flux_hllc(left, right, unit_normal, area, gamma, max_wave_speed,
                             acoustic_scale, all_speed_flux_fix, all_speed_mach_cutoff,
                             all_speed_f_min, all_speed_ramp_weight, debug);
    case EulerFluxScheme::kRusanov:
    default:
      if (debug != nullptr) {
        debug->used_diffusive_fallback = false;
      }
      return rusanov_flux(left, right, unit_normal, gamma, max_wave_speed, acoustic_scale);
  }
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
