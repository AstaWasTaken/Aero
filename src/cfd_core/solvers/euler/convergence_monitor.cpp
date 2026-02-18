#include "cfd_core/solvers/euler/convergence_monitor.hpp"

#include <algorithm>
#include <cmath>

namespace cfd::core {
ConvergenceMonitor::ConvergenceMonitor(const ConvergenceMonitorConfig config) : config_(config) {
  if (config_.force_mean_drift_tol <= 0.0f) {
    config_.force_mean_drift_tol = config_.force_stability_tol;
  }
}

float ConvergenceMonitor::update_residual(const float residual_l2) {
  const float bounded_residual = std::max(residual_l2, 1.0e-20f);
  if (initial_residual_l2_ < 0.0f) {
    initial_residual_l2_ = bounded_residual;
  }
  return bounded_residual / std::max(initial_residual_l2_, 1.0e-20f);
}

ForceDrift ConvergenceMonitor::update_forces(const ForceCoefficients& forces) {
  ForceDrift drift;
  if (has_previous_force_) {
    drift.dcl = std::abs(forces.cl - previous_force_.cl);
    drift.dcd = std::abs(forces.cd - previous_force_.cd);
    drift.dcm = std::abs(forces.cm - previous_force_.cm);
  }
  previous_force_ = forces;
  has_previous_force_ = true;

  const int required = std::max(config_.force_stability_window, 1);
  force_window_.push_back(forces);
  while (static_cast<int>(force_window_.size()) > 2 * required) {
    force_window_.pop_front();
  }
  return drift;
}

bool ConvergenceMonitor::forces_stable() const {
  const int required = std::max(config_.force_stability_window, 1);
  if (static_cast<int>(force_window_.size()) < required) {
    return false;
  }

  const int begin = static_cast<int>(force_window_.size()) - required;
  const ForceCoefficients& first = force_window_[static_cast<std::size_t>(begin)];
  float cl_min = first.cl;
  float cl_max = first.cl;
  float cd_min = first.cd;
  float cd_max = first.cd;
  float cm_min = first.cm;
  float cm_max = first.cm;

  for (int i = begin; i < static_cast<int>(force_window_.size()); ++i) {
    const ForceCoefficients& sample = force_window_[static_cast<std::size_t>(i)];
    cl_min = std::min(cl_min, sample.cl);
    cl_max = std::max(cl_max, sample.cl);
    cd_min = std::min(cd_min, sample.cd);
    cd_max = std::max(cd_max, sample.cd);
    cm_min = std::min(cm_min, sample.cm);
    cm_max = std::max(cm_max, sample.cm);
  }

  return (cl_max - cl_min) <= config_.force_stability_tol &&
         (cd_max - cd_min) <= config_.force_stability_tol &&
         (cm_max - cm_min) <= config_.force_stability_tol;
}

ForceMeanDrift ConvergenceMonitor::force_mean_drift() const {
  ForceMeanDrift drift;
  const int required = std::max(config_.force_stability_window, 1);
  const int needed = 2 * required;
  if (static_cast<int>(force_window_.size()) < needed) {
    return drift;
  }

  const int prev_begin = static_cast<int>(force_window_.size()) - needed;
  const int curr_begin = static_cast<int>(force_window_.size()) - required;
  float cl_prev = 0.0f;
  float cd_prev = 0.0f;
  float cm_prev = 0.0f;
  float cl_curr = 0.0f;
  float cd_curr = 0.0f;
  float cm_curr = 0.0f;
  for (int i = prev_begin; i < curr_begin; ++i) {
    const ForceCoefficients& sample = force_window_[static_cast<std::size_t>(i)];
    cl_prev += sample.cl;
    cd_prev += sample.cd;
    cm_prev += sample.cm;
  }
  for (int i = curr_begin; i < static_cast<int>(force_window_.size()); ++i) {
    const ForceCoefficients& sample = force_window_[static_cast<std::size_t>(i)];
    cl_curr += sample.cl;
    cd_curr += sample.cd;
    cm_curr += sample.cm;
  }
  const float inv = 1.0f / static_cast<float>(required);
  drift.dcl = std::abs(cl_curr * inv - cl_prev * inv);
  drift.dcd = std::abs(cd_curr * inv - cd_prev * inv);
  drift.dcm = std::abs(cm_curr * inv - cm_prev * inv);
  drift.valid = true;
  return drift;
}

bool ConvergenceMonitor::forces_mean_drift_stable() const {
  const ForceMeanDrift drift = force_mean_drift();
  if (!drift.valid) {
    return false;
  }
  return drift.dcl <= config_.force_mean_drift_tol &&
         drift.dcd <= config_.force_mean_drift_tol &&
         drift.dcm <= config_.force_mean_drift_tol;
}

bool ConvergenceMonitor::forces_physically_converged() const {
  return forces_stable() && forces_mean_drift_stable();
}

bool ConvergenceMonitor::converged(const int iter, const float residual_ratio) const {
  if (iter + 1 < config_.min_iterations) {
    return false;
  }
  if (residual_ratio > config_.residual_ratio_target) {
    return false;
  }
  return forces_physically_converged();
}
}  // namespace cfd::core
