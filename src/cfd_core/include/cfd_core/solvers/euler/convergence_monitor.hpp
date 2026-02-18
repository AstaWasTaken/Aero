#pragma once

#include "cfd_core/post/forces.hpp"

#include <deque>

namespace cfd::core {
struct ForceDrift {
  float dcl = 0.0f;
  float dcd = 0.0f;
  float dcm = 0.0f;
};

struct ForceMeanDrift {
  float dcl = 0.0f;
  float dcd = 0.0f;
  float dcm = 0.0f;
  bool valid = false;
};

struct ConvergenceMonitorConfig {
  float residual_ratio_target = 1.0e-3f;
  float force_stability_tol = 2.0e-5f;
  int force_stability_window = 6;
  int min_iterations = 40;
  float force_mean_drift_tol = 2.0e-5f;
};

class ConvergenceMonitor {
 public:
  explicit ConvergenceMonitor(ConvergenceMonitorConfig config);

  float update_residual(float residual_l2);
  ForceDrift update_forces(const ForceCoefficients& forces);
  bool forces_stable() const;
  ForceMeanDrift force_mean_drift() const;
  bool forces_mean_drift_stable() const;
  bool forces_physically_converged() const;
  bool converged(int iter, float residual_ratio) const;

 private:
  ConvergenceMonitorConfig config_;
  float initial_residual_l2_ = -1.0f;
  bool has_previous_force_ = false;
  ForceCoefficients previous_force_;
  std::deque<ForceCoefficients> force_window_;
};
}  // namespace cfd::core
