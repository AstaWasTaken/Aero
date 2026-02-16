# Validation Plan

## Objective

Build confidence step-by-step, starting from deterministic parity checks and moving toward physically meaningful aerodynamic trend validation.

## Implemented Validation (Current)

### Automated native tests

- `cfd_core_smoke`: build/import surface sanity.
- `cfd_scalar_parity`: scalar residual CPU/CUDA parity.
- `cfd_euler_regression`: Euler run remains bounded and produces expected outputs.
- `cfd_euler_parity`: Euler residual CPU/CUDA parity for the same mesh/state.

### Automated Python checks

- Import checks for `cfd` and `cfd_core`.
- CLI smoke run for scalar case output generation.

## Near-Term Validation Expansion

### 2D Euler quality checks

- Residual trend behavior over mesh refinement levels.
- Force coefficient sensitivity to CFL and iteration controls.
- Wall Cp shape consistency at low Mach attached flow.

### Boundary-condition robustness

- Farfield radius sensitivity.
- Slip-wall consistency checks across angle of attack.

## Mid-Term Validation (After Viscous/RANS Work)

- RANS baseline comparisons for NACA0012-style cases.
- Turbulence-model parameter sensitivity sweeps.
- Grid-convergence trend checks for integrated coefficients.

## Longer-Term Validation (3D/Unsteady)

- 3D extruded consistency checks against 2D sections.
- Finite-wing trend validation (lift slope and induced drag behavior).
- Unsteady sanity cases for time-step sensitivity and phase consistency.

## Reporting Artifacts

- `residuals.csv`
- `forces.csv`
- `cp_wall.csv` (Euler airfoil runs)
- `field_0000.vtu`
- `resolved_case.yaml`
- `run.log`
