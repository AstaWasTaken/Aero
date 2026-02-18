# Low-Mach Notes

## Status

- Preconditioning is still experimental.
- It is only active when:
  - case sets `precond: on`
  - env var `AERO_ENABLE_EXPERIMENTAL_PRECOND=1` is set
- For normal low-Mach runs, preferred path is:
  - `all_speed_flux_fix: on`
  - preconditioning off

## Core idea

The solver scales acoustic stiffness using a bounded factor:

- `M = |u| / a`
- `beta = clamp(max(M, mach_ref, mach_min), beta_min, beta_max)`

At low Mach, this reduces acoustic dominance. At higher Mach, behavior returns toward baseline.

## Where scaling is used

1. Flux dissipation:
- acoustic terms are scaled
- all-speed correction can reduce pressure dissipation at low Mach

2. Stabilization floor:
- a stagnation-safe floor prevents dissipation collapse near zero velocity zones
- configured via `stabilization_mach_floor_*` and ramp settings

3. Pseudo-time step sizing:
- local spectral radius scaling changes local pseudo-time step limits

## Main knobs

- `precond`
- `precond_mach_ref`
- `precond_mach_min`
- `precond_beta_min`
- `precond_beta_max`
- `all_speed_flux_fix`
- `all_speed_mach_cutoff`
- `all_speed_f_min`
- `all_speed_ramp_start_iter`
- `all_speed_ramp_iters`
- `stabilization_mach_floor_start`
- `stabilization_mach_floor_target`
- `stabilization_mach_floor_k_start`
- `stabilization_mach_floor_k_target`
- `stabilization_ramp_iters`

## Practical recommendation

For stable low-Mach sweeps:

1. keep `precond: off`
2. enable `all_speed_flux_fix: on`
3. use ramped activation (`all_speed_ramp_*`)
4. keep stabilization floor ramp active
