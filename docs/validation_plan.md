# Validation Plan

## Goal

Keep confidence high while iterating on numerics, especially low-Mach behavior.

## Current automated checks

Native C++ tests:

- `cfd_core_smoke`
- `cfd_scalar_parity`
- `cfd_euler_regression`
- `cfd_euler_aoa0_regression`
- `cfd_euler_aoa0_coeff`
- `cfd_euler_wall_sanity`
- `cfd_euler_uniform_invariance`
- `cfd_euler_parity`
- `cfd_euler_low_mach_asymptotic`

Python checks:

- import checks for `cfd` and `cfd_core`
- CLI smoke output generation

## Minimum regression gate

Run this before merging solver changes:

```bash
ctest --test-dir build/cpu -R "cfd_euler_regression|cfd_euler_uniform_invariance|cfd_euler_wall_sanity|cfd_euler_low_mach_asymptotic" --output-on-failure
```

Also run:

```bash
pytest -q tests/python
```

## Baseline data

Canonical low-Mach mechanism baseline files:

- `tests/data/baselines/low_mach_euler/resolved_case.yaml`
- `tests/data/baselines/low_mach_euler/expected.yaml`

These keep the low-Mach mechanism test reproducible without storing huge run folders.

## Key run artifacts to inspect

- `residuals.csv`
- `forces.csv`
- `euler_diagnostics.csv`
- `cp_wall.csv`
- `field_0000.vtu`
- `run.log`

## Next validation expansion

- mesh sensitivity for Euler coefficients
- farfield-radius sensitivity checks
- stronger low-Mach trend tracking over more Mach points
- future RANS/3D validation once those paths are implemented
