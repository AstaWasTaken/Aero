# Developer Guide

## Scope

This repo is centered on:

- 2D Euler airfoil solves (main path)
- scalar demo path (smoke/parity support)
- CPU backend with optional CUDA backend

## Code map

- `src/cfd_core/include/cfd_core/` public C++ interfaces
- `src/cfd_core/solvers/euler/` Euler solver modules
- `src/cfd_core/solvers/euler_solver.cpp` Euler run driver
- `src/cfd_core/numerics/` flux/reconstruction/scalar residual path
- `src/cfd_core/cuda/` CUDA backend and kernels
- `src/cfd_core/post/` force and Cp post-processing
- `python/cfd/` Python CLI and case utilities
- `tests/cpp/` native tests
- `tests/python/` Python smoke tests

## Build and test loop

CPU:

```bash
cmake -S . -B build/cpu -DCFD_ENABLE_CUDA=OFF -DCFD_BUILD_TESTS=ON -DCFD_BUILD_PYTHON=ON
cmake --build build/cpu --config Release
ctest --test-dir build/cpu --output-on-failure
```

Targeted Euler checks:

```bash
ctest --test-dir build/cpu -R "cfd_euler_regression|cfd_euler_uniform_invariance|cfd_euler_wall_sanity|cfd_euler_low_mach_asymptotic" --output-on-failure
```

Python:

```bash
python -m pip install -e python
pytest -q tests/python
```

## Runtime workflow

1. Resolve and run a case with `cfd run`.
2. Outputs are written to your selected output directory.
3. Post-process with `cfd post` or ParaView (`field_0000.vtu`).

Example:

```bash
cfd run cases/naca0012_euler_2d.yaml --backend cpu --out results/euler_cpu_demo
```

## Important low-Mach knobs

Common `numerics` keys you will tune most often:

- `precond`
- `precond_mach_ref`, `precond_mach_min`
- `all_speed_flux_fix`
- `all_speed_mach_cutoff`
- `all_speed_f_min`
- `all_speed_ramp_start_iter`, `all_speed_ramp_iters`
- `stabilization_mach_floor_k_start`, `stabilization_mach_floor_k_target`
- `stabilization_ramp_iters`
