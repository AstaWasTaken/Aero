# AeroCFD - v0.1.1

AeroCFD is my CFD solver project for 2D airfoil cases, with CPU and optional CUDA backends.
Right now the main focus is steady 2D compressible Euler runs (plus a small scalar demo path).

## What is implemented

- C++20 core library (`cfd_core`) with CMake build.
- Optional CUDA backend (auto-disabled if CUDA compiler is not found).
- 2D Euler airfoil solver with:
  - HLLC and Rusanov fluxes
  - minmod/venkat limiters
  - low-Mach controls and all-speed flux fix options
  - force/Cp outputs and VTU export
- Python layer (`cfd`) for CLI workflows (`run`, `sweep`, `post`, `ui`).
- C++ regression/sanity tests + Python smoke tests.

## Repo layout

- `src/cfd_core/` native solver code
- `src/cfd_core/cuda/` CUDA kernels/backend
- `src/cfd_core/bindings/` pybind11 module
- `python/cfd/` Python CLI and utilities
- `cases/` active case YAML files
- `tests/` C++ and Python tests
- `tools/clean_artifacts.py` cleanup utility for generated files
- `docs/` developer/validation notes

## Build

### CPU build

```bash
cmake -S . -B build/cpu -DCFD_ENABLE_CUDA=OFF -DCFD_BUILD_TESTS=ON -DCFD_BUILD_PYTHON=ON
cmake --build build/cpu --config Release
```

Quick script:

```bash
bash scripts/build_cpu.sh
```

### CUDA build (if available)

```bash
cmake -S . -B build/cuda -DCFD_ENABLE_CUDA=ON -DCFD_BUILD_TESTS=ON -DCFD_BUILD_PYTHON=ON
cmake --build build/cuda --config Release
```

Quick scripts:

```bash
bash scripts/build_cuda.sh
```

Windows:

```bat
scripts\build_cuda.cmd
```

## Python install

From repo root:

```bash
python -m pip install -e python
```

Check CLI:

```bash
cfd --help
```

## Run examples

Scalar demo:

```bash
cfd run cases/scalar_advect_demo.yaml --backend cpu --out results/scalar_demo
```

Euler 2D (CPU):

```bash
cfd run cases/naca0012_euler_2d.yaml --backend cpu --out results/euler_cpu_demo
```

Euler 2D (CUDA):

```bash
cfd run cases/naca0012_euler_2d.yaml --backend cuda --out results/euler_cuda_demo
```

AoA sweep:

```bash
cfd sweep cases/naca0012_euler_2d_aoa_sweep.yaml --backend cpu --out results/naca0012_sweep
```

Typical Euler output files:

- `field_0000.vtu`
- `forces.csv`
- `residuals.csv`
- `cp_wall.csv`
- `wall_flux.csv`
- `run.log`

## Tests

Run all C++ tests:

```bash
ctest --test-dir build/cpu --output-on-failure
```

Run the main Euler regression/sanity subset:

```bash
ctest --test-dir build/cpu -R "cfd_euler_regression|cfd_euler_uniform_invariance|cfd_euler_wall_sanity|cfd_euler_low_mach_asymptotic" --output-on-failure
```

Run Python tests:

```bash
pytest -q tests/python
```

## Notes

- Canonical low-Mach mechanism baseline data is in `tests/data/baselines/low_mach_euler/`.
- See `cases/README.md` for current active case files.
- More details: `docs/developer_guide.md`, `docs/validation_plan.md`, `docs/low_mach_preconditioning.md`.
