# Developer Guide

## Architecture

- `src/cfd_core/include/cfd_core`: public native interfaces.
- `src/cfd_core/mesh`: mesh generation for demo and airfoil cases.
- `src/cfd_core/numerics`: scalar residual runtime path and backend dispatch.
- `src/cfd_core/solvers/euler_solver.cpp`: 2D Euler solver loop, CPU/CUDA residual path selection.
- `src/cfd_core/cuda`: CUDA kernels and backend management.
- `src/cfd_core/io`: VTU writers for scalar and Euler fields.
- `src/cfd_core/post`: force and Cp post-processing.
- `src/cfd_core/bindings`: pybind11 module (`cfd_core`).
- `python/cfd`: Python CLI, case resolution, sweep/post helpers.

## Key Native Entry Points

- Case dispatch:
  - `src/cfd_core/numerics/residual_stub.cpp`
  - `cfd::core::run_case(...)`
- Scalar run path:
  - `cfd::core::run_scalar_case(...)`
- Euler run path:
  - `src/cfd_core/solvers/euler_solver.cpp`
  - `cfd::core::run_euler_airfoil_case(...)`

## CUDA Integration Notes

- Scalar CUDA:
  - `src/cfd_core/cuda/kernels.cu`
  - `src/cfd_core/cuda/cuda_backend.cu`
- Euler CUDA (M2.5):
  - `src/cfd_core/cuda/euler_kernels.cu`
  - `src/cfd_core/cuda/euler_backend.cu`
- Public CUDA API:
  - `src/cfd_core/include/cfd_core/cuda_backend.hpp`
- Build wiring:
  - `src/cfd_core/cuda/CMakeLists.txt`
  - `src/cfd_core/CMakeLists.txt`

## Python Integration Model

- Python loads and validates YAML.
- Python writes a normalized native case config into the output folder.
- pybind executes native `run_case(...)`.
- Native side writes VTU/CSV outputs and returns summary metrics.

## Practical Developer Loop

1. Configure/build native targets.
2. Install Python package editable (`python -m pip install -e python`) when needed.
3. Run a case through `python -m cfd.cli`.
4. Run C++ and Python tests.
5. Re-run format tools before commit.

## Common Commands

CPU build:

```bash
cmake -S . -B build/cpu -DCFD_ENABLE_CUDA=OFF -DCFD_BUILD_TESTS=ON -DCFD_BUILD_PYTHON=ON
cmake --build build/cpu --config Release
ctest --test-dir build/cpu --output-on-failure
```

CUDA build:

```bash
cmake -S . -B build/cuda -DCFD_ENABLE_CUDA=ON -DCFD_BUILD_TESTS=ON -DCFD_BUILD_PYTHON=ON
cmake --build build/cuda --config Release
ctest --test-dir build/cuda --output-on-failure
```

Windows helper:

```bat
scripts\build_cuda.cmd
```

## Current Technical TODOs

- `TODO(cuda):` keep Euler state/residual resident on GPU across iterations.
- `TODO(cuda):` move Euler gradient calculation to GPU.
- `TODO(cuda):` support implicit/JFNK-style GPU reuse of residual operator.
- `TODO(physics):` extend from Euler to viscous Navier-Stokes / RANS models.
