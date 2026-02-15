# Developer Guide

## Architecture

- `src/cfd_core/include/cfd_core`: public C++ interfaces.
- `src/cfd_core/mesh`: mesh and topology scaffolding.
- `src/cfd_core/numerics`: residual and solver stubs.
- `src/cfd_core/io`: VTU writer stubs.
- `src/cfd_core/cuda`: CUDA backend placeholders and kernel stubs.
- `src/cfd_core/bindings`: pybind11 module (`cfd_core`).
- `python/cfd`: user layer (CLI, case handling, sweeps, post, UI).

## Where to implement future logic

- Fluxes and reconstruction:
  - Start in `src/cfd_core/numerics/residual_stub.cpp` with split files for convective/viscous fluxes.
- Linear/nonlinear solver loops:
  - Extend `solver.hpp` and introduce dedicated solver source files under `src/cfd_core/numerics`.
- Mesh and geometry:
  - Extend `mesh.hpp`/`geometry.hpp`, then populate `src/cfd_core/mesh`.
- VTK output:
  - Replace placeholder writer in `src/cfd_core/io/io_vtk_stub.cpp` with cell/field exports.
- CUDA kernels:
  - Add kernels in `src/cfd_core/cuda/kernels.cu` and launch orchestration in `cuda_backend.cu`.

## Python integration model

- Python parses and validates YAML.
- Python writes a resolved case snapshot to output directory.
- pybind11 call executes native stub run and returns summary dictionary.
- Python post-processing reads CSV outputs and generates plots.

## Coding conventions

- C++: C++20, RAII, explicit interfaces, deterministic outputs.
- Python: typed function signatures, clear error messages, minimal side effects.
- TODO markers:
  - `TODO(physics): ...`
  - `TODO(numerics): ...`
  - `TODO(cuda): ...`

## Developer loop

1. Configure CMake (`build/cpu` or `build/cuda`).
2. Install Python package in editable mode.
3. Run CLI demo case.
4. Run smoke tests.
5. Run format scripts before commit.