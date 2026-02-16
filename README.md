# AeroCFD Project

AeroCFD is my student CFD solver project for analyzing 2D airfoils (and eventually 3D wings), with a focus on useful aerodynamic outputs (Cp, Cl, Cd, Cm) and GPU acceleration with CUDA.

The codebase is no longer just a placeholder wiring demo. It now has working scalar and Euler paths, CPU and CUDA backend selection, test coverage for parity/regression, and deterministic VTU/CSV outputs.

## Current Implementation Status

### Implemented now

- C++20 native core (`cfd_core`) with optional CUDA backend.
- Deterministic unstructured mesh generation for:
  - demo scalar mesh
  - 2D airfoil O-grid-like triangular mesh
- Scalar advection residual assembly:
  - CPU path
  - CUDA face kernel path
- 2D compressible Euler airfoil solver:
  - Rusanov flux
  - MUSCL + minmod limiter
  - slip wall and farfield boundary conditions
  - force and wall Cp post-processing
- Euler residual CUDA acceleration (M2.5 scope):
  - face reconstruction + flux + residual accumulation on GPU
  - pseudo-time iteration loop remains CPU-driven for now
- Output pipeline:
  - `field_0000.vtu`
  - `residuals.csv`
  - `forces.csv`
  - `cp_wall.csv` (Euler case)
  - `run.log`
- Python package + CLI:
  - `cfd run`
  - `cfd sweep`
  - `cfd post`
  - `cfd ui`
- Tests:
  - C++ smoke test
  - scalar CPU/CUDA parity test
  - Euler regression test
  - Euler CPU/CUDA residual parity test
  - Python import/CLI smoke tests

### Planned next

- Extend from Euler to viscous Navier-Stokes and RANS models.
- Move more of the Euler loop to GPU (state/residual residency, gradients, implicit/JFNK paths).
- Expand 3D wing workflow and validation cases.

## Repository Layout

- `src/cfd_core`: native core, numerics, mesh, IO, post, solver
- `src/cfd_core/cuda`: CUDA kernels and backend code
- `src/cfd_core/bindings`: pybind11 module
- `python/cfd`: Python API and CLI
- `cases`: sample YAML case files
- `tests`: C++ and Python tests
- `scripts`: helper build/run scripts
- `docs`: project outline, developer guide, validation plan

## Build

### Linux/macOS quick scripts

CPU:

```bash
bash scripts/build_cpu.sh
```

CUDA:

```bash
bash scripts/build_cuda.sh
```

### Windows quick script

From repo root:

```bat
scripts\build_cuda.cmd
```

This script sets up the VS dev environment (if needed), configures CUDA build, builds, and runs CTest.

### Manual CMake build

CPU-only:

```bash
cmake -S . -B build/cpu -DCFD_ENABLE_CUDA=OFF -DCFD_BUILD_PYTHON=ON -DCFD_BUILD_TESTS=ON
cmake --build build/cpu --config Release
ctest --test-dir build/cpu --output-on-failure
```

CUDA-enabled:

```bash
cmake -S . -B build/cuda -DCFD_ENABLE_CUDA=ON -DCFD_BUILD_PYTHON=ON -DCFD_BUILD_TESTS=ON
cmake --build build/cuda --config Release
ctest --test-dir build/cuda --output-on-failure
```

If CUDA is requested but not detected, CMake prints a warning and continues with CPU-only backend.

## Install Python Package (Editable)

From repository root:

```bash
python -m pip install -e python
```

Sanity check:

```bash
python -c "import cfd; print(cfd.__version__)"
python -m cfd.cli --help
```

## Run Cases

Scalar demo:

```bash
python -m cfd.cli run cases/scalar_advect_demo.yaml --backend cpu --out results/scalar_demo
```

Euler airfoil (CPU residual):

```bash
python -m cfd.cli run cases/naca0012_euler_2d.yaml --backend cpu --out results/euler_cpu_demo
```

Euler airfoil (CUDA residual):

```bash
python -m cfd.cli run cases/naca0012_euler_2d.yaml --backend cuda --out results/euler_cuda_demo
```

Typical Euler outputs in the run folder:

- `field_0000.vtu`
- `cp_wall.csv`
- `forces.csv`
- `residuals.csv`
- `run.log`

## Open VTU in ParaView

1. Start ParaView.
2. Open `results/<run_name>/field_0000.vtu`.
3. Click **Apply**.

## Tests

Python:

```bash
pytest -q tests/python
```

C++ via CTest:

```bash
ctest --test-dir build/cpu --output-on-failure
```

CUDA builds include CPU/CUDA parity tests. If CUDA runtime is unavailable, parity tests skip gracefully.

## Formatting

```bash
bash scripts/format_all.sh
```

- C++: `clang-format`
- Python: `ruff` + `black`

## Roadmap and Docs

- `docs/project_outline.md`
- `docs/developer_guide.md`
- `docs/validation_plan.md`
