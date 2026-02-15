# AeroCFD Project

AeroCFD is my CFD solver project for analysing 2D airfoils and 3D finite wings, with a focus on producing accurate aerodynamic results (e.g., Cp distributions and force coefficients) while keeping execution fast using GPU acceleration (CUDA).

Right now, the repository is set up as a working foundation: the build system, C++/CUDA core layout, Python bindings, CLI workflow, and output pipeline are in place. The current implementation only includes a simple deterministic “hello-world” compute path so that the architecture is verified end-to-end before I begin adding the full CFD physics and numerical methods.

## Planned Features

**Core solver**
- 2D airfoil + 3D wing support (shared finite-volume core)
- Unstructured mesh support (cells/faces + boundary patches)
- Compressible **Euler → Navier–Stokes → RANS**
- Turbulence models: **Spalart–Allmaras (SA)** and **k–ω SST**
- Optional bonus: **transition model** + **URANS (dual-time stepping)**

**Numerics**
- Second-order reconstruction (MUSCL + limiter)
- Gradient reconstruction (least-squares / Green–Gauss)
- Robust boundary conditions (farfield, wall, symmetry, periodic)
- Convergence monitoring (residuals + force convergence)

**Performance**
- **CUDA-first GPU acceleration** for flux/residual/gradient/turbulence kernels
- CPU/OpenMP fallback for compatibility + validation
- Profiling + optimisation (minimise host↔device transfers)

**Outputs & usability**
- Cp distribution + force/moment coefficients (Cl, Cd, Cm)
- VTU/VTK export for **ParaView** visualisation
- CLI workflow: `cfd run`, `cfd sweep`, `cfd post`
- Optional Streamlit UI for quick case setup + plots

## Features Repository

- C++20 core library (`cfd_core`) with CPU and optional CUDA backend toggle.
- pybind11 extension module (`cfd_core`) exposed to Python.
- Python package (`cfd`) with CLI subcommands: `run`, `sweep`, `post`, `ui`.
- YAML case loading + schema validation + resolved snapshot writing.
- Dummy run output generation:
  - `resolved_case.yaml`
  - `run.log`
  - `residuals.csv`
  - `forces.csv`
  - `field_0000.vtu`
- Smoke tests for C++ and Python.

## Repository layout

- `src/cfd_core`: native core and CUDA/backend stubs
- `src/cfd_core/bindings`: pybind11 module
- `python/cfd`: Python API and CLI
- `cases`: sample YAML inputs
- `tests`: C++ and Python smoke tests
- `docs`: project scope, architecture, validation plan

## Build the C++ project (CPU-only)

```bash
cmake -S . -B build/cpu -DCFD_ENABLE_CUDA=OFF -DCFD_BUILD_PYTHON=ON -DCFD_BUILD_TESTS=ON
cmake --build build/cpu --config Release
ctest --test-dir build/cpu --output-on-failure
```

## Build the C++ project (CUDA enabled)

```bash
cmake -S . -B build/cuda -DCFD_ENABLE_CUDA=ON -DCFD_BUILD_PYTHON=ON -DCFD_BUILD_TESTS=ON
cmake --build build/cuda --config Release
ctest --test-dir build/cuda --output-on-failure
```

If CUDA is requested but no CUDA compiler is available, CMake prints a warning and falls back to CPU mode.

## Install Python package (editable)

Run from repository root:

```bash
python -m pip install -e python
```

Sanity check:

```bash
python -c "import cfd; print(cfd.__version__)"
cfd --help
```

## Run the demo case

```bash
cfd run cases/naca0012_2d.yaml --backend cpu --out results/demo
```

Generated outputs in `results/demo` are deterministic placeholders for integration validation.

## Open VTU output in ParaView

1. Start ParaView.
2. Open `results/demo/field_0000.vtu`.
3. Click **Apply**.

The VTU currently contains a minimal placeholder unstructured grid.

## Run tests

Python smoke test:

```bash
pytest -q tests/python
```

C++ smoke test (via CTest):

```bash
ctest --test-dir build/cpu --output-on-failure
```

## Formatting

```bash
bash scripts/format_all.sh
```

- C++: `clang-format`
- Python: `ruff` + `black`

## Roadmap

See:

- `docs/project_outline.md`
- `docs/developer_guide.md`
- `docs/validation_plan.md`