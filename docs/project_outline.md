# Project Outline

## Mission

Develop a production-grade CFD solver focused on aerodynamic analysis of airfoils and wings with a CUDA-first
acceleration strategy and a robust CPU fallback.

## Scope by stage

### Stage 0 (this scaffold)

- C++20 core build + CUDA option + Python bindings.
- YAML-driven cases and deterministic output plumbing.
- CLI and optional Streamlit entry point.

### Stage 1 (Euler / baseline RANS scaffold)

- 2D finite-volume core for structured and unstructured meshes.
- Inviscid flux implementation and residual assembly framework.
- Boundary condition framework (farfield, wall, symmetry).

### Stage 2 (RANS + turbulence)

- Compressible RANS with SST baseline.
- Transitional modeling hooks and infrastructure.
- Steady-state convergence controls and monitoring.

### Stage 3 (3D and finite wings)

- Extruded 3D consistency checks from 2D sections.
- Full finite-wing meshing and solver support.
- Aerodynamic coefficients, induced effects, and trend validation.

### Stage 4 (URANS and advanced capabilities)

- Time-accurate URANS stepping.
- CUDA kernels for residuals, gradients, and linear solves.
- Multi-case sweep throughput optimization.

## Backend strategy

- Default backend: CPU (always available).
- Optional backend: CUDA (`CFD_ENABLE_CUDA=ON`).
- Shared interfaces for future HIP/SYCL exploration.

## Output and tooling

- Native output target: VTU/VTK for ParaView.
- Python post-processing for convergence and force history plots.
- Repeatable case snapshot capture for traceability.