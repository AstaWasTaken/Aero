# Project Outline

## Mission

Build a reliable CFD solver for aerodynamic analysis of airfoils (and later wings), with a CUDA-first acceleration path and a CPU fallback for validation and portability.

## Current Progress Snapshot

### M1 complete

- Deterministic unstructured mesh generation (demo path).
- Scalar advection residual assembly on CPU and CUDA.
- CUDA face-kernel pattern with atomic accumulation.
- VTU output and CLI wiring.

### M2 complete

- 2D compressible Euler airfoil solver on CPU.
- Rusanov flux + MUSCL/minmod reconstruction.
- Slip-wall and farfield boundary conditions.
- Cp/Cl/Cd/Cm post-processing.
- Euler regression test coverage.

### M2.5 complete

- CUDA acceleration for Euler residual assembly hot path:
  - face reconstruction
  - Rusanov face flux
  - per-cell residual accumulation
- CPU-driven pseudo-time loop retained for now.
- CPU/CUDA Euler residual parity test added.

## Next Stages

### Near term

- Keep state/residual resident on GPU across iterations.
- Move gradient construction to GPU.
- Strengthen solver robustness and convergence controls.

### Mid term

- Extend from Euler to viscous Navier-Stokes / RANS.
- Add turbulence-model infrastructure (SA / SST).
- Expand case coverage and validation datasets.

### Long term

- 3D finite-wing workflows.
- Unsteady pathway (URANS / dual-time stepping).
- Throughput optimization for sweeps and larger runs.

## Backend Strategy

- Default backend: CPU.
- Optional backend: CUDA (`CFD_ENABLE_CUDA=ON` and runtime GPU availability).
- Shared interfaces between CPU and CUDA code paths where possible.

## Outputs and Tooling

- VTU output for ParaView.
- CSV outputs for residual and force history.
- Case snapshot + run log for reproducibility.
- Python CLI for `run`, `sweep`, `post`, and optional UI.
