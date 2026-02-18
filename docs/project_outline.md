# Project Outline

## Snapshot (Feb 17, 2026)

- Project: AeroCFD
- Current phase: `P2` (Euler), with all-speed steady robustness work in progress
- Direction: CUDA-first unstructured finite-volume solver with CPU fallback
- End goal: reliable aerodynamic outputs for airfoils first, then finite wings

## Mission

Build a practical, verification-driven CFD solver that is:

- accurate enough to trust trends (`Cp`, `Cl`, `Cd`, `Cm`)
- fast enough for sweeps (AoA/Mach/Re)
- structured enough to extend from Euler -> Navier-Stokes -> RANS

## Guiding principles

- correctness before speed
- one FV framework for 2D and 3D
- keep major fields GPU-resident where possible
- every run should be reproducible (resolved config + logs + CSV + VTU)
- regression gate stays green before new model work

## Scope

Core path:

- Euler foundation (current)
- laminar Navier-Stokes
- RANS SA, then RANS SST

Optional/bonus path:

- transition model (`gamma-Re_theta` style)
- URANS with dual-time stepping

Non-goals for early releases:

- LES/DNS
- moving/deforming mesh
- reacting multi-species flow
- full MPI scaling

## Numerical direction

- spatial: cell-centered FV, 2nd-order MUSCL reconstruction
- gradients: Green-Gauss baseline, least-squares production path
- inviscid flux: Rusanov baseline, HLLC production
- low-Mach handling: all-speed flux treatment and controlled continuation
- steady strategy: robust pseudo-time convergence with force-based stopping

## Milestones and definition of done

### P0 - Repo skeleton + CI + CLI

- DoD: project builds cleanly, smoke tests pass, basic case run works

### P1 - Mesh/metrics + VTU output

- DoD: mesh/metric sanity checks pass, ParaView output is correct

### P2 - Euler baseline 

- DoD: stable convergence on reference airfoil cases, regression ladder established, CPU/CUDA parity coverage for core residual behavior

### P3 - Laminar Navier-Stokes

- DoD: viscous terms and no-slip walls verified, laminar benchmarks behave physically

### P4 - RANS SA

- DoD: robust steady convergence and reasonable AoA/Re trend behavior

### P5 - RANS SST

- DoD: improved separated-flow behavior vs SA and stable runs on broader cases

### P6 - Transition (bonus)

- DoD: transition-sensitive drag trends improve on at least one reference case

### P7 - URANS (bonus)

- DoD: dual-time stepping produces stable unsteady histories and restartable runs

### P8 - GPU optimization pass

- DoD: profiling shows meaningful speedup on target cases vs CPU baseline

### P9 - UX/reporting polish

- DoD: sweep -> plots/report bundle workflow is one-command and reproducible

## Validation ladder (high level)

- unit checks: geometry/metrics, invariance, BC sanity
- regression checks: fixed-case coefficient windows and trend checks
- parity checks: CPU vs CUDA for critical kernels/metrics
- physics checks: `AoA=0` symmetry, low-Mach mechanism behavior, refinement trends

## Main risks and mitigations

- low-Mach instability: staged continuation + mechanism tests + conservative defaults
- implicit complexity creep: keep explicit/LTS path strong first
- unstructured GPU atomic bottlenecks: profile first, then consider coloring/two-pass options
- RANS fragility: strict regression gate and robust startup/limiter settings
