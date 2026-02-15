# Validation Plan

## Goals

Create staged confidence in solver behavior from simple 2D checks to 3D finite-wing trends.

## Stage A: 2D airfoil baselines

- NACA0012 low-Mach attached-flow trends.
- Angle-of-attack sweep for monotonic lift slope behavior.
- Convergence monitoring for residual decay stability.

## Stage B: 2D robustness checks

- Sensitivity to mesh refinement and CFL settings.
- Mild shock-containing transonic cases for robustness checks.
- Boundary-condition consistency tests.

## Stage C: 3D consistency

- Extruded 2D geometry in 3D grid to validate dimensional consistency.
- Compare sectional loads against 2D references.

## Stage D: finite-wing trend checks

- Rectangular wing baseline: induced drag and lift slope trends.
- Tapered wing trend comparisons across moderate AR values.
- Grid convergence trends for integrated force coefficients.

## Stage E: unsteady pathway

- URANS oscillator sanity case.
- Time-step sensitivity and phase-consistency checks.

## Reporting outputs

- Residual history plots.
- Force coefficient histories (`CL`, `CD`, `CM`).
- Snapshot artifacts (`resolved_case.yaml`, `run.log`, `*.vtu`).