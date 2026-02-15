"""Case loading and validation utilities.

This file owns user-facing YAML schema handling while the native layer remains focused
on compute kernels and deterministic output generation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

DEFAULT_CASE: dict[str, Any] = {
    "name": "unnamed_case",
    "geometry": {
        "type": "airfoil2d",
        "profile": "NACA0012",
        "chord": 1.0,
        "span": 1.0,
    },
    "mesh": {
        "generator": "structured_c",
        "cells": 20000,
    },
    "flow": {
        "mach": 0.15,
        "reynolds": 6.0e6,
        "aoa_deg": 2.0,
    },
    "solver": {
        "equations": "rans",
        "turbulence_model": "sst",
        "iterations": 50,
        "cfl": 1.5,
    },
    "output": {
        "format": "vtu",
        "write_interval": 50,
    },
}


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_case(case_path: str | Path) -> dict[str, Any]:
    path = Path(case_path)
    if not path.exists():
        raise FileNotFoundError(f"Case file not found: {path}")

    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}

    if not isinstance(data, dict):
        raise ValueError("Case YAML must contain a top-level mapping.")

    return data


def validate_case(case: dict[str, Any]) -> None:
    required_sections = ["geometry", "flow", "solver"]
    for section in required_sections:
        if section not in case:
            raise ValueError(f"Missing required section '{section}' in case config.")

    geometry_type = case.get("geometry", {}).get("type")
    if geometry_type not in {"airfoil2d", "wing3d"}:
        raise ValueError("geometry.type must be either 'airfoil2d' or 'wing3d'.")

    solver_iterations = case.get("solver", {}).get("iterations", 0)
    if int(solver_iterations) <= 0:
        raise ValueError("solver.iterations must be a positive integer.")


def resolve_case(case: dict[str, Any], backend: str | None = None) -> dict[str, Any]:
    resolved = _deep_merge(DEFAULT_CASE, case)
    validate_case(resolved)

    if backend:
        resolved.setdefault("runtime", {})
        resolved["runtime"]["backend"] = backend

    return resolved


def next_output_dir(root: str | Path = "results", prefix: str = "run") -> Path:
    root_path = Path(root)
    root_path.mkdir(parents=True, exist_ok=True)

    index = 1
    while True:
        candidate = root_path / f"{prefix}_{index:03d}"
        if not candidate.exists():
            return candidate
        index += 1


def prepare_output_dir(out_dir: str | Path | None, prefix: str = "run") -> Path:
    if out_dir is None:
        output = next_output_dir(prefix=prefix)
    else:
        output = Path(out_dir)

    output.mkdir(parents=True, exist_ok=True)
    return output


def write_resolved_case(case: dict[str, Any], out_dir: str | Path) -> Path:
    output_path = Path(out_dir) / "resolved_case.yaml"
    with output_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(case, handle, sort_keys=True)
    return output_path