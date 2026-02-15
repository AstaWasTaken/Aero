"""Sweep orchestration for multi-case studies (stub implementation)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from cfd import run_case_native
from cfd.case import load_case, prepare_output_dir, resolve_case, write_resolved_case


def run_sweep(
    sweep_path: str | Path,
    jobs: int = 1,
    backend: str = "cpu",
    out_dir: str | Path | None = None,
) -> dict[str, Any]:
    sweep_file = Path(sweep_path)
    if not sweep_file.exists():
        raise FileNotFoundError(f"Sweep file not found: {sweep_file}")

    with sweep_file.open("r", encoding="utf-8") as handle:
        sweep_cfg = yaml.safe_load(handle) or {}

    if not isinstance(sweep_cfg, dict):
        raise ValueError("Sweep YAML must contain a top-level mapping.")

    base_case = sweep_cfg.get("base_case")
    if not base_case:
        raise ValueError("Sweep YAML must define 'base_case'.")

    base_case_path = (sweep_file.parent / str(base_case)).resolve()
    base_case_cfg = load_case(base_case_path)

    aoa_values = sweep_cfg.get("sweep", {}).get("aoa_deg", [base_case_cfg.get("flow", {}).get("aoa_deg", 0.0)])
    output_root = prepare_output_dir(out_dir, prefix="sweep")

    summaries: list[dict[str, Any]] = []
    # TODO(numerics): Replace serial sweep loop with robust process/thread execution model.
    for index, aoa in enumerate(aoa_values):
        run_dir = output_root / f"run_{index:03d}"
        run_dir.mkdir(parents=True, exist_ok=True)

        resolved_case = resolve_case(base_case_cfg, backend=backend)
        resolved_case.setdefault("flow", {})
        resolved_case["flow"]["aoa_deg"] = float(aoa)
        write_resolved_case(resolved_case, run_dir)

        summary = run_case_native(str(base_case_path), str(run_dir), backend)
        summary["aoa_deg"] = float(aoa)
        summaries.append(summary)

    return {
        "status": "ok",
        "jobs_requested": int(jobs),
        "output_root": str(output_root),
        "runs": summaries,
    }