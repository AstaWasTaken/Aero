"""Case loading and validation utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

DEFAULT_SCALAR_CASE: dict[str, Any] = {
    "case_type": "scalar_advect_demo",
    "name": "scalar_advect_demo",
    "mesh": {
        "generator": "demo_tri_2x2",
    },
    "flow": {
        "u_inf": [1.0, 0.35, 0.0],
    },
    "scalar": {
        "inflow_phi": 1.0,
    },
    "solver": {
        "iterations": 1,
    },
}

DEFAULT_EULER_CASE: dict[str, Any] = {
    "case_type": "euler_airfoil_2d",
    "name": "naca0012_euler_2d",
    "geometry": {
        "type": "airfoil2d",
        "profile": "NACA0012",
        "chord": 1.0,
    },
    "mesh": {
        "generator": "structured_o",
        "circumferential": 160,
        "radial": 48,
        "farfield_radius": 15.0,
        "radial_stretch": 1.4,
    },
    "flow": {
        "mach": 0.15,
        "aoa_deg": 2.0,
        "p_inf": 101325.0,
        "t_inf": 288.15,
    },
    "solver": {
        "equations": "euler",
        "iterations": 300,
        "min_iterations": 40,
    },
    "numerics": {
        "gamma": 1.4,
        "gas_constant": 287.05,
        "cfl_start": 0.2,
        "cfl_max": 1.2,
        "cfl_ramp_iters": 120,
        "residual_reduction_target": 1e-3,
        "force_stability_tol": 2e-5,
    },
    "post": {
        "x_ref": 0.25,
        "y_ref": 0.0,
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


def _infer_case_type(case: dict[str, Any]) -> str:
    case_type = str(case.get("case_type", "")).strip().lower()
    if case_type:
        return case_type

    equations = str(case.get("solver", {}).get("equations", "")).strip().lower()
    if equations in {"scalar_advection", "scalar"}:
        return "scalar_advect_demo"
    if equations in {"euler", "compressible_euler"}:
        return "euler_airfoil_2d"
    return "scalar_advect_demo"


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
    case_type = _infer_case_type(case)
    if case_type not in {"scalar_advect_demo", "euler_airfoil_2d"}:
        raise ValueError("case_type must be 'scalar_advect_demo' or 'euler_airfoil_2d'.")

    if case_type == "scalar_advect_demo":
        u_inf = case.get("flow", {}).get("u_inf", [])
        if not isinstance(u_inf, list) or len(u_inf) != 3:
            raise ValueError("scalar flow.u_inf must be a 3-component list.")
        return

    geometry = case.get("geometry", {})
    flow = case.get("flow", {})
    solver = case.get("solver", {})
    mesh = case.get("mesh", {})

    if str(geometry.get("type", "")).strip().lower() != "airfoil2d":
        raise ValueError("euler_airfoil_2d requires geometry.type=airfoil2d.")
    if int(solver.get("iterations", 0)) <= 0:
        raise ValueError("solver.iterations must be positive.")
    if int(mesh.get("circumferential", 0)) < 32:
        raise ValueError("mesh.circumferential must be >= 32.")
    if int(mesh.get("radial", 0)) < 4:
        raise ValueError("mesh.radial must be >= 4.")
    if float(flow.get("mach", 0.0)) < 0.0:
        raise ValueError("flow.mach must be >= 0.")


def resolve_case(case: dict[str, Any], backend: str | None = None) -> dict[str, Any]:
    case_type = _infer_case_type(case)
    defaults = DEFAULT_SCALAR_CASE if case_type == "scalar_advect_demo" else DEFAULT_EULER_CASE
    resolved = _deep_merge(defaults, case)
    resolved["case_type"] = case_type
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


def _as_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _as_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def write_native_case_config(case: dict[str, Any], out_dir: str | Path) -> Path:
    """Write a deterministic key/value case file consumed by the native core."""

    case_type = _infer_case_type(case)
    lines: list[str] = [f"case_type={case_type}"]

    if case_type == "scalar_advect_demo":
        flow = case.get("flow", {})
        scalar = case.get("scalar", {})
        u_inf = flow.get("u_inf", [1.0, 0.35, 0.0])
        if not isinstance(u_inf, list) or len(u_inf) != 3:
            u_inf = [1.0, 0.35, 0.0]
        lines.append(
            "u_inf="
            + ",".join(f"{_as_float(component, 0.0):.9g}" for component in u_inf[:3])
        )
        lines.append(f"inflow_phi={_as_float(scalar.get('inflow_phi', 1.0), 1.0):.9g}")
    else:
        geometry = case.get("geometry", {})
        mesh = case.get("mesh", {})
        flow = case.get("flow", {})
        solver = case.get("solver", {})
        numerics = case.get("numerics", {})
        post = case.get("post", {})

        profile = str(geometry.get("profile", "NACA0012"))
        naca_digits = "".join(ch for ch in profile if ch.isdigit())
        if len(naca_digits) < 4:
            naca_digits = "0012"
        naca_digits = naca_digits[:4]

        airfoil_file = str(geometry.get("airfoil_file", "")).strip()
        if airfoil_file:
            lines.append("airfoil_source=file")
            lines.append(f"airfoil_file={airfoil_file}")
        else:
            lines.append("airfoil_source=naca4")
            lines.append(f"naca_code={naca_digits}")

        lines.append(f"chord={_as_float(geometry.get('chord', 1.0), 1.0):.9g}")
        lines.append(f"num_circumferential={_as_int(mesh.get('circumferential', 160), 160)}")
        lines.append(f"num_radial={_as_int(mesh.get('radial', 48), 48)}")
        lines.append(f"farfield_radius={_as_float(mesh.get('farfield_radius', 15.0), 15.0):.9g}")
        lines.append(f"radial_stretch={_as_float(mesh.get('radial_stretch', 1.4), 1.4):.9g}")

        lines.append(f"mach={_as_float(flow.get('mach', 0.15), 0.15):.9g}")
        lines.append(f"aoa_deg={_as_float(flow.get('aoa_deg', 2.0), 2.0):.9g}")
        lines.append(f"p_inf={_as_float(flow.get('p_inf', 101325.0), 101325.0):.9g}")
        lines.append(f"t_inf={_as_float(flow.get('t_inf', 288.15), 288.15):.9g}")
        rho_inf = flow.get("rho_inf", None)
        if rho_inf is not None:
            lines.append(f"rho_inf={_as_float(rho_inf, 0.0):.9g}")

        lines.append(f"iterations={_as_int(solver.get('iterations', 300), 300)}")
        lines.append(f"min_iterations={_as_int(solver.get('min_iterations', 40), 40)}")
        lines.append(f"gamma={_as_float(numerics.get('gamma', 1.4), 1.4):.9g}")
        lines.append(
            f"gas_constant={_as_float(numerics.get('gas_constant', 287.05), 287.05):.9g}"
        )
        lines.append(f"cfl_start={_as_float(numerics.get('cfl_start', 0.2), 0.2):.9g}")
        lines.append(f"cfl_max={_as_float(numerics.get('cfl_max', 1.2), 1.2):.9g}")
        lines.append(f"cfl_ramp_iters={_as_int(numerics.get('cfl_ramp_iters', 120), 120)}")
        lines.append(
            "residual_reduction_target="
            f"{_as_float(numerics.get('residual_reduction_target', 1e-3), 1e-3):.9g}"
        )
        lines.append(
            f"force_stability_tol={_as_float(numerics.get('force_stability_tol', 2e-5), 2e-5):.9g}"
        )
        lines.append(f"x_ref={_as_float(post.get('x_ref', 0.25), 0.25):.9g}")
        lines.append(f"y_ref={_as_float(post.get('y_ref', 0.0), 0.0):.9g}")

    output_path = Path(out_dir) / "native_case.cfg"
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_path
