"""Command-line entry point for AeroCFD."""

from __future__ import annotations

import argparse
import importlib.util
import subprocess
import sys
from pathlib import Path

from cfd import __version__, core_module_path, cuda_available, native_core_loaded, run_case_native
from cfd.case import (
  load_case,
  prepare_output_dir,
  resolve_case,
  write_native_case_config,
  write_resolved_case,
)
from cfd.post import generate_plots
from cfd.sweep import run_sweep


def _build_parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser(prog="cfd", description="AeroCFD command-line interface")
  parser.add_argument("--version", action="store_true", help="Show package version and exit.")

  subparsers = parser.add_subparsers(dest="command")

  run_parser = subparsers.add_parser("run", help="Run one CFD case from YAML.")
  run_parser.add_argument("case_yaml", type=Path, help="Path to case YAML file.")
  run_parser.add_argument("--backend", default="cpu", choices=["cpu", "cuda"], help="Backend to use.")
  run_parser.add_argument("--out", type=Path, default=None, help="Output directory.")

  sweep_parser = subparsers.add_parser("sweep", help="Run a sweep YAML.")
  sweep_parser.add_argument("sweep_yaml", type=Path, help="Path to sweep YAML file.")
  sweep_parser.add_argument("--jobs", type=int, default=1, help="Requested job count (stub is serial).")
  sweep_parser.add_argument("--backend", default="cpu", choices=["cpu", "cuda"], help="Backend to use.")
  sweep_parser.add_argument("--out", type=Path, default=None, help="Sweep output directory.")

  post_parser = subparsers.add_parser("post", help="Generate plots from CSV outputs.")
  post_parser.add_argument("results_dir", type=Path, help="Results directory.")

  subparsers.add_parser("ui", help="Launch Streamlit UI if installed.")
  return parser


def _run_command(args: argparse.Namespace) -> int:
  backend = str(args.backend).strip().lower()
  if backend == "cuda" and not cuda_available():
    print("CUDA backend requested but unavailable in this build.")
    print("Rebuild with CUDA enabled and a detected CUDA compiler.")
    return 2

  output_dir = prepare_output_dir(args.out, prefix="run")
  try:
    user_case = load_case(args.case_yaml)
    resolved_case = resolve_case(user_case, backend=backend)
    case_type = str(resolved_case.get("case_type", "scalar_advect_demo"))
    if case_type == "euler_airfoil_2d" and not native_core_loaded:
      print("Euler airfoil runs require native cfd_core extension, but Python fallback is loaded.")
      print(f"Loaded module: {core_module_path}")
      return 3

    write_resolved_case(resolved_case, output_dir)
    native_case = write_native_case_config(resolved_case, output_dir)
    summary = run_case_native(str(native_case), str(output_dir), backend)
  except Exception as exc:  # pragma: no cover - CLI error path
    print(f"Run failed: {exc}")
    return 1

  print(f"Run completed. Outputs: {output_dir}")
  if resolved_case.get("case_type") == "euler_airfoil_2d":
    print(
      f"Final coefficients: Cl={summary.get('cl', float('nan')):.6f}, "
      f"Cd={summary.get('cd', float('nan')):.6f}, "
      f"Cm={summary.get('cm', float('nan')):.6f}"
    )
  print(summary)
  return 0


def _sweep_command(args: argparse.Namespace) -> int:
  try:
    result = run_sweep(args.sweep_yaml, jobs=max(1, int(args.jobs)), backend=args.backend, out_dir=args.out)
  except Exception as exc:  # pragma: no cover - CLI error path
    print(f"Sweep failed: {exc}")
    return 1

  print(f"Sweep completed. Outputs: {result['output_root']}")
  print({"runs": len(result["runs"]), "jobs_requested": result["jobs_requested"]})
  return 0


def _post_command(args: argparse.Namespace) -> int:
  plots = generate_plots(args.results_dir)
  if plots:
    print("Generated plots:")
    for plot in plots:
      print(f" - {plot}")
  else:
    print("No plot inputs found. Expected residuals.csv and/or forces.csv.")
  return 0


def _ui_command() -> int:
  if importlib.util.find_spec("streamlit") is None:
    print("Streamlit is not installed. Install UI extras with: python -m pip install -e python[ui]")
    return 0

  ui_script = Path(__file__).with_name("ui_streamlit.py")
  subprocess.run([sys.executable, "-m", "streamlit", "run", str(ui_script)], check=False)
  return 0


def main(argv: list[str] | None = None) -> None:
  parser = _build_parser()
  args = parser.parse_args(argv)

  if args.version:
    print(f"cfd {__version__}")
    raise SystemExit(0)

  if args.command is None:
    parser.print_help()
    raise SystemExit(0)

  if args.command == "run":
    raise SystemExit(_run_command(args))
  if args.command == "sweep":
    raise SystemExit(_sweep_command(args))
  if args.command == "post":
    raise SystemExit(_post_command(args))
  if args.command == "ui":
    raise SystemExit(_ui_command())

  raise SystemExit(2)


if __name__ == "__main__":
  main()
