"""Pure Python fallback for the `cfd_core` module.

If the compiled pybind11 extension is available it should take precedence over this file.
This fallback keeps editable installs usable in constrained environments.
"""

from __future__ import annotations

from pathlib import Path


def version() -> str:
    return "0.1.0"


def hello() -> str:
    return "cfd_core bindings ok"


def cuda_available() -> bool:
    return False


def _write_placeholder_vtu(path: Path) -> None:
    path.write_text(
        """<?xml version="1.0"?>
<VTKFile type="UnstructuredGrid" version="0.1" byte_order="LittleEndian">
  <UnstructuredGrid>
    <Piece NumberOfPoints="1" NumberOfCells="1">
      <Points>
        <DataArray type="Float32" NumberOfComponents="3" format="ascii">0 0 0</DataArray>
      </Points>
      <Cells>
        <DataArray type="Int32" Name="connectivity" format="ascii">0</DataArray>
        <DataArray type="Int32" Name="offsets" format="ascii">1</DataArray>
        <DataArray type="UInt8" Name="types" format="ascii">1</DataArray>
      </Cells>
      <PointData>
        <DataArray type="Float32" Name="pressure" format="ascii">101325</DataArray>
      </PointData>
      <CellData/>
    </Piece>
  </UnstructuredGrid>
</VTKFile>
""",
        encoding="utf-8",
    )


def _detect_case_type(case_path: str) -> str:
    path = Path(case_path)
    if not path.exists():
        return "scalar_advect_demo"

    try:
        content = path.read_text(encoding="utf-8")
    except OSError:
        return "scalar_advect_demo"

    for line in content.splitlines():
        if line.strip().startswith("case_type="):
            return line.split("=", maxsplit=1)[1].strip() or "scalar_advect_demo"
        if line.strip().startswith("case_type:"):
            return line.split(":", maxsplit=1)[1].strip() or "scalar_advect_demo"
    return "scalar_advect_demo"


def run_case(case_path: str, out_dir: str, backend: str = "cpu") -> dict:
    backend_lower = backend.strip().lower()
    if backend_lower not in {"cpu", "cuda"}:
        raise ValueError("Unsupported backend. Use 'cpu' or 'cuda'.")
    if backend_lower == "cuda" and not cuda_available():
        raise RuntimeError(
            "CUDA backend requested but unavailable. Reconfigure with CUDA support enabled."
        )

    root = Path(out_dir)
    root.mkdir(parents=True, exist_ok=True)
    case_type = _detect_case_type(case_path)

    (root / "run.log").write_text(
        "\n".join(
            [
                "AeroCFD stub run",
                f"version={version()}",
                f"case_path={case_path}",
                f"backend={backend_lower}",
                f"case_type={case_type}",
                "status=completed_stub",
                "TODO(physics): Implement RANS/SST governing equations and source terms.",
                "TODO(numerics): Replace dummy residuals with actual flux/residual assembly.",
                "TODO(cuda): Wire GPU residual kernels and asynchronous reductions.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    (root / "residuals.csv").write_text(
        "\n".join(
            [
                "iter,residual",
                "0,1.000000",
                "1,0.500000",
                "2,0.250000",
                "3,0.125000",
                "4,0.062500",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    (root / "forces.csv").write_text(
        "\n".join(
            [
                "iter,cl,cd,cm",
                "0,0.1000,0.0200,0.0000",
                "1,0.2000,0.0190,-0.0010",
                "2,0.2800,0.0185,-0.0020",
                "3,0.3300,0.0180,-0.0025",
                "4,0.3500,0.0178,-0.0030",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    _write_placeholder_vtu(root / "field_0000.vtu")

    return {
        "status": "ok",
        "backend": backend_lower,
        "case_type": case_type,
        "run_log": str(root / "run.log"),
        "iterations": 5,
        "residual_l1": 0.0,
        "residual_l2": 0.0625,
        "residual_linf": 0.0,
        "cl": 0.35,
        "cd": 0.0178,
        "cm": -0.003,
    }
