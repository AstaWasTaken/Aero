import cfd
import cfd_core
import pytest
from cfd.cli import main as cfd_main
from pathlib import Path


def test_import_and_version() -> None:
    assert isinstance(cfd.__version__, str)
    assert cfd.__version__
    assert isinstance(cfd_core.version(), str)


def test_hello() -> None:
    assert cfd_core.hello() == "cfd_core bindings ok"


def test_cli_run_creates_vtu(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    case_path = repo_root / "cases" / "scalar_advect_demo.yaml"
    out_dir = tmp_path / "scalar_demo"

    with pytest.raises(SystemExit) as raised:
        cfd_main(["run", str(case_path), "--backend", "cpu", "--out", str(out_dir)])

    assert raised.value.code == 0
    assert (out_dir / "field_0000.vtu").exists()
    assert (out_dir / "residuals.csv").exists()
