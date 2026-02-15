"""Python user layer for the AeroCFD scaffold.

The heavy compute path is owned by the `cfd_core` pybind11 module.
"""

from __future__ import annotations

__version__ = "0.1.0"

try:
    import cfd_core as _core
except ImportError as exc:
    raise ImportError(
        "Failed to import native module `cfd_core`. "
        "Install in editable mode with `python -m pip install -e python`."
    ) from exc

version = _core.version
hello = _core.hello
run_case_native = _core.run_case
cuda_available = _core.cuda_available

__all__ = [
    "__version__",
    "version",
    "hello",
    "run_case_native",
    "cuda_available",
]