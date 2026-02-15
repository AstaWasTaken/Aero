"""Python user layer for the AeroCFD scaffold.

The heavy compute path is owned by the `cfd_core` pybind11 module.
"""

from __future__ import annotations

import os
from pathlib import Path
import sys

__version__ = "0.1.0"

if sys.platform == "win32" and hasattr(os, "add_dll_directory"):
    dll_candidates: list[str] = []
    env_dirs = os.environ.get("CFD_DLL_DIRS", "").strip()
    if env_dirs:
        dll_candidates.extend([p for p in env_dirs.split(";") if p])
    dll_candidates.extend(
        [
            r"C:\Strawberry\c\bin",
            r"C:\msys64\mingw64\bin",
        ]
    )

    seen: set[str] = set()
    for candidate in dll_candidates:
        directory = str(Path(candidate))
        if directory in seen:
            continue
        seen.add(directory)
        if Path(directory).is_dir():
            try:
                os.add_dll_directory(directory)
            except OSError:
                pass

try:
    import cfd_core as _core
except ImportError as exc:
    raise ImportError(
        "Failed to import native module `cfd_core`. "
        "Install in editable mode with `python -m pip install -e python`, and on Windows "
        "set CFD_DLL_DIRS to your MinGW runtime DLL folder (for example C:\\Strawberry\\c\\bin)."
    ) from exc

version = _core.version
hello = _core.hello
run_case_native = _core.run_case
cuda_available = _core.cuda_available
core_module_path = str(getattr(_core, "__file__", ""))
native_core_loaded = core_module_path.endswith((".pyd", ".so", ".dylib"))

__all__ = [
    "__version__",
    "version",
    "hello",
    "run_case_native",
    "cuda_available",
    "core_module_path",
    "native_core_loaded",
]
