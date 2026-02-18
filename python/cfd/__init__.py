"""Python user layer for the AeroCFD scaffold.

The heavy compute path is owned by the `cfd_core` pybind11 module.
"""

from __future__ import annotations

import importlib.machinery
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


def _find_native_core_module_dirs() -> list[Path]:
    module_dirs: list[Path] = []
    seen: set[str] = set()

    env_dirs = os.environ.get("CFD_CORE_MODULE_DIRS", "").strip()
    if env_dirs:
        for entry in env_dirs.split(os.pathsep):
            if not entry:
                continue
            path = Path(entry).resolve()
            key = str(path)
            if key not in seen and path.is_dir():
                seen.add(key)
                module_dirs.append(path)

    python_root = Path(__file__).resolve().parents[1]
    repo_root = python_root.parent
    build_root = repo_root / "build"
    if build_root.is_dir():
        build_dirs = sorted(
            (path for path in build_root.iterdir() if path.is_dir()),
            key=lambda path: (0 if "cuda" in path.name.lower() else 1, path.name.lower()),
        )
        for build_dir in build_dirs:
            bindings_dir = build_dir / "src" / "cfd_core" / "bindings"
            if not bindings_dir.is_dir():
                continue
            has_native_module = any(
                bindings_dir.glob(f"cfd_core*{suffix}")
                for suffix in importlib.machinery.EXTENSION_SUFFIXES
            )
            if not has_native_module:
                continue
            key = str(bindings_dir.resolve())
            if key not in seen:
                seen.add(key)
                module_dirs.append(bindings_dir.resolve())

    return module_dirs


for module_dir in reversed(_find_native_core_module_dirs()):
    module_dir_str = str(module_dir)
    if module_dir_str not in sys.path:
        sys.path.insert(0, module_dir_str)

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
