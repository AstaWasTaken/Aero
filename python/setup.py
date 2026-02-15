from __future__ import annotations

import sys
from pathlib import Path
import shutil

from setuptools import Command, setup


class MinimalBdistWheel(Command):
    """Lightweight wheel command for offline editable installs in this scaffold."""

    description = "minimal bdist_wheel command"
    user_options = [
        ("bdist-dir=", "b", "temporary build directory"),
        ("dist-dir=", "d", "directory for final built distributions"),
        ("plat-name=", "p", "platform name"),
        ("python-tag=", None, "python tag"),
        ("abi-tag=", None, "abi tag"),
        ("universal", None, "build universal wheel"),
    ]
    boolean_options = ["universal"]

    def initialize_options(self) -> None:
        self.bdist_dir = None
        self.dist_dir = None
        self.plat_name = None
        self.python_tag = None
        self.abi_tag = None
        self.universal = False

    def finalize_options(self) -> None:
        if self.dist_dir is None:
            self.dist_dir = "dist"

    def run(self) -> None:
        # Editable installs use `write_wheelfile` and `get_tag` directly.
        return None

    def get_tag(self) -> tuple[str, str, str]:
        py_tag = self.python_tag or f"py{sys.version_info.major}"
        abi_tag = self.abi_tag or "none"
        plat_tag = self.plat_name or "any"
        return py_tag, abi_tag, plat_tag

    def write_wheelfile(self, wheelfile_base: str, generator: str = "setuptools") -> None:
        wheel_path = Path(wheelfile_base) / "WHEEL"
        py_tag, abi_tag, plat_tag = self.get_tag()
        wheel_path.write_text(
            "\n".join(
                [
                    "Wheel-Version: 1.0",
                    f"Generator: {generator}",
                    "Root-Is-Purelib: true",
                    f"Tag: {py_tag}-{abi_tag}-{plat_tag}",
                    "",
                ]
            ),
            encoding="utf-8",
        )

    @staticmethod
    def egg2dist(egg_info_dir: str, dist_info_dir: str) -> None:
        src = Path(egg_info_dir)
        dst = Path(dist_info_dir)
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)

        pkg_info = dst / "PKG-INFO"
        metadata = dst / "METADATA"
        if pkg_info.exists():
            pkg_info.replace(metadata)


if __name__ == "__main__":
    setup(cmdclass={"bdist_wheel": MinimalBdistWheel})
