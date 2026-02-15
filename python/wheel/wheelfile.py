"""Tiny subset of wheel.wheelfile used by setuptools editable_wheel."""

from __future__ import annotations

from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile


class WheelFile(ZipFile):
    def __init__(self, file: str | Path, mode: str = "w") -> None:
        super().__init__(file=file, mode=mode, compression=ZIP_DEFLATED)
        self._record_written = False

    def write_files(self, base_dir: str | Path) -> None:
        base_path = Path(base_dir)
        for path in sorted(base_path.rglob("*")):
            if path.is_file():
                arcname = path.relative_to(base_path).as_posix()
                self.write(path, arcname=arcname)

    def _record_name(self) -> str:
        for name in self.namelist():
            marker = ".dist-info/"
            if marker in name:
                return f"{name.split(marker, 1)[0]}{marker}RECORD"
        return "RECORD"

    def close(self) -> None:
        if self.fp is not None and not self._record_written and self.mode in {"w", "x", "a"}:
            record_name = self._record_name()
            if record_name not in self.namelist():
                lines = [f"{name},,\n" for name in self.namelist() if name != record_name]
                lines.append(f"{record_name},,\n")
                super().writestr(record_name, "".join(lines))
            self._record_written = True

        super().close()
