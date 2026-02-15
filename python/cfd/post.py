"""Post-processing utilities for placeholder CFD outputs."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _load_csv(csv_path: Path) -> np.ndarray | None:
    if not csv_path.exists():
        return None

    data = np.genfromtxt(csv_path, delimiter=",", names=True)
    if data.size == 0:
        return None
    return data


def generate_plots(results_dir: str | Path) -> list[Path]:
    root = Path(results_dir)
    root.mkdir(parents=True, exist_ok=True)

    output_files: list[Path] = []

    residuals = _load_csv(root / "residuals.csv")
    if residuals is not None:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(residuals["iter"], residuals["residual"], marker="o")
        ax.set_yscale("log")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Residual")
        ax.set_title("Residual History")
        fig.tight_layout()
        residual_plot = root / "residuals.png"
        fig.savefig(residual_plot, dpi=160)
        plt.close(fig)
        output_files.append(residual_plot)

    forces = _load_csv(root / "forces.csv")
    if forces is not None:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(forces["iter"], forces["cl"], label="CL")
        ax.plot(forces["iter"], forces["cd"], label="CD")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Coefficient")
        ax.set_title("Force Coefficients")
        ax.legend(loc="best")
        fig.tight_layout()
        force_plot = root / "forces.png"
        fig.savefig(force_plot, dpi=160)
        plt.close(fig)
        output_files.append(force_plot)

    return output_files