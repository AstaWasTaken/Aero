"""Minimal Streamlit UI for AeroCFD scaffold workflows."""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd
import streamlit as st

from cfd import run_case_native
from cfd.case import load_case, resolve_case, write_resolved_case


def main() -> None:
    st.set_page_config(page_title="AeroCFD UI", layout="wide")
    st.title("AeroCFD UI (stub)")

    uploaded = st.file_uploader("Upload a case YAML", type=["yaml", "yml"])
    backend = st.selectbox("Backend", ["cpu", "cuda"], index=0)
    out_dir = st.text_input("Output directory", value="results/ui_demo")

    if st.button("Run Case"):
        if uploaded is None:
            st.warning("Upload a YAML case file before running.")
        else:
            with TemporaryDirectory() as tmpdir:
                case_path = Path(tmpdir) / "uploaded_case.yaml"
                case_path.write_bytes(uploaded.getvalue())

                resolved = resolve_case(load_case(case_path), backend=backend)
                out_path = Path(out_dir)
                out_path.mkdir(parents=True, exist_ok=True)
                write_resolved_case(resolved, out_path)

                summary = run_case_native(str(case_path), str(out_path), backend)
                st.success(f"Run completed (stub). Outputs in: {out_path}")
                st.json(summary)

    st.subheader("Placeholder charts")
    residuals_path = Path(out_dir) / "residuals.csv"
    forces_path = Path(out_dir) / "forces.csv"

    if residuals_path.exists():
        st.line_chart(pd.read_csv(residuals_path).set_index("iter"))
    else:
        st.info("residuals.csv not found yet.")

    if forces_path.exists():
        st.line_chart(pd.read_csv(forces_path).set_index("iter"))
    else:
        st.info("forces.csv not found yet.")


if __name__ == "__main__":
    main()