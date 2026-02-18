#!/usr/bin/env bash
set -euo pipefail

python -m pip install -e python
cfd run cases/naca0012_euler_2d.yaml --backend cpu --out results/regression_demo
pytest -q tests/python
