#!/usr/bin/env bash
set -euo pipefail

if command -v clang-format >/dev/null 2>&1; then
  find src tests -type f \( -name "*.cpp" -o -name "*.hpp" -o -name "*.cu" -o -name "*.cuh" \) -print0 | \
    xargs -0 clang-format -i
else
  echo "clang-format not found; skipping C++ formatting"
fi

ruff check python tests --fix
black python tests