#!/usr/bin/env bash
set -euo pipefail

cmake -S . -B build/cuda -DCFD_ENABLE_CUDA=ON -DCFD_BUILD_PYTHON=ON -DCFD_BUILD_TESTS=ON
cmake --build build/cuda --config Release
ctest --test-dir build/cuda --output-on-failure