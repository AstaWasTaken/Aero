#!/usr/bin/env bash
set -euo pipefail

cmake -S . -B build/cpu -DCFD_ENABLE_CUDA=OFF -DCFD_BUILD_PYTHON=ON -DCFD_BUILD_TESTS=ON
cmake --build build/cpu --config Release
ctest --test-dir build/cpu --output-on-failure