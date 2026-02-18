# Docs Index

- `developer_guide.md`: practical build/test/run workflow and code map
- `validation_plan.md`: current regression gate and baseline policy
- `low_mach_preconditioning.md`: low-Mach controls and recommended usage
- `project_outline.md`: milestone status and roadmap

Quick start:

```bash
cmake -S . -B build/cpu -DCFD_ENABLE_CUDA=OFF -DCFD_BUILD_TESTS=ON -DCFD_BUILD_PYTHON=ON
cmake --build build/cpu --config Release
ctest --test-dir build/cpu --output-on-failure
```
