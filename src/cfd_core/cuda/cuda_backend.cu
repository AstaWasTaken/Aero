#include "kernels.cuh"

namespace cfd::cuda_backend {
void warmup_backend() {
  // TODO(cuda): Add stream setup, memory pools, and backend diagnostics.
  launch_noop_kernel();
}
}  // namespace cfd::cuda_backend