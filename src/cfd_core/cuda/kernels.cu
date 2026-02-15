#include "kernels.cuh"

#include <cuda_runtime.h>

namespace {
__global__ void noop_kernel() {
  // TODO(cuda): Replace with real flux/residual kernels.
}
}  // namespace

namespace cfd::cuda_backend {
void launch_noop_kernel() {
  noop_kernel<<<1, 1>>>();
  cudaDeviceSynchronize();
}
}  // namespace cfd::cuda_backend