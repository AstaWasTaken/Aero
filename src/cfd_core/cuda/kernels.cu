#include "kernels.cuh"

#include <cuda_runtime.h>

#include <string>

namespace {
__global__ void scalar_face_kernel(const int num_faces, const int* face_owner,
                                   const int* face_neighbor, const float* face_normal,
                                   const float* face_area, const float* phi,
                                   const float inflow_phi, const float u_x, const float u_y,
                                   const float u_z, float* residual) {
  const int face = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (face >= num_faces) {
    return;
  }

  const int owner = face_owner[face];
  const int neighbor = face_neighbor[face];

  const float nx = face_normal[3 * face + 0];
  const float ny = face_normal[3 * face + 1];
  const float nz = face_normal[3 * face + 2];
  const float un = (u_x * nx + u_y * ny + u_z * nz) * face_area[face];

  float upwind_phi = phi[owner];
  if (neighbor >= 0) {
    upwind_phi = (un >= 0.0f) ? phi[owner] : phi[neighbor];
  } else if (un < 0.0f) {
    upwind_phi = inflow_phi;
  }

  const float flux = un * upwind_phi;
  atomicAdd(&residual[owner], -flux);
  if (neighbor >= 0) {
    atomicAdd(&residual[neighbor], flux);
  }
}
}  // namespace

namespace cfd::cuda_backend {
bool launch_scalar_face_kernel(const int num_faces, const int* d_face_owner,
                               const int* d_face_neighbor, const float* d_face_normal,
                               const float* d_face_area, const float* d_phi,
                               const float inflow_phi, const float u_x, const float u_y,
                               const float u_z, float* d_residual,
                               std::string* error_message) {
  constexpr int threads_per_block = 256;
  const int blocks = (num_faces + threads_per_block - 1) / threads_per_block;
  scalar_face_kernel<<<blocks, threads_per_block>>>(
    num_faces, d_face_owner, d_face_neighbor, d_face_normal, d_face_area, d_phi, inflow_phi, u_x,
    u_y, u_z, d_residual);

  const cudaError_t launch_status = cudaGetLastError();
  if (launch_status != cudaSuccess) {
    if (error_message != nullptr) {
      *error_message = std::string("kernel launch failed: ") + cudaGetErrorString(launch_status);
    }
    return false;
  }

  const cudaError_t sync_status = cudaDeviceSynchronize();
  if (sync_status != cudaSuccess) {
    if (error_message != nullptr) {
      *error_message = std::string("kernel synchronize failed: ") + cudaGetErrorString(sync_status);
    }
    return false;
  }

  return true;
}
}  // namespace cfd::cuda_backend
