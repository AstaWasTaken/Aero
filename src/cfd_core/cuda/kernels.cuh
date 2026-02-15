#pragma once

#include <string>

namespace cfd::cuda_backend {
bool launch_scalar_face_kernel(int num_faces, const int* d_face_owner, const int* d_face_neighbor,
                               const float* d_face_normal, const float* d_face_area,
                               const float* d_phi, float inflow_phi, float u_x, float u_y,
                               float u_z, float* d_residual, std::string* error_message);
}  // namespace cfd::cuda_backend
