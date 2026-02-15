#pragma once

#include "cfd_core/mesh.hpp"
#include "cfd_core/numerics/euler_flux.hpp"

#include <array>
#include <vector>

namespace cfd::core {
enum class LimiterType {
  kMinmod = 0,
};

struct PrimitiveGradients {
  // [cell][var][component], var=(rho,u,v,w,p), component=(dx,dy)
  std::vector<float> values;
};

PrimitiveGradients compute_green_gauss_gradients(const UnstructuredMesh& mesh,
                                                 const std::vector<PrimitiveState>& primitive);
float apply_limiter(LimiterType limiter, float a, float b);

void reconstruct_interior_face_states(const UnstructuredMesh& mesh,
                                      const std::vector<PrimitiveState>& primitive,
                                      const PrimitiveGradients& gradients, int face,
                                      LimiterType limiter, PrimitiveState* left,
                                      PrimitiveState* right);
}  // namespace cfd::core
