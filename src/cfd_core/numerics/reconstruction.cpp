#include "cfd_core/numerics/reconstruction.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>

namespace cfd::core {
namespace {
constexpr int kNumVars = 5;
constexpr float kRhoFloor = 1.0e-8f;
constexpr float kPressureFloor = 1.0e-8f;

float primitive_var(const PrimitiveState& primitive, const int var) {
  switch (var) {
    case 0:
      return primitive.rho;
    case 1:
      return primitive.u;
    case 2:
      return primitive.v;
    case 3:
      return primitive.w;
    case 4:
      return primitive.p;
    default:
      return 0.0f;
  }
}

void set_primitive_var(PrimitiveState* primitive, const int var, const float value) {
  if (primitive == nullptr) {
    return;
  }
  switch (var) {
    case 0:
      primitive->rho = value;
      break;
    case 1:
      primitive->u = value;
      break;
    case 2:
      primitive->v = value;
      break;
    case 3:
      primitive->w = value;
      break;
    case 4:
      primitive->p = value;
      break;
    default:
      break;
  }
}

float& gradient_ref(std::vector<float>& values, const int cell, const int var, const int component) {
  return values[static_cast<std::size_t>(((cell * kNumVars + var) * 2 + component))];
}

float gradient_value(const std::vector<float>& values, const int cell, const int var,
                     const int component) {
  return values[static_cast<std::size_t>(((cell * kNumVars + var) * 2 + component))];
}
}  // namespace

PrimitiveGradients compute_green_gauss_gradients(const UnstructuredMesh& mesh,
                                                 const std::vector<PrimitiveState>& primitive) {
  if (primitive.size() != static_cast<std::size_t>(mesh.num_cells)) {
    throw std::invalid_argument("Primitive field size must match mesh.num_cells.");
  }

  PrimitiveGradients gradients;
  gradients.values.assign(static_cast<std::size_t>(mesh.num_cells) * kNumVars * 2, 0.0f);

  for (int face = 0; face < mesh.num_faces; ++face) {
    const int owner = mesh.face_owner[face];
    const int neighbor = mesh.face_neighbor[face];
    const float nxA = mesh.face_normal[3 * face + 0] * mesh.face_area[face];
    const float nyA = mesh.face_normal[3 * face + 1] * mesh.face_area[face];

    for (int var = 0; var < kNumVars; ++var) {
      const float q_owner = primitive_var(primitive[owner], var);
      float q_face = q_owner;
      if (neighbor >= 0) {
        const float q_neighbor = primitive_var(primitive[neighbor], var);
        q_face = 0.5f * (q_owner + q_neighbor);
      }

      gradient_ref(gradients.values, owner, var, 0) += q_face * nxA;
      gradient_ref(gradients.values, owner, var, 1) += q_face * nyA;
      if (neighbor >= 0) {
        gradient_ref(gradients.values, neighbor, var, 0) -= q_face * nxA;
        gradient_ref(gradients.values, neighbor, var, 1) -= q_face * nyA;
      }
    }
  }

  for (int cell = 0; cell < mesh.num_cells; ++cell) {
    const float inv_vol = 1.0f / std::max(mesh.cell_volume[cell], 1.0e-12f);
    for (int var = 0; var < kNumVars; ++var) {
      gradient_ref(gradients.values, cell, var, 0) *= inv_vol;
      gradient_ref(gradients.values, cell, var, 1) *= inv_vol;
    }
  }

  return gradients;
}

float apply_limiter(const LimiterType limiter, const float a, const float b) {
  if (limiter != LimiterType::kMinmod) {
    return a;
  }
  if (a * b <= 0.0f) {
    return 0.0f;
  }
  return (std::abs(a) < std::abs(b)) ? a : b;
}

void reconstruct_interior_face_states(const UnstructuredMesh& mesh,
                                      const std::vector<PrimitiveState>& primitive,
                                      const PrimitiveGradients& gradients, const int face,
                                      const LimiterType limiter, PrimitiveState* left,
                                      PrimitiveState* right) {
  const int owner = mesh.face_owner[face];
  const int neighbor = mesh.face_neighbor[face];
  if (neighbor < 0) {
    throw std::invalid_argument("Boundary face passed into interior reconstruction.");
  }
  if (left == nullptr || right == nullptr) {
    return;
  }

  const std::array<float, 2> face_center = {
    mesh.face_center[3 * face + 0],
    mesh.face_center[3 * face + 1],
  };
  const std::array<float, 2> owner_center = {
    mesh.cell_center[3 * owner + 0],
    mesh.cell_center[3 * owner + 1],
  };
  const std::array<float, 2> neighbor_center = {
    mesh.cell_center[3 * neighbor + 0],
    mesh.cell_center[3 * neighbor + 1],
  };

  const std::array<float, 2> d_owner = {
    face_center[0] - owner_center[0],
    face_center[1] - owner_center[1],
  };
  const std::array<float, 2> d_neighbor = {
    face_center[0] - neighbor_center[0],
    face_center[1] - neighbor_center[1],
  };

  *left = primitive[owner];
  *right = primitive[neighbor];

  for (int var = 0; var < kNumVars; ++var) {
    const float qo = primitive_var(primitive[owner], var);
    const float qn = primitive_var(primitive[neighbor], var);

    const float grad_o_x = gradient_value(gradients.values, owner, var, 0);
    const float grad_o_y = gradient_value(gradients.values, owner, var, 1);
    const float grad_n_x = gradient_value(gradients.values, neighbor, var, 0);
    const float grad_n_y = gradient_value(gradients.values, neighbor, var, 1);

    const float delta_cell = qn - qo;
    const float delta_owner = grad_o_x * d_owner[0] + grad_o_y * d_owner[1];
    const float delta_neighbor = grad_n_x * d_neighbor[0] + grad_n_y * d_neighbor[1];

    const float ql = qo + apply_limiter(limiter, delta_owner, delta_cell);
    const float qr = qn + apply_limiter(limiter, delta_neighbor, -delta_cell);

    set_primitive_var(left, var, ql);
    set_primitive_var(right, var, qr);
  }

  left->rho = std::max(left->rho, kRhoFloor);
  right->rho = std::max(right->rho, kRhoFloor);
  left->p = std::max(left->p, kPressureFloor);
  right->p = std::max(right->p, kPressureFloor);
}
}  // namespace cfd::core
