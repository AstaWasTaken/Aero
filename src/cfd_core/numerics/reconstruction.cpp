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
constexpr int kMinComponents = 2;
constexpr int kMaxComponents = 3;

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

int resolve_num_components(const UnstructuredMesh& mesh) {
  return std::clamp(mesh.dimension, kMinComponents, kMaxComponents);
}

float& gradient_ref(std::vector<float>& values, const int num_components, const int cell,
                    const int var, const int component) {
  return values[static_cast<std::size_t>(((cell * kNumVars + var) * num_components + component))];
}

float gradient_value(const std::vector<float>& values, const int num_components, const int cell,
                     const int var, const int component) {
  return values[static_cast<std::size_t>(((cell * kNumVars + var) * num_components + component))];
}
}  // namespace

PrimitiveGradients compute_green_gauss_gradients(const UnstructuredMesh& mesh,
                                                 const std::vector<PrimitiveState>& primitive) {
  if (primitive.size() != static_cast<std::size_t>(mesh.num_cells)) {
    throw std::invalid_argument("Primitive field size must match mesh.num_cells.");
  }

  PrimitiveGradients gradients;
  gradients.num_components = resolve_num_components(mesh);
  gradients.values.assign(static_cast<std::size_t>(mesh.num_cells) * kNumVars *
                            gradients.num_components,
                          0.0f);

  for (int face = 0; face < mesh.num_faces; ++face) {
    const int owner = mesh.face_owner[face];
    const int neighbor = mesh.face_neighbor[face];

    for (int var = 0; var < kNumVars; ++var) {
      const float q_owner = primitive_var(primitive[owner], var);
      float q_face = q_owner;
      if (neighbor >= 0) {
        const float q_neighbor = primitive_var(primitive[neighbor], var);
        q_face = 0.5f * (q_owner + q_neighbor);
      }

      for (int component = 0; component < gradients.num_components; ++component) {
        const float nA = mesh.face_normal[3 * face + component] * mesh.face_area[face];
        gradient_ref(gradients.values, gradients.num_components, owner, var, component) +=
          q_face * nA;
        if (neighbor >= 0) {
          gradient_ref(gradients.values, gradients.num_components, neighbor, var, component) -=
            q_face * nA;
        }
      }
    }
  }

  for (int cell = 0; cell < mesh.num_cells; ++cell) {
    const float inv_vol = 1.0f / std::max(mesh.cell_volume[cell], 1.0e-12f);
    for (int var = 0; var < kNumVars; ++var) {
      for (int component = 0; component < gradients.num_components; ++component) {
        gradient_ref(gradients.values, gradients.num_components, cell, var, component) *= inv_vol;
      }
    }
  }

  return gradients;
}

float apply_limiter(const LimiterType limiter, const float a, const float b) {
  if (limiter == LimiterType::kVenkat) {
    if (a * b <= 0.0f) {
      return 0.0f;
    }
    const float eps2 = 1.0e-12f * (1.0f + a * a + b * b);
    const float numerator = (b * b + 2.0f * a * b + eps2) * a;
    const float denominator = a * a + 2.0f * b * b + a * b + eps2;
    if (std::abs(denominator) <= 1.0e-20f) {
      return 0.0f;
    }
    const float limited = numerator / denominator;
    if (!std::isfinite(limited)) {
      return 0.0f;
    }
    if (a > 0.0f) {
      return std::clamp(limited, 0.0f, a);
    }
    return std::clamp(limited, a, 0.0f);
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

  const int num_components = std::clamp(gradients.num_components, kMinComponents, kMaxComponents);

  const std::array<float, 3> face_center = {
    mesh.face_center[3 * face + 0],
    mesh.face_center[3 * face + 1],
    mesh.face_center[3 * face + 2],
  };
  const std::array<float, 3> owner_center = {
    mesh.cell_center[3 * owner + 0],
    mesh.cell_center[3 * owner + 1],
    mesh.cell_center[3 * owner + 2],
  };
  const std::array<float, 3> neighbor_center = {
    mesh.cell_center[3 * neighbor + 0],
    mesh.cell_center[3 * neighbor + 1],
    mesh.cell_center[3 * neighbor + 2],
  };

  const std::array<float, 3> d_owner = {
    face_center[0] - owner_center[0],
    face_center[1] - owner_center[1],
    face_center[2] - owner_center[2],
  };
  const std::array<float, 3> d_neighbor = {
    face_center[0] - neighbor_center[0],
    face_center[1] - neighbor_center[1],
    face_center[2] - neighbor_center[2],
  };

  *left = primitive[owner];
  *right = primitive[neighbor];

  for (int var = 0; var < kNumVars; ++var) {
    const float qo = primitive_var(primitive[owner], var);
    const float qn = primitive_var(primitive[neighbor], var);

    const float delta_cell = qn - qo;
    float delta_owner = 0.0f;
    float delta_neighbor = 0.0f;
    for (int component = 0; component < num_components; ++component) {
      delta_owner +=
        gradient_value(gradients.values, num_components, owner, var, component) * d_owner[component];
      delta_neighbor += gradient_value(gradients.values, num_components, neighbor, var, component) *
                        d_neighbor[component];
    }

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
