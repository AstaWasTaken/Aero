#include "cfd_core/solvers/euler/mesh_geometry.hpp"

#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

namespace cfd::core {
namespace {
constexpr float kMinNormalNorm = 1.0e-12f;

std::vector<std::string> build_face_patch_name(const UnstructuredMesh& mesh) {
  std::vector<std::string> patch_name(static_cast<std::size_t>(mesh.num_faces), std::string());
  for (const auto& patch : mesh.boundary_patches) {
    for (int i = 0; i < patch.face_count; ++i) {
      const int face = patch.start_face + i;
      patch_name[static_cast<std::size_t>(face)] = patch.name;
    }
  }
  return patch_name;
}

bool is_wall_patch(const std::string& patch_name) {
  return patch_name == "wall";
}

bool is_farfield_patch(const std::string& patch_name) {
  return patch_name.empty() || patch_name == "farfield" || patch_name == "boundary" ||
         patch_name == "wake" || patch_name == "cut";
}
}  // namespace

std::array<float, 3> read_face_unit_normal(const UnstructuredMesh& mesh, const int face) {
  return {
    mesh.face_normal[3 * face + 0],
    mesh.face_normal[3 * face + 1],
    mesh.face_normal[3 * face + 2],
  };
}

float read_face_measure(const UnstructuredMesh& mesh, const int face) {
  return mesh.face_area[face];
}

float read_cell_volume(const UnstructuredMesh& mesh, const int cell) {
  return std::max(mesh.cell_volume[cell], 1.0e-12f);
}

FaceNormalQuality normalize_face_normals(UnstructuredMesh* mesh, const float tolerance) {
  FaceNormalQuality quality;
  if (mesh == nullptr) {
    return quality;
  }

  for (int face = 0; face < mesh->num_faces; ++face) {
    const float nx = mesh->face_normal[3 * face + 0];
    const float ny = mesh->face_normal[3 * face + 1];
    const float nz = mesh->face_normal[3 * face + 2];
    const float norm = std::sqrt(nx * nx + ny * ny + nz * nz);
    if (!std::isfinite(norm) || norm < kMinNormalNorm) {
      ++quality.invalid_count;
      mesh->face_normal[3 * face + 0] = 1.0f;
      mesh->face_normal[3 * face + 1] = 0.0f;
      mesh->face_normal[3 * face + 2] = 0.0f;
      continue;
    }

    const float deviation = std::abs(norm - 1.0f);
    quality.max_norm_deviation = std::max(quality.max_norm_deviation, deviation);
    if (deviation > tolerance) {
      const float inv_norm = 1.0f / norm;
      mesh->face_normal[3 * face + 0] = nx * inv_norm;
      mesh->face_normal[3 * face + 1] = ny * inv_norm;
      mesh->face_normal[3 * face + 2] = nz * inv_norm;
      ++quality.normalized_count;
    }
  }

  return quality;
}

std::vector<EulerBoundaryConditionType> build_euler_boundary_types(const UnstructuredMesh& mesh) {
  std::vector<EulerBoundaryConditionType> boundary_type(
    static_cast<std::size_t>(mesh.num_faces), EulerBoundaryConditionType::kInterior);
  const std::vector<std::string> face_patch_name = build_face_patch_name(mesh);

  for (int face = 0; face < mesh.num_faces; ++face) {
    if (mesh.face_neighbor[face] >= 0) {
      continue;
    }
    const std::string& patch_name = face_patch_name[static_cast<std::size_t>(face)];
    if (is_wall_patch(patch_name)) {
      boundary_type[static_cast<std::size_t>(face)] = EulerBoundaryConditionType::kSlipWall;
      continue;
    }
    if (is_farfield_patch(patch_name)) {
      boundary_type[static_cast<std::size_t>(face)] = EulerBoundaryConditionType::kFarfield;
      continue;
    }
    boundary_type[static_cast<std::size_t>(face)] = EulerBoundaryConditionType::kFarfield;
  }

  return boundary_type;
}

std::vector<int> collect_boundary_faces(const UnstructuredMesh& mesh,
                                        const std::vector<EulerBoundaryConditionType>& boundary_type,
                                        const EulerBoundaryConditionType requested_type) {
  std::vector<int> faces;
  faces.reserve(static_cast<std::size_t>(mesh.num_faces) / 4);
  for (int face = 0; face < mesh.num_faces; ++face) {
    if (mesh.face_neighbor[face] >= 0) {
      continue;
    }
    if (boundary_type[static_cast<std::size_t>(face)] != requested_type) {
      continue;
    }
    faces.push_back(face);
  }
  return faces;
}
}  // namespace cfd::core
