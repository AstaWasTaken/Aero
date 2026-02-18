#pragma once

#include "cfd_core/mesh.hpp"

#include <array>
#include <vector>

namespace cfd::core {
enum class EulerBoundaryConditionType : int {
  kInterior = 0,
  kFarfield = 1,
  kSlipWall = 2,
};

struct FaceNormalQuality {
  int normalized_count = 0;
  int invalid_count = 0;
  float max_norm_deviation = 0.0f;
};

std::array<float, 3> read_face_unit_normal(const UnstructuredMesh& mesh, int face);
float read_face_measure(const UnstructuredMesh& mesh, int face);
float read_cell_volume(const UnstructuredMesh& mesh, int cell);

FaceNormalQuality normalize_face_normals(UnstructuredMesh* mesh, float tolerance = 1.0e-5f);
std::vector<EulerBoundaryConditionType> build_euler_boundary_types(const UnstructuredMesh& mesh);
std::vector<int> collect_boundary_faces(const UnstructuredMesh& mesh,
                                        const std::vector<EulerBoundaryConditionType>& boundary_type,
                                        EulerBoundaryConditionType requested_type);
}  // namespace cfd::core
