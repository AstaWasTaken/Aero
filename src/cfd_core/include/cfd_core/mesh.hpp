#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace cfd::core {
struct BoundaryPatchRange {
  std::string name;
  int start_face = 0;
  int face_count = 0;
};

struct UnstructuredMesh {
  int dimension = 2;
  int num_cells = 0;
  int num_faces = 0;

  // Face data: normals are unit normals pointing out of owner.
  std::vector<int> face_owner;
  std::vector<int> face_neighbor;
  std::vector<int> face_vertices;  // v0,v1 per face
  std::vector<float> face_normal;  // xyz per face
  std::vector<float> face_center;  // xyz per face
  std::vector<float> face_area;
  std::vector<float> cell_volume;
  std::vector<float> cell_center;  // xyz per cell

  std::vector<BoundaryPatchRange> boundary_patches;

  // Minimal geometry for VTU output.
  std::vector<float> points;  // xyz per point
  std::vector<int> cell_connectivity;
  std::vector<int> cell_offsets;
  std::vector<std::uint8_t> cell_types;
};

UnstructuredMesh make_demo_tri_mesh_2x2();
}  // namespace cfd::core
