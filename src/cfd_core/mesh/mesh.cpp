#include "cfd_core/mesh.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <map>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace cfd::core {
namespace {
struct FaceBuildData {
  int owner = -1;
  int neighbor = -1;
  int v0 = -1;
  int v1 = -1;
  float nx = 0.0f;
  float ny = 0.0f;
  float nz = 0.0f;
  float area = 0.0f;
};

constexpr float kEps = 1.0e-6f;

std::string classify_boundary_patch(const float x_mid, const float y_mid) {
  if (std::abs(x_mid - 0.0f) < kEps) {
    return "left";
  }
  if (std::abs(x_mid - 2.0f) < kEps) {
    return "right";
  }
  if (std::abs(y_mid - 0.0f) < kEps) {
    return "bottom";
  }
  if (std::abs(y_mid - 2.0f) < kEps) {
    return "top";
  }
  return "boundary";
}
}  // namespace

UnstructuredMesh make_demo_tri_mesh_2x2() {
  UnstructuredMesh mesh;
  mesh.dimension = 2;

  auto point_id = [](const int i, const int j) { return j * 3 + i; };

  std::vector<std::array<float, 3>> points;
  points.reserve(9);
  for (int j = 0; j < 3; ++j) {
    for (int i = 0; i < 3; ++i) {
      points.push_back({static_cast<float>(i), static_cast<float>(j), 0.0f});
    }
  }

  std::vector<std::array<int, 3>> cells;
  cells.reserve(8);
  for (int j = 0; j < 2; ++j) {
    for (int i = 0; i < 2; ++i) {
      const int p00 = point_id(i, j);
      const int p10 = point_id(i + 1, j);
      const int p01 = point_id(i, j + 1);
      const int p11 = point_id(i + 1, j + 1);
      cells.push_back({p00, p10, p11});
      cells.push_back({p00, p11, p01});
    }
  }

  mesh.num_cells = static_cast<int>(cells.size());
  mesh.cell_volume.resize(mesh.num_cells, 0.0f);
  mesh.cell_center.resize(static_cast<std::size_t>(mesh.num_cells) * 3, 0.0f);

  for (int c = 0; c < mesh.num_cells; ++c) {
    const auto& tri = cells[c];
    const auto& a = points[tri[0]];
    const auto& b = points[tri[1]];
    const auto& d = points[tri[2]];

    const float abx = b[0] - a[0];
    const float aby = b[1] - a[1];
    const float adx = d[0] - a[0];
    const float ady = d[1] - a[1];
    const float signed_area2 = abx * ady - aby * adx;
    const float area = 0.5f * std::abs(signed_area2);
    if (area < kEps) {
      throw std::runtime_error("Degenerate cell detected in demo mesh.");
    }

    mesh.cell_volume[c] = area;
    mesh.cell_center[3 * c + 0] = (a[0] + b[0] + d[0]) / 3.0f;
    mesh.cell_center[3 * c + 1] = (a[1] + b[1] + d[1]) / 3.0f;
    mesh.cell_center[3 * c + 2] = 0.0f;
  }

  std::map<std::pair<int, int>, int> edge_to_face;
  std::vector<FaceBuildData> faces;
  faces.reserve(24);

  for (int c = 0; c < mesh.num_cells; ++c) {
    const auto& tri = cells[c];
    const std::array<std::pair<int, int>, 3> edges = {
      std::pair<int, int>{tri[0], tri[1]},
      std::pair<int, int>{tri[1], tri[2]},
      std::pair<int, int>{tri[2], tri[0]},
    };

    for (const auto& edge : edges) {
      const int v0 = edge.first;
      const int v1 = edge.second;
      const std::pair<int, int> key {std::min(v0, v1), std::max(v0, v1)};

      const auto found = edge_to_face.find(key);
      if (found == edge_to_face.end()) {
        FaceBuildData face;
        face.owner = c;
        face.v0 = v0;
        face.v1 = v1;

        const auto& p0 = points[v0];
        const auto& p1 = points[v1];
        const float dx = p1[0] - p0[0];
        const float dy = p1[1] - p0[1];
        const float length = std::sqrt(dx * dx + dy * dy);
        if (length < kEps) {
          throw std::runtime_error("Degenerate face detected in demo mesh.");
        }

        face.nx = dy / length;
        face.ny = -dx / length;
        face.nz = 0.0f;
        face.area = length;

        const int idx = static_cast<int>(faces.size());
        edge_to_face.emplace(key, idx);
        faces.push_back(face);
      } else {
        faces[found->second].neighbor = c;
      }
    }
  }

  std::vector<int> interior_faces;
  std::vector<int> left_faces;
  std::vector<int> right_faces;
  std::vector<int> bottom_faces;
  std::vector<int> top_faces;
  std::vector<int> other_boundary_faces;
  interior_faces.reserve(faces.size());
  left_faces.reserve(faces.size());
  right_faces.reserve(faces.size());
  bottom_faces.reserve(faces.size());
  top_faces.reserve(faces.size());
  other_boundary_faces.reserve(faces.size());

  for (int f = 0; f < static_cast<int>(faces.size()); ++f) {
    const auto& face = faces[f];
    if (face.neighbor >= 0) {
      interior_faces.push_back(f);
      continue;
    }

    const auto& p0 = points[face.v0];
    const auto& p1 = points[face.v1];
    const float x_mid = 0.5f * (p0[0] + p1[0]);
    const float y_mid = 0.5f * (p0[1] + p1[1]);
    const std::string patch_name = classify_boundary_patch(x_mid, y_mid);
    if (patch_name == "left") {
      left_faces.push_back(f);
    } else if (patch_name == "right") {
      right_faces.push_back(f);
    } else if (patch_name == "bottom") {
      bottom_faces.push_back(f);
    } else if (patch_name == "top") {
      top_faces.push_back(f);
    } else {
      other_boundary_faces.push_back(f);
    }
  }

  std::vector<int> face_order;
  face_order.reserve(faces.size());
  face_order.insert(face_order.end(), interior_faces.begin(), interior_faces.end());
  face_order.insert(face_order.end(), left_faces.begin(), left_faces.end());
  face_order.insert(face_order.end(), right_faces.begin(), right_faces.end());
  face_order.insert(face_order.end(), bottom_faces.begin(), bottom_faces.end());
  face_order.insert(face_order.end(), top_faces.begin(), top_faces.end());
  face_order.insert(face_order.end(), other_boundary_faces.begin(), other_boundary_faces.end());

  mesh.num_faces = static_cast<int>(face_order.size());
  mesh.face_owner.resize(mesh.num_faces);
  mesh.face_neighbor.resize(mesh.num_faces);
  mesh.face_vertices.resize(static_cast<std::size_t>(mesh.num_faces) * 2);
  mesh.face_area.resize(mesh.num_faces);
  mesh.face_normal.resize(static_cast<std::size_t>(mesh.num_faces) * 3);
  mesh.face_center.resize(static_cast<std::size_t>(mesh.num_faces) * 3);

  for (int new_face = 0; new_face < mesh.num_faces; ++new_face) {
    const auto& old_face = faces[face_order[new_face]];
    mesh.face_owner[new_face] = old_face.owner;
    mesh.face_neighbor[new_face] = old_face.neighbor;
    mesh.face_vertices[2 * new_face + 0] = old_face.v0;
    mesh.face_vertices[2 * new_face + 1] = old_face.v1;
    mesh.face_area[new_face] = old_face.area;
    mesh.face_normal[3 * new_face + 0] = old_face.nx;
    mesh.face_normal[3 * new_face + 1] = old_face.ny;
    mesh.face_normal[3 * new_face + 2] = old_face.nz;
    const auto& p0 = points[old_face.v0];
    const auto& p1 = points[old_face.v1];
    mesh.face_center[3 * new_face + 0] = 0.5f * (p0[0] + p1[0]);
    mesh.face_center[3 * new_face + 1] = 0.5f * (p0[1] + p1[1]);
    mesh.face_center[3 * new_face + 2] = 0.0f;
  }

  int patch_start = static_cast<int>(interior_faces.size());
  auto push_patch = [&](const std::string& patch_name, const std::vector<int>& patch_faces) {
    if (patch_faces.empty()) {
      return;
    }
    BoundaryPatchRange patch;
    patch.name = patch_name;
    patch.start_face = patch_start;
    patch.face_count = static_cast<int>(patch_faces.size());
    mesh.boundary_patches.push_back(std::move(patch));
    patch_start += static_cast<int>(patch_faces.size());
  };

  push_patch("left", left_faces);
  push_patch("right", right_faces);
  push_patch("bottom", bottom_faces);
  push_patch("top", top_faces);
  push_patch("boundary", other_boundary_faces);

  mesh.points.reserve(points.size() * 3);
  for (const auto& point : points) {
    mesh.points.push_back(point[0]);
    mesh.points.push_back(point[1]);
    mesh.points.push_back(point[2]);
  }

  mesh.cell_connectivity.reserve(cells.size() * 3);
  mesh.cell_offsets.reserve(cells.size());
  mesh.cell_types.reserve(cells.size());
  int running_offset = 0;
  for (const auto& cell : cells) {
    mesh.cell_connectivity.push_back(cell[0]);
    mesh.cell_connectivity.push_back(cell[1]);
    mesh.cell_connectivity.push_back(cell[2]);
    running_offset += 3;
    mesh.cell_offsets.push_back(running_offset);
    mesh.cell_types.push_back(5);  // VTK_TRIANGLE
  }

  return mesh;
}
}  // namespace cfd::core
