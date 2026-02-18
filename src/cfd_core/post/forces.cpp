#include "cfd_core/post/forces.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <map>
#include <stdexcept>
#include <utility>
#include <vector>

namespace cfd::core {
namespace {
constexpr float kPi = 3.14159265358979323846f;
constexpr bool kFaceNormalIsUnit = true;

struct EdgeKey {
  int a = -1;
  int b = -1;

  bool operator<(const EdgeKey& other) const {
    if (a != other.a) {
      return a < other.a;
    }
    return b < other.b;
  }
};

EdgeKey make_edge_key(const int v0, const int v1) {
  if (v0 < v1) {
    return {v0, v1};
  }
  return {v1, v0};
}

std::array<float, 2> point_xy(const UnstructuredMesh& mesh, const int vertex) {
  return {mesh.points[3 * vertex + 0], mesh.points[3 * vertex + 1]};
}

std::vector<int> order_wall_faces(const UnstructuredMesh& mesh, const std::vector<int>& faces) {
  if (faces.size() <= 2) {
    return faces;
  }

  std::map<EdgeKey, int> edge_to_face;
  std::map<int, std::vector<int>> vertex_neighbors;
  for (const int face : faces) {
    const int v0 = mesh.face_vertices[2 * face + 0];
    const int v1 = mesh.face_vertices[2 * face + 1];
    edge_to_face.emplace(make_edge_key(v0, v1), face);
    vertex_neighbors[v0].push_back(v1);
    vertex_neighbors[v1].push_back(v0);
  }

  int start_vertex = -1;
  float best_x = -1.0e30f;
  float best_y = -1.0e30f;
  for (const auto& [vertex, _neighbors] : vertex_neighbors) {
    const auto p = point_xy(mesh, vertex);
    if (p[0] > best_x || (std::abs(p[0] - best_x) < 1.0e-8f && p[1] > best_y)) {
      best_x = p[0];
      best_y = p[1];
      start_vertex = vertex;
    }
  }

  if (start_vertex < 0) {
    return faces;
  }

  const auto& first_neighbors = vertex_neighbors[start_vertex];
  if (first_neighbors.size() < 2) {
    return faces;
  }

  int first_next = first_neighbors[0];
  const auto p_start = point_xy(mesh, start_vertex);
  const auto p_n0 = point_xy(mesh, first_neighbors[0]);
  const auto p_n1 = point_xy(mesh, first_neighbors[1]);
  const float y0 = p_n0[1] - p_start[1];
  const float y1 = p_n1[1] - p_start[1];
  if (y1 > y0) {
    first_next = first_neighbors[1];
  }

  std::vector<int> ordered_faces;
  ordered_faces.reserve(faces.size());
  int previous = start_vertex;
  int current = first_next;

  for (std::size_t step = 0; step < faces.size(); ++step) {
    const auto face_it = edge_to_face.find(make_edge_key(previous, current));
    if (face_it == edge_to_face.end()) {
      ordered_faces.clear();
      break;
    }
    ordered_faces.push_back(face_it->second);

    const auto& neighbors = vertex_neighbors[current];
    int next_vertex = -1;
    for (const int candidate : neighbors) {
      if (candidate != previous) {
        next_vertex = candidate;
        break;
      }
    }
    if (next_vertex < 0) {
      ordered_faces.clear();
      break;
    }
    previous = current;
    current = next_vertex;
    if (previous == start_vertex) {
      break;
    }
  }

  if (ordered_faces.size() != faces.size()) {
    std::vector<int> fallback = faces;
    std::sort(fallback.begin(), fallback.end(), [&](const int lhs, const int rhs) {
      const float xl = mesh.face_center[3 * lhs + 0];
      const float yl = mesh.face_center[3 * lhs + 1];
      const float xr = mesh.face_center[3 * rhs + 0];
      const float yr = mesh.face_center[3 * rhs + 1];
      const float al = std::atan2(yl, xl);
      const float ar = std::atan2(yr, xr);
      return al > ar;
    });
    return fallback;
  }

  return ordered_faces;
}

std::array<float, 3> face_nA(const UnstructuredMesh& mesh, const int face) {
  const std::array<float, 3> body_outward_normal = {
    -mesh.face_normal[3 * face + 0],
    -mesh.face_normal[3 * face + 1],
    -mesh.face_normal[3 * face + 2],
  };
  if (kFaceNormalIsUnit) {
    const float area = mesh.face_area[face];
    return {
      body_outward_normal[0] * area,
      body_outward_normal[1] * area,
      body_outward_normal[2] * area,
    };
  }
  return body_outward_normal;
}
}  // namespace

std::vector<int> find_patch_faces(const UnstructuredMesh& mesh, const std::string& patch_name) {
  int total_faces = 0;
  for (const auto& patch : mesh.boundary_patches) {
    if (patch.name != patch_name) {
      continue;
    }
    total_faces += patch.face_count;
  }
  if (total_faces <= 0) {
    return {};
  }

  std::vector<int> faces;
  faces.reserve(static_cast<std::size_t>(total_faces));
  for (const auto& patch : mesh.boundary_patches) {
    if (patch.name != patch_name) {
      continue;
    }
    for (int i = 0; i < patch.face_count; ++i) {
      faces.push_back(patch.start_face + i);
    }
  }
  return faces;
}

std::vector<WallCpSample> extract_wall_cp(const UnstructuredMesh& mesh,
                                          const std::vector<float>& cell_pressure,
                                          const FreestreamReference& reference,
                                          const std::string& wall_patch) {
  if (cell_pressure.size() != static_cast<std::size_t>(mesh.num_cells)) {
    throw std::invalid_argument("Pressure vector size must match mesh.num_cells.");
  }

  std::vector<int> wall_faces = find_patch_faces(mesh, wall_patch);
  wall_faces = order_wall_faces(mesh, wall_faces);

  const float q_inf =
    0.5f * reference.rho_inf * reference.speed_inf * reference.speed_inf + 1.0e-12f;

  std::vector<WallCpSample> samples;
  samples.reserve(wall_faces.size());
  float s = 0.0f;

  for (const int face : wall_faces) {
    const int owner = mesh.face_owner[face];
    const float p = cell_pressure[owner];
    const float cp = (p - reference.p_inf) / q_inf;
    const float x = mesh.face_center[3 * face + 0];
    const float y = mesh.face_center[3 * face + 1];
    samples.push_back({s, x, y, cp});
    s += mesh.face_area[face];
  }

  return samples;
}

PressureForceDiagnostics compute_pressure_force_diagnostics(
  const UnstructuredMesh& mesh, const std::vector<float>& cell_pressure,
  const FreestreamReference& reference, const std::string& wall_patch) {
  if (cell_pressure.size() != static_cast<std::size_t>(mesh.num_cells)) {
    throw std::invalid_argument("Pressure vector size must match mesh.num_cells.");
  }

  const std::vector<int> wall_faces = find_patch_faces(mesh, wall_patch);
  PressureForceDiagnostics diagnostics;
  diagnostics.integrated_face_count = static_cast<int>(wall_faces.size());
  diagnostics.normal_is_unit = kFaceNormalIsUnit;

  for (const int face : wall_faces) {
    if (face < 0 || face >= mesh.num_faces) {
      throw std::runtime_error("Wall face index is out of mesh face bounds.");
    }
    if (mesh.face_neighbor[face] >= 0) {
      throw std::runtime_error("Wall face set contains an interior face.");
    }

    const int owner = mesh.face_owner[face];
    if (owner < 0 || owner >= mesh.num_cells) {
      throw std::runtime_error("Wall face owner is out of mesh cell bounds.");
    }

    const std::array<float, 3> nA = face_nA(mesh, face);
    const float p = cell_pressure[owner];
    const float gauge_p = p - reference.p_inf;

    diagnostics.sum_nA_x += nA[0];
    diagnostics.sum_nA_y += nA[1];
    diagnostics.sum_nA_z += nA[2];
    diagnostics.fx_abs -= p * nA[0];
    diagnostics.fy_abs -= p * nA[1];
    diagnostics.fz_abs -= p * nA[2];
    diagnostics.fx_gauge -= gauge_p * nA[0];
    diagnostics.fy_gauge -= gauge_p * nA[1];
    diagnostics.fz_gauge -= gauge_p * nA[2];
  }

  return diagnostics;
}

ForceCoefficients integrate_pressure_forces(const UnstructuredMesh& mesh,
                                            const std::vector<float>& cell_pressure,
                                            const FreestreamReference& reference,
                                            const std::string& wall_patch,
                                            const bool subtract_p_inf) {
  if (cell_pressure.size() != static_cast<std::size_t>(mesh.num_cells)) {
    throw std::invalid_argument("Pressure vector size must match mesh.num_cells.");
  }

  const std::vector<int> wall_faces = find_patch_faces(mesh, wall_patch);
  const PressureForceDiagnostics diagnostics =
    compute_pressure_force_diagnostics(mesh, cell_pressure, reference, wall_patch);
  const float q_inf =
    0.5f * reference.rho_inf * reference.speed_inf * reference.speed_inf + 1.0e-12f;
  const float alpha = reference.aoa_deg * (kPi / 180.0f);
  const float cos_a = std::cos(alpha);
  const float sin_a = std::sin(alpha);
  const float chord = std::max(reference.chord, 1.0e-8f);
  const float s_ref = chord * 1.0f;

  ForceCoefficients coeffs;
  coeffs.fx = subtract_p_inf ? diagnostics.fx_gauge : diagnostics.fx_abs;
  coeffs.fy = subtract_p_inf ? diagnostics.fy_gauge : diagnostics.fy_abs;
  for (const int face : wall_faces) {
    const int owner = mesh.face_owner[face];
    const float p = cell_pressure[owner];
    const float pressure = subtract_p_inf ? (p - reference.p_inf) : p;
    const std::array<float, 3> nA = face_nA(mesh, face);
    const float dfx = -pressure * nA[0];
    const float dfy = -pressure * nA[1];

    const float x = mesh.face_center[3 * face + 0] - reference.x_ref;
    const float y = mesh.face_center[3 * face + 1] - reference.y_ref;
    coeffs.moment += x * dfy - y * dfx;
  }

  coeffs.drag = coeffs.fx * cos_a + coeffs.fy * sin_a;
  coeffs.lift = -coeffs.fx * sin_a + coeffs.fy * cos_a;

  coeffs.cd = coeffs.drag / (q_inf * s_ref);
  coeffs.cl = coeffs.lift / (q_inf * s_ref);
  coeffs.cm = coeffs.moment / (q_inf * s_ref * chord);
  return coeffs;
}
}  // namespace cfd::core
