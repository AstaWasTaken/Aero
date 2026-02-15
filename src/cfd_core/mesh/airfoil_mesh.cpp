#include "cfd_core/mesh/airfoil_mesh.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <limits>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace cfd::core {
namespace {
constexpr float kPi = 3.14159265358979323846f;
constexpr float kEps = 1.0e-7f;

struct EdgeKey {
  int vmin = -1;
  int vmax = -1;

  bool operator<(const EdgeKey& other) const {
    if (vmin != other.vmin) {
      return vmin < other.vmin;
    }
    return vmax < other.vmax;
  }
};

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

EdgeKey make_edge_key(const int a, const int b) {
  if (a < b) {
    return {a, b};
  }
  return {b, a};
}

float clamp_positive(const float value, const float minimum) {
  return std::max(value, minimum);
}

float signed_area_polygon(const std::vector<std::array<float, 2>>& points) {
  if (points.size() < 3) {
    return 0.0f;
  }

  double area2 = 0.0;
  const std::size_t n = points.size();
  for (std::size_t i = 0; i < n; ++i) {
    const std::size_t j = (i + 1) % n;
    area2 += static_cast<double>(points[i][0]) * static_cast<double>(points[j][1]) -
             static_cast<double>(points[j][0]) * static_cast<double>(points[i][1]);
  }
  return static_cast<float>(0.5 * area2);
}

float distance2(const std::array<float, 2>& a, const std::array<float, 2>& b) {
  const float dx = a[0] - b[0];
  const float dy = a[1] - b[1];
  return dx * dx + dy * dy;
}

std::vector<std::array<float, 2>> remove_duplicate_points(
  const std::vector<std::array<float, 2>>& input) {
  std::vector<std::array<float, 2>> unique_points;
  unique_points.reserve(input.size());
  for (const auto& point : input) {
    if (!unique_points.empty() && distance2(unique_points.back(), point) < 1.0e-12f) {
      continue;
    }
    unique_points.push_back(point);
  }
  if (unique_points.size() >= 2 && distance2(unique_points.front(), unique_points.back()) < 1.0e-12f) {
    unique_points.pop_back();
  }
  return unique_points;
}

std::vector<std::array<float, 2>> resample_closed_curve(
  const std::vector<std::array<float, 2>>& input, const int target_count) {
  if (target_count < 8) {
    throw std::invalid_argument("Target curve resolution must be >= 8.");
  }
  const auto points = remove_duplicate_points(input);
  if (points.size() < 8) {
    throw std::invalid_argument("Airfoil contour has too few points.");
  }

  const std::size_t n = points.size();
  std::vector<float> cumulative(n + 1, 0.0f);
  for (std::size_t i = 0; i < n; ++i) {
    const std::size_t j = (i + 1) % n;
    cumulative[i + 1] = cumulative[i] + std::sqrt(distance2(points[i], points[j]));
  }

  const float total_length = cumulative.back();
  if (total_length < kEps) {
    throw std::invalid_argument("Airfoil contour has near-zero length.");
  }

  std::vector<std::array<float, 2>> sampled;
  sampled.reserve(static_cast<std::size_t>(target_count));

  std::size_t segment = 0;
  for (int k = 0; k < target_count; ++k) {
    const float target_s = total_length * static_cast<float>(k) / static_cast<float>(target_count);
    while (segment + 1 < cumulative.size() && cumulative[segment + 1] < target_s) {
      ++segment;
    }
    const std::size_t i0 = segment % n;
    const std::size_t i1 = (segment + 1) % n;
    const float s0 = cumulative[segment];
    const float s1 = cumulative[segment + 1];
    const float denom = std::max(s1 - s0, kEps);
    const float t = (target_s - s0) / denom;
    sampled.push_back({points[i0][0] + t * (points[i1][0] - points[i0][0]),
                       points[i0][1] + t * (points[i1][1] - points[i0][1])});
  }

  return sampled;
}

std::string extract_naca_digits(const std::string& code) {
  std::string digits;
  digits.reserve(code.size());
  for (char ch : code) {
    if (ch >= '0' && ch <= '9') {
      digits.push_back(ch);
    }
  }
  if (digits.size() < 4) {
    throw std::invalid_argument("NACA code must contain four digits.");
  }
  return digits.substr(0, 4);
}

std::vector<std::array<float, 2>> load_airfoil_coordinate_file(const std::string& path,
                                                               const float chord) {
  std::ifstream in(path);
  if (!in) {
    throw std::runtime_error("Failed to open airfoil coordinate file: " + path);
  }

  std::vector<std::array<float, 2>> points;
  points.reserve(512);

  std::string line;
  while (std::getline(in, line)) {
    std::istringstream iss(line);
    float x = 0.0f;
    float y = 0.0f;
    if (!(iss >> x >> y)) {
      continue;
    }
    points.push_back({x * chord, y * chord});
  }

  points = remove_duplicate_points(points);
  if (points.size() < 16) {
    throw std::invalid_argument("Coordinate file contains too few airfoil points.");
  }

  if (signed_area_polygon(points) < 0.0f) {
    std::reverse(points.begin(), points.end());
  }

  return points;
}

UnstructuredMesh build_triangular_mesh(
  const std::vector<std::array<float, 2>>& points2d, const std::vector<std::array<int, 3>>& tris_in,
  const std::map<EdgeKey, std::string>& boundary_edge_patch) {
  if (points2d.empty() || tris_in.empty()) {
    throw std::invalid_argument("Cannot build mesh from empty point/cell arrays.");
  }

  std::vector<std::array<int, 3>> triangles = tris_in;
  for (auto& tri : triangles) {
    const auto& a = points2d[tri[0]];
    const auto& b = points2d[tri[1]];
    const auto& c = points2d[tri[2]];
    const float area2 = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]);
    if (area2 < 0.0f) {
      std::swap(tri[1], tri[2]);
    }
  }

  UnstructuredMesh mesh;
  mesh.dimension = 2;
  mesh.num_cells = static_cast<int>(triangles.size());
  mesh.cell_volume.resize(mesh.num_cells, 0.0f);
  mesh.cell_center.resize(static_cast<std::size_t>(mesh.num_cells) * 3, 0.0f);

  for (int c = 0; c < mesh.num_cells; ++c) {
    const auto& tri = triangles[static_cast<std::size_t>(c)];
    const auto& a = points2d[tri[0]];
    const auto& b = points2d[tri[1]];
    const auto& d = points2d[tri[2]];

    const float area2 = (b[0] - a[0]) * (d[1] - a[1]) - (b[1] - a[1]) * (d[0] - a[0]);
    const float area = 0.5f * std::abs(area2);
    if (area < kEps) {
      throw std::runtime_error("Degenerate triangle detected while building airfoil mesh.");
    }

    mesh.cell_volume[c] = area;
    mesh.cell_center[3 * c + 0] = (a[0] + b[0] + d[0]) / 3.0f;
    mesh.cell_center[3 * c + 1] = (a[1] + b[1] + d[1]) / 3.0f;
    mesh.cell_center[3 * c + 2] = 0.0f;
  }

  std::map<EdgeKey, int> edge_to_face;
  std::vector<FaceBuildData> faces;
  faces.reserve(static_cast<std::size_t>(mesh.num_cells) * 2);

  for (int c = 0; c < mesh.num_cells; ++c) {
    const auto& tri = triangles[static_cast<std::size_t>(c)];
    const std::array<std::pair<int, int>, 3> edges = {
      std::pair<int, int>{tri[0], tri[1]},
      std::pair<int, int>{tri[1], tri[2]},
      std::pair<int, int>{tri[2], tri[0]},
    };

    for (const auto& edge : edges) {
      const int v0 = edge.first;
      const int v1 = edge.second;
      const EdgeKey key = make_edge_key(v0, v1);
      const auto found = edge_to_face.find(key);
      if (found == edge_to_face.end()) {
        FaceBuildData face;
        face.owner = c;
        face.v0 = v0;
        face.v1 = v1;

        const auto& p0 = points2d[v0];
        const auto& p1 = points2d[v1];
        const float dx = p1[0] - p0[0];
        const float dy = p1[1] - p0[1];
        const float length = std::sqrt(dx * dx + dy * dy);
        if (length < kEps) {
          throw std::runtime_error("Degenerate face detected while building airfoil mesh.");
        }
        face.nx = dy / length;
        face.ny = -dx / length;
        face.nz = 0.0f;
        face.area = length;

        const int face_index = static_cast<int>(faces.size());
        edge_to_face.emplace(key, face_index);
        faces.push_back(face);
      } else {
        faces[found->second].neighbor = c;
      }
    }
  }

  std::vector<int> interior_faces;
  std::map<std::string, std::vector<int>> boundary_faces;
  interior_faces.reserve(faces.size());

  for (int f = 0; f < static_cast<int>(faces.size()); ++f) {
    if (faces[static_cast<std::size_t>(f)].neighbor >= 0) {
      interior_faces.push_back(f);
      continue;
    }
    const EdgeKey key = make_edge_key(faces[static_cast<std::size_t>(f)].v0,
                                      faces[static_cast<std::size_t>(f)].v1);
    const auto it = boundary_edge_patch.find(key);
    const std::string patch_name = (it == boundary_edge_patch.end()) ? "boundary" : it->second;
    boundary_faces[patch_name].push_back(f);
  }

  std::vector<int> face_order;
  face_order.reserve(faces.size());
  face_order.insert(face_order.end(), interior_faces.begin(), interior_faces.end());

  auto append_patch = [&](const std::string& patch_name) {
    const auto it = boundary_faces.find(patch_name);
    if (it == boundary_faces.end()) {
      return;
    }
    face_order.insert(face_order.end(), it->second.begin(), it->second.end());
  };

  append_patch("wall");
  append_patch("farfield");
  append_patch("wake");
  append_patch("cut");

  for (const auto& [patch_name, patch_faces] : boundary_faces) {
    if (patch_name == "wall" || patch_name == "farfield" || patch_name == "wake" ||
        patch_name == "cut") {
      continue;
    }
    face_order.insert(face_order.end(), patch_faces.begin(), patch_faces.end());
  }

  mesh.num_faces = static_cast<int>(face_order.size());
  mesh.face_owner.resize(mesh.num_faces);
  mesh.face_neighbor.resize(mesh.num_faces);
  mesh.face_vertices.resize(static_cast<std::size_t>(mesh.num_faces) * 2, -1);
  mesh.face_normal.resize(static_cast<std::size_t>(mesh.num_faces) * 3, 0.0f);
  mesh.face_center.resize(static_cast<std::size_t>(mesh.num_faces) * 3, 0.0f);
  mesh.face_area.resize(mesh.num_faces, 0.0f);

  for (int new_face = 0; new_face < mesh.num_faces; ++new_face) {
    const FaceBuildData& src = faces[static_cast<std::size_t>(face_order[static_cast<std::size_t>(new_face)])];
    mesh.face_owner[new_face] = src.owner;
    mesh.face_neighbor[new_face] = src.neighbor;
    mesh.face_vertices[2 * new_face + 0] = src.v0;
    mesh.face_vertices[2 * new_face + 1] = src.v1;
    mesh.face_normal[3 * new_face + 0] = src.nx;
    mesh.face_normal[3 * new_face + 1] = src.ny;
    mesh.face_normal[3 * new_face + 2] = src.nz;
    mesh.face_area[new_face] = src.area;
    const auto& p0 = points2d[src.v0];
    const auto& p1 = points2d[src.v1];
    mesh.face_center[3 * new_face + 0] = 0.5f * (p0[0] + p1[0]);
    mesh.face_center[3 * new_face + 1] = 0.5f * (p0[1] + p1[1]);
    mesh.face_center[3 * new_face + 2] = 0.0f;
  }

  int patch_start = static_cast<int>(interior_faces.size());
  auto push_patch_range = [&](const std::string& patch_name) {
    const auto it = boundary_faces.find(patch_name);
    if (it == boundary_faces.end() || it->second.empty()) {
      return;
    }
    BoundaryPatchRange patch;
    patch.name = patch_name;
    patch.start_face = patch_start;
    patch.face_count = static_cast<int>(it->second.size());
    mesh.boundary_patches.push_back(std::move(patch));
    patch_start += static_cast<int>(it->second.size());
  };

  push_patch_range("wall");
  push_patch_range("farfield");
  push_patch_range("wake");
  push_patch_range("cut");
  for (const auto& [patch_name, patch_faces] : boundary_faces) {
    if (patch_name == "wall" || patch_name == "farfield" || patch_name == "wake" ||
        patch_name == "cut") {
      continue;
    }
    if (patch_faces.empty()) {
      continue;
    }
    BoundaryPatchRange patch;
    patch.name = patch_name;
    patch.start_face = patch_start;
    patch.face_count = static_cast<int>(patch_faces.size());
    mesh.boundary_patches.push_back(std::move(patch));
    patch_start += static_cast<int>(patch_faces.size());
  }

  mesh.points.reserve(points2d.size() * 3);
  for (const auto& point : points2d) {
    mesh.points.push_back(point[0]);
    mesh.points.push_back(point[1]);
    mesh.points.push_back(0.0f);
  }

  mesh.cell_connectivity.reserve(triangles.size() * 3);
  mesh.cell_offsets.reserve(triangles.size());
  mesh.cell_types.reserve(triangles.size());
  int running_offset = 0;
  for (const auto& tri : triangles) {
    mesh.cell_connectivity.push_back(tri[0]);
    mesh.cell_connectivity.push_back(tri[1]);
    mesh.cell_connectivity.push_back(tri[2]);
    running_offset += 3;
    mesh.cell_offsets.push_back(running_offset);
    mesh.cell_types.push_back(static_cast<std::uint8_t>(5));
  }

  return mesh;
}
}  // namespace

std::vector<std::array<float, 2>> generate_naca4_profile(const std::string& code,
                                                          const int num_circumferential,
                                                          const float chord) {
  if (chord <= 0.0f) {
    throw std::invalid_argument("Airfoil chord must be positive.");
  }
  const std::string digits = extract_naca_digits(code);

  const float m = static_cast<float>(digits[0] - '0') / 100.0f;
  const float p = static_cast<float>(digits[1] - '0') / 10.0f;
  const float t = static_cast<float>(10 * (digits[2] - '0') + (digits[3] - '0')) / 100.0f;

  const int points_per_surface = std::max(16, num_circumferential / 2 + 1);
  std::vector<std::array<float, 2>> upper(static_cast<std::size_t>(points_per_surface));
  std::vector<std::array<float, 2>> lower(static_cast<std::size_t>(points_per_surface));

  for (int i = 0; i < points_per_surface; ++i) {
    const float beta = kPi * static_cast<float>(i) / static_cast<float>(points_per_surface - 1);
    const float x = 0.5f * (1.0f - std::cos(beta));
    const float sqrt_x = std::sqrt(std::max(x, 0.0f));
    const float yt =
      5.0f * t *
      (0.2969f * sqrt_x - 0.1260f * x - 0.3516f * x * x + 0.2843f * x * x * x -
       0.1036f * x * x * x * x);

    float yc = 0.0f;
    float dyc_dx = 0.0f;
    if (m > 0.0f && p > 0.0f && p < 1.0f) {
      if (x < p) {
        yc = m / (p * p) * (2.0f * p * x - x * x);
        dyc_dx = 2.0f * m / (p * p) * (p - x);
      } else {
        const float one_minus_p = 1.0f - p;
        yc = m / (one_minus_p * one_minus_p) * (1.0f - 2.0f * p + 2.0f * p * x - x * x);
        dyc_dx = 2.0f * m / (one_minus_p * one_minus_p) * (p - x);
      }
    }

    const float theta = std::atan(dyc_dx);
    const float xu = x - yt * std::sin(theta);
    const float yu = yc + yt * std::cos(theta);
    const float xl = x + yt * std::sin(theta);
    const float yl = yc - yt * std::cos(theta);

    upper[static_cast<std::size_t>(i)] = {xu * chord, yu * chord};
    lower[static_cast<std::size_t>(i)] = {xl * chord, yl * chord};
  }

  std::vector<std::array<float, 2>> contour;
  contour.reserve(static_cast<std::size_t>(2 * points_per_surface - 1));
  for (int i = points_per_surface - 1; i >= 0; --i) {
    contour.push_back(upper[static_cast<std::size_t>(i)]);
  }
  for (int i = 1; i < points_per_surface; ++i) {
    contour.push_back(lower[static_cast<std::size_t>(i)]);
  }

  if (signed_area_polygon(contour) < 0.0f) {
    std::reverse(contour.begin(), contour.end());
  }

  return resample_closed_curve(contour, std::max(32, num_circumferential));
}

UnstructuredMesh make_airfoil_ogrid_mesh(const AirfoilMeshConfig& config) {
  if (config.num_circumferential < 32) {
    throw std::invalid_argument("num_circumferential must be >= 32.");
  }
  if (config.num_radial < 4) {
    throw std::invalid_argument("num_radial must be >= 4.");
  }

  std::vector<std::array<float, 2>> contour;
  if (config.airfoil_source == "file") {
    contour = load_airfoil_coordinate_file(config.coordinate_file, config.chord);
    contour = resample_closed_curve(contour, config.num_circumferential);
  } else {
    contour = generate_naca4_profile(config.naca_code, config.num_circumferential, config.chord);
  }

  if (signed_area_polygon(contour) < 0.0f) {
    std::reverse(contour.begin(), contour.end());
  }

  std::array<float, 2> center {0.0f, 0.0f};
  for (const auto& p : contour) {
    center[0] += p[0];
    center[1] += p[1];
  }
  center[0] /= static_cast<float>(contour.size());
  center[1] /= static_cast<float>(contour.size());

  const float farfield_radius = clamp_positive(config.farfield_radius, 2.5f) * config.chord;
  const float stretch = clamp_positive(config.radial_stretch, 1.0f);
  const int n_theta = config.num_circumferential;
  const int n_radial = config.num_radial;

  const auto point_id = [n_theta](const int i, const int j) { return j * n_theta + i; };

  std::vector<std::array<float, 2>> points;
  points.reserve(static_cast<std::size_t>((n_radial + 1) * n_theta));

  for (int j = 0; j <= n_radial; ++j) {
    const float eta = static_cast<float>(j) / static_cast<float>(n_radial);
    float sigma = eta;
    if (stretch > 1.0f + 1.0e-4f) {
      sigma = (std::pow(stretch, eta) - 1.0f) / (stretch - 1.0f);
    }

    for (int i = 0; i < n_theta; ++i) {
      const auto& inner = contour[static_cast<std::size_t>(i)];
      float dx = inner[0] - center[0];
      float dy = inner[1] - center[1];
      float radius = std::sqrt(dx * dx + dy * dy);
      if (radius < kEps) {
        dx = 1.0f;
        dy = 0.0f;
        radius = 1.0f;
      }
      const float inv_r = 1.0f / radius;
      const float ox = center[0] + dx * inv_r * farfield_radius;
      const float oy = center[1] + dy * inv_r * farfield_radius;
      points.push_back({inner[0] + sigma * (ox - inner[0]), inner[1] + sigma * (oy - inner[1])});
    }
  }

  std::vector<std::array<int, 3>> triangles;
  triangles.reserve(static_cast<std::size_t>(2 * n_theta * n_radial));
  for (int j = 0; j < n_radial; ++j) {
    for (int i = 0; i < n_theta; ++i) {
      const int i1 = (i + 1) % n_theta;
      const int a = point_id(i, j);
      const int b = point_id(i1, j);
      const int c = point_id(i1, j + 1);
      const int d = point_id(i, j + 1);
      triangles.push_back({a, b, c});
      triangles.push_back({a, c, d});
    }
  }

  std::map<EdgeKey, std::string> boundary_edge_patch;
  for (int i = 0; i < n_theta; ++i) {
    const int i1 = (i + 1) % n_theta;
    boundary_edge_patch.emplace(make_edge_key(point_id(i, 0), point_id(i1, 0)), "wall");
    boundary_edge_patch.emplace(make_edge_key(point_id(i, n_radial), point_id(i1, n_radial)),
                                "farfield");
  }

  return build_triangular_mesh(points, triangles, boundary_edge_patch);
}
}  // namespace cfd::core
