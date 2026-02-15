#pragma once

#include "cfd_core/mesh.hpp"

#include <string>
#include <vector>

namespace cfd::core {
struct FreestreamReference {
  float rho_inf = 1.225f;
  float p_inf = 101325.0f;
  float aoa_deg = 0.0f;
  float speed_inf = 1.0f;
  float chord = 1.0f;
  float x_ref = 0.25f;
  float y_ref = 0.0f;
};

struct WallCpSample {
  float s = 0.0f;
  float x = 0.0f;
  float y = 0.0f;
  float cp = 0.0f;
};

struct ForceCoefficients {
  float fx = 0.0f;
  float fy = 0.0f;
  float lift = 0.0f;
  float drag = 0.0f;
  float moment = 0.0f;
  float cl = 0.0f;
  float cd = 0.0f;
  float cm = 0.0f;
};

std::vector<int> find_patch_faces(const UnstructuredMesh& mesh, const std::string& patch_name);
std::vector<WallCpSample> extract_wall_cp(const UnstructuredMesh& mesh,
                                          const std::vector<float>& cell_pressure,
                                          const FreestreamReference& reference,
                                          const std::string& wall_patch = "wall");
ForceCoefficients integrate_pressure_forces(const UnstructuredMesh& mesh,
                                            const std::vector<float>& cell_pressure,
                                            const FreestreamReference& reference,
                                            const std::string& wall_patch = "wall");
}  // namespace cfd::core
