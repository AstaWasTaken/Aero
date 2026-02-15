#pragma once

#include "cfd_core/mesh.hpp"

#include <array>
#include <string>
#include <vector>

namespace cfd::core {
struct AirfoilMeshConfig {
  std::string airfoil_source = "naca4";
  std::string naca_code = "0012";
  std::string coordinate_file;
  float chord = 1.0f;
  int num_circumferential = 160;
  int num_radial = 48;
  float farfield_radius = 15.0f;  // in chord lengths
  float radial_stretch = 1.4f;
};

std::vector<std::array<float, 2>> generate_naca4_profile(const std::string& code,
                                                          int num_circumferential,
                                                          float chord);
UnstructuredMesh make_airfoil_ogrid_mesh(const AirfoilMeshConfig& config);
}  // namespace cfd::core
