#pragma once

#include <string>

namespace cfd::core {
enum class GeometryKind {
  Airfoil2D,
  Wing3D,
};

struct GeometrySpec {
  GeometryKind kind = GeometryKind::Airfoil2D;
  std::string name = "placeholder";
  double chord = 1.0;
  double span = 1.0;
};
}  // namespace cfd::core