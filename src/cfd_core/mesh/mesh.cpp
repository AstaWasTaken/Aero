#include "cfd_core/mesh.hpp"

namespace cfd::core {
MeshSummary create_stub_mesh(const int dimension) {
  MeshSummary mesh;
  mesh.dimension = dimension;
  mesh.cell_count = dimension == 3 ? 128 : 64;
  return mesh;
}
}  // namespace cfd::core