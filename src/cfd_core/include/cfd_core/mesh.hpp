#pragma once

#include <cstddef>

namespace cfd::core {
struct MeshSummary {
  int dimension = 2;
  std::size_t cell_count = 0;
};

MeshSummary create_stub_mesh(int dimension);
}  // namespace cfd::core