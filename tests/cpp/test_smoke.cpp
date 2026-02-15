#include "cfd_core/solver.hpp"
#include "cfd_core/version.hpp"

int main() {
  if (cfd::core::version().empty()) {
    return 1;
  }

  if (cfd::core::hello().empty()) {
    return 2;
  }

  return 0;
}