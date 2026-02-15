#include "cfd_core/solver.hpp"
#include "cfd_core/version.hpp"

#include <iostream>

int main(int argc, char** argv) {
  std::cout << "AeroCFD native CLI stub\n";
  std::cout << "version=" << cfd::core::version() << "\n";
  std::cout << "hello=" << cfd::core::hello() << "\n";
  if (argc > 1) {
    std::cout << "This native CLI is a placeholder. Use Python CLI (`cfd`) for workflows.\n";
    std::cout << "First argument received: " << argv[1] << "\n";
  }
  return 0;
}