#include "cfd_core/mesh.hpp"
#include "cfd_core/numerics/euler_flux.hpp"
#include "cfd_core/numerics/reconstruction.hpp"
#include "cfd_core/solvers/euler/mesh_geometry.hpp"
#include "cfd_core/solvers/euler/preconditioning.hpp"
#include "cfd_core/solvers/euler/residual_assembly.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <vector>

int main() {
  const cfd::core::UnstructuredMesh mesh = cfd::core::make_demo_tri_mesh_2x2();
  const int num_cells = mesh.num_cells;
  if (num_cells <= 0) {
    std::cerr << "Demo mesh has no cells.\n";
    return 2;
  }

  constexpr float gamma = 1.4f;
  const cfd::core::PrimitiveState uniform = {
    1.05f,
    0.32f,
    -0.08f,
    0.03f,
    0.9f,
  };
  std::vector<cfd::core::PrimitiveState> primitive(static_cast<std::size_t>(num_cells), uniform);
  const cfd::core::PrimitiveGradients gradients =
    cfd::core::compute_green_gauss_gradients(mesh, primitive);
  const std::vector<cfd::core::EulerBoundaryConditionType> boundary_type =
    cfd::core::build_euler_boundary_types(mesh);
  const cfd::core::ConservativeState farfield =
    cfd::core::primitive_to_conservative(uniform, gamma);
  const cfd::core::LowMachPreconditionConfig precondition = {
    true,
    0.15f,
    1.0e-3f,
    0.2f,
    1.0f,
  };

  std::vector<float> residual(static_cast<std::size_t>(num_cells) * 5, 0.0f);
  std::vector<float> spectral_radius(static_cast<std::size_t>(num_cells), 0.0f);
  const cfd::core::EulerResidualAssemblyConfig assembly = {
    &mesh,
    &primitive,
    &gradients,
    &boundary_type,
    farfield,
    gamma,
    cfd::core::EulerFluxScheme::kHllc,
    cfd::core::LimiterType::kVenkat,
    true,
    false,
    0.3f,
    0.0f,
    false,
    precondition,
  };
  cfd::core::assemble_euler_residual_cpu(assembly, &residual, &spectral_radius);

  float max_abs_residual = 0.0f;
  for (const float value : residual) {
    max_abs_residual = std::max(max_abs_residual, std::abs(value));
  }
  if (!std::isfinite(max_abs_residual) || max_abs_residual > 1.0e-6f) {
    std::cerr << "Uniform-flow residual invariance failed. max_abs_residual=" << max_abs_residual
              << "\n";
    return 3;
  }

  return 0;
}
