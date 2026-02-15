#include "cfd_core/backend.hpp"
#include "cfd_core/mesh.hpp"
#include "cfd_core/io_vtk.hpp"
#include "cfd_core/solver.hpp"
#include "cfd_core/version.hpp"

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#if CFD_HAS_CUDA
#include "cfd_core/cuda_backend.hpp"
#endif

#if defined(_OPENMP)
#include <omp.h>
#endif

namespace {
std::vector<float> make_default_phi(const cfd::core::UnstructuredMesh& mesh) {
  std::vector<float> phi(static_cast<std::size_t>(mesh.num_cells), 1.0f);
  for (int c = 0; c < mesh.num_cells; ++c) {
    const float x = mesh.cell_center[3 * c + 0];
    const float y = mesh.cell_center[3 * c + 1];
    phi[c] = 1.0f + 0.25f * x - 0.15f * y;
  }
  return phi;
}

std::vector<float> compute_scalar_residual_cpu(const cfd::core::UnstructuredMesh& mesh,
                                               const std::vector<float>& phi,
                                               const std::array<float, 3>& u_inf,
                                               const float inflow_phi) {
  const int num_cells = mesh.num_cells;
  const int num_faces = mesh.num_faces;
  int thread_count = 1;

#if defined(_OPENMP)
  thread_count = std::max(1, omp_get_max_threads());
#endif

  std::vector<float> local_accum(static_cast<std::size_t>(thread_count) * num_cells, 0.0f);

#if defined(_OPENMP)
#pragma omp parallel
#endif
  {
    int thread_id = 0;
#if defined(_OPENMP)
    thread_id = omp_get_thread_num();
#endif
    float* residual_local = local_accum.data() + static_cast<std::size_t>(thread_id) * num_cells;

#if defined(_OPENMP)
#pragma omp for schedule(static)
#endif
    for (int face = 0; face < num_faces; ++face) {
      const int owner = mesh.face_owner[face];
      const int neighbor = mesh.face_neighbor[face];
      const float nx = mesh.face_normal[3 * face + 0];
      const float ny = mesh.face_normal[3 * face + 1];
      const float nz = mesh.face_normal[3 * face + 2];
      const float un = (u_inf[0] * nx + u_inf[1] * ny + u_inf[2] * nz) * mesh.face_area[face];

      float upwind_phi = phi[owner];
      if (neighbor >= 0) {
        upwind_phi = (un >= 0.0f) ? phi[owner] : phi[neighbor];
      } else if (un < 0.0f) {
        upwind_phi = inflow_phi;
      }

      const float flux = un * upwind_phi;
      residual_local[owner] -= flux;
      if (neighbor >= 0) {
        residual_local[neighbor] += flux;
      }
    }
  }

  std::vector<float> residual(static_cast<std::size_t>(num_cells), 0.0f);
  for (int thread_id = 0; thread_id < thread_count; ++thread_id) {
    const float* local = local_accum.data() + static_cast<std::size_t>(thread_id) * num_cells;
    for (int c = 0; c < num_cells; ++c) {
      residual[c] += local[c];
    }
  }

  return residual;
}

void normalize_residual_by_volume(const cfd::core::UnstructuredMesh& mesh,
                                  std::vector<float>* residual) {
  if (residual == nullptr) {
    return;
  }
  for (int c = 0; c < mesh.num_cells; ++c) {
    const float volume = mesh.cell_volume[c];
    if (volume > 0.0f) {
      (*residual)[c] /= volume;
    }
  }
}

cfd::core::ScalarResidualNorms compute_norms(const std::vector<float>& residual) {
  cfd::core::ScalarResidualNorms norms;
  double l1 = 0.0;
  double l2_sum = 0.0;
  double linf = 0.0;
  for (const float value : residual) {
    const double mag = std::abs(static_cast<double>(value));
    l1 += mag;
    l2_sum += mag * mag;
    linf = std::max(linf, mag);
  }

  norms.l1 = static_cast<float>(l1);
  norms.l2 = static_cast<float>(std::sqrt(l2_sum));
  norms.linf = static_cast<float>(linf);
  return norms;
}

void write_residuals_csv(const std::filesystem::path& file_path,
                         const cfd::core::ScalarResidualNorms& norms) {
  std::ofstream out(file_path, std::ios::trunc);
  out << "iter,residual,l1,l2,linf\n";
  out << "0," << norms.l2 << "," << norms.l1 << "," << norms.l2 << "," << norms.linf << "\n";
}

void write_forces_csv(const std::filesystem::path& file_path,
                      const cfd::core::ScalarResidualNorms& norms) {
  std::ofstream out(file_path, std::ios::trunc);
  out << "iter,cl,cd,cm\n";
  out << "0," << 0.0f << "," << norms.l2 << "," << 0.0f << "\n";
}
}  // namespace

namespace cfd::core {
bool cuda_available() {
#if CFD_HAS_CUDA
  return true;
#else
  return false;
#endif
}

std::string normalize_backend(std::string requested_backend) {
  std::transform(requested_backend.begin(), requested_backend.end(), requested_backend.begin(),
                 [](unsigned char c) { return static_cast<char>(std::tolower(c)); });

  if (requested_backend.empty()) {
    requested_backend = "cpu";
  }

  if (requested_backend != "cpu" && requested_backend != "cuda") {
    throw std::invalid_argument("Unsupported backend. Use 'cpu' or 'cuda'.");
  }

  if (requested_backend == "cuda" && !cuda_available()) {
    throw std::runtime_error(
      "CUDA backend requested but unavailable. Reconfigure with a CUDA compiler and "
      "-DCFD_ENABLE_CUDA=ON.");
  }

  return requested_backend;
}

std::string hello() {
  return "cfd_core bindings ok";
}

ScalarRunResult run_scalar_case(const UnstructuredMesh& mesh, const ScalarCaseConfig& config,
                                const std::string& backend) {
  if (mesh.num_cells <= 0 || mesh.num_faces <= 0) {
    throw std::invalid_argument("Mesh must contain cells and faces.");
  }
  if (mesh.face_owner.size() != static_cast<std::size_t>(mesh.num_faces) ||
      mesh.face_neighbor.size() != static_cast<std::size_t>(mesh.num_faces) ||
      mesh.face_area.size() != static_cast<std::size_t>(mesh.num_faces) ||
      mesh.face_normal.size() != static_cast<std::size_t>(mesh.num_faces) * 3 ||
      mesh.cell_volume.size() != static_cast<std::size_t>(mesh.num_cells)) {
    throw std::invalid_argument("Mesh face/cell array sizes are inconsistent.");
  }

  ScalarRunResult result;
  result.backend = normalize_backend(backend);

  result.phi = config.phi;
  if (result.phi.empty()) {
    result.phi = make_default_phi(mesh);
  }
  if (result.phi.size() != static_cast<std::size_t>(mesh.num_cells)) {
    throw std::invalid_argument("Scalar field phi size must match mesh.num_cells.");
  }

  if (result.backend == "cpu") {
    result.residual =
      compute_scalar_residual_cpu(mesh, result.phi, config.u_inf, config.inflow_phi);
  } else {
#if CFD_HAS_CUDA
    std::string error_message;
    const bool ok = cfd::cuda_backend::compute_scalar_residual_cuda(
      mesh, result.phi, config.u_inf, config.inflow_phi, &result.residual, &error_message);
    if (!ok) {
      throw std::runtime_error("CUDA scalar residual failed: " + error_message);
    }
#else
    throw std::runtime_error("CUDA backend requested but unavailable.");
#endif
  }

  normalize_residual_by_volume(mesh, &result.residual);
  result.residual_norms = compute_norms(result.residual);

  const std::filesystem::path output_dir = config.output_dir.empty() ? "." : config.output_dir;
  std::filesystem::create_directories(output_dir);
  result.residuals_csv_path = output_dir / "residuals.csv";
  result.vtu_path = output_dir / "field_0000.vtu";

  write_residuals_csv(result.residuals_csv_path, result.residual_norms);
  const bool vtu_ok = write_scalar_cell_vtu(result.vtu_path, mesh, result.phi, result.residual);
  if (!vtu_ok) {
    throw std::runtime_error("Failed to write VTU output.");
  }

  return result;
}

RunSummary run_case(const std::string& case_path, const std::string& out_dir,
                    const std::string& backend) {
  const std::string resolved_backend = normalize_backend(backend);
  const std::filesystem::path output_dir(out_dir);
  std::filesystem::create_directories(output_dir);

  const UnstructuredMesh mesh = make_demo_tri_mesh_2x2();
  ScalarCaseConfig config;
  config.u_inf = {1.0f, 0.35f, 0.0f};
  config.inflow_phi = 1.0f;
  config.phi = make_default_phi(mesh);
  config.output_dir = output_dir;

  const ScalarRunResult scalar_result = run_scalar_case(mesh, config, resolved_backend);

  const std::filesystem::path run_log_path = output_dir / "run.log";
  const std::filesystem::path forces_path = output_dir / "forces.csv";

  {
    std::ofstream log(run_log_path, std::ios::trunc);
    log << "AeroCFD scalar advection demo\n";
    log << "version=" << version() << "\n";
    log << "case_path=" << case_path << "\n";
    log << "backend=" << resolved_backend << "\n";
    log << "num_cells=" << mesh.num_cells << "\n";
    log << "num_faces=" << mesh.num_faces << "\n";
    log << "residual_l1=" << scalar_result.residual_norms.l1 << "\n";
    log << "residual_l2=" << scalar_result.residual_norms.l2 << "\n";
    log << "residual_linf=" << scalar_result.residual_norms.linf << "\n";
    // TODO(physics): Swap scalar advection with Euler fluxes.
    // TODO(physics): Extend to Navier-Stokes viscous terms.
    // TODO(turbulence): Integrate SA/SST transport equations.
  }

  write_forces_csv(forces_path, scalar_result.residual_norms);

  RunSummary summary;
  summary.status = "ok";
  summary.backend = resolved_backend;
  summary.run_log = run_log_path.string();
  summary.iterations = 1;
  summary.residual_l1 = scalar_result.residual_norms.l1;
  summary.residual_l2 = scalar_result.residual_norms.l2;
  summary.residual_linf = scalar_result.residual_norms.linf;
  return summary;
}
}  // namespace cfd::core
