#include "cfd_core/backend.hpp"
#include "cfd_core/io_vtk.hpp"
#include "cfd_core/solver.hpp"
#include "cfd_core/version.hpp"

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>

namespace {
void write_residuals_csv(const std::filesystem::path& file_path) {
  std::ofstream out(file_path, std::ios::trunc);
  out << "iter,residual\n";
  out << "0,1.000000\n";
  out << "1,0.500000\n";
  out << "2,0.250000\n";
  out << "3,0.125000\n";
  out << "4,0.062500\n";
}

void write_forces_csv(const std::filesystem::path& file_path) {
  std::ofstream out(file_path, std::ios::trunc);
  out << "iter,cl,cd,cm\n";
  out << "0,0.1000,0.0200,0.0000\n";
  out << "1,0.2000,0.0190,-0.0010\n";
  out << "2,0.2800,0.0185,-0.0020\n";
  out << "3,0.3300,0.0180,-0.0025\n";
  out << "4,0.3500,0.0178,-0.0030\n";
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

RunSummary run_case(const std::string& case_path, const std::string& out_dir,
                    const std::string& backend) {
  // TODO(physics): Hook RANS/SST state assembly here.
  // TODO(numerics): Replace deterministic CSV generation with real residual updates.
  // TODO(cuda): Add GPU execution path and async host/device data movement.
  const std::string resolved_backend = normalize_backend(backend);
  const std::filesystem::path output_dir(out_dir);
  std::filesystem::create_directories(output_dir);

  const std::filesystem::path run_log_path = output_dir / "run.log";
  const std::filesystem::path residuals_path = output_dir / "residuals.csv";
  const std::filesystem::path forces_path = output_dir / "forces.csv";
  const std::filesystem::path field_path = output_dir / "field_0000.vtu";

  {
    std::ofstream log(run_log_path, std::ios::trunc);
    log << "AeroCFD stub run\n";
    log << "version=" << version() << "\n";
    log << "case_path=" << case_path << "\n";
    log << "backend=" << resolved_backend << "\n";
    log << "status=completed_stub\n";
    log << "TODO(physics): Implement RANS/SST governing equations and source terms.\n";
    log << "TODO(numerics): Replace dummy residuals with actual flux/residual assembly.\n";
    log << "TODO(cuda): Wire GPU residual kernels and asynchronous reductions.\n";
  }

  write_residuals_csv(residuals_path);
  write_forces_csv(forces_path);
  const bool vtk_ok = write_placeholder_vtu(field_path);
  if (!vtk_ok) {
    throw std::runtime_error("Failed to write placeholder VTU output.");
  }

  RunSummary summary;
  summary.status = "ok";
  summary.backend = resolved_backend;
  summary.run_log = run_log_path.string();
  summary.iterations = 5;
  return summary;
}
}  // namespace cfd::core
