#include "cfd_core/backend.hpp"
#include "cfd_core/io_vtk.hpp"
#include "cfd_core/mesh.hpp"
#include "cfd_core/solvers/euler_solver.hpp"
#include "cfd_core/solver.hpp"
#include "cfd_core/version.hpp"

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#if CFD_HAS_CUDA
#include "cfd_core/cuda_backend.hpp"
#endif

#if defined(_OPENMP)
#include <omp.h>
#endif

namespace {
constexpr float kPi = 3.14159265358979323846f;

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

std::string trim_copy(const std::string& value) {
  std::size_t begin = 0;
  while (begin < value.size() && std::isspace(static_cast<unsigned char>(value[begin]))) {
    ++begin;
  }
  std::size_t end = value.size();
  while (end > begin && std::isspace(static_cast<unsigned char>(value[end - 1]))) {
    --end;
  }
  return value.substr(begin, end - begin);
}

std::unordered_map<std::string, std::string> parse_case_kv(const std::filesystem::path& path) {
  std::unordered_map<std::string, std::string> kv;
  std::ifstream in(path);
  if (!in) {
    return kv;
  }

  std::string line;
  while (std::getline(in, line)) {
    const std::string stripped = trim_copy(line);
    if (stripped.empty() || stripped[0] == '#') {
      continue;
    }

    std::size_t sep = stripped.find('=');
    if (sep == std::string::npos) {
      sep = stripped.find(':');
    }
    if (sep == std::string::npos) {
      continue;
    }

    const std::string key = trim_copy(stripped.substr(0, sep));
    const std::string value = trim_copy(stripped.substr(sep + 1));
    if (!key.empty()) {
      kv[key] = value;
    }
  }

  return kv;
}

std::string get_string(const std::unordered_map<std::string, std::string>& kv, const char* key,
                       const std::string& fallback) {
  const auto it = kv.find(key);
  if (it == kv.end() || it->second.empty()) {
    return fallback;
  }
  return it->second;
}

bool has_key(const std::unordered_map<std::string, std::string>& kv, const char* key) {
  return kv.find(key) != kv.end();
}

float get_float(const std::unordered_map<std::string, std::string>& kv, const char* key,
                const float fallback) {
  const auto it = kv.find(key);
  if (it == kv.end()) {
    return fallback;
  }
  try {
    return std::stof(it->second);
  } catch (...) {
    return fallback;
  }
}

int get_int(const std::unordered_map<std::string, std::string>& kv, const char* key,
            const int fallback) {
  const auto it = kv.find(key);
  if (it == kv.end()) {
    return fallback;
  }
  try {
    return std::stoi(it->second);
  } catch (...) {
    return fallback;
  }
}

std::array<float, 3> get_vec3(const std::unordered_map<std::string, std::string>& kv,
                              const char* key, const std::array<float, 3>& fallback) {
  const auto it = kv.find(key);
  if (it == kv.end()) {
    return fallback;
  }
  std::array<float, 3> values = fallback;
  std::string text = it->second;
  std::replace(text.begin(), text.end(), ';', ',');
  std::istringstream iss(text);
  std::string token;
  int idx = 0;
  while (std::getline(iss, token, ',') && idx < 3) {
    try {
      values[idx] = std::stof(trim_copy(token));
      ++idx;
    } catch (...) {
      return fallback;
    }
  }
  if (idx < 3) {
    return fallback;
  }
  return values;
}

std::string to_lower_copy(std::string value) {
  std::transform(value.begin(), value.end(), value.begin(),
                 [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  return value;
}

cfd::core::EulerFluxScheme parse_flux_scheme(const std::string& value,
                                             const cfd::core::EulerFluxScheme fallback) {
  const std::string normalized = to_lower_copy(trim_copy(value));
  if (normalized == "hllc") {
    return cfd::core::EulerFluxScheme::kHllc;
  }
  if (normalized == "rusanov") {
    return cfd::core::EulerFluxScheme::kRusanov;
  }
  return fallback;
}

cfd::core::LimiterType parse_limiter_type(const std::string& value,
                                          const cfd::core::LimiterType fallback) {
  const std::string normalized = to_lower_copy(trim_copy(value));
  if (normalized == "venkat" || normalized == "venkatakrishnan") {
    return cfd::core::LimiterType::kVenkat;
  }
  if (normalized == "minmod") {
    return cfd::core::LimiterType::kMinmod;
  }
  return fallback;
}

bool parse_on_off(const std::string& value, const bool fallback) {
  const std::string normalized = to_lower_copy(trim_copy(value));
  if (normalized == "on" || normalized == "true" || normalized == "yes" || normalized == "1") {
    return true;
  }
  if (normalized == "off" || normalized == "false" || normalized == "no" || normalized == "0") {
    return false;
  }
  return fallback;
}
}  // namespace

namespace cfd::core {
bool cuda_available() {
#if CFD_HAS_CUDA
  return cfd::cuda_backend::cuda_runtime_available(nullptr);
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
      "CUDA backend requested but unavailable. Build with -DCFD_ENABLE_CUDA=ON and ensure a "
      "CUDA-capable device is visible.");
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
      mesh.face_vertices.size() != static_cast<std::size_t>(mesh.num_faces) * 2 ||
      mesh.face_center.size() != static_cast<std::size_t>(mesh.num_faces) * 3 ||
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
  const auto case_kv = parse_case_kv(case_path);
  const std::string case_type = get_string(case_kv, "case_type", "scalar_advect_demo");

  const std::filesystem::path run_log_path = output_dir / "run.log";

  if (case_type == "euler_airfoil_2d") {
    EulerAirfoilCaseConfig euler_config;
    euler_config.output_dir = output_dir;
    euler_config.iterations = get_int(case_kv, "iterations", euler_config.iterations);
    euler_config.min_iterations = get_int(case_kv, "min_iterations", euler_config.min_iterations);
    euler_config.cfl_start = get_float(case_kv, "cfl_start", euler_config.cfl_start);
    euler_config.cfl_max = get_float(case_kv, "cfl_max", euler_config.cfl_max);
    euler_config.cfl_ramp_iters = get_int(case_kv, "cfl_ramp_iters", euler_config.cfl_ramp_iters);
    euler_config.residual_reduction_target =
      get_float(case_kv, "residual_reduction_target", euler_config.residual_reduction_target);
    euler_config.force_stability_tol =
      get_float(case_kv, "force_stability_tol", euler_config.force_stability_tol);
    euler_config.force_stability_window =
      get_int(case_kv, "force_stability_window", euler_config.force_stability_window);
    euler_config.force_mean_drift_tol =
      get_float(case_kv, "force_mean_drift_tol", euler_config.force_mean_drift_tol);
    euler_config.startup_first_order_iters =
      get_int(case_kv, "startup_first_order_iters", euler_config.startup_first_order_iters);
    euler_config.rk_stages = get_int(case_kv, "rk_stages", euler_config.rk_stages);
    euler_config.local_time_stepping = parse_on_off(
      get_string(case_kv, "local_time_stepping", euler_config.local_time_stepping ? "on" : "off"),
      euler_config.local_time_stepping);
    euler_config.gamma = get_float(case_kv, "gamma", euler_config.gamma);
    euler_config.gas_constant = get_float(case_kv, "gas_constant", euler_config.gas_constant);
    euler_config.mach = get_float(case_kv, "mach", euler_config.mach);
    euler_config.aoa_deg = get_float(case_kv, "aoa_deg", euler_config.aoa_deg);
    euler_config.p_inf = get_float(case_kv, "p_inf", euler_config.p_inf);
    euler_config.t_inf = get_float(case_kv, "t_inf", euler_config.t_inf);
    euler_config.rho_inf = get_float(case_kv, "rho_inf", euler_config.rho_inf);
    euler_config.x_ref = get_float(case_kv, "x_ref", euler_config.x_ref);
    euler_config.y_ref = get_float(case_kv, "y_ref", euler_config.y_ref);
    euler_config.flux_scheme = parse_flux_scheme(
      get_string(case_kv, "flux", "rusanov"), euler_config.flux_scheme);
    euler_config.limiter = parse_limiter_type(
      get_string(case_kv, "limiter", "minmod"), euler_config.limiter);

    const bool has_low_mach_fix_key = has_key(case_kv, "low_mach_fix");
    const bool low_mach_fix_token = parse_on_off(
      get_string(case_kv, "low_mach_fix", euler_config.low_mach_fix ? "on" : "off"),
      euler_config.low_mach_fix);
    const bool has_precond_key = has_key(case_kv, "precond") || has_key(case_kv, "precond_on");
    const std::string precond_token = has_key(case_kv, "precond")
                                        ? get_string(case_kv, "precond", "on")
                                        : get_string(case_kv, "precond_on", "on");
    if (has_precond_key) {
      euler_config.precond_on = parse_on_off(precond_token, euler_config.precond_on);
    } else {
      euler_config.precond_on = low_mach_fix_token;
    }

    euler_config.precond_mach_ref = get_float(
      case_kv, "precond_mach_ref",
      get_float(case_kv, "mach_ref", euler_config.precond_mach_ref));
    euler_config.precond_mach_min =
      get_float(case_kv, "precond_mach_min", euler_config.precond_mach_min);
    euler_config.precond_beta_min =
      get_float(case_kv, "precond_beta_min", euler_config.precond_beta_min);
    euler_config.precond_beta_max =
      get_float(case_kv, "precond_beta_max", euler_config.precond_beta_max);
    euler_config.precond_farfield_bc = parse_on_off(
      get_string(case_kv, "precond_farfield_bc", euler_config.precond_farfield_bc ? "on" : "off"),
      euler_config.precond_farfield_bc);
    const float stabilization_floor_fallback = get_float(
      case_kv, "stabilization_mach_floor", euler_config.stabilization_mach_floor_target);
    euler_config.stabilization_mach_floor_start = get_float(
      case_kv, "stabilization_mach_floor_start", stabilization_floor_fallback);
    euler_config.stabilization_mach_floor_target = get_float(
      case_kv, "stabilization_mach_floor_target", stabilization_floor_fallback);
    euler_config.stabilization_mach_floor_k_start = get_float(
      case_kv, "stabilization_mach_floor_k_start", euler_config.stabilization_mach_floor_k_start);
    euler_config.stabilization_mach_floor_k_target = get_float(
      case_kv, "stabilization_mach_floor_k_target", euler_config.stabilization_mach_floor_k_target);
    euler_config.stabilization_ramp_iters =
      get_int(case_kv, "stabilization_ramp_iters", euler_config.stabilization_ramp_iters);

    euler_config.low_mach_fix = has_low_mach_fix_key ? low_mach_fix_token : euler_config.precond_on;
    euler_config.mach_cutoff = get_float(case_kv, "mach_cutoff", euler_config.mach_cutoff);
    euler_config.all_speed_flux_fix = parse_on_off(
      get_string(case_kv, "all_speed_flux_fix", euler_config.all_speed_flux_fix ? "on" : "off"),
      euler_config.all_speed_flux_fix);
    euler_config.all_speed_mach_cutoff = get_float(case_kv, "all_speed_mach_cutoff",
                                                   euler_config.all_speed_mach_cutoff);
    if (euler_config.all_speed_mach_cutoff <= 0.0f) {
      euler_config.all_speed_mach_cutoff = 0.25f;
    }
    euler_config.all_speed_f_min = get_float(
      case_kv, "all_speed_f_min",
      get_float(case_kv, "all_speed_theta_min", euler_config.all_speed_f_min));
    if (euler_config.all_speed_f_min <= 0.0f) {
      euler_config.all_speed_f_min = 0.05f;
    }
    euler_config.all_speed_f_min = std::clamp(euler_config.all_speed_f_min, 1.0e-6f, 1.0f);
    euler_config.all_speed_ramp_start_iter =
      get_int(case_kv, "all_speed_ramp_start_iter", euler_config.all_speed_ramp_start_iter);
    euler_config.all_speed_ramp_iters =
      get_int(case_kv, "all_speed_ramp_iters", euler_config.all_speed_ramp_iters);
    euler_config.all_speed_ramp_start_iter = std::max(euler_config.all_speed_ramp_start_iter, 0);
    euler_config.all_speed_ramp_iters = std::max(euler_config.all_speed_ramp_iters, 0);
    euler_config.all_speed_staged_controller =
      parse_on_off(get_string(case_kv, "all_speed_staged_controller",
                              euler_config.all_speed_staged_controller ? "on" : "off"),
                   euler_config.all_speed_staged_controller);
    euler_config.all_speed_stage_a_min_iters =
      std::max(get_int(case_kv, "all_speed_stage_a_min_iters",
                       euler_config.all_speed_stage_a_min_iters),
               0);
    euler_config.all_speed_stage_a_max_iters =
      std::max(get_int(case_kv, "all_speed_stage_a_max_iters",
                       euler_config.all_speed_stage_a_max_iters),
               0);
    euler_config.all_speed_stage_f_min_high = get_float(
      case_kv, "all_speed_stage_f_min_high", euler_config.all_speed_stage_f_min_high);
    euler_config.all_speed_stage_f_min_mid = get_float(
      case_kv, "all_speed_stage_f_min_mid", euler_config.all_speed_stage_f_min_mid);
    euler_config.all_speed_stage_f_min_low = get_float(
      case_kv, "all_speed_stage_f_min_low", euler_config.all_speed_stage_f_min_low);
    euler_config.all_speed_pjump_wall_target = get_float(
      case_kv, "all_speed_pjump_wall_target", euler_config.all_speed_pjump_wall_target);
    euler_config.all_speed_pjump_spike_factor = get_float(
      case_kv, "all_speed_pjump_spike_factor", euler_config.all_speed_pjump_spike_factor);
    euler_config.all_speed_pjump_hold_iters =
      std::max(get_int(case_kv, "all_speed_pjump_hold_iters",
                       euler_config.all_speed_pjump_hold_iters),
               1);
    euler_config.all_speed_freeze_iters =
      std::max(get_int(case_kv, "all_speed_freeze_iters", euler_config.all_speed_freeze_iters), 0);
    euler_config.all_speed_cfl_drop_factor = get_float(
      case_kv, "all_speed_cfl_drop_factor", euler_config.all_speed_cfl_drop_factor);
    euler_config.all_speed_cfl_restore_factor = get_float(
      case_kv, "all_speed_cfl_restore_factor", euler_config.all_speed_cfl_restore_factor);
    euler_config.all_speed_cfl_min_scale = get_float(
      case_kv, "all_speed_cfl_min_scale", euler_config.all_speed_cfl_min_scale);
    euler_config.aoa0_symmetry_enforce = parse_on_off(
      get_string(case_kv, "aoa0_symmetry_enforce",
                 euler_config.aoa0_symmetry_enforce ? "on" : "off"),
      euler_config.aoa0_symmetry_enforce);
    euler_config.aoa0_symmetry_enforce_interval =
      std::max(get_int(case_kv, "aoa0_symmetry_enforce_interval",
                       euler_config.aoa0_symmetry_enforce_interval),
               0);
    if (euler_config.force_mean_drift_tol <= 0.0f) {
      euler_config.force_mean_drift_tol = euler_config.force_stability_tol;
    }

    euler_config.mesh.airfoil_source =
      get_string(case_kv, "airfoil_source", euler_config.mesh.airfoil_source);
    euler_config.mesh.naca_code = get_string(case_kv, "naca_code", euler_config.mesh.naca_code);
    euler_config.mesh.coordinate_file =
      get_string(case_kv, "airfoil_file", euler_config.mesh.coordinate_file);
    euler_config.mesh.chord = get_float(case_kv, "chord", euler_config.mesh.chord);
    euler_config.mesh.num_circumferential =
      get_int(case_kv, "num_circumferential", euler_config.mesh.num_circumferential);
    euler_config.mesh.num_radial = get_int(case_kv, "num_radial", euler_config.mesh.num_radial);
    euler_config.mesh.farfield_radius =
      get_float(case_kv, "farfield_radius", euler_config.mesh.farfield_radius);
    euler_config.mesh.radial_stretch =
      get_float(case_kv, "radial_stretch", euler_config.mesh.radial_stretch);

    const EulerRunResult euler_result = run_euler_airfoil_case(euler_config, resolved_backend);
    const EulerIterationRecord& final_record = euler_result.history.back();
    float rho_inf = euler_config.rho_inf;
    if (rho_inf <= 0.0f) {
      rho_inf = euler_config.p_inf / (euler_config.gas_constant * euler_config.t_inf);
    }
    const float aoa_rad = euler_config.aoa_deg * (kPi / 180.0f);
    const float a_inf =
      std::sqrt(euler_config.gamma * euler_config.p_inf / std::max(rho_inf, 1.0e-12f));
    const float speed_inf = std::max(euler_config.mach, 0.0f) * a_inf;
    const float u_inf = speed_inf * std::cos(aoa_rad);
    const float v_inf = speed_inf * std::sin(aoa_rad);
    const float chord = std::max(euler_config.mesh.chord, 1.0e-8f);
    const float s_ref = chord * 1.0f;
    const float q_inf = 0.5f * rho_inf * speed_inf * speed_inf + 1.0e-12f;
    const FreestreamReference reference = {
      rho_inf,
      euler_config.p_inf,
      euler_config.aoa_deg,
      speed_inf,
      chord,
      euler_config.x_ref,
      euler_config.y_ref,
    };
    const PressureForceDiagnostics pressure_diag =
      compute_pressure_force_diagnostics(euler_result.mesh, euler_result.p, reference, "wall");
    const ForceCoefficients forces_abs =
      integrate_pressure_forces(euler_result.mesh, euler_result.p, reference, "wall", false);
    const ForceCoefficients forces_gauge =
      integrate_pressure_forces(euler_result.mesh, euler_result.p, reference, "wall", true);
    const float sum_nA_mag = std::sqrt(pressure_diag.sum_nA_x * pressure_diag.sum_nA_x +
                                       pressure_diag.sum_nA_y * pressure_diag.sum_nA_y);

    std::cout << "aoa_deg=" << euler_config.aoa_deg << "\n";
    std::cout << "aoa_rad=" << aoa_rad << "\n";
    std::cout << "u_inf=" << u_inf << "\n";
    std::cout << "v_inf=" << v_inf << "\n";
    std::cout << "S_ref=" << s_ref << "\n";
    std::cout << "wall_faces_integrated=" << pressure_diag.integrated_face_count << "\n";
    std::cout << "sum_nA_x=" << pressure_diag.sum_nA_x << "\n";
    std::cout << "sum_nA_y=" << pressure_diag.sum_nA_y << "\n";
    std::cout << "sum_nA_mag=" << sum_nA_mag << "\n";
    std::cout << "Fx_abs=" << pressure_diag.fx_abs << "\n";
    std::cout << "Fy_abs=" << pressure_diag.fy_abs << "\n";
    std::cout << "Fx_gauge=" << pressure_diag.fx_gauge << "\n";
    std::cout << "Fy_gauge=" << pressure_diag.fy_gauge << "\n";

    {
      std::ofstream log(run_log_path, std::ios::trunc);
      log << "AeroCFD Euler airfoil 2D\n";
      log << "version=" << version() << "\n";
      log << "case_path=" << case_path << "\n";
      log << "backend=" << resolved_backend << "\n";
      log << "case_type=" << case_type << "\n";
      log << "num_cells=" << euler_result.mesh.num_cells << "\n";
      log << "num_faces=" << euler_result.mesh.num_faces << "\n";
      log << "iterations=" << euler_result.history.size() << "\n";
      log << "residual_l1=" << final_record.residual_l1 << "\n";
      log << "residual_l2=" << final_record.residual_l2 << "\n";
      log << "residual_linf=" << final_record.residual_linf << "\n";
      log << "cl=" << forces_gauge.cl << "\n";
      log << "cd=" << forces_gauge.cd << "\n";
      log << "cm=" << forces_gauge.cm << "\n";
      log << "aoa_deg=" << euler_config.aoa_deg << "\n";
      log << "aoa_rad=" << aoa_rad << "\n";
      log << "u_inf=" << u_inf << "\n";
      log << "v_inf=" << v_inf << "\n";
      log << "wall_faces_integrated=" << pressure_diag.integrated_face_count << "\n";
      log << "normal_is_unit=" << (pressure_diag.normal_is_unit ? 1 : 0) << "\n";
      log << "sum_nA_x=" << pressure_diag.sum_nA_x << "\n";
      log << "sum_nA_y=" << pressure_diag.sum_nA_y << "\n";
      log << "sum_nA_mag=" << sum_nA_mag << "\n";
      log << "Fx_abs=" << pressure_diag.fx_abs << "\n";
      log << "Fy_abs=" << pressure_diag.fy_abs << "\n";
      log << "Fx_gauge=" << pressure_diag.fx_gauge << "\n";
      log << "Fy_gauge=" << pressure_diag.fy_gauge << "\n";
      log << "q_inf=" << q_inf << "\n";
      log << "V_inf=" << speed_inf << "\n";
      log << "rho_inf=" << rho_inf << "\n";
      log << "S_ref=" << s_ref << "\n";
      log << "chord=" << chord << "\n";
      log << "Fx=" << forces_gauge.fx << "\n";
      log << "Fy=" << forces_gauge.fy << "\n";
      log << "L=" << forces_gauge.lift << "\n";
      log << "D=" << forces_gauge.drag << "\n";
      log << "cl_abs=" << forces_abs.cl << "\n";
      log << "cd_abs=" << forces_abs.cd << "\n";
      log << "cm_abs=" << forces_abs.cm << "\n";
      log << "flux="
          << (euler_config.flux_scheme == EulerFluxScheme::kHllc ? "hllc" : "rusanov") << "\n";
      log << "limiter="
          << (euler_config.limiter == LimiterType::kVenkat ? "venkat" : "minmod") << "\n";
      log << "low_mach_fix=" << (euler_config.low_mach_fix ? "on" : "off") << "\n";
      log << "mach_cutoff=" << euler_config.mach_cutoff << "\n";
      log << "all_speed_flux_fix=" << (euler_config.all_speed_flux_fix ? "on" : "off") << "\n";
      log << "all_speed_mach_cutoff=" << euler_config.all_speed_mach_cutoff << "\n";
      log << "all_speed_f_min=" << euler_config.all_speed_f_min << "\n";
      log << "all_speed_ramp_start_iter=" << euler_config.all_speed_ramp_start_iter << "\n";
      log << "all_speed_ramp_iters=" << euler_config.all_speed_ramp_iters << "\n";
      log << "all_speed_staged_controller="
          << (euler_config.all_speed_staged_controller ? "on" : "off") << "\n";
      log << "all_speed_stage_a_min_iters=" << euler_config.all_speed_stage_a_min_iters << "\n";
      log << "all_speed_stage_a_max_iters=" << euler_config.all_speed_stage_a_max_iters << "\n";
      log << "all_speed_stage_f_min_high=" << euler_config.all_speed_stage_f_min_high << "\n";
      log << "all_speed_stage_f_min_mid=" << euler_config.all_speed_stage_f_min_mid << "\n";
      log << "all_speed_stage_f_min_low=" << euler_config.all_speed_stage_f_min_low << "\n";
      log << "all_speed_pjump_wall_target=" << euler_config.all_speed_pjump_wall_target << "\n";
      log << "all_speed_pjump_spike_factor=" << euler_config.all_speed_pjump_spike_factor << "\n";
      log << "all_speed_pjump_hold_iters=" << euler_config.all_speed_pjump_hold_iters << "\n";
      log << "all_speed_freeze_iters=" << euler_config.all_speed_freeze_iters << "\n";
      log << "all_speed_cfl_drop_factor=" << euler_config.all_speed_cfl_drop_factor << "\n";
      log << "all_speed_cfl_restore_factor=" << euler_config.all_speed_cfl_restore_factor << "\n";
      log << "all_speed_cfl_min_scale=" << euler_config.all_speed_cfl_min_scale << "\n";
      log << "aoa0_symmetry_enforce=" << (euler_config.aoa0_symmetry_enforce ? "on" : "off")
          << "\n";
      log << "aoa0_symmetry_enforce_interval=" << euler_config.aoa0_symmetry_enforce_interval
          << "\n";
      log << "precond=" << (euler_config.precond_on ? "on" : "off") << "\n";
      log << "precond_mach_ref=" << euler_config.precond_mach_ref << "\n";
      log << "precond_mach_min=" << euler_config.precond_mach_min << "\n";
      log << "precond_beta_min=" << euler_config.precond_beta_min << "\n";
      log << "precond_beta_max=" << euler_config.precond_beta_max << "\n";
      log << "precond_farfield_bc=" << (euler_config.precond_farfield_bc ? "on" : "off") << "\n";
      log << "stabilization_mach_floor_start=" << euler_config.stabilization_mach_floor_start
          << "\n";
      log << "stabilization_mach_floor_target=" << euler_config.stabilization_mach_floor_target
          << "\n";
      log << "stabilization_mach_floor_k_start=" << euler_config.stabilization_mach_floor_k_start
          << "\n";
      log << "stabilization_mach_floor_k_target="
          << euler_config.stabilization_mach_floor_k_target << "\n";
      log << "stabilization_ramp_iters=" << euler_config.stabilization_ramp_iters << "\n";
      log << "precond_mach_ref_effective=" << euler_result.precond_mach_ref_effective << "\n";
      log << "precond_beta_min_effective=" << euler_result.precond_beta_min_effective << "\n";
      log << "precond_beta_max_effective=" << euler_result.precond_beta_max_effective << "\n";
      log << "stabilization_mach_floor_start_effective="
          << euler_result.stabilization_mach_floor_start_effective << "\n";
      log << "stabilization_mach_floor_target_effective="
          << euler_result.stabilization_mach_floor_target_effective << "\n";
      log << "stabilization_mach_floor_k_start_effective="
          << euler_result.stabilization_mach_floor_k_start_effective << "\n";
      log << "stabilization_mach_floor_k_target_effective="
          << euler_result.stabilization_mach_floor_k_target_effective << "\n";
      log << "stabilization_ramp_iters_effective="
          << euler_result.stabilization_ramp_iters_effective << "\n";
      log << "stabilization_mach_floor_final=" << euler_result.stabilization_mach_floor_final
          << "\n";
      log << "beta_min=" << euler_result.beta_min << "\n";
      log << "beta_max=" << euler_result.beta_max << "\n";
      log << "beta_mean=" << euler_result.beta_mean << "\n";
      log << "mach_local_min=" << euler_result.mach_local_min << "\n";
      log << "mach_local_max=" << euler_result.mach_local_max << "\n";
      log << "mach_local_mean=" << euler_result.mach_local_mean << "\n";
      log << "acoustic_scale_min=" << euler_result.acoustic_scale_min << "\n";
      log << "acoustic_scale_max=" << euler_result.acoustic_scale_max << "\n";
      log << "acoustic_scale_mean=" << euler_result.acoustic_scale_mean << "\n";
      log << "acoustic_scale_final_min=" << euler_result.acoustic_scale_final_min << "\n";
      log << "acoustic_scale_final_max=" << euler_result.acoustic_scale_final_max << "\n";
      log << "acoustic_scale_final_mean=" << euler_result.acoustic_scale_final_mean << "\n";
      log << "wall_adjacent_acoustic_scale_min=" << euler_result.wall_adjacent_acoustic_scale_min
          << "\n";
      log << "wall_adjacent_acoustic_scale_p01=" << euler_result.wall_adjacent_acoustic_scale_p01
          << "\n";
      log << "wall_adjacent_acoustic_scale_p50=" << euler_result.wall_adjacent_acoustic_scale_p50
          << "\n";
      log << "wall_adjacent_stabilization_scale_min="
          << euler_result.wall_adjacent_stabilization_scale_min << "\n";
      log << "wall_adjacent_stabilization_scale_p01="
          << euler_result.wall_adjacent_stabilization_scale_p01 << "\n";
      log << "wall_adjacent_stabilization_scale_p50="
          << euler_result.wall_adjacent_stabilization_scale_p50 << "\n";
      log << "wall_adjacent_acoustic_scale_final_min="
          << euler_result.wall_adjacent_acoustic_scale_final_min << "\n";
      log << "wall_adjacent_acoustic_scale_final_p01="
          << euler_result.wall_adjacent_acoustic_scale_final_p01 << "\n";
      log << "wall_adjacent_acoustic_scale_final_p50="
          << euler_result.wall_adjacent_acoustic_scale_final_p50 << "\n";
      log << "dissipation_debug_csv=" << euler_result.dissipation_debug_csv_path.string() << "\n";
      log << "startup_first_order_iters=" << euler_config.startup_first_order_iters << "\n";
      log << "rk_stages=" << euler_config.rk_stages << "\n";
      log << "local_time_stepping=" << (euler_config.local_time_stepping ? "on" : "off") << "\n";
      log << "force_stability_window=" << euler_config.force_stability_window << "\n";
      log << "force_mean_drift_tol=" << euler_config.force_mean_drift_tol << "\n";
      log << "force_window_stable_final=" << (euler_result.force_window_stable_final ? 1 : 0)
          << "\n";
      log << "force_mean_drift_stable_final="
          << (euler_result.force_mean_drift_stable_final ? 1 : 0) << "\n";
      log << "force_physical_converged_final="
          << (euler_result.force_physical_converged_final ? 1 : 0) << "\n";
      log << "force_mean_dcl_final=" << euler_result.force_mean_dcl_final << "\n";
      log << "force_mean_dcd_final=" << euler_result.force_mean_dcd_final << "\n";
      log << "force_mean_dcm_final=" << euler_result.force_mean_dcm_final << "\n";
      log << "pressure_fluctuation_nd_max_final="
          << euler_result.pressure_fluctuation_nd_max_final << "\n";
      log << "pressure_fluctuation_nd_wall_final="
          << euler_result.pressure_fluctuation_nd_wall_final << "\n";
      log << "pressure_fluctuation_nd_farfield_final="
          << euler_result.pressure_fluctuation_nd_farfield_final << "\n";
      log << "cd_times_mach_final=" << euler_result.cd_times_mach_final << "\n";
      log << "max_wall_mass_flux=" << euler_result.max_wall_mass_flux << "\n";
      // TODO(numerics): Add implicit pseudo-time and local Jacobian preconditioning.
      // TODO(physics): Extend from Euler to viscous Navier-Stokes terms.
    }

    RunSummary summary;
    summary.status = "ok";
    summary.backend = resolved_backend;
    summary.case_type = case_type;
    summary.run_log = run_log_path.string();
    summary.iterations = static_cast<int>(euler_result.history.size());
    summary.residual_l1 = final_record.residual_l1;
    summary.residual_l2 = final_record.residual_l2;
    summary.residual_linf = final_record.residual_linf;
    summary.cl = forces_gauge.cl;
    summary.cd = forces_gauge.cd;
    summary.cm = forces_gauge.cm;
    return summary;
  }

  const UnstructuredMesh mesh = make_demo_tri_mesh_2x2();
  ScalarCaseConfig config;
  config.u_inf = get_vec3(case_kv, "u_inf", {1.0f, 0.35f, 0.0f});
  config.inflow_phi = get_float(case_kv, "inflow_phi", 1.0f);
  config.phi = make_default_phi(mesh);
  config.output_dir = output_dir;

  const ScalarRunResult scalar_result = run_scalar_case(mesh, config, resolved_backend);

  const std::filesystem::path forces_path = output_dir / "forces.csv";
  {
    std::ofstream log(run_log_path, std::ios::trunc);
    log << "AeroCFD scalar advection demo\n";
    log << "version=" << version() << "\n";
    log << "case_path=" << case_path << "\n";
    log << "backend=" << resolved_backend << "\n";
    log << "case_type=scalar_advect_demo\n";
    log << "num_cells=" << mesh.num_cells << "\n";
    log << "num_faces=" << mesh.num_faces << "\n";
    log << "residual_l1=" << scalar_result.residual_norms.l1 << "\n";
    log << "residual_l2=" << scalar_result.residual_norms.l2 << "\n";
    log << "residual_linf=" << scalar_result.residual_norms.linf << "\n";
  }

  write_forces_csv(forces_path, scalar_result.residual_norms);

  RunSummary summary;
  summary.status = "ok";
  summary.backend = resolved_backend;
  summary.case_type = "scalar_advect_demo";
  summary.run_log = run_log_path.string();
  summary.iterations = 1;
  summary.residual_l1 = scalar_result.residual_norms.l1;
  summary.residual_l2 = scalar_result.residual_norms.l2;
  summary.residual_linf = scalar_result.residual_norms.linf;
  summary.cl = 0.0f;
  summary.cd = scalar_result.residual_norms.l2;
  summary.cm = 0.0f;
  return summary;
}
}  // namespace cfd::core
