#pragma once

#include "cfd_core/mesh.hpp"

#include <array>
#include <filesystem>
#include <string>
#include <vector>

namespace cfd::core {
struct RunSummary {
  std::string status;
  std::string backend;
  std::string case_type;
  std::string run_log;
  int iterations = 0;
  float residual_l1 = 0.0f;
  float residual_l2 = 0.0f;
  float residual_linf = 0.0f;
  float cl = 0.0f;
  float cd = 0.0f;
  float cm = 0.0f;
};

struct ScalarCaseConfig {
  std::array<float, 3> u_inf {1.0f, 0.0f, 0.0f};
  float inflow_phi = 0.0f;
  std::vector<float> phi;
  std::filesystem::path output_dir = ".";
};

struct ScalarResidualNorms {
  float l1 = 0.0f;
  float l2 = 0.0f;
  float linf = 0.0f;
};

struct ScalarRunResult {
  std::string backend;
  ScalarResidualNorms residual_norms;
  std::vector<float> phi;
  std::vector<float> residual;
  std::filesystem::path residuals_csv_path;
  std::filesystem::path vtu_path;
};

std::string hello();
ScalarRunResult run_scalar_case(const UnstructuredMesh& mesh, const ScalarCaseConfig& config,
                                const std::string& backend = "cpu");
RunSummary run_case(const std::string& case_path, const std::string& out_dir,
                    const std::string& backend = "cpu");
}  // namespace cfd::core
