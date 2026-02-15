#pragma once

#include <string>

namespace cfd::core {
struct RunSummary {
  std::string status;
  std::string backend;
  std::string run_log;
  int iterations = 0;
};

std::string hello();
RunSummary run_case(const std::string& case_path, const std::string& out_dir,
                    const std::string& backend = "cpu");
}  // namespace cfd::core