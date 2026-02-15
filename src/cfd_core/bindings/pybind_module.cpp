#include "cfd_core/backend.hpp"
#include "cfd_core/solver.hpp"
#include "cfd_core/version.hpp"

#include <pybind11/pybind11.h>

namespace py = pybind11;

PYBIND11_MODULE(cfd_core, m) {
  m.doc() = "pybind11 bindings for AeroCFD scaffold";

  m.def("version", []() { return cfd::core::version(); }, "Return native core version string.");
  m.def("hello", []() { return cfd::core::hello(); }, "Return hello-world marker from C++ core.");
  m.def(
    "cuda_available",
    []() { return cfd::core::cuda_available(); },
    "Return True when the CUDA backend is available in this build.");

  m.def(
    "run_case",
    [](const std::string& case_path, const std::string& out_dir, const std::string& backend) {
      const cfd::core::RunSummary summary = cfd::core::run_case(case_path, out_dir, backend);
      py::dict result;
      result["status"] = summary.status;
      result["backend"] = summary.backend;
      result["case_type"] = summary.case_type;
      result["run_log"] = summary.run_log;
      result["iterations"] = summary.iterations;
      result["residual_l1"] = summary.residual_l1;
      result["residual_l2"] = summary.residual_l2;
      result["residual_linf"] = summary.residual_linf;
      result["cl"] = summary.cl;
      result["cd"] = summary.cd;
      result["cm"] = summary.cm;
      return result;
    },
    py::arg("case_path"), py::arg("out_dir"), py::arg("backend") = "cpu",
    "Run a native case file and generate VTU/CSV outputs.");
}
