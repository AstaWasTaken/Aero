include(FetchContent)

find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
find_package(pybind11 CONFIG QUIET)

if(NOT pybind11_FOUND)
  message(STATUS "pybind11 not found in current environment. Fetching with FetchContent.")
  FetchContent_Declare(
    pybind11
    GIT_REPOSITORY https://github.com/pybind/pybind11.git
    GIT_TAG v2.12.0
  )
  FetchContent_MakeAvailable(pybind11)
endif()

pybind11_add_module(cfd_core_py pybind_module.cpp)

target_link_libraries(cfd_core_py PRIVATE cfd_core_lib)
target_compile_features(cfd_core_py PRIVATE cxx_std_20)
set_target_properties(cfd_core_py PROPERTIES OUTPUT_NAME cfd_core)

install(TARGETS cfd_core_py
  LIBRARY DESTINATION .
  RUNTIME DESTINATION .
  ARCHIVE DESTINATION .
)