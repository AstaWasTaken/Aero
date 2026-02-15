#include "cfd_core/io_vtk.hpp"

#include <fstream>

namespace cfd::core {
bool write_placeholder_vtu(const std::filesystem::path& output_path) {
  std::ofstream out(output_path, std::ios::trunc);
  if (!out) {
    return false;
  }

  out << "<?xml version=\"1.0\"?>\n";
  out << "<VTKFile type=\"UnstructuredGrid\" version=\"0.1\" byte_order=\"LittleEndian\">\n";
  out << "  <UnstructuredGrid>\n";
  out << "    <Piece NumberOfPoints=\"1\" NumberOfCells=\"1\">\n";
  out << "      <Points>\n";
  out << "        <DataArray type=\"Float32\" NumberOfComponents=\"3\" format=\"ascii\">0 0 0</DataArray>\n";
  out << "      </Points>\n";
  out << "      <Cells>\n";
  out << "        <DataArray type=\"Int32\" Name=\"connectivity\" format=\"ascii\">0</DataArray>\n";
  out << "        <DataArray type=\"Int32\" Name=\"offsets\" format=\"ascii\">1</DataArray>\n";
  out << "        <DataArray type=\"UInt8\" Name=\"types\" format=\"ascii\">1</DataArray>\n";
  out << "      </Cells>\n";
  out << "      <PointData>\n";
  out << "        <DataArray type=\"Float32\" Name=\"pressure\" format=\"ascii\">101325</DataArray>\n";
  out << "      </PointData>\n";
  out << "      <CellData/>\n";
  out << "    </Piece>\n";
  out << "  </UnstructuredGrid>\n";
  out << "</VTKFile>\n";

  return true;
}
}  // namespace cfd::core