#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/tuple.h>
#include "mol.hpp"

namespace nb = nanobind;
using namespace nb::literals;

NB_MODULE(parallel_rdkit_backend, m) {
    m.doc() = "Parallel RDKit molecule processing backend";

    m.def("msready_smiles", &parallel_rdkit::msready_smiles, "smiles"_a,
          "Transforms a SMILES string into an MS-Ready SMILES string.");

    m.def("msready_smiles_parallel", &parallel_rdkit::msready_smiles_parallel, "smiles"_a,
          "Parallel MS-Ready transformation of SMILES.");

    m.def("sanitize_smiles_parallel", &parallel_rdkit::sanitize_smiles_parallel, "smiles"_a,
          "Parallel SMILES sanitization.");

    m.def("inchi_to_smiles_parallel", &parallel_rdkit::inchi_to_smiles_parallel, "inchis"_a,
          "Parallel InChI to SMILES conversion.");

    m.def("smiles_to_inchi_parallel", &parallel_rdkit::smiles_to_inchi_parallel, "smiles"_a,
          "Parallel SMILES to InChI conversion.");

    m.def("smiles_to_inchikey_parallel", &parallel_rdkit::smiles_to_inchikey_parallel, "smiles"_a,
          "Parallel SMILES to InChIKey conversion.");

    m.def("msready_inchi_inchikey_parallel", &parallel_rdkit::msready_inchi_inchikey_parallel, "smiles"_a,
          "Parallel conversion to MS-Ready SMILES, InChI, and InChIKey simultaneously.");
}
