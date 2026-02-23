#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/tuple.h>
#include "mol.hpp"

namespace nb = nanobind;
using namespace nb::literals;

NB_MODULE(parallel_rdkit_backend, m) {
    m.doc() = "Parallel RDKit molecule processing backend";

    nb::class_<parallel_rdkit::FingerprintOptions>(m, "FingerprintOptions")
        .def(nb::init<>())
        .def_rw("fp_type", &parallel_rdkit::FingerprintOptions::fp_type)
        .def_rw("fp_method", &parallel_rdkit::FingerprintOptions::fp_method)
        .def_rw("fpSize", &parallel_rdkit::FingerprintOptions::fpSize)
        .def_rw("radius", &parallel_rdkit::FingerprintOptions::radius)
        .def_rw("useBondTypes", &parallel_rdkit::FingerprintOptions::useBondTypes)
        .def_rw("minPath", &parallel_rdkit::FingerprintOptions::minPath)
        .def_rw("maxPath", &parallel_rdkit::FingerprintOptions::maxPath)
        .def_rw("numBitsPerFeature", &parallel_rdkit::FingerprintOptions::numBitsPerFeature)
        .def_rw("use2D", &parallel_rdkit::FingerprintOptions::use2D)
        .def_rw("minDistance", &parallel_rdkit::FingerprintOptions::minDistance)
        .def_rw("maxDistance", &parallel_rdkit::FingerprintOptions::maxDistance)
        .def_rw("countSimulation", &parallel_rdkit::FingerprintOptions::countSimulation)
        .def_rw("includeChirality", &parallel_rdkit::FingerprintOptions::includeChirality)
        .def_rw("targetSize", &parallel_rdkit::FingerprintOptions::targetSize);

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

    m.def("get_fingerprints_parallel", &parallel_rdkit::get_fingerprints_parallel, "smiles"_a, "opts"_a,
          "Parallel fingerprint generation.");
}
