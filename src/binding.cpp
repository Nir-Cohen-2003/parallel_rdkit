#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/tuple.h>
#include "mol.hpp"
#include "similarity.hpp"
#include "similarity_io.hpp"
#include "screen_smarts.hpp"
#include "stoned.hpp"

namespace nb = nanobind;
using namespace nb::literals;

namespace {
using DenseArray = nb::ndarray<nb::numpy, float, nb::c_contig>;
using MaskArray = nb::ndarray<nb::numpy, bool, nb::c_contig>;
using I64Array = nb::ndarray<nb::numpy, std::int64_t, nb::c_contig>;
using FloatArray = nb::ndarray<nb::numpy, float, nb::c_contig>;
void check_matrix(const DenseArray &a, std::size_t n, std::size_t m) {
    if (a.ndim() != 2 || a.shape(0) != n || a.shape(1) != m || !a.data()) throw std::invalid_argument("dense output must be C-contiguous float32 with requested shape");
}
void check_mask(const MaskArray &a, std::size_t n) {
    if (a.ndim() != 1 || a.shape(0) != n || !a.data()) throw std::invalid_argument("mask output must be one-dimensional bool");
}
void native_dense_into(const std::vector<std::string>& left, const std::vector<std::string>& right,
 const parallel_rdkit::FingerprintOptions& opts, bool assume_sanitized, std::size_t batch_size,
 std::size_t tile_size, DenseArray out, MaskArray left_valid, MaskArray right_valid) {
    check_matrix(out,left.size(),right.size()); check_mask(left_valid,left.size()); check_mask(right_valid,right.size());
    parallel_rdkit::compute_similarity_dense_into(left,right,opts,assume_sanitized,batch_size,tile_size,
      out.data(), reinterpret_cast<std::uint8_t *>(left_valid.data()), reinterpret_cast<std::uint8_t *>(right_valid.data()));
}
std::vector<std::int64_t> native_coo_count_into(const std::vector<std::string>& left, const std::vector<std::string>& right,
 const parallel_rdkit::FingerprintOptions& opts, bool assume_sanitized, double threshold, std::size_t batch_size,
 std::size_t tile_size, MaskArray left_valid, MaskArray right_valid) {
    check_mask(left_valid,left.size()); check_mask(right_valid,right.size());
    return parallel_rdkit::count_similarity_coo(left,right,opts,assume_sanitized,threshold,batch_size,tile_size,
      reinterpret_cast<std::uint8_t *>(left_valid.data()), reinterpret_cast<std::uint8_t *>(right_valid.data()));
}
void native_coo_fill_into(const std::vector<std::string>& left, const std::vector<std::string>& right,
 const parallel_rdkit::FingerprintOptions& opts, bool assume_sanitized, double threshold, std::size_t batch_size,
 std::size_t tile_size, I64Array rows, I64Array cols, FloatArray values) {
    if (rows.ndim()!=1 || cols.ndim()!=1 || values.ndim()!=1 ||
        rows.shape(0)!=cols.shape(0) || rows.shape(0)!=values.shape(0) ||
        (rows.shape(0) && (!rows.data() || !cols.data() || !values.data())))
        throw std::invalid_argument("COO buffers must be equal-length one-dimensional arrays");
    parallel_rdkit::fill_similarity_coo(left, right, opts, assume_sanitized,
        threshold, batch_size, tile_size, rows.shape(0), rows.data(), cols.data(),
        values.data());
}
}

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

    m.def("msready_smiles", &parallel_rdkit::msready_smiles, "smiles"_a, "silent"_a = true,
          nb::call_guard<nb::gil_scoped_release>(),
          "Transforms a SMILES string into an MS-Ready SMILES string.");

    m.def("msready_smiles_parallel", &parallel_rdkit::msready_smiles_parallel, "smiles"_a, "silent"_a = true,
          nb::call_guard<nb::gil_scoped_release>(),
          "Parallel MS-Ready transformation of SMILES.");

    m.def("sanitize_smiles_parallel", &parallel_rdkit::sanitize_smiles_parallel, "smiles"_a,
          nb::call_guard<nb::gil_scoped_release>(),
          "Parallel SMILES sanitization.");

    m.def("inchi_to_smiles_parallel", &parallel_rdkit::inchi_to_smiles_parallel, "inchis"_a,
          nb::call_guard<nb::gil_scoped_release>(),
          "Parallel InChI to SMILES conversion.");

    m.def("smiles_to_inchi_parallel", &parallel_rdkit::smiles_to_inchi_parallel, "smiles"_a,
          nb::call_guard<nb::gil_scoped_release>(),
          "Parallel SMILES to InChI conversion.");

    m.def("smiles_to_inchikey_parallel", &parallel_rdkit::smiles_to_inchikey_parallel, "smiles"_a,
          nb::call_guard<nb::gil_scoped_release>(),
          "Parallel SMILES to InChIKey conversion.");

    m.def("msready_inchi_inchikey_parallel", &parallel_rdkit::msready_inchi_inchikey_parallel, "smiles"_a, "silent"_a = true,
          nb::call_guard<nb::gil_scoped_release>(),
          "Parallel conversion to MS-Ready SMILES, InChI, and InChIKey simultaneously.");

    m.def("get_fingerprints_parallel", &parallel_rdkit::get_fingerprints_parallel, "smiles"_a, "opts"_a,
          nb::call_guard<nb::gil_scoped_release>(),
          "Parallel fingerprint generation.");

    m.def("cross_similarity_dense_into", &native_dense_into,
          "left"_a, "right"_a, "opts"_a, "assume_sanitized"_a = false,
          "batch_size"_a = 4096, "tile_size"_a = 256, "out"_a,
          "left_valid"_a, "right_valid"_a,
          nb::call_guard<nb::gil_scoped_release>());
    m.def("cross_similarity_dense_to_file", &parallel_rdkit::write_similarity_dense_npy,
          "left"_a, "right"_a, "opts"_a, "assume_sanitized"_a = false,
          "batch_size"_a = 4096, "tile_size"_a = 256, "output_path"_a,
          "overwrite"_a = false,
          nb::call_guard<nb::gil_scoped_release>(),
          "Compute dense similarity in native batches and atomically publish a NumPy file.");
    m.def("cross_similarity_coo_count_into", &native_coo_count_into,
          "left"_a, "right"_a, "opts"_a, "assume_sanitized"_a = false,
          "threshold"_a, "batch_size"_a = 4096, "tile_size"_a = 256,
          "left_valid"_a, "right_valid"_a,
          nb::call_guard<nb::gil_scoped_release>());
    m.def("cross_similarity_coo_fill_into", &native_coo_fill_into,
          "left"_a, "right"_a, "opts"_a, "assume_sanitized"_a = false,
          "threshold"_a, "batch_size"_a = 4096, "tile_size"_a = 256,
          "rows"_a, "columns"_a, "values"_a,
          nb::call_guard<nb::gil_scoped_release>());

    m.def("smiles_to_formula_parallel", &parallel_rdkit::smiles_to_formula_parallel, "smiles"_a,
          nb::call_guard<nb::gil_scoped_release>(),
          "Parallel molecular formula computation. Returns a flattened (n*12) list of int64 counts in element order H, C, N, O, F, Na, P, S, Cl, K, Br, I.");

    // ScreenSmarts bindings
    nb::class_<parallel_rdkit::ScreenSmartsOptions>(m, "ScreenSmartsOptions")
        .def(nb::init<>())
        .def_rw("mode", &parallel_rdkit::ScreenSmartsOptions::mode)
        .def_rw("batch_size", &parallel_rdkit::ScreenSmartsOptions::batch_size)
        .def_rw("cache_path", &parallel_rdkit::ScreenSmartsOptions::cache_path)
        .def_rw("output_path", &parallel_rdkit::ScreenSmartsOptions::output_path);

    m.def("screen_smarts_direct", &parallel_rdkit::screen_smarts_direct, 
          "smiles_file"_a, "smarts_list"_a, "cache_path"_a,
          nb::call_guard<nb::gil_scoped_release>(),
          "Screen molecules against SMARTS patterns (direct mode). Returns N x M matrix.");

    m.def("screen_smarts_streaming", &parallel_rdkit::screen_smarts_streaming,
          "smiles_file"_a, "smarts_list"_a, "batch_size"_a, "cache_path"_a, "output_path"_a,
          nb::call_guard<nb::gil_scoped_release>(),
          "Screen molecules against SMARTS patterns (streaming mode). Writes to output_path.npy, returns molecule count.");

    // STONED bindings
    m.def("randomize_smiles_parallel", &parallel_rdkit::randomize_smiles_parallel,
          "smiles"_a, "num_samples"_a,
          nb::call_guard<nb::gil_scoped_release>(),
          "Generate randomized SMILES orderings in parallel. Returns a flattened list of length smiles.size() * num_samples.");

    m.def("tanimoto_scores_parallel", &parallel_rdkit::tanimoto_scores_parallel,
          "smiles"_a, "target_smi"_a, "opts"_a,
          nb::call_guard<nb::gil_scoped_release>(),
          "Compute Tanimoto similarity scores between a list of SMILES and a target SMILES in parallel.");
}
