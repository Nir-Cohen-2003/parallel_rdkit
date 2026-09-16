#include "similarity.hpp"
#include <GraphMol/GraphMol.h>
#include <GraphMol/SmilesParse/SmilesParse.h>
#include <GraphMol/Fingerprints/FingerprintGenerator.h>
#include <GraphMol/Fingerprints/MorganGenerator.h>
#include <GraphMol/Fingerprints/RDKitFPGenerator.h>
#include <GraphMol/Fingerprints/AtomPairGenerator.h>
#include <GraphMol/Fingerprints/TopologicalTorsionGenerator.h>
#include <GraphMol/Fingerprints/MACCS.h>
#include <DataStructs/BitVects.h>
#include <DataStructs/SparseIntVect.h>
#include <bit>
#include <cmath>
#include <limits>
#include <memory>
#include <stdexcept>
#include <atomic>
#include <exception>
#include <mutex>
#include <omp.h>

namespace parallel_rdkit {
using namespace RDKit;
namespace {
std::size_t bit_count(const PackedLibrary& f, std::size_t row) {
    return f.popcount[row];
}
float score(const PackedLibrary& a, std::size_t i, const PackedLibrary& b, std::size_t j) {
    const auto *x = a.data.data() + i * a.words;
    const auto *y = b.data.data() + j * b.words;
    std::uint64_t intersection = 0;
    // These fixed trip counts are deliberately explicit: compilers can unroll
    // them while the generic path handles every other positive fingerprint size.
    if (a.words == 16 && b.words == 16) for (int k=0;k<16;++k) intersection += std::popcount(x[k] & y[k]);
    else if (a.words == 32 && b.words == 32) for (int k=0;k<32;++k) intersection += std::popcount(x[k] & y[k]);
    else if (a.words == 64 && b.words == 64) for (int k=0;k<64;++k) intersection += std::popcount(x[k] & y[k]);
    else for (std::size_t k=0;k<a.words;++k) intersection += std::popcount(x[k] & y[k]);
    const std::uint64_t uni = std::uint64_t(bit_count(a,i)) + bit_count(b,j) - intersection;
    return uni == 0 ? 0.0f : static_cast<float>(static_cast<double>(intersection) / static_cast<double>(uni));
}

std::unique_ptr<FingerprintGenerator<std::uint64_t>> generator(const FingerprintOptions& o) {
    const auto n = static_cast<std::size_t>(o.fpSize);
    if (o.fp_type == "morgan") return std::unique_ptr<FingerprintGenerator<std::uint64_t>>(MorganFingerprint::getMorganGenerator<std::uint64_t>(o.radius, o.countSimulation, o.includeChirality, o.useBondTypes, false, nullptr, nullptr, n));
    if (o.fp_type == "rdkit") return std::unique_ptr<FingerprintGenerator<std::uint64_t>>(RDKitFP::getRDKitFPGenerator<std::uint64_t>(o.minPath, o.maxPath, true, true, true, nullptr, o.countSimulation, {1,2,4,8}, n, o.numBitsPerFeature));
    if (o.fp_type == "atompair") return std::unique_ptr<FingerprintGenerator<std::uint64_t>>(AtomPair::getAtomPairGenerator<std::uint64_t>(o.minDistance, o.maxDistance, o.includeChirality, o.use2D, nullptr, o.countSimulation, n));
    if (o.fp_type == "torsion") return std::unique_ptr<FingerprintGenerator<std::uint64_t>>(TopologicalTorsion::getTopologicalTorsionGenerator<std::uint64_t>(o.includeChirality, o.targetSize, nullptr, o.countSimulation, n));
    return nullptr;
}
void set_bit(PackedLibrary& out, std::size_t row, std::size_t bit) {
    if (bit >= out.bits) bit %= out.bits;
    out.data[row*out.words + bit/64] |= std::uint64_t(1) << (bit % 64);
}
}

void validate_similarity_options(const FingerprintOptions& opts) {
    if (opts.fp_type != "morgan" && opts.fp_type != "rdkit" &&
        opts.fp_type != "atompair" && opts.fp_type != "torsion" &&
        opts.fp_type != "maccs")
        throw std::invalid_argument("unsupported fingerprint family");
    if (opts.fp_method != "GetFingerprint" && opts.fp_method != "GetSparseFingerprint")
        throw std::invalid_argument("count fingerprint methods are not supported");
    if (opts.fpSize <= 0) throw std::invalid_argument("fpSize must be positive");
    if (opts.radius < 0) throw std::invalid_argument("radius must be nonnegative");
    if (opts.minPath < 0 || opts.maxPath < 0 || opts.minPath > opts.maxPath)
        throw std::invalid_argument("path bounds must be nonnegative with minPath <= maxPath");
    if (opts.numBitsPerFeature <= 0)
        throw std::invalid_argument("numBitsPerFeature must be positive");
    if (opts.minDistance < 0 || opts.maxDistance < 0 ||
        opts.minDistance > opts.maxDistance)
        throw std::invalid_argument("distance bounds must be nonnegative with minDistance <= maxDistance");
    if (opts.targetSize <= 0)
        throw std::invalid_argument("targetSize must be positive");
}

PackedLibrary prepare_similarity_library(const std::vector<std::string>& smiles,
                                         const FingerprintOptions& opts, bool) {
    validate_similarity_options(opts);
    const std::size_t bits = opts.fp_type == "maccs" ? 167u : static_cast<std::size_t>(opts.fpSize);
    PackedLibrary out{bits, (bits + 63) / 64, {}, std::vector<std::uint32_t>(smiles.size(), 0), std::vector<std::uint8_t>(smiles.size(), 0)};
    if (bits > std::numeric_limits<std::size_t>::max() / sizeof(std::uint64_t) /
                    (smiles.size() ? smiles.size() : 1))
        throw std::overflow_error("fingerprint allocation overflow");
    out.data.assign(smiles.size() * out.words, 0);
    std::exception_ptr first_error;
    std::mutex error_mutex;
    std::atomic<bool> failed{false};

    #pragma omp parallel
    {
        std::unique_ptr<FingerprintGenerator<std::uint64_t>> fpgen;
        try {
            if (opts.fp_type != "maccs") fpgen = generator(opts);
        } catch (...) {
            std::lock_guard<std::mutex> lock(error_mutex);
            if (!first_error) first_error = std::current_exception();
            failed.store(true, std::memory_order_release);
        }
        #pragma omp for schedule(static)
        for (long long ii = 0; ii < static_cast<long long>(smiles.size()); ++ii) {
            if (failed.load(std::memory_order_acquire)) continue;
            const std::size_t i = static_cast<std::size_t>(ii);
            std::unique_ptr<ROMol> mol;
            try {
                mol.reset(SmilesToMol(smiles[i]));
            } catch (...) {
                continue; // parser/sanitization failures are invalid positions
            }
            if (!mol) continue;
            try {
                if (opts.fp_type == "maccs") {
                    std::unique_ptr<ExplicitBitVect> fp(
                        MACCSFingerprints::getFingerprintAsBitVect(*mol));
                    if (!fp) continue;
                    for (unsigned int k = 0; k < fp->getNumBits() && k < bits; ++k)
                        if (fp->getBit(k)) set_bit(out, i, k);
                } else if (opts.fp_method == "GetFingerprint") {
                    std::unique_ptr<ExplicitBitVect> fp(fpgen->getFingerprint(*mol));
                    if (!fp) continue;
                    for (unsigned int k = 0; k < fp->getNumBits() && k < bits; ++k)
                        if (fp->getBit(k)) set_bit(out, i, k);
                } else {
                    std::unique_ptr<SparseBitVect> fp(fpgen->getSparseFingerprint(*mol));
                    if (!fp) continue;
                    for (int k : *fp->getBitSet()) set_bit(out, i, static_cast<std::size_t>(k));
                }
                if (out.words && bits % 64)
                    out.data[(i + 1) * out.words - 1] &=
                        (std::uint64_t(1) << (bits % 64)) - 1;
                std::uint32_t count = 0;
                for (std::size_t k = 0; k < out.words; ++k)
                    count += std::popcount(out.data[i * out.words + k]);
                out.popcount[i] = count;
                out.valid[i] = 1;
            } catch (...) {
                std::lock_guard<std::mutex> lock(error_mutex);
                if (!first_error) first_error = std::current_exception();
                failed.store(true, std::memory_order_release);
            }
        }
    }
    if (first_error) std::rethrow_exception(first_error);
    return out;
}

void compute_similarity_packed_tile(const PackedLibrary& left,
                                    std::size_t left_start,
                                    const PackedLibrary& right,
                                    std::size_t right_start,
                                    std::size_t rows, std::size_t columns,
                                    std::size_t right_stride, float *out) {
    if (left_start > left.valid.size() || rows > left.valid.size() - left_start ||
        right_start > right.valid.size() || columns > right.valid.size() - right_start)
        throw std::out_of_range("similarity tile exceeds prepared library");
    if (right_stride < columns || (rows && columns && !out))
        throw std::invalid_argument("invalid similarity tile output buffer");
    #pragma omp parallel for collapse(2) schedule(static)
    for (long long x = 0; x < static_cast<long long>(rows); ++x)
        for (long long y = 0; y < static_cast<long long>(columns); ++y) {
            const std::size_t i = left_start + static_cast<std::size_t>(x);
            const std::size_t j = right_start + static_cast<std::size_t>(y);
            out[static_cast<std::size_t>(x) * right_stride + static_cast<std::size_t>(y)] =
                (left.valid[i] && right.valid[j]) ? score(left, i, right, j) :
                std::numeric_limits<float>::quiet_NaN();
        }
}

void compute_similarity_dense_into(const std::vector<std::string>& left,
 const std::vector<std::string>& right, const FingerprintOptions& opts,
 bool assume_sanitized, std::size_t batch_size, std::size_t tile_size,
 float *out, std::uint8_t *left_valid, std::uint8_t *right_valid) {
    if (!batch_size || !tile_size) throw std::invalid_argument("batch and tile sizes must be positive");
    if (right.size() && left.size() > std::numeric_limits<std::size_t>::max()/right.size()) throw std::overflow_error("matrix size overflow");
    auto a=prepare_similarity_library(left,opts,assume_sanitized), b=prepare_similarity_library(right,opts,assume_sanitized);
    std::copy(a.valid.begin(), a.valid.end(), left_valid);
    std::copy(b.valid.begin(), b.valid.end(), right_valid);
    for (std::size_t i0=0;i0<left.size();i0+=batch_size) for (std::size_t j0=0;j0<right.size();j0+=batch_size)
      for (std::size_t i=i0;i<std::min(i0+batch_size,left.size());i+=tile_size) for (std::size_t j=j0;j<std::min(j0+batch_size,right.size());j+=tile_size) {
        const std::size_t rows = std::min(tile_size, left.size() - i);
        const std::size_t columns = std::min(tile_size, right.size() - j);
        compute_similarity_packed_tile(a, i, b, j, rows, columns,
                                       right.size(), out + i * right.size() + j);
      }
}

std::tuple<std::vector<float>, std::vector<std::uint8_t>, std::vector<std::uint8_t>> compute_similarity_dense(
 const std::vector<std::string>& left, const std::vector<std::string>& right,
 const FingerprintOptions& opts, bool assume_sanitized, std::size_t batch_size, std::size_t tile_size) {
    if (!batch_size || !tile_size) throw std::invalid_argument("batch and tile sizes must be positive");
    if (right.size() && left.size() > std::numeric_limits<std::size_t>::max()/right.size()) throw std::overflow_error("matrix size overflow");
    auto a=prepare_similarity_library(left,opts,assume_sanitized), b=prepare_similarity_library(right,opts,assume_sanitized);
    std::vector<float> result(left.size()*right.size(), 0.0f);
    for (std::size_t i0=0;i0<left.size();i0+=batch_size) for (std::size_t j0=0;j0<right.size();j0+=batch_size)
      for (std::size_t i=i0;i<std::min(i0+batch_size,left.size());i+=tile_size) for (std::size_t j=j0;j<std::min(j0+batch_size,right.size());j+=tile_size) {
        const std::size_t rows = std::min(tile_size, left.size() - i);
        const std::size_t columns = std::min(tile_size, right.size() - j);
        compute_similarity_packed_tile(a, i, b, j, rows, columns,
                                       right.size(), result.data() + i * right.size() + j);
      }
    return {std::move(result), std::move(a.valid), std::move(b.valid)};
}

std::vector<std::int64_t> count_similarity_coo(const std::vector<std::string>& left,
 const std::vector<std::string>& right, const FingerprintOptions& opts, bool assume_sanitized,
 double threshold, std::size_t batch_size, std::size_t tile_size, std::uint8_t *left_valid,
 std::uint8_t *right_valid) {
    if (!std::isfinite(threshold) || threshold<0 || threshold>1) throw std::invalid_argument("threshold must be in [0,1]");
    if (!batch_size || !tile_size) throw std::invalid_argument("batch and tile sizes must be positive");
    auto a=prepare_similarity_library(left,opts,assume_sanitized), b=prepare_similarity_library(right,opts,assume_sanitized);
    std::copy(a.valid.begin(), a.valid.end(), left_valid); std::copy(b.valid.begin(), b.valid.end(), right_valid);
    std::vector<std::int64_t> counts(left.size(),0);
    #pragma omp parallel for schedule(static)
    for (long long ii=0; ii<static_cast<long long>(left.size()); ++ii) { std::size_t i=static_cast<std::size_t>(ii); if(a.valid[i]) for(std::size_t j=0;j<right.size();++j) if(b.valid[j] && static_cast<double>(score(a,i,b,j))>=threshold) ++counts[i]; }
    return counts;
}
void fill_similarity_coo(const std::vector<std::string>& left, const std::vector<std::string>& right,
 const FingerprintOptions& opts, bool assume_sanitized, double threshold, std::size_t batch_size,
 std::size_t tile_size, std::size_t capacity, std::int64_t *rows, std::int64_t *cols,
 float *values) {
    if (!std::isfinite(threshold) || threshold < 0.0 || threshold > 1.0)
        throw std::invalid_argument("threshold must be in [0,1]");
    if (!batch_size || !tile_size) throw std::invalid_argument("batch and tile sizes must be positive");
    auto a = prepare_similarity_library(left, opts, assume_sanitized);
    auto b = prepare_similarity_library(right, opts, assume_sanitized);
    std::vector<std::int64_t> counts(left.size(), 0);
    #pragma omp parallel for schedule(static)
    for (long long ii = 0; ii < static_cast<long long>(left.size()); ++ii) {
        const std::size_t i = static_cast<std::size_t>(ii);
        if (a.valid[i])
            for (std::size_t j = 0; j < right.size(); ++j)
                if (b.valid[j] && static_cast<double>(score(a, i, b, j)) >= threshold)
                    ++counts[i];
    }
    std::vector<std::int64_t> offsets(left.size() + 1, 0);
    for (std::size_t i = 0; i < left.size(); ++i)
        offsets[i + 1] = offsets[i] + counts[i];
    const std::size_t nnz = static_cast<std::size_t>(offsets.back());
    if (capacity != nnz)
        throw std::invalid_argument("COO buffers must have exactly the expected number of entries");
    #pragma omp parallel for schedule(static)
    for (long long ii = 0; ii < static_cast<long long>(left.size()); ++ii) {
        const std::size_t i = static_cast<std::size_t>(ii);
        std::size_t cursor = static_cast<std::size_t>(offsets[i]);
        if (!a.valid[i]) continue;
        for (std::size_t j = 0; j < right.size(); ++j) {
            if (!b.valid[j]) continue;
            const float v = score(a, i, b, j);
            if (static_cast<double>(v) >= threshold) {
                rows[cursor] = static_cast<std::int64_t>(i);
                cols[cursor] = static_cast<std::int64_t>(j);
                values[cursor] = v;
                ++cursor;
            }
        }
    }
}

std::tuple<std::vector<std::int64_t>, std::vector<std::int64_t>, std::vector<float>, std::vector<std::uint8_t>, std::vector<std::uint8_t>> compute_similarity_coo(
 const std::vector<std::string>& left, const std::vector<std::string>& right, const FingerprintOptions& opts,
 bool assume_sanitized, double threshold, std::size_t batch_size, std::size_t tile_size) {
    if (!std::isfinite(threshold) || threshold<0 || threshold>1) throw std::invalid_argument("threshold must be in [0,1]");
    if (!batch_size || !tile_size) throw std::invalid_argument("batch and tile sizes must be positive");
    auto a=prepare_similarity_library(left,opts,assume_sanitized), b=prepare_similarity_library(right,opts,assume_sanitized);
    std::vector<std::int64_t> counts(left.size(),0);
    #pragma omp parallel for schedule(static)
    for (long long ii=0; ii<static_cast<long long>(left.size()); ++ii) {
        std::size_t i = static_cast<std::size_t>(ii);
        if(a.valid[i]) for(std::size_t j=0;j<right.size();++j)
            if(b.valid[j] && static_cast<double>(score(a,i,b,j))>=threshold) ++counts[i];
    }
    std::vector<std::int64_t> offsets(left.size()+1,0); for(std::size_t i=0;i<left.size();++i) offsets[i+1]=offsets[i]+counts[i];
    std::vector<std::int64_t> rows(offsets.back()), cols(offsets.back()); std::vector<float> values(offsets.back());
    #pragma omp parallel for schedule(static)
    for (long long ii=0; ii<static_cast<long long>(left.size()); ++ii) {
        const std::size_t i = static_cast<std::size_t>(ii);
        std::size_t cursor = static_cast<std::size_t>(offsets[i]);
        if (!a.valid[i]) continue;
        for (std::size_t j=0; j<right.size(); ++j) if (b.valid[j]) {
            const float v = score(a, i, b, j);
            if (static_cast<double>(v) >= threshold) {
                rows[cursor] = static_cast<std::int64_t>(i);
                cols[cursor] = static_cast<std::int64_t>(j);
                values[cursor] = v;
                ++cursor;
            }
        }
    }
    return {std::move(rows),std::move(cols),std::move(values),std::move(a.valid),std::move(b.valid)};
}
}
