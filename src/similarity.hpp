#pragma once

#include "mol.hpp"
#include <cstdint>
#include <string>
#include <tuple>
#include <vector>

namespace parallel_rdkit {
void validate_similarity_options(const FingerprintOptions& opts);

struct PackedLibrary {
    std::size_t bits = 0;
    std::size_t words = 0;
    std::vector<std::uint64_t> data;
    std::vector<std::uint32_t> popcount;
    std::vector<std::uint8_t> valid;
};

PackedLibrary prepare_similarity_library(const std::vector<std::string>& smiles,
                                         const FingerprintOptions& opts,
                                         bool assume_sanitized = false);
void compute_similarity_dense_into(const std::vector<std::string>& left,
                                   const std::vector<std::string>& right,
                                   const FingerprintOptions& opts,
                                   bool assume_sanitized, std::size_t batch_size,
                                   std::size_t tile_size, float *out,
                                   std::uint8_t *left_valid,
                                   std::uint8_t *right_valid);
std::tuple<std::vector<float>, std::vector<std::uint8_t>, std::vector<std::uint8_t>>
compute_similarity_dense(const std::vector<std::string>& left,
                         const std::vector<std::string>& right,
                         const FingerprintOptions& opts,
                         bool assume_sanitized, std::size_t batch_size,
                         std::size_t tile_size);
std::vector<std::int64_t> count_similarity_coo(const std::vector<std::string>& left,
                                               const std::vector<std::string>& right,
                                               const FingerprintOptions& opts,
                                               bool assume_sanitized, double threshold,
                                               std::size_t batch_size, std::size_t tile_size,
                                               std::uint8_t *left_valid,
                                               std::uint8_t *right_valid);
void fill_similarity_coo(const std::vector<std::string>& left,
                         const std::vector<std::string>& right,
                         const FingerprintOptions& opts, bool assume_sanitized,
                         double threshold, std::size_t batch_size,
                         std::size_t tile_size, std::size_t capacity,
                         std::int64_t *rows, std::int64_t *cols, float *values);

// Compute one disjoint score tile from already prepared libraries. The caller
// owns the tile buffer, whose size is rows * columns.
void compute_similarity_packed_tile(const PackedLibrary& left,
                                    std::size_t left_start,
                                    const PackedLibrary& right,
                                    std::size_t right_start,
                                    std::size_t rows, std::size_t columns,
                                    std::size_t right_stride, float *out);
std::tuple<std::vector<std::int64_t>, std::vector<std::int64_t>, std::vector<float>,
           std::vector<std::uint8_t>, std::vector<std::uint8_t>>
compute_similarity_coo(const std::vector<std::string>& left,
                       const std::vector<std::string>& right,
                       const FingerprintOptions& opts, bool assume_sanitized,
                       double threshold, std::size_t batch_size,
                       std::size_t tile_size);
}
