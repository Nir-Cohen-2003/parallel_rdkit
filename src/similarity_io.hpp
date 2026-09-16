#pragma once

#include "similarity.hpp"
#include <cstdint>
#include <string>
#include <tuple>
#include <vector>

namespace parallel_rdkit {
inline constexpr const char *similarity_file_format = "npy-float32-c-order";

// Compute and publish a dense similarity matrix as a NumPy .npy file.  The
// returned masks retain the original input positions (zero means invalid).
// The file is published only after all computation and writes succeed.
std::tuple<std::vector<std::uint8_t>, std::vector<std::uint8_t>>
write_similarity_dense_npy(const std::vector<std::string>& left,
                          const std::vector<std::string>& right,
                          const FingerprintOptions& opts,
                          bool assume_sanitized,
                          std::size_t batch_size,
                          std::size_t tile_size,
                          const std::string& output_path,
                          bool overwrite);
}
