#pragma once

#include <string>
#include <vector>
#include <cstdint>

namespace parallel_rdkit {

struct ScreenSmartsOptions {
    std::string mode = "direct";  // "direct" or "streaming"
    int batch_size = 10000;       // for streaming mode
    std::string cache_path = "";  // path to cache file (empty = no caching)
    std::string output_path = ""; // for streaming: .npy output file path
};

// Direct mode: returns full N x M matrix
std::vector<std::vector<uint8_t>> screen_smarts_direct(
    const std::string& smiles_file,
    const std::vector<std::string>& smarts_list,
    const std::string& cache_path);

// Streaming mode: processes in batches and writes to output .npy file
// Returns number of molecules processed
size_t screen_smarts_streaming(
    const std::string& smiles_file,
    const std::vector<std::string>& smarts_list,
    int batch_size,
    const std::string& cache_path,
    const std::string& output_path);

} // namespace parallel_rdkit
