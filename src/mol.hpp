#include <cstdint>
#pragma once

#include <string>
#include <vector>
#include <tuple>

namespace parallel_rdkit {

struct FingerprintOptions {
    std::string fp_type = "morgan";
    std::string fp_method = "GetFingerprint";
    int fpSize = 2048;
    int radius = 2;
    bool useBondTypes = true;
    int minPath = 1;
    int maxPath = 7;
    int numBitsPerFeature = 2;
    bool use2D = true;
    int minDistance = 1;
    int maxDistance = 30;
    bool countSimulation = true;
    bool includeChirality = false;
    int targetSize = 4;
};

std::string msready_smiles(const std::string& smiles, bool silent = true);

std::vector<std::string> msready_smiles_parallel(const std::vector<std::string>& smiles, bool silent = true);
std::vector<std::string> sanitize_smiles_parallel(const std::vector<std::string>& smiles);
std::vector<std::string> inchi_to_smiles_parallel(const std::vector<std::string>& inchis);
std::vector<std::string> smiles_to_inchi_parallel(const std::vector<std::string>& smiles);
std::vector<std::string> smiles_to_inchikey_parallel(const std::vector<std::string>& smiles);
std::tuple<std::vector<std::string>, std::vector<std::string>, std::vector<std::string>> msready_inchi_inchikey_parallel(const std::vector<std::string>& smiles, bool silent = true);

std::tuple<std::vector<float>, std::vector<uint8_t>> get_fingerprints_parallel(const std::vector<std::string>& smiles, const FingerprintOptions& opts);

// Return a flattened (n * 12) vector of int64_t counts for the formula array.
// Element order: H, C, N, O, F, Na, P, S, Cl, K, Br, I.
// Invalid SMILES produce a row of zeros.
std::vector<int64_t> smiles_to_formula_parallel(const std::vector<std::string>& smiles);

} // namespace parallel_rdkit
