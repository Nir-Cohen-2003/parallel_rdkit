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

std::string msready_smiles(const std::string& smiles);

std::vector<std::string> msready_smiles_parallel(const std::vector<std::string>& smiles);
std::vector<std::string> sanitize_smiles_parallel(const std::vector<std::string>& smiles);
std::vector<std::string> inchi_to_smiles_parallel(const std::vector<std::string>& inchis);
std::vector<std::string> smiles_to_inchi_parallel(const std::vector<std::string>& smiles);
std::vector<std::string> smiles_to_inchikey_parallel(const std::vector<std::string>& smiles);
std::tuple<std::vector<std::string>, std::vector<std::string>, std::vector<std::string>> msready_inchi_inchikey_parallel(const std::vector<std::string>& smiles);

std::vector<float> get_fingerprints_parallel(const std::vector<std::string>& smiles, const FingerprintOptions& opts);

} // namespace parallel_rdkit
