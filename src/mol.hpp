#pragma once

#include <string>
#include <vector>
#include <tuple>

namespace parallel_rdkit {

std::string msready_smiles(const std::string& smiles);

std::vector<std::string> msready_smiles_parallel(const std::vector<std::string>& smiles);
std::vector<std::string> sanitize_smiles_parallel(const std::vector<std::string>& smiles);
std::vector<std::string> inchi_to_smiles_parallel(const std::vector<std::string>& inchis);
std::vector<std::string> smiles_to_inchi_parallel(const std::vector<std::string>& smiles);
std::vector<std::string> smiles_to_inchikey_parallel(const std::vector<std::string>& smiles);
std::vector<std::tuple<std::string, std::string, std::string>> msready_inchi_inchikey_parallel(const std::vector<std::string>& smiles);

} // namespace parallel_rdkit
