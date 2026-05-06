#pragma once

#include <string>
#include <vector>
#include "mol.hpp"

namespace parallel_rdkit {

/**
 * Generate num_samples randomized SMILES orderings for each input SMILES.
 * Uses OpenMP parallelization over the input list.
 *
 * Parameters:
 *   smiles      - Input SMILES strings.
 *   num_samples - Number of randomized variants to generate per SMILES.
 *
 * Returns:
 *   Flattened vector of length smiles.size() * num_samples.
 *   Invalid inputs produce empty strings.
 */
std::vector<std::string> randomize_smiles_parallel(const std::vector<std::string>& smiles, int num_samples);

/**
 * Compute Tanimoto similarity scores between a list of SMILES and a target SMILES.
 * Fingerprints are generated in parallel using the existing fingerprint infrastructure.
 *
 * Parameters:
 *   smiles     - Query SMILES strings.
 *   target_smi - Target SMILES string.
 *   opts       - FingerprintOptions controlling fingerprint type and size.
 *
 * Returns:
 *   Vector of Tanimoto scores (0.0 for invalid molecules).
 */
std::vector<float> tanimoto_scores_parallel(const std::vector<std::string>& smiles,
                                            const std::string& target_smi,
                                            const FingerprintOptions& opts);

} // namespace parallel_rdkit
