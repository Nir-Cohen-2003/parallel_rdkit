#include "stoned.hpp"

#include <GraphMol/GraphMol.h>
#include <GraphMol/SmilesParse/SmilesParse.h>
#include <GraphMol/SmilesParse/SmilesWrite.h>
#include <GraphMol/MolOps.h>

#include <omp.h>
#include <memory>
#include <algorithm>
#include <cmath>

namespace parallel_rdkit {

using namespace RDKit;

std::vector<std::string> randomize_smiles_parallel(const std::vector<std::string>& smiles, int num_samples) {
    long n = static_cast<long>(smiles.size());
    std::vector<std::string> results;
    if (n == 0 || num_samples <= 0) {
        return results;
    }
    results.resize(static_cast<size_t>(n) * static_cast<size_t>(num_samples));

    #pragma omp parallel for schedule(static, 500)
    for (long i = 0; i < n; ++i) {
        try {
            std::unique_ptr<ROMol> mol(SmilesToMol(smiles[i]));
            if (!mol) {
                for (int j = 0; j < num_samples; ++j) {
                    results[static_cast<size_t>(i) * num_samples + j] = "";
                }
                continue;
            }

            for (int j = 0; j < num_samples; ++j) {
                try {
                    // Kekulize modifies the molecule in place; make a copy per sample.
                    // Kekulize requires an RWMol (read-write), not ROMol.
                    std::unique_ptr<RWMol> mol_copy(new RWMol(*mol));
                    MolOps::Kekulize(*mol_copy);
                    // MolToSmiles(mol, doIsomericSmiles=false, doKekule=true,
                    //             rootedAtAtom=-1, canonical=false,
                    //             allBondsExplicit=false, allHsExplicit=false, doRandom=true)
                    results[static_cast<size_t>(i) * num_samples + j] =
                        MolToSmiles(*mol_copy, false, true, -1, false, false, false, true);
                } catch (...) {
                    results[static_cast<size_t>(i) * num_samples + j] = "";
                }
            }
        } catch (...) {
            for (int j = 0; j < num_samples; ++j) {
                results[static_cast<size_t>(i) * num_samples + j] = "";
            }
        }
    }

    return results;
}

std::vector<float> tanimoto_scores_parallel(const std::vector<std::string>& smiles,
                                            const std::string& target_smi,
                                            const FingerprintOptions& opts) {
    long n = static_cast<long>(smiles.size());
    std::vector<float> scores(n, 0.0f);

    if (n == 0) {
        return scores;
    }

    // Combine target + queries into a single batch for the existing parallel fingerprinter.
    std::vector<std::string> all_smiles;
    all_smiles.reserve(1 + smiles.size());
    all_smiles.push_back(target_smi);
    all_smiles.insert(all_smiles.end(), smiles.begin(), smiles.end());

    auto fps_and_valid = get_fingerprints_parallel(all_smiles, opts);
    const std::vector<float>& fps = std::get<0>(fps_and_valid);
    const std::vector<uint8_t>& valid = std::get<1>(fps_and_valid);

    size_t fpSize = static_cast<size_t>(opts.fpSize);
    bool is_count = (opts.fp_method.find("Count") != std::string::npos);

    // If target is invalid, all scores stay 0.0.
    if (valid.empty() || !valid[0]) {
        return scores;
    }

    // Extract target fingerprint.
    const float* target_fp = fps.data();

    #pragma omp parallel for schedule(static, 500)
    for (long i = 0; i < n; ++i) {
        if (!valid[static_cast<size_t>(i) + 1]) {
            scores[i] = 0.0f;
            continue;
        }

        const float* query_fp = fps.data() + (static_cast<size_t>(i) + 1) * fpSize;
        float common = 0.0f;
        float union_sum = 0.0f;

        if (is_count) {
            for (size_t j = 0; j < fpSize; ++j) {
                float a = target_fp[j];
                float b = query_fp[j];
                common += std::min(a, b);
                union_sum += std::max(a, b);
            }
        } else {
            for (size_t j = 0; j < fpSize; ++j) {
                float a = target_fp[j];
                float b = query_fp[j];
                common += a * b;
                union_sum += a + b - a * b;
            }
        }

        if (union_sum > 0.0f) {
            scores[i] = common / union_sum;
        } else {
            scores[i] = 0.0f;
        }
    }

    return scores;
}

} // namespace parallel_rdkit
