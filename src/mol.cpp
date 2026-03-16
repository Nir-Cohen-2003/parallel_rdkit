#include <cstdint>
#include "mol.hpp"
#include <GraphMol/GraphMol.h>
#include <GraphMol/SmilesParse/SmilesParse.h>
#include <GraphMol/SmilesParse/SmilesWrite.h>
#include <GraphMol/MolStandardize/MolStandardize.h>
#include <GraphMol/MolStandardize/Tautomer.h>
#include <GraphMol/FileParsers/FileParsers.h>
#include <GraphMol/inchi.h>
#include <RDGeneral/RDLog.h>
#include <RDGeneral/Invariant.h>

#include <GraphMol/Fingerprints/FingerprintGenerator.h>
#include <GraphMol/Fingerprints/MorganGenerator.h>
#include <GraphMol/Fingerprints/RDKitFPGenerator.h>
#include <GraphMol/Fingerprints/AtomPairGenerator.h>
#include <GraphMol/Fingerprints/TopologicalTorsionGenerator.h>
#include <GraphMol/Fingerprints/MACCS.h>
#include <DataStructs/BitVects.h>
#include <DataStructs/SparseIntVect.h>

#include <omp.h>
#include <memory>
#include <tuple>

namespace parallel_rdkit {

using namespace RDKit;

// Helper function to check if a molecule has any carbon atoms
bool has_carbon(const ROMol& mol) {
    for (const auto& atom : mol.atoms()) {
        if (atom->getAtomicNum() == 6) {
            return true;
        }
    }
    return false;
}

std::string msready_smiles(const std::string& smiles) {
    std::unique_ptr<RWMol> mol;
    try {
        mol.reset(SmilesToMol(smiles));
    } catch (...) {
        return "";
    }
    if (!mol) return "";

    try {
        // 1. Cleanup (Metal Disconnection, Normalization, Reionization)
        MolStandardize::CleanupParameters cleanup_params;
        std::unique_ptr<RWMol> clean_mol(MolStandardize::cleanup(*mol, cleanup_params));
        if (!clean_mol) return "";

        // 2. Fragment Parent (Salt Stripping)
        std::unique_ptr<RWMol> frag_parent(MolStandardize::fragmentParent(*clean_mol, cleanup_params));
        if (!frag_parent) return "";

        // 3. Charge Parent (Neutralization)
        std::unique_ptr<RWMol> charge_parent(MolStandardize::chargeParent(*frag_parent, cleanup_params));
        if (!charge_parent) return "";

        // 4. Tautomer Canonicalization
        MolStandardize::TautomerEnumerator te;
        te.setRemoveBondStereo(true);
        te.setRemoveSp3Stereo(true);
        
        std::unique_ptr<ROMol> ms_ready_mol(te.canonicalize(*charge_parent));
        if (!ms_ready_mol) return "";

        // 5. Check if result has any carbon atoms (organic part)
        if (!has_carbon(*ms_ready_mol)) {
            return "<INORGANIC>";
        }

        // 6. Final SMILES generation (isomeric=false, canonical=true)
        return MolToSmiles(*ms_ready_mol, false, false);
    } catch (...) {
        return "";
    }
}

template <typename Func, typename R = std::invoke_result_t<Func, std::string>>
std::vector<R> process_parallel(const std::vector<std::string>& inputs, Func func) {
    long n = inputs.size();
    std::vector<R> results(n);

    // Use static schedule with chunk size of 500 as requested
    #pragma omp parallel for schedule(static, 500)
    for (long i = 0; i < n; ++i) {
        try {
            results[i] = func(inputs[i]);
        } catch (...) {
            // we should not normally reach here if func catches its own exceptions
            // but for safety, assign a default-constructed R if possible
            results[i] = R{};
        }
    }

    return results;
}

std::vector<std::string> msready_smiles_parallel(const std::vector<std::string>& smiles) {
    return process_parallel(smiles, msready_smiles);
}

std::vector<std::string> sanitize_smiles_parallel(const std::vector<std::string>& smiles) {
    return process_parallel(smiles, [](const std::string& s) -> std::string {
        try {
            std::unique_ptr<ROMol> mol(SmilesToMol(s));
            return mol ? MolToSmiles(*mol, false, false) : std::string("");
        } catch (...) {
            return "";
        }
    });
}

// Single-threaded versions for INCHI operations (not thread-safe)
// These are called from Python multiprocessing to ensure thread safety

std::vector<std::string> inchi_to_smiles_parallel(const std::vector<std::string>& inchis) {
    long n = inchis.size();
    std::vector<std::string> results(n);
    
    // Single-threaded: INCHI library is not thread-safe
    for (long i = 0; i < n; ++i) {
        try {
            ExtraInchiReturnValues rv;
            std::unique_ptr<ROMol> mol(InchiToMol(inchis[i], rv));
            results[i] = mol ? MolToSmiles(*mol, false, false) : std::string("");
        } catch (...) {
            results[i] = "";
        }
    }
    
    return results;
}

std::vector<std::string> smiles_to_inchi_parallel(const std::vector<std::string>& smiles) {
    long n = smiles.size();
    std::vector<std::string> results(n);
    
    // Single-threaded: INCHI library is not thread-safe
    for (long i = 0; i < n; ++i) {
        try {
            std::unique_ptr<ROMol> mol(SmilesToMol(smiles[i]));
            if (!mol) {
                results[i] = "";
            } else {
                ExtraInchiReturnValues rv;
                results[i] = MolToInchi(*mol, rv);
            }
        } catch (...) {
            results[i] = "";
        }
    }
    
    return results;
}

std::vector<std::string> smiles_to_inchikey_parallel(const std::vector<std::string>& smiles) {
    long n = smiles.size();
    std::vector<std::string> results(n);
    
    // Single-threaded: INCHI library is not thread-safe
    for (long i = 0; i < n; ++i) {
        try {
            std::unique_ptr<ROMol> mol(SmilesToMol(smiles[i]));
            if (!mol) {
                results[i] = "";
            } else {
                results[i] = MolToInchiKey(*mol);
            }
        } catch (...) {
            results[i] = "";
        }
    }
    
    return results;
}

std::tuple<std::vector<std::string>, std::vector<std::string>, std::vector<std::string>> msready_inchi_inchikey_parallel(const std::vector<std::string>& smiles) {
    long n = smiles.size();
    std::vector<std::string> msready_vec(n);
    std::vector<std::string> inchi_vec(n);
    std::vector<std::string> inchikey_vec(n);
    
    // Single-threaded: INCHI library is not thread-safe
    for (long i = 0; i < n; ++i) {
        std::string inchi = "";
        std::string inchikey = "";
        std::string msready = "";
        
        std::unique_ptr<RWMol> mol;
        try {
            mol.reset(SmilesToMol(smiles[i]));
        } catch (...) {
            // Invalid SMILES (e.g., bad valence) - return empty strings
            msready_vec[i] = "";
            inchi_vec[i] = "";
            inchikey_vec[i] = "";
            continue;
        }
        
        if (!mol) {
            msready_vec[i] = "";
            inchi_vec[i] = "";
            inchikey_vec[i] = "";
            continue;
        }
        
        // InChI library is not thread-safe - single threaded execution
        try {
            ExtraInchiReturnValues rv;
            inchi = MolToInchi(*mol, rv);
            inchikey = MolToInchiKey(*mol);
        } catch (...) {
            inchi = "";
            inchikey = "";
        }
        
        try {
            MolStandardize::CleanupParameters cleanup_params;
            std::unique_ptr<RWMol> clean_mol(MolStandardize::cleanup(*mol, cleanup_params));
            if (!clean_mol) {
                msready = "";
            } else {
                std::unique_ptr<RWMol> frag_parent(MolStandardize::fragmentParent(*clean_mol, cleanup_params));
                if (!frag_parent) {
                    msready = "";
                } else {
                    std::unique_ptr<RWMol> charge_parent(MolStandardize::chargeParent(*frag_parent, cleanup_params));
                    if (!charge_parent) {
                        msready = "";
                    } else {
                        MolStandardize::TautomerEnumerator te;
                        te.setRemoveBondStereo(true);
                        te.setRemoveSp3Stereo(true);
                        std::unique_ptr<ROMol> ms_ready_mol(te.canonicalize(*charge_parent));
                        if (!ms_ready_mol) {
                            msready = "";
                        } else {
                            // Check if result has any carbon atoms (organic part)
                            if (!has_carbon(*ms_ready_mol)) {
                                msready = "<INORGANIC>";
                            } else {
                                msready = MolToSmiles(*ms_ready_mol, false, false);
                            }
                        }
                    }
                }
            }
        } catch (...) {}
        
        msready_vec[i] = msready;
        inchi_vec[i] = inchi;
        inchikey_vec[i] = inchikey;
    }
    
    return {msready_vec, inchi_vec, inchikey_vec};
}

std::tuple<std::vector<float>, std::vector<uint8_t>> get_fingerprints_parallel(const std::vector<std::string>& smiles, const FingerprintOptions& opts) {
    long n = smiles.size();
    size_t fpSize = opts.fpSize;
    std::vector<float> results(n * fpSize, 0.0f);
    std::vector<uint8_t> valid(n, 0);

    #pragma omp parallel
    {
        std::unique_ptr<FingerprintGenerator<std::uint64_t>> fpgen;
        if (opts.fp_type == "morgan") {
            fpgen.reset(MorganFingerprint::getMorganGenerator<std::uint64_t>(opts.radius, opts.countSimulation, opts.includeChirality, opts.useBondTypes, false, nullptr, nullptr, fpSize));
        } else if (opts.fp_type == "rdkit") {
            fpgen.reset(RDKitFP::getRDKitFPGenerator<std::uint64_t>(opts.minPath, opts.maxPath, true, true, true, nullptr, opts.countSimulation, {1, 2, 4, 8}, fpSize, opts.numBitsPerFeature));
        } else if (opts.fp_type == "atompair") {
            fpgen.reset(AtomPair::getAtomPairGenerator<std::uint64_t>(opts.minDistance, opts.maxDistance, opts.includeChirality, opts.use2D, nullptr, opts.countSimulation, fpSize));
        } else if (opts.fp_type == "torsion") {
            fpgen.reset(TopologicalTorsion::getTopologicalTorsionGenerator<std::uint64_t>(opts.includeChirality, opts.targetSize, nullptr, opts.countSimulation, fpSize));
        }

        #pragma omp for schedule(static, 500)
        for (long i = 0; i < n; ++i) {
            try {
                std::unique_ptr<ROMol> mol(SmilesToMol(smiles[i]));
                if (!mol) continue;

                if (opts.fp_type == "maccs") {
                    std::unique_ptr<ExplicitBitVect> fp(RDKit::MACCSFingerprints::getFingerprintAsBitVect(*mol));
                    if (fp) {
                        for (unsigned int j = 0; j < fp->getNumBits() && j < fpSize; ++j) {
                            if (fp->getBit(j)) results[i * fpSize + j] = 1.0f;
                        }
                        valid[i] = 1;
                    }
                } else if (fpgen) {
                    if (opts.fp_method == "GetFingerprint") {
                        std::unique_ptr<ExplicitBitVect> fp(fpgen->getFingerprint(*mol));
                        if (fp) {
                            for (unsigned int j = 0; j < fp->getNumBits() && j < fpSize; ++j) {
                                if (fp->getBit(j)) results[i * fpSize + j] = 1.0f;
                            }
                            valid[i] = 1;
                        }
                    } else if (opts.fp_method == "GetCountFingerprint") {
                        std::unique_ptr<SparseIntVect<std::uint32_t>> fp(fpgen->getCountFingerprint(*mol));
                        if (fp) {
                            for (const auto& it : fp->getNonzeroElements()) {
                                results[i * fpSize + (it.first % fpSize)] += it.second;
                            }
                            valid[i] = 1;
                        }
                    } else if (opts.fp_method == "GetSparseFingerprint") {
                        std::unique_ptr<SparseBitVect> fp(fpgen->getSparseFingerprint(*mol));
                        if (fp) {
                            for (int bit : *fp->getBitSet()) {
                                results[i * fpSize + (static_cast<size_t>(bit) % fpSize)] = 1.0f;
                            }
                            valid[i] = 1;
                        }
                    } else if (opts.fp_method == "GetSparseCountFingerprint") {
                        std::unique_ptr<SparseIntVect<std::uint64_t>> fp(fpgen->getSparseCountFingerprint(*mol));
                        if (fp) {
                            for (const auto& it : fp->getNonzeroElements()) {
                                results[i * fpSize + (it.first % fpSize)] += it.second;
                            }
                            valid[i] = 1;
                        }
                    }
                }
            } catch (...) {
                // If any exception arises for a single molecule, it remains marked as invalid (valid[i] = 0)
                // and we continue with the next molecule.
            }
        }
    }
    return {results, valid};
}

} // namespace parallel_rdkit
