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

#include <omp.h>
#include <memory>
#include <tuple>

namespace parallel_rdkit {

using namespace RDKit;

std::string msready_smiles(const std::string& smiles) {
    std::unique_ptr<RWMol> mol(SmilesToMol(smiles));
    if (!mol) return "";

    try {
        // 1. Cleanup (Metal Disconnection, Normalization, Reionization)
        std::unique_ptr<RWMol> clean_mol(MolStandardize::cleanup(*mol));

        // 2. Fragment Parent (Salt Stripping)
        std::unique_ptr<RWMol> frag_parent(MolStandardize::fragmentParent(*clean_mol));

        // 3. Charge Parent (Neutralization)
        std::unique_ptr<RWMol> charge_parent(MolStandardize::chargeParent(*frag_parent));

        // 4. Tautomer Canonicalization
        MolStandardize::TautomerEnumerator te;
        te.setRemoveBondStereo(true);
        te.setRemoveSp3Stereo(true);
        
        std::unique_ptr<ROMol> ms_ready_mol(te.canonicalize(*charge_parent));

        // 5. Final SMILES generation (isomeric=false, canonical=true)
        return MolToSmiles(*ms_ready_mol, {false, true});
    } catch (...) {
        return "";
    }
}

template <typename Func, typename R = std::invoke_result_t<Func, std::string>>
std::vector<R> process_parallel(const std::vector<std::string>& inputs, Func func) {
    size_t n = inputs.size();
    std::vector<R> results(n);

    // Use static schedule with chunk size of 500 as requested
    #pragma omp parallel for schedule(static, 500)
    for (size_t i = 0; i < n; ++i) {
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
    return process_parallel(smiles, [](const std::string& s) {
        std::unique_ptr<ROMol> mol(SmilesToMol(s));
        return mol ? MolToSmiles(*mol, {false, true}) : std::string("");
    });
}

std::vector<std::string> inchi_to_smiles_parallel(const std::vector<std::string>& inchis) {
    return process_parallel(inchis, [](const std::string& inchi) {
        ExtraInchiReturnValues rv;
        std::unique_ptr<ROMol> mol(InchiToMol(inchi, rv));
        return mol ? MolToSmiles(*mol, {false, true}) : std::string("");
    });
}

std::vector<std::string> smiles_to_inchi_parallel(const std::vector<std::string>& smiles) {
    return process_parallel(smiles, [](const std::string& s) {
        std::unique_ptr<ROMol> mol(SmilesToMol(s));
        if (!mol) return std::string("");
        ExtraInchiReturnValues rv;
        return MolToInchi(*mol, rv);
    });
}

std::vector<std::string> smiles_to_inchikey_parallel(const std::vector<std::string>& smiles) {
    return process_parallel(smiles, [](const std::string& s) {
        std::unique_ptr<ROMol> mol(SmilesToMol(s));
        if (!mol) return std::string("");
        return MolToInchiKey(*mol);
    });
}

std::vector<std::tuple<std::string, std::string, std::string>> msready_inchi_inchikey_parallel(const std::vector<std::string>& smiles) {
    return process_parallel(smiles, [](const std::string& s) -> std::tuple<std::string, std::string, std::string> {
        std::unique_ptr<RWMol> mol(SmilesToMol(s));
        if (!mol) return {"", "", ""};
        
        std::string inchi = "";
        std::string inchikey = "";
        std::string msready = "";
        
        try {
            ExtraInchiReturnValues rv;
            inchi = MolToInchi(*mol, rv);
            inchikey = MolToInchiKey(*mol);
        } catch (...) {}
        
        try {
            std::unique_ptr<RWMol> clean_mol(MolStandardize::cleanup(*mol));
            std::unique_ptr<RWMol> frag_parent(MolStandardize::fragmentParent(*clean_mol));
            std::unique_ptr<RWMol> charge_parent(MolStandardize::chargeParent(*frag_parent));
            MolStandardize::TautomerEnumerator te;
            te.setRemoveBondStereo(true);
            te.setRemoveSp3Stereo(true);
            std::unique_ptr<ROMol> ms_ready_mol(te.canonicalize(*charge_parent));
            msready = MolToSmiles(*ms_ready_mol, {false, true});
        } catch (...) {}
        
        return {msready, inchi, inchikey};
    });
}

} // namespace parallel_rdkit
