#include "screen_smarts.hpp"
#include <GraphMol/RDKitBase.h>
#include <GraphMol/SmilesParse/SmilesParse.h>
#include <GraphMol/Fingerprints/Fingerprints.h>

// Suppress C++20 deprecation warning about implicit 'this' capture in RDKit headers
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated"
#elif defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated"
#endif

#include <GraphMol/SubstructLibrary/SubstructLibrary.h>

#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic pop
#elif defined(__clang__)
#pragma clang diagnostic pop
#endif

#include <DataStructs/ExplicitBitVect.h>

#include <iostream>
#include <fstream>
#include <sstream>
#include <memory>
#include <omp.h>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/serialization/vector.hpp>
#include <chrono>

namespace parallel_rdkit {

using namespace RDKit;

// Simple hash function for cache invalidation
std::string compute_file_hash(const std::string& filepath) {
    std::ifstream file(filepath, std::ios::binary);
    if (!file) return "";
    
    // Use file size + modification time as a simple hash
    file.seekg(0, std::ios::end);
    size_t size = file.tellg();
    file.seekg(0, std::ios::beg);
    
    // Simple XOR hash of first 1KB and last 1KB
    size_t hash = size;
    char buffer[1024];
    
    // Read first 1KB
    file.read(buffer, std::min(size_t(1024), size));
    for (size_t i = 0; i < file.gcount(); ++i) {
        hash = hash * 31 + static_cast<unsigned char>(buffer[i]);
    }
    
    // Read last 1KB if file is large enough
    if (size > 2048) {
        file.seekg(-1024, std::ios::end);
        file.read(buffer, 1024);
        for (size_t i = 0; i < file.gcount(); ++i) {
            hash = hash * 31 + static_cast<unsigned char>(buffer[i]);
        }
    }
    
    return std::to_string(hash);
}

// Cache structure
struct SmartsCache {
    std::string smiles_file_hash;
    std::vector<std::string> smarts_list;
    std::vector<std::vector<uint8_t>> matrix;
    
    template<class Archive>
    void serialize(Archive& ar, const unsigned int version) {
        ar & smiles_file_hash;
        ar & smarts_list;
        ar & matrix;
    }
};

// Parse SMARTS patterns
std::vector<std::unique_ptr<ROMol>> parse_smarts(const std::vector<std::string>& smarts_list) {
    std::vector<std::unique_ptr<ROMol>> queries;
    queries.reserve(smarts_list.size());
    for (const auto& sm : smarts_list) {
        queries.emplace_back(SmartsToMol(sm));
    }
    return queries;
}

// Count lines in file efficiently
size_t count_lines(const std::string& filepath) {
    std::ifstream file(filepath);
    size_t count = 0;
    std::string line;
    while (std::getline(file, line)) {
        if (!line.empty()) ++count;
    }
    return count;
}

// Process a batch of SMILES and return matches for each SMARTS.
//
// BATCH APPROACH (fixes OOM):
//   1. Parse every SMILES in the batch ONCE (parallel) and compute its pattern
//      fingerprint. KEEP the parsed mol — do not discard it.
//   2. Build a SubstructLibrary with MolHolder (stores parsed mols in memory)
//      + PatternHolder (fingerprint pre-filter). Because MolHolder::getMol()
//      returns the already-parsed mol, there is NO re-parsing during matching.
//
//   The old code used CachedTrustedSmilesMolHolder, which stores raw SMILES
//   strings and re-parses via SmilesToMol() on EVERY getMol() call. Since
//   getMatches() calls getMol() for each of the M SMARTS queries, every
//   molecule was parsed up to M times per batch. That allocation churn grew
//   RSS monotonically across batches until OOM — even in streaming mode.
//
// Memory is bounded by batch_size: N parsed ROMols + N fingerprints + the
// N x M result matrix. Everything is freed when process_batch returns.
std::vector<std::vector<uint8_t>> process_batch(
    const std::vector<std::string>& smiles_batch,
    const std::vector<std::unique_ptr<ROMol>>& queries) {
    
    size_t N = smiles_batch.size();
    size_t M = queries.size();
    
    // Phase 1: Parse all SMILES in parallel — once per molecule, no exceptions.
    // Store parsed mols and fingerprints for the library build below.
    std::vector<boost::shared_ptr<ROMol>> parsed_mols(N);
    std::vector<ExplicitBitVect*> computed_fps(N, nullptr);
    
    #pragma omp parallel for schedule(dynamic)
    for (long i = 0; i < static_cast<long>(N); ++i) {
        SmilesParserParams params;
        params.sanitize = false;
        RWMol* m = SmilesToMol(smiles_batch[i], params);
        if (m) {
            try {
                m->updatePropertyCache();  // match CachedTrustedSmilesMolHolder behaviour
            } catch (...) {
                delete m;
                m = nullptr;
            }
        }
        if (m) {
            parsed_mols[i] = boost::shared_ptr<ROMol>(m);
            computed_fps[i] = PatternFingerprintMol(*m);
        }
    }
    
    // Phase 2: Build SubstructLibrary with MolHolder (parsed mols kept in
    // memory — getMol returns the shared_ptr, no SmilesToMol re-parse) and
    // PatternHolder (fingerprint filter skips impossible matches cheaply).
    boost::shared_ptr<MolHolder> molHolder(new MolHolder());
    boost::shared_ptr<PatternHolder> fps(new PatternHolder());
    molHolder->getMols().reserve(N);
    
    for (size_t i = 0; i < N; ++i) {
        if (parsed_mols[i]) {
            molHolder->getMols().push_back(parsed_mols[i]);
            if (computed_fps[i]) {
                fps->addFingerprint(computed_fps[i]);
            } else {
                fps->addFingerprint(new ExplicitBitVect(2048));
            }
        } else {
            // Empty placeholder keeps index alignment with smiles_batch.
            molHolder->getMols().push_back(boost::make_shared<ROMol>());
            fps->addFingerprint(new ExplicitBitVect(2048));
        }
    }
    // Release our references — the library now owns the mols/fingerprints.
    parsed_mols.clear();
    
    SubstructLibrary lib(molHolder, fps);
    
    // Phase 3: Run each SMARTS against the entire batch. getMatches uses the
    // fingerprint pre-filter and reads pre-parsed mols (zero re-parsing).
    // maxResults=-1 returns ALL matching molecule indices — no short-circuit.
    // numThreads=-1 lets SubstructLibrary parallelise internally.
    std::vector<std::vector<uint8_t>> bit_matrix(N, std::vector<uint8_t>(M, 0));
    for (size_t j = 0; j < M; ++j) {
        if (!queries[j]) continue;
        std::vector<unsigned int> matches = lib.getMatches(
            *queries[j],
            false,  // recursionPossible
            false,  // useChirality
            false,  // useQueryQueryMatches
            -1,     // numThreads (all)
            -1      // maxResults (all matching molecules)
        );
        for (unsigned int match_idx : matches) {
            bit_matrix[match_idx][j] = 1;
        }
    }
    
    return bit_matrix;
}

std::vector<std::vector<uint8_t>> screen_smarts_direct(
    const std::string& smiles_file,
    const std::vector<std::string>& smarts_list,
    const std::string& cache_path) {
    
    // Check cache if path provided
    if (!cache_path.empty()) {
        std::ifstream cache_in(cache_path, std::ios::binary);
        if (cache_in) {
            try {
                boost::archive::binary_iarchive ia(cache_in);
                SmartsCache cache;
                ia >> cache;
                
                // Verify cache validity
                if (cache.smarts_list == smarts_list && 
                    cache.smiles_file_hash == compute_file_hash(smiles_file)) {
                    return cache.matrix;
                }
            } catch (...) {
                // Cache invalid, continue with computation
            }
        }
    }
    
    // Read all SMILES
    std::vector<std::string> smiles_list;
    {
        std::ifstream infile(smiles_file);
        std::string line;
        while (std::getline(infile, line)) {
            if (!line.empty()) {
                smiles_list.push_back(line);
            }
        }
    }
    
    // Parse SMARTS
    auto queries = parse_smarts(smarts_list);

    // Process in batches to bound peak memory (only ~numThreads live mols).
    constexpr size_t BATCH_SIZE = 64000;
    std::vector<std::vector<uint8_t>> result;
    result.reserve(smiles_list.size());

    for (size_t start = 0; start < smiles_list.size(); start += BATCH_SIZE) {
        size_t end = std::min(start + BATCH_SIZE, smiles_list.size());
        std::vector<std::string> batch(smiles_list.begin() + start, smiles_list.begin() + end);
        auto batch_result = process_batch(batch, queries);
        result.insert(result.end(),
                      std::make_move_iterator(batch_result.begin()),
                      std::make_move_iterator(batch_result.end()));
    }
    
    // Save cache if path provided
    if (!cache_path.empty()) {
        try {
            std::ofstream cache_out(cache_path, std::ios::binary);
            boost::archive::binary_oarchive oa(cache_out);
            SmartsCache cache{compute_file_hash(smiles_file), smarts_list, result};
            oa << cache;
        } catch (...) {
            // Failed to write cache, ignore
        }
    }
    
    return result;
}

size_t screen_smarts_streaming(
    const std::string& smiles_file,
    const std::vector<std::string>& smarts_list,
    int batch_size,
    const std::string& cache_path,
    const std::string& output_path) {
    
    // Parse SMARTS once
    auto queries = parse_smarts(smarts_list);
    size_t M = smarts_list.size();
    
    // Open input file
    std::ifstream infile(smiles_file);
    if (!infile) {
        throw std::runtime_error("Cannot open SMILES file: " + smiles_file);
    }
    
    // Open output file for writing numpy array
    std::ofstream outfile(output_path, std::ios::binary);
    if (!outfile) {
        throw std::runtime_error("Cannot open output file: " + output_path);
    }
    
    // Count total lines first for numpy header
    size_t total_mols = count_lines(smiles_file);
    
    // Write numpy header
    // Format: \x93NUMPY + version (1.0) + header_len + header_dict
    outfile.write("\x93NUMPY", 6);
    outfile.write("\x01\x00", 2);  // version 1.0
    
    // Build header dictionary
    std::string header = "{'descr': '|b1', 'fortran_order': False, 'shape': (" + 
                        std::to_string(total_mols) + ", " + std::to_string(M) + "), }";
    // Pad to 64-byte alignment
    size_t header_len = header.length();
    size_t padding = 64 - ((8 + 2 + 2 + header_len + 1) % 64);
    if (padding == 64) padding = 0;
    header += std::string(padding, ' ') + "\n";
    header_len = header.length();
    
    uint16_t len = static_cast<uint16_t>(header_len);
    outfile.write(reinterpret_cast<const char*>(&len), 2);
    outfile.write(header.c_str(), header_len);
    
    // Process in batches
    std::vector<std::string> batch;
    batch.reserve(batch_size);
    size_t total_processed = 0;
    std::string line;
    
    while (std::getline(infile, line)) {
        if (line.empty()) continue;
        batch.push_back(line);
        
        if (batch.size() >= static_cast<size_t>(batch_size)) {
            auto matrix = process_batch(batch, queries);
            // Write results
            for (const auto& row : matrix) {
                outfile.write(reinterpret_cast<const char*>(row.data()), row.size());
            }
            total_processed += batch.size();
            batch.clear();
        }
    }
    
    // Process remaining
    if (!batch.empty()) {
        auto matrix = process_batch(batch, queries);
        for (const auto& row : matrix) {
            outfile.write(reinterpret_cast<const char*>(row.data()), row.size());
        }
        total_processed += batch.size();
    }
    
    // Save cache metadata (just the hash and params, not the data)
    if (!cache_path.empty()) {
        try {
            std::ofstream cache_out(cache_path + ".meta", std::ios::binary);
            boost::archive::binary_oarchive oa(cache_out);
            SmartsCache cache{compute_file_hash(smiles_file), smarts_list, {}};
            oa << cache;
        } catch (...) {
            // Failed to write cache, ignore
        }
    }
    
    return total_processed;
}

} // namespace parallel_rdkit
