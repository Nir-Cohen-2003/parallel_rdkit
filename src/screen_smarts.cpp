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

// Process a batch of SMILES and return matches for each SMARTS
std::vector<std::vector<uint8_t>> process_batch(
    const std::vector<std::string>& smiles_batch,
    const std::vector<std::unique_ptr<ROMol>>& queries) {
    
    size_t N = smiles_batch.size();
    size_t M = queries.size();
    
    // Generate fingerprints in parallel
    std::vector<ExplicitBitVect*> computed_fps(N, nullptr);
    
    #pragma omp parallel for schedule(dynamic)
    for (size_t i = 0; i < N; ++i) {
        SmilesParserParams params;
        params.sanitize = false;
        std::unique_ptr<ROMol> mol(SmilesToMol(smiles_batch[i], params));
        if (mol) {
            computed_fps[i] = PatternFingerprintMol(*mol);
        }
    }
    
    // Build SubstructLibrary for this batch
    boost::shared_ptr<CachedTrustedSmilesMolHolder> mols(new CachedTrustedSmilesMolHolder());
    boost::shared_ptr<PatternHolder> fps(new PatternHolder());
    
    for (size_t i = 0; i < N; ++i) {
        mols->addSmiles(smiles_batch[i]);
        if (computed_fps[i]) {
            fps->addFingerprint(computed_fps[i]);
        } else {
            fps->addFingerprint(new ExplicitBitVect(2048));
        }
    }
    
    SubstructLibrary lib(mols, fps);
    
    // Query each SMARTS
    std::vector<std::vector<uint8_t>> bit_matrix(N, std::vector<uint8_t>(M, 0));
    for (size_t j = 0; j < M; ++j) {
        if (!queries[j]) continue;
        // Use explicit types to avoid ambiguity
        std::vector<unsigned int> matches = lib.getMatches(
            *queries[j], 
            false,  // recursionPossible
            false,  // useChirality  
            false,  // useQueryQueryMatches
            -1,     // numThreads
            -1      // maxResults
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
    
    // Process all at once
    auto result = process_batch(smiles_list, queries);
    
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
