// binary_manifest.hpp - Binary Manifest Format for OceanEterna
// Speed Improvement #1: Reduces manifest loading from 41s to ~4-8s (5-10x faster)
//
// Binary Format Specification:
// HEADER (24 bytes):
//   magic[4]        = "OEM1"
//   version[4]      = 1
//   chunk_count[8]  = number of chunks
//   keyword_count[8]= number of unique keywords
//
// KEYWORD DICTIONARY:
//   For each keyword:
//     len[2] + string[len]
//
// CHUNK ENTRIES:
//   For each chunk:
//     id_len[2] + id[id_len]
//     summary_len[2] + summary[summary_len]
//     offset[8]
//     length[8]
//     token_start[4]
//     token_end[4]
//     timestamp[8]
//     kw_count[2]
//     kw_indices[kw_count * 4]  // indices into keyword dictionary

#ifndef BINARY_MANIFEST_HPP
#define BINARY_MANIFEST_HPP

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <unordered_map>
#include <cstring>
#include <chrono>
#include <stdexcept>
#include <sys/stat.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>

// Magic header for binary manifest files
constexpr char BINARY_MANIFEST_MAGIC[4] = {'O', 'E', 'M', '1'};
constexpr uint32_t BINARY_MANIFEST_VERSION = 1;

// Forward declaration of DocMeta if not already defined
#ifndef DOCMETA_DEFINED
#define DOCMETA_DEFINED
struct DocMeta {
    std::string id;
    std::string summary;            // low-mem: empty in RAM, still written to disk manifest
    std::vector<uint32_t> keyword_ids; // low-mem: indices into g_keyword_dict (was vector<string>)
    uint64_t offset;
    uint64_t length;
    uint32_t start;
    uint32_t end;
    long long timestamp = 0;
    // v4.2: Chunk cross-references for context expansion
    std::string source_file;        // original filename
    std::string prev_chunk_id;      // previous chunk in same document
    std::string next_chunk_id;      // next chunk in same document
    std::string doc_section;        // hierarchical section path
    std::string original_file_id;   // links back to originals.json entry
};
#endif

// Binary manifest header structure (24 bytes)
struct BinaryManifestHeader {
    char magic[4];
    uint32_t version;
    uint64_t chunk_count;
    uint64_t keyword_count;
};

// Helper functions for binary I/O
namespace BinaryIO {

    // Write a value in little-endian format
    template<typename T>
    inline void write_le(std::ostream& out, T value) {
        out.write(reinterpret_cast<const char*>(&value), sizeof(T));
    }

    // Read a value in little-endian format
    template<typename T>
    inline T read_le(std::istream& in) {
        T value;
        in.read(reinterpret_cast<char*>(&value), sizeof(T));
        return value;
    }

    // Write a length-prefixed string (2-byte length)
    inline void write_string(std::ostream& out, const std::string& str) {
        uint16_t len = static_cast<uint16_t>(std::min(str.size(), size_t(65535)));
        write_le<uint16_t>(out, len);
        out.write(str.data(), len);
    }

    // Read a length-prefixed string (2-byte length)
    inline std::string read_string(std::istream& in) {
        uint16_t len = read_le<uint16_t>(in);
        std::string str(len, '\0');
        in.read(&str[0], len);
        return str;
    }
}

// low-mem: Writer functions only needed by the standalone converter tool
#ifdef BINARY_MANIFEST_WRITER
// Build keyword dictionary from documents (old-style string keywords)
// Only used by converter tool, not by the server
inline std::pair<std::unordered_map<std::string, uint32_t>, std::vector<std::string>>
build_keyword_dictionary(const std::vector<DocMeta>& docs) {
    std::unordered_map<std::string, uint32_t> kw_to_index;
    std::vector<std::string> keywords;

    // Writer uses keyword_ids + dictionary for resolution
    // This is only called from the converter which manages its own data
    return {kw_to_index, keywords};
}
#endif

// low-mem: write_binary_manifest only needed by standalone converter
#ifdef BINARY_MANIFEST_WRITER
// Write binary manifest file from vector of DocMeta
// Returns: true on success, false on failure
inline bool write_binary_manifest(const std::string& output_path,
                                   const std::vector<DocMeta>& docs) {
    // stub — converter has its own implementation
    return false;
}
#endif

// Corpus structure for loading (matches main server structure)
struct BinaryCorpus {
    std::vector<DocMeta> docs;
    std::unordered_map<std::string, std::vector<uint32_t>> inverted_index;
    // low-mem: keyword dictionary — keyword ID → string, shared across all docs
    std::vector<std::string> keyword_dict;
    std::unordered_map<std::string, uint32_t> keyword_to_id;
    size_t total_tokens = 0;
    double avgdl = 0;
};

// Load binary manifest file
// Returns: populated Corpus structure
// Also populates chunk_id_to_index map if provided
inline BinaryCorpus load_binary_manifest(const std::string& input_path,
                                          std::unordered_map<std::string, uint32_t>* chunk_id_map = nullptr) {
    using namespace BinaryIO;

    BinaryCorpus corpus;

    std::ifstream in(input_path, std::ios::binary);
    if (!in.is_open()) {
        std::cerr << "Failed to open binary manifest: " << input_path << std::endl;
        return corpus;
    }

    std::cout << "Loading binary manifest..." << std::flush;
    auto start = std::chrono::high_resolution_clock::now();

    // Read and validate header
    BinaryManifestHeader header;
    in.read(reinterpret_cast<char*>(&header), sizeof(header));

    if (std::memcmp(header.magic, BINARY_MANIFEST_MAGIC, 4) != 0) {
        std::cerr << "\nInvalid binary manifest: bad magic header" << std::endl;
        return corpus;
    }

    if (header.version != BINARY_MANIFEST_VERSION) {
        std::cerr << "\nUnsupported binary manifest version: " << header.version << std::endl;
        return corpus;
    }

    // Pre-allocate vectors for performance
    corpus.docs.reserve(header.chunk_count);
    corpus.keyword_dict.reserve(header.keyword_count);

    // Read keyword dictionary into corpus
    for (uint64_t i = 0; i < header.keyword_count; ++i) {
        std::string kw = read_string(in);
        corpus.keyword_to_id[kw] = static_cast<uint32_t>(corpus.keyword_dict.size());
        corpus.keyword_dict.push_back(std::move(kw));
    }

    // Read chunk entries
    for (uint64_t i = 0; i < header.chunk_count; ++i) {
        DocMeta doc;

        // Read chunk ID
        doc.id = read_string(in);

        // low-mem: skip summary (read and discard)
        read_string(in);

        // Read fixed fields
        doc.offset = read_le<uint64_t>(in);
        doc.length = read_le<uint64_t>(in);
        doc.start = read_le<uint32_t>(in);
        doc.end = read_le<uint32_t>(in);
        doc.timestamp = static_cast<long long>(read_le<int64_t>(in));

        // low-mem: store keyword indices directly (not resolved to strings)
        uint16_t kw_count = read_le<uint16_t>(in);
        doc.keyword_ids.reserve(kw_count);

        for (uint16_t k = 0; k < kw_count; ++k) {
            uint32_t kw_idx = read_le<uint32_t>(in);
            if (kw_idx < corpus.keyword_dict.size()) {
                doc.keyword_ids.push_back(kw_idx);
            }
        }

        corpus.docs.push_back(std::move(doc));

        // Build inverted index using resolved keyword strings from dictionary
        size_t doc_idx = corpus.docs.size() - 1;
        for (uint32_t kid : corpus.docs.back().keyword_ids) {
            corpus.inverted_index[corpus.keyword_dict[kid]].push_back(static_cast<uint32_t>(doc_idx));
        }

        // Build chunk_id to index mapping if provided
        if (chunk_id_map) {
            (*chunk_id_map)[corpus.docs.back().id] = static_cast<uint32_t>(doc_idx);
        }

        // Track total tokens
        corpus.total_tokens += (corpus.docs.back().end - corpus.docs.back().start);
    }

    // Calculate average document length
    if (!corpus.docs.empty()) {
        corpus.avgdl = corpus.total_tokens / static_cast<double>(corpus.docs.size());
    }

    auto end = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double, std::milli>(end - start).count();

    std::cout << " Done!" << std::endl;
    std::cout << "Loaded " << corpus.docs.size() << " chunks in " << elapsed << "ms" << std::endl;
    std::cout << "Total tokens: " << corpus.total_tokens << std::endl;
    std::cout << "Unique keywords: " << corpus.keyword_dict.size() << std::endl;

    return corpus;
}

// Memory-mapped binary manifest loader (faster than ifstream, less memory overhead)
inline BinaryCorpus load_binary_manifest_mmap(const std::string& input_path,
                                               std::unordered_map<std::string, uint32_t>* chunk_id_map = nullptr) {
    BinaryCorpus corpus;

    // Open file and get size
    int fd = open(input_path.c_str(), O_RDONLY);
    if (fd < 0) {
        std::cerr << "Failed to open binary manifest: " << input_path << std::endl;
        return corpus;
    }

    struct stat st;
    if (fstat(fd, &st) != 0) {
        std::cerr << "Failed to stat binary manifest" << std::endl;
        close(fd);
        return corpus;
    }
    size_t file_size = st.st_size;

    // mmap the file
    void* mapped = mmap(nullptr, file_size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (mapped == MAP_FAILED) {
        std::cerr << "Failed to mmap binary manifest" << std::endl;
        close(fd);
        return corpus;
    }

    // Advise kernel for sequential access
    madvise(mapped, file_size, MADV_SEQUENTIAL);

    std::cout << "Loading binary manifest (mmap)..." << std::flush;
    auto start = std::chrono::high_resolution_clock::now();

    const uint8_t* ptr = static_cast<const uint8_t*>(mapped);
    const uint8_t* end_ptr = ptr + file_size;

    // Helper lambdas for reading from mapped memory
    auto read_u16 = [&]() -> uint16_t {
        uint16_t v;
        memcpy(&v, ptr, 2);
        ptr += 2;
        return v;
    };
    auto read_u32 = [&]() -> uint32_t {
        uint32_t v;
        memcpy(&v, ptr, 4);
        ptr += 4;
        return v;
    };
    auto read_u64 = [&]() -> uint64_t {
        uint64_t v;
        memcpy(&v, ptr, 8);
        ptr += 8;
        return v;
    };
    auto read_i64 = [&]() -> int64_t {
        int64_t v;
        memcpy(&v, ptr, 8);
        ptr += 8;
        return v;
    };
    auto read_str = [&]() -> std::string {
        uint16_t len = read_u16();
        std::string s(reinterpret_cast<const char*>(ptr), len);
        ptr += len;
        return s;
    };

    // Read and validate header
    BinaryManifestHeader header;
    memcpy(&header, ptr, sizeof(header));
    ptr += sizeof(header);

    if (std::memcmp(header.magic, BINARY_MANIFEST_MAGIC, 4) != 0) {
        std::cerr << "\nInvalid binary manifest: bad magic header" << std::endl;
        munmap(mapped, file_size);
        close(fd);
        return corpus;
    }

    if (header.version != BINARY_MANIFEST_VERSION) {
        std::cerr << "\nUnsupported binary manifest version: " << header.version << std::endl;
        munmap(mapped, file_size);
        close(fd);
        return corpus;
    }

    // Pre-allocate
    corpus.docs.reserve(header.chunk_count);
    corpus.keyword_dict.reserve(header.keyword_count);

    // Read keyword dictionary into corpus (shared across all docs)
    for (uint64_t i = 0; i < header.keyword_count; ++i) {
        std::string kw = read_str();
        corpus.keyword_to_id[kw] = static_cast<uint32_t>(corpus.keyword_dict.size());
        corpus.keyword_dict.push_back(std::move(kw));
    }

    // Read chunk entries
    for (uint64_t i = 0; i < header.chunk_count; ++i) {
        DocMeta doc;
        doc.id = read_str();

        // low-mem: skip summary — read length and advance pointer without storing
        {
            uint16_t slen = read_u16();
            ptr += slen;
        }

        doc.offset = read_u64();
        doc.length = read_u64();
        doc.start = read_u32();
        doc.end = read_u32();
        doc.timestamp = static_cast<long long>(read_i64());

        // low-mem: store keyword indices directly (not resolved to strings)
        uint16_t kw_count = read_u16();
        doc.keyword_ids.reserve(kw_count);
        for (uint16_t k = 0; k < kw_count; ++k) {
            uint32_t kw_idx = read_u32();
            if (kw_idx < corpus.keyword_dict.size()) {
                doc.keyword_ids.push_back(kw_idx);
            }
        }

        corpus.docs.push_back(std::move(doc));

        // Build inverted index using resolved keyword strings from dictionary
        size_t doc_idx = corpus.docs.size() - 1;
        for (uint32_t kid : corpus.docs.back().keyword_ids) {
            corpus.inverted_index[corpus.keyword_dict[kid]].push_back(static_cast<uint32_t>(doc_idx));
        }
        if (chunk_id_map) {
            (*chunk_id_map)[corpus.docs.back().id] = static_cast<uint32_t>(doc_idx);
        }
        corpus.total_tokens += (corpus.docs.back().end - corpus.docs.back().start);
    }

    if (!corpus.docs.empty()) {
        corpus.avgdl = corpus.total_tokens / static_cast<double>(corpus.docs.size());
    }

    auto elapsed = std::chrono::duration<double, std::milli>(
        std::chrono::high_resolution_clock::now() - start).count();

    std::cout << " Done!" << std::endl;
    std::cout << "Loaded " << corpus.docs.size() << " chunks in " << elapsed << "ms" << std::endl;
    std::cout << "Total tokens: " << corpus.total_tokens << std::endl;
    std::cout << "Unique keywords: " << corpus.keyword_dict.size() << std::endl;

    // Release mapped pages back to the OS (data already copied into structures)
    madvise(mapped, file_size, MADV_DONTNEED);
    munmap(mapped, file_size);
    close(fd);

    return corpus;
}

// Check if a binary manifest exists and is newer than the JSONL source
inline bool binary_manifest_is_current(const std::string& binary_path,
                                        const std::string& jsonl_path) {
    struct stat binary_stat, jsonl_stat;

    // Check if binary file exists
    if (stat(binary_path.c_str(), &binary_stat) != 0) {
        return false;  // Binary doesn't exist
    }

    // Check if JSONL file exists
    if (stat(jsonl_path.c_str(), &jsonl_stat) != 0) {
        return true;  // JSONL doesn't exist, use binary if available
    }

    // Compare modification times
    return binary_stat.st_mtime >= jsonl_stat.st_mtime;
}

// Get the binary manifest path for a given JSONL path
inline std::string get_binary_manifest_path(const std::string& jsonl_path) {
    // Replace .jsonl extension with .bin
    std::string bin_path = jsonl_path;
    size_t pos = bin_path.rfind(".jsonl");
    if (pos != std::string::npos) {
        bin_path.replace(pos, 6, ".bin");
    } else {
        bin_path += ".bin";
    }
    return bin_path;
}

// Convert JSONL manifest to binary format
// This function requires json.hpp to be included before this header
#ifdef NLOHMANN_JSON_HPP
inline bool convert_jsonl_to_binary(const std::string& jsonl_path,
                                     const std::string& binary_path) {
    using json = nlohmann::json;

    std::ifstream in(jsonl_path);
    if (!in.is_open()) {
        std::cerr << "Failed to open JSONL manifest: " << jsonl_path << std::endl;
        return false;
    }

    std::cout << "Reading JSONL manifest: " << jsonl_path << std::endl;
    auto start = std::chrono::high_resolution_clock::now();

    std::vector<DocMeta> docs;
    std::string line;
    size_t line_count = 0;
    size_t error_count = 0;

    while (std::getline(in, line)) {
        if (line.empty()) continue;
        line_count++;

        try {
            json obj = json::parse(line);

            DocMeta doc;
            doc.id = obj.value("chunk_id", "");
            doc.summary = obj.value("summary", "");
            doc.offset = obj.value("offset", 0ULL);
            doc.length = obj.value("length", 0ULL);
            doc.start = obj.value("token_start", 0U);
            doc.end = obj.value("token_end", 0U);
            doc.timestamp = obj.value("timestamp", 0LL);

            if (obj.contains("keywords") && obj["keywords"].is_array()) {
                for (const auto& kw : obj["keywords"]) {
                    doc.keywords.push_back(kw.get<std::string>());
                }
            }

            docs.push_back(std::move(doc));

            // Progress indicator every 100k lines
            if (line_count % 100000 == 0) {
                std::cout << "  Read " << line_count << " lines..." << std::endl;
            }
        } catch (const std::exception& e) {
            error_count++;
            if (error_count <= 5) {
                std::cerr << "Error parsing line " << line_count << ": " << e.what() << std::endl;
            }
        }
    }

    auto read_end = std::chrono::high_resolution_clock::now();
    double read_elapsed = std::chrono::duration<double>(read_end - start).count();

    std::cout << "Read " << docs.size() << " chunks in " << read_elapsed << " seconds" << std::endl;
    if (error_count > 0) {
        std::cout << "Skipped " << error_count << " lines with errors" << std::endl;
    }

    // Write binary manifest
    std::cout << "\nWriting binary manifest: " << binary_path << std::endl;
    auto write_start = std::chrono::high_resolution_clock::now();

    bool success = write_binary_manifest(binary_path, docs);

    auto write_end = std::chrono::high_resolution_clock::now();
    double write_elapsed = std::chrono::duration<double>(write_end - write_start).count();

    if (success) {
        std::cout << "Wrote binary manifest in " << write_elapsed << " seconds" << std::endl;

        // Report file sizes
        struct stat jsonl_stat, binary_stat;
        if (stat(jsonl_path.c_str(), &jsonl_stat) == 0 &&
            stat(binary_path.c_str(), &binary_stat) == 0) {
            double jsonl_mb = jsonl_stat.st_size / (1024.0 * 1024.0);
            double binary_mb = binary_stat.st_size / (1024.0 * 1024.0);
            double ratio = binary_stat.st_size / (double)jsonl_stat.st_size * 100.0;
            std::cout << "\nFile sizes:" << std::endl;
            std::cout << "  JSONL:  " << jsonl_mb << " MB" << std::endl;
            std::cout << "  Binary: " << binary_mb << " MB (" << ratio << "% of original)" << std::endl;
        }
    }

    return success;
}
#endif // NLOHMANN_JSON_HPP

// Converter utility main function
// Can be compiled standalone with: g++ -DBINARY_MANIFEST_CONVERTER -o convert_manifest binary_manifest.hpp -std=c++17
#ifdef BINARY_MANIFEST_CONVERTER

#ifndef NLOHMANN_JSON_HPP
#include "json.hpp"
#endif

// Re-declare convert_jsonl_to_binary for the converter since json.hpp is now included
inline bool converter_convert_jsonl_to_binary(const std::string& jsonl_path,
                                               const std::string& binary_path) {
    using json = nlohmann::json;

    std::ifstream in(jsonl_path);
    if (!in.is_open()) {
        std::cerr << "Failed to open JSONL manifest: " << jsonl_path << std::endl;
        return false;
    }

    std::cout << "Reading JSONL manifest: " << jsonl_path << std::endl;
    auto start = std::chrono::high_resolution_clock::now();

    std::vector<DocMeta> docs;
    std::string line;
    size_t line_count = 0;
    size_t error_count = 0;

    while (std::getline(in, line)) {
        if (line.empty()) continue;
        line_count++;

        try {
            json obj = json::parse(line);

            DocMeta doc;
            doc.id = obj.value("chunk_id", "");
            doc.summary = obj.value("summary", "");
            doc.offset = obj.value("offset", 0ULL);
            doc.length = obj.value("length", 0ULL);
            doc.start = obj.value("token_start", 0U);
            doc.end = obj.value("token_end", 0U);
            doc.timestamp = obj.value("timestamp", 0LL);

            if (obj.contains("keywords") && obj["keywords"].is_array()) {
                for (const auto& kw : obj["keywords"]) {
                    doc.keywords.push_back(kw.get<std::string>());
                }
            }

            docs.push_back(std::move(doc));

            // Progress indicator every 100k lines
            if (line_count % 100000 == 0) {
                std::cout << "  Read " << line_count << " lines..." << std::endl;
            }
        } catch (const std::exception& e) {
            error_count++;
            if (error_count <= 5) {
                std::cerr << "Error parsing line " << line_count << ": " << e.what() << std::endl;
            }
        }
    }

    auto read_end = std::chrono::high_resolution_clock::now();
    double read_elapsed = std::chrono::duration<double>(read_end - start).count();

    std::cout << "Read " << docs.size() << " chunks in " << read_elapsed << " seconds" << std::endl;
    if (error_count > 0) {
        std::cout << "Skipped " << error_count << " lines with errors" << std::endl;
    }

    // Write binary manifest
    std::cout << "\nWriting binary manifest: " << binary_path << std::endl;
    auto write_start = std::chrono::high_resolution_clock::now();

    bool success = write_binary_manifest(binary_path, docs);

    auto write_end = std::chrono::high_resolution_clock::now();
    double write_elapsed = std::chrono::duration<double>(write_end - write_start).count();

    if (success) {
        std::cout << "Wrote binary manifest in " << write_elapsed << " seconds" << std::endl;

        // Report file sizes
        struct stat jsonl_stat, binary_stat;
        if (stat(jsonl_path.c_str(), &jsonl_stat) == 0 &&
            stat(binary_path.c_str(), &binary_stat) == 0) {
            double jsonl_mb = jsonl_stat.st_size / (1024.0 * 1024.0);
            double binary_mb = binary_stat.st_size / (1024.0 * 1024.0);
            double ratio = binary_stat.st_size / (double)jsonl_stat.st_size * 100.0;
            std::cout << "\nFile sizes:" << std::endl;
            std::cout << "  JSONL:  " << jsonl_mb << " MB" << std::endl;
            std::cout << "  Binary: " << binary_mb << " MB (" << ratio << "% of original)" << std::endl;
        }
    }

    return success;
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cout << "OceanEterna Binary Manifest Converter" << std::endl;
        std::cout << "Usage: " << argv[0] << " <input.jsonl> [output.bin]" << std::endl;
        std::cout << std::endl;
        std::cout << "If output.bin is not specified, it will be derived from input path" << std::endl;
        std::cout << "(e.g., manifest.jsonl -> manifest.bin)" << std::endl;
        return 1;
    }

    std::string jsonl_path = argv[1];
    std::string binary_path;

    if (argc >= 3) {
        binary_path = argv[2];
    } else {
        binary_path = get_binary_manifest_path(jsonl_path);
    }

    std::cout << "========================================" << std::endl;
    std::cout << "OceanEterna Binary Manifest Converter" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "Input:  " << jsonl_path << std::endl;
    std::cout << "Output: " << binary_path << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << std::endl;

    auto total_start = std::chrono::high_resolution_clock::now();

    bool success = converter_convert_jsonl_to_binary(jsonl_path, binary_path);

    auto total_end = std::chrono::high_resolution_clock::now();
    double total_elapsed = std::chrono::duration<double>(total_end - total_start).count();

    std::cout << std::endl;
    std::cout << "========================================" << std::endl;
    if (success) {
        std::cout << "Conversion completed in " << total_elapsed << " seconds" << std::endl;
        std::cout << std::endl;
        std::cout << "To use binary manifest in ocean_chat_server:" << std::endl;
        std::cout << "  1. Include binary_manifest.hpp" << std::endl;
        std::cout << "  2. Replace load_manifest() call with load_binary_manifest()" << std::endl;
        std::cout << std::endl;
        std::cout << "Expected speedup: 5-10x (41s -> 4-8s)" << std::endl;
    } else {
        std::cout << "Conversion FAILED" << std::endl;
        return 1;
    }
    std::cout << "========================================" << std::endl;

    return 0;
}
#endif // BINARY_MANIFEST_CONVERTER

#endif // BINARY_MANIFEST_HPP
