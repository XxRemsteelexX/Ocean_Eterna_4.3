// test_binary_manifest.cpp - Test program to verify binary manifest loading
// Compile with: g++ -std=c++17 -O2 -o test_binary_manifest test_binary_manifest.cpp

#include <iostream>
#include "json.hpp"
#include "binary_manifest.hpp"

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cout << "Usage: " << argv[0] << " <manifest.bin>" << std::endl;
        return 1;
    }

    std::string bin_path = argv[1];

    std::cout << "Loading binary manifest: " << bin_path << std::endl;
    std::cout << std::endl;

    std::unordered_map<std::string, uint32_t> chunk_id_map;
    BinaryCorpus corpus = load_binary_manifest(bin_path, &chunk_id_map);

    if (corpus.docs.empty()) {
        std::cerr << "Failed to load binary manifest!" << std::endl;
        return 1;
    }

    std::cout << std::endl;
    std::cout << "=== Verification ===" << std::endl;
    std::cout << "Documents loaded: " << corpus.docs.size() << std::endl;
    std::cout << "Inverted index size: " << corpus.inverted_index.size() << " keywords" << std::endl;
    std::cout << "Chunk ID map size: " << chunk_id_map.size() << " entries" << std::endl;
    std::cout << "Average document length: " << corpus.avgdl << std::endl;
    std::cout << std::endl;

    // Show first 3 documents
    std::cout << "=== First 3 Documents ===" << std::endl;
    for (size_t i = 0; i < std::min(size_t(3), corpus.docs.size()); ++i) {
        const auto& doc = corpus.docs[i];
        std::cout << "Doc " << i << ":" << std::endl;
        std::cout << "  ID: " << doc.id << std::endl;
        std::cout << "  Summary: " << doc.summary.substr(0, 80) << "..." << std::endl;
        std::cout << "  Offset: " << doc.offset << ", Length: " << doc.length << std::endl;
        std::cout << "  Tokens: " << doc.start << " - " << doc.end << std::endl;
        std::cout << "  Keywords: ";
        for (size_t k = 0; k < std::min(size_t(5), doc.keywords.size()); ++k) {
            std::cout << doc.keywords[k];
            if (k < doc.keywords.size() - 1) std::cout << ", ";
        }
        if (doc.keywords.size() > 5) std::cout << "...";
        std::cout << std::endl;
        std::cout << std::endl;
    }

    // Check inverted index
    std::cout << "=== Sample Inverted Index Entry ===" << std::endl;
    if (!corpus.inverted_index.empty()) {
        auto it = corpus.inverted_index.begin();
        std::cout << "Keyword: \"" << it->first << "\" -> " << it->second.size() << " documents" << std::endl;
    }

    std::cout << std::endl;
    std::cout << "Binary manifest test PASSED!" << std::endl;

    return 0;
}
