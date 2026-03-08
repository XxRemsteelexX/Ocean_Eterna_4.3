/**
 * BM25S Sparse Matrix Optimization Engine
 * ========================================
 *
 * Speed Improvement #2: Pre-computed BM25 Scores
 * Speed Improvement #3: BlockWeakAnd Query Optimization
 *
 * Problem: Traditional BM25 calculates TF-IDF scores for ALL documents at query time.
 *          With 5M documents, this takes ~500ms per query.
 *
 * Solution #2: Pre-compute BM25 contribution for each term in each document during indexing.
 *              At query time, just iterate through query terms and accumulate pre-computed scores.
 *
 * Solution #3: BlockWeakAnd - Divide posting lists into blocks, track max score per block.
 *              Skip blocks that cannot contribute to top-K results (early termination).
 *
 * Performance:
 *   - Search time: 500ms -> 10-50ms (10-50x faster with BM25S)
 *   - BlockWeakAnd: 2-5x additional speedup on top of BM25S
 *   - Memory overhead: ~500MB-1GB for score matrix + ~1-2% for block index
 *   - Indexing time: slightly longer (acceptable tradeoff)
 *
 * Data Structures:
 *   score_matrix: unordered_map<term, vector<pair<doc_id, score>>>
 *     - Only stores non-zero scores (sparse representation)
 *     - Posting lists pre-sorted by score for early termination
 *     - Scores stored as float (4 bytes) to save memory
 *
 *   block_index: unordered_map<term, vector<BlockInfo>>
 *     - Each posting list divided into 128-document blocks
 *     - BlockInfo stores (start_idx, end_idx, max_score)
 *     - Enables skipping blocks during search
 *
 * Author: OceanEterna Team
 * Version: 1.1 (Added BlockWeakAnd)
 */

#ifndef BM25S_ENGINE_HPP
#define BM25S_ENGINE_HPP

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <algorithm>
#include <functional>
#include <cmath>
#include <chrono>
#include <cstring>
#include <sys/stat.h>
#include <omp.h>

namespace bm25s {

// ============================================================================
// Configuration
// ============================================================================

struct BM25SConfig {
    float k1 = 1.5f;          // Term frequency saturation parameter
    float b = 0.75f;          // Length normalization parameter
    int min_df = 1;           // Minimum document frequency to include term
    int max_df_percent = 95;  // Maximum DF as percentage (exclude very common terms)
    bool compress_scores = false;  // Optional: quantize scores to reduce memory
    int score_bits = 16;      // If compressing: bits per score (8 or 16)
};

// ============================================================================
// Posting List Entry: (doc_id, pre-computed score)
// ============================================================================

struct PostingEntry {
    uint32_t doc_id;
    float score;

    PostingEntry() : doc_id(0), score(0.0f) {}
    PostingEntry(uint32_t id, float s) : doc_id(id), score(s) {}

    // For sorting by score (descending) for early termination potential
    bool operator>(const PostingEntry& other) const {
        return score > other.score;
    }
};

// ============================================================================
// Block Info for BlockWeakAnd Optimization (Speed Improvement #3)
// ============================================================================

/**
 * BlockInfo stores metadata for a block of documents in a posting list.
 * Used for early termination during search - blocks with max_score below
 * the current threshold can be skipped entirely.
 */
struct BlockInfo {
    uint32_t start_idx;     // Start index in posting list
    uint32_t end_idx;       // End index in posting list (exclusive)
    float max_score;        // Maximum score in this block

    BlockInfo() : start_idx(0), end_idx(0), max_score(0.0f) {}
    BlockInfo(uint32_t start, uint32_t end, float max_s)
        : start_idx(start), end_idx(end), max_score(max_s) {}
};

// ============================================================================
// BM25S Index: The Pre-computed Score Matrix
// ============================================================================

class BM25SIndex {
public:
    // Score matrix: term -> sorted posting list of (doc_id, pre-computed_score)
    std::unordered_map<std::string, std::vector<PostingEntry>> score_matrix;

    // Block index for early termination (Speed Improvement #3: BlockWeakAnd)
    std::unordered_map<std::string, std::vector<BlockInfo>> block_index;
    static const int BLOCK_SIZE = 128;
    bool block_index_built = false;

    // Metadata
    size_t num_documents = 0;
    size_t num_terms = 0;
    size_t total_postings = 0;
    double avgdl = 0.0;
    BM25SConfig config;

    // Statistics for diagnostics
    double build_time_ms = 0.0;
    size_t memory_bytes = 0;

    // ========================================================================
    // Memory estimation
    // ========================================================================

    size_t estimate_memory() const {
        size_t bytes = 0;

        // Hash map overhead (approximate)
        bytes += score_matrix.bucket_count() * sizeof(void*);

        // For each term
        for (const auto& [term, postings] : score_matrix) {
            // String storage
            bytes += term.capacity() + sizeof(std::string);
            // Vector overhead
            bytes += sizeof(std::vector<PostingEntry>);
            // Posting entries
            bytes += postings.capacity() * sizeof(PostingEntry);
        }

        // Block index memory (if built)
        if (block_index_built) {
            bytes += block_index.bucket_count() * sizeof(void*);
            for (const auto& [term, blocks] : block_index) {
                bytes += term.capacity() + sizeof(std::string);
                bytes += sizeof(std::vector<BlockInfo>);
                bytes += blocks.capacity() * sizeof(BlockInfo);
            }
        }

        return bytes;
    }

    // ========================================================================
    // Clear index
    // ========================================================================

    void clear() {
        score_matrix.clear();
        block_index.clear();
        block_index_built = false;
        num_documents = 0;
        num_terms = 0;
        total_postings = 0;
        avgdl = 0.0;
        build_time_ms = 0.0;
        memory_bytes = 0;
    }

    // ========================================================================
    // Build Block Index for BlockWeakAnd Optimization
    // ========================================================================

    /**
     * Build the block index for early termination during search.
     *
     * Divides each posting list into blocks of BLOCK_SIZE documents,
     * storing the maximum score within each block. During search,
     * blocks with max_score below the current threshold can be skipped.
     *
     * Call this after score_matrix is populated (either after build or load).
     *
     * Performance impact:
     *   - 2-5x speedup on top of BM25S for queries with common terms
     *   - Memory overhead: ~1-2% additional
     */
    void build_block_index() {
        if (score_matrix.empty()) {
            std::cerr << "[BM25S] Cannot build block index: score_matrix is empty" << std::endl;
            return;
        }

        auto start_time = std::chrono::high_resolution_clock::now();
        std::cout << "[BM25S] Building block index (BLOCK_SIZE=" << BLOCK_SIZE << ")..." << std::flush;

        block_index.clear();
        block_index.reserve(score_matrix.size());

        size_t total_blocks = 0;

        // Build blocks for each term's posting list
        #pragma omp parallel for schedule(dynamic, 100) reduction(+:total_blocks)
        for (size_t i = 0; i < score_matrix.bucket_count(); i++) {
            for (auto it = score_matrix.begin(i); it != score_matrix.end(i); ++it) {
                const std::string& term = it->first;
                const std::vector<PostingEntry>& postings = it->second;

                if (postings.empty()) continue;

                std::vector<BlockInfo> blocks;
                blocks.reserve((postings.size() + BLOCK_SIZE - 1) / BLOCK_SIZE);

                // Divide posting list into blocks
                for (size_t start = 0; start < postings.size(); start += BLOCK_SIZE) {
                    size_t end = std::min(start + BLOCK_SIZE, postings.size());

                    // Find max score in this block
                    float max_score = 0.0f;
                    for (size_t j = start; j < end; j++) {
                        if (postings[j].score > max_score) {
                            max_score = postings[j].score;
                        }
                    }

                    blocks.emplace_back(
                        static_cast<uint32_t>(start),
                        static_cast<uint32_t>(end),
                        max_score
                    );
                    total_blocks++;
                }

                blocks.shrink_to_fit();

                #pragma omp critical
                {
                    block_index[term] = std::move(blocks);
                }
            }
        }

        block_index_built = true;

        auto end_time = std::chrono::high_resolution_clock::now();
        double elapsed_ms = std::chrono::duration<double, std::milli>(end_time - start_time).count();

        std::cout << " Done! (" << elapsed_ms << " ms)" << std::endl;
        std::cout << "[BM25S] Block index: " << total_blocks << " blocks for "
                  << block_index.size() << " terms" << std::endl;

        // Update memory estimate
        memory_bytes = estimate_memory();
    }

    // ========================================================================
    // Check if index is built
    // ========================================================================

    bool is_built() const {
        return num_documents > 0 && !score_matrix.empty();
    }
};

// ============================================================================
// Search Result
// ============================================================================

struct BM25SHit {
    uint32_t doc_id;
    float score;

    BM25SHit() : doc_id(0), score(0.0f) {}
    BM25SHit(uint32_t id, float s) : doc_id(id), score(s) {}
};

// ============================================================================
// Build BM25S Index from Corpus
// ============================================================================

/**
 * Build the pre-computed BM25S score matrix from a corpus.
 *
 * Template parameters allow flexibility in document representation.
 *
 * @param documents Vector of documents, each containing keywords
 * @param get_keywords Function to extract keywords from a document
 * @param config BM25S configuration parameters
 * @return Populated BM25SIndex
 *
 * Algorithm:
 *   1. Calculate document frequencies for all terms
 *   2. Calculate average document length
 *   3. For each document and each term:
 *      - Compute BM25 score contribution
 *      - Store in score_matrix[term]
 *   4. Sort each posting list by score (descending)
 */
template<typename DocType, typename GetKeywordsFn>
BM25SIndex build_bm25s_index(
    const std::vector<DocType>& documents,
    GetKeywordsFn get_keywords,
    const BM25SConfig& config = BM25SConfig()
) {
    BM25SIndex index;
    index.config = config;
    index.num_documents = documents.size();

    auto start_time = std::chrono::high_resolution_clock::now();

    const size_t N = documents.size();
    if (N == 0) return index;

    std::cout << "[BM25S] Building index for " << N << " documents..." << std::endl;

    // ========================================================================
    // Phase 1: Calculate document frequencies and average document length
    // ========================================================================

    std::cout << "[BM25S] Phase 1: Calculating document frequencies..." << std::flush;

    std::unordered_map<std::string, uint32_t> doc_freq;
    size_t total_doc_length = 0;

    // Use OpenMP for parallel DF calculation
    std::vector<std::unordered_map<std::string, uint32_t>> thread_df(omp_get_max_threads());
    std::vector<size_t> thread_lengths(omp_get_max_threads(), 0);

    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        auto& local_df = thread_df[tid];
        size_t local_length = 0;

        #pragma omp for schedule(dynamic, 1000)
        for (size_t i = 0; i < N; i++) {
            const auto& keywords = get_keywords(documents[i]);
            local_length += keywords.size();

            // Count unique terms in this document
            std::unordered_set<std::string> seen;
            for (const auto& kw : keywords) {
                std::string term = kw;
                // Lowercase for consistency
                std::transform(term.begin(), term.end(), term.begin(), ::tolower);
                if (seen.insert(term).second) {
                    local_df[term]++;
                }
            }
        }

        thread_lengths[tid] = local_length;
    }

    // Merge thread-local results
    for (int t = 0; t < omp_get_max_threads(); t++) {
        total_doc_length += thread_lengths[t];
        for (const auto& [term, count] : thread_df[t]) {
            doc_freq[term] += count;
        }
    }

    index.avgdl = static_cast<double>(total_doc_length) / N;
    std::cout << " Done. Found " << doc_freq.size() << " unique terms." << std::endl;
    std::cout << "[BM25S] Average document length: " << index.avgdl << std::endl;

    // ========================================================================
    // Phase 2: Filter terms by document frequency
    // ========================================================================

    std::cout << "[BM25S] Phase 2: Filtering terms..." << std::flush;

    uint32_t max_df = static_cast<uint32_t>(N * config.max_df_percent / 100);

    std::vector<std::string> valid_terms;
    for (const auto& [term, df] : doc_freq) {
        if (df >= config.min_df && df <= max_df) {
            valid_terms.push_back(term);
        }
    }

    std::cout << " Done. " << valid_terms.size() << " terms after filtering." << std::endl;

    // Pre-allocate score matrix
    for (const auto& term : valid_terms) {
        index.score_matrix[term].reserve(doc_freq[term]);
    }

    // ========================================================================
    // Phase 3: Compute BM25 scores for each (term, document) pair
    // ========================================================================

    std::cout << "[BM25S] Phase 3: Computing BM25 scores..." << std::flush;

    const float k1 = config.k1;
    const float b = config.b;
    const float avgdl_f = static_cast<float>(index.avgdl);

    // Thread-local posting lists to avoid contention
    std::vector<std::unordered_map<std::string, std::vector<PostingEntry>>> thread_postings(omp_get_max_threads());

    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        auto& local_postings = thread_postings[tid];

        // Pre-allocate local posting lists
        for (const auto& term : valid_terms) {
            local_postings[term].reserve(doc_freq[term] / omp_get_max_threads() + 100);
        }

        #pragma omp for schedule(dynamic, 1000)
        for (size_t doc_id = 0; doc_id < N; doc_id++) {
            const auto& keywords = get_keywords(documents[doc_id]);
            int doc_len = static_cast<int>(keywords.size());

            // Count term frequencies in this document
            std::unordered_map<std::string, int> term_freq;
            for (const auto& kw : keywords) {
                std::string term = kw;
                std::transform(term.begin(), term.end(), term.begin(), ::tolower);
                term_freq[term]++;
            }

            // Pre-compute length normalization factor
            float norm_factor = k1 * (1.0f - b + b * doc_len / avgdl_f);

            // Calculate BM25 score for each term
            for (const auto& [term, tf] : term_freq) {
                auto df_it = doc_freq.find(term);
                if (df_it == doc_freq.end()) continue;

                uint32_t df = df_it->second;
                if (df < config.min_df || df > max_df) continue;

                // IDF: log((N - df + 0.5) / (df + 0.5) + 1.0)
                float idf = std::log((N - df + 0.5f) / (df + 0.5f) + 1.0f);

                // BM25 score contribution: idf * (tf * (k1 + 1)) / (tf + norm)
                float score = idf * (tf * (k1 + 1.0f)) / (tf + norm_factor);

                if (score > 0.0f) {
                    local_postings[term].emplace_back(static_cast<uint32_t>(doc_id), score);
                }
            }
        }
    }

    std::cout << " Done." << std::endl;

    // ========================================================================
    // Phase 4: Merge thread-local postings and sort by score
    // ========================================================================

    std::cout << "[BM25S] Phase 4: Merging and sorting posting lists..." << std::flush;

    size_t total_postings = 0;

    // Merge in parallel
    #pragma omp parallel for schedule(dynamic, 100) reduction(+:total_postings)
    for (size_t i = 0; i < valid_terms.size(); i++) {
        const std::string& term = valid_terms[i];
        auto& main_list = index.score_matrix[term];

        // Collect all entries for this term
        for (int t = 0; t < omp_get_max_threads(); t++) {
            auto it = thread_postings[t].find(term);
            if (it != thread_postings[t].end()) {
                main_list.insert(main_list.end(), it->second.begin(), it->second.end());
            }
        }

        // Sort by score (descending) for early termination potential
        std::sort(main_list.begin(), main_list.end(),
                  [](const PostingEntry& a, const PostingEntry& b) {
                      return a.score > b.score;
                  });

        // Shrink to fit
        main_list.shrink_to_fit();

        total_postings += main_list.size();
    }

    // Clear thread-local storage
    thread_postings.clear();
    thread_postings.shrink_to_fit();

    std::cout << " Done." << std::endl;

    // ========================================================================
    // Finalize
    // ========================================================================

    index.num_terms = index.score_matrix.size();
    index.total_postings = total_postings;

    auto end_time = std::chrono::high_resolution_clock::now();
    index.build_time_ms = std::chrono::duration<double, std::milli>(end_time - start_time).count();
    index.memory_bytes = index.estimate_memory();

    std::cout << "[BM25S] Index built successfully!" << std::endl;
    std::cout << "[BM25S] Statistics:" << std::endl;
    std::cout << "  - Documents: " << index.num_documents << std::endl;
    std::cout << "  - Terms: " << index.num_terms << std::endl;
    std::cout << "  - Total postings: " << index.total_postings << std::endl;
    std::cout << "  - Build time: " << index.build_time_ms << " ms" << std::endl;
    std::cout << "  - Memory: " << index.memory_bytes / (1024 * 1024) << " MB" << std::endl;

    return index;
}

// ============================================================================
// Fast Search Using Pre-computed Scores
// ============================================================================

/**
 * Search the BM25S index with pre-computed scores.
 *
 * This is the fast path - no TF-IDF calculation at query time!
 *
 * @param index The pre-built BM25S index
 * @param query_terms Vector of query terms (already tokenized and lowercased)
 * @param topk Number of top results to return
 * @return Vector of BM25SHit results sorted by score
 *
 * Algorithm:
 *   1. For each query term, look up posting list in score_matrix
 *   2. Accumulate scores for each document
 *   3. Return top-k documents by score
 *
 * Time complexity: O(sum of posting list lengths for query terms + N log k)
 * Much faster than O(N * |query|) for traditional BM25
 */
std::vector<BM25SHit> search_bm25s(
    const BM25SIndex& index,
    const std::vector<std::string>& query_terms,
    int topk
) {
    if (!index.is_built() || query_terms.empty()) {
        return {};
    }

    // Accumulate scores for each document
    // Using a hash map for sparse accumulation
    std::unordered_map<uint32_t, float> doc_scores;
    doc_scores.reserve(topk * 100);  // Reasonable initial capacity

    // Process each query term
    for (const std::string& term : query_terms) {
        // Lowercase the term
        std::string lower_term = term;
        std::transform(lower_term.begin(), lower_term.end(), lower_term.begin(), ::tolower);

        // Look up posting list
        auto it = index.score_matrix.find(lower_term);
        if (it == index.score_matrix.end()) {
            continue;  // Term not in index
        }

        // Accumulate scores from posting list
        const auto& postings = it->second;
        for (const PostingEntry& entry : postings) {
            doc_scores[entry.doc_id] += entry.score;
        }
    }

    // Convert to vector for sorting
    std::vector<BM25SHit> results;
    results.reserve(doc_scores.size());

    for (const auto& [doc_id, score] : doc_scores) {
        results.emplace_back(doc_id, score);
    }

    // Partial sort for top-k
    if (results.size() > static_cast<size_t>(topk)) {
        std::partial_sort(results.begin(), results.begin() + topk, results.end(),
                         [](const BM25SHit& a, const BM25SHit& b) {
                             return a.score > b.score;
                         });
        results.resize(topk);
    } else {
        std::sort(results.begin(), results.end(),
                 [](const BM25SHit& a, const BM25SHit& b) {
                     return a.score > b.score;
                 });
    }

    return results;
}

/**
 * Parallel search using OpenMP for even faster query processing.
 * Useful when query has many terms.
 */
std::vector<BM25SHit> search_bm25s_parallel(
    const BM25SIndex& index,
    const std::vector<std::string>& query_terms,
    int topk
) {
    if (!index.is_built() || query_terms.empty()) {
        return {};
    }

    // Thread-local score accumulators
    std::vector<std::unordered_map<uint32_t, float>> thread_scores(omp_get_max_threads());

    // Process query terms in parallel
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        auto& local_scores = thread_scores[tid];

        #pragma omp for schedule(dynamic)
        for (size_t i = 0; i < query_terms.size(); i++) {
            std::string lower_term = query_terms[i];
            std::transform(lower_term.begin(), lower_term.end(), lower_term.begin(), ::tolower);

            auto it = index.score_matrix.find(lower_term);
            if (it == index.score_matrix.end()) continue;

            const auto& postings = it->second;
            for (const PostingEntry& entry : postings) {
                local_scores[entry.doc_id] += entry.score;
            }
        }
    }

    // Merge thread-local scores
    std::unordered_map<uint32_t, float> doc_scores;
    for (const auto& local : thread_scores) {
        for (const auto& [doc_id, score] : local) {
            doc_scores[doc_id] += score;
        }
    }

    // Convert to vector and sort
    std::vector<BM25SHit> results;
    results.reserve(doc_scores.size());

    for (const auto& [doc_id, score] : doc_scores) {
        results.emplace_back(doc_id, score);
    }

    if (results.size() > static_cast<size_t>(topk)) {
        std::partial_sort(results.begin(), results.begin() + topk, results.end(),
                         [](const BM25SHit& a, const BM25SHit& b) {
                             return a.score > b.score;
                         });
        results.resize(topk);
    } else {
        std::sort(results.begin(), results.end(),
                 [](const BM25SHit& a, const BM25SHit& b) {
                     return a.score > b.score;
                 });
    }

    return results;
}

// ============================================================================
// BlockWeakAnd Search with Early Termination (Speed Improvement #3)
// ============================================================================

/**
 * Search using BlockWeakAnd algorithm for early termination.
 *
 * This builds on BM25S by leveraging the block index to skip blocks
 * that cannot possibly contribute to the top-K results.
 *
 * @param index The pre-built BM25S index (must have block_index built)
 * @param query_terms Vector of query terms (already tokenized and lowercased)
 * @param topk Number of top results to return
 * @return Vector of (doc_id, score) pairs sorted by score
 *
 * Algorithm:
 *   1. Calculate upper bounds for each query term from block max scores
 *   2. Sort terms by upper bound (descending) - process high-value terms first
 *   3. For each term:
 *      - For each block:
 *        - If block.max_score < threshold / num_terms, skip block
 *        - Otherwise, process all documents in the block
 *   4. Maintain a min-heap of top-K scores to update threshold dynamically
 *
 * Performance:
 *   - 2-5x additional speedup on top of BM25S
 *   - Most effective for queries with common terms
 *   - Falls back to regular search if block index not built
 */
std::vector<std::pair<uint32_t, float>> search_bm25s_blockmax(
    const BM25SIndex& index,
    const std::vector<std::string>& query_terms,
    int topk
) {
    // Fall back to regular search if block index not built
    if (!index.block_index_built) {
        auto hits = search_bm25s(index, query_terms, topk);
        std::vector<std::pair<uint32_t, float>> results;
        results.reserve(hits.size());
        for (const auto& hit : hits) {
            results.emplace_back(hit.doc_id, hit.score);
        }
        return results;
    }

    if (!index.is_built() || query_terms.empty()) {
        return {};
    }

    // Structure to track term info for processing order
    struct TermInfo {
        std::string term;
        float upper_bound;      // Sum of all block max scores
        float max_block_score;  // Maximum single block score (for tighter pruning)
    };

    // Calculate upper bounds for each query term
    std::vector<TermInfo> term_infos;
    term_infos.reserve(query_terms.size());

    for (const std::string& term : query_terms) {
        std::string lower_term = term;
        std::transform(lower_term.begin(), lower_term.end(), lower_term.begin(), ::tolower);

        auto block_it = index.block_index.find(lower_term);
        if (block_it == index.block_index.end()) {
            continue;  // Term not in index
        }

        const auto& blocks = block_it->second;
        float upper_bound = 0.0f;
        float max_block_score = 0.0f;

        for (const BlockInfo& block : blocks) {
            upper_bound += block.max_score;
            if (block.max_score > max_block_score) {
                max_block_score = block.max_score;
            }
        }

        term_infos.push_back({lower_term, upper_bound, max_block_score});
    }

    if (term_infos.empty()) {
        return {};
    }

    // Sort terms by upper bound (descending) - process high-value terms first
    std::sort(term_infos.begin(), term_infos.end(),
              [](const TermInfo& a, const TermInfo& b) {
                  return a.upper_bound > b.upper_bound;
              });

    const size_t num_terms = term_infos.size();

    // Document score accumulator
    std::unordered_map<uint32_t, float> doc_scores;
    doc_scores.reserve(topk * 100);

    // Min-heap to track top-K scores (for dynamic threshold)
    // We use a vector and maintain it as a min-heap
    std::vector<float> top_scores;
    top_scores.reserve(topk + 1);

    float threshold = 0.0f;  // Current k-th best score

    // Statistics for debugging/tuning
    size_t blocks_processed = 0;
    size_t blocks_skipped = 0;

    // Process each term
    for (size_t term_idx = 0; term_idx < term_infos.size(); term_idx++) {
        const TermInfo& tinfo = term_infos[term_idx];

        auto score_it = index.score_matrix.find(tinfo.term);
        auto block_it = index.block_index.find(tinfo.term);

        if (score_it == index.score_matrix.end() || block_it == index.block_index.end()) {
            continue;
        }

        const auto& postings = score_it->second;
        const auto& blocks = block_it->second;

        // Calculate remaining upper bound from subsequent terms
        float remaining_upper_bound = 0.0f;
        for (size_t i = term_idx + 1; i < term_infos.size(); i++) {
            remaining_upper_bound += term_infos[i].upper_bound;
        }

        // Process each block
        for (const BlockInfo& block : blocks) {
            // Early termination check:
            // If this block's max score + remaining upper bound < threshold,
            // then even the best document in this block can't make top-K
            float potential_contribution = block.max_score + remaining_upper_bound;

            if (potential_contribution < threshold && threshold > 0.0f) {
                blocks_skipped++;
                continue;  // Skip this block
            }

            blocks_processed++;

            // Process all documents in this block
            for (uint32_t i = block.start_idx; i < block.end_idx; i++) {
                const PostingEntry& entry = postings[i];
                float& doc_score = doc_scores[entry.doc_id];
                doc_score += entry.score;

                // Update threshold if we have enough documents
                if (doc_scores.size() >= static_cast<size_t>(topk)) {
                    // Add to min-heap
                    top_scores.push_back(doc_score);
                    std::push_heap(top_scores.begin(), top_scores.end(), std::greater<float>());

                    // Keep only top-K
                    while (top_scores.size() > static_cast<size_t>(topk)) {
                        std::pop_heap(top_scores.begin(), top_scores.end(), std::greater<float>());
                        top_scores.pop_back();
                    }

                    // Update threshold (minimum of top-K)
                    if (!top_scores.empty()) {
                        threshold = top_scores.front();
                    }
                }
            }
        }
    }

    // Final threshold update from all accumulated scores
    for (const auto& [doc_id, score] : doc_scores) {
        top_scores.push_back(score);
        std::push_heap(top_scores.begin(), top_scores.end(), std::greater<float>());
        while (top_scores.size() > static_cast<size_t>(topk)) {
            std::pop_heap(top_scores.begin(), top_scores.end(), std::greater<float>());
            top_scores.pop_back();
        }
    }

    // Get final threshold
    if (!top_scores.empty()) {
        threshold = top_scores.front();
    }

    // Convert to result vector, filtering by threshold
    std::vector<std::pair<uint32_t, float>> results;
    results.reserve(doc_scores.size());

    for (const auto& [doc_id, score] : doc_scores) {
        if (score >= threshold || results.size() < static_cast<size_t>(topk)) {
            results.emplace_back(doc_id, score);
        }
    }

    // Partial sort for top-k
    if (results.size() > static_cast<size_t>(topk)) {
        std::partial_sort(results.begin(), results.begin() + topk, results.end(),
                         [](const std::pair<uint32_t, float>& a, const std::pair<uint32_t, float>& b) {
                             return a.second > b.second;
                         });
        results.resize(topk);
    } else {
        std::sort(results.begin(), results.end(),
                 [](const std::pair<uint32_t, float>& a, const std::pair<uint32_t, float>& b) {
                     return a.second > b.second;
                 });
    }

    // Debug output (can be disabled in production)
    #ifdef BM25S_DEBUG
    std::cout << "[BM25S BlockMax] Blocks processed: " << blocks_processed
              << ", skipped: " << blocks_skipped
              << " (" << (100.0 * blocks_skipped / (blocks_processed + blocks_skipped)) << "% pruned)"
              << std::endl;
    #endif

    return results;
}

/**
 * Convenience wrapper that returns BM25SHit instead of pair.
 * Use this for consistency with other search functions.
 */
std::vector<BM25SHit> search_bm25s_blockmax_hits(
    const BM25SIndex& index,
    const std::vector<std::string>& query_terms,
    int topk
) {
    auto results = search_bm25s_blockmax(index, query_terms, topk);
    std::vector<BM25SHit> hits;
    hits.reserve(results.size());

    for (const auto& [doc_id, score] : results) {
        hits.emplace_back(doc_id, score);
    }

    return hits;
}

// ============================================================================
// Persistence: Save/Load Index to/from Disk
// ============================================================================

// Magic number and version for binary format
constexpr uint32_t BM25S_MAGIC = 0x424D3235;  // "BM25" in hex
constexpr uint32_t BM25S_VERSION = 1;

/**
 * Save BM25S index to binary file.
 *
 * Format:
 *   [Header]
 *     - Magic number (4 bytes)
 *     - Version (4 bytes)
 *     - Num documents (8 bytes)
 *     - Num terms (8 bytes)
 *     - Total postings (8 bytes)
 *     - Average document length (8 bytes, double)
 *     - Config: k1 (4 bytes), b (4 bytes)
 *   [Terms]
 *     For each term:
 *       - Term length (4 bytes)
 *       - Term string (variable)
 *       - Posting list size (4 bytes)
 *       - Postings: [doc_id (4 bytes), score (4 bytes)] * size
 */
bool save_bm25s_index(const BM25SIndex& index, const std::string& path) {
    std::ofstream file(path, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "[BM25S] Failed to open file for writing: " << path << std::endl;
        return false;
    }

    std::cout << "[BM25S] Saving index to " << path << "..." << std::flush;
    auto start = std::chrono::high_resolution_clock::now();

    // Write header
    file.write(reinterpret_cast<const char*>(&BM25S_MAGIC), 4);
    file.write(reinterpret_cast<const char*>(&BM25S_VERSION), 4);

    uint64_t num_docs = index.num_documents;
    uint64_t num_terms = index.num_terms;
    uint64_t total_post = index.total_postings;
    double avgdl = index.avgdl;
    float k1 = index.config.k1;
    float b = index.config.b;

    file.write(reinterpret_cast<const char*>(&num_docs), 8);
    file.write(reinterpret_cast<const char*>(&num_terms), 8);
    file.write(reinterpret_cast<const char*>(&total_post), 8);
    file.write(reinterpret_cast<const char*>(&avgdl), 8);
    file.write(reinterpret_cast<const char*>(&k1), 4);
    file.write(reinterpret_cast<const char*>(&b), 4);

    // Write terms and posting lists
    for (const auto& [term, postings] : index.score_matrix) {
        // Term
        uint32_t term_len = static_cast<uint32_t>(term.length());
        file.write(reinterpret_cast<const char*>(&term_len), 4);
        file.write(term.data(), term_len);

        // Posting list
        uint32_t post_size = static_cast<uint32_t>(postings.size());
        file.write(reinterpret_cast<const char*>(&post_size), 4);

        // Write postings as contiguous array
        for (const auto& entry : postings) {
            file.write(reinterpret_cast<const char*>(&entry.doc_id), 4);
            file.write(reinterpret_cast<const char*>(&entry.score), 4);
        }
    }

    file.close();

    auto end = std::chrono::high_resolution_clock::now();
    double elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();

    // Get file size
    std::ifstream check(path, std::ios::binary | std::ios::ate);
    size_t file_size = check.tellg();
    check.close();

    std::cout << " Done! (" << elapsed_ms << " ms, "
              << file_size / (1024 * 1024) << " MB)" << std::endl;

    return true;
}

/**
 * Load BM25S index from binary file.
 */
bool load_bm25s_index(BM25SIndex& index, const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "[BM25S] Failed to open file for reading: " << path << std::endl;
        return false;
    }

    std::cout << "[BM25S] Loading index from " << path << "..." << std::flush;
    auto start = std::chrono::high_resolution_clock::now();

    index.clear();

    // Read and verify header
    uint32_t magic, version;
    file.read(reinterpret_cast<char*>(&magic), 4);
    file.read(reinterpret_cast<char*>(&version), 4);

    if (magic != BM25S_MAGIC) {
        std::cerr << " Error: Invalid magic number" << std::endl;
        return false;
    }

    if (version != BM25S_VERSION) {
        std::cerr << " Error: Unsupported version " << version << std::endl;
        return false;
    }

    uint64_t num_docs, num_terms, total_post;
    double avgdl;
    float k1, b;

    file.read(reinterpret_cast<char*>(&num_docs), 8);
    file.read(reinterpret_cast<char*>(&num_terms), 8);
    file.read(reinterpret_cast<char*>(&total_post), 8);
    file.read(reinterpret_cast<char*>(&avgdl), 8);
    file.read(reinterpret_cast<char*>(&k1), 4);
    file.read(reinterpret_cast<char*>(&b), 4);

    index.num_documents = num_docs;
    index.num_terms = num_terms;
    index.total_postings = total_post;
    index.avgdl = avgdl;
    index.config.k1 = k1;
    index.config.b = b;

    // Pre-allocate hash map
    index.score_matrix.reserve(num_terms);

    // Read terms and posting lists
    std::string term_buffer;
    for (size_t i = 0; i < num_terms; i++) {
        // Read term
        uint32_t term_len;
        file.read(reinterpret_cast<char*>(&term_len), 4);

        term_buffer.resize(term_len);
        file.read(&term_buffer[0], term_len);

        // Read posting list
        uint32_t post_size;
        file.read(reinterpret_cast<char*>(&post_size), 4);

        std::vector<PostingEntry>& postings = index.score_matrix[term_buffer];
        postings.resize(post_size);

        // Read postings
        for (uint32_t j = 0; j < post_size; j++) {
            file.read(reinterpret_cast<char*>(&postings[j].doc_id), 4);
            file.read(reinterpret_cast<char*>(&postings[j].score), 4);
        }
    }

    file.close();

    index.memory_bytes = index.estimate_memory();

    auto end = std::chrono::high_resolution_clock::now();
    double elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();

    std::cout << " Done! (" << elapsed_ms << " ms)" << std::endl;
    std::cout << "[BM25S] Loaded " << index.num_documents << " documents, "
              << index.num_terms << " terms, "
              << index.total_postings << " postings" << std::endl;
    std::cout << "[BM25S] Memory: " << index.memory_bytes / (1024 * 1024) << " MB" << std::endl;

    return true;
}

// ============================================================================
// Utility: Check if index file exists and is current
// ============================================================================

/**
 * Get the default BM25S index path for a given manifest file.
 */
inline std::string get_bm25s_index_path(const std::string& manifest_path) {
    size_t dot_pos = manifest_path.rfind('.');
    if (dot_pos != std::string::npos) {
        return manifest_path.substr(0, dot_pos) + ".bm25s";
    }
    return manifest_path + ".bm25s";
}

/**
 * Check if BM25S index file exists and is newer than manifest.
 */
inline bool bm25s_index_is_current(const std::string& index_path, const std::string& manifest_path) {
    struct stat index_stat, manifest_stat;

    if (stat(index_path.c_str(), &index_stat) != 0) {
        return false;  // Index doesn't exist
    }

    if (stat(manifest_path.c_str(), &manifest_stat) != 0) {
        return false;  // Manifest doesn't exist
    }

    // Index is current if it's newer than manifest
    return index_stat.st_mtime >= manifest_stat.st_mtime;
}

// ============================================================================
// Integration Helper: Convert between BM25S hits and existing Hit structure
// ============================================================================

/**
 * Wrapper to make integration with existing code easier.
 * Converts BM25SHit to the existing Hit structure used in ocean_chat_server.cpp
 *
 * Usage in ocean_chat_server.cpp:
 *
 *   #include "bm25s_engine.hpp"
 *
 *   // Global BM25S index
 *   bm25s::BM25SIndex global_bm25s_index;
 *
 *   // In load_manifest() or after:
 *   global_bm25s_index = bm25s::build_bm25s_index(
 *       global_corpus.docs,
 *       [](const DocMeta& doc) -> const vector<string>& { return doc.keywords; }
 *   );
 *
 *   // Replace search_bm25() with:
 *   vector<Hit> search_bm25_fast(const Corpus& corpus, const string& query, int topk) {
 *       vector<string> query_terms = extract_text_keywords(query);
 *       auto bm25s_hits = bm25s::search_bm25s(global_bm25s_index, query_terms, topk);
 *       return bm25s::convert_hits(bm25s_hits);
 *   }
 */

template<typename HitType>
std::vector<HitType> convert_hits(const std::vector<BM25SHit>& bm25s_hits) {
    std::vector<HitType> hits;
    hits.reserve(bm25s_hits.size());

    for (const auto& bm25s_hit : bm25s_hits) {
        HitType hit;
        hit.doc_idx = bm25s_hit.doc_id;
        hit.score = static_cast<double>(bm25s_hit.score);
        hits.push_back(hit);
    }

    return hits;
}

// ============================================================================
// Optional: Compressed Score Storage (for memory-constrained environments)
// ============================================================================

/**
 * Quantize float scores to 16-bit integers for memory savings.
 * Reduces posting entry size from 8 bytes to 6 bytes (25% savings).
 *
 * Score range is normalized to [0, max_score] -> [0, 65535]
 */
struct CompressedPostingEntry {
    uint32_t doc_id;
    uint16_t quantized_score;

    CompressedPostingEntry() : doc_id(0), quantized_score(0) {}
    CompressedPostingEntry(uint32_t id, uint16_t s) : doc_id(id), quantized_score(s) {}
};

/**
 * Compress a posting list by quantizing scores.
 * Returns the max score used for dequantization.
 */
inline float compress_posting_list(
    const std::vector<PostingEntry>& input,
    std::vector<CompressedPostingEntry>& output
) {
    if (input.empty()) {
        output.clear();
        return 0.0f;
    }

    // Find max score
    float max_score = 0.0f;
    for (const auto& entry : input) {
        if (entry.score > max_score) max_score = entry.score;
    }

    // Quantize
    output.resize(input.size());
    float scale = (max_score > 0) ? (65535.0f / max_score) : 0.0f;

    for (size_t i = 0; i < input.size(); i++) {
        output[i].doc_id = input[i].doc_id;
        output[i].quantized_score = static_cast<uint16_t>(input[i].score * scale);
    }

    return max_score;
}

/**
 * Decompress a quantized score.
 */
inline float decompress_score(uint16_t quantized, float max_score) {
    return (quantized / 65535.0f) * max_score;
}

// ============================================================================
// Incremental Index Update (for adding new documents without full rebuild)
// ============================================================================

/**
 * Add a single document to an existing BM25S index.
 *
 * Note: This is less efficient than batch building, but useful for
 * real-time updates like adding conversation chunks.
 *
 * The avgdl is not updated - call update_avgdl() periodically if needed.
 */
template<typename GetKeywordsFn>
void add_document_to_index(
    BM25SIndex& index,
    uint32_t doc_id,
    const std::vector<std::string>& keywords,
    GetKeywordsFn /* unused - for API consistency */
) {
    if (keywords.empty()) return;

    const size_t N = index.num_documents + 1;
    const float k1 = index.config.k1;
    const float b = index.config.b;
    const float avgdl_f = static_cast<float>(index.avgdl);

    int doc_len = static_cast<int>(keywords.size());
    float norm_factor = k1 * (1.0f - b + b * doc_len / avgdl_f);

    // Count term frequencies
    std::unordered_map<std::string, int> term_freq;
    for (const auto& kw : keywords) {
        std::string term = kw;
        std::transform(term.begin(), term.end(), term.begin(), ::tolower);
        term_freq[term]++;
    }

    // Add to score matrix
    for (const auto& [term, tf] : term_freq) {
        // Estimate DF (use existing posting list size + 1)
        auto it = index.score_matrix.find(term);
        size_t df = (it != index.score_matrix.end()) ? it->second.size() + 1 : 1;

        // Calculate IDF
        float idf = std::log((N - df + 0.5f) / (df + 0.5f) + 1.0f);

        // Calculate BM25 score
        float score = idf * (tf * (k1 + 1.0f)) / (tf + norm_factor);

        if (score > 0.0f) {
            // Add to posting list (maintain sorted order)
            auto& postings = index.score_matrix[term];

            // Find insertion point (keep sorted by descending score)
            auto insert_pos = std::lower_bound(
                postings.begin(), postings.end(),
                PostingEntry(doc_id, score),
                [](const PostingEntry& a, const PostingEntry& b) {
                    return a.score > b.score;
                }
            );

            postings.insert(insert_pos, PostingEntry(doc_id, score));
            index.total_postings++;
        }
    }

    index.num_documents++;
}

/**
 * Update average document length after adding multiple documents.
 */
template<typename DocType, typename GetKeywordsFn>
void update_avgdl(
    BM25SIndex& index,
    const std::vector<DocType>& documents,
    GetKeywordsFn get_keywords
) {
    size_t total_length = 0;
    for (const auto& doc : documents) {
        total_length += get_keywords(doc).size();
    }
    index.avgdl = static_cast<double>(total_length) / documents.size();
    index.num_documents = documents.size();
}

} // namespace bm25s

#endif // BM25S_ENGINE_HPP
