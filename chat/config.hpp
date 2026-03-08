#pragma once
// OceanEterna v4 Configuration System
// Config struct, JSON loader, and environment variable overrides

#include <string>
#include <fstream>
#include <iostream>

// Forward declaration - json.hpp must be included before this header
using json = nlohmann::json;
using namespace std;

// Helper: read environment variable with fallback
inline string get_env_or(const char* name, const string& fallback) {
    const char* val = getenv(name);
    return val ? string(val) : fallback;
}

// Configuration struct -- loaded from config.json with env var overrides
struct Config {
    struct {
        int port = 8888;
        string host = "0.0.0.0";
    } server;

    struct {
        bool use_external = true;
        string external_url = "https://routellm.abacus.ai/v1/chat/completions";
        string external_model = "gpt-5-mini";
        string local_url = "http://127.0.0.1:1234/v1/chat/completions";
        string local_model = "qwen/qwen3-32b";
        string api_key;  // from env OCEAN_API_KEY
        int timeout_sec = 30;
        int max_retries = 3;
        int retry_backoff_ms = 1000;
    } llm;

    struct {
        int top_k = 8;
        double k1 = 1.5;
        double b = 0.75;
        int context_window = 1;      // v4.2: adjacent chunks to fetch (0-3)
        int max_context_chars = 32000; // v4.2: max LLM context size (~8000 tokens)
        double score_threshold = 0.2;  // v4.2: drop results below this fraction of top score
    } search;

    struct {
        string manifest = "corpus/manifest.jsonl";
        string storage = "corpus/storage.bin";
        string chapter_guide = "corpus/chapter_guide.json";
    } corpus;

    struct {
        bool enabled = false;
        string url = "http://127.0.0.1:8889/rerank";
        int timeout_ms = 500;
        int candidate_count = 50;  // how many BM25 results to send to reranker
    } reranker;

    struct {
        bool enabled = false;
        string api_key = "";
    } auth;

    struct {
        bool enabled = false;
        int requests_per_minute = 60;
    } rate_limit;
};

inline Config load_config(const string& path) {
    Config cfg;
    ifstream f(path);
    if (f.is_open()) {
        try {
            json j = json::parse(f);
            if (j.contains("server")) {
                cfg.server.port = j["server"].value("port", cfg.server.port);
                cfg.server.host = j["server"].value("host", cfg.server.host);
            }
            if (j.contains("llm")) {
                cfg.llm.use_external = j["llm"].value("use_external", cfg.llm.use_external);
                cfg.llm.external_url = j["llm"].value("external_url", cfg.llm.external_url);
                cfg.llm.external_model = j["llm"].value("external_model", cfg.llm.external_model);
                cfg.llm.local_url = j["llm"].value("local_url", cfg.llm.local_url);
                cfg.llm.local_model = j["llm"].value("local_model", cfg.llm.local_model);
                cfg.llm.timeout_sec = j["llm"].value("timeout_sec", cfg.llm.timeout_sec);
                cfg.llm.max_retries = j["llm"].value("max_retries", cfg.llm.max_retries);
                cfg.llm.retry_backoff_ms = j["llm"].value("retry_backoff_ms", cfg.llm.retry_backoff_ms);
            }
            if (j.contains("search")) {
                cfg.search.top_k = j["search"].value("top_k", cfg.search.top_k);
                cfg.search.k1 = j["search"].value("bm25_k1", cfg.search.k1);
                cfg.search.b = j["search"].value("bm25_b", cfg.search.b);
                cfg.search.context_window = j["search"].value("context_window", cfg.search.context_window);
                cfg.search.max_context_chars = j["search"].value("max_context_chars", cfg.search.max_context_chars);
                cfg.search.score_threshold = j["search"].value("score_threshold", cfg.search.score_threshold);
            }
            if (j.contains("corpus")) {
                cfg.corpus.manifest = j["corpus"].value("manifest", cfg.corpus.manifest);
                cfg.corpus.storage = j["corpus"].value("storage", cfg.corpus.storage);
                cfg.corpus.chapter_guide = j["corpus"].value("chapter_guide", cfg.corpus.chapter_guide);
            }
            if (j.contains("reranker")) {
                cfg.reranker.enabled = j["reranker"].value("enabled", cfg.reranker.enabled);
                cfg.reranker.url = j["reranker"].value("url", cfg.reranker.url);
                cfg.reranker.timeout_ms = j["reranker"].value("timeout_ms", cfg.reranker.timeout_ms);
                cfg.reranker.candidate_count = j["reranker"].value("candidate_count", cfg.reranker.candidate_count);
            }
            if (j.contains("auth")) {
                cfg.auth.enabled = j["auth"].value("enabled", cfg.auth.enabled);
                cfg.auth.api_key = j["auth"].value("api_key", cfg.auth.api_key);
            }
            if (j.contains("rate_limit")) {
                cfg.rate_limit.enabled = j["rate_limit"].value("enabled", cfg.rate_limit.enabled);
                cfg.rate_limit.requests_per_minute = j["rate_limit"].value("requests_per_minute", cfg.rate_limit.requests_per_minute);
            }
            cout << "Loaded config from " << path << endl;
        } catch (const exception& e) {
            cerr << "Warning: Failed to parse " << path << ": " << e.what() << endl;
            cerr << "Using default configuration." << endl;
        }
    } else {
        cout << "No config.json found, using defaults." << endl;
    }

    // Environment variable overrides (highest priority)
    cfg.llm.api_key = get_env_or("OCEAN_API_KEY", "");
    string env_url = get_env_or("OCEAN_API_URL", "");
    if (!env_url.empty()) cfg.llm.external_url = env_url;
    string env_model = get_env_or("OCEAN_MODEL", "");
    if (!env_model.empty()) cfg.llm.external_model = env_model;
    string env_server_key = get_env_or("OCEAN_SERVER_API_KEY", "");
    if (!env_server_key.empty()) cfg.auth.api_key = env_server_key;

    return cfg;
}
