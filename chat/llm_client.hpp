#pragma once
// OceanEterna v4 LLM Client
// CURL-based API calls with retry + exponential backoff
// Requires: json.hpp, curl/curl.h, config.hpp (g_config), DEBUG macros

#include <string>
#include <chrono>
#include <thread>
#include <functional>
#include <sstream>

// CURL callback for capturing response body
inline size_t curl_write_cb(void* contents, size_t size, size_t nmemb, std::string* s) {
    size_t new_length = size * nmemb;
    s->append((char*)contents, new_length);
    return new_length;
}

// Step 24: Streaming context for SSE parsing
struct StreamingContext {
    std::function<void(const std::string&)> token_callback;
    std::string buffer;           // Partial line buffer
    std::string full_response;    // Accumulated complete response
    bool error_occurred = false;
    std::string error_message;
};

// Step 24: CURL callback for streaming SSE responses
inline size_t curl_streaming_cb(void* contents, size_t size, size_t nmemb, StreamingContext* ctx) {
    size_t new_length = size * nmemb;
    ctx->buffer.append((char*)contents, new_length);

    // Process complete lines from buffer
    size_t pos;
    while ((pos = ctx->buffer.find('\n')) != std::string::npos) {
        std::string line = ctx->buffer.substr(0, pos);
        ctx->buffer.erase(0, pos + 1);

        // Remove carriage return if present
        if (!line.empty() && line.back() == '\r') {
            line.pop_back();
        }

        // Skip empty lines
        if (line.empty()) continue;

        // SSE format: "data: {...}" or "data: [DONE]"
        if (line.substr(0, 6) == "data: ") {
            std::string data = line.substr(6);

            // Check for stream end
            if (data == "[DONE]") {
                continue;
            }

            // Parse JSON chunk
            try {
                json chunk = json::parse(data);
                if (chunk.contains("choices") && !chunk["choices"].empty()) {
                    auto& choice = chunk["choices"][0];
                    if (choice.contains("delta") && choice["delta"].contains("content")) {
                        auto& content = choice["delta"]["content"];
                        if (!content.is_null()) {
                            std::string token = content.get<std::string>();
                            ctx->full_response += token;
                            if (ctx->token_callback) {
                                ctx->token_callback(token);
                            }
                        }
                    }
                }
                // Check for error in chunk
                if (chunk.contains("error")) {
                    ctx->error_occurred = true;
                    ctx->error_message = chunk["error"].dump();
                }
            } catch (const std::exception& e) {
                // Ignore parse errors for partial chunks
                DEBUG_LOG("SSE parse warning: " + std::string(e.what()));
            }
        }
    }

    return new_length;
}

// Step 24: Streaming LLM query with token callback
// Returns {full_response, elapsed_ms} - calls token_callback for each token as it arrives
inline std::pair<std::string, double> query_llm_streaming(
    const std::string& prompt,
    std::function<void(const std::string&)> token_callback) {

    CURL* curl = curl_easy_init();
    if (!curl) return {"ERROR", 0};

    auto start = std::chrono::high_resolution_clock::now();

    json request;
    try {
        if (USE_EXTERNAL_API) {
            request["model"] = EXTERNAL_MODEL;
        } else {
            request["model"] = LOCAL_MODEL;
        }
        request["messages"] = {{{"role", "user"}, {"content", prompt}}};
        request["temperature"] = 0.3;
        request["max_tokens"] = 500;
        request["stream"] = true;  // Enable streaming
        request["stop"] = json::array({"\n\n", "Question:", "Context:"});
    } catch (...) {
        curl_easy_cleanup(curl);
        return {"ERROR", 0};
    }

    const std::string& api_url = USE_EXTERNAL_API ? EXTERNAL_API_URL : LOCAL_LLM_URL;
    curl_easy_setopt(curl, CURLOPT_URL, api_url.c_str());

    struct curl_slist *headers = NULL;
    headers = curl_slist_append(headers, "Content-Type: application/json");
    headers = curl_slist_append(headers, "Accept: text/event-stream");

    if (USE_EXTERNAL_API) {
        std::string auth_header = "Authorization: Bearer " + EXTERNAL_API_KEY;
        headers = curl_slist_append(headers, auth_header.c_str());
    }

    curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);

    std::string request_body = request.dump();
    curl_easy_setopt(curl, CURLOPT_POSTFIELDS, request_body.c_str());

    // Set up streaming context
    StreamingContext ctx;
    ctx.token_callback = token_callback;

    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, curl_streaming_cb);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &ctx);
    curl_easy_setopt(curl, CURLOPT_TIMEOUT, LLM_TIMEOUT_SEC);

    CURLcode res = curl_easy_perform(curl);

    auto end = std::chrono::high_resolution_clock::now();
    double elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();

    curl_slist_free_all(headers);
    curl_easy_cleanup(curl);

    if (res != CURLE_OK) {
        return {"ERROR: " + std::string(curl_easy_strerror(res)), elapsed_ms};
    }

    if (ctx.error_occurred) {
        return {"ERROR: " + ctx.error_message, elapsed_ms};
    }

    // Trim whitespace from response
    std::string answer = ctx.full_response;
    answer.erase(0, answer.find_first_not_of(" \n\r\t"));
    if (!answer.empty()) {
        answer.erase(answer.find_last_not_of(" \n\r\t") + 1);
    }

    if (answer.empty()) {
        return {"ERROR: LLM returned empty response", elapsed_ms};
    }

    return {answer, elapsed_ms};
}

// Single LLM call (no retry)
inline std::pair<std::string, double> query_llm_once(const std::string& prompt) {
    CURL* curl = curl_easy_init();
    if (!curl) return {"ERROR", 0};

    auto start = std::chrono::high_resolution_clock::now();

    json request;
    try {
        if (USE_EXTERNAL_API) {
            request["model"] = EXTERNAL_MODEL;
        } else {
            request["model"] = LOCAL_MODEL;
        }
        request["messages"] = {{{"role", "user"}, {"content", prompt}}};
        request["temperature"] = 0.3;
        request["max_tokens"] = 500;
        request["stream"] = false;
        request["stop"] = json::array({"\n\n", "Question:", "Context:"});
    } catch (...) {
        return {"ERROR", 0};
    }

    const std::string& api_url = USE_EXTERNAL_API ? EXTERNAL_API_URL : LOCAL_LLM_URL;
    curl_easy_setopt(curl, CURLOPT_URL, api_url.c_str());

    struct curl_slist *headers = NULL;
    headers = curl_slist_append(headers, "Content-Type: application/json");

    if (USE_EXTERNAL_API) {
        std::string auth_header = "Authorization: Bearer " + EXTERNAL_API_KEY;
        headers = curl_slist_append(headers, auth_header.c_str());
    }

    curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);

    std::string request_body = request.dump();
    curl_easy_setopt(curl, CURLOPT_POSTFIELDS, request_body.c_str());

    std::string response_string;
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, curl_write_cb);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response_string);
    curl_easy_setopt(curl, CURLOPT_TIMEOUT, LLM_TIMEOUT_SEC);

    CURLcode res = curl_easy_perform(curl);

    auto end = std::chrono::high_resolution_clock::now();
    double elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();

    curl_slist_free_all(headers);
    curl_easy_cleanup(curl);

    if (res != CURLE_OK) {
        return {"ERROR: " + std::string(curl_easy_strerror(res)), elapsed_ms};
    }

    DEBUG_LOG("===== RAW RESPONSE =====");
    DEBUG_LOG(response_string.substr(0, 500));
    DEBUG_LOG("========================");

    try {
        json response = json::parse(response_string);

#ifdef DEBUG_MODE
        std::cout << "LLM Response parsed: " << response.dump().substr(0, 500) << std::endl;
        std::cout << "Has choices: " << response.contains("choices") << std::endl;
        if (response.contains("choices")) {
            std::cout << "Choices is array: " << response["choices"].is_array() << std::endl;
            std::cout << "Choices empty: " << response["choices"].empty() << std::endl;
        }
#endif

        if (!response.contains("choices") || response["choices"].empty()) {
            return {"ERROR: No choices in LLM response. Response: " + response.dump().substr(0, 200), elapsed_ms};
        }

        auto& choice = response["choices"][0];
        if (!choice.contains("message") || !choice["message"].contains("content")) {
            return {"ERROR: No content in LLM response", elapsed_ms};
        }

        if (choice["message"]["content"].is_null()) {
            return {"ERROR: LLM returned null content", elapsed_ms};
        }

        std::string answer = choice["message"]["content"].get<std::string>();

        answer.erase(0, answer.find_first_not_of(" \n\r\t"));
        answer.erase(answer.find_last_not_of(" \n\r\t") + 1);

        if (answer.empty()) {
            return {"ERROR: LLM returned empty response", elapsed_ms};
        }

        return {answer, elapsed_ms};
    } catch (const std::exception& e) {
        return {std::string("ERROR: Failed to parse LLM response: ") + e.what() + "\nResponse: " + response_string.substr(0, 200), elapsed_ms};
    }
}

// Query LLM with retry + exponential backoff
inline std::pair<std::string, double> query_llm(const std::string& prompt) {
    int max_retries = g_config.llm.max_retries;
    int backoff_ms = g_config.llm.retry_backoff_ms;
    double total_elapsed = 0;

    for (int attempt = 0; attempt <= max_retries; attempt++) {
        auto [answer, elapsed_ms] = query_llm_once(prompt);
        total_elapsed += elapsed_ms;

        if (answer.substr(0, 5) != "ERROR") {
            return {answer, total_elapsed};
        }

        if (attempt == max_retries) {
            return {answer, total_elapsed};
        }

        bool retryable = (answer.find("timeout") != std::string::npos ||
                         answer.find("TIMEOUT") != std::string::npos ||
                         answer.find("rate limit") != std::string::npos ||
                         answer.find("Rate limit") != std::string::npos ||
                         answer.find("502") != std::string::npos ||
                         answer.find("503") != std::string::npos ||
                         answer.find("CURLE_OPERATION_TIMEDOUT") != std::string::npos ||
                         answer.find("CURLE_COULDNT_CONNECT") != std::string::npos);

        if (!retryable) {
            return {answer, total_elapsed};
        }

        int delay = backoff_ms * (1 << attempt);
        std::cerr << "LLM call failed (attempt " << (attempt + 1) << "/" << (max_retries + 1)
             << "), retrying in " << delay << "ms: " << answer.substr(0, 80) << std::endl;
        std::this_thread::sleep_for(std::chrono::milliseconds(delay));
    }

    return {"ERROR: Max retries exceeded", total_elapsed};
}
