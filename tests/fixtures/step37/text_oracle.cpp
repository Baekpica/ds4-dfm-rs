// Standalone GGUF oracle against the pinned stepfun-ai/llama.cpp step3.7.
// Identical token input and full-vocabulary dumps as test_step37_forward.
#include "llama.h"
#include "ggml-backend.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <vector>
#include <cstring>
#include <string>

static bool trace(struct ggml_tensor *t, bool ask, void *data) {
    if (!*static_cast<bool *>(data)) { return false; }
    const char *dir = std::getenv("STEP37_TRACE");
    if (!dir || !*dir) { return false; }
    const char *prefixes[] = {"attn_norm_in-", "attn_norm-", "Qcur_pos-", "Kcur_pos-",
        "Vcur-", "attn_out-", "attn_gated-", "attn_proj-", "ffn_inp-", "ffn_norm-",
        "ffn_out-", "ffn_moe_out-", "ffn_shared_out-", "ffn_moe_topk-", "ffn_moe_weights_scaled-"};
    bool selected = false;
    for (auto p : prefixes) { selected |= std::strncmp(t->name, p, std::strlen(p)) == 0; }
    if (ask || !selected) { return selected; }
    const auto path = std::string(dir) + "/" + t->name + ".bin";
    std::vector<char> bytes(ggml_nbytes(t));
    ggml_backend_tensor_get(t, bytes.data(), 0, bytes.size());
    std::ofstream out(path, std::ios::binary);
    out.write(bytes.data(), bytes.size());
    if (!out) { std::abort(); }
    std::fprintf(stderr, "trace %s %s [%lld,%lld,%lld,%lld] contiguous=%d\n", t->name,
        ggml_type_name(t->type), (long long)t->ne[0], (long long)t->ne[1],
        (long long)t->ne[2], (long long)t->ne[3], ggml_is_contiguous(t));
    return true;
}

static int run_case(llama_model *model, const char *input_path, const char *output_path,
                    unsigned cap, unsigned decode) {
    std::fprintf(stderr, "Oracle case %s -> %s\n", input_path, output_path);
    std::ifstream input(input_path);
    std::vector<llama_token> tokens;
    int token;
    while (input >> token) { tokens.push_back(token); }
    if (tokens.empty()) { return 2; }
    const int vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    for (int t : tokens) { if (t < 0 || t >= vocab) { return 2; } }
    auto cp = llama_context_default_params();
    bool tracing = true;
    cp.n_ctx = tokens.size() + decode + 1;
    cp.n_batch = cap;
    cp.n_ubatch = cap;
    cp.n_threads = 16;
    cp.n_threads_batch = 16;
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    if (std::getenv("STEP37_NO_FA")) { cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED; }
    if (std::getenv("STEP37_TRACE")) { cp.cb_eval = trace; cp.cb_eval_user_data = &tracing; }
    auto *ctx = llama_init_from_model(model, cp);
    if (!ctx) { return 1; }
    const auto start = std::chrono::steady_clock::now();
    for (unsigned pos = 0; pos < tokens.size();) {
        const unsigned n = std::min(cap, unsigned(tokens.size() - pos));
        if (llama_decode(ctx, llama_batch_get_one(tokens.data() + pos, n))) { return 1; }
        pos += n;
    }
    std::ofstream output(output_path, std::ios::binary);
    if (!output) { return 2; }
    const auto prefill_end = std::chrono::steady_clock::now();
    tracing = false;
    std::fprintf(stderr, "Oracle prefill %zu tokens %.6f seconds\n", tokens.size(),
        std::chrono::duration<double>(prefill_end - start).count());
    for (unsigned i = 0; i <= decode; i++) {
        const float *logits = llama_get_logits_ith(ctx, -1);
        if (!logits) { return 1; }
        for (int j = 0; j < vocab; j++) { if (!std::isfinite(logits[j])) { return 1; } }
        llama_token best = std::max_element(logits, logits + vocab) - logits;
        std::printf("%u %d %.9g\n", i, best, logits[best]);
        output.write(reinterpret_cast<const char *>(logits), vocab * sizeof(float));
        if (!output) { return 1; }
        if (i == decode) { break; }
        if (llama_decode(ctx, llama_batch_get_one(&best, 1))) { return 1; }
    }
    std::fprintf(stderr, "Oracle decode %u evaluations %.6f seconds\n", decode,
        std::chrono::duration<double>(std::chrono::steady_clock::now() - prefill_end).count());
    llama_free(ctx);
    return 0;
}

int main(int argc, char **argv) {
    if (argc != 6) { return 2; }
    const unsigned cap = std::strtoul(argv[4], nullptr, 10);
    const unsigned decode = std::strtoul(argv[5], nullptr, 10);
    if (!cap || cap > 4096 || decode > 1024) { return 2; }
    llama_backend_init();
    auto mp = llama_model_default_params();
    mp.n_gpu_layers = 99;
    auto *model = llama_model_load_from_file(argv[1], mp);
    if (!model) { return 1; }
    int rc = 0;
    if (argv[2][0] != '@') {
        rc = run_case(model, argv[2], argv[3], cap, decode);
    } else {
        std::ifstream manifest(argv[2] + 1);
        std::string input, output;
        unsigned count = 0;
        while (manifest >> input) {
            if (!(manifest >> output)) { rc = 2; break; }
            count++;
            rc = run_case(model, input.c_str(), output.c_str(), cap, decode);
            if (rc) { break; }
        }
        if (!count) { rc = 2; }
    }
    llama_model_free(model);
    llama_backend_free();
    return rc;
}
