/* Qwen3.8 two-row MTP verify parity.
 *
 *   ./tests/test_qwen4exp_verify <first-model-shard.gguf> [steps]
 *
 * Session A commits tokens one row at a time (the serial oracle).  Session
 * B verifies the same tokens with qwen4exp_graph_verify_pair: one two-row
 * pass over the committed token and a draft, rolled back through
 * qwen4exp_graph_verify_rollback whenever the draft is rejected.  Every
 * step compares both rows' logits bit for bit against the oracle and then
 * every persistent state the pass advanced: Gated DeltaNet recurrent and
 * convolution states, the PLE convolution state and n-gram hash, and the
 * sparse-attention caches up to the committed length.  Drafts alternate
 * between the oracle's successor (accepted) and a wrong token (rejected)
 * so both frontiers are exercised across at least one index block.
 */
#include "../ds4.c"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double bench_now(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static const char *kPrompt =
    "The DGX Spark places a GB10 system on a chip next to 128 gigabytes of "
    "unified memory, which is why a serving engine written for it treats "
    "every byte of weight traffic as the resource that decides tokens per "
    "second. A decoder that streams a mixture-of-experts model through the "
    "memory bus can hide almost nothing behind compute, so the engine keeps "
    "activations resident, quantizes the expert matrices to a mixed recipe, "
    "and pins the per-layer embedding table to the solid state drive where a "
    "bounded page cache serves the n-gram rows on demand. The runtime is "
    "organised as one long-lived weight owner and a restartable worker: the "
    "owner maps the shards once, repacks the dense Q8 tensors into an "
    "aligned layout, and exports the mapping over a manifest, while the "
    "worker imports it, allocates its recurrent states and sparse-attention "
    "caches, and answers chat completions on a plain HTTP port. When a "
    "request arrives the prompt is prefilled in fixed chunks so that the "
    "gated delta-net recurrence, the block-indexed sparse attention and the "
    "routed experts all see a batch that is wide enough to amortise their "
    "launches, and the checkpointed states are written to a disk-backed "
    "cache so that a follow-up turn with the same prefix resumes without "
    "recomputing it. Decoding is different: each generated token needs the "
    "full stack of forty-eight layers with a single row, the hyper-"
    "connection streams are mixed in bfloat16 with a row-stable kernel so "
    "that a continuation reproduces the serial oracle exactly, and the "
    "embedded one-layer drafter proposes one successor that the target "
    "confirms or rejects. The measurement discipline that goes with all of "
    "this is strict: a change is compared against the previous adopted "
    "state on identical prompts with three fresh workers per variant, the "
    "kernel metric for decode is milliseconds per verify step rather than "
    "tokens per second, and nothing is published without the git revision, "
    "the artifact revision, the exact command, the token counts and the "
    "correctness fixture result attached to the number.";

typedef struct {
    ds4_session *session;
    ds4_qwen_gpu_graph *graph;
} lane;

static int compare_tensor(const char *label, const ds4_gpu_tensor *a,
                          const ds4_gpu_tensor *b, uint64_t bytes,
                          float *scratch_a, float *scratch_b) {
    if (bytes == 0u) return 0;
    if (!ds4_gpu_tensor_read(a, 0, scratch_a, bytes) ||
        !ds4_gpu_tensor_read(b, 0, scratch_b, bytes)) {
        fprintf(stderr, "  %s: tensor read failed (%llu bytes)\n", label,
                (unsigned long long)bytes);
        return 1;
    }
    if (memcmp(scratch_a, scratch_b, bytes) == 0) return 0;
    const uint64_t n = bytes / sizeof(float);
    uint64_t diff = 0;
    double max_abs = 0.0;
    for (uint64_t i = 0; i < n; i++) {
        if (scratch_a[i] != scratch_b[i]) {
            diff++;
            const double d = fabs((double)scratch_a[i] - (double)scratch_b[i]);
            if (d > max_abs) max_abs = d;
        }
    }
    fprintf(stderr, "  %s: %llu of %llu values differ, max abs %.3e\n", label,
            (unsigned long long)diff, (unsigned long long)n, max_abs);
    return 1;
}

static int compare_logits(const char *label, const float *a, const float *b) {
    if (memcmp(a, b, (size_t)DS4_N_VOCAB * sizeof(float)) == 0) return 0;
    uint64_t diff = 0;
    double max_abs = 0.0;
    for (uint32_t i = 0; i < DS4_N_VOCAB; i++) {
        if (a[i] != b[i]) {
            diff++;
            const double d = fabs((double)a[i] - (double)b[i]);
            if (d > max_abs) max_abs = d;
        }
    }
    fprintf(stderr, "  %s: %llu logits differ, max abs %.3e, argmax %d vs %d\n",
            label, (unsigned long long)diff, max_abs,
            sample_argmax(a, DS4_N_VOCAB), sample_argmax(b, DS4_N_VOCAB));
    return 1;
}

static int compare_states(const ds4_qwen_gpu_graph *ga,
                          const ds4_qwen_gpu_graph *gb,
                          float *scratch_a, float *scratch_b) {
    int failures = 0;
    char label[96];
    if (ga->length != gb->length) {
        fprintf(stderr, "  graph length %u vs %u\n", ga->length, gb->length);
        failures++;
    }
    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        if (ds4_qwen4exp_layer_is_full_attention(il)) {
            const ds4_qwen_qsa_state *a = &ga->qsa_state[il];
            const ds4_qwen_qsa_state *b = &gb->qsa_state[il];
            if (a->length != b->length) {
                fprintf(stderr, "  layer %u qsa length %u vs %u\n", il,
                        a->length, b->length);
                failures++;
                continue;
            }
            const uint64_t kv_bytes = (uint64_t)a->length * a->kv_heads *
                a->head_dim * sizeof(float);
            const uint64_t index_bytes =
                (uint64_t)a->length * a->index_head_dim * sizeof(float);
            const uint64_t pooled_bytes = (uint64_t)(a->length / a->ratio) *
                a->index_head_dim * sizeof(float);
            snprintf(label, sizeof(label), "layer %u qsa k_cache", il);
            failures += compare_tensor(label, a->k_cache, b->k_cache, kv_bytes,
                                       scratch_a, scratch_b);
            snprintf(label, sizeof(label), "layer %u qsa v_cache", il);
            failures += compare_tensor(label, a->v_cache, b->v_cache, kv_bytes,
                                       scratch_a, scratch_b);
            snprintf(label, sizeof(label), "layer %u qsa raw_index", il);
            failures += compare_tensor(label, a->raw_index, b->raw_index,
                                       index_bytes, scratch_a, scratch_b);
            snprintf(label, sizeof(label), "layer %u qsa pooled_index", il);
            failures += compare_tensor(label, a->pooled_index, b->pooled_index,
                                       pooled_bytes, scratch_a, scratch_b);
            continue;
        }
        const ds4_qwen_gdn_state *a = &ga->gdn_state[il];
        const ds4_qwen_gdn_state *b = &gb->gdn_state[il];
        snprintf(label, sizeof(label), "layer %u gdn recurrent", il);
        failures += compare_tensor(label, a->recurrent, b->recurrent,
                                   ds4_gpu_tensor_bytes(a->recurrent),
                                   scratch_a, scratch_b);
        snprintf(label, sizeof(label), "layer %u gdn conv", il);
        failures += compare_tensor(label, a->conv, b->conv,
                                   ds4_gpu_tensor_bytes(a->conv),
                                   scratch_a, scratch_b);
    }
    failures += compare_tensor("ple conv_state", ga->ple.conv_state,
                               gb->ple.conv_state,
                               ds4_gpu_tensor_bytes(ga->ple.conv_state),
                               scratch_a, scratch_b);
    if (memcmp(&ga->ple.hash_state, &gb->ple.hash_state,
               sizeof(ga->ple.hash_state)) != 0) {
        fprintf(stderr, "  ple hash state differs\n");
        failures++;
    }
    return failures;
}

int main(int argc, char **argv) {
    if (argc < 2 || argc > 3) {
        fprintf(stderr, "usage: %s <first-model-shard.gguf> [steps]\n", argv[0]);
        return 2;
    }
    const int steps = argc == 3 ? atoi(argv[2]) : 48;
    if (setenv("DS4_SESSION_LAZY_GRAPH", "1", 1) != 0 ||
        setenv("DS4_NO_BOOT_PREWARM", "1", 1) != 0 ||
        setenv("DS4_MEMGOV", "observe", 1) != 0) {
        perror("setenv");
        return 1;
    }
    ds4_engine_options opt = {0};
    opt.model_path = argv[1];
    opt.backend = DS4_BACKEND_CUDA;
    opt.n_threads = 8;
    opt.mtp_draft_tokens = 2;
    opt.defer_boot_prewarm = true;
    ds4_engine *engine = NULL;
    if (ds4_engine_open(&engine, &opt) != 0) {
        fprintf(stderr, "Qwen engine open failed\n");
        return 1;
    }
    int failed = 0;
    char err[256] = "";
    ds4_tokens prompt = {0};
    ds4_tokenize_text(engine, kPrompt, &prompt);
    lane a = {0}, b = {0};
    float *logits_pair = NULL, *oracle0 = NULL, *oracle1 = NULL;
    float *scratch_a = NULL, *scratch_b = NULL;
    if (prompt.len <= 0) {
        fprintf(stderr, "tokenizer failed\n");
        failed = 1;
        goto cleanup;
    }
    if (ds4_session_create(&a.session, engine, 2048) != 0 ||
        ds4_session_create(&b.session, engine, 2048) != 0 ||
        ds4_session_sync(a.session, &prompt, err, sizeof(err)) != 0 ||
        ds4_session_sync(b.session, &prompt, err, sizeof(err)) != 0) {
        fprintf(stderr, "session sync failed: %s\n", err);
        failed = 1;
        goto cleanup;
    }
    a.graph = &a.session->qwen_graph;
    b.graph = &b.session->qwen_graph;
    if (!a.session->qwen_graph_ready || !b.session->qwen_graph_ready ||
        a.graph->length != (uint32_t)prompt.len ||
        b.graph->length != (uint32_t)prompt.len) {
        fprintf(stderr, "graphs not ready after sync\n");
        failed = 1;
        goto cleanup;
    }
    const uint64_t largest = 64u << 20;
    logits_pair = xmalloc(2u * (size_t)DS4_N_VOCAB * sizeof(float));
    oracle0 = xmalloc((size_t)DS4_N_VOCAB * sizeof(float));
    oracle1 = xmalloc((size_t)DS4_N_VOCAB * sizeof(float));
    scratch_a = xmalloc(largest);
    scratch_b = xmalloc(largest);
    ds4_engine *e = a.session->engine;
    if (!qwen4exp_graph_verify_ensure(b.graph)) {
        fprintf(stderr, "verify checkpoint allocation failed\n");
        failed = 1;
        goto cleanup;
    }
    printf("prompt %d tokens, %d verify steps, index ratio %u\n", prompt.len,
           steps, b.graph->qsa_state[3].ratio);

    int token = ds4_session_argmax(a.session);
    int hits = 0, misses = 0, logit_failures = 0, state_failures = 0;
    double serial_s = 0.0, pair_s = 0.0;
    for (int step = 0; step < steps; step++) {
        const uint32_t pos = a.graph->length;
        /* Oracle: commit `token`, learn its successor. */
        double t0 = bench_now();
        if (ds4_session_eval(a.session, token, err, sizeof(err)) != 0) {
            fprintf(stderr, "oracle eval failed at %u: %s\n", pos, err);
            failed = 1;
            goto cleanup;
        }
        ds4_gpu_synchronize();
        serial_s += bench_now() - t0;
        memcpy(oracle0, a.session->logits, (size_t)DS4_N_VOCAB * sizeof(float));
        const int successor = sample_argmax(oracle0, DS4_N_VOCAB);
        const bool want_hit = step % 3 != 2;
        const int draft = want_hit ? successor
                                   : (successor + 1) % (int)DS4_N_VOCAB;
        if (want_hit) {
            t0 = bench_now();
            if (ds4_session_eval(a.session, draft, err, sizeof(err)) != 0) {
                fprintf(stderr, "oracle draft eval failed at %u: %s\n",
                        pos + 1u, err);
                failed = 1;
                goto cleanup;
            }
            ds4_gpu_synchronize();
            serial_s += bench_now() - t0;
            memcpy(oracle1, a.session->logits,
                   (size_t)DS4_N_VOCAB * sizeof(float));
        }

        /* Pair lane: one two-row pass, then accept or roll back. */
        t0 = bench_now();
        if (!qwen4exp_graph_verify_pair(
                b.graph, &e->model, &e->weights, e->qwen_ple_store,
                e->qwen_ple_cuda, token, draft, pos, logits_pair)) {
            fprintf(stderr, "verify pair failed at %u\n", pos);
            failed = 1;
            goto cleanup;
        }
        ds4_gpu_synchronize();
        const bool hit = sample_argmax(logits_pair, DS4_N_VOCAB) == draft;
        if (!hit && !qwen4exp_graph_verify_rollback(
                        b.graph, e->qwen_ple_store, pos)) {
            fprintf(stderr, "rollback failed at %u\n", pos);
            failed = 1;
            goto cleanup;
        }
        ds4_gpu_synchronize();
        pair_s += bench_now() - t0;

        int step_failures = 0;
        if (hit != want_hit) {
            fprintf(stderr, "step %d: expected %s, pair lane %s\n", step,
                    want_hit ? "hit" : "miss", hit ? "hit" : "miss");
            step_failures++;
        }
        step_failures += compare_logits("row 0 logits", logits_pair, oracle0);
        if (hit)
            step_failures += compare_logits("row 1 logits",
                                            logits_pair + DS4_N_VOCAB, oracle1);
        logit_failures += step_failures;
        const int state_diff = compare_states(a.graph, b.graph,
                                              scratch_a, scratch_b);
        state_failures += state_diff;
        if (step_failures || state_diff)
            fprintf(stderr, "step %d (pos %u, %s): %d logit / %d state "
                    "mismatches\n", step, pos, hit ? "hit" : "miss",
                    step_failures, state_diff);
        if (hit) {
            hits++;
            token = sample_argmax(oracle1, DS4_N_VOCAB);
        } else {
            misses++;
            token = successor;
        }
        if (hit != want_hit) break;
    }
    printf("steps %d: %d hits, %d misses, %d logit mismatches, %d state "
           "mismatches\n", steps, hits, misses, logit_failures, state_failures);
    printf("wall: serial one-row passes %.3f s, pair passes %.3f s "
           "(%.1f%% of serial)\n", serial_s, pair_s,
           serial_s > 0.0 ? 100.0 * pair_s / serial_s : 0.0);
    if (logit_failures || state_failures) failed = 1;
    printf("%s\n", failed ? "FAIL" : "PASS");

cleanup:
    free(scratch_b);
    free(scratch_a);
    free(oracle1);
    free(oracle0);
    free(logits_pair);
    ds4_session_free(b.session);
    ds4_session_free(a.session);
    ds4_tokens_free(&prompt);
    ds4_engine_close(engine);
    return failed ? 1 : 0;
}
