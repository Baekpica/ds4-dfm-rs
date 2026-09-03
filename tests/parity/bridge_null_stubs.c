/* Link stubs so tests/parity/bridge_null_oracle can use ds4_bridge.o
 * without the CUDA engine. NULL-handle tests never reach these. */

#include "ds4.h"
#include "ds4_distributed.h"
#include "native/bridge/ds4_host_load.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define STUB(name) \
    do { \
        fprintf(stderr, "bridge_null_oracle stub called: %s\n", name); \
        abort(); \
    } while (0)

unsigned bridge_payload_load_calls;
int64_t bridge_payload_load_offset;
uint64_t bridge_payload_load_bytes;
int bridge_routed_quant_bits;
unsigned bridge_boot_prewarm_calls;
int bridge_sync_rc;
unsigned bridge_sync_calls;
unsigned bridge_progress_sets;
unsigned bridge_progress_clears;
int bridge_progress_active;
int bridge_batch_max_seq;
int bridge_static_batch_calls;
int bridge_static_batch_n;
int bridge_static_batch_prompt_lens[8];
int bridge_static_batch_max_new[8];
int bridge_static_batch_eos[8];
int bridge_bank_committed;
int bridge_bank_tokens[8];
uint64_t bridge_bank_generation;
unsigned bridge_bank_save_calls;
int bridge_bank_save_result;
unsigned bridge_bank_load_calls;
int64_t bridge_bank_load_offset;
uint64_t bridge_bank_load_bytes;
int bridge_cont_run;
ds4_cont_seq_stats bridge_cont_stats;
int bridge_dist_open_enabled;
int bridge_dist_prepare_calls;
int bridge_dist_run_calls;
int bridge_dist_run_ctx;
int bridge_dist_run_result;
int bridge_dist_route_ready;
ds4_engine_options bridge_dist_engine_options;
ds4_dist_options bridge_dist_run_options;

static ds4_session_progress_fn bridge_progress;
static void *bridge_progress_ud;

void ds4_host_tensor_dir_install(const ds4_host_tensor_dir *d) { (void)d; }
void ds4_host_tensor_dir_clear(void) {}
void ds4_host_shape_install(const ds4_host_shape *s) { (void)s; }
void ds4_host_shape_clear(void) {}
void ds4_host_vocab_install(const ds4_host_vocab *v) { (void)v; }
void ds4_host_vocab_clear(void) {}
void ds4_host_bind_map_install(const ds4_host_bind_map *m) { (void)m; }
void ds4_host_bind_map_clear(void) {}
void ds4_host_mtp_bind_map_install(const ds4_host_bind_map *m) { (void)m; }
void ds4_host_mtp_bind_map_clear(void) {}
void ds4_host_dspark_bind_map_install(const ds4_host_bind_map *m) { (void)m; }
void ds4_host_dspark_bind_map_clear(void) {}

int ds4_engine_open(ds4_engine **out, const ds4_engine_options *opt) {
    if (!bridge_dist_open_enabled || !out || !opt) {
        STUB("ds4_engine_open");
    }
    bridge_dist_engine_options = *opt;
    *out = malloc(1);
    return *out ? 0 : 1;
}
void ds4_engine_close(ds4_engine *e) {
    if (!bridge_dist_open_enabled || !e) {
        STUB("ds4_engine_close");
    }
    free(e);
}
void ds4_engine_boot_prewarm(ds4_engine *e) {
    if (!e) STUB("ds4_engine_boot_prewarm");
    bridge_boot_prewarm_calls++;
}
int ds4_engine_model_id(ds4_engine *e) { (void)e; STUB("ds4_engine_model_id"); }
int ds4_engine_routed_quant_bits(ds4_engine *e) {
    if (!e) STUB("ds4_engine_routed_quant_bits");
    return bridge_routed_quant_bits;
}
int ds4_engine_session_graph_fit_quote(ds4_engine *e, int ctx_size,
                                       ds4_session_graph_fit_quote *q) {
    (void)e; (void)ctx_size; (void)q;
    STUB("ds4_engine_session_graph_fit_quote");
}
int ds4_session_create(ds4_session **out, ds4_engine *e, int ctx_size) {
    (void)out; (void)e; (void)ctx_size; STUB("ds4_session_create");
}
void ds4_session_free(ds4_session *s) { (void)s; STUB("ds4_session_free"); }
int ds4_session_power(ds4_session *s) {
    (void)s; STUB("ds4_session_power");
}
int ds4_session_set_power(ds4_session *s, int power_percent) {
    (void)s; (void)power_percent; STUB("ds4_session_set_power");
}
int ds4_session_sync(ds4_session *s, const ds4_tokens *prompt, char *err, size_t errlen) {
    (void)s; (void)err; (void)errlen;
    bridge_sync_calls++;
    if (bridge_progress) {
        bridge_progress(bridge_progress_ud, "prefill_display", 1024, prompt->len);
        bridge_progress(bridge_progress_ud, "prefill_chunk", 0, prompt->len);
        bridge_progress(bridge_progress_ud, "prefill_chunk", 4096, prompt->len);
        bridge_progress(bridge_progress_ud, "prefill_chunk", prompt->len, prompt->len);
    }
    return bridge_sync_rc;
}
int ds4_engine_vision_probe(ds4_engine *e,
                            const uint8_t *encoded, size_t encoded_len,
                            ds4_vision_image_info *out,
                            char *err, size_t errlen) {
    (void)e; (void)encoded; (void)encoded_len; (void)out; (void)err; (void)errlen;
    STUB("ds4_engine_vision_probe");
}
int ds4_engine_vision_encode_memory(ds4_engine *e,
                                    const uint8_t *encoded, size_t encoded_len,
                                    ds4_vision_embedding *out,
                                    char *err, size_t errlen) {
    (void)e; (void)encoded; (void)encoded_len; (void)out; (void)err; (void)errlen;
    STUB("ds4_engine_vision_encode_memory");
}
void ds4_vision_embedding_free(ds4_vision_embedding *embedding) {
    (void)embedding;
    STUB("ds4_vision_embedding_free");
}
int ds4_session_sync_multimodal(ds4_session *s,
                                const ds4_tokens *prompt,
                                const ds4_vision_span *spans,
                                uint32_t span_count,
                                char *err, size_t errlen) {
    (void)s; (void)prompt; (void)spans; (void)span_count; (void)err; (void)errlen;
    STUB("ds4_session_sync_multimodal");
}
void ds4_session_set_progress(ds4_session *s, ds4_session_progress_fn fn, void *ud) {
    (void)s;
    bridge_progress = fn;
    bridge_progress_ud = ud;
    bridge_progress_active = fn != NULL;
    if (fn) {
        bridge_progress_sets++;
    } else {
        bridge_progress_clears++;
    }
}
int ds4_session_eval(ds4_session *s, int token, char *err, size_t errlen) {
    (void)s; (void)token; (void)err; (void)errlen; STUB("ds4_session_eval");
}
int ds4_session_eval_speculative_argmax(ds4_session *s, int first_token,
                                        int max_tokens, int eos_token,
                                        int *accepted, int accepted_cap,
                                        char *err, size_t errlen) {
    (void)s; (void)first_token; (void)max_tokens; (void)eos_token;
    (void)accepted; (void)accepted_cap; (void)err; (void)errlen;
    STUB("ds4_session_eval_speculative_argmax");
}
int ds4_session_layer_slice_reset(ds4_session *s, char *err, size_t errlen) {
    (void)s; (void)err; (void)errlen;
    STUB("ds4_session_layer_slice_reset");
}
int ds4_session_eval_layer_slice(ds4_session *s,
                                 const int *tokens,
                                 uint32_t n_tokens,
                                 uint32_t pos0,
                                 uint32_t layer_start,
                                 uint32_t layer_end,
                                 const float *input_hc,
                                 float *output_hc,
                                 bool output_logits,
                                 float *logits,
                                 char *err,
                                 size_t errlen) {
    (void)s; (void)tokens; (void)n_tokens; (void)pos0;
    (void)layer_start; (void)layer_end; (void)input_hc; (void)output_hc;
    (void)output_logits; (void)logits; (void)err; (void)errlen;
    STUB("ds4_session_eval_layer_slice");
}
int ds4_session_argmax(ds4_session *s) { (void)s; STUB("ds4_session_argmax"); }
int ds4_session_argmax_excluding(ds4_session *s, int excluded_id) {
    (void)s; (void)excluded_id; STUB("ds4_session_argmax_excluding");
}
int ds4_session_pos(ds4_session *s) { (void)s; STUB("ds4_session_pos"); }
int ds4_session_ctx(ds4_session *s) { (void)s; STUB("ds4_session_ctx"); }
int ds4_session_graph_pending(const ds4_session *s) {
    (void)s; STUB("ds4_session_graph_pending");
}
void ds4_session_rewind(ds4_session *s, int pos) {
    (void)s; (void)pos; STUB("ds4_session_rewind");
}
void ds4_session_invalidate(ds4_session *s) { (void)s; STUB("ds4_session_invalidate"); }
uint64_t ds4_session_generation(const ds4_session *s) {
    (void)s; STUB("ds4_session_generation");
}
int ds4_session_prefill_cap(ds4_session *s) { (void)s; STUB("ds4_session_prefill_cap"); }
int ds4_session_exaone_rewind_span(ds4_session *s) {
    (void)s; STUB("ds4_session_exaone_rewind_span");
}
int ds4_session_distributed_route_ready(ds4_session *s, char *err, size_t errlen) {
    (void)err; (void)errlen;
    if (!s) {
        STUB("ds4_session_distributed_route_ready");
    }
    return bridge_dist_route_ready;
}
int ds4_session_sample(ds4_session *s, float temperature, int top_k, float top_p,
                       float min_p, uint64_t *rng) {
    (void)s; (void)temperature; (void)top_k; (void)top_p; (void)min_p; (void)rng;
    STUB("ds4_session_sample");
}
int ds4_session_save_payload(ds4_session *s, FILE *fp, char *err, size_t errlen) {
    (void)s; (void)fp; (void)err; (void)errlen; STUB("ds4_session_save_payload");
}
int ds4_session_save_layer_payload(ds4_session *s, FILE *fp,
                                   uint32_t layer_start, uint32_t layer_end,
                                   char *err, size_t errlen) {
    (void)s; (void)fp; (void)layer_start; (void)layer_end;
    (void)err; (void)errlen;
    STUB("ds4_session_save_layer_payload");
}
int ds4_session_load_layer_payload(ds4_session *s, FILE *fp,
                                   uint64_t payload_bytes,
                                   const int *tokens, uint32_t n_tokens,
                                   uint32_t layer_start, uint32_t layer_end,
                                   char *err, size_t errlen) {
    (void)s; (void)fp; (void)payload_bytes; (void)tokens; (void)n_tokens;
    (void)layer_start; (void)layer_end; (void)err; (void)errlen;
    STUB("ds4_session_load_layer_payload");
}
int ds4_session_load_payload(ds4_session *s, FILE *fp, uint64_t payload_bytes,
                             char *err, size_t errlen) {
    (void)s; (void)err; (void)errlen;
    bridge_payload_load_calls++;
    bridge_payload_load_offset = (int64_t)ftello(fp);
    bridge_payload_load_bytes = payload_bytes;
    return 0;
}
int ds4_session_save_snapshot(ds4_session *s, ds4_session_snapshot *snap,
                              char *err, size_t errlen) {
    (void)s; (void)snap; (void)err; (void)errlen;
    STUB("ds4_session_save_snapshot");
}
int ds4_session_load_snapshot(ds4_session *s, const ds4_session_snapshot *snap,
                              char *err, size_t errlen) {
    (void)s; (void)snap; (void)err; (void)errlen;
    STUB("ds4_session_load_snapshot");
}
void ds4_session_snapshot_free(ds4_session_snapshot *snap) { (void)snap; }
void ds4_tokenize_text(ds4_engine *e, const char *text, ds4_tokens *out) {
    (void)e; (void)text; (void)out; STUB("ds4_tokenize_text");
}
void ds4_encode_chat_prompt(ds4_engine *e, const char *system, const char *prompt,
                            ds4_think_mode think_mode, ds4_tokens *out) {
    (void)e; (void)system; (void)prompt; (void)think_mode; (void)out;
    STUB("ds4_encode_chat_prompt");
}
int ds4_session_top_logprobs(ds4_session *s, ds4_token_score *out, int k) {
    (void)s; (void)out; (void)k; STUB("ds4_session_top_logprobs");
}
int ds4_session_copy_logits(ds4_session *s, float *out, int cap) {
    (void)s; (void)out; (void)cap; STUB("ds4_session_copy_logits");
}
int ds4_session_output_head_bench(ds4_session *s, int iters, FILE *fp,
                                  char *err, size_t errlen) {
    (void)s; (void)iters; (void)fp; (void)err; (void)errlen;
    STUB("ds4_session_output_head_bench");
}
void ds4_tokenize_rendered_chat(ds4_engine *e, const char *text, ds4_tokens *out) {
    (void)e; (void)text; (void)out; STUB("ds4_tokenize_rendered_chat");
}
void ds4_tokens_free(ds4_tokens *tv) { (void)tv; STUB("ds4_tokens_free"); }
char *ds4_token_text(ds4_engine *e, int token, size_t *len) {
    (void)e; (void)token; (void)len; STUB("ds4_token_text");
}
int ds4_token_eos(ds4_engine *e) { (void)e; STUB("ds4_token_eos"); }
bool ds4_token_is_stop(ds4_engine *e, int token) {
    (void)e; (void)token; STUB("ds4_token_is_stop");
}

int ds4_batch_ctx_create_fit(ds4_engine *e, int ctx_size, int max_seq,
                             int max_total_tokens, ds4_batch_ctx **out,
                             char *err, size_t errlen) {
    (void)e; (void)ctx_size; (void)max_seq; (void)max_total_tokens;
    (void)out; (void)err; (void)errlen;
    STUB("ds4_batch_ctx_create_fit");
}
void ds4_batch_ctx_destroy(ds4_batch_ctx *ctx) { (void)ctx; }
int ds4_batch_ctx_max_seq(const ds4_batch_ctx *ctx) {
    (void)ctx;
    return bridge_batch_max_seq;
}
int ds4_batch_ctx_raw_cap(const ds4_batch_ctx *ctx) { (void)ctx; return 0; }
int ds4_batch_ctx_seq_cap(const ds4_batch_ctx *ctx) { (void)ctx; return 0; }
bool ds4_batch_ctx_supports_partial_reuse(const ds4_batch_ctx *ctx) {
    (void)ctx;
    return false;
}
uint64_t ds4_batch_ctx_trim_free(ds4_batch_ctx *ctx, uint64_t want_bytes) {
    (void)ctx;
    (void)want_bytes;
    return 0;
}
int ds4_engine_batched_generate_ctx(ds4_batch_ctx *ctx,
                                    const ds4_tokens *prompts, int n,
                                    const int *max_new_tokens,
                                    const int *eos_ids,
                                    ds4_batch_gen_result *out,
                                    char *err, size_t errlen) {
    (void)ctx; (void)err; (void)errlen;
    bridge_static_batch_calls++;
    bridge_static_batch_n = n;
    for (int i = 0; i < n; i++) {
        bridge_static_batch_prompt_lens[i] = prompts[i].len;
        bridge_static_batch_max_new[i] = max_new_tokens[i];
        bridge_static_batch_eos[i] = eos_ids[i];
    }
    out[0].tokens = malloc(2 * sizeof(*out[0].tokens));
    out[1].tokens = malloc(3 * sizeof(*out[1].tokens));
    if (!out[0].tokens || !out[1].tokens) {
        free(out[0].tokens);
        free(out[1].tokens);
        out[0].tokens = NULL;
        out[1].tokens = NULL;
        return 1;
    }
    out[0].tokens[0] = 101;
    out[0].tokens[1] = 102;
    out[0].n_tokens = 2;
    out[0].finish = 1;
    out[1].tokens[0] = 201;
    out[1].tokens[1] = 202;
    out[1].tokens[2] = 203;
    out[1].n_tokens = 3;
    out[1].finish = 0;
    return 0;
}
int ds4_batch_ctx_bank_committed(const ds4_batch_ctx *ctx, int bank,
                                 const int **tokens) {
    (void)ctx;
    if (bank < 0 || bank >= bridge_batch_max_seq) {
        if (tokens) *tokens = NULL;
        return 0;
    }
    if (tokens) {
        *tokens = bridge_bank_committed > 0 ? bridge_bank_tokens : NULL;
    }
    return bridge_bank_committed;
}
uint64_t ds4_batch_ctx_bank_generation(const ds4_batch_ctx *ctx, int bank) {
    (void)ctx;
    return bank >= 0 && bank < bridge_batch_max_seq ? bridge_bank_generation : 0;
}
int ds4_cont_bank_save_payload(ds4_batch_ctx *ctx, uint32_t bank, FILE *fp,
                               char *err, size_t errlen) {
    (void)ctx; (void)bank;
    bridge_bank_save_calls++;
    if (bridge_bank_save_result != 0) {
        if (err && errlen) snprintf(err, errlen, "bank save failed");
        return bridge_bank_save_result;
    }
    return fwrite("BANK", 1, 4, fp) == 4 ? 0 : 1;
}
int ds4_cont_bank_restore_payload(ds4_batch_ctx *ctx, uint32_t bank, FILE *fp,
                                  uint64_t payload_bytes,
                                  char *err, size_t errlen) {
    (void)ctx; (void)bank; (void)err; (void)errlen;
    bridge_bank_load_calls++;
    bridge_bank_load_offset = (int64_t)ftello(fp);
    bridge_bank_load_bytes = payload_bytes;
    return 0;
}
int ds4_engine_continuous_generate(ds4_batch_ctx *ctx,
                                   int (*admit)(void *ud, ds4_cont_request *req),
                                   int (*on_token)(void *ud, void *user, int token),
                                   void (*on_done)(void *ud, void *user,
                                                   const int *tokens, int n, int finish),
                                   void *ud, char *err, size_t errlen) {
    (void)ctx; (void)admit; (void)on_token;
    (void)err; (void)errlen;
    if (bridge_cont_run) {
        static const int tokens[] = {1, 2, 3, 4, 5};
        on_done(ud, (void *)42, tokens, 5, 1);
        return 0;
    }
    STUB("ds4_engine_continuous_generate");
}

int ds4_cont_last_done_stats(const ds4_batch_ctx *ctx,
                             ds4_cont_seq_stats *out) {
    (void)ctx;
    if (!bridge_cont_run || !out) {
        return 0;
    }
    *out = bridge_cont_stats;
    return 1;
}

int ds4_qwen_image_probe(const uint8_t *data, size_t data_len,
                         ds4_qwen_image_info *info,
                         char *err, size_t errlen) {
    if (!data || data_len != 3 || !info) {
        if (err && errlen) snprintf(err, errlen, "bad image probe");
        return 1;
    }
    *info = (ds4_qwen_image_info){
        .source_width = 1,
        .source_height = 2,
        .resized_width = 256,
        .resized_height = 512,
        .grid_h = 32,
        .grid_w = 16,
        .token_count = 128,
    };
    return 0;
}

int ds4_qwen_image_pixel_hash(const uint8_t *data, size_t data_len,
                              uint64_t *hash,
                              char *err, size_t errlen) {
    if (!data || data_len != 3 || !hash) {
        if (err && errlen) snprintf(err, errlen, "bad image hash");
        return 1;
    }
    *hash = UINT64_C(0x0123456789abcdef);
    return 0;
}

int ds4_gpu_mem_census_read(int consumer_class, int domain, ds4_mem_cell *out) {
    (void)consumer_class; (void)domain; (void)out;
    return 1;
}
uint64_t ds4_gpu_mem_census_faults(void) { return 0; }
uint64_t ds4_gpu_mem_census_epoch_begin(void) { return 0; }
int ds4_gpu_mem_census_epoch_verify(uint64_t began) { return began == 0; }
int ds4_gpu_mem_observe(ds4_mem_observation *out) {
    if (out) {
        memset(out, 0, sizeof(*out));
        out->status = DS4_MEMOBS_UNSUPPORTED;
        out->source = DS4_MEMOBS_SRC_NONE;
    }
    return 1;
}
uint64_t ds4_gpu_substrate_outstanding(void) { return 0; }

int ds4_dist_prepare_engine_options(const ds4_dist_options *opt,
                                    ds4_engine_options *engine,
                                    char *err, size_t errlen) {
    (void)err; (void)errlen;
    if (!opt || !engine) {
        STUB("ds4_dist_prepare_engine_options");
    }
    bridge_dist_prepare_calls++;
    engine->distributed = *opt;
    if (opt->role != DS4_DISTRIBUTED_NONE) {
        engine->load_slice = true;
        engine->load_layer_start = opt->layers.start;
        engine->load_layer_end = opt->layers.has_output ? UINT32_MAX : opt->layers.end;
        engine->load_output = opt->layers.has_output ||
                              opt->role == DS4_DISTRIBUTED_COORDINATOR;
    }
    return 0;
}

int ds4_dist_run(ds4_engine *engine, const ds4_dist_options *opt,
                 const ds4_dist_generation_options *gen) {
    if (!engine || !opt || !gen) {
        STUB("ds4_dist_run");
    }
    bridge_dist_run_calls++;
    bridge_dist_run_options = *opt;
    bridge_dist_run_ctx = gen->ctx_size;
    return bridge_dist_run_result;
}
