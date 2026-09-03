#ifndef DS4_BRIDGE_H
#define DS4_BRIDGE_H

#include <stddef.h>
#include <stdint.h>

#include "ds4_host_load.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Narrow Rust ↔ native ABI.  Do not include ds4.h from Rust.  Handles are
 * opaque; the C structs below may contain ds4_engine / ds4_session pointers
 * but those layouts are not part of this contract.
 *
 * Freeze: do not add new ds4_bridge_* except create / load / session /
 * prefill / decode / KV / destroy.  Existing extras stay; do not mass-delete.
 * Policy: docs/rust-migration/FFI_CONTRACT.md */

typedef struct ds4_bridge_model ds4_bridge_model;
typedef struct ds4_bridge_session ds4_bridge_session;
typedef struct ds4_bridge_snapshot ds4_bridge_snapshot;

/* Durable prefill frontiers only. Callback runs on the sync calling thread. */
typedef void (*ds4_bridge_prefill_fn)(void *ud, int32_t current, int32_t total);

enum {
    DS4_BRIDGE_BACKEND_CUDA = 0,
    DS4_BRIDGE_BACKEND_METAL = 1,
    DS4_BRIDGE_BACKEND_CPU = 2
};

enum {
    DS4_BRIDGE_DISTRIBUTED_NONE = 0,
    DS4_BRIDGE_DISTRIBUTED_COORDINATOR = 1,
    DS4_BRIDGE_DISTRIBUTED_WORKER = 2
};

/* Rust owns these strings for the lifetime of the returned model.  The
 * structure is translated into ds4_distributed_options inside the bridge;
 * neither its layout nor the ds4.h layout crosses into Rust application code. */
typedef struct {
    int32_t role;              /* DS4_BRIDGE_DISTRIBUTED_* */
    uint32_t layer_start;
    uint32_t layer_end;
    int32_t has_output;
    const char *listen_host;   /* optional; borrowed for model lifetime */
    int32_t listen_port;
    const char *coordinator_host; /* optional; borrowed for model lifetime */
    int32_t coordinator_port;
    uint32_t prefill_chunk;
    uint32_t prefill_window;
    uint32_t activation_bits;
    int32_t replay_check;
    int32_t debug;
} ds4_bridge_distributed_options;

#define DS4_BRIDGE_MAX_DIMS 8

typedef struct {
    const char *name;         /* borrowed for the call */
    uint32_t required;        /* 1 = weights_bind required_tensor */
    uint32_t ndim;
    uint64_t dim[DS4_BRIDGE_MAX_DIMS];
    uint32_t type;
    uint64_t rel_offset;
    uint64_t abs_offset;
    uint64_t bytes;
    uint32_t shard;
    uint32_t found;
} ds4_bridge_bind_slot;

typedef struct {
    const char *path;         /* borrowed for the call */
    uint64_t size;
    uint64_t base;
} ds4_bridge_shard;

/* Host tensor inventory + weights_bind name table.  Native bind consumes
 * this plan: check before ds4_engine_open, match after C parse when an
 * engine is open. */
typedef struct {
    uint32_t n_slots;
    const ds4_bridge_bind_slot *slots;
    uint32_t n_shards;
    const ds4_bridge_shard *shards;
    uint64_t data_pos;
    uint64_t alignment;
    uint64_t page;
} ds4_bridge_bind_plan;

typedef struct {
    const char *model_path;
    const char *vision_path;  /* optional GLM-5.3 vision encoder; borrowed */
    int backend;              /* DS4_BRIDGE_BACKEND_* */
    int n_threads;
    int defer_boot_prewarm;   /* nonzero => skip boot prewarm inside open */
    int32_t power_percent;    /* 1..100; native default is 100 */
    int32_t warm_weights;
    int32_t quality;
    const ds4_bridge_bind_plan *plan; /* optional; borrowed for the call */
    const ds4_host_tensor_dir *tensors; /* optional full inventory; borrowed */
    const ds4_host_shape *shape; /* optional; skip C validate when set */
    const ds4_host_vocab *vocab; /* optional; skip C vocab_load when set */
    const ds4_host_bind_map *bind; /* optional; skip C name walk when set */
    /* DeepSeek-only sibling support models.  Paths open through the same
     * native model_open; the optional maps are host-resolved name->index
     * tables for THAT sibling's tensor dir, and when installed native
     * skips that sibling's C layout check. */
    const char *mtp_path;                 /* optional; borrowed */
    const char *dspark_path;              /* optional; borrowed */
    const ds4_host_bind_map *mtp_bind;    /* optional; borrowed */
    const ds4_host_bind_map *dspark_bind; /* optional; borrowed */
    int32_t mtp_draft_tokens;             /* 0 => native default 1 */
    float mtp_margin;
    const char *directional_steering_file; /* optional; borrowed */
    float directional_steering_attn;
    float directional_steering_ffn;
} ds4_bridge_model_open_options;

typedef struct {
    const uint8_t *data;      /* encoded PNG/JPEG; borrowed for the call */
    size_t data_len;
    uint32_t token_offset;
} ds4_bridge_vision_input;

typedef struct {
    uint32_t source_width;
    uint32_t source_height;
    uint32_t content_width;
    uint32_t content_height;
    uint32_t padded_width;
    uint32_t padded_height;
    uint32_t grid_height;
    uint32_t grid_width;
    uint32_t token_count;
} ds4_bridge_vision_info;

/* All functions: 0 on success, nonzero on failure.  err is optional; when
 * provided it is NUL-terminated on failure.  Token pointers are borrowed
 * for the duration of the call only. */

int ds4_bridge_bind_plan_check(const ds4_bridge_bind_plan *plan,
                               char *err, size_t errlen);
int ds4_bridge_bind_plan_match(const ds4_bridge_bind_plan *host,
                               const ds4_bridge_bind_plan *native,
                               char *err, size_t errlen);

int ds4_bridge_model_open(ds4_bridge_model **out,
                          const ds4_bridge_model_open_options *opt,
                          char *err, size_t errlen);
int ds4_bridge_model_open_distributed(
        ds4_bridge_model **out,
        const ds4_bridge_model_open_options *opt,
        const ds4_bridge_distributed_options *distributed,
        char *err, size_t errlen);
/* Worker mode stays on the proven C DS4D runtime for this first strangler
 * slice.  Rust owns parsing/lifecycle; wire bytes, reconnect, telemetry,
 * prefetch, forwarding, and snapshot behavior remain unchanged. */
int ds4_bridge_model_run_distributed_worker(ds4_bridge_model *m,
                                            int32_t ctx_size,
                                            char *err, size_t errlen);
void ds4_bridge_model_boot_prewarm(ds4_bridge_model *m);
void ds4_bridge_model_free(ds4_bridge_model *m);

int ds4_bridge_session_create(ds4_bridge_session **out,
                              ds4_bridge_model *m,
                              int ctx_size,
                              char *err, size_t errlen);
void ds4_bridge_session_free(ds4_bridge_session *s);

int ds4_bridge_session_sync(ds4_bridge_session *s,
                            const int32_t *tokens, int n_tokens,
                            char *err, size_t errlen);
int ds4_bridge_model_vision_probe(ds4_bridge_model *m,
                                  const uint8_t *data, size_t data_len,
                                  ds4_bridge_vision_info *info,
                                  char *err, size_t errlen);
int ds4_bridge_session_sync_vision(ds4_bridge_session *s,
                                   const int32_t *tokens, int n_tokens,
                                   const ds4_bridge_vision_input *images,
                                   uint32_t image_count,
                                   char *err, size_t errlen);
int ds4_bridge_session_sync_cb(ds4_bridge_session *s,
                               const int32_t *tokens, int n_tokens,
                               ds4_bridge_prefill_fn progress, void *ud,
                               char *err, size_t errlen);
int ds4_bridge_eval(ds4_bridge_session *s, int32_t token,
                    char *err, size_t errlen);
int ds4_bridge_eval_speculative_argmax(ds4_bridge_session *s,
                                       int32_t first_token,
                                       int32_t max_tokens,
                                       int32_t eos_token,
                                       int32_t *accepted,
                                       int32_t accepted_cap,
                                       char *err, size_t errlen);
int ds4_bridge_session_eval_layer_slice(ds4_bridge_session *s,
                                        const int32_t *tokens,
                                        uint32_t n_tokens,
                                        uint32_t pos0,
                                        uint32_t layer_start,
                                        uint32_t layer_end,
                                        const float *input_hc,
                                        float *output_hc,
                                        int32_t output_logits,
                                        float *logits,
                                        char *err, size_t errlen);
int ds4_bridge_session_layer_slice_reset(ds4_bridge_session *s,
                                         char *err, size_t errlen);
int ds4_bridge_session_argmax(ds4_bridge_session *s);
int ds4_bridge_session_argmax_excluding(ds4_bridge_session *s,
                                        int32_t excluded_id);
int ds4_bridge_session_pos(ds4_bridge_session *s);
int ds4_bridge_session_ctx(ds4_bridge_session *s);
/* 1 while the S6 lazy graph alloc is still deferred; a pending session can be
 * re-created at a different ctx for free (ds4_session_graph_pending). */
int ds4_bridge_session_graph_pending(ds4_bridge_session *s);
int ds4_bridge_session_power(ds4_bridge_session *s);
int ds4_bridge_session_set_power(ds4_bridge_session *s, int power_percent);
void ds4_bridge_session_rewind(ds4_bridge_session *s, int pos);
void ds4_bridge_session_invalidate(ds4_bridge_session *s);
uint64_t ds4_bridge_session_generation(ds4_bridge_session *s);
int ds4_bridge_session_prefill_cap(ds4_bridge_session *s);
int ds4_bridge_session_exaone_rewind_span(ds4_bridge_session *s);
/* 1 ready, 0 still incomplete, -1 error (same as ds4.h). */
int ds4_bridge_session_distributed_route_ready(ds4_bridge_session *s,
                                               char *err, size_t errlen);
int ds4_bridge_session_sample(ds4_bridge_session *s,
                              float temperature, int top_k, float top_p, float min_p,
                              uint64_t *rng);
int ds4_bridge_session_save_payload(ds4_bridge_session *s, const char *path,
                                    char *err, size_t errlen);
int ds4_bridge_session_load_payload(ds4_bridge_session *s, const char *path,
                                    char *err, size_t errlen);
int ds4_bridge_session_load_payload_range(ds4_bridge_session *s, const char *path,
                                          uint64_t offset, uint64_t length,
                                          char *err, size_t errlen);
int ds4_bridge_session_save_layer_payload(ds4_bridge_session *s, const char *path,
                                          uint32_t layer_start, uint32_t layer_end,
                                          char *err, size_t errlen);
int ds4_bridge_session_load_layer_payload(ds4_bridge_session *s, const char *path,
                                          uint64_t payload_bytes,
                                          const int32_t *tokens, uint32_t n_tokens,
                                          uint32_t layer_start, uint32_t layer_end,
                                          char *err, size_t errlen);

int ds4_bridge_snapshot_create(ds4_bridge_snapshot **out,
                               char *err, size_t errlen);
void ds4_bridge_snapshot_free(ds4_bridge_snapshot *snap);
uint64_t ds4_bridge_snapshot_len(const ds4_bridge_snapshot *snap);
int ds4_bridge_session_save_snapshot(ds4_bridge_session *s,
                                     ds4_bridge_snapshot *snap,
                                     char *err, size_t errlen);
int ds4_bridge_session_load_snapshot(ds4_bridge_session *s,
                                     const ds4_bridge_snapshot *snap,
                                     char *err, size_t errlen);

/* Caller-owned token / text buffers.  n_out is always written on a
 * successful length discovery, including the "buffer too small" error. */
int ds4_bridge_tokenize_text(ds4_bridge_model *m, const char *text,
                             int32_t *out, int cap, int *n_out,
                             char *err, size_t errlen);
int ds4_bridge_tokenize_rendered_chat(ds4_bridge_model *m, const char *text,
                                      int32_t *out, int cap, int *n_out,
                                      char *err, size_t errlen);
int ds4_bridge_token_text(ds4_bridge_model *m, int32_t token,
                          char *out, size_t cap, size_t *n_out,
                          char *err, size_t errlen);
int ds4_bridge_token_eos(ds4_bridge_model *m);
int ds4_bridge_token_is_stop(ds4_bridge_model *m, int32_t token);
int ds4_bridge_model_id(ds4_bridge_model *m);
int ds4_bridge_model_routed_quant_bits(ds4_bridge_model *m);

/* Thin wrap of ds4_engine_session_graph_fit_quote. Fields match the
 * native quote; this ABI is copied, not the ds4.h layout. */
typedef struct ds4_bridge_graph_fit_quote {
    int32_t fits;
    int32_t fail_open;
    uint64_t need_bytes;
    uint64_t headroom_bytes;
    uint64_t avail_bytes;
    uint64_t deficit_bytes;
} ds4_bridge_graph_fit_quote;

int ds4_bridge_session_graph_fit_quote(ds4_bridge_model *m, int ctx_size,
                                       ds4_bridge_graph_fit_quote *q);

/* CLI chat-template encode (ds4_encode_chat_prompt): system may be NULL,
 * think_mode is ds4_think_mode 0..3.  Same buffer contract as tokenize. */
int ds4_bridge_encode_chat_prompt(ds4_bridge_model *m, const char *system,
                                  const char *prompt, int think_mode,
                                  int32_t *out, int cap, int *n_out,
                                  char *err, size_t errlen);

/* Post-prefill distribution head (proof harness --dump-logprobs).
 * Copies up to k entries; returns the count, -1 on a NULL session. */
typedef struct {
    int32_t id;
    float logit;
    float logprob;
} ds4_bridge_token_score;

int ds4_bridge_session_top_logprobs(ds4_bridge_session *s,
                                    ds4_bridge_token_score *out, int k);
int ds4_bridge_session_copy_logits(ds4_bridge_session *s, float *out, int cap);
int ds4_bridge_session_output_head_bench(ds4_bridge_session *s,
                                         int iters, const char *path,
                                         char *err, size_t errlen);

/* Live CUDA memgov census.  Process-global after backend init; no model
 * handle.  Counts match ds4_mem_census.h (DS4_MEMC__COUNT x DS4_MEMD__COUNT).
 * supported=0 means the backend keeps no census (Metal/CPU/stubs): porcelain
 * renders ABSENCE, never a zero family. */
#define DS4_BRIDGE_MEMC_COUNT 17
#define DS4_BRIDGE_MEMD_COUNT 2

typedef struct {
    uint64_t requested;
    uint64_t committed;
    uint64_t freed_requested;
    uint64_t freed_committed;
    uint64_t alloc_calls;
    uint64_t free_calls;
} ds4_bridge_mem_cell;

typedef struct {
    int32_t supported;          /* 1 = coherent CUDA census image */
    uint64_t faults;
    uint64_t epoch;
    uint64_t torn_fallbacks;
    ds4_bridge_mem_cell cells[DS4_BRIDGE_MEMC_COUNT][DS4_BRIDGE_MEMD_COUNT];
} ds4_bridge_mem_census;

typedef struct {
    int32_t status;             /* 0 ok, 1 unsupported, 2 query_error */
    int32_t source;             /* 0 none, 1 cuda_free, 2 meminfo_available */
    uint64_t free_bytes;
    uint64_t total_bytes;
    uint64_t cuda_free_bytes;
    uint64_t meminfo_avail_bytes;
} ds4_bridge_mem_observe;

int ds4_bridge_mem_census_snap(ds4_bridge_mem_census *out);
int ds4_bridge_mem_observe_snap(ds4_bridge_mem_observe *out);
uint64_t ds4_bridge_mem_substrate_outstanding(void);

typedef struct {
    const uint8_t *data;
    size_t data_len;
    uint32_t token_offset;
    uint32_t grid_h;
    uint32_t grid_w;
} ds4_bridge_qwen_image_input;

typedef struct {
    uint32_t source_width;
    uint32_t source_height;
    uint32_t resized_width;
    uint32_t resized_height;
    uint32_t grid_h;
    uint32_t grid_w;
    uint32_t token_count;
} ds4_bridge_qwen_image_info;

int ds4_bridge_qwen_image_probe(const uint8_t *data, size_t data_len,
                                ds4_bridge_qwen_image_info *info,
                                char *err, size_t errlen);
int ds4_bridge_qwen_image_pixel_hash(const uint8_t *data, size_t data_len,
                                     uint64_t *hash,
                                     char *err, size_t errlen);

/* Continuous batching (mid-flight admit/evict) over a persistent batch
 * context.  Mirrors ds4_batch_ctx / ds4_engine_continuous_generate with
 * a narrow request struct; the engine's rolling scheduler stays native.
 * All callbacks run on the calling thread.  `user` is an opaque
 * per-request handle echoed back verbatim; `ud` is the caller context
 * given to ds4_bridge_continuous_generate. */
typedef struct ds4_bridge_batch_ctx ds4_bridge_batch_ctx;

int ds4_bridge_batch_ctx_create_fit(ds4_bridge_model *m, int ctx_size,
                                    int max_seq, int max_total_tokens,
                                    ds4_bridge_batch_ctx **out,
                                    char *err, size_t errlen);
void ds4_bridge_batch_ctx_destroy(ds4_bridge_batch_ctx *c);
int ds4_bridge_batch_ctx_max_seq(ds4_bridge_batch_ctx *c);
int ds4_bridge_batch_ctx_raw_cap(ds4_bridge_batch_ctx *c);
int ds4_bridge_batch_ctx_seq_cap(ds4_bridge_batch_ctx *c);
int ds4_bridge_batch_ctx_supports_partial_reuse(ds4_bridge_batch_ctx *c);
uint64_t ds4_bridge_batch_ctx_trim_free(ds4_bridge_batch_ctx *c, uint64_t want_bytes);

/* Static greedy batch over the persistent native context.  Prompts and output
 * arrays are caller-owned.  Successful token streams are copied contiguously,
 * in request order, into out_tokens; out_lengths partitions that buffer and
 * out_finish uses 1=EOS, 0=budget.  Native ds4_batch_gen_result allocations are
 * always released inside the bridge before return. */
int ds4_bridge_batch_ctx_generate_static(
        ds4_bridge_batch_ctx *c,
        const int32_t *const *prompt_tokens,
        const int32_t *prompt_lengths,
        const int32_t *max_new_tokens,
        const int32_t *eos_ids,
        int32_t n,
        int32_t *out_tokens,
        int32_t out_tokens_cap,
        int32_t *out_lengths,
        int32_t *out_finish,
        char *err, size_t errlen);

/* Opaque durable-bank seam.  Host metadata stays outside the native batch;
 * committed tokens are copied into caller storage and payload bytes move only
 * through files owned by the caller. */
int ds4_bridge_batch_ctx_bank_snapshot(ds4_bridge_batch_ctx *c, int32_t bank,
                                       int32_t *tokens, int32_t cap,
                                       int32_t *n_tokens,
                                       uint64_t *generation,
                                       char *err, size_t errlen);
int ds4_bridge_batch_ctx_bank_save_payload(ds4_bridge_batch_ctx *c,
                                           int32_t bank, const char *path,
                                           char *err, size_t errlen);
int ds4_bridge_batch_ctx_bank_load_payload_range(ds4_bridge_batch_ctx *c,
                                                 int32_t bank,
                                                 const char *path,
                                                 uint64_t offset,
                                                 uint64_t length,
                                                 char *err, size_t errlen);

typedef struct {
    const int32_t *tokens;  /* caller-owned; keep alive until on_done */
    int32_t n;
    ds4_bridge_qwen_image_input images[4];
    uint32_t image_count;
    int32_t max_new;
    int32_t eos;            /* < 0 => engine default */
    void *user;
    float temperature;      /* <= 0 => greedy argmax */
    int32_t top_k;
    float top_p;
    float min_p;
    uint64_t seed;
    /* Optional (NULL disables).  Same contracts as ds4_cont_request:
     * sample_override returns DS4_SAMPLE_OVERRIDE_* encoding; alive
     * returns 0 to abandon a pending admission; on_admitted returns 0
     * to cancel before prefill (n_cached + n_computed == n). */
    int (*sample_override)(void *ud, void *user);
    int (*alive)(void *ud, void *user);
    int (*on_admitted)(void *ud, void *user, int n_cached, int n_computed,
                       int bank);
    int32_t place_bank;     /* bank id + 1; 0 = engine's choice */
    int32_t n_cached;       /* committed prefix length; 0 = cold */
    int32_t *bank_used;     /* OUT (optional): placed bank id */
    int32_t fork_bank;      /* source bank id + 1; 0 = no fork */
} ds4_bridge_cont_request;

typedef struct {
    double decode_ms;
    uint32_t decode_tokens;
    uint32_t decode_steps;
} ds4_bridge_cont_stats;

int ds4_bridge_continuous_generate(
    ds4_bridge_batch_ctx *c,
    int (*admit)(void *ud, ds4_bridge_cont_request *req),
    int (*on_token)(void *ud, void *user, int32_t token),
    void (*on_done)(void *ud, void *user, const int32_t *tokens, int32_t n,
                    int32_t finish, const ds4_bridge_cont_stats *stats),
    void *ud, char *err, size_t errlen);

#ifdef __cplusplus
}
#endif

#endif
