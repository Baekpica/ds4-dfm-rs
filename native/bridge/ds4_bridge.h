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
    int32_t ssd_streaming;
    int32_t ssd_streaming_cold;
    uint32_t ssd_streaming_cache_experts;
    uint64_t ssd_streaming_cache_bytes;
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
    const float *pixels; /* borrowed F32 BTHWC [token_count,2,40,40,3] */
    uint64_t pixel_count;
    uint32_t token_offset;
    uint32_t token_count;
} ds4_bridge_inkling_pixels;

typedef struct {
    const float *pixels; /* borrowed normalized CHW [3,edge,edge] */
    uint64_t pixel_count;
    uint32_t token_offset;
    uint32_t token_count;
    uint32_t edge;
} ds4_bridge_step37_pixels;

typedef struct {
    const int32_t *codes; /* borrowed [token_count,80], each in [0,15] */
    uint64_t code_count;
    uint32_t token_offset;
    uint32_t token_count;
} ds4_bridge_inkling_audio;

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
/* Successful drafter IPC import; absence/fallback returns zero. */
int ds4_bridge_drafter_shared(ds4_bridge_model *m);

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
int ds4_bridge_sync_step37(ds4_bridge_session *s,
                            const int32_t *tokens, int n_tokens,
                            const ds4_bridge_step37_pixels *crops, uint32_t crop_count,
                            char *err, size_t errlen);
int ds4_bridge_sync_inkling(ds4_bridge_session *s,
                             const int32_t *tokens, int n_tokens,
                             const ds4_bridge_inkling_pixels *images, uint32_t image_count,
                             const ds4_bridge_inkling_audio *audios, uint32_t audio_count,
                             char *err, size_t errlen);
int ds4_bridge_session_sync_cb(ds4_bridge_session *s,
                               const int32_t *tokens, int n_tokens,
                               ds4_bridge_prefill_fn progress, void *ud,
                               char *err, size_t errlen);
int ds4_bridge_eval(ds4_bridge_session *s, int32_t token,
                    char *err, size_t errlen);
int ds4_bridge_inkling_trial(ds4_bridge_session *s, int32_t first, int32_t max_tokens,
                              int32_t *tokens, int32_t *target, int32_t cap,
                              char *err, size_t errlen);
int ds4_bridge_inkling_commit(ds4_bridge_session *s, int32_t keep, char *err, size_t errlen);
int ds4_bridge_glm53_trial(ds4_bridge_session *s, int32_t first, int32_t max_tokens,
                           int32_t *tokens, int32_t *target, int32_t cap,
                           char *err, size_t errlen);
int ds4_bridge_glm53_commit(ds4_bridge_session *s, int32_t keep, char *err, size_t errlen);
int ds4_bridge_ling31_trial(ds4_bridge_session *s, int32_t first, int32_t max_tokens,
                            int32_t *tokens, int32_t *target, int32_t cap,
                            char *err, size_t errlen);
int ds4_bridge_ling31_commit(ds4_bridge_session *s, int32_t keep, char *err, size_t errlen);
int ds4_bridge_step37_trial(ds4_bridge_session *s, int32_t first, int32_t max_tokens,
                              int32_t *tokens, int32_t *target, int32_t cap,
                              char *err, size_t errlen);
int ds4_bridge_step37_commit(ds4_bridge_session *s, int32_t keep, char *err, size_t errlen);
int ds4_bridge_naive_trial(ds4_bridge_session *s, int32_t first, int32_t max_tokens,
                            int32_t *tokens, int32_t *target, int32_t cap, char *err, size_t errlen);
int ds4_bridge_naive_commit(ds4_bridge_session *s, int32_t keep, char *err, size_t errlen);
int ds4_bridge_beam_trial(ds4_bridge_session *s, int32_t first, int32_t max_tokens,
                          int32_t *tokens, int32_t *target, int32_t cap, char *err, size_t errlen);
int ds4_bridge_beam_commit(ds4_bridge_session *s, int32_t keep, char *err, size_t errlen);
int ds4_bridge_minimax_trial(ds4_bridge_session *s, int32_t first, int32_t max_tokens,
                             int32_t *tokens, int32_t *target, int32_t cap, char *err, size_t errlen);
int ds4_bridge_minimax_commit(ds4_bridge_session *s, int32_t keep, char *err, size_t errlen);
int ds4_bridge_iquest_trial(ds4_bridge_session *s, int32_t first, int32_t max_tokens,
                           int32_t *tokens, int32_t *target, int32_t cap, char *err, size_t errlen);
int ds4_bridge_iquest_commit(ds4_bridge_session *s, int32_t keep, char *err, size_t errlen);
int ds4_bridge_dots3_enabled(ds4_bridge_session *s);
int ds4_bridge_dots3_trial(ds4_bridge_session *s, int32_t first, int32_t max_tokens,
                              int32_t *tokens, int32_t *target, int32_t cap,
                              char *err, size_t errlen);
int ds4_bridge_dots3_commit(ds4_bridge_session *s, int32_t keep, char *err, size_t errlen);
int ds4_bridge_mimo2_trial(ds4_bridge_session *s, int32_t first, int32_t max_tokens,
                           int32_t *tokens, int32_t *target, int32_t cap,
                           char *err, size_t errlen);
int ds4_bridge_mimo2_commit(ds4_bridge_session *s, int32_t keep, char *err, size_t errlen);
int ds4_bridge_mimo2_sync_media(ds4_bridge_session *s, const int32_t *tokens, int n_tokens,
                                const uint32_t *starts, const uint32_t *counts,
                                const float *const *rows, uint32_t nspans, uint64_t tag,
                                char *err, size_t errlen);
int ds4_bridge_mimo2_encode_vision(ds4_bridge_session *s, const float *patches,
                                   uint32_t n_patches, uint32_t grid_h, uint32_t grid_w,
                                   float *out, char *err, size_t errlen);
int ds4_bridge_mimo2_encode_audio(ds4_bridge_session *s, const float *mel, uint32_t frames,
                                  float *out, uint32_t cap, uint32_t *rows,
                                  char *err, size_t errlen);
int ds4_bridge_minimax_encode_vision(ds4_bridge_session *s, const float *patches, uint32_t grid_t,
                                     uint32_t grid_h, uint32_t grid_w, float *out, char *err, size_t errlen);
int ds4_bridge_minimax_sync_media(ds4_bridge_session *s, const int32_t *tokens, int n_tokens,
                                  const uint32_t *starts, const uint32_t *counts,
                                  const float *const *rows, uint32_t nspans, uint64_t tag,
                                  char *err, size_t errlen);
int ds4_bridge_eval_speculative_argmax(ds4_bridge_session *s,
                                       int32_t first_token,
                                       int32_t max_tokens,
                                       int32_t eos_token,
                                       int32_t *accepted,
                                       int32_t accepted_cap,
                                       char *err, size_t errlen);

/* ---- V4.1 (ds41) run surface (P5) -----------------------------------------
 * The engine's single-request server path IS the one-shot generate with two
 * callbacks (server_generate_v41.c:412-414 sets the prefill progress hook and
 * calls ds4_engine_v41_generate_argmax with v41_emit); this is that same
 * surface across the bridge, so it is the family's decode entry, not a new
 * category.  The Rust host hashes and preads the engram rows
 * (crates/ds4-core/src/engram.rs); the native owns the pinned buffers and
 * hands the rows provider the exact block the forward is about to run.  All
 * callbacks run on the calling thread; emit returning nonzero stops the
 * generate (the CLI's EOS convention) and progress returning nonzero aborts
 * the prefill.  The switches mirror the engine's CLI globals (cli_diag.c:58):
 * dspark mode 0 off / 1 default on / 2 explicit --dspark. */
typedef int (*ds4_bridge_v41_rows_fn)(uint32_t pos0, const int32_t *tokens, int n,
                                      uint8_t *const *dst, void *ud);
/* P2: the per-layer collect hook (the engine's v41_ejob_wait call site,
 * core_v41_engram.c:441) — the native forward calls it at each engram layer
 * and only then uploads the layer's rows; NULL keeps the synchronous
 * protocol (prepare filled everything, the graph records no flag wait). */
typedef int (*ds4_bridge_v41_wait_fn)(int k, void *ud);
/* The cross-block prefetch hook: the chunk loop declares the next prefill
 * block before running the current one (ds41_forward.h's
 * ds41_engram_feed.prefetch_next); nonzero refuses the run. */
typedef int (*ds4_bridge_v41_next_fn)(uint32_t pos0, const int32_t *tokens, int n, void *ud);
/* Required with rows/wait/next. Drain provider writes before row buffers are
 * freed, including on an aborted forward. Nonzero reports a completed failed
 * drain; no write may remain in flight. Layer waits do not replace it. */
typedef int (*ds4_bridge_v41_drain_fn)(void *ud);
typedef int (*ds4_bridge_v41_emit_fn)(int32_t token, void *ud);
typedef int (*ds4_bridge_v41_progress_fn)(void *ud, const char *event, int32_t current, int32_t total);
void ds4_bridge_v41_set_dspark(int mode);
void ds4_bridge_v41_set_graph(int on);
void ds4_bridge_v41_set_emit_trace(int on);
void ds4_bridge_v41_set_prof(int on);
/* The V4.1 KV prefix reuse (U-B, core_v41_kvreuse.c): the --no-kv-reuse
 * switch, the last run's restored token count (the server's usage
 * cached_tokens), the read-only peek that seeds the host-side engram history
 * before the run, and the mounted-amplifier identity the native drops the
 * checkpoint on (the engine's switch_plugins stand-in). */
void ds4_bridge_v41_kvreuse_set(int on);
int ds4_bridge_v41_kvreuse_last_reused(void);
uint32_t ds4_bridge_v41_kvreuse_peek(const int32_t *prompt, int n, const uint64_t *sig);
void ds4_bridge_v41_kvreuse_note_mount(int has_mount, uint64_t hash);
/* T's key (the prompt it was saved with, then the ids the state committed),
 * which is the host store's turn entry text: answers the count, filling
 * `out` when it covers `cap`.  0 = no turn checkpoint. */
uint32_t ds4_bridge_v41_kvreuse_turn_key(int32_t *out, uint32_t cap);
/* The disk half (item 4): the payload export/import the Rust store drives
 * (crates/ds4-kv, v41_disk.rs).  export writes the named class's checkpoint
 * payload (meta + spans; the store's text block carries the tokens) to a
 * host-named path, 1 = written; import loads it back from (path, offset,
 * bytes) plus the token block the caller matched, 1 = loaded, into the slot
 * the payload's meta names.  cls is the class to write: 0 = the prompt-end
 * checkpoint K, 1 = the run-end turn checkpoint T (the native v41_kv_class,
 * ds4_ds41_kvreuse.inc; the Rust side names the same two in ds4-sys).  Both
 * are gated by --no-kv-reuse and the class's slot being valid. */
int ds4_bridge_v41_kvreuse_export(const char *path, int cls);
int ds4_bridge_v41_kvreuse_import(const char *path, uint64_t offset, uint64_t bytes,
                                  const int32_t *tokens, uint32_t n);
/* The last v41 run's speculation account (the engine's
 * ds4_engine_v41_last_spec_stats, core_v41_api.c:41-46): rounds/offered/
 * accepted, cleared by the next generate entry.  Any pointer may be NULL. */
void ds4_bridge_v41_last_spec_stats(int *rounds, int *offered, int *accepted);
/* The per-request decode-sampling face (the engine's
 * ds4_engine_set_decode_sampling, core_v41_api.c:10-13; the struct and its
 * semantics at ds41_forward.h).  All-zero = bare argmax; NULL resets to that.
 * The engine's server sets this once per request before the generate
 * (server_generate_v41.c:409); the Rust host does the same through this
 * bridge.  The struct is laid out like ds4_decode_sampling, spelled out here
 * so the header stays self-contained for the Rust side.  greedy_fn/greedy_ud:
 * the engine's tool-syntax greedy hook (ds41_forward.h; the Rust host answers
 * through the ds4-server accumulator). */
typedef int (*ds4_bridge_v41_greedy_fn)(void *ud, const int32_t *ahead, uint32_t n);
typedef struct {
    float temperature, top_p, min_p; int32_t top_k; uint64_t seed;
    float freq_penalty, presence_penalty;
    float dry_multiplier, dry_base; int32_t dry_allowed_length;
    ds4_bridge_v41_greedy_fn greedy_fn; void *greedy_ud;
} ds4_bridge_v41_sampling;
void ds4_bridge_v41_set_sampling(const ds4_bridge_v41_sampling *sp);
/* --mem-budget-mb (unit Z): the engine's process memory red line and its
 * L1 pre-flight load gate (core_util.c:198-256); call before opening the
 * model.  mb <= 0 leaves the budget unset (no-op gate). */
void ds4_bridge_set_mem_budget_mb(int mb);
/* The V4.1 zchain sidecar mount (unit Z).  The engine mounts ② / ③ per
 * state (v41_state_plugins, core_v41_state.c:77-87 -> v41_amp_load,
 * core_v41_amp.c:227-270); the Rust reader (crates/ds4-core/src/zchain.rs,
 * P2/P3) reads and merges the directories, and these three entries are the
 * device side it drives: the layer's merged gain table, the layer's route
 * bias, and the process-wide rb unload the engine performs before any rb
 * mount (core_v41_amp.c:114).  All return 1 on success, 0 on refusal. */
int ds4_bridge_v41_zchain_gr_set(ds4_bridge_model *m, uint32_t il, const float *data, uint32_t n_expert, uint32_t n_embd);
int ds4_bridge_v41_zchain_rb_set(ds4_bridge_model *m, uint32_t il, const float *delta, uint32_t n_expert);
void ds4_bridge_v41_zchain_rb_clear(void);
/* The request's images (the engine's ds4_v41_images, core_v41.h:409-417).  The
 * Rust host owns preprocessing (crates/ds4-core/src/v41_media.rs) and the
 * prompt expansion: patches is the f32 [n_vit_h*n_vit_w][3*patch*patch] on the
 * bf16 grid the tower consumes, row_of maps the expanded prompt's positions to
 * embed rows (-1 = text), and n_rows is the sum of the span lengths.  Every
 * pointer is borrowed for the duration of the call only; NULL = a text request
 * (no tower, byte-identical to before). */
typedef struct {
    const float *patches;
    uint64_t patches_len; /* f32 elements */
    uint32_t n_vit_h, n_vit_w, n_llm_h, n_llm_w;
} ds4_bridge_v41_image;
typedef struct {
    const ds4_bridge_v41_image *img;
    int n;
    const int32_t *row_of;
    const uint64_t *sig_of;
    uint32_t np, n_rows;
} ds4_bridge_v41_images;
/* The serving admission's byte account (server_sched_v41.c:58): the prefill
 * peak for a prompt, plus the image set's own peak.  Both are engine-side
 * formulas (ds4_v41_req_prefill_bytes / ds4_v41_images_bytes); the caller
 * compares 2x their sum against MemAvailable. */
uint64_t ds4_bridge_v41_prefill_bytes(ds4_bridge_model *m, int n_prompt);
uint64_t ds4_bridge_v41_images_bytes(const ds4_bridge_v41_images *images);
/* The golden gate's tower entry (the engine's v41_vision_run,
 * core_v41_vision.c:116-131, called by core_v41_vision_gate.c:80):
 * has_vision is the gate's upfront check (:45); vision_run returns malloc'd
 * aligner rows [n_llm_h*n_llm_w][DS4_N_EMBD] for one patch set (NULL when the
 * model carries no tower, the patch extent is invalid, or the forward fails),
 * freed with vision_rows_free. patches_len counts f32 elements. */
int ds4_bridge_v41_has_vision(ds4_bridge_model *m);
float *ds4_bridge_v41_vision_run(ds4_bridge_model *m, const float *patches,
                                 uint64_t patches_len,
                                 uint32_t n_vit_h, uint32_t n_vit_w,
                                 uint32_t *n_llm_h, uint32_t *n_llm_w);
void ds4_bridge_v41_vision_rows_free(float *rows);
int ds4_bridge_v41_generate(ds4_bridge_model *m,
                            const int32_t *prompt, int n_prompt, int n_predict,
                            int no_engram, int verify_k,
                            const ds4_bridge_v41_images *images,
                            ds4_bridge_v41_rows_fn rows, void *rows_ud,
                            ds4_bridge_v41_wait_fn wait,
                            ds4_bridge_v41_next_fn next,
                            ds4_bridge_v41_drain_fn drain,
                            ds4_bridge_v41_emit_fn emit, void *emit_ud,
                            ds4_bridge_v41_progress_fn progress, void *progress_ud,
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
/* The host's draw of one verify row from its sampler and stream. */
typedef int (*ds4_bridge_spec_draw_fn)(void *ud, const float *row, uint32_t vocab,
                                       uint64_t *rng);
/* Speculative trials draw verify rows with `draw` until cleared (NULL). */
int ds4_bridge_session_set_spec_sampler(ds4_bridge_session *s, ds4_bridge_spec_draw_fn draw,
                                        void *ud, uint64_t *rng);
int ds4_bridge_session_sample_excluding(ds4_bridge_session *s,
                                        float temperature, int top_k,
                                        float top_p, float min_p,
                                        uint64_t *rng, int excluded_id);
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
/* Bytes a snapshot of the session's checkpoint would take (0: none). */
uint64_t ds4_bridge_session_payload_bytes(ds4_bridge_session *s);
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

/* Process-global speculative counters. Each field is read atomically;
 * the caller owns out, and NULL is a no-op. No model handle is needed. */
typedef struct {
    uint64_t drafts;
    uint64_t hits;
    uint64_t quench;
} ds4_bridge_spec_metrics;

void ds4_bridge_spec_snapshot(ds4_bridge_spec_metrics *out);

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
    int (*sample_exclude)(void *ud, void *user);
    /* Call-scoped Step trial buffers; return an accepted prefix in [1,n].
     * Rust owns this pure policy. NULL disables bank speculation. */
    int (*step_accept)(const int32_t *tokens, const int32_t *target, int n, int eos);
    int (*alive)(void *ud, void *user);
    int (*on_admitted)(void *ud, void *user, int n_cached, int n_computed,
                       int bank);
    int32_t place_bank;     /* bank id + 1; 0 = engine's choice */
    int32_t n_cached;       /* committed prefix length; 0 = cold */
    int32_t *bank_used;     /* OUT (optional): placed bank id */
    int32_t fork_bank;      /* source bank id + 1; 0 = no fork */
    int32_t checkpoint_at;  /* canonical history prefill boundary; 0 disables */
    void (*on_checkpoint)(void *ud, void *user, int bank, int current);
} ds4_bridge_cont_request;

typedef struct {
    double decode_ms;
    uint32_t decode_tokens;
    uint32_t decode_steps;
    /* Draft rows this sequence ran.  0 means ordinary decode: MTP weights
     * being resident is not "this request speculated". */
    uint64_t spec_drafts;
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
