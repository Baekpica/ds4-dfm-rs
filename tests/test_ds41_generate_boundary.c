/* The production generate/forward/feed functions run unchanged. Bounded host
 * tensors replace CUDA, and a deterministic rejection keeps every frontier
 * reachable without numerical inference or a model fixture. */
#define _POSIX_C_SOURCE 200809L
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include "../ds4_gpu.h"
#include "../ds41_forward.h"

#define DS4_VARIANT_V41 41u
#define DS4_MODEL_VARIANT DS4_VARIANT_V41
#define DS4_V41_CHUNK 2048u
#define DS4_MTP_MAX_BLOCK 8u
#define DS4_V41_DRAFT_GROWS 8u
#define DS4_MAX_LAYER 2u
#define DS4_MTP_MAX_TOWERS 2u
#define DS4_V41_MAX_ENGRAM DS41_MAX_ENGRAM
#define DS4_N_HC 4u
#define DS4_N_SWA 8u
#define DS4_N_VOCAB 4u
#define DS4_N_LAYER 2u
#define DS4_N_HEAD_DIM 8u
#define DS4_N_EMBD 8u
#define DS4_V41_CKV_BYTES 32u
#define DS4_V41_IDXK_BYTES 32u

typedef struct { int unused; } ds4_tensor;
typedef struct {
    struct {
        struct { const ds4_tensor *main_proj, *markov_embd, *confidence; } v41_towers;
    } weights;
} ds4_engine;
struct ds4_gpu_tensor { uint64_t bytes; unsigned char *data; };
#include "ds41_boundary_types.inc"

static struct {
    uint32_t ctx, n_engram, engram_max_ngram, engram_heads, engram_head_dim;
    uint32_t mtp_towers, mtp_block, n_mtp_target;
    uint8_t is_kv_source[DS4_MAX_LAYER];
} g_ds4_v41;
static ds4_decode_sampling g_decode_sampling;
int g_ds4_v41_prof, g_ds4_v41_emit_trace;
static int g_ds4_v41_decoder_full;
static int g_v41_last_spec_rounds, g_v41_last_spec_off, g_v41_last_spec_acc;
static int (*g_v41_progress)(void *, const char *, int, int);
static void *g_v41_progress_ud;
static int g_spec, g_pad_state, g_emitted, g_expected, g_body_calls;
static int g_graph, g_graph_pending, g_launches, g_waits;
static uint32_t g_snap_past, g_alloc_cap, g_peak_block;
static uint32_t g_vision_first;
static int g_vision_calls;
typedef struct { int unused; } v41_sched;

static void *xmalloc(size_t bytes) { return malloc(bytes); }
static void *xmalloc_zeroed(size_t n, size_t size) { return calloc(n, size); }
static double now_sec(void) { return (double)clock() / CLOCKS_PER_SEC; }
ds4_gpu_tensor *ds4_gpu_tensor_alloc(uint64_t bytes) {
    ds4_gpu_tensor *t = calloc(1, sizeof *t);
    if (t) { t->bytes = bytes; t->data = calloc(1, (size_t)bytes); }
    return t;
}
void ds4_gpu_tensor_free(ds4_gpu_tensor *t) {
    if (t) { free(t->data); free(t); }
}
int ds4_gpu_tensor_write(ds4_gpu_tensor *t, uint64_t offset, const void *src, uint64_t bytes) {
    if (!t || offset > t->bytes || bytes > t->bytes - offset) { return 0; }
    memcpy(t->data + offset, src, (size_t)bytes);
    return 1;
}
int ds4_gpu_tensor_read(const ds4_gpu_tensor *t, uint64_t offset, void *dst, uint64_t bytes) {
    if (!t || offset > t->bytes || bytes > t->bytes - offset) { return 0; }
    memcpy(dst, t->data + offset, (size_t)bytes);
    return 1;
}
const void *ds4_gpu_tensor_ptr(const ds4_gpu_tensor *t) { return t ? t->data : NULL; }
void *ds4_gpu_host_alloc(uint64_t bytes) { return calloc(1, (size_t)bytes); }
void ds4_gpu_host_free(void *ptr) { free(ptr); }
int ds4_gpu_begin_commands(void) { return 1; }
int ds4_gpu_end_commands(void) { return 1; }
int ds4_gpu_synchronize(void) { return 1; }
int ds4_gpu_tensor_fill_f32(ds4_gpu_tensor *t, float value, uint64_t n) {
    if (!t || n > t->bytes / 4u) { return 0; }
    for (uint64_t i = 0; i < n; i++) { ((float *)t->data)[i] = value; }
    return 1;
}

static void v41_alloc_dense(ds4_v41_state *st, uint32_t cap, uint32_t rows, bool *ok) {
    g_alloc_cap = cap;
    /* Padding only isolates the second, independently undersized feed buffer. */
    st->cap_tok = g_pad_state && cap < 6u ? 6u : cap;
    st->tok = ds4_gpu_tensor_alloc(st->cap_tok * 4u);
    st->pos = ds4_gpu_tensor_alloc(st->cap_tok * 4u);
    st->vrows = ds4_gpu_tensor_alloc(st->cap_tok * 4u);
    st->pre_mix = ds4_gpu_tensor_alloc(st->cap_tok * DS4_N_HC * 4u);
    st->logits = ds4_gpu_tensor_alloc((uint64_t)rows * DS4_N_VOCAB * 4u);
    *ok = st->tok && st->pos && st->pre_mix && st->logits;
}
static void v41_alloc_attn_rows(ds4_v41_state *st, uint32_t cap, bool *ok) {}
static void v41_alloc_attn_cache(ds4_v41_state *st, uint32_t cap, bool *ok) {}
static uint32_t ds4_layer_compress_ratio(uint32_t il) { return 4u; }
static ds4_gpu_tensor *v41_alloc(uint64_t bytes, bool *ok) {
    ds4_gpu_tensor *t = ds4_gpu_tensor_alloc(bytes);
    if (!t) { *ok = false; }
    return t;
}
static void v41_state_free(ds4_v41_state *st) {
    ds4_gpu_tensor_free(st->tok); ds4_gpu_tensor_free(st->pos);
    ds4_gpu_tensor_free(st->vrows);
    ds4_gpu_tensor_free(st->pre_mix); ds4_gpu_tensor_free(st->logits);
    ds4_gpu_tensor_free(st->wintmp); ds4_gpu_tensor_free(st->mainh);
    for (uint32_t il = 0; il < DS4_N_LAYER; il++) { ds4_gpu_tensor_free(st->win[il]); }
}
#include "ds41_boundary_alloc.inc"
static bool v41_forward_body(ds4_engine *e, ds4_v41_state *st) {
    if (g_emitted >= g_expected) { return false; }
    g_body_calls++;
    if (st->n > g_peak_block) { g_peak_block = st->n; }
    for (uint32_t i = 0; i < st->logits->bytes / 4u; i++) {
        ((float *)st->logits->data)[i] = i % DS4_N_VOCAB == 1u ? 1.0f : 0.0f;
    }
    return true;
}
static int v41_dspark_mode(void) { return g_spec; }
/* Keep the real forward's dispatch and row mapping; replace only tower math. */
static bool v41_vision_encode(ds4_engine *e, struct ds4_v41_images *imgs, uint32_t first_row) {
    if (imgs->encoded) { return true; }
    g_vision_first = first_row;
    g_vision_calls++;
    imgs->row_base = first_row;
    imgs->encoded = 1;
    return true;
}
#include "ds41_boundary_feed.inc"
#include "ds41_boundary_forward.inc"

static bool v41_draft_alloc(ds4_engine *e, ds4_v41_draft *dr) {
    memset(dr, 0, sizeof *dr); dr->block = g_ds4_v41.mtp_block; return true;
}
static void v41_draft_free(ds4_v41_draft *dr) {}
static int v41_draft_launch(ds4_engine *e, ds4_v41_state *st, ds4_v41_draft *dr, int32_t tok, uint32_t pos) {
    for (uint32_t i = 1; i <= dr->block; i++) { dr->host_ids[i] = 2; }
    return 2;
}
static bool v41_draft_wait(ds4_v41_draft *dr) { return true; }
static uint32_t v41_draft_pick_k(const float *conf, uint32_t block, float *val, v41_sched *cal) { return block; }
static void v41_sched_cost(v41_sched *cal, uint32_t n, double ms) {}
static void v41_sched_draft_cost(v41_sched *cal, double ms) {}
static void v41_sched_observe(v41_sched *cal, const float *conf, bool accepted) {}
static uint32_t v41_kvreuse_restore(ds4_v41_state *st, const int *ids, uint32_t n, const uint64_t *sig) { return 0; }
static void v41_kvreuse_save(ds4_v41_state *st, const int *ids, uint32_t n, const uint64_t *sig) {}
static void v41_kvreuse_save_turn(ds4_v41_state *st, const int32_t *ids, uint32_t np,
                                  const int32_t *gen, uint32_t ng, const uint64_t *sig) {}
static void v41_turn_ids_add(v41_turn_ids *t, const int32_t *ids, uint32_t n) {}
static void v41_turn_ids_free(v41_turn_ids *t) {}
static bool v41_next_token(ds4_v41_state *st, ds4_gpu_tensor *am, uint32_t row, float *buf, uint64_t *rng, v41_hist *hist, int32_t *tok) { *tok = 1; return true; }
static bool v41_device_next(ds4_v41_state *st, ds4_gpu_tensor *am, uint32_t row, uint32_t n, const int32_t *batch, int32_t *want) {
    for (uint32_t i = 0; i < n; i++) { want[i] = 1; }
    return true;
}
static uint8_t *ds4_decode_breakers(ds4_engine *e, uint32_t n) { return NULL; }
static int ds4_token_eos(ds4_engine *e) { return 3; }
static void v41_gen_dump_row(ds4_v41_state *st, FILE *fd) {}
static bool v41_graph_ready(ds4_v41_state *st) { return g_graph != 0; }
static bool v41_graph_launch(ds4_engine *e, ds4_v41_state *st, int32_t tok, int verify) {
    g_launches++;
    if (g_graph_pending || !v41_forward(e, st, &tok, 1u)) { return false; }
    g_graph_pending = 1;
    return true;
}
static bool v41_graph_wait(ds4_engine *e, ds4_v41_state *st, int32_t *tok) {
    if (!g_graph_pending) { return false; }
    g_graph_pending = 0; g_waits++; *tok = 1;
    return true;
}
static bool v41_graph_batch_ready(ds4_v41_state *st, uint32_t n) { return false; }
static bool v41_graph_batch_launch(ds4_engine *e, ds4_v41_state *st, const int32_t *ids, uint32_t n) { return false; }
static bool v41_graph_batch_wait(ds4_engine *e, ds4_v41_state *st, int32_t *ids) { return false; }
static void v41_graph_batch_host(ds4_v41_state *st, double *p, double *l, uint32_t *n) {}
static void v41_graph_batch_prep(ds4_v41_state *st, double *b, double *e, double *c, double *a) {}
static void v41_graph_capture_cost(ds4_v41_state *st, double *t, uint32_t *n) {}
static bool v41_spec_snapshot(ds4_v41_state *st, uint32_t n) { g_snap_past = st->n_past; return true; }
static bool v41_spec_rollback(ds4_v41_state *st, uint32_t keep) { st->n_past = g_snap_past + keep; return true; }
#include "ds41_boundary_generate.inc"

static int rows(uint32_t pos, const int *ids, int n, uint8_t *const *dst, void *ud) { return 0; }
static int emit(int tok, void *ud) { g_emitted++; return 0; }

static int run_case(uint32_t np, int predict, uint32_t ctx, int spec, int pad) {
    ds4_engine engine = {0};
    const ds4_tensor weight = {0};
    engine.weights.v41_towers.main_proj = &weight;
    engine.weights.v41_towers.markov_embd = &weight;
    engine.weights.v41_towers.confidence = &weight;
    int prompt[6] = {1, 1, 1, 1, 1, 1};
    g_ds4_v41.ctx = ctx; g_ds4_v41.n_engram = 1u;
    g_ds4_v41.engram_max_ngram = 2u; g_ds4_v41.engram_heads = 1u; g_ds4_v41.engram_head_dim = 32u;
    g_ds4_v41.mtp_towers = 1u;
    g_ds4_v41.mtp_block = ctx < 64u ? DS4_MTP_MAX_BLOCK : 5u;
    g_ds4_v41.n_mtp_target = 1u;
    g_spec = spec; g_pad_state = pad; g_emitted = 0; g_peak_block = 0;
    g_alloc_cap = 0; g_body_calls = 0;
    g_graph_pending = 0; g_launches = 0; g_waits = 0;
    const int room = (int)(ctx - np);
    const int expected = predict < room ? predict : room;
    g_expected = expected;
    ds4_v41_feed *feed = ds4_v41_feed_open(&engine, (int)np, rows, NULL, NULL, NULL, NULL, NULL);
    if (!feed) { return 1; }
    const int rc = ds4_v41_generate_argmax(&engine, prompt, (int)np, predict, 0, 5, NULL, &feed->feed, emit, NULL);
    const uint32_t feed_cap = feed->rows_cap;
    ds4_v41_feed_close(feed);
    const int graph_ok = !g_graph || (g_launches == expected - 1 && g_waits == g_launches && !g_graph_pending);
    if (rc != 0 || g_emitted != expected || g_body_calls != expected || !graph_ok) {
        printf("FAIL prompt=%u output=%d ctx=%u mtp=%d pad-state=%d graph=%d: rc=%d emitted=%d/%d state-cap=%u feed-cap=%u forward=%d/%d launch/wait=%d/%d\n",
               np, predict, ctx, spec, pad, g_graph, rc, g_emitted, expected, g_alloc_cap, feed_cap,
               g_body_calls, expected, g_launches, g_waits);
        return 1;
    }
    return 0;
}

static int vision_suffix(void) {
    ds4_engine engine = {0};
    const int32_t ids[] = {1};
    int32_t row_of[] = {-1, 0, 1, -1, 2, 3, -1};
    struct ds4_v41_images images = {.np = 7u, .n_rows = 4u, .row_of = row_of};
    g_ds4_v41.ctx = 16u;
    g_ds4_v41.n_engram = 0;
    g_expected = 16; g_emitted = 0; g_pad_state = 0;
    g_vision_calls = 0;
    ds4_v41_state st;
    bool ok = v41_state_alloc(&st, 1u, 16u, 1u);
    st.vimg = &images;
    st.n_past = 3u;  /* The first image's rows already live in restored KV. */
    unsetenv("DS41_VISION_REENCODE");
    ok = ok && v41_forward(&engine, &st, ids, 1u) && g_vision_calls == 0;
    ok = ok && v41_forward(&engine, &st, ids, 1u) && g_vision_calls == 1 && g_vision_first == 2u;
    ok = ok && ((int32_t *)st.vrows->data)[0] == 0 && st.himg[4] == 1;
    ok = ok && v41_forward(&engine, &st, ids, 1u) && g_vision_calls == 1;
    ok = ok && ((int32_t *)st.vrows->data)[0] == 1 && st.himg[5] == 1;
    ok = ok && v41_forward(&engine, &st, ids, 1u) && g_vision_calls == 1 && st.himg[6] == 0;

    /* The diagnostic control retains the original full-tower work. */
    st.n_past = 3u; images.encoded = 0; g_vision_calls = 0;
    setenv("DS41_VISION_REENCODE", "1", 1);
    ok = ok && v41_forward(&engine, &st, ids, 1u) && g_vision_calls == 1 && g_vision_first == 0u;
    unsetenv("DS41_VISION_REENCODE");
    v41_state_free(&st);
    printf("vision suffix dispatch and row mapping: %s\n", ok ? "PASS" : "FAIL");
    return !ok;
}

int main(void) {
    const uint32_t prompts[] = {1u, 2u, 5u, 6u};
    const int outputs[] = {1, 16};
    int failures = vision_suffix(), cases = 1;
    for (uint32_t p = 0; p < 4u; p++) {
        for (uint32_t n = 0; n < 2u; n++) {
            for (int spec = 0; spec <= 1; spec++) {
                failures += run_case(prompts[p], outputs[n], 64u, spec, 0); cases++;
            }
            failures += run_case(prompts[p], outputs[n], 64u, 1, 1); cases++;
        }
    }
    const uint32_t contexts[] = {8u, 11u};
    for (uint32_t c = 0; c < 2u; c++) {
        for (uint32_t n = 0; n < 2u; n++) {
            failures += run_case(6u, outputs[n], contexts[c], 1, 0); cases++;
        }
    }
    /* The last output needs no next graph launch; earlier launches must drain. */
    g_graph = 1;
    for (uint32_t p = 0; p < 4u; p++) {
        for (uint32_t n = 0; n < 2u; n++) {
            failures += run_case(prompts[p], outputs[n], 64u, 0, 0); cases++;
        }
    }
    printf("ds41 generate boundary gate: %s (%d/%d failures)\n", failures ? "FAIL" : "PASS", failures, cases);
    return failures ? 1 : 0;
}
