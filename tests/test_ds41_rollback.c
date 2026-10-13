/* Actual rollback runs under mock CUDA copies/readbacks. Comparing every
 * restored lane with its checkpoint is independent of cross-width MoE FP. */
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define DS4_N_LAYER 3u
#define DS4_N_HEAD_DIM 8u
#define DS4_N_SWA 16u
#define DS4_MTP_MAX_BLOCK 8u
#define ROW_BYTES (DS4_N_HEAD_DIM * sizeof(float))
#define BATCH_ROWS 6u
#define START_POS 17u
#define MOCK_CTX 32u

typedef struct { uint8_t *data; uint64_t bytes; } ds4_gpu_tensor;
typedef struct {
    uint32_t n_past, ctx, snap_past, snap_on, snap_n;
    ds4_gpu_tensor *win[DS4_N_LAYER], *snap_win[DS4_N_LAYER];
    ds4_gpu_tensor *cpre_kv[DS4_N_LAYER], *cpre_sc[DS4_N_LAYER];
    ds4_gpu_tensor *snap_cpre_kv[DS4_N_LAYER], *snap_cpre_sc[DS4_N_LAYER];
    uint32_t ng_src[DS4_N_LAYER], cpend[DS4_N_LAYER], snap_cpend[DS4_N_LAYER];
    ds4_gpu_tensor *mainh;
    uint32_t mainh_cap, mainh_n;
    int64_t mainh_end;
} ds4_v41_state;

static struct { uint8_t is_kv_source[DS4_N_LAYER]; } g_ds4_v41 = {{1u, 1u, 0u}};
static const uint32_t g_ratio[DS4_N_LAYER] = {2u, 1u, 2u};
enum check_mode { CHECK_OFF, CHECK_ON, CHECK_ZERO };
enum fault {
    CLEAN, WRONG_WINDOW, WRONG_KV, WRONG_SC, READ_FAIL,
    WRONG_PAST, WRONG_GROUPS, WRONG_PENDING, WRONG_SNAP_PENDING,
    WRONG_MAINH_COUNT, WRONG_MAINH_END, SYNC_FAIL, SHORT_WINDOW, SHORT_SNAPSHOT
};
static enum fault g_fault;
static ds4_v41_state *g_state;
static uint32_t g_reads, g_syncs;
static int g_bad_read_size;

static uint32_t ds4_layer_compress_ratio(uint32_t il) { return g_ratio[il]; }
static uint64_t ds4_gpu_tensor_bytes(const ds4_gpu_tensor *t) { return t ? t->bytes : 0u; }
static int ds4_gpu_tensor_copy(ds4_gpu_tensor *dst, uint64_t doff,
                               const ds4_gpu_tensor *src, uint64_t soff, uint64_t bytes) {
    if (!dst || !src || doff > dst->bytes || bytes > dst->bytes - doff ||
        soff > src->bytes || bytes > src->bytes - soff) { return 0; }
    memcpy(dst->data + doff, src->data + soff, (size_t)bytes);
    if ((g_fault == WRONG_KV && dst == g_state->cpre_kv[0]) ||
        (g_fault == WRONG_SC && dst == g_state->cpre_sc[0])) {
        dst->data[doff + bytes - 1u] ^= 1u;
    }
    return 1;
}
static int ds4_gpu_v41_win_ring_snap_tensor(ds4_gpu_tensor *win, ds4_gpu_tensor *snap,
                                           uint32_t pos, uint32_t first, uint32_t n,
                                           uint32_t window, uint32_t dim, int back, const void *posd) {
    (void)dim; (void)posd;
    for (uint32_t i = first; i < n; i++) {
        const uint64_t cell = (uint64_t)((pos + i) % window) * ROW_BYTES;
        if (back) {
            memcpy(win->data + cell, snap->data + (uint64_t)i * ROW_BYTES, ROW_BYTES);
            if (g_fault == WRONG_WINDOW && win == g_state->win[0] && i + 1u == n) {
                win->data[cell + ROW_BYTES - 1u] ^= 1u;
            }
        } else {
            memcpy(snap->data + (uint64_t)i * ROW_BYTES, win->data + cell, ROW_BYTES);
        }
    }
    return 1;
}
static int ds4_gpu_synchronize(void) {
    g_syncs++;
    switch (g_fault) {
        case WRONG_PAST: g_state->n_past++; break;
        case WRONG_GROUPS: g_state->ng_src[0]++; break;
        case WRONG_PENDING: g_state->cpend[0]++; break;
        case WRONG_SNAP_PENDING: g_state->snap_cpend[0]++; break;
        case WRONG_MAINH_COUNT: g_state->mainh_n++; break;
        case WRONG_MAINH_END: g_state->mainh_end++; break;
        case SYNC_FAIL: return 0;
        case SHORT_WINDOW: g_state->win[0]->bytes = ROW_BYTES; break;
        case SHORT_SNAPSHOT: g_state->snap_win[0]->bytes = ROW_BYTES; break;
        default: break;
    }
    return 1;
}
static int ds4_gpu_tensor_read(const ds4_gpu_tensor *src, uint64_t off, void *dst, uint64_t bytes) {
    g_reads++;
    if (bytes != ROW_BYTES) { g_bad_read_size = 1; return 0; }
    if (g_fault == READ_FAIL || !src || off > src->bytes || bytes > src->bytes - off) { return 0; }
    memcpy(dst, src->data + off, (size_t)bytes);
    return 1;
}

#include "ds41_rollback_under_test.inc"

static ds4_gpu_tensor *tensor(uint32_t rows, uint8_t value) {
    ds4_gpu_tensor *t = calloc(1, sizeof *t);
    if (!t) { abort(); }
    t->bytes = rows * ROW_BYTES; t->data = malloc((size_t)t->bytes);
    if (!t->data) { abort(); }
    for (uint32_t i = 0; i < rows; i++) { memset(t->data + i * ROW_BYTES, value + i, ROW_BYTES); }
    return t;
}
static void free_tensor(ds4_gpu_tensor *t) {
    if (t) { free(t->data); free(t); }
}

static int run_case(const char *name, uint32_t keep, uint32_t mainh_n,
                     enum check_mode mode, enum fault fault, int expected) {
    ds4_v41_state st = {0};
    st.ctx = MOCK_CTX; st.snap_past = START_POS; st.snap_n = BATCH_ROWS;
    st.n_past = START_POS + BATCH_ROWS; st.snap_on = 1u;
    st.mainh_cap = DS4_N_SWA; st.mainh_n = mainh_n;
    st.mainh_end = START_POS + BATCH_ROWS - 1u; st.mainh = tensor(1u, 0x70u);
    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        st.win[il] = tensor(DS4_N_SWA, 0x10u + il * DS4_N_SWA);
        st.snap_win[il] = tensor(DS4_MTP_MAX_BLOCK + 1u, 0x80u);
        for (uint32_t i = 0; i < BATCH_ROWS; i++) {
            const uint32_t cell = (START_POS + i) % DS4_N_SWA;
            memcpy(st.snap_win[il]->data + i * ROW_BYTES, st.win[il]->data + cell * ROW_BYTES, ROW_BYTES);
            memset(st.win[il]->data + cell * ROW_BYTES, 0xd0u + i, ROW_BYTES);
        }
        if (!g_ds4_v41.is_kv_source[il]) { continue; }
        st.ng_src[il] = st.n_past / g_ratio[il]; st.cpend[il] = st.n_past % g_ratio[il];
        st.snap_cpend[il] = START_POS % g_ratio[il];
        if (g_ratio[il] == 1u) { continue; }
        st.cpre_kv[il] = tensor(g_ratio[il] + DS4_MTP_MAX_BLOCK + 1u, 0xe0u);
        st.cpre_sc[il] = tensor(g_ratio[il] + DS4_MTP_MAX_BLOCK + 1u, 0xe0u);
        st.snap_cpre_kv[il] = tensor(g_ratio[il] + DS4_MTP_MAX_BLOCK + 1u, 0x40u);
        st.snap_cpre_sc[il] = tensor(g_ratio[il] + DS4_MTP_MAX_BLOCK + 1u, 0x60u);
    }
    if (mode == CHECK_ON) { setenv("DS41_ROLLBACK_CHECK", "1", 1); }
    else if (mode == CHECK_ZERO) { setenv("DS41_ROLLBACK_CHECK", "0", 1); }
    else { unsetenv("DS41_ROLLBACK_CHECK"); }
    g_state = &st; g_fault = fault; g_reads = 0; g_syncs = 0; g_bad_read_size = 0;
    const int result = v41_spec_rollback(&st, keep);
    const int observed = mode != CHECK_ON ? (g_reads == 0u && g_syncs == 0u) :
        (g_syncs > 0u && (fault >= WRONG_PAST || g_reads > 0u));
    const uint32_t required_reads = 2u * DS4_N_LAYER * (BATCH_ROWS - keep) +
        4u * ((START_POS + keep) % g_ratio[0]);
    const int coverage = mode != CHECK_ON || fault != CLEAN || g_reads == required_reads;
    const int failed = result != expected || !observed || !coverage || g_bad_read_size;
    if (failed) {
        printf("FAIL %s: result=%d/%d reads=%u syncs=%u observed=%d bad-read-size=%d\n",
               name, result, expected, g_reads, g_syncs, observed, g_bad_read_size);
    }
    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        free_tensor(st.win[il]); free_tensor(st.snap_win[il]);
        free_tensor(st.cpre_kv[il]); free_tensor(st.cpre_sc[il]);
        free_tensor(st.snap_cpre_kv[il]); free_tensor(st.snap_cpre_sc[il]);
    }
    free_tensor(st.mainh);
    return failed;
}

int main(void) {
    int failures = 0;
    failures += run_case("default off", 2u, DS4_N_SWA, CHECK_OFF, WRONG_WINDOW, 1);
    failures += run_case("explicit zero", 2u, DS4_N_SWA, CHECK_ZERO, WRONG_WINDOW, 1);
    failures += run_case("rejected tail", 2u, DS4_N_SWA, CHECK_ON, CLEAN, 1);
    failures += run_case("no pending rows", 1u, DS4_N_SWA, CHECK_ON, CLEAN, 1);
    failures += run_case("whole batch accepted", BATCH_ROWS, DS4_N_SWA, CHECK_ON, CLEAN, 1);
    failures += run_case("short mainh run", 2u, BATCH_ROWS, CHECK_ON, CLEAN, 1);
    failures += run_case("wrong window lane", 2u, DS4_N_SWA, CHECK_ON, WRONG_WINDOW, 0);
    failures += run_case("wrong pending KV", 2u, DS4_N_SWA, CHECK_ON, WRONG_KV, 0);
    failures += run_case("wrong pending gate", 2u, DS4_N_SWA, CHECK_ON, WRONG_SC, 0);
    failures += run_case("readback failure", 2u, DS4_N_SWA, CHECK_ON, READ_FAIL, 0);
    failures += run_case("wrong position frontier", 2u, DS4_N_SWA, CHECK_ON, WRONG_PAST, 0);
    failures += run_case("wrong source groups", 2u, DS4_N_SWA, CHECK_ON, WRONG_GROUPS, 0);
    failures += run_case("wrong source remainder", 2u, DS4_N_SWA, CHECK_ON, WRONG_PENDING, 0);
    failures += run_case("wrong snapshot remainder", 2u, DS4_N_SWA, CHECK_ON, WRONG_SNAP_PENDING, 0);
    failures += run_case("wrong mainh count", 2u, DS4_N_SWA, CHECK_ON, WRONG_MAINH_COUNT, 0);
    failures += run_case("wrong mainh frontier", 2u, DS4_N_SWA, CHECK_ON, WRONG_MAINH_END, 0);
    failures += run_case("synchronization failure", 2u, DS4_N_SWA, CHECK_ON, SYNC_FAIL, 0);
    failures += run_case("short restored window", 2u, DS4_N_SWA, CHECK_ON, SHORT_WINDOW, 0);
    failures += run_case("short snapshot window", 2u, DS4_N_SWA, CHECK_ON, SHORT_SNAPSHOT, 0);
    printf("ds41 rollback plumbing gate: %s (%d/19 failures)\n", failures ? "FAIL" : "PASS", failures);
    return failures ? 1 : 0;
}
