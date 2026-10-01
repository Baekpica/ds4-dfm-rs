/* Regression guard for the token-tile attention ldmatrix address operands.
 *
 * Why this test exists
 * The three ldmatrix helpers in ds4_cuda.cu handed ptxas a 32-bit
 * __cvta_generic_to_shared offset in an "r" operand; ptxas then emitted an
 * address adjusted by the shared-window base (IADD3 R74, R15, -c[0x0][0x18],
 * RZ immediately before LDSM.16.M88.2 R74, [R74]) and every ldmatrix faulted
 * with an illegal shared access.  The token-tile attention kernel is the only
 * caller of those helpers, and nothing on a machine without a DeepSeek V4 or
 * GLM 5.3 artifact exercised it.  This test drives the production entry at that
 * kernel's own geometry with synthetic tensors and binds the output to a
 * double-precision reference, so the path runs - and computes the attention it
 * claims to - anywhere the CUDA backend initializes.
 *
 * The dispatch under test is the committed one: ds4_cuda.cu selects the
 * token-tile kernel for n_tokens >= 128, head_dim == 512, n_head == 64,
 * window == 128 and a compressed cache.  Running with DS4_ATTN_TOKENTILE=0
 * sends the same cases through the pre-existing fallback, which must match the
 * same reference.
 *
 * The reference is the attention the dispatcher documents for the non-indexed
 * (dense compressed-cache) callers: for token t at position p = pos0 + t the
 * keys are the raw window [p - 127, p] plus the compressed rows
 * [0, (p + 1) / ratio) clamped to n_comp, scored as dot(q, k) / sqrt(512), and
 * softmaxed with the per-head sink as a denominator-only term.  It shares no
 * code with the kernels it checks.
 *
 * One case per process (the test takes the case index), because this backend
 * does not survive a second attention case in the same process on every
 * machine: measured on the RTX 4070 SUPER / CUDA 13.3 box, the second case's
 * first 16 MiB upload fails with cudaErrorInvalidValue, while the same case
 * passes alone and a loop of allocation, upload, attention and readback
 * rounds with one case's geometry runs clean.  tests/cuda_long_context_smoke.c
 * fails its second case the same way at the committed revision, so it is not
 * this test's structure that breaks and not something this test can fix.
 */
#include "ds4_gpu.h"

#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum {
    N_TOKENS = 128,
    N_HEAD = 64,
    HEAD_DIM = 512,          /* kTTHeadDim: the tile kernel's only head dim */
    WINDOW = 128,            /* kTTRawWindow */
    WINDOW_FIRST = WINDOW - 1u,
    MAX_KEYS = 1024,         /* >= WINDOW + n_comp for every case below */
    /* The raw ring must cover every position the window can reach; otherwise
     * the kernel legitimately wraps its slot arithmetic and the modeled
     * attention would no longer be the one under test. */
    RING_COVERS_WINDOW = N_TOKENS + WINDOW_FIRST,
};

/* Relative limit against the output peak.  The kernel scores fp32 queries
 * against fp16 mirror rows and softmaxes in fp16 probabilities, so it cannot
 * be bit-identical to an fp64 reference; the measured spread is two orders
 * below this. */
static const double kParityLimit = 4.0e-3;

typedef enum {
    KV_PATTERN_DISTINCT,     /* every row and query differs: catches a wrong row */
    KV_PATTERN_CONSTANT,     /* q = 0, every row identical: catches a wrong scale */
} kv_pattern;

typedef struct {
    const char *label;
    uint32_t pos0;
    uint32_t n_raw;
    uint32_t raw_cap;
    uint32_t n_comp;
    uint32_t comp_cap;
    uint32_t ratio;
    kv_pattern pattern;
} attn_case;

/* Small deterministic values: a real softmax shape, no overflow, and a pattern
 * that differs per row so a wrong base address cannot match by accident. */
static float pat(uint32_t i) {
    return ((float)((i * 2654435761u) >> 8) / 8388608.0f - 1.0f) * 0.05f;
}

static void fill_distinct(float *dst, uint64_t n, uint32_t seed) {
    for (uint64_t i = 0; i < n; i++) {
        dst[i] = pat((uint32_t)(i + seed));
    }
}

/* Compressed rows token t can see, as attention_tokentile_dense_build_kernel
 * computes them for a caller without per-row positions. */
static uint32_t visible_comp_rows(const attn_case *c, uint32_t t) {
    const uint32_t qpos = c->pos0 + t;
    const uint32_t visible = (qpos + 1u) / c->ratio;
    return visible > c->n_comp ? c->n_comp : visible;
}

/* Position of the first raw ring row, as the launcher computes it. */
static uint32_t first_raw_pos(const attn_case *c) {
    return c->pos0 + N_TOKENS - c->n_raw;
}

/* Where the ring must start for the reference to model every window the
 * kernels read: the earliest position any window reaches. */
static uint32_t ring_start(const attn_case *c) {
    return c->pos0 > WINDOW_FIRST ? c->pos0 - WINDOW_FIRST : 0u;
}

/* Rows the raw mirror zero-fills: the window reaching past position 0. */
static uint32_t raw_row_min(const attn_case *c) {
    uint32_t available = c->n_raw - N_TOKENS;
    if (available > c->pos0) {
        available = c->pos0;
    }
    if (available > WINDOW_FIRST) {
        available = WINDOW_FIRST;
    }
    return WINDOW_FIRST - available;
}

/* Every failure path here is a CUDA call; name the runtime error so a failing
 * run says what the backend did, not only which step was reached. */
static void report_cuda_error(const char *label, const char *step) {
    const cudaError_t err = cudaGetLastError();
    fprintf(stderr, "%s: %s failed: %s (%s)\n", label, step,
            cudaGetErrorName(err), cudaGetErrorString(err));
}

static double dot_f64(const float *a, const float *b) {
    double s = 0.0;
    for (uint32_t d = 0; d < HEAD_DIM; d++) {
        s += (double)a[d] * (double)b[d];
    }
    return s;
}

/* One output row: softmax over the visible raw window plus the visible
 * compressed rows, sink included in the denominator only. */
static void ref_row(const attn_case *c,
                    uint32_t t,
                    const float *q_row,
                    const float *raw_kv,
                    const float *comp_kv,
                    float sink,
                    double *out) {
    const double scale = 1.0 / sqrt((double)HEAD_DIM);
    const uint32_t ring0 = first_raw_pos(c);
    const uint32_t row_min = raw_row_min(c);
    const uint32_t n_comp_vis = visible_comp_rows(c, t);
    uint32_t nk = 0;
    double scores[MAX_KEYS];
    const float *keys[MAX_KEYS];

    /* The window of this token: mirror rows [t, t + WINDOW - 1] hold positions
     * [p - 127, p]; the first row_min of them are zero-filled and unselected. */
    for (uint32_t w = 0; w < WINDOW; w++) {
        const uint32_t mirror_row = t + w;
        if (mirror_row < row_min) {
            continue;
        }
        const uint32_t pos = c->pos0 + mirror_row - WINDOW_FIRST;
        if (pos < ring0) {
            continue;
        }
        keys[nk] = raw_kv + (uint64_t)(pos - ring0) * HEAD_DIM;
        scores[nk] = dot_f64(q_row, keys[nk]) * scale;
        nk++;
    }
    for (uint32_t comp = 0; comp < n_comp_vis; comp++) {
        keys[nk] = comp_kv + (uint64_t)comp * HEAD_DIM;
        scores[nk] = dot_f64(q_row, keys[nk]) * scale;
        nk++;
    }
    if (nk == 0u || nk >= MAX_KEYS) {
        return;
    }

    double m = (double)sink;
    for (uint32_t i = 0; i < nk; i++) {
        if (scores[i] > m) {
            m = scores[i];
        }
    }
    double den = exp((double)sink - m);
    double acc[HEAD_DIM];
    memset(acc, 0, sizeof(acc));
    for (uint32_t i = 0; i < nk; i++) {
        const double p = exp(scores[i] - m);
        den += p;
        for (uint32_t d = 0; d < HEAD_DIM; d++) {
            acc[d] += p * (double)keys[i][d];
        }
    }
    for (uint32_t d = 0; d < HEAD_DIM; d++) {
        out[d] = acc[d] / den;
    }
}

static int run_case(const attn_case *c) {
    const uint64_t q_count = (uint64_t)N_TOKENS * N_HEAD * HEAD_DIM;
    const uint64_t raw_count = (uint64_t)c->raw_cap * HEAD_DIM;
    const uint64_t comp_count = (uint64_t)c->n_comp * HEAD_DIM;
    const uint32_t ring0 = first_raw_pos(c);

    if (ring0 != ring_start(c) || (uint64_t)ring0 + c->n_raw > c->raw_cap) {
        fprintf(stderr, "%s: ring must start at %u and fit raw_cap=%u (has %u, n_raw=%u)\n",
                c->label, ring_start(c), c->raw_cap, ring0, c->n_raw);
        return 1;
    }
    if ((uint32_t)MAX_KEYS < WINDOW + c->n_comp) {
        fprintf(stderr, "%s: MAX_KEYS too small for n_comp=%u\n", c->label, c->n_comp);
        return 1;
    }

    float *sinks = (float *)malloc(N_HEAD * sizeof(float));
    float *q_host = (float *)malloc(q_count * sizeof(float));
    float *raw_host = (float *)malloc(raw_count * sizeof(float));
    float *comp_host = (float *)malloc(comp_count * sizeof(float));
    float *heads_host = (float *)calloc(q_count, sizeof(float));
    if (!sinks || !q_host || !raw_host || !comp_host || !heads_host) {
        return 1;
    }
    for (uint32_t h = 0; h < N_HEAD; h++) {
        sinks[h] = 0.05f * (float)((int)(h % 5u) - 2);
    }
    if (c->pattern == KV_PATTERN_DISTINCT) {
        fill_distinct(q_host, q_count, 1u);
        fill_distinct(raw_host, raw_count, 2u);
        fill_distinct(comp_host, comp_count, 3u);
    } else {
        /* A constant cache with a zero query makes every key equally likely, so
         * every output element must be that constant whatever the kernel
         * selects; 0.125 is exact in both fp32 and fp16. */
        for (uint64_t i = 0; i < q_count; i++) {
            q_host[i] = 0.0f;
        }
        for (uint64_t i = 0; i < raw_count; i++) {
            raw_host[i] = 0.125f;
        }
        for (uint64_t i = 0; i < comp_count; i++) {
            comp_host[i] = 0.125f;
        }
    }

    ds4_gpu_tensor *heads = ds4_gpu_tensor_alloc(q_count * sizeof(float));
    ds4_gpu_tensor *q = ds4_gpu_tensor_alloc(q_count * sizeof(float));
    ds4_gpu_tensor *raw = ds4_gpu_tensor_alloc(raw_count * sizeof(float));
    ds4_gpu_tensor *comp = ds4_gpu_tensor_alloc(comp_count * sizeof(float));
    int rc = 1;
    if (!heads || !q || !raw || !comp) {
        fprintf(stderr, "%s: tensor allocation failed\n", c->label);
    } else if (!ds4_gpu_tensor_write(q, 0, q_host, q_count * sizeof(float))) {
        report_cuda_error(c->label, "q upload");
    } else if (!ds4_gpu_tensor_write(raw, 0, raw_host, raw_count * sizeof(float))) {
        report_cuda_error(c->label, "raw upload");
    } else if (!ds4_gpu_tensor_write(comp, 0, comp_host, comp_count * sizeof(float))) {
        report_cuda_error(c->label, "comp upload");
    } else if (!ds4_gpu_attention_decode_mixed_batch_heads_tensor(
            heads,
            sinks,
            (uint64_t)N_HEAD * sizeof(float),
            0,                    /* sinks_offset */
            q,
            raw,
            comp,
            0,                    /* comp_kv_f16: FP32 rows */
            NULL,                 /* comp_mask */
            0,                    /* use_comp_mask */
            N_TOKENS,
            c->pos0,
            c->n_raw,
            c->raw_cap,
            0,                    /* raw_start */
            c->n_comp,
            c->comp_cap,
            WINDOW,
            c->ratio,
            N_HEAD,
            HEAD_DIM,
            NULL,                 /* comp_fp8 */
            NULL,                 /* comp_scale */
            NULL,                 /* positions */
            NULL,                 /* seq_id */
            0,                    /* allow_mseq_heads8 */
            NULL,                 /* scalars */
            UINT32_MAX,           /* il_for_decode1 */
            c->pos0)) {           /* tt_run_pos0: one consecutive-position run */
        report_cuda_error(c->label, "attention dispatch rejected");
    } else if (!ds4_gpu_synchronize()) {
        report_cuda_error(c->label, "attention kernel");
    } else if (!ds4_gpu_tensor_read(heads, 0, heads_host, q_count * sizeof(float))) {
        report_cuda_error(c->label, "heads readback");
    } else {
        double worst_abs = 0.0, biggest = 0.0;
        uint32_t worst_t = 0, worst_h = 0, worst_d = 0;
        double *ref = (double *)malloc(HEAD_DIM * sizeof(double));
        if (!ref) {
            fprintf(stderr, "%s: reference buffer allocation failed\n", c->label);
        } else {
            for (uint32_t t = 0; t < N_TOKENS; t++) {
                for (uint32_t h = 0; h < N_HEAD; h++) {
                    const uint64_t base = ((uint64_t)t * N_HEAD + h) * HEAD_DIM;
                    ref_row(c, t, q_host + base, raw_host, comp_host, sinks[h], ref);
                    for (uint32_t d = 0; d < HEAD_DIM; d++) {
                        const double got = (double)heads_host[base + d];
                        const double want = ref[d];
                        const double diff = fabs(got - want);
                        if (fabs(got) > biggest) {
                            biggest = fabs(got);
                        }
                        if (diff > worst_abs) {
                            worst_abs = diff;
                            worst_t = t;
                            worst_h = h;
                            worst_d = d;
                        }
                    }
                }
            }
            free(ref);
            const double rel = biggest > 0.0 ? worst_abs / biggest : worst_abs;
            const int ok = rel <= kParityLimit;
            printf("%-34s keys<=%u max_abs=%.3g max_rel=%.3g limit=%.1g worst=(t%u h%u d%u) %s\n",
                   c->label, (unsigned)(WINDOW + c->n_comp), worst_abs, rel, kParityLimit,
                   worst_t, worst_h, worst_d, ok ? "PASS" : "FAIL");
            if (ok) {
                rc = 0;
            }
        }
    }

    ds4_gpu_tensor_free(comp);
    ds4_gpu_tensor_free(raw);
    ds4_gpu_tensor_free(q);
    ds4_gpu_tensor_free(heads);
    free(heads_host);
    free(comp_host);
    free(raw_host);
    free(q_host);
    free(sinks);
    return rc;
}

int main(int argc, char **argv) {
    /* An optional case index runs one case only, which keeps the cases
     * separable when a machine misbehaves after a first backend round. */
    const int only = argc > 1 ? atoi(argv[1]) : -1;
    /* The window fully in the ring and every compressed row visible from the
     * last token: the widest row set the kernel stages. */
    const attn_case wide = {
        .label = "distinct, full window + comp",
        .pos0 = 256u,
        .n_raw = RING_COVERS_WINDOW,
        .raw_cap = 512u,
        .n_comp = 128u,
        .comp_cap = 128u,
        .ratio = 4u,
        .pattern = KV_PATTERN_DISTINCT,
    };
    /* A ring that ends at the current position and a ratio that leaves the
     * compressed stage partly masked: the partial (nr < 32) stages. */
    const attn_case partial = {
        .label = "distinct, partial window + mask",
        .pos0 = 64u,
        .n_raw = 64u + N_TOKENS,   /* ring starts at position 0 */
        .raw_cap = 512u,
        .n_comp = 64u,
        .comp_cap = 64u,
        .ratio = 2u,
        .pattern = KV_PATTERN_DISTINCT,
    };
    const attn_case constant = {
        .label = "constant cache, zero query",
        .pos0 = 128u,
        .n_raw = RING_COVERS_WINDOW,
        .raw_cap = 512u,
        .n_comp = 128u,
        .comp_cap = 128u,
        .ratio = 8u,
        .pattern = KV_PATTERN_CONSTANT,
    };

    if (!ds4_gpu_init()) {
        fprintf(stderr, "cuda_tokentile_ldmatrix: CUDA backend init failed\n");
        return 1;
    }
    int rc = 0;
    if ((only < 0 || only == 0) && run_case(&wide) != 0) rc = 1;
    if ((only < 0 || only == 1) && run_case(&partial) != 0) rc = 1;
    if ((only < 0 || only == 2) && run_case(&constant) != 0) rc = 1;
    ds4_gpu_cleanup();
    if (rc == 0) {
        puts("cuda_tokentile_ldmatrix: OK");
    }
    return rc;
}
