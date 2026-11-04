/* Deciding measurement for the CPU-expert phase (handoff 2026-11-04).
 *
 * One real routed-expert matrix, decoded and dotted on the CPU with the same
 * arithmetic the CUDA decode uses (cuda/gemma_vq_decode.cuh:74-93): a 13-bit
 * index into the layer's 8192-word E4M3 book, plus one 8-bit residual index
 * into the 256-word residual book, the row's f16 gain, a bf16 rounding of the
 * weight and of the activation, and an f32 FMA chain. The question this
 * answers: does 240 experts x 3 matrices x 1,982,464 weights (1.43 G weights
 * per token, 720 of these records) land under the device path's ~94 ms?
 *
 * The record is layer 5, expert 0, down_proj of the R13+8 release, from
 * `ds4 -m <release> --inventory`:
 *   model.layers.5.experts.0.down_proj                abs 6295195648 408448 B
 *   model.layers.5.experts.down_proj.vq_codebook      abs 6295130112  65536 B
 *   model.layers.5.experts.0.down_proj.res            abs 6347479040 247936 B
 *   model.layers.5.experts.down_proj.vq_residual_book abs 6347476992   2048 B
 * The GVQ1 header is FORMAT.md's v1 layout (crates/ds4-core/src/beam_vq.rs:
 * 143-200): magic, rows at +8, cols at +12, k at +16, book at +24, and the
 * offsets [header=128, main, plane, size] at +32/+40/+48/+56. GVR1 is the
 * same 128-byte shape (crates/ds4-core/src/glm_vq.rs:185-225): one index byte
 * per 8-weight vector at +128.
 *
 * This is a throughput instrument, not the port: it reads the one record,
 * measures the GEMV, and prints the extrapolation. It does not reproduce the
 * warp XOR reduction order (each row is summed sequentially here), so it is
 * not a byte-identity reference.
 *
 * build: cc -O3 -mavx2 -mf16c -pthread -o tests/gemma4_cpu_expert_gemv \
 *            tests/gemma4_cpu_expert_gemv.c
 * usage: tests/gemma4_cpu_expert_gemv <trunk.gguf> [threads] [iters]
 */
#include <errno.h>
#include <immintrin.h>
#include <math.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define GVQ_OFF 6295195648ULL
#define GVQ_BYTES 408448ULL
#define BOOK_OFF 6295130112ULL
#define BOOK_BYTES 65536ULL
#define RES_OFF 6347479040ULL
#define RES_BYTES 247936ULL
#define RES_BOOK_OFF 6347476992ULL
#define RES_BOOK_BYTES 2048ULL

#define VECTOR 8u
#define HEADER 128u

static float e4m3[256];

/* OCP E4M3 (the format __nv_fp8_e4m3 holds): 1 sign, 4 exponent bits with
 * bias 7, 3 mantissa bits, subnormals at 2^-6 * m/8, and 0x7F/0xFF are NaN
 * (E4M3 FN has no infinities). */
static void init_e4m3(void) {
    for (int i = 0; i < 256; i++) {
        const int sign = (i >> 7) & 1, exp = (i >> 3) & 15, man = i & 7;
        float v;
        if (exp == 0) {
            v = (man / 8.0f) * ldexpf(1.0f, -6);
        } else if (exp == 15 && man == 7) {
            v = NAN;
        } else {
            v = (1.0f + man / 8.0f) * ldexpf(1.0f, exp - 7);
        }
        e4m3[i] = sign ? -v : v;
    }
}

/* bf16 round-to-nearest-even, the same value __float2bfloat16_rn produces. */
static inline float bf16_rn(float f) {
    uint32_t u;
    memcpy(&u, &f, sizeof(u));
    u = (u + 0x7FFFu + ((u >> 16) & 1u)) & 0xFFFF0000u;
    memcpy(&f, &u, sizeof(f));
    return f;
}

static uint8_t *read_span(FILE *f, uint64_t off, size_t n) {
    uint8_t *p = malloc(n);
    if (!p) { fprintf(stderr, "alloc %zu failed\n", n); exit(2); }
    if (fseeko(f, (off_t)off, SEEK_SET) != 0 || fread(p, 1, n, f) != n) {
        fprintf(stderr, "read %zu at %llu failed: %s\n", n,
            (unsigned long long)off, strerror(errno));
        exit(2);
    }
    return p;
}

static uint32_t u32_at(const uint8_t *p, size_t o) {
    uint32_t v;
    memcpy(&v, p + o, sizeof(v));
    return v;
}

struct gvq {
    const uint8_t *gain;
    const uint8_t *main;
    const uint8_t *plane;
    uint32_t rows, cols;
    uint64_t nvec;
};

struct gvr {
    const uint8_t *res;
    uint32_t rows, cols;
};

static void parse_gvq(struct gvq *g, const uint8_t *p, uint64_t bytes) {
    if (memcmp(p, "GVQ1", 4) != 0 || p[4] != 1 || p[5] != 0 || p[6] != 13 ||
        p[7] != VECTOR || u32_at(p, 16) != 8192 || u32_at(p, 20) != 1) {
        fprintf(stderr, "not an R13 GVQ1 v1 record\n");
        exit(2);
    }
    g->rows = u32_at(p, 8);
    g->cols = u32_at(p, 12);
    if (u32_at(p, 32) != HEADER || u32_at(p, 56) != bytes) {
        fprintf(stderr, "noncanonical GVQ1 offsets\n");
        exit(2);
    }
    g->gain = p + HEADER;
    g->main = p + u32_at(p, 40);
    g->plane = p + u32_at(p, 48);
    g->nvec = (uint64_t)g->rows * (g->cols / VECTOR);
}

static void parse_gvr(struct gvr *r, const uint8_t *p, uint64_t bytes) {
    if (memcmp(p, "GVR1", 4) != 0 || p[4] != 1 || p[5] != 0 || p[6] != 8 ||
        p[7] != VECTOR || u32_at(p, 16) != 256 || u32_at(p, 20) != 1) {
        fprintf(stderr, "not a GVR8 residual record\n");
        exit(2);
    }
    r->rows = u32_at(p, 8);
    r->cols = u32_at(p, 12);
    if (u32_at(p, 32) != HEADER || u32_at(p, 56) != bytes) {
        fprintf(stderr, "noncanonical GVR1 offsets\n");
        exit(2);
    }
    r->res = p + HEADER;
}

/* Index i of the GVQ1 v1 stream (cuda/gemma_vq_decode.cuh:55-66): bits
 * 12*i..12*i+11 least significant first, two indices in three bytes, the 13th
 * bit at bit i%8 of the plane. */
static inline uint32_t gvq_index(const struct gvq *g, uint64_t v) {
    const uint64_t pair = v / 2 * 3;
    uint32_t index = (v & 1u)
        ? ((uint32_t)g->main[pair + 1] >> 4) | ((uint32_t)g->main[pair + 2] << 4)
        : (uint32_t)g->main[pair] | (((uint32_t)g->main[pair + 1] & 15u) << 8);
    index |= ((uint32_t)(g->plane[v / 8] >> (v % 8)) & 1u) << 12;
    return index;
}

struct job {
    const struct gvq *g;
    const struct gvr *r;
    const uint8_t *book;
    const uint8_t *rbook;
    const float *x;
    float *y;
    uint32_t row0, row1;
};

static void *gemv_rows(void *arg) {
    const struct job *j = arg;
    const struct gvq *g = j->g;
    const struct gvr *r = j->r;
    const uint64_t nvec_row = g->cols / VECTOR;
    for (uint32_t row = j->row0; row < j->row1; row++) {
        const float gain = _cvtsh_ss(*(const uint16_t *)(g->gain + 2u * row));
        float acc = 0.0f;
        const uint64_t base = (uint64_t)row * nvec_row;
        for (uint64_t k = 0; k < nvec_row; k++) {
            const uint64_t v = base + k;
            const uint32_t idx = gvq_index(g, v);
            const uint8_t *wb = j->book + (size_t)idx * VECTOR;
            const uint8_t *rb = j->rbook + (size_t)r->res[v] * VECTOR;
            const float *xp = j->x + k * VECTOR;
            for (uint32_t d = 0; d < VECTOR; d++) {
                const float word = e4m3[wb[d]] + e4m3[rb[d]];
                const float w = bf16_rn(gain * word);
                acc = fmaf(w, bf16_rn(xp[d]), acc);
            }
        }
        j->y[row] = acc;
    }
    return NULL;
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s <trunk.gguf> [threads] [iters]\n", argv[0]);
        return 2;
    }
    const char *path = argv[1];
    int threads = argc > 2 ? atoi(argv[2]) : 20;
    int iters = argc > 3 ? atoi(argv[3]) : 10;
    if (threads < 1) { threads = 1; }
    if (iters < 1) { iters = 1; }
    init_e4m3();

    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "open %s: %s\n", path, strerror(errno)); return 2; }
    uint8_t *gvq_p = read_span(f, GVQ_OFF, GVQ_BYTES);
    uint8_t *book_p = read_span(f, BOOK_OFF, BOOK_BYTES);
    uint8_t *res_p = read_span(f, RES_OFF, RES_BYTES);
    uint8_t *rbook_p = read_span(f, RES_BOOK_OFF, RES_BOOK_BYTES);
    fclose(f);

    struct gvq g;
    struct gvr r;
    parse_gvq(&g, gvq_p, GVQ_BYTES);
    parse_gvr(&r, res_p, RES_BYTES);
    if (r.rows != g.rows || r.cols != g.cols) {
        fprintf(stderr, "residual shape %ux%u != matrix %ux%u\n",
            r.rows, r.cols, g.rows, g.cols);
        return 2;
    }

    float *x = malloc((size_t)g.cols * sizeof(float));
    float *y = malloc((size_t)g.rows * sizeof(float));
    if (!x || !y) { return 2; }
    for (uint32_t i = 0; i < g.cols; i++) { x[i] = sinf((float)i * 0.017f); }

    printf("{\"test\":\"cpu_expert_gemv\",\"record\":\"layers.5.experts.0.down_proj\","
        "\"rows\":%u,\"cols\":%u,\"weights\":%llu,\"gvq_bytes\":%llu,"
        "\"res_bytes\":%llu,\"threads\":%d,\"iters\":%d}\n",
        g.rows, g.cols, (unsigned long long)g.rows * g.cols,
        (unsigned long long)GVQ_BYTES, (unsigned long long)RES_BYTES, threads, iters);

    /* Warm the pages and the books before the clock starts. */
    struct job whole = { &g, &r, book_p, rbook_p, x, y, 0, g.rows };
    gemv_rows(&whole);

    double best = 1e30, sum = 0.0;
    for (int it = 0; it < iters; it++) {
        struct timespec a, b;
        clock_gettime(CLOCK_MONOTONIC, &a);
        pthread_t tid[64];
        struct job jobs[64];
        for (int t = 0; t < threads; t++) {
            uint32_t r0 = (uint32_t)((uint64_t)g.rows * t / threads);
            uint32_t r1 = (uint32_t)((uint64_t)g.rows * (t + 1) / threads);
            jobs[t] = (struct job){ &g, &r, book_p, rbook_p, x, y, r0, r1 };
            if (pthread_create(&tid[t], NULL, gemv_rows, &jobs[t]) != 0) {
                fprintf(stderr, "pthread_create failed\n");
                return 2;
            }
        }
        for (int t = 0; t < threads; t++) { pthread_join(tid[t], NULL); }
        clock_gettime(CLOCK_MONOTONIC, &b);
        const double ms = (b.tv_sec - a.tv_sec) * 1e3 + (b.tv_nsec - a.tv_nsec) / 1e6;
        if (ms < best) { best = ms; }
        sum += ms;
    }
    const double bytes = (double)GVQ_BYTES + (double)RES_BYTES;
    printf("{\"matrix_ms_best\":%.3f,\"matrix_ms_mean\":%.3f,"
        "\"expert_ms\":%.3f,\"token_ms_240_experts\":%.3f,"
        "\"matrix_gbs\":%.2f}\n",
        best, sum / iters, best * 3.0, best * 720.0,
        bytes / (best * 1e6));
    free(x); free(y);
    free(gvq_p); free(book_p); free(res_p); free(rbook_p);
    return 0;
}
