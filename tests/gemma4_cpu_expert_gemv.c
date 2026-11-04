/* The fair CPU-expert ceiling (agy review 5.A, 2026-11-04).
 *
 * One real routed-expert matrix, decoded and dotted on the CPU with the same
 * arithmetic the CUDA decode uses (cuda/gemma_vq_decode.cuh:74-93): a 13-bit
 * index into the layer's 8192-word E4M3 book, plus one 8-bit residual index
 * into the 256-word residual book, the row's f16 gain, a bf16 rounding of the
 * weight and of the activation, and an f32 FMA chain. The question: does
 * 240 experts x 3 matrices x 1,982,464 weights (1.43 G weights per token, 720
 * of these records) land under the device path's ~94 ms?
 *
 * The first instrument (kept here as the scalar reference) answered with a
 * warm-L3 scalar loop, 0.387 ms/matrix, extrapolated to 278.4 ms/token. The
 * agy review found that number unfair in both directions: pthread_create and
 * join sat inside the timed region, every weight paid a scalar bf16_rn, one
 * serial FMA chain carried the row, and the same 656 KiB record was looped
 * from a 33 MiB L3. It asked for four changes, all here:
 *
 *  1. a persistent barrier threadpool, 32-row chunks stolen atomically, so no
 *     thread is created inside the timed region (ggml-cpu.c:1695-1739);
 *  2. codebooks pre-expanded to FP16 once at load, where every finite E4M3
 *     value is exact (1 sign, 5 exp, 10 mantissa holds 4 exp, 3 mantissa),
 *     turning each codeword into one 128-bit load plus vcvtph2ps;
 *  3. an AVX2 + F16C inner loop with four __m256 accumulators, unrolled four
 *     codewords (32 weights) per iteration, and the row gain hoisted out of
 *     the dot. The hoist is Class-B, not bit-identical: the scalar path
 *     rounds gain*word to bf16 per weight, this one scales the finished dot.
 *     The exact variant below prices that difference;
 *  4. a cold-DRAM sweep over 720 record copies (450.7 MiB, 13.6x the L3), so
 *     the reported token number is the real expert read, not an extrapolation
 *     from one cached record.
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
 * This is a throughput instrument, not the port: it does not reproduce the
 * warp XOR reduction order (each row is summed in AVX2 lane order here), so it
 * is not a byte-identity reference. It reports its distance from the scalar
 * reference so the cost of that difference is visible.
 *
 * build: cc -O3 -mavx2 -mfma -mf16c -pthread -o tests/gemma4_cpu_expert_gemv \
 *            tests/gemma4_cpu_expert_gemv.c -lm
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
#define BOOK_WORDS 8192u
#define RES_WORDS 256u
#define COLD_MATRICES 720u   /* 240 active experts x (gate, up, down) */
#define CHUNK_ROWS 32u

static float e4m3[256];
static uint16_t main_fp16[BOOK_WORDS * VECTOR];
static uint16_t res_fp16[RES_WORDS * VECTOR];

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

/* The 256 E4M3 values into FP16, exactly, and prove it by converting back.
 * A finite E4M3 value needs at most 10 mantissa bits of range and 11 bits of
 * exponent reach; FP16 carries 10 mantissa bits and 5 exponent bits, and the
 * smallest E4M3 subnormal is 2^-9, far inside FP16's 2^-24. */
static void expand_books(const uint8_t *book, const uint8_t *rbook) {
    int bad = 0;
    for (uint32_t i = 0; i < BOOK_WORDS * VECTOR; i++) {
        const float v = e4m3[book[i]];
        main_fp16[i] = _cvtss_sh(v, 0);
        if (!isnan(v) && _cvtsh_ss(main_fp16[i]) != v) { bad++; }
    }
    for (uint32_t i = 0; i < RES_WORDS * VECTOR; i++) {
        const float v = e4m3[rbook[i]];
        res_fp16[i] = _cvtss_sh(v, 0);
        if (!isnan(v) && _cvtsh_ss(res_fp16[i]) != v) { bad++; }
    }
    if (bad) {
        fprintf(stderr, "E4M3->FP16 not exact on %d entries\n", bad);
        exit(2);
    }
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

/* Scalar reference: the first instrument's loop, unchanged in arithmetic. */
static void gemv_rows_scalar(const struct gvq *g, const struct gvr *r,
        const uint8_t *book, const uint8_t *rbook, const float *x, float *y,
        uint32_t row0, uint32_t row1) {
    const uint64_t nvec_row = g->cols / VECTOR;
    for (uint32_t row = row0; row < row1; row++) {
        const float gain = _cvtsh_ss(*(const uint16_t *)(g->gain + 2u * row));
        float acc = 0.0f;
        const uint64_t base = (uint64_t)row * nvec_row;
        for (uint64_t k = 0; k < nvec_row; k++) {
            const uint64_t v = base + k;
            const uint32_t idx = gvq_index(g, v);
            const uint8_t *wb = book + (size_t)idx * VECTOR;
            const uint8_t *rb = rbook + (size_t)r->res[v] * VECTOR;
            const float *xp = x + k * VECTOR;
            for (uint32_t d = 0; d < VECTOR; d++) {
                const float word = e4m3[wb[d]] + e4m3[rb[d]];
                const float w = bf16_rn(gain * word);
                acc = fmaf(w, bf16_rn(xp[d]), acc);
            }
        }
        y[row] = acc;
    }
}

/* Horizontal sum of four accumulators into one broadcast lane. */
static inline __m256 hsum4x8(__m256 a, __m256 b, __m256 c, __m256 d) {
    const __m256 s = _mm256_add_ps(_mm256_add_ps(a, b), _mm256_add_ps(c, d));
    const __m128 lo = _mm256_castps256_ps128(s);
    const __m128 hi = _mm256_extractf128_ps(s, 1);
    __m128 t = _mm_add_ps(lo, hi);
    t = _mm_hadd_ps(t, t);
    t = _mm_hadd_ps(t, t);
    return _mm256_set1_ps(_mm_cvtss_f32(t));
}

/* bf16 round-to-nearest-even, vectorized, the same value bf16_rn produces. */
static inline __m256 bf16_rn8(__m256 w) {
    __m256i u = _mm256_castps_si256(w);
    const __m256i lsb = _mm256_and_si256(_mm256_srli_epi32(u, 16),
        _mm256_set1_epi32(1));
    u = _mm256_add_epi32(u, _mm256_add_epi32(_mm256_set1_epi32(0x7FFF), lsb));
    u = _mm256_and_si256(u, _mm256_set1_epi32((int)0xFFFF0000u));
    return _mm256_castsi256_ps(u);
}

/* Four consecutive 13-bit indices from six bytes plus one plane byte. The
 * group starts at an 8-vector boundary (checked at startup), so the four
 * plane bits share one byte at bit k%8. */
#define GVQ_INDEX4(ix, pb, sh, i0, i1, i2, i3) do {                        \
    (i0) = (uint32_t)(ix)[0] | (((uint32_t)(ix)[1] & 15u) << 8);           \
    (i1) = ((uint32_t)(ix)[1] >> 4) | ((uint32_t)(ix)[2] << 4);            \
    (i2) = (uint32_t)(ix)[3] | (((uint32_t)(ix)[4] & 15u) << 8);           \
    (i3) = ((uint32_t)(ix)[4] >> 4) | ((uint32_t)(ix)[5] << 4);            \
    (i0) |= ((uint32_t)((pb) >> (sh)) & 1u) << 12;                         \
    (i1) |= ((uint32_t)((pb) >> ((sh) + 1u)) & 1u) << 12;                  \
    (i2) |= ((uint32_t)((pb) >> ((sh) + 2u)) & 1u) << 12;                  \
    (i3) |= ((uint32_t)((pb) >> ((sh) + 3u)) & 1u) << 12;                  \
} while (0)

/* AVX2 + F16C, gain hoisted out of the dot (Class-B, see the header). */
static void gemv_rows_hoisted(const struct gvq *g, const struct gvr *r,
        const float *x, float *y, uint32_t row0, uint32_t row1) {
    const uint32_t nvec = g->cols / VECTOR;
    for (uint32_t row = row0; row < row1; row++) {
        const float gain = _cvtsh_ss(*(const uint16_t *)(g->gain + 2u * row));
        const uint64_t base = (uint64_t)row * nvec;
        const uint8_t *ix = g->main + (base >> 1) * 3u;
        const uint8_t *pl = g->plane + (base >> 3);
        const uint8_t *ri = r->res + base;
        const float *xp = x;
        __m256 a0 = _mm256_setzero_ps(), a1 = a0, a2 = a0, a3 = a0;
        for (uint32_t k = 0; k < nvec; k += 4u) {
            uint32_t i0, i1, i2, i3;
            GVQ_INDEX4(ix, pl[k >> 3], k & 7u, i0, i1, i2, i3);
            __m256 w0 = _mm256_add_ps(
                _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(main_fp16 + (size_t)i0 * 8))),
                _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(res_fp16 + (size_t)ri[k] * 8))));
            __m256 w1 = _mm256_add_ps(
                _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(main_fp16 + (size_t)i1 * 8))),
                _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(res_fp16 + (size_t)ri[k + 1u] * 8))));
            __m256 w2 = _mm256_add_ps(
                _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(main_fp16 + (size_t)i2 * 8))),
                _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(res_fp16 + (size_t)ri[k + 2u] * 8))));
            __m256 w3 = _mm256_add_ps(
                _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(main_fp16 + (size_t)i3 * 8))),
                _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(res_fp16 + (size_t)ri[k + 3u] * 8))));
            a0 = _mm256_fmadd_ps(w0, _mm256_loadu_ps(xp), a0);
            a1 = _mm256_fmadd_ps(w1, _mm256_loadu_ps(xp + 8), a1);
            a2 = _mm256_fmadd_ps(w2, _mm256_loadu_ps(xp + 16), a2);
            a3 = _mm256_fmadd_ps(w3, _mm256_loadu_ps(xp + 24), a3);
            ix += 6; xp += 32;
        }
        y[row] = _mm_cvtss_f32(_mm256_castps256_ps128(hsum4x8(a0, a1, a2, a3))) * gain;
    }
}

/* The same kernel with the scalar path's per-weight bf16 rounding of
 * gain*word kept. Slower; it exists to price the hoist, not to run a token. */
static void gemv_rows_exact(const struct gvq *g, const struct gvr *r,
        const float *x, float *y, uint32_t row0, uint32_t row1) {
    const uint32_t nvec = g->cols / VECTOR;
    for (uint32_t row = row0; row < row1; row++) {
        const float gain = _cvtsh_ss(*(const uint16_t *)(g->gain + 2u * row));
        const __m256 g8 = _mm256_set1_ps(gain);
        const uint64_t base = (uint64_t)row * nvec;
        const uint8_t *ix = g->main + (base >> 1) * 3u;
        const uint8_t *pl = g->plane + (base >> 3);
        const uint8_t *ri = r->res + base;
        const float *xp = x;
        __m256 a0 = _mm256_setzero_ps(), a1 = a0, a2 = a0, a3 = a0;
        for (uint32_t k = 0; k < nvec; k += 4u) {
            uint32_t i0, i1, i2, i3;
            GVQ_INDEX4(ix, pl[k >> 3], k & 7u, i0, i1, i2, i3);
            __m256 w0 = bf16_rn8(_mm256_mul_ps(g8, _mm256_add_ps(
                _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(main_fp16 + (size_t)i0 * 8))),
                _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(res_fp16 + (size_t)ri[k] * 8))))));
            __m256 w1 = bf16_rn8(_mm256_mul_ps(g8, _mm256_add_ps(
                _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(main_fp16 + (size_t)i1 * 8))),
                _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(res_fp16 + (size_t)ri[k + 1u] * 8))))));
            __m256 w2 = bf16_rn8(_mm256_mul_ps(g8, _mm256_add_ps(
                _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(main_fp16 + (size_t)i2 * 8))),
                _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(res_fp16 + (size_t)ri[k + 2u] * 8))))));
            __m256 w3 = bf16_rn8(_mm256_mul_ps(g8, _mm256_add_ps(
                _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(main_fp16 + (size_t)i3 * 8))),
                _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(res_fp16 + (size_t)ri[k + 3u] * 8))))));
            a0 = _mm256_fmadd_ps(w0, _mm256_loadu_ps(xp), a0);
            a1 = _mm256_fmadd_ps(w1, _mm256_loadu_ps(xp + 8), a1);
            a2 = _mm256_fmadd_ps(w2, _mm256_loadu_ps(xp + 16), a2);
            a3 = _mm256_fmadd_ps(w3, _mm256_loadu_ps(xp + 24), a3);
            ix += 6; xp += 32;
        }
        y[row] = _mm_cvtss_f32(_mm256_castps256_ps128(hsum4x8(a0, a1, a2, a3)));
    }
}

/* Persistent barrier threadpool: one barrier to release, one to join. Every
 * participant steals 32-row chunks from the sweep's atomic cursor, so a slow
 * core never holds a row block while another sits idle. */
struct sweep {
    const struct gvq *g;
    const struct gvr *r;
    const uint8_t *book, *rbook;   /* scalar mode only */
    const float *x;                /* raw, scalar mode rounds it itself */
    const float *xb;               /* bf16-rounded once, vector modes */
    float *y;
    uint32_t nmat, rows;
    int64_t next, total;
    int mode;                      /* 0 hoisted, 1 exact, 2 scalar */
};

static pthread_barrier_t bar_start, bar_done;
static struct sweep *volatile g_sw;
static volatile int g_stop;

static void sweep_run(struct sweep *s) {
    const uint32_t per = (s->rows + CHUNK_ROWS - 1u) / CHUNK_ROWS;
    int64_t c;
    while ((c = __atomic_fetch_add(&s->next, 1, __ATOMIC_RELAXED)) < s->total) {
        const uint32_t m = (uint32_t)(c / per);
        const uint32_t r0 = (uint32_t)(c % per) * CHUNK_ROWS;
        uint32_t r1 = r0 + CHUNK_ROWS;
        if (r1 > s->rows) { r1 = s->rows; }
        float *y = s->y + (size_t)m * s->rows;
        if (s->mode == 2) {
            gemv_rows_scalar(&s->g[m], &s->r[m], s->book, s->rbook, s->x, y, r0, r1);
        } else if (s->mode == 1) {
            gemv_rows_exact(&s->g[m], &s->r[m], s->xb, y, r0, r1);
        } else {
            gemv_rows_hoisted(&s->g[m], &s->r[m], s->xb, y, r0, r1);
        }
    }
}

static void *worker(void *arg) {
    (void)arg;
    for (;;) {
        pthread_barrier_wait(&bar_start);
        if (g_stop) { break; }
        sweep_run(g_sw);
        pthread_barrier_wait(&bar_done);
    }
    return NULL;
}

/* One timed sweep. The caller has already armed s->next/total/g_sw. */
static double sweep_once(struct sweep *s) {
    struct timespec a, b;
    clock_gettime(CLOCK_MONOTONIC, &a);
    pthread_barrier_wait(&bar_start);
    sweep_run(s);
    pthread_barrier_wait(&bar_done);
    clock_gettime(CLOCK_MONOTONIC, &b);
    return (b.tv_sec - a.tv_sec) * 1e3 + (b.tv_nsec - a.tv_nsec) / 1e6;
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s <trunk.gguf> [threads] [iters]\n", argv[0]);
        return 2;
    }
    const char *path = argv[1];
    int threads = argc > 2 ? atoi(argv[2]) : 28;
    int iters = argc > 3 ? atoi(argv[3]) : 5;
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
    if (g.cols % (VECTOR * 8u)) {
        fprintf(stderr, "cols %u is not a multiple of 64: the plane byte would "
            "not cover four indices\n", g.cols);
        return 2;
    }
    expand_books(book_p, rbook_p);

    const uint32_t rows = g.rows, cols = g.cols;
    float *x = malloc((size_t)cols * sizeof(float));
    float *xb = malloc((size_t)cols * sizeof(float));
    float *y_scalar = malloc((size_t)rows * sizeof(float));
    float *y_fast = malloc((size_t)rows * sizeof(float));
    float *y_exact = malloc((size_t)rows * sizeof(float));
    if (!x || !xb || !y_scalar || !y_fast || !y_exact) { return 2; }
    for (uint32_t i = 0; i < cols; i++) {
        x[i] = sinf((float)i * 0.017f);
        xb[i] = bf16_rn(x[i]);   /* the scalar path rounds x per weight; do it once */
    }

    /* The first instrument's loop, timed alone: the number this replaces. */
    double scalar_best = 1e30;
    for (int it = 0; it < iters; it++) {
        struct timespec a, b;
        clock_gettime(CLOCK_MONOTONIC, &a);
        gemv_rows_scalar(&g, &r, book_p, rbook_p, x, y_scalar, 0, rows);
        clock_gettime(CLOCK_MONOTONIC, &b);
        const double ms = (b.tv_sec - a.tv_sec) * 1e3 + (b.tv_nsec - a.tv_nsec) / 1e6;
        if (ms < scalar_best) { scalar_best = ms; }
    }

    /* Numeric check against the scalar reference on the one record. */
    gemv_rows_hoisted(&g, &r, xb, y_fast, 0, rows);
    gemv_rows_exact(&g, &r, xb, y_exact, 0, rows);
    double max_hoist = 0.0, max_exact = 0.0;
    double max_abs_hoist = 0.0, max_abs_exact = 0.0, max_abs_y = 0.0;
    uint32_t worst_row = 0;
    for (uint32_t i = 0; i < rows; i++) {
        const double den = fabs((double)y_scalar[i]);
        const double dh = fabs((double)y_fast[i] - (double)y_scalar[i]);
        const double de = fabs((double)y_exact[i] - (double)y_scalar[i]);
        if (den > max_abs_y) { max_abs_y = den; }
        if (dh > max_abs_hoist) { max_abs_hoist = dh; }
        if (de > max_abs_exact) { max_abs_exact = de; }
        if (den > 1e-9) {
            if (dh / den > max_hoist) { max_hoist = dh / den; worst_row = i; }
            if (de / den > max_exact) { max_exact = de / den; }
        }
    }
    if (getenv("G4_DEBUG")) {
        /* Independent double-precision dot of the same decoded record. */
        const uint32_t row = worst_row;
        const uint64_t nvec_row = cols / VECTOR;
        const uint64_t base = (uint64_t)row * nvec_row;
        double dref = 0.0, dsum = 0.0;
        for (uint64_t k = 0; k < nvec_row; k++) {
            const uint32_t idx = gvq_index(&g, base + k);
            for (uint32_t d = 0; d < VECTOR; d++) {
                const double w = (double)e4m3[book_p[(size_t)idx * VECTOR + d]] +
                    (double)e4m3[rbook_p[(size_t)r.res[base + k] * VECTOR + d]];
                dref += w * (double)x[k * VECTOR + d];
                dsum += fabs(w * (double)x[k * VECTOR + d]);
            }
        }
        fprintf(stderr, "debug: row %u double_ref=%.9g sum|terms|=%.9g "
            "scalar=%.9g fast=%.9g exact=%.9g\n", row, dref, dsum,
            (double)y_scalar[row], (double)y_fast[row], (double)y_exact[row]);
        fprintf(stderr, "debug: max|y|=%.6g max_abs_hoist=%.6g max_abs_exact=%.6g "
            "worst_row=%u scalar=%.9g fast=%.9g exact=%.9g\n",
            max_abs_y, max_abs_hoist, max_abs_exact, worst_row,
            (double)y_scalar[worst_row], (double)y_fast[worst_row],
            (double)y_exact[worst_row]);
        for (uint32_t v = 0; v < 4; v++) {
            const uint32_t idx = gvq_index(&g, v);
            fprintf(stderr, "debug: v=%u idx=%u ridx=%u book0=%g fp16_0=%g "
                "book1=%g fp16_1=%g\n", v, idx, (unsigned)r.res[v],
                (double)e4m3[book_p[(size_t)idx * VECTOR]],
                (double)_cvtsh_ss(main_fp16[(size_t)idx * VECTOR]),
                (double)e4m3[book_p[(size_t)idx * VECTOR + 1]],
                (double)_cvtsh_ss(main_fp16[(size_t)idx * VECTOR + 1]));
        }
        fprintf(stderr, "debug: x[0]=%.9g xb[0]=%.9g\n",
            (double)x[0], (double)xb[0]);
    }

    /* Persistent pool: threads-1 workers, main participates in every sweep. */
    const int workers = threads - 1;
    pthread_barrier_init(&bar_start, NULL, (unsigned)threads);
    pthread_barrier_init(&bar_done, NULL, (unsigned)threads);
    pthread_t *tid = workers > 0 ? calloc((size_t)workers, sizeof(*tid)) : NULL;
    for (int t = 0; t < workers; t++) {
        if (pthread_create(&tid[t], NULL, worker, NULL) != 0) {
            fprintf(stderr, "pthread_create failed\n");
            return 2;
        }
    }

    struct gvq warm_g = g;
    struct gvr warm_r = r;
    struct sweep sw = { &warm_g, &warm_r, book_p, rbook_p, x, xb, y_fast, 1,
        rows, 0, (int64_t)((rows + CHUNK_ROWS - 1u) / CHUNK_ROWS), 0 };

    /* Warm single-record matrix, the same shape the first instrument timed. */
    double warm_best = 1e30, warm_sum = 0.0;
    for (int it = 0; it < iters; it++) {
        sw.next = 0;
        g_sw = &sw;
        const double ms = sweep_once(&sw);
        if (ms < warm_best) { warm_best = ms; }
        warm_sum += ms;
    }

    /* Exact-rounding variant, warm, to price the hoist. */
    sw.mode = 1;
    double exact_best = 1e30;
    for (int it = 0; it < iters; it++) {
        sw.next = 0;
        g_sw = &sw;
        const double ms = sweep_once(&sw);
        if (ms < exact_best) { exact_best = ms; }
    }
    sw.mode = 0;

    /* Cold sweep: 720 distinct record copies, 450.7 MiB, 13.6x the L3. */
    uint8_t *cold_g = malloc((size_t)COLD_MATRICES * GVQ_BYTES);
    uint8_t *cold_r = malloc((size_t)COLD_MATRICES * RES_BYTES);
    struct gvq *cg = calloc(COLD_MATRICES, sizeof(*cg));
    struct gvr *cr = calloc(COLD_MATRICES, sizeof(*cr));
    float *cy = malloc((size_t)COLD_MATRICES * rows * sizeof(float));
    if (!cold_g || !cold_r || !cg || !cr || !cy) {
        fprintf(stderr, "cold buffer alloc failed\n");
        return 2;
    }
    for (uint32_t m = 0; m < COLD_MATRICES; m++) {
        memcpy(cold_g + (size_t)m * GVQ_BYTES, gvq_p, GVQ_BYTES);
        memcpy(cold_r + (size_t)m * RES_BYTES, res_p, RES_BYTES);
        parse_gvq(&cg[m], cold_g + (size_t)m * GVQ_BYTES, GVQ_BYTES);
        parse_gvr(&cr[m], cold_r + (size_t)m * RES_BYTES, RES_BYTES);
    }
    struct sweep cs = { cg, cr, book_p, rbook_p, x, xb, cy, COLD_MATRICES, rows, 0,
        (int64_t)COLD_MATRICES * ((rows + CHUNK_ROWS - 1u) / CHUNK_ROWS), 0 };
    double cold_best = 1e30, cold_sum = 0.0;
    for (int it = 0; it < iters; it++) {
        cs.next = 0;
        g_sw = &cs;
        const double ms = sweep_once(&cs);
        if (ms < cold_best) { cold_best = ms; }
        cold_sum += ms;
    }

    /* The first instrument's loop over the same 720 cold records: what its
     * 278.4 ms warm-L3 extrapolation actually costs when nothing is cached. */
    cs.mode = 2;
    double scalar_cold_best = 1e30;
    const int scalar_iters = iters < 3 ? iters : 3;
    for (int it = 0; it < scalar_iters; it++) {
        cs.next = 0;
        g_sw = &cs;
        const double ms = sweep_once(&cs);
        if (ms < scalar_cold_best) { scalar_cold_best = ms; }
    }
    cs.mode = 0;

    /* Teardown: release the pool and join it. */
    g_stop = 1;
    pthread_barrier_wait(&bar_start);
    for (int t = 0; t < workers; t++) { pthread_join(tid[t], NULL); }
    pthread_barrier_destroy(&bar_start);
    pthread_barrier_destroy(&bar_done);
    free(tid);

    const double bytes = (double)GVQ_BYTES + (double)RES_BYTES;
    const double cold_bytes = bytes * (double)COLD_MATRICES;
    printf("{\"test\":\"cpu_expert_gemv_avx2\",\"record\":\"layers.5.experts.0.down_proj\","
        "\"rows\":%u,\"cols\":%u,\"weights\":%llu,\"gvq_bytes\":%llu,"
        "\"res_bytes\":%llu,\"threads\":%d,\"iters\":%d,\"cold_matrices\":%u,"
        "\"scalar_matrix_ms\":%.3f,\"avx2_matrix_ms\":%.3f,\"avx2_matrix_ms_mean\":%.3f,"
        "\"avx2_matrix_gbs\":%.2f,\"exact_matrix_ms\":%.3f,"
        "\"cold_token_ms\":%.3f,\"cold_token_ms_mean\":%.3f,\"cold_token_gbs\":%.2f,"
        "\"scalar_cold_token_ms\":%.3f,"
        "\"cold_bytes\":%llu,\"max_rel_diff_hoisted\":%.3e,\"max_rel_diff_exact\":%.3e,"
        "\"gain_hoist\":true,\"fp16_books\":\"exact\"}\n",
        rows, cols, (unsigned long long)rows * cols,
        (unsigned long long)GVQ_BYTES, (unsigned long long)RES_BYTES,
        threads, iters, COLD_MATRICES,
        scalar_best,
        warm_best, warm_sum / iters,
        bytes / (warm_best * 1e6), exact_best,
        cold_best, cold_sum / iters, cold_bytes / (cold_best * 1e6),
        scalar_cold_best,
        (unsigned long long)cold_bytes, max_hoist, max_exact);

    free(x); free(xb); free(y_scalar); free(y_fast); free(y_exact);
    free(gvq_p); free(book_p); free(res_p); free(rbook_p);
    free(cold_g); free(cold_r); free(cg); free(cr); free(cy);
    return 0;
}
