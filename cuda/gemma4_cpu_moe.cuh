#ifndef DS4_GEMMA4_CPU_MOE_CUH
#define DS4_GEMMA4_CPU_MOE_CUH

/* Host routed-expert path for a discrete card (DS4_GEMMA4_CPU_MOE=1).
 *
 * The 4070 SUPER cannot hold the R13+8 expert records (about 7.5 GiB) beside
 * a 128K KV cache, so this path leaves them host-cold in the model mmap --
 * the state the discrete stream map already installs -- and runs the routed
 * projections on the CPU with the GPU kernels' arithmetic:
 *
 *   w  = bf16(gain * word)        cuda/gemma_vq_decode.cuh:57-93
 *   xv = bf16(x)
 *   acc = fmaf(w, xv, acc)
 *   mid = bf16(gelu(gate) * bf16(up))   cuda/gemma4_primitives.cuh:33-37,
 *                                       g4_gelu_kernel:182-187
 *   out = bf16(out + down * weight)     g4_moe_scatter_kernel:821-832
 *
 * The books are expanded to FP16 once per layer (every finite E4M3 value is
 * exact in FP16) and cached by record index. One persistent pool runs two
 * sweeps per layer: gate+up for every routed expert, then down; the GELU is
 * a third sweep. Chunks are 32 rows, stolen atomically.
 *
 * This is a measurement path, not a serving default: it exists so Gemma can
 * run on the 4070 at 128K without the SSD slot pool. */

#include <math.h>
#include <pthread.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <atomic>

/* The routed-expert kernels below are AVX2 + F16C intrinsics: the 4070's
 * host lane, where the expert records stay in the model mmap. The CUDA
 * translation unit includes this header on every target (the Spark is
 * aarch64), so the implementation is compiled only on x86 and the other
 * architectures get the explicit stubs at the end of the file. */
#if defined(__x86_64__) || defined(__i386__)
#define G4_CPU_MOE_AVX2 1
#include <immintrin.h>
#else
#define G4_CPU_MOE_AVX2 0
#endif

#if G4_CPU_MOE_AVX2

#define G4_CPU_CHUNK 32u
#define G4_CPU_GELU_CHUNK 64u
#define G4_CPU_BOOKS 128

/* ------------------------------------------------------------- arithmetic */

static float g4_cpu_e4m3_tab[256];
static int g4_cpu_e4m3_ready;

/* OCP E4M3, the format __nv_fp8_e4m3 holds and glmvq_e4m3 converts. */
static void g4_cpu_e4m3_init(void) {
    if (g4_cpu_e4m3_ready) { return; }
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
        g4_cpu_e4m3_tab[i] = sign ? -v : v;
    }
    g4_cpu_e4m3_ready = 1;
}

/* __float2bfloat16_rn, the rounding g4_vq_bf16 and g4_round use. */
static inline float g4_cpu_bf16(float f) {
    uint32_t u;
    memcpy(&u, &f, sizeof(u));
    u = (u + 0x7FFFu + ((u >> 16) & 1u)) & 0xFFFF0000u;
    memcpy(&f, &u, sizeof(f));
    return f;
}

static inline __m256 g4_cpu_bf16x8(__m256 w) {
    __m256i u = _mm256_castps_si256(w);
    const __m256i lsb = _mm256_and_si256(_mm256_srli_epi32(u, 16),
        _mm256_set1_epi32(1));
    u = _mm256_add_epi32(u, _mm256_add_epi32(_mm256_set1_epi32(0x7FFF), lsb));
    u = _mm256_and_si256(u, _mm256_set1_epi32((int)0xFFFF0000u));
    return _mm256_castsi256_ps(u);
}

/* cuda/gemma4_primitives.cuh:33-37: the reference's accurate tanh, evaluated
 * in double, one bf16 rounding on the input and one on the result. */
static inline float g4_cpu_gelu(float x) {
    const float v = g4_cpu_bf16(x);
    const float cube = 0.044715f * v * v * v;
    const float t = (float)tanh(0.7978845608028654 * (double)(v + cube));
    return g4_cpu_bf16(0.5f * v * (1.0f + t));
}

/* g4_gelu_kernel:186. */
static inline float g4_cpu_gelu_mid(float gate, float up) {
    return g4_cpu_bf16(g4_cpu_gelu(gate) * g4_cpu_bf16(up));
}

/* ------------------------------------------------------------ book cache */

struct g4_cpu_bookent {
    uint32_t index;
    uint16_t *fp16;
};

static struct g4_cpu_bookent g4_cpu_books[G4_CPU_BOOKS];
static int g4_cpu_nbooks;

static const uint16_t *g4_cpu_book(const void *map, const ds4_gvq_model *model,
                                   uint32_t index) {
    for (int i = 0; i < g4_cpu_nbooks; i++) {
        if (g4_cpu_books[i].index == index) { return g4_cpu_books[i].fp16; }
    }
    if (index >= model->records.size() || g4_cpu_nbooks == G4_CPU_BOOKS) {
        return NULL;
    }
    const ds4_gvq_record &r = model->records[index];
    if (r.type != GVQ_TYPE_BOOK || (r.bytes != 65536u && r.bytes != 2048u)) {
        return NULL;
    }
    const uint8_t *src = (const uint8_t *)map + r.offset;
    uint16_t *dst = (uint16_t *)malloc((size_t)r.bytes * sizeof(uint16_t));
    if (!dst) { return NULL; }
    for (uint64_t i = 0; i < r.bytes; i++) {
        dst[i] = _cvtss_sh(g4_cpu_e4m3_tab[src[i]], 0);
    }
    g4_cpu_books[g4_cpu_nbooks].index = index;
    g4_cpu_books[g4_cpu_nbooks].fp16 = dst;
    g4_cpu_nbooks++;
    return dst;
}

/* ------------------------------------------------------------ record parse */

struct g4_cpu_mat {
    const uint8_t *gain, *main, *plane, *res;
    const uint16_t *book, *rbook;
    uint32_t rows, cols;
    uint64_t nvec;
};

static inline uint32_t g4_cpu_u32(const uint8_t *p, size_t o) {
    uint32_t v;
    memcpy(&v, p + o, sizeof(v));
    return v;
}

static int g4_cpu_mat_parse(struct g4_cpu_mat *m, const void *map,
                            const ds4_gvq_model *model, uint32_t index) {
    memset(m, 0, sizeof(*m));
    if (!map || index >= model->records.size()) { return 0; }
    const ds4_gvq_record &r = model->records[index];
    if (r.type != GVQ_TYPE_VQ || r.bytes < 128u) { return 0; }
    const uint8_t *p = (const uint8_t *)map + r.offset;
    if (memcmp(p, "GVQ1", 4) || p[4] != 1 || p[5] != 0 || p[6] != 13 ||
        p[7] != 8 || g4_cpu_u32(p, 16) != 8192u || g4_cpu_u32(p, 20) != 1u) {
        return 0;
    }
    m->rows = g4_cpu_u32(p, 8);
    m->cols = g4_cpu_u32(p, 12);
    if (m->rows != r.rows || m->cols != r.cols || (m->cols % 64u)) { return 0; }
    if (g4_cpu_u32(p, 32) != 128u || g4_cpu_u32(p, 56) != r.bytes) { return 0; }
    m->gain = p + 128;
    m->main = p + g4_cpu_u32(p, 40);
    m->plane = p + g4_cpu_u32(p, 48);
    m->book = g4_cpu_book(map, model, r.book);
    if (!m->book) { return 0; }
    if (r.res == DS4_GVQ_NO_LINK || r.res >= model->records.size()) { return 0; }
    const ds4_gvq_record &rr = model->records[r.res];
    if (rr.type != GVQ_TYPE_RES || rr.bytes < 128u) { return 0; }
    const uint8_t *q = (const uint8_t *)map + rr.offset;
    if (memcmp(q, "GVR1", 4) || q[4] != 1 || q[5] != 0 || q[6] != 8 ||
        q[7] != 8 || g4_cpu_u32(q, 16) != 256u || g4_cpu_u32(q, 20) != 1u ||
        g4_cpu_u32(q, 32) != 128u || g4_cpu_u32(q, 56) != rr.bytes) {
        return 0;
    }
    m->res = q + 128;
    m->rbook = g4_cpu_book(map, model, rr.book);
    if (!m->rbook) { return 0; }
    m->nvec = (uint64_t)m->rows * (m->cols / 8u);
    return 1;
}

/* Four consecutive 13-bit indices from six bytes plus one plane byte. */
#define G4_CPU_INDEX4(ix, pb, sh, i0, i1, i2, i3) do {                     \
    (i0) = (uint32_t)(ix)[0] | (((uint32_t)(ix)[1] & 15u) << 8);           \
    (i1) = ((uint32_t)(ix)[1] >> 4) | ((uint32_t)(ix)[2] << 4);            \
    (i2) = (uint32_t)(ix)[3] | (((uint32_t)(ix)[4] & 15u) << 8);           \
    (i3) = ((uint32_t)(ix)[4] >> 4) | ((uint32_t)(ix)[5] << 4);            \
    (i0) |= ((uint32_t)((pb) >> (sh)) & 1u) << 12;                         \
    (i1) |= ((uint32_t)((pb) >> ((sh) + 1u)) & 1u) << 12;                  \
    (i2) |= ((uint32_t)((pb) >> ((sh) + 2u)) & 1u) << 12;                  \
    (i3) |= ((uint32_t)((pb) >> ((sh) + 3u)) & 1u) << 12;                  \
} while (0)

/* One row block of the R13+8 GEMV, the exact-rounding form. */
static void g4_cpu_gemv_rows(const struct g4_cpu_mat *m, const float *x, float *y,
                             uint32_t r0, uint32_t r1) {
    const uint32_t nvec = m->cols / 8u;
    for (uint32_t row = r0; row < r1; row++) {
        const float gain = _cvtsh_ss(*(const uint16_t *)(m->gain + 2u * row));
        const __m256 g8 = _mm256_set1_ps(gain);
        const uint64_t base = (uint64_t)row * nvec;
        const uint8_t *ix = m->main + (base >> 1) * 3u;
        const uint8_t *pl = m->plane + (base >> 3);
        const uint8_t *ri = m->res + base;
        const float *xp = x;
        __m256 a0 = _mm256_setzero_ps(), a1 = a0, a2 = a0, a3 = a0;
        for (uint32_t k = 0; k < nvec; k += 4u) {
            uint32_t i0, i1, i2, i3;
            G4_CPU_INDEX4(ix, pl[k >> 3], k & 7u, i0, i1, i2, i3);
            __m256 w0 = g4_cpu_bf16x8(_mm256_mul_ps(g8, _mm256_add_ps(
                _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(m->book + (size_t)i0 * 8))),
                _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(m->rbook + (size_t)ri[k] * 8))))));
            __m256 w1 = g4_cpu_bf16x8(_mm256_mul_ps(g8, _mm256_add_ps(
                _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(m->book + (size_t)i1 * 8))),
                _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(m->rbook + (size_t)ri[k + 1u] * 8))))));
            __m256 w2 = g4_cpu_bf16x8(_mm256_mul_ps(g8, _mm256_add_ps(
                _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(m->book + (size_t)i2 * 8))),
                _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(m->rbook + (size_t)ri[k + 2u] * 8))))));
            __m256 w3 = g4_cpu_bf16x8(_mm256_mul_ps(g8, _mm256_add_ps(
                _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(m->book + (size_t)i3 * 8))),
                _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(m->rbook + (size_t)ri[k + 3u] * 8))))));
            a0 = _mm256_fmadd_ps(w0, _mm256_loadu_ps(xp), a0);
            a1 = _mm256_fmadd_ps(w1, _mm256_loadu_ps(xp + 8), a1);
            a2 = _mm256_fmadd_ps(w2, _mm256_loadu_ps(xp + 16), a2);
            a3 = _mm256_fmadd_ps(w3, _mm256_loadu_ps(xp + 24), a3);
            ix += 6; xp += 32;
        }
        const __m256 s = _mm256_add_ps(_mm256_add_ps(a0, a1), _mm256_add_ps(a2, a3));
        const __m128 lo = _mm256_castps256_ps128(s);
        const __m128 hi = _mm256_extractf128_ps(s, 1);
        __m128 t = _mm_add_ps(lo, hi);
        t = _mm_hadd_ps(t, t);
        t = _mm_hadd_ps(t, t);
        y[row] = _mm_cvtss_f32(t);
    }
}

/* --------------------------------------------------------------- the pool */

struct g4_cpu_job {
    const struct g4_cpu_mat *m;
    const float *x;
    float *y;
};

struct g4_cpu_pool {
    int threads, started, stop;
    std::atomic<uint32_t> arrived, gen;
    int64_t next, total;
    uint32_t chunks;
    int phase;                      /* 0 gemv, 1 gelu */
    const struct g4_cpu_job *jobs;
    uint32_t njobs;
    float *gelu_gate, *gelu_up, *gelu_mid;
};

static struct g4_cpu_pool g4_cpu_pool;

/* Sense-reversing barrier over the pool plus the caller. */
static void g4_cpu_bar(struct g4_cpu_pool *p) {
    const uint32_t gen = p->gen.load(std::memory_order_acquire);
    if (p->arrived.fetch_add(1, std::memory_order_acq_rel) + 1u ==
        (uint32_t)p->threads) {
        p->arrived.store(0, std::memory_order_relaxed);
        p->gen.fetch_add(1, std::memory_order_release);
    } else {
        while (p->gen.load(std::memory_order_acquire) == gen) { _mm_pause(); }
    }
}

static void g4_cpu_work(struct g4_cpu_pool *p) {
    int64_t c;
    while ((c = __atomic_fetch_add(&p->next, 1, __ATOMIC_RELAXED)) < p->total) {
        const uint32_t j = (uint32_t)(c / p->chunks);
        const uint32_t k = (uint32_t)(c % p->chunks);
        if (p->phase == 0) {
            const struct g4_cpu_job *job = &p->jobs[j];
            uint32_t r0 = k * G4_CPU_CHUNK;
            uint32_t r1 = r0 + G4_CPU_CHUNK;
            if (r1 > job->m->rows) { r1 = job->m->rows; }
            g4_cpu_gemv_rows(job->m, job->x, job->y, r0, r1);
        } else {
            const uint32_t e = j;
            uint32_t i0 = k * G4_CPU_GELU_CHUNK;
            uint32_t i1 = i0 + G4_CPU_GELU_CHUNK;
            if (i1 > G4_FF) { i1 = G4_FF; }
            for (uint32_t i = i0; i < i1; i++) {
                p->gelu_mid[(size_t)e * G4_FF + i] =
                    g4_cpu_gelu_mid(p->gelu_gate[(size_t)e * G4_FF + i],
                                    p->gelu_up[(size_t)e * G4_FF + i]);
            }
        }
    }
}

static void *g4_cpu_thread(void *arg) {
    struct g4_cpu_pool *p = (struct g4_cpu_pool *)arg;
    for (;;) {
        g4_cpu_bar(p);
        if (p->stop) { break; }
        g4_cpu_work(p);
        g4_cpu_bar(p);
    }
    return NULL;
}

static int g4_cpu_pool_init(void) {
    if (g4_cpu_pool.started) { return 1; }
    const int t = (int)ds4_gemma4_knob_cpu_threads();
    g4_cpu_pool.threads = t;
    g4_cpu_pool.arrived.store(0);
    g4_cpu_pool.gen.store(0);
    g4_cpu_e4m3_init();
    for (int i = 0; i < t - 1; i++) {
        pthread_t tid;
        if (pthread_create(&tid, NULL, g4_cpu_thread, &g4_cpu_pool) != 0) {
            fprintf(stderr, "ds4: Gemma-4 CPU MoE pool could not start %d threads\n", t);
            return 0;
        }
        pthread_detach(tid);
    }
    g4_cpu_pool.started = 1;
    return 1;
}

static void g4_cpu_sweep(int phase, const struct g4_cpu_job *jobs, uint32_t njobs,
                         uint32_t chunks, float *gate, float *up, float *mid) {
    struct g4_cpu_pool *p = &g4_cpu_pool;
    p->phase = phase;
    p->jobs = jobs;
    p->njobs = njobs;
    p->chunks = chunks;
    p->gelu_gate = gate;
    p->gelu_up = up;
    p->gelu_mid = mid;
    p->next = 0;
    p->total = (int64_t)njobs * chunks;
    g4_cpu_bar(p);
    g4_cpu_work(p);
    g4_cpu_bar(p);
}

/* ------------------------------------------------------------- the layer */

static float g4_cpu_xr[G4_EMBED];
static float g4_cpu_g[G4_USED][G4_FF];
static float g4_cpu_u[G4_USED][G4_FF];
static float g4_cpu_mid[G4_USED][G4_FF];
static float g4_cpu_d[G4_USED][G4_EMBED];

static int g4_cpu_moe_layer(const void *map, const ds4_gvq_model *model,
                            const uint32_t *matrices, const int32_t *routes,
                            const float *weights, const float *x, float *out) {
    struct { uint32_t expert; float w; } slot[G4_USED];
    struct g4_cpu_mat gate[G4_USED], up[G4_USED], down[G4_USED];
    struct g4_cpu_job jobs[2u * G4_USED];

    for (uint32_t i = 0; i < G4_USED; i++) {
        slot[i].expert = (uint32_t)routes[i];
        slot[i].w = weights[i];
        if (slot[i].expert >= G4_EXPERTS) { return 0; }
    }
    /* The reference adds a token's experts in ascending id order (the grouped
     * list is expert-major), so the bf16 accumulator sees that sequence. */
    for (uint32_t i = 1; i < G4_USED; i++) {
        const auto tmp = slot[i];
        uint32_t j = i;
        while (j && slot[j - 1u].expert > tmp.expert) { slot[j] = slot[j - 1u]; j--; }
        slot[j] = tmp;
    }
    for (uint32_t i = 0; i < G4_USED; i++) {
        const uint32_t e = slot[i].expert;
        if (!g4_cpu_mat_parse(&gate[i], map, model, matrices[e]) ||
            !g4_cpu_mat_parse(&up[i], map, model, matrices[G4_EXPERTS + e]) ||
            !g4_cpu_mat_parse(&down[i], map, model, matrices[2u * G4_EXPERTS + e])) {
            return 0;
        }
        if (gate[i].rows != G4_FF || gate[i].cols != G4_EMBED ||
            up[i].rows != G4_FF || up[i].cols != G4_EMBED ||
            down[i].rows != G4_EMBED || down[i].cols != G4_FF) {
            return 0;
        }
    }
    for (uint32_t i = 0; i < G4_EMBED; i++) { g4_cpu_xr[i] = g4_cpu_bf16(x[i]); }
    for (uint32_t i = 0; i < G4_USED; i++) {
        jobs[i] = {&gate[i], g4_cpu_xr, g4_cpu_g[i]};
        jobs[G4_USED + i] = {&up[i], g4_cpu_xr, g4_cpu_u[i]};
    }
    g4_cpu_sweep(0, jobs, 2u * G4_USED,
                 (G4_FF + G4_CPU_CHUNK - 1u) / G4_CPU_CHUNK, NULL, NULL, NULL);
    g4_cpu_sweep(1, NULL, G4_USED,
                 (G4_FF + G4_CPU_GELU_CHUNK - 1u) / G4_CPU_GELU_CHUNK,
                 &g4_cpu_g[0][0], &g4_cpu_u[0][0], &g4_cpu_mid[0][0]);
    for (uint32_t i = 0; i < G4_USED; i++) {
        jobs[i] = {&down[i], g4_cpu_mid[i], g4_cpu_d[i]};
    }
    g4_cpu_sweep(0, jobs, G4_USED,
                 (G4_EMBED + G4_CPU_CHUNK - 1u) / G4_CPU_CHUNK, NULL, NULL, NULL);
    for (uint32_t i = 0; i < G4_EMBED; i++) { out[i] = 0.0f; }
    for (uint32_t s = 0; s < G4_USED; s++) {
        const float w = slot[s].w;
        const float *d = g4_cpu_d[s];
        for (uint32_t i = 0; i < G4_EMBED; i++) {
            out[i] = g4_cpu_bf16(out[i] + d[i] * w);
        }
    }
    return 1;
}

/* One expert, one row: the down output of the host pipeline, for the
 * cache-miss fallback (the graph adds it with the ordered scatter). */
static int g4_cpu_moe_expert(const void *map, const ds4_gvq_model *model,
                             const uint32_t *matrices, uint32_t expert,
                             const float *x, float *out) {
    struct g4_cpu_mat gate, up, down;
    struct g4_cpu_job jobs[2], dj;
    if (!map || !model || !matrices || !x || !out || expert >= G4_EXPERTS) { return 0; }
    if (!g4_cpu_pool_init()) { return 0; }
    if (!g4_cpu_mat_parse(&gate, map, model, matrices[expert]) ||
        !g4_cpu_mat_parse(&up, map, model, matrices[G4_EXPERTS + expert]) ||
        !g4_cpu_mat_parse(&down, map, model, matrices[2u * G4_EXPERTS + expert])) {
        return 0;
    }
    if (gate.rows != G4_FF || gate.cols != G4_EMBED ||
        up.rows != G4_FF || up.cols != G4_EMBED ||
        down.rows != G4_EMBED || down.cols != G4_FF) {
        return 0;
    }
    for (uint32_t i = 0; i < G4_EMBED; i++) { g4_cpu_xr[i] = g4_cpu_bf16(x[i]); }
    jobs[0] = {&gate, g4_cpu_xr, g4_cpu_g[0]};
    jobs[1] = {&up, g4_cpu_xr, g4_cpu_u[0]};
    g4_cpu_sweep(0, jobs, 2u, (G4_FF + G4_CPU_CHUNK - 1u) / G4_CPU_CHUNK, NULL, NULL, NULL);
    g4_cpu_sweep(1, NULL, 1u, (G4_FF + G4_CPU_GELU_CHUNK - 1u) / G4_CPU_GELU_CHUNK,
                 g4_cpu_g[0], g4_cpu_u[0], g4_cpu_mid[0]);
    dj = {&down, g4_cpu_mid[0], g4_cpu_d[0]};
    g4_cpu_sweep(0, &dj, 1u, (G4_EMBED + G4_CPU_CHUNK - 1u) / G4_CPU_CHUNK, NULL, NULL, NULL);
    memcpy(out, g4_cpu_d[0], (size_t)G4_EMBED * sizeof(float));
    return 1;
}

/* Decode entry: rows tokens of G4_USED routes each, all through the host. */
static int g4_cpu_moe_run_tensors(const void *map, const ds4_gvq_model *model,
                                  ds4_gpu_tensor *out, const ds4_gpu_tensor *input,
                                  const ds4_gpu_tensor *weights, const uint32_t *matrices,
                                  const int32_t *routes, uint32_t rows) {
    if (!map || !model || !out || !input || !weights || !matrices || !routes || !rows) {
        return 0;
    }
    if (!g4_cpu_pool_init()) { return 0; }
    const uint64_t in_bytes = (uint64_t)rows * G4_EMBED * sizeof(float);
    const uint64_t w_bytes = (uint64_t)rows * G4_USED * sizeof(float);
    if (input->bytes < in_bytes || weights->bytes < w_bytes || out->bytes < in_bytes) {
        return 0;
    }
    static float *xin, *wbuf, *xout;
    static uint32_t cap;
    if (rows > cap) {
        free(xin); free(wbuf); free(xout);
        xin = (float *)malloc((size_t)in_bytes);
        wbuf = (float *)malloc((size_t)w_bytes);
        xout = (float *)malloc((size_t)in_bytes);
        cap = rows;
        if (!xin || !wbuf || !xout) { cap = 0; return 0; }
    }
    cudaStream_t s = ds4_current_stream();
    if (cudaStreamSynchronize(s) != cudaSuccess) { return 0; }
    if (cudaMemcpyAsync(xin, input->ptr, (size_t)in_bytes, cudaMemcpyDeviceToHost, s) !=
            cudaSuccess ||
        cudaMemcpyAsync(wbuf, weights->ptr, (size_t)w_bytes, cudaMemcpyDeviceToHost, s) !=
            cudaSuccess ||
        cudaStreamSynchronize(s) != cudaSuccess) {
        return 0;
    }
    for (uint32_t t = 0; t < rows; t++) {
        if (!g4_cpu_moe_layer(map, model, matrices, routes + (uint64_t)t * G4_USED,
                              wbuf + (uint64_t)t * G4_USED,
                              xin + (uint64_t)t * G4_EMBED,
                              xout + (uint64_t)t * G4_EMBED)) {
            return 0;
        }
    }
    if (getenv("DS4_GEMMA4_CPU_MOE_DEBUG")) {
        static int reported;        if (!reported) {
            reported = 1;
            const ds4_gvq_record &rg = model->records[matrices[(uint32_t)routes[0]]];
            fprintf(stderr, "ds4: Gemma-4 CPU MoE active: rows %u threads %d "
                "route0 %d w0 %.4f gate off %llu bytes %llu rows %u cols %u "
                "x %.4f %.4f %.4f out %.4f %.4f %.4f\n",
                rows, g4_cpu_pool.threads, routes[0], (double)wbuf[0],
                (unsigned long long)rg.offset, (unsigned long long)rg.bytes,
                rg.rows, rg.cols, (double)xin[0], (double)xin[1], (double)xin[2],
                (double)xout[0], (double)xout[1], (double)xout[2]);
        }
    }
    if (getenv("DS4_GEMMA4_CPU_MOE_DUMP")) {
        static int dumped;
        if (!dumped) {
            dumped = 1;
            FILE *fp = fopen("/tmp/g4-cpu-out.bin", "wb");
            if (fp) {
                fwrite(xout, sizeof(float), G4_EMBED, fp);
                fclose(fp);
            }
        }
    }
    if (cudaMemcpyAsync(out->ptr, xout, (size_t)in_bytes, cudaMemcpyHostToDevice, s) !=
        cudaSuccess) {
        return 0;
    }
    return cudaStreamSynchronize(s) == cudaSuccess;
}

#else /* !G4_CPU_MOE_AVX2 */

/* Explicit unavailable path: a graph that asks for the host experts fails
 * loudly instead of silently computing them somewhere else. */
static int g4_cpu_moe_unavailable(void) {
    static int reported;
    if (!reported) {
        reported = 1;
        fprintf(stderr, "ds4: Gemma-4 CPU MoE is an x86 AVX2 path; this build "
            "is not x86, so DS4_GEMMA4_CPU_MOE is unavailable here\n");
    }
    return 0;
}

static int g4_cpu_moe_expert(const void *map, const ds4_gvq_model *model,
                             const uint32_t *matrices, uint32_t expert,
                             const float *x, float *out) {
    (void)map; (void)model; (void)matrices; (void)expert; (void)x; (void)out;
    return g4_cpu_moe_unavailable();
}

static int g4_cpu_moe_run_tensors(const void *map, const ds4_gvq_model *model,
                                  ds4_gpu_tensor *out, const ds4_gpu_tensor *input,
                                  const ds4_gpu_tensor *weights, const uint32_t *matrices,
                                  const int32_t *routes, uint32_t rows) {
    (void)map; (void)model; (void)out; (void)input; (void)weights;
    (void)matrices; (void)routes; (void)rows;
    return g4_cpu_moe_unavailable();
}

#endif /* G4_CPU_MOE_AVX2 */

#endif
