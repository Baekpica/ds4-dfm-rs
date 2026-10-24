/* h3_gvq.c — the rung A / A+8 encoder for MiniMax-H3, and the tool's own gates.
 *
 * The format lives in h3_gvq.h (GVQ1 v1 + GVR8, field by field). This file is
 * the encoder: the method is /data/YoungAi/docs/TECHNICAL.md section 4.1, and
 * the parameters are /data/minimax-m3.1-vq-recipe/recipe.json, i.e. what the
 * M3.1 release actually shipped:
 *
 *   init             equidistant         no random source (TECHNICAL.md 4.1.3)
 *   pool             stratified          systematic sample of the vector list
 *   lloyd_iterations 32
 *   mode             imatrix             column weighting; uniform here, which
 *                                        is the reference's own zero-corpus
 *                                        mode (TECHNICAL.md 4.1, "Zero corpus")
 *   gain_refit       ls                  least squares, against stored indices
 *   codewords        E4M3FN, rounded to stored precision *before* the final
 *                    assignment (TECHNICAL.md 4.1.4)
 *   pool density     >= 1000 vectors per codeword; "a pool 4x thinner lost
 *                    0.96% on 13-bit layers" (TECHNICAL.md 4.1)
 *   residual         GVR8 K=256, 8 bits, 32 iterations, equidistant init,
 *                    target w/g - C1[i1] against the stored gain and index,
 *                    weights g^2 x column profile, rounding FP32->F16->E4M3FN
 *                    (recipe.json "residual", verbatim)
 *
 * Three assignment passes, each with its own arithmetic:
 *   Lloyd    pool vectors (already row-RMS-normalized) against the codebook
 *   initial  the whole matrix against rms[row] * codebook -- the same quantity
 *            the pool was trained on, used only to give the gain refit indices
 *   final    the whole matrix against gain[row] * codebook, where gain is the
 *            least-squares refit rounded to F16: exactly what the reader
 *            reconstructs, so the reported error is the delivered error
 *
 * Determinism. Per-vector work is independent, so no result depends on the
 * other threads. The centroid update reduces per-thread double accumulators in
 * fixed thread order, so the sum is identical at any --threads value; selftest
 * asserts byte equality across thread counts.
 *
 * Usage:
 *   h3_gvq selftest [dir]
 *   h3_gvq encode <safetensors-dir> <tensor> <prefix> [opts]
 *   h3_gvq check  <prefix> [--src <safetensors-dir> --name <tensor>]
 *
 * encode writes <prefix>.gvq, <prefix>.book, [<prefix>.gvr, <prefix>.rbook],
 * <prefix>.report. check reads them back through the strict reader.
 */
#define _POSIX_C_SOURCE 200809L
#include <errno.h>
#include <math.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#include "ds4_st41.h"      /* multi-shard safetensors index (locate, mmap) */
#include "ds4_float.h"     /* ds4_bf16_to_f32 */
#include "onebit_quant.h"  /* go1b_fp32_to_fp16 / go1b_fp16_to_fp32 */
#include "h3_gvq.h"        /* the format, both directions */

#define DIM ((int)GVQ_VECTOR)

static double now_s(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec + (double)t.tv_nsec * 1e-9;
}

static int nthreads_default(void) {
    const long n = sysconf(_SC_NPROCESSORS_ONLN);
    return (n > 0 && n < 256) ? (int)n : 8;
}

/* ---- one-job pthread pool ------------------------------------------------ */
typedef void (*par_fn)(void *ud, int t, int nt);
typedef struct { par_fn fn; void *ud; int t, nt; } par_job;

static void *par_tramp(void *arg) {
    par_job *j = (par_job *)arg;
    j->fn(j->ud, j->t, j->nt);
    return NULL;
}

static void par_run(par_fn fn, void *ud, int nt) {
    if (nt <= 1) { fn(ud, 0, 1); return; }
    pthread_t *th = (pthread_t *)calloc((size_t)nt, sizeof(pthread_t));
    par_job *jobs = (par_job *)calloc((size_t)nt, sizeof(par_job));
    if (!th || !jobs) { free(th); free(jobs); fn(ud, 0, 1); return; }
    for (int t = 0; t < nt; t++) {
        jobs[t].fn = fn; jobs[t].ud = ud; jobs[t].t = t; jobs[t].nt = nt;
        if (pthread_create(&th[t], NULL, par_tramp, &jobs[t]) != 0) {
            fn(ud, t, nt);
            th[t] = (pthread_t)0;
        }
    }
    for (int t = 0; t < nt; t++) { if (th[t]) { pthread_join(th[t], NULL); } }
    free(th);
    free(jobs);
}

/* ---- options and results ------------------------------------------------ */
typedef struct {
    int k;                /* 8192 (rung A) or 4096 (rung B) */
    int iters;            /* Lloyd rounds: 32 */
    int r_iters;          /* residual Lloyd rounds: 32 */
    long pool_per_code;   /* vectors per codeword in the pool: 1000 */
    int threads;
    int residual;
    uint32_t book_id;
} enc_opts;

typedef struct {
    double pool, init, lloyd, book, gain, final, residual, write, metrics, total;
    double assign_s, assign_vec;
} enc_times;

typedef struct {
    uint8_t *book;        /* k*dim E4M3 */
    float *bookf;         /* k*dim F32, decoded back from book */
    uint16_t *gain;       /* rows, F16 */
    float *gainf;         /* rows, F32 as stored */
    uint32_t *idx;
    uint8_t *rbook;
    uint8_t *ridx;
    long nidx;
} enc_out;

static void enc_out_free(enc_out *o) {
    free(o->book); free(o->bookf); free(o->gain); free(o->gainf);
    free(o->idx); free(o->rbook); free(o->ridx);
    memset(o, 0, sizeof *o);
}

/* ---- passes ------------------------------------------------------------- */

/* Row RMS. A zero row gets 1.0 so nothing divides by zero; its LS gain lands
 * on +0, which FORMAT.md defines as the zero-row encoding. */
typedef struct { const float *W; int rows, cols; float *g0; } rms_job;

static void rms_rows(void *ud, int t, int nt) {
    rms_job *j = (rms_job *)ud;
    for (int r = t; r < j->rows; r += nt) {
        const float *w = j->W + (size_t)r * (size_t)j->cols;
        double s = 0;
        for (int c = 0; c < j->cols; c++) { s += (double)w[c] * w[c]; }
        const double ms = s / (double)j->cols;
        j->g0[r] = ms > 0 ? (float)sqrt(ms) : 1.0f;
    }
}

/* The stratified pool: every stratum of the flat vector list contributes one
 * vector, so the sample is spread over every row. pos[] keeps the global
 * position, which the residual stage reuses to read the stored index. */
typedef struct {
    const float *W; const float *g0;
    int cols; long nv, np;
    long *pos;
    float *PV;
} pool_job;

static void pool_build(void *ud) {
    pool_job *j = (pool_job *)ud;
    const int vpr = j->cols / DIM;
    for (long i = 0; i < j->np; i++) {
        const long p = (long)((long long)i * j->nv / j->np);
        const int r = (int)(p / vpr), jj = (int)(p % vpr);
        const float *w = j->W + (size_t)r * (size_t)j->cols + (size_t)jj * DIM;
        const float g = j->g0[r];
        j->pos[i] = p;
        for (int d = 0; d < DIM; d++) { j->PV[(size_t)i * DIM + d] = w[d] / g; }
    }
}

/* d(v, c) = sum_d colw[base+d] * (v_d - c_d)^2, first strict minimum wins a tie
 * (the tie rule of vq_qc.h:59-79, so a tie breaks on codeword index, never on
 * thread order). Direct form, not the expanded c2 - 2 v.c: with a per-column
 * weight profile the expanded form's c2 term depends on the vector's column
 * base, so the direct form is the one that stays correct when the imatrix
 * profile arrives. */
static inline int argmin8(const float *v, const float *colw, const float *bookf, int k) {
    const double w0 = colw[0], w1 = colw[1], w2 = colw[2], w3 = colw[3];
    const double w4 = colw[4], w5 = colw[5], w6 = colw[6], w7 = colw[7];
    const double v0 = v[0], v1 = v[1], v2 = v[2], v3 = v[3];
    const double v4 = v[4], v5 = v[5], v6 = v[6], v7 = v[7];
    double best = 1e300;
    int bi = 0;
    for (int c = 0; c < k; c++) {
        const float *cc = bookf + (size_t)c * DIM;
        const double d0 = v0 - (double)cc[0], d1 = v1 - (double)cc[1];
        const double d2 = v2 - (double)cc[2], d3 = v3 - (double)cc[3];
        const double d4 = v4 - (double)cc[4], d5 = v5 - (double)cc[5];
        const double d6 = v6 - (double)cc[6], d7 = v7 - (double)cc[7];
        const double s = w0 * d0 * d0 + w1 * d1 * d1 + w2 * d2 * d2 + w3 * d3 * d3 +
                         w4 * d4 * d4 + w5 * d5 * d5 + w6 * d6 * d6 + w7 * d7 * d7;
        if (s < best) { best = s; bi = c; }
    }
    return bi;
}

typedef struct {
    const float *V; long nv;
    const float *bookf; int k;
    const float *colw; const int *colbase;
    uint32_t *idx;
} pool_assign_job;

static void pool_assign(void *ud, int t, int nt) {
    pool_assign_job *j = (pool_assign_job *)ud;
    for (long i = t; i < j->nv; i += nt) {
        j->idx[i] = (uint32_t)argmin8(j->V + (size_t)i * DIM, j->colw + j->colbase[i],
                                     j->bookf, j->k);
    }
}

/* Whole-matrix assignment against gain[row] * codeword. */
typedef struct {
    const float *W; const float *gainf;
    int rows, cols;
    const float *bookf; int k;
    const float *colw;
    uint32_t *idx;
} final_job;

static void final_assign(void *ud, int t, int nt) {
    final_job *j = (final_job *)ud;
    const int vpr = j->cols / DIM;
    for (int r = t; r < j->rows; r += nt) {
        const float *wrow = j->W + (size_t)r * (size_t)j->cols;
        const float g = j->gainf[r];
        uint32_t *orow = j->idx + (size_t)r * vpr;
        for (int jj = 0; jj < vpr; jj++) {
            const float *seg = wrow + (size_t)jj * DIM;
            const float *cw = j->colw + (size_t)jj * DIM;
            double best = 1e300;
            int bi = 0;
            for (int c = 0; c < j->k; c++) {
                const float *cc = j->bookf + (size_t)c * DIM;
                double s = 0;
                for (int d = 0; d < DIM; d++) {
                    const double dd = (double)seg[d] - (double)g * (double)cc[d];
                    s += (double)cw[d] * dd * dd;
                }
                if (s < best) { best = s; bi = c; }
            }
            orow[jj] = (uint32_t)bi;
        }
    }
}

/* Lloyd M-step. */
typedef struct {
    const float *V; long nv;
    const uint32_t *idx;
    const float *uw;
    int k, nt;
    double **acc, **cnt;
    float *bookf;
} update_job;

static void update_run(void *ud, int t, int nt) {
    update_job *j = (update_job *)ud;
    double *acc = j->acc[t], *cnt = j->cnt[t];
    memset(acc, 0, sizeof(double) * (size_t)j->k * DIM);
    memset(cnt, 0, sizeof(double) * (size_t)j->k);
    for (long i = t; i < j->nv; i += nt) {
        const uint32_t c = j->idx[i];
        const double u = j->uw ? (double)j->uw[i] : 1.0;
        const float *v = j->V + (size_t)i * DIM;
        double *a = acc + (size_t)c * DIM;
        for (int d = 0; d < DIM; d++) { a[d] += u * (double)v[d]; }
        cnt[c] += u;
    }
}

static void update_reduce(update_job *j) {
    for (int c = 0; c < j->k; c++) {
        double n = 0;
        for (int t = 0; t < j->nt; t++) { n += j->cnt[t][c]; }
        if (n <= 0) { continue; }   /* empty codeword keeps its centroid */
        for (int d = 0; d < DIM; d++) {
            double s = 0;
            for (int t = 0; t < j->nt; t++) { s += j->acc[t][(size_t)c * DIM + d]; }
            j->bookf[(size_t)c * DIM + d] = (float)(s / n);
        }
    }
}

/* Least-squares gain: g_r = sum w*c / sum c^2 against the stored codebook. */
typedef struct {
    const float *W; const float *bookf; const uint32_t *idx;
    int rows, cols;
    const float *colw;
    float *gainf; uint16_t *gain16;
} gain_job;

static void gain_rows(void *ud, int t, int nt) {
    gain_job *j = (gain_job *)ud;
    const int vpr = j->cols / DIM;
    for (int r = t; r < j->rows; r += nt) {
        const float *wrow = j->W + (size_t)r * (size_t)j->cols;
        double num = 0, den = 0;
        for (int jj = 0; jj < vpr; jj++) {
            const float *seg = wrow + (size_t)jj * DIM;
            const float *cw = j->colw + (size_t)jj * DIM;
            const float *cc = j->bookf + (size_t)j->idx[(size_t)r * vpr + jj] * DIM;
            for (int d = 0; d < DIM; d++) {
                num += (double)cw[d] * (double)seg[d] * (double)cc[d];
                den += (double)cw[d] * (double)cc[d] * (double)cc[d];
            }
        }
        double g = den > 0 ? num / den : 0.0;
        if (!isfinite(g) || g < 0) { g = 0; }
        if (g > 65504.0) { g = 65504.0; }
        const uint16_t h = go1b_fp32_to_fp16((float)g);
        j->gain16[r] = h;
        j->gainf[r] = go1b_fp16_to_fp32(h);
    }
}

/* Residual target: w/g - C1[i1], the stored gain and the stored index. */
typedef struct { const float *W; const float *gainf; const uint8_t *c1; const uint32_t *idx;
                 int cols; long nv; const long *pos; float *T; float *uw; } target_job;

static void target_build(void *ud) {
    target_job *j = (target_job *)ud;
    const int vpr = j->cols / DIM;
    for (long i = 0; i < j->nv; i++) {
        const long p = j->pos[i];
        const int r = (int)(p / vpr);
        const float g = j->gainf[r];
        const float *w = j->W + (size_t)p * DIM;
        const uint8_t *c1 = j->c1 + (size_t)j->idx[(size_t)p] * DIM;
        float *t = j->T + (size_t)i * DIM;
        for (int d = 0; d < DIM; d++) {
            t[d] = g > 0 ? w[d] / g - ds4_e4m3fn_to_f32(c1[d]) : 0.0f;
        }
        j->uw[i] = g * g;
    }
}

/* Whole-matrix residual assignment; stage 1 is fixed. */
typedef struct {
    const float *W; const float *gainf;
    int rows, cols;
    const uint8_t *c1; const uint32_t *idx1;
    const float *rbookf; int k;
    const float *colw;
    uint8_t *ridx;
} res_final_job;

static void res_final_assign(void *ud, int t, int nt) {
    res_final_job *j = (res_final_job *)ud;
    const int vpr = j->cols / DIM;
    for (int r = t; r < j->rows; r += nt) {
        const float *wrow = j->W + (size_t)r * (size_t)j->cols;
        const float g = j->gainf[r];
        for (int jj = 0; jj < vpr; jj++) {
            const size_t p = (size_t)r * vpr + jj;
            const uint8_t *c1 = j->c1 + (size_t)j->idx1[p] * DIM;
            const float *seg = wrow + (size_t)jj * DIM;
            const float *cw = j->colw + (size_t)jj * DIM;
            float tt[DIM];
            for (int d = 0; d < DIM; d++) {
                tt[d] = g > 0 ? seg[d] / g - ds4_e4m3fn_to_f32(c1[d]) : 0.0f;
            }
            j->ridx[p] = (uint8_t)argmin8(tt, cw, j->rbookf, j->k);
        }
    }
}

/* ---- encoder ------------------------------------------------------------ */
static int acc_alloc(update_job *j, int nt, int k) {
    j->acc = (double **)calloc((size_t)nt, sizeof(double *));
    j->cnt = (double **)calloc((size_t)nt, sizeof(double *));
    if (!j->acc || !j->cnt) { return -1; }
    for (int t = 0; t < nt; t++) {
        j->acc[t] = (double *)malloc(sizeof(double) * (size_t)k * DIM);
        j->cnt[t] = (double *)malloc(sizeof(double) * (size_t)k);
        if (!j->acc[t] || !j->cnt[t]) { return -1; }
    }
    return 0;
}
static void acc_free(update_job *j, int nt) {
    if (j->acc) { for (int t = 0; t < nt; t++) { free(j->acc[t]); } }
    if (j->cnt) { for (int t = 0; t < nt; t++) { free(j->cnt[t]); } }
    free(j->acc); free(j->cnt);
    j->acc = NULL; j->cnt = NULL;
}

/* Encode one matrix. Everything the caller needs to write a payload. */
static int encode_matrix(const float *W, int rows, int cols, const char *name,
                         const enc_opts *o, enc_out *out, enc_times *tm, FILE *log) {
    memset(tm, 0, sizeof *tm);
    memset(out, 0, sizeof *out);
    if (!gvq_shape_ok(rows, cols)) {
        fprintf(stderr, "%s: %dx%d is not a GVQ1 shape (cols must be a multiple of 64)\n",
                name, rows, cols);
        return -1;
    }
    const int vpr = cols / DIM;
    const long nv = (long)rows * (long)vpr;
    const int k = o->k;
    const int nt = o->threads > 0 ? o->threads : nthreads_default();
    const double t_start = now_s();
    int rc = -1;
    float *g0 = NULL, *colw = NULL, *PV = NULL;
    long *pos = NULL;
    int *colbase = NULL;
    uint32_t *pidx = NULL;
    update_job uj;
    memset(&uj, 0, sizeof uj);

    out->book = (uint8_t *)malloc(gvq_book_bytes(k));
    out->bookf = (float *)malloc(sizeof(float) * (size_t)k * DIM);
    out->gain = (uint16_t *)malloc(sizeof(uint16_t) * (size_t)rows);
    out->gainf = (float *)malloc(sizeof(float) * (size_t)rows);
    out->idx = (uint32_t *)malloc(sizeof(uint32_t) * (size_t)nv);
    g0 = (float *)malloc(sizeof(float) * (size_t)rows);
    colw = (float *)malloc(sizeof(float) * (size_t)cols);
    if (!out->book || !out->bookf || !out->gain || !out->gainf || !out->idx || !g0 || !colw) {
        fprintf(stderr, "%s: out of memory\n", name);
        goto done;
    }
    for (int c = 0; c < cols; c++) { colw[c] = 1.0f; }   /* zero corpus */

    /* 1. row RMS */
    double t0 = now_s();
    { rms_job rj = { W, rows, cols, g0 }; par_run(rms_rows, &rj, nt); }
    tm->pool = now_s() - t0;

    /* 2. the stratified pool */
    long np = (long)k * o->pool_per_code;
    if (np > nv) { np = nv; }
    if (np < (long)k) {
        fprintf(stderr, "%s: %ld vectors cannot train %d codewords\n", name, nv, k);
        goto done;
    }
    PV = (float *)malloc(sizeof(float) * (size_t)np * DIM);
    pos = (long *)malloc(sizeof(long) * (size_t)np);
    colbase = (int *)malloc(sizeof(int) * (size_t)np);
    if (!PV || !pos || !colbase) { fprintf(stderr, "%s: out of memory for the pool\n", name); goto done; }
    t0 = now_s();
    { pool_job pj = { W, g0, cols, nv, np, pos, PV }; pool_build(&pj); }
    for (long i = 0; i < np; i++) { colbase[i] = (int)((pos[i] % vpr) * DIM); }
    tm->pool += now_s() - t0;
    if (log) {
        fprintf(log, "pool: %ld sampled of %ld vectors (%.1f per codeword); k=%d dim=%d iters=%d threads=%d\n",
                np, nv, (double)np / (double)k, k, DIM, o->iters, nt);
    }

    /* 3. equidistant init */
    t0 = now_s();
    for (int c = 0; c < k; c++) {
        const long i = (long)((long long)c * np / k);
        memcpy(out->bookf + (size_t)c * DIM, PV + (size_t)i * DIM, sizeof(float) * DIM);
    }
    tm->init = now_s() - t0;

    /* 4. Lloyd on the pool */
    pidx = (uint32_t *)malloc(sizeof(uint32_t) * (size_t)np);
    if (!pidx || acc_alloc(&uj, nt, k)) { fprintf(stderr, "%s: out of memory\n", name); goto done; }
    t0 = now_s();
    for (int it = 0; it < o->iters; it++) {
        const double ta = now_s();
        pool_assign_job pj = { PV, np, out->bookf, k, colw, colbase, pidx };
        par_run(pool_assign, &pj, nt);
        tm->assign_s += now_s() - ta;
        tm->assign_vec += (double)np;
        uj.V = PV; uj.nv = np; uj.idx = pidx; uj.uw = NULL; uj.k = k; uj.nt = nt;
        uj.bookf = out->bookf;
        par_run(update_run, &uj, nt);
        update_reduce(&uj);
    }
    tm->lloyd = now_s() - t0;

    /* 5. codewords to E4M3, and use the stored values from here on */
    t0 = now_s();
    for (int c = 0; c < k; c++) {
        for (int d = 0; d < DIM; d++) {
            const uint8_t b = ds4_e4m3fn_f32_to_byte(out->bookf[(size_t)c * DIM + d]);
            out->book[(size_t)c * DIM + d] = b;
            out->bookf[(size_t)c * DIM + d] = ds4_e4m3fn_to_f32(b);
        }
    }
    tm->book = now_s() - t0;

    /* 6. provisional assignment against the row RMS, then the LS gain refit */
    t0 = now_s();
    {
        final_job fj = { W, g0, rows, cols, out->bookf, k, colw, out->idx };
        const double ta = now_s();
        par_run(final_assign, &fj, nt);
        tm->assign_s += now_s() - ta;
        tm->assign_vec += (double)nv;
        gain_job gj = { W, out->bookf, out->idx, rows, cols, colw, out->gainf, out->gain };
        par_run(gain_rows, &gj, nt);
    }
    tm->gain = now_s() - t0;

    /* 7. final assignment against the stored gain and codebook */
    t0 = now_s();
    {
        final_job fj = { W, out->gainf, rows, cols, out->bookf, k, colw, out->idx };
        const double ta = now_s();
        par_run(final_assign, &fj, nt);
        tm->assign_s += now_s() - ta;
        tm->assign_vec += (double)nv;
    }
    tm->final = now_s() - t0;

    /* 8. GVR8 */
    if (o->residual) {
        t0 = now_s();
        out->rbook = (uint8_t *)malloc(gvq_book_bytes((int)GVR_K));
        out->ridx = (uint8_t *)malloc((size_t)nv);
        float *rbookf = (float *)malloc(sizeof(float) * (size_t)GVR_K * DIM);
        float *T = (float *)malloc(sizeof(float) * (size_t)np * DIM);
        float *uw = (float *)malloc(sizeof(float) * (size_t)np);
        uint32_t *ridx_pool = (uint32_t *)malloc(sizeof(uint32_t) * (size_t)np);
        update_job ruj;
        memset(&ruj, 0, sizeof ruj);
        if (!out->rbook || !out->ridx || !rbookf || !T || !uw || !ridx_pool ||
            acc_alloc(&ruj, nt, (int)GVR_K)) {
            fprintf(stderr, "%s: out of memory for the residual stage\n", name);
            free(rbookf); free(T); free(uw); free(ridx_pool);
            acc_free(&ruj, nt);
            goto done;
        }
        { target_job tj = { W, out->gainf, out->book, out->idx, cols, np, pos, T, uw };
          target_build(&tj); }
        for (int c = 0; c < (int)GVR_K; c++) {
            const long i = (long)((long long)c * np / (long)GVR_K);
            memcpy(rbookf + (size_t)c * DIM, T + (size_t)i * DIM, sizeof(float) * DIM);
        }
        for (int it = 0; it < o->r_iters; it++) {
            pool_assign_job pj = { T, np, rbookf, (int)GVR_K, colw, colbase, ridx_pool };
            const double ta = now_s();
            par_run(pool_assign, &pj, nt);
            tm->assign_s += now_s() - ta;
            tm->assign_vec += (double)np;
            ruj.V = T; ruj.nv = np; ruj.idx = ridx_pool; ruj.uw = uw; ruj.k = (int)GVR_K;
            ruj.nt = nt; ruj.bookf = rbookf;
            par_run(update_run, &ruj, nt);
            update_reduce(&ruj);
        }
        /* rounding chain FP32 -> F16 -> E4M3FN (recipe.json, verbatim) */
        for (int c = 0; c < (int)GVR_K; c++) {
            for (int d = 0; d < DIM; d++) {
                const float f16 = go1b_fp16_to_fp32(go1b_fp32_to_fp16(rbookf[(size_t)c * DIM + d]));
                const uint8_t b = ds4_e4m3fn_f32_to_byte(f16);
                out->rbook[(size_t)c * DIM + d] = b;
                rbookf[(size_t)c * DIM + d] = ds4_e4m3fn_to_f32(b);
            }
        }
        { res_final_job rj = { W, out->gainf, rows, cols, out->book, out->idx, rbookf,
                               (int)GVR_K, colw, out->ridx };
          const double ta = now_s();
          par_run(res_final_assign, &rj, nt);
          tm->assign_s += now_s() - ta;
          tm->assign_vec += (double)nv; }
        acc_free(&ruj, nt);
        free(rbookf); free(T); free(uw); free(ridx_pool);
        tm->residual = now_s() - t0;
    }

    out->nidx = nv;
    tm->total = now_s() - t_start;
    rc = 0;

done:
    acc_free(&uj, nt);
    free(pidx); free(PV); free(pos); free(colbase); free(g0); free(colw);
    if (rc) { enc_out_free(out); }
    return rc;
}

/* ---- file helpers ------------------------------------------------------- */
static int write_file(const char *path, const void *p, size_t n) {
    FILE *f = fopen(path, "wb");
    if (!f) { fprintf(stderr, "cannot write %s: %s\n", path, strerror(errno)); return -1; }
    const size_t w = n ? fwrite(p, 1, n, f) : 0;
    const int bad = (fclose(f) != 0) || (w != n);
    if (bad) { fprintf(stderr, "short write to %s\n", path); return -1; }
    return 0;
}

static uint8_t *read_file(const char *path, size_t *n) {
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "cannot read %s: %s\n", path, strerror(errno)); return NULL; }
    if (fseek(f, 0, SEEK_END) != 0) { fclose(f); return NULL; }
    const long sz = ftell(f);
    if (sz < 0) { fclose(f); return NULL; }
    rewind(f);
    *n = (size_t)sz;
    uint8_t *p = (uint8_t *)malloc(*n ? *n : 1);
    if (!p || (*n && fread(p, 1, *n, f) != *n)) { fclose(f); free(p); return NULL; }
    fclose(f);
    return p;
}

/* ---- one matrix from the checkpoint, widened to F32 --------------------- */
static int load_source(const char *dir, const char *name, float **W, int *rows, int *cols) {
    v41_st S;
    memset(&S, 0, sizeof S);
    if (v41_st_open(&S, dir) != 0) { fprintf(stderr, "%s: cannot index %s\n", name, dir); return -1; }
    const v41_st_ent *E = v41_st_find(&S, name);
    if (!E) { fprintf(stderr, "%s: not in %s\n", name, dir); v41_st_close(&S); return -1; }
    if (E->nd != 2) { fprintf(stderr, "%s: rank %d, not a matrix\n", name, E->nd); v41_st_close(&S); return -1; }
    const uint8_t *raw = v41_st_data(&S, E);
    if (!raw) { v41_st_close(&S); return -1; }
    const int64_t n = v41_st_numel(E);
    float *w = (float *)malloc(sizeof(float) * (size_t)n);
    if (!w) { v41_st_close(&S); return -1; }
    if (!strcmp(E->dtype, "BF16")) {
        const uint16_t *h = (const uint16_t *)raw;
        for (int64_t i = 0; i < n; i++) { w[i] = ds4_bf16_to_f32(h[i]); }
    } else if (!strcmp(E->dtype, "F32")) {
        memcpy(w, raw, (size_t)n * 4);
    } else if (!strcmp(E->dtype, "F16")) {
        const uint16_t *h = (const uint16_t *)raw;
        for (int64_t i = 0; i < n; i++) { w[i] = go1b_fp16_to_fp32(h[i]); }
    } else {
        fprintf(stderr, "%s: dtype %s is not supported\n", name, E->dtype);
        free(w); v41_st_close(&S); return -1;
    }
    *W = w;
    *rows = (int)E->shape[0];
    *cols = (int)E->shape[1];
    v41_st_close(&S);
    return 0;
}

/* cos and relative L2 of a reconstruction against the source. */
static void quality(const float *W, const float *Wq, size_t n, double *cos, double *rel) {
    double dot = 0, na = 0, nb = 0, se = 0;
    for (size_t i = 0; i < n; i++) {
        const double a = W[i], b = Wq[i];
        dot += a * b; na += a * a; nb += b * b;
        const double d = a - b; se += d * d;
    }
    *cos = dot / (sqrt(na) * sqrt(nb) + 1e-30);
    *rel = sqrt(se / (na + 1e-30));
}

/* ---- selftest ----------------------------------------------------------- */
static int st_fail;
#define ST(cond, ...) do { if (!(cond)) { printf("  FAIL "); printf(__VA_ARGS__); printf("\n"); st_fail++; } } while (0)

static void synth(float *W, size_t n, uint32_t seed) {
    uint32_t s = seed;
    for (size_t i = 0; i < n; i++) {
        s = s * 1664525u + 1013904223u;
        W[i] = ((float)(s >> 8) / 16777216.0f - 0.5f) * 2.0f;
    }
}

/* The value rule, computed straight from the encoder's arrays. Independent of
 * the payload's packing, so it is a real oracle for gvq_decode. */
static void oracle(const enc_out *o, int rows, int cols, int with_res, float *Wq) {
    const int vpr = cols / DIM;
    for (int r = 0; r < rows; r++) {
        const float g = go1b_fp16_to_fp32(o->gain[r]);
        for (int jj = 0; jj < vpr; jj++) {
            const size_t p = (size_t)r * vpr + jj;
            const uint8_t *c1 = o->book + (size_t)o->idx[p] * DIM;
            const uint8_t *c2 = with_res ? o->rbook + (size_t)o->ridx[p] * DIM : NULL;
            for (int d = 0; d < DIM; d++) {
                Wq[p * DIM + d] = g * gvq_elem(c1, c2, d);
            }
        }
    }
}

/* One mutation, then the reader must refuse the payload. */
static void reject_case(const uint8_t *pay, size_t sz, const uint8_t *book,
                        const char *what, void (*mut)(uint8_t *, size_t)) {
    uint8_t *m = (uint8_t *)malloc(sz);
    char err[160] = "";
    memcpy(m, pay, sz);
    mut(m, sz);
    if (gvq_check(m, sz, book, NULL, err, sizeof err) == 0) {
        printf("  FAIL accepted a mutated payload: %s\n", what);
        st_fail++;
    } else {
        printf("  ok   rejected: %-34s (%s)\n", what, err);
    }
    free(m);
}

static void m_magic(uint8_t *p, size_t n) { (void)n; p[0] = 'X'; }
static void m_version(uint8_t *p, size_t n) { (void)n; p[4] = 2; }
static void m_bits(uint8_t *p, size_t n) { (void)n; p[6] = 12; }
static void m_dim(uint8_t *p, size_t n) { (void)n; p[7] = 4; }
static void m_k(uint8_t *p, size_t n) { (void)n; gvq_put32(p + 16, 3000); }
static void m_prec(uint8_t *p, size_t n) { (void)n; gvq_put32(p + 20, 2); }
static void m_flags(uint8_t *p, size_t n) { (void)n; gvq_put32(p + 28, 1); }
static void m_gain_off(uint8_t *p, size_t n) { (void)n; gvq_put64(p + 32, 256); }
static void m_main_off(uint8_t *p, size_t n) { (void)n; gvq_put64(p + 40, gvq_get64(p + 40) + 128); }
static void m_plane_off(uint8_t *p, size_t n) { (void)n; gvq_put64(p + 48, gvq_get64(p + 48) + 128); }
static void m_size(uint8_t *p, size_t n) { (void)n; gvq_put64(p + 56, gvq_get64(p + 56) + 128); }
static void m_body(uint8_t *p, size_t n) { p[n - 200] ^= 0x01; }
static void m_book_hash(uint8_t *p, size_t n) { (void)n; p[64] ^= 0x01; }
static void m_gain_neg(uint8_t *p, size_t n) { (void)n; gvq_put16(p + GVQ_HEADER, (uint16_t)(gvq_get16(p + GVQ_HEADER) | 0x8000)); }
static void m_gain_inf(uint8_t *p, size_t n) { (void)n; gvq_put16(p + GVQ_HEADER, 0x7c00); }
static void m_pad_head(uint8_t *p, size_t n) { (void)n; p[GVQ_HEADER + 2u * gvq_get32(p + 8)] = 1; }
static void m_pad_tail(uint8_t *p, size_t n) { p[n - 1] = 1; }

/* A mutation that also recomputes the body checksum: the rules it targets are
 * then the reason for the rejection, not the checksum that guards them. */
static void reseal(uint8_t *p, size_t n) { ds4_sha256(p + GVQ_HEADER, n - GVQ_HEADER, p + 96); }
static void m_gain_neg_s(uint8_t *p, size_t n) { m_gain_neg(p, n); reseal(p, n); }
static void m_gain_inf_s(uint8_t *p, size_t n) { m_gain_inf(p, n); reseal(p, n); }
static void m_pad_head_s(uint8_t *p, size_t n) { m_pad_head(p, n); reseal(p, n); }
static void m_pad_tail_s(uint8_t *p, size_t n) { m_pad_tail(p, n); reseal(p, n); }
static void m_pad_plane_s(uint8_t *p, size_t n) {
    const size_t plane = gvq_plane_off((int)gvq_get32(p + 8), (int)gvq_get32(p + 12));
    p[plane - 1] = 1;
    reseal(p, n);
}

static int selftest(const char *dir) {
    /* k, iters, r_iters, pool_per_code, threads, residual, book_id */
    const enc_opts o = { (int)GVQ_K, 2, 2, 1, 4, 0, 0 };
    const int rows = 128, cols = 512;
    const size_t n = (size_t)rows * cols;
    float *W = (float *)malloc(sizeof(float) * n);
    float *Wq = (float *)malloc(sizeof(float) * n);
    float *Wo = (float *)malloc(sizeof(float) * n);
    enc_out a;
    enc_times tm;
    const size_t sz = gvq_bytes(rows, cols, (int)GVQ_K);
    uint8_t *pay = (uint8_t *)malloc(sz);
    uint8_t *pay2 = (uint8_t *)malloc(sz);
    int rc = 1;

    memset(&a, 0, sizeof a);
    if (!W || !Wq || !Wo || !pay || !pay2) {
        fprintf(stderr, "selftest: out of memory\n");
        goto out;
    }

    printf("== byte account\n");
    {
        /* 128 + 2R gains, then R*C*3/16 of stream, R*C/64 of plane, each region
         * 128-aligned: the same four expressions glm_vq.rs:140-160 checks. */
        const size_t main = gvq_align(128 + 2 * (size_t)rows);
        const size_t main_end = main + n / 16 * 3;
        const size_t plane = gvq_align(main_end);
        const size_t want = gvq_align(plane + n / 64);
        printf("  %dx%d: main=%zu plane=%zu size=%zu  (stream %zu B, gains %d B, book %zu B, %.4f bpw)\n",
               rows, cols, main, plane, gvq_bytes(rows, cols, (int)GVQ_K),
               n / 16 * 3, 2 * rows, gvq_book_bytes((int)GVQ_K),
               8.0 * (double)gvq_bytes(rows, cols, (int)GVQ_K) / (double)n);
        ST(gvq_bytes(rows, cols, (int)GVQ_K) == want,
           "gvq_bytes disagrees with the format arithmetic");
        ST(gvq_bytes(rows, cols, (int)GVQ_K12) ==
           gvq_align(gvq_align(128 + 2 * (size_t)rows) + n / 16 * 3), "12-bit byte account");
    }

    printf("== encode: 13-bit, K=8192, dim 8, E4M3 book\n");
    synth(W, n, 0x1234u);
    if (encode_matrix(W, rows, cols, "selftest", &o, &a, &tm, stdout) != 0) { goto out; }
    {
        const size_t wrote = gvq_write(pay, rows, cols, (int)GVQ_K, 0, a.book, a.gain, a.idx);
        ST(wrote == sz, "writer wrote %zu bytes, the account says %zu", wrote, sz);
    }

    printf("== strict read\n");
    {
        gvq_hdr h;
        char err[160] = "";
        ST(gvq_check(pay, sz, a.book, &h, err, sizeof err) == 0, "own payload rejected: %s", err);
        ST(h.rows == rows && h.cols == cols && h.k == (int)GVQ_K && h.size == sz, "header fields");
        ST(h.book_id == 0, "book id");
        printf("  ok   accepted (%dx%d, k=%d, %zu B, %.4f bits/weight)\n",
               h.rows, h.cols, h.k, h.size, 8.0 * (double)h.size / (double)n);
    }

    printf("== indices and values round-trip\n");
    {
        uint32_t *idx = (uint32_t *)malloc(sizeof(uint32_t) * (size_t)a.nidx);
        int bad = 0;
        gvq_unpack(pay, idx);
        for (long i = 0; i < a.nidx; i++) { if (idx[i] != a.idx[i]) { bad++; } }
        ST(bad == 0, "%d of %ld indices did not survive the 12+1 stream", bad, a.nidx);
        printf("  %s %ld indices through the 12-bit stream and the 1-bit plane\n",
               bad ? "FAIL" : "ok  ", a.nidx);
        free(idx);

        gvq_decode(pay, a.book, NULL, NULL, Wq);
        oracle(&a, rows, cols, 0, Wo);
        int diff = 0;
        for (size_t i = 0; i < n; i++) { if (Wq[i] != Wo[i]) { diff++; } }
        ST(diff == 0, "%d decoded values differ from the value rule", diff);
        double cos, rel;
        quality(W, Wq, n, &cos, &rel);
        printf("  ok   decode == value rule on %zu values; cos=%.6f relL2=%.4f%%\n",
               n, cos, 100.0 * rel);
    }

    printf("== determinism across thread counts\n");
    {
        enc_out x;
        enc_times tx;
        enc_opts o1 = o;
        o1.threads = 1;
        memset(&x, 0, sizeof x);
        if (encode_matrix(W, rows, cols, "selftest", &o1, &x, &tx, NULL) != 0) { goto out; }
        const size_t w2 = gvq_write(pay2, rows, cols, (int)GVQ_K, 0, x.book, x.gain, x.idx);
        ST(w2 == sz && memcmp(pay, pay2, sz) == 0,
           "a 1-thread encode differs from the 4-thread one");
        printf("  %s 1-thread and 4-thread payloads byte-identical\n",
               (w2 == sz && memcmp(pay, pay2, sz) == 0) ? "ok  " : "FAIL");
        enc_out_free(&x);
    }

    printf("== residual stage (GVR8)\n");
    {
        enc_opts orr = o;
        orr.residual = 1;
        orr.book_id = 7;
        enc_out r;
        enc_times tr;
        memset(&r, 0, sizeof r);
        if (encode_matrix(W, rows, cols, "selftest", &orr, &r, &tr, NULL) != 0) { goto out; }
        const size_t rsz = gvr_bytes(rows, cols);
        uint8_t *gvr = (uint8_t *)malloc(rsz);
        uint8_t *r1 = (uint8_t *)malloc(sz);
        const size_t rw = gvr_write(gvr, rows, cols, orr.book_id + 1, r.rbook, r.ridx);
        ST(rw == rsz, "GVR8 writer wrote %zu, the account says %zu", rw, rsz);
        {
            gvr_hdr gh;
            char err[160] = "";
            ST(gvr_check(gvr, rsz, r.rbook, &gh, err, sizeof err) == 0, "GVR8 rejected: %s", err);
            ST(gh.rows == rows && gh.cols == cols && gh.size == rsz &&
               gh.book_id == orr.book_id + 1, "GVR8 header fields");
        }
        /* stage 1 must not change when the residual stage is armed */
        const size_t rw1 = gvq_write(r1, rows, cols, (int)GVQ_K, 0, r.book, r.gain, r.idx);
        ST(rw1 == sz && memcmp(r1, pay, sz) == 0,
           "the stage-1 payload changed when the residual stage was added");
        printf("  %s stage-1 payload byte-identical with and without the residual\n",
               (rw1 == sz && memcmp(r1, pay, sz) == 0) ? "ok  " : "FAIL");

        gvq_decode(r1, r.book, gvr, r.rbook, Wq);
        oracle(&r, rows, cols, 1, Wo);
        int diff = 0;
        for (size_t i = 0; i < n; i++) { if (Wq[i] != Wo[i]) { diff++; } }
        ST(diff == 0, "%d two-stage values differ from the value rule", diff);
        double c1, e1, c2, e2;
        gvq_decode(r1, r.book, NULL, NULL, Wo);
        quality(W, Wo, n, &c1, &e1);
        quality(W, Wq, n, &c2, &e2);
        printf("  ok   two-stage decode == value rule; stage1 cos=%.6f rel=%.4f%%  "
               "two-stage cos=%.6f rel=%.4f%%\n", c1, 100.0 * e1, c2, 100.0 * e2);
        ST(c2 >= c1 - 1e-12, "the residual stage made the reconstruction worse");

        printf("== GVR8 rejections\n");
        {
            uint8_t *m = (uint8_t *)malloc(rsz);
            char err[160];
            int bad = 0;
            memcpy(m, gvr, rsz); m[0] = 'X';
            bad += gvr_check(m, rsz, r.rbook, NULL, err, sizeof err) == 0;
            memcpy(m, gvr, rsz); m[6] = 13;
            bad += gvr_check(m, rsz, r.rbook, NULL, err, sizeof err) == 0;
            memcpy(m, gvr, rsz); gvq_put32(m + 16, GVQ_K);
            bad += gvr_check(m, rsz, r.rbook, NULL, err, sizeof err) == 0;
            memcpy(m, gvr, rsz); m[rsz - 1] = 1;
            bad += gvr_check(m, rsz, r.rbook, NULL, err, sizeof err) == 0;
            memcpy(m, gvr, rsz); m[rsz / 2] ^= 1;
            bad += gvr_check(m, rsz, r.rbook, NULL, err, sizeof err) == 0;
            ST(bad == 0, "%d of 5 GVR8 mutations were accepted", bad);
            printf("  %s 5 GVR8 mutations rejected (magic, bits, K, padding, checksum)\n",
                   bad ? "FAIL" : "ok  ");
            free(m);
        }
        free(r1);
        free(gvr);
        enc_out_free(&r);
    }

    printf("== GVQ1 rejections\n");
    {
        reject_case(pay, sz, a.book, "bad magic", m_magic);
        reject_case(pay, sz, a.book, "bad version", m_version);
        reject_case(pay, sz, a.book, "index bits vs K", m_bits);
        reject_case(pay, sz, a.book, "vector dim", m_dim);
        reject_case(pay, sz, a.book, "K field", m_k);
        reject_case(pay, sz, a.book, "precision", m_prec);
        reject_case(pay, sz, a.book, "flags", m_flags);
        reject_case(pay, sz, a.book, "gain offset", m_gain_off);
        reject_case(pay, sz, a.book, "main offset", m_main_off);
        reject_case(pay, sz, a.book, "plane offset", m_plane_off);
        reject_case(pay, sz, a.book, "size field", m_size);
        reject_case(pay, sz, a.book, "body checksum", m_body);
        reject_case(pay, sz, a.book, "codebook checksum", m_book_hash);
        reject_case(pay, sz, a.book, "negative gain (resealed)", m_gain_neg_s);
        reject_case(pay, sz, a.book, "infinite gain (resealed)", m_gain_inf_s);
        {
            gvq_hdr h;
            char err[160] = "";
            ST(gvq_check(pay, sz - 1, a.book, &h, err, sizeof err) != 0,
               "truncated payload accepted");
            printf("  ok   rejected: %-34s (%s)\n", "truncated payload", err);
            uint8_t *bb = (uint8_t *)malloc(gvq_book_bytes((int)GVQ_K));
            memcpy(bb, a.book, gvq_book_bytes((int)GVQ_K));
            bb[3] = 0x7f;
            ST(gvq_check(pay, sz, bb, &h, err, sizeof err) != 0, "NaN codeword accepted");
            printf("  %s rejected: %-34s\n", "ok  ", "NaN codeword");
            free(bb);
        }
    }

    printf("== padding regions (all three exercised and enforced)\n");
    {
        /* rows=100 leaves 56 bytes between the gains and the stream; cols=1280
         * puts 64 bytes in front of the plane and 48 at the tail. */
        const int pr = 100, pc = 1280;
        const enc_opts po = { (int)GVQ_K, 1, 1, 1, 2, 0, 0 };
        const size_t psz = gvq_bytes(pr, pc, (int)GVQ_K);
        uint8_t *pp = (uint8_t *)malloc(psz);
        float *pw = (float *)malloc(sizeof(float) * (size_t)pr * pc);
        enc_out pe;
        enc_times pt;
        memset(&pe, 0, sizeof pe);
        synth(pw, (size_t)pr * pc, 0x99u);
        if (!pp || !pw || encode_matrix(pw, pr, pc, "pad", &po, &pe, &pt, NULL) != 0) {
            free(pp); free(pw);
            goto out;
        }
        gvq_write(pp, pr, pc, (int)GVQ_K, 0, pe.book, pe.gain, pe.idx);
        {
            const size_t gap0 = gvq_main_off(pr) - (GVQ_HEADER + 2u * (size_t)pr);
            const size_t gap1 = gvq_plane_off(pr, pc) - (gvq_main_off(pr) + (size_t)pr * pc / 16 * 3);
            const size_t end = gvq_plane_off(pr, pc) + (size_t)pr * pc / 64;
            printf("  %dx%d: gaps %zu / %zu / %zu bytes\n", pr, pc, gap0, gap1, psz - end);
            ST(gap0 > 0 && gap1 > 0 && psz - end > 0,
               "the fixture does not exercise a padding gap");
            reject_case(pp, psz, pe.book, "nonzero padding after gains", m_pad_head_s);
            reject_case(pp, psz, pe.book, "nonzero padding before plane", m_pad_plane_s);
            reject_case(pp, psz, pe.book, "nonzero padding at the tail", m_pad_tail_s);
            gvq_hdr h;
            char err[160] = "";
            ST(gvq_check(pp, psz, pe.book, &h, err, sizeof err) == 0,
               "padding fixture rejected: %s", err);
            printf("  ok   padding fixture accepted\n");
        }
        enc_out_free(&pe);
        free(pp);
        free(pw);
    }

    printf("== 12-bit path (K=4096, no plane)\n");
    {
        const int r2 = 64, c2 = 512;
        const enc_opts o2 = { (int)GVQ_K12, 1, 1, 1, 2, 0, 0 };
        const size_t s2 = gvq_bytes(r2, c2, (int)GVQ_K12);
        uint8_t *p2 = (uint8_t *)malloc(s2);
        float *w2 = (float *)malloc(sizeof(float) * (size_t)r2 * c2);
        float *q2 = (float *)malloc(sizeof(float) * (size_t)r2 * c2);
        float *q2o = (float *)malloc(sizeof(float) * (size_t)r2 * c2);
        enc_out e2;
        enc_times t2;
        memset(&e2, 0, sizeof e2);
        synth(w2, (size_t)r2 * c2, 0x55u);
        if (!p2 || !w2 || !q2 || !q2o ||
            encode_matrix(w2, r2, c2, "k12", &o2, &e2, &t2, NULL) != 0) {
            free(p2); free(w2); free(q2); free(q2o);
            goto out;
        }
        ST(gvq_write(p2, r2, c2, (int)GVQ_K12, 0, e2.book, e2.gain, e2.idx) == s2,
           "12-bit writer size");
        {
            gvq_hdr h;
            char err[160] = "";
            ST(gvq_check(p2, s2, e2.book, &h, err, sizeof err) == 0,
               "12-bit payload rejected: %s", err);
            ST(gvq_get64(p2 + 48) == 0, "a 12-bit payload must have no plane offset");
            ST(p2[6] == GVQ_BITS12, "index bits are not 12");
            gvq_decode(p2, e2.book, NULL, NULL, q2);
            oracle(&e2, r2, c2, 0, q2o);
            int diff = 0;
            for (size_t i = 0; i < (size_t)r2 * c2; i++) { if (q2[i] != q2o[i]) { diff++; } }
            ST(diff == 0, "%d 12-bit values differ from the value rule", diff);
            uint32_t *ix = (uint32_t *)malloc(sizeof(uint32_t) * (size_t)e2.nidx);
            gvq_unpack(p2, ix);
            int bad = 0;
            for (long i = 0; i < e2.nidx; i++) { if (ix[i] != e2.idx[i]) { bad++; } }
            ST(bad == 0, "%d 12-bit indices differ", bad);
            free(ix);
            printf("  ok   12-bit payload %zu B, no plane, %ld indices and values exact\n",
                   s2, e2.nidx);
        }
        enc_out_free(&e2);
        free(p2); free(w2); free(q2); free(q2o);
    }

    rc = st_fail ? 1 : 0;
    printf("\n%s: %d failure(s)\n", rc ? "SELFTEST FAILED" : "SELFTEST PASSED", st_fail);
out:
    if (dir && *dir) { printf("(note: the selftest needs no checkpoint; the argument is ignored)\n"); }
    enc_out_free(&a);
    free(W); free(Wq); free(Wo); free(pay); free(pay2);
    return rc;
}

/* ---- encode / check ----------------------------------------------------- */
static void usage(void) {
    fprintf(stderr,
        "usage:\n"
        "  h3_gvq selftest [dir]\n"
        "  h3_gvq encode <safetensors-dir> <tensor> <prefix> [--k 8192|4096] [--iters N]\n"
        "                [--res-iters N] [--pool-per-code N] [--threads N] [--residual]\n"
        "                [--book-id N]\n"
        "  h3_gvq check <prefix> [--src <safetensors-dir> --name <tensor>]\n");
}

static int encode_cmd(int argc, char **argv) {
    if (argc < 5) { usage(); return 2; }
    const char *dir = argv[2], *name = argv[3], *prefix = argv[4];
    enc_opts o = { (int)GVQ_K, 32, 32, 1000, 0, 0, 0 };
    for (int i = 5; i < argc; i++) {
        if (!strcmp(argv[i], "--k") && i + 1 < argc) { o.k = atoi(argv[++i]); }
        else if (!strcmp(argv[i], "--iters") && i + 1 < argc) { o.iters = atoi(argv[++i]); }
        else if (!strcmp(argv[i], "--res-iters") && i + 1 < argc) { o.r_iters = atoi(argv[++i]); }
        else if (!strcmp(argv[i], "--pool-per-code") && i + 1 < argc) { o.pool_per_code = atol(argv[++i]); }
        else if (!strcmp(argv[i], "--threads") && i + 1 < argc) { o.threads = atoi(argv[++i]); }
        else if (!strcmp(argv[i], "--book-id") && i + 1 < argc) { o.book_id = (uint32_t)strtoul(argv[++i], NULL, 10); }
        else if (!strcmp(argv[i], "--residual")) { o.residual = 1; }
        else { fprintf(stderr, "unknown option %s\n", argv[i]); usage(); return 2; }
    }
    if (o.k != (int)GVQ_K && o.k != (int)GVQ_K12) { fprintf(stderr, "--k must be 8192 or 4096\n"); return 2; }

    float *W = NULL;
    int rows = 0, cols = 0;
    if (load_source(dir, name, &W, &rows, &cols) != 0) { return 1; }
    const size_t n = (size_t)rows * (size_t)cols;
    const size_t sz = gvq_bytes(rows, cols, o.k);
    uint8_t *pay = (uint8_t *)malloc(sz);
    char path[1024], report[1024];
    enc_out e;
    enc_times tm;
    memset(&e, 0, sizeof e);
    FILE *log = stderr;

    fprintf(stderr, "== %s  %dx%d  (%zu weights, %ld vectors, k=%d, E4M3 book)\n",
            name, rows, cols, n, (long)(n / DIM), o.k);
    if (!pay || encode_matrix(W, rows, cols, name, &o, &e, &tm, log) != 0) { free(W); free(pay); return 1; }

    const size_t wrote = gvq_write(pay, rows, cols, o.k, o.book_id, e.book, e.gain, e.idx);
    const double tw = now_s();
    snprintf(path, sizeof path, "%s.gvq", prefix);
    snprintf(report, sizeof report, "%s.report", prefix);
    size_t wrote_gvr = 0;
    if (write_file(path, pay, wrote) != 0) { free(W); free(pay); enc_out_free(&e); return 1; }
    snprintf(path, sizeof path, "%s.book", prefix);
    if (write_file(path, e.book, gvq_book_bytes(o.k)) != 0) { free(W); free(pay); enc_out_free(&e); return 1; }
    if (o.residual) {
        const size_t rsz = gvr_bytes(rows, cols);
        uint8_t *gvr = (uint8_t *)malloc(rsz);
        wrote_gvr = gvr_write(gvr, rows, cols, o.book_id + 1, e.rbook, e.ridx);
        snprintf(path, sizeof path, "%s.gvr", prefix);
        const int bad = !gvr || write_file(path, gvr, wrote_gvr) != 0;
        free(gvr);
        if (bad) { free(W); free(pay); enc_out_free(&e); return 1; }
        snprintf(path, sizeof path, "%s.rbook", prefix);
        if (write_file(path, e.rbook, gvq_book_bytes((int)GVR_K)) != 0) { free(W); free(pay); enc_out_free(&e); return 1; }
    }
    tm.write = now_s() - tw;

    /* metrics: decode the payload this process will ship, twice (stage 1 alone,
     * then with the residual), against the source matrix. */
    const double tq = now_s();
    float *Wq = (float *)malloc(sizeof(float) * n);
    uint8_t *gvr_in = NULL;
    size_t gvr_sz = 0;
    if (o.residual) {
        snprintf(path, sizeof path, "%s.gvr", prefix);
        gvr_in = read_file(path, &gvr_sz);
    }
    double cos1 = 0, rel1 = 0, cos = 0, rel = 0;
    if (Wq) {
        gvq_decode(pay, e.book, NULL, NULL, Wq);
        quality(W, Wq, n, &cos1, &rel1);
        if (o.residual && gvr_in) {
            gvq_decode(pay, e.book, gvr_in, e.rbook, Wq);
            quality(W, Wq, n, &cos, &rel);
        } else { cos = cos1; rel = rel1; }
    }
    tm.metrics = now_s() - tq;

    char err[160] = "";
    gvq_hdr h;
    const int ok = gvq_check(pay, wrote, e.book, &h, err, sizeof err) == 0;
    const double bpw = 8.0 * (double)(wrote + wrote_gvr + gvq_book_bytes(o.k) +
                                      (o.residual ? gvq_book_bytes((int)GVR_K) : 0)) / (double)n;

    FILE *rep = fopen(report, "w");
    FILE *outs[2] = { stderr, rep };
    for (int k2 = 0; k2 < 2; k2++) {
        FILE *f = outs[k2];
        if (!f) { continue; }
        fprintf(f, "name %s\nrows %d\ncols %d\nweights %zu\nvectors %ld\nk %d\niters %d\npool %ld\nthreads %d\n",
                name, rows, cols, n, (long)(n / DIM), o.k, o.iters, o.pool_per_code,
                o.threads > 0 ? o.threads : nthreads_default());
        fprintf(f, "PROFILE pool=%.3f init=%.3f lloyd=%.3f book=%.3f gain=%.3f final=%.3f residual=%.3f write=%.3f metrics=%.3f total=%.3f\n",
                tm.pool, tm.init, tm.lloyd, tm.book, tm.gain, tm.final, tm.residual, tm.write,
                tm.metrics, tm.total);
        fprintf(f, "RATE assign_vectors=%.0f assign_seconds=%.3f Mvec_per_s=%.3f\n",
                tm.assign_vec, tm.assign_s, tm.assign_s > 0 ? tm.assign_vec / tm.assign_s / 1e6 : 0.0);
        fprintf(f, "BYTES gvq=%zu gvr=%zu book=%zu rbook=%zu total=%zu bits_per_weight=%.4f\n",
                wrote, wrote_gvr, gvq_book_bytes(o.k),
                o.residual ? gvq_book_bytes((int)GVR_K) : (size_t)0,
                wrote + wrote_gvr + gvq_book_bytes(o.k) +
                    (o.residual ? gvq_book_bytes((int)GVR_K) : 0), bpw);
        fprintf(f, "QUALITY cos_stage1=%.6f rel_stage1=%.6f cos=%.6f rel=%.6f\n", cos1, rel1, cos, rel);
        fprintf(f, "READER %s%s%s\n", ok ? "ok" : "FAILED", ok ? "" : ": ", ok ? "" : err);
        if (k2 == 1) {
            fprintf(f, "recipe minimax-h3-fl2va rung=%s dim=8 k=%d index_bits=%d book=e4m3 gain=f16 "
                       "init=equidistant lloyd=%d pool=%ld-per-code weighting=uniform gain_refit=ls "
                       "residual=%s book_id=%u\n",
                    o.k == (int)GVQ_K ? "A" : "B", o.k, o.k == (int)GVQ_K ? 13 : 12, o.iters,
                    o.pool_per_code, o.residual ? "GVR8" : "none", o.book_id);
        }
    }
    if (rep) { fclose(rep); }
    free(gvr_in); free(Wq); free(W); free(pay); enc_out_free(&e);
    if (!ok) { fprintf(stderr, "FAILED: the payload does not pass its own reader: %s\n", err); return 1; }
    return 0;
}

static int check_cmd(int argc, char **argv) {
    if (argc < 3) { usage(); return 2; }
    const char *prefix = argv[2], *src = NULL, *name = NULL;
    for (int i = 3; i < argc; i++) {
        if (!strcmp(argv[i], "--src") && i + 1 < argc) { src = argv[++i]; }
        else if (!strcmp(argv[i], "--name") && i + 1 < argc) { name = argv[++i]; }
        else { fprintf(stderr, "unknown option %s\n", argv[i]); usage(); return 2; }
    }
    char path[1024];
    size_t n = 0, bn = 0;
    snprintf(path, sizeof path, "%s.gvq", prefix);
    uint8_t *pay = read_file(path, &n);
    snprintf(path, sizeof path, "%s.book", prefix);
    uint8_t *book = read_file(path, &bn);
    if (!pay || !book) { free(pay); free(book); return 1; }
    gvq_hdr h;
    char err[160] = "";
    if (gvq_check(pay, n, book, &h, err, sizeof err) != 0) {
        fprintf(stderr, "REJECTED: %s\n", err);
        free(pay); free(book); return 1;
    }
    if (bn != gvq_book_bytes(h.k)) {
        fprintf(stderr, "REJECTED: the codebook file is %zu B, k=%d needs %zu\n",
                bn, h.k, gvq_book_bytes(h.k));
        free(pay); free(book); return 1;
    }
    printf("GVQ1 OK  %dx%d  k=%d  bits=%d  book_id=%u  size=%zu  %.4f bits/weight\n",
           h.rows, h.cols, h.k, pay[6], h.book_id, h.size,
           8.0 * (double)h.size / ((double)h.rows * (double)h.cols));

    /* the residual, when the prefix carries one */
    snprintf(path, sizeof path, "%s.gvr", prefix);
    size_t gn = 0, rbn = 0;
    uint8_t *gvr = read_file(path, &gn);
    uint8_t *rbook = NULL;
    if (gvr) {
        snprintf(path, sizeof path, "%s.rbook", prefix);
        rbook = read_file(path, &rbn);
        gvr_hdr gh;
        if (!rbook) { fprintf(stderr, "a .gvr without a .rbook is incomplete\n"); }
        else if (gvr_check(gvr, gn, rbook, &gh, err, sizeof err) != 0) {
            fprintf(stderr, "REJECTED (residual): %s\n", err);
            free(gvr); free(rbook); free(pay); free(book); return 1;
        } else if (gh.rows != h.rows || gh.cols != h.cols) {
            fprintf(stderr, "REJECTED: residual shape %dx%d against the matrix's %dx%d\n",
                    gh.rows, gh.cols, h.rows, h.cols);
            free(gvr); free(rbook); free(pay); free(book); return 1;
        } else if (h.k != (int)GVQ_K) {
            fprintf(stderr, "REJECTED: a residual stage needs a K=8192 parent\n");
            free(gvr); free(rbook); free(pay); free(book); return 1;
        } else if (gh.book_id != h.book_id + 1) {
            fprintf(stderr, "REJECTED: residual book id %u is not the parent's %u + 1\n",
                    gh.book_id, h.book_id);
            free(gvr); free(rbook); free(pay); free(book); return 1;
        } else {
            printf("GVR1 OK  %dx%d  k=%u  8-bit  book_id=%u  size=%zu\n",
                   gh.rows, gh.cols, GVR_K, gh.book_id, gh.size);
        }
    }

    /* decode, and verify the value rule independently of the decoder */
    const size_t ns = (size_t)h.rows * (size_t)h.cols;
    float *Wq = (float *)malloc(sizeof(float) * ns);
    if (!Wq) { free(gvr); free(rbook); free(pay); free(book); return 1; }
    gvq_decode(pay, book, gvr, rbook, Wq);
    {
        uint32_t *idx = (uint32_t *)malloc(sizeof(uint32_t) * (ns / DIM));
        uint8_t *ridx = gvr ? (uint8_t *)malloc(ns / DIM) : NULL;
        gvq_unpack(pay, idx);
        if (gvr) { gvr_unpack(gvr, ridx); }
        int bad = 0;
        for (int r = 0; r < h.rows; r++) {
            const float g = go1b_fp16_to_fp32(gvq_get16(pay + GVQ_HEADER + 2u * (size_t)r));
            for (size_t jj = 0; jj < (size_t)h.cols / DIM; jj++) {
                const size_t p = (size_t)r * ((size_t)h.cols / DIM) + jj;
                const uint8_t *c1 = book + (size_t)idx[p] * DIM;
                const uint8_t *c2 = ridx ? rbook + (size_t)ridx[p] * DIM : NULL;
                for (int d = 0; d < DIM; d++) {
                    const float want = g * gvq_elem(c1, c2, d);
                    if (Wq[p * DIM + d] != want) { bad++; }
                }
            }
        }
        printf("VALUE RULE %s  %zu values, %d disagree\n", bad ? "FAILED" : "OK", ns, bad);
        free(idx); free(ridx);
        if (bad) { free(Wq); free(gvr); free(rbook); free(pay); free(book); return 1; }
    }

    if (src && name) {
        float *W = NULL;
        int rows = 0, cols = 0;
        if (load_source(src, name, &W, &rows, &cols) != 0) {
            free(Wq); free(gvr); free(rbook); free(pay); free(book); return 1;
        }
        if (rows != h.rows || cols != h.cols) {
            fprintf(stderr, "source is %dx%d, payload is %dx%d\n", rows, cols, h.rows, h.cols);
        } else {
            double cos, rel;
            quality(W, Wq, ns, &cos, &rel);
            printf("QUALITY cos=%.6f relL2=%.6f (against %s:%s)\n", cos, rel, src, name);
        }
        free(W);
    }
    free(Wq); free(gvr); free(rbook); free(pay); free(book);
    return 0;
}

int main(int argc, char **argv) {
    if (argc < 2) { usage(); return 2; }
    if (!strcmp(argv[1], "selftest")) { return selftest(argc > 2 ? argv[2] : NULL) ? 1 : 0; }
    if (!strcmp(argv[1], "encode")) { return encode_cmd(argc, argv); }
    if (!strcmp(argv[1], "check")) { return check_cmd(argc, argv); }
    usage();
    return 2;
}
