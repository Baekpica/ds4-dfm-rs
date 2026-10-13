/* test_ds41_engram.cu — P4-3 gate: the engram gate kernel
 * (ds4_gpu_v41_engram_gate_tensor) and the row dequant
 * (ds4_gpu_v41_engram_rows_tensor), plus the read path's device primitives
 * (pinned alloc -> zero-copy upload -> spin-flag wait).
 *
 *   test_ds41_engram <engram.img> <engram.cases.txt> <engram.ref.f32> <rows.bin> <rows.ref.f32>
 *
 * The fixture (tests/fixtures/ds41/engram/gen_engram.py) carries the
 * artifact's own geometry (E=5120, HC=4, head_dim=256, eps=1e-20) at n_tok=2:
 *   dense  — random h/key/val: the f32 reduction order and the fast-math expf
 *            differ from the emulation, and the output is bf16-quantized, so
 *            the compare is at one-bf16-ulp distance.
 *   absorb — h on the bf16 grid with a tiny val: the update lands under half
 *            an ulp, so h must come back BIT-EXACT.
 *   zero   — h = 0: the pure gate path, bf16r(gate*val).
 * The rows half must be bit-exact: every product is exact and both sides run
 * the same bf16r.
 */
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "ds4_gpu.h"

/* Real-data mode: dense wkv against the f64 emulation (see real_main). */
#define TOL_SCALE 1e-3f

typedef struct {
    char kind[16];
    uint32_t n_tok;
    float *h;
    float *kv;
} gate_case;

static void *xread(const char *path, uint64_t *size) {
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "test_ds41_engram: cannot open %s\n", path); exit(2); }
    fseek(f, 0, SEEK_END);
    const long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    /* Page-aligned: the image is the "model map" and the resolver's register
     * tier page-rounds ranges (ds4_cuda.cu:1795-1803). */
    void *buf = NULL;
    if (posix_memalign(&buf, 4096, (size_t)n) != 0 || !buf) { fprintf(stderr, "test_ds41_engram: alloc %s\n", path); exit(2); }
    if (fread(buf, 1, (size_t)n, f) != (size_t)n) { fprintf(stderr, "test_ds41_engram: read %s\n", path); exit(2); }
    fclose(f);
    *size = (uint64_t)n;
    return buf;
}

static uint32_t hex32(const char *s) { return (uint32_t)strtoul(s, NULL, 16); }

/* bf16 ulp at |x|: the grid spacing of the coarser neighbor of x. */
static float bf16_ulp_at(float x) {
    if (!(x > 0.0f)) return 0.0f;
    int e = 0;
    (void)frexpf(x, &e);
    return ldexpf(1.0f, e - 9);
}

static gate_case *parse_cases(const char *path, uint32_t *n_cases, uint32_t *e_out, uint32_t *hc_out,
                              uint64_t *q_off, uint64_t *k_off, float *eps) {
    FILE *f = fopen(path, "r");
    if (!f) { fprintf(stderr, "test_ds41_engram: cannot open %s\n", path); exit(2); }
    gate_case *cases = NULL;
    uint32_t n = 0, cap = 0, e = 0, hc = 0;
    char *line = NULL;
    size_t line_cap = 0;
    while (getline(&line, &line_cap, f) > 0) {
        if (line[0] == '#' || line[0] == '\n') continue;
        if (strncmp(line, "gate ", 5) == 0) {
            char *tok = strtok(line + 5, " \t\n");
            while (tok) {
                char k[16] = {0};
                char v[64] = {0};
                if (sscanf(tok, "%15[^=]=%63s", k, v) == 2) {
                    if (strcmp(k, "q_off") == 0) *q_off = strtoull(v, NULL, 10);
                    else if (strcmp(k, "k_off") == 0) *k_off = strtoull(v, NULL, 10);
                    else if (strcmp(k, "e") == 0) e = (uint32_t)strtoul(v, NULL, 10);
                    else if (strcmp(k, "hc") == 0) hc = (uint32_t)strtoul(v, NULL, 10);
                    else if (strcmp(k, "eps") == 0) *eps = strtof(v, NULL);
                }
                tok = strtok(NULL, " \t\n");
            }
            continue;
        }
        if (strncmp(line, "case ", 5) == 0) {
            if (n == cap) { cap = cap ? cap * 2u : 8u; cases = (gate_case *)realloc(cases, cap * sizeof *cases); }
            gate_case *c = &cases[n++];
            memset(c, 0, sizeof *c);
            unsigned idx = 0;
            if (sscanf(line, "case %u %15s n=%u", &idx, c->kind, &c->n_tok) != 3) { fprintf(stderr, "test_ds41_engram: bad case line\n"); exit(2); }
            continue;
        }
        gate_case *c = &cases[n - 1];
        char *tok = strtok(line, " \t\n");
        if (!tok) continue;
        if (strcmp(tok, "h") == 0) {
            c->h = (float *)malloc((size_t)c->n_tok * hc * e * 4);
            for (uint32_t i = 0; i < c->n_tok * hc * e && (tok = strtok(NULL, " \t\n")); i++) { uint32_t b = hex32(tok); memcpy(&c->h[i], &b, 4); }
        } else if (strcmp(tok, "kv") == 0) {
            c->kv = (float *)malloc((size_t)c->n_tok * (hc + 1u) * e * 4);
            for (uint32_t i = 0; i < c->n_tok * (hc + 1u) * e && (tok = strtok(NULL, " \t\n")); i++) { uint32_t b = hex32(tok); memcpy(&c->kv[i], &b, 4); }
        }
    }
    free(line);
    fclose(f);
    *n_cases = n;
    *e_out = e;
    *hc_out = hc;
    return cases;
}

/* Real-data gate (the Spark, the golden erows): the builder
 * (crates/ds4-core/examples/ds41_engram_real.rs) preads the rows the engine's
 * own erows ids name, decodes them and computes the layer's wkv reference.
 * This mode runs the device chain on those exact bytes: rows bit-exact, wkv
 * within the dense criterion (the f32 GEMV vs the f64 emulation).
 *
 *   test_ds41_engram --real <wkv.img> <rows.bin> <erows.ref.f32> <wkv.ref.f32> <in_dim> <out_dim> <n_tok>
 */
static int real_main(int argc, char **argv) {
    if (argc != 9) {
        fprintf(stderr, "usage: %s --real <wkv.img> <rows.bin> <erows.ref.f32> <wkv.ref.f32> <in_dim> <out_dim> <n_tok>\n", argv[0]);
        return 2;
    }
    const uint32_t in_dim = (uint32_t)strtoul(argv[6], NULL, 10);
    const uint32_t out_dim = (uint32_t)strtoul(argv[7], NULL, 10);
    const uint32_t n_tok = (uint32_t)strtoul(argv[8], NULL, 10);
    const uint32_t hd = 256;
    if (!in_dim || !out_dim || !n_tok || (in_dim % hd) || (in_dim % 512u)) {
        fprintf(stderr, "test_ds41_engram: bad geometry\n");
        return 2;
    }
    uint64_t img_bytes = 0, rows_bytes = 0, erows_ref_bytes = 0, wkv_ref_bytes = 0;
    void *img = xread(argv[2], &img_bytes);
    uint8_t *rows = (uint8_t *)xread(argv[3], &rows_bytes);
    float *erows_ref = (float *)xread(argv[4], &erows_ref_bytes);
    float *wkv_ref = (float *)xread(argv[5], &wkv_ref_bytes);
    const uint32_t n_rows = n_tok * (in_dim / hd);
    const uint64_t stride = hd + hd / 32u;
    if (rows_bytes != (uint64_t)n_rows * stride || erows_ref_bytes != (uint64_t)n_rows * hd * 4 ||
        wkv_ref_bytes != (uint64_t)n_tok * out_dim * 4) {
        fprintf(stderr, "test_ds41_engram: file sizes do not match the geometry\n");
        return 2;
    }
    if (!ds4_gpu_init()) { fprintf(stderr, "test_ds41_engram: ds4_gpu_init failed\n"); return 2; }

    ds4_gpu_tensor *raw_t = ds4_gpu_tensor_alloc(rows_bytes);
    ds4_gpu_tensor *rows_t = ds4_gpu_tensor_alloc((uint64_t)n_rows * hd * 4);
    ds4_gpu_tensor *ekv_t = ds4_gpu_tensor_alloc((uint64_t)n_tok * out_dim * 4);
    if (!raw_t || !rows_t || !ekv_t) { fprintf(stderr, "test_ds41_engram: tensor alloc\n"); return 2; }
    void *pinned = ds4_gpu_host_alloc(rows_bytes);
    if (!pinned) { fprintf(stderr, "test_ds41_engram: pinned alloc\n"); return 2; }
    memcpy(pinned, rows, rows_bytes);
    if (!ds4_gpu_tensor_write_zerocopy(raw_t, 0, pinned, rows_bytes)) { fprintf(stderr, "test_ds41_engram: zerocopy upload refused\n"); return 2; }
    if (!ds4_gpu_v41_engram_rows_tensor(rows_t, raw_t, n_rows, hd)) { fprintf(stderr, "test_ds41_engram: rows entry refused\n"); return 2; }

    int n_fail = 0;
    float *rows_got = (float *)malloc((size_t)n_rows * hd * 4);
    ds4_gpu_tensor_read(rows_t, 0, rows_got, (uint64_t)n_rows * hd * 4);
    {
        uint32_t diff = 0;
        for (uint64_t i = 0; i < (uint64_t)n_rows * hd; i++) if (rows_got[i] != erows_ref[i]) diff++;
        if (diff) n_fail++;
        printf("real rows: %s (%llu values, %u bit-diffs)\n", diff ? "FAIL" : "PASS",
               (unsigned long long)((uint64_t)n_rows * hd), diff);
    }

    if (!ds4_gpu_v41_matmul_fp8blk_tensor(ekv_t, img, img_bytes, 0, in_dim, out_dim, rows_t, n_tok)) {
        fprintf(stderr, "test_ds41_engram: wkv entry refused\n");
        return 2;
    }
    float *got = (float *)malloc((size_t)n_tok * out_dim * 4);
    ds4_gpu_tensor_read(ekv_t, 0, got, (uint64_t)n_tok * out_dim * 4);
    {
        uint64_t n = (uint64_t)n_tok * out_dim;
        float max_abs = 0.0f, max_ref = 0.0f;
        uint64_t n_diff = 0;
        for (uint64_t i = 0; i < n; i++) {
            const float d = fabsf(got[i] - wkv_ref[i]);
            if (got[i] != wkv_ref[i]) n_diff++;
            if (d > max_abs) max_abs = d;
            if (fabsf(wkv_ref[i]) > max_ref) max_ref = fabsf(wkv_ref[i]);
        }
        const int pass = max_abs <= TOL_SCALE * max_ref;
        if (!pass) n_fail++;
        printf("real wkv: %s (n=%u in=%u out=%u max rel %.3e max abs %.3e scale %.3e diffs %llu)\n",
               pass ? "PASS" : "FAIL", n_tok, in_dim, out_dim,
               max_ref > 0.0f ? max_abs / max_ref : 0.0f, max_abs, max_ref, (unsigned long long)n_diff);
    }
    printf("DS41 engram real gate: %s\n", n_fail ? "FAIL" : "PASS");
    return n_fail ? 1 : 0;
}

int main(int argc, char **argv) {
    if (argc >= 2 && strcmp(argv[1], "--real") == 0) return real_main(argc, argv);
    if (argc != 6) {
        fprintf(stderr, "usage: %s <engram.img> <engram.cases.txt> <engram.ref.f32> <rows.bin> <rows.ref.f32>\n", argv[0]);
        return 2;
    }
    uint64_t img_bytes = 0, ref_bytes = 0, rows_bytes = 0, rows_ref_bytes = 0;
    void *img = xread(argv[1], &img_bytes);
    float *ref = (float *)xread(argv[3], &ref_bytes);
    uint8_t *rows = (uint8_t *)xread(argv[4], &rows_bytes);
    float *rows_ref = (float *)xread(argv[5], &rows_ref_bytes);
    uint32_t n_cases = 0, E = 0, HC = 0;
    uint64_t q_off = 0, k_off = 0;
    float eps = 0.0f;
    gate_case *cases = parse_cases(argv[2], &n_cases, &E, &HC, &q_off, &k_off, &eps);
    if (!E || !HC) { fprintf(stderr, "test_ds41_engram: missing gate descriptor\n"); return 2; }
    if (!ds4_gpu_init()) { fprintf(stderr, "test_ds41_engram: ds4_gpu_init failed\n"); return 2; }

    int n_fail = 0;
    uint64_t ref_off = 0;

    /* ---- the read path's device primitives ----
     * Pinned alloc -> zero-copy upload of the row bytes -> read back. The
     * forward uploads engram rows exactly this way; the rows kernel below
     * then consumes the uploaded bytes. */
    ds4_gpu_tensor *raw_t = ds4_gpu_tensor_alloc(rows_bytes);
    ds4_gpu_tensor *rows_t = ds4_gpu_tensor_alloc((uint64_t)(rows_bytes / 264) * 256 * 4);
    if (!raw_t || !rows_t) { fprintf(stderr, "test_ds41_engram: tensor alloc\n"); return 2; }
    void *pinned = ds4_gpu_host_alloc(rows_bytes);
    if (!pinned) { fprintf(stderr, "test_ds41_engram: pinned alloc\n"); return 2; }
    memcpy(pinned, rows, rows_bytes);
    if (!ds4_gpu_tensor_write_zerocopy(raw_t, 0, pinned, rows_bytes)) { fprintf(stderr, "test_ds41_engram: zerocopy upload refused\n"); return 2; }
    uint8_t *raw_back = (uint8_t *)malloc(rows_bytes);
    ds4_gpu_tensor_read(raw_t, 0, raw_back, rows_bytes);
    if (memcmp(raw_back, rows, rows_bytes) != 0) { fprintf(stderr, "test_ds41_engram: zerocopy upload mismatch\n"); n_fail++; }
    else printf("read path: pinned zero-copy upload %llu B PASS\n", (unsigned long long)rows_bytes);

    /* ---- spin-flag wait: the host sets the flag while the kernel polls ---- */
    int32_t *flags = (int32_t *)ds4_gpu_host_alloc(3 * 4);
    flags[0] = 0; flags[1] = 1; flags[2] = 0;   /* flag, want, err */
    if (!ds4_gpu_host_flag_wait(&flags[0], &flags[1], &flags[2])) { fprintf(stderr, "test_ds41_engram: flag wait refused\n"); n_fail++; }
    struct timespec ts = {0, 50 * 1000 * 1000};
    nanosleep(&ts, NULL);
    flags[0] = 1;
    if (cudaDeviceSynchronize() != cudaSuccess || flags[2] != 0) {
        fprintf(stderr, "test_ds41_engram: flag wait failed (err %d)\n", flags[2]);
        n_fail++;
    } else printf("read path: host flag wait PASS\n");

    /* ---- rows dequant: bit-exact ---- */
    const uint32_t n_rows = (uint32_t)(rows_bytes / 264);
    if (!ds4_gpu_v41_engram_rows_tensor(rows_t, raw_t, n_rows, 256)) { fprintf(stderr, "test_ds41_engram: rows entry refused\n"); return 2; }
    float *rows_got = (float *)malloc((size_t)n_rows * 256 * 4);
    ds4_gpu_tensor_read(rows_t, 0, rows_got, (uint64_t)n_rows * 256 * 4);
    {
        uint32_t diff = 0;
        for (uint32_t i = 0; i < n_rows * 256u; i++) if (rows_got[i] != rows_ref[i]) diff++;
        const int pass = diff == 0;
        if (!pass) n_fail++;
        printf("rows: %s (%u values, %u bit-diffs)\n", pass ? "PASS" : "FAIL", n_rows * 256u, diff);
    }

    /* ---- the gate ---- */
    ds4_gpu_tensor *hc_t = ds4_gpu_tensor_alloc((uint64_t)E * HC * 8u);
    ds4_gpu_tensor *kv_t = ds4_gpu_tensor_alloc((uint64_t)E * (HC + 1u) * 8u);
    if (!hc_t || !kv_t) { fprintf(stderr, "test_ds41_engram: gate tensor alloc\n"); return 2; }
    for (uint32_t ci = 0; ci < n_cases; ci++) {
        gate_case *c = &cases[ci];
        const uint64_t n_hc = (uint64_t)c->n_tok * HC * E, n_kv = (uint64_t)c->n_tok * (HC + 1u) * E;
        ds4_gpu_tensor_write(hc_t, 0, c->h, n_hc * 4);
        ds4_gpu_tensor_write(kv_t, 0, c->kv, n_kv * 4);
        if (!ds4_gpu_v41_engram_gate_tensor(hc_t, kv_t, img, img_bytes, q_off, k_off, E, HC, c->n_tok, eps)) {
            fprintf(stderr, "test_ds41_engram: gate case %u refused\n", ci);
            n_fail++; ref_off += n_hc; continue;
        }
        float *got = (float *)malloc(n_hc * 4);
        ds4_gpu_tensor_read(hc_t, 0, got, n_hc * 4);
        const float *want = ref + ref_off;
        int bit_exact = 1, n_diff = 0, over_bound = 0;
        float max_ulp_dist = 0.0f;
        float worst_d = 0.0f; uint64_t worst_i = 0;
        for (uint64_t i = 0; i < n_hc; i++) {
            if (got[i] == want[i]) continue;
            bit_exact = 0; n_diff++;
            const uint64_t t = i / ((uint64_t)HC * E), c_ = (i / E) % HC, d_ = i % E;
            const float hin = c->h[i];
            const float val = c->kv[(t * (HC + 1u) + HC) * E + d_];
            /* Bound: one bf16 grid step at the output magnitude (the rounding
             * boundary flip) plus the propagated gate error relative to the
             * terms being summed.  The device's gate runs f32 lane/tree sums
             * and the fast-math expf against the emulation's f64; h and
             * gate*val cancel down to ~1e-4 on some elements, where a ~1e-6
             * absolute gate difference is worth several steps of the OUTPUT's
             * own ulp — measuring against the output alone would fail a
             * correct kernel (measured worst: 9.5e-7 at a 1.5e-4 output,
             * val 0.32, i.e. 3e-6 relative to the update scale). */
            const float out_m = fmaxf(fabsf(got[i]), fabsf(want[i]));
            const float term_m = fmaxf(fmaxf(out_m, fabsf(hin)), fabsf(val));
            const float bound = 1e-5f * term_m + bf16_ulp_at(out_m);
            const float d = fabsf(got[i] - want[i]);
            if (d > bound) over_bound++;
            const float dist = bf16_ulp_at(out_m) > 0.0f ? d / bf16_ulp_at(out_m) : 0.0f;
            if (dist > max_ulp_dist) { max_ulp_dist = dist; worst_d = d; worst_i = i; }
        }
        if (n_diff) {
            const uint64_t t = worst_i / ((uint64_t)HC * E), c_ = (worst_i / E) % HC, d_ = worst_i % E;
            const float val = c->kv[(t * (HC + 1u) + HC) * E + d_];
            fprintf(stderr, "  worst: tok %llu hc %llu d %llu got %a want %a |d| %.3e ulp %.3e val %a\n",
                    (unsigned long long)t, (unsigned long long)c_, (unsigned long long)d_,
                    got[worst_i], want[worst_i], worst_d, bf16_ulp_at(fmaxf(fabsf(got[worst_i]), fabsf(want[worst_i]))), val);
        }
        /* absorb must be bit-exact; dense/zero within the bound above. */
        const int pass = (strcmp(c->kind, "absorb") == 0) ? bit_exact : (over_bound == 0);
        if (!pass) n_fail++;
        printf("gate case %u %s n=%u %s (bit-exact %d diffs %d over-bound %d max ulp dist %.3f)\n",
               ci, c->kind, c->n_tok, pass ? "PASS" : "FAIL", bit_exact, n_diff, over_bound, max_ulp_dist);
        ref_off += n_hc;
        free(got);
    }
    if (ref_off != ref_bytes / 4) {
        fprintf(stderr, "test_ds41_engram: consumed %llu ref values, file has %llu\n",
                (unsigned long long)ref_off, (unsigned long long)(ref_bytes / 4));
        return 2;
    }
    printf("DS41 engram gate: %s (%u cases, %s)\n", n_fail ? "FAIL" : "PASS", n_cases,
           n_fail ? "see the failing case above" : "rows bit-exact, gate within the term-relative bound, read path live");
    return n_fail ? 1 : 0;
}
