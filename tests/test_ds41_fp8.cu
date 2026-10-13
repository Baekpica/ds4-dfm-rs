/* test_ds41_fp8.cu — P4-3 gate: the fp8_32x32 entries
 * (ds4_gpu_v41_matmul_fp8blk_tensor / _round_ / _grouped_) against the Rust
 * emulation (crates/ds4-core/examples/ds41_fp8_ref.rs).
 *
 *   test_ds41_fp8 <fp8.img> <fp8.cases.txt> <fp8.ref.f32>
 *
 * The fixture (tests/fixtures/ds41/fp8/gen_fp8.py) carries two case kinds:
 *   onehot — a single 1.0 per token, so every dot is a single-term sum and the
 *            entry must be BIT-EXACT against the emulation.
 *   dense  — full activations: the f32 accumulation order (warp shuffles, FMA
 *            contraction) differs from the emulation's f64 sum, so the compare
 *            is within TOL_SCALE of the reference vector's own scale.
 *
 * Round-out cases add one bf16 quantum to the allowance: the final bf16
 * rounding can flip a value that sits on a bf16 boundary (the same mechanism
 * the P4-2 criterion records, one rounding stage shorter here).
 *
 * The image is a malloc'd "model map": the entries resolve their device
 * pointer through the native range resolver, exactly the path the forward
 * will take.
 */
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "ds4_gpu.h"

/* Dense cases: max |delta| against the reference vector's own scale. */
#define TOL_SCALE 1e-3f

typedef struct {
    char kind[16];
    char entry[16];
    uint32_t n_tok;
    int round_out;
    uint32_t col;
    float *x;
} fp8_case;

typedef struct {
    char name[16];
    uint64_t off;
    uint32_t rows, cols;      /* plain: out/in; grouped: groups*rank / gdim */
    uint32_t groups, gdim, rank;
} fp8_tensor;

static void *xread(const char *path, uint64_t *size) {
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "test_ds41_fp8: cannot open %s\n", path); exit(2); }
    fseek(f, 0, SEEK_END);
    const long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    /* Page-aligned: the image is the "model map", and the range resolver's
     * register tier page-rounds ranges — a non-aligned base makes neighbor
     * tensors' registrations overlap (ds4_cuda.cu:1795-1803). */
    void *buf = NULL;
    if (posix_memalign(&buf, 4096, (size_t)n) != 0 || !buf) { fprintf(stderr, "test_ds41_fp8: alloc %s\n", path); exit(2); }
    if (fread(buf, 1, (size_t)n, f) != (size_t)n) { fprintf(stderr, "test_ds41_fp8: read %s\n", path); exit(2); }
    fclose(f);
    *size = (uint64_t)n;
    return buf;
}

static uint32_t hex32(const char *s) { return (uint32_t)strtoul(s, NULL, 16); }

/* bf16 ulp at |x| (7 explicit mantissa bits: 2^(e-8) for x = m*2^e, m in [1,2)). */
static float bf16_ulp_at(float x) {
    if (!(x > 0.0f)) return 0.0f;
    int e = 0;
    (void)frexpf(x, &e);
    return ldexpf(1.0f, e - 9);
}

static fp8_case *parse_cases(const char *path, fp8_tensor *tensors, uint32_t *n_tensors, uint32_t *n_cases) {
    FILE *f = fopen(path, "r");
    if (!f) { fprintf(stderr, "test_ds41_fp8: cannot open %s\n", path); exit(2); }
    fp8_case *cases = NULL;
    uint32_t nt = 0, n = 0, cap = 0;
    /* getline, not a fixed buffer: an x line is n_tok*in_dim hex words (up to
     * ~110 KB here), and a short fgets buffer silently truncates the
     * activation — the P4-2 harness lesson. */
    char *line = NULL;
    size_t line_cap = 0;
    while (getline(&line, &line_cap, f) > 0) {
        if (line[0] == '#' || line[0] == '\n') continue;
        if (strncmp(line, "tensor ", 7) == 0) {
            fp8_tensor *t = &tensors[nt++];
            memset(t, 0, sizeof *t);
            t->groups = 1;
            char *tok = strtok(line + 7, " \t\n");
            snprintf(t->name, sizeof t->name, "%s", tok ? tok : "");
            while ((tok = strtok(NULL, " \t\n"))) {
                char k[16] = {0};
                unsigned long long v = 0;
                if (sscanf(tok, "%15[^=]=%llu", k, &v) != 2) { fprintf(stderr, "test_ds41_fp8: bad tensor line\n"); exit(2); }
                if (strcmp(k, "off") == 0) t->off = v;
                else if (strcmp(k, "in") == 0) t->cols = (uint32_t)v;
                else if (strcmp(k, "out") == 0) t->rows = (uint32_t)v;
                else if (strcmp(k, "groups") == 0) t->groups = (uint32_t)v;
                else if (strcmp(k, "gdim") == 0) t->gdim = (uint32_t)v;
                else if (strcmp(k, "rank") == 0) t->rank = (uint32_t)v;
            }
            if (t->groups > 1) { t->rows = t->groups * t->rank; t->cols = t->gdim; }
            continue;
        }
        if (strncmp(line, "case ", 5) == 0) {
            if (n == cap) { cap = cap ? cap * 2u : 8u; cases = (fp8_case *)realloc(cases, cap * sizeof *cases); }
            fp8_case *c = &cases[n++];
            memset(c, 0, sizeof *c);
            unsigned idx = 0;
            if (sscanf(line, "case %u %15s entry=%15s n=%u round=%d col=%u",
                       &idx, c->kind, c->entry, &c->n_tok, &c->round_out, &c->col) != 6) {
                fprintf(stderr, "test_ds41_fp8: bad case line\n"); exit(2);
            }
            continue;
        }
        fp8_case *c = &cases[n - 1];
        char *tok = strtok(line, " \t\n");
        if (tok && strcmp(tok, "x") == 0) {
            const fp8_tensor *t = NULL;
            for (uint32_t i = 0; i < nt; i++) if (strcmp(tensors[i].name, strcmp(c->entry, "grouped") == 0 ? "grouped" : "plain") == 0) t = &tensors[i];
            const uint32_t in_dim = t->groups > 1 ? t->groups * t->gdim : t->cols;
            c->x = (float *)malloc((size_t)c->n_tok * in_dim * 4);
            for (uint32_t i = 0; i < c->n_tok * in_dim && (tok = strtok(NULL, " \t\n")); i++) {
                uint32_t b = hex32(tok);
                memcpy(&c->x[i], &b, 4);
            }
        }
    }
    free(line);
    fclose(f);
    *n_tensors = nt;
    *n_cases = n;
    return cases;
}

int main(int argc, char **argv) {
    if (argc != 4) {
        fprintf(stderr, "usage: %s <fp8.img> <fp8.cases.txt> <fp8.ref.f32>\n", argv[0]);
        return 2;
    }
    uint64_t img_bytes = 0, ref_bytes = 0;
    void *img = xread(argv[1], &img_bytes);
    float *ref = (float *)xread(argv[3], &ref_bytes);
    fp8_tensor tensors[8];
    uint32_t n_tensors = 0, n_cases = 0;
    fp8_case *cases = parse_cases(argv[2], tensors, &n_tensors, &n_cases);

    const fp8_tensor *plain = NULL, *grouped = NULL;
    for (uint32_t i = 0; i < n_tensors; i++) {
        if (strcmp(tensors[i].name, "plain") == 0) plain = &tensors[i];
        if (strcmp(tensors[i].name, "grouped") == 0) grouped = &tensors[i];
    }
    if (!plain || !grouped) { fprintf(stderr, "test_ds41_fp8: image is missing a tensor\n"); return 2; }
    /* The >8 arm runs cuBLAS; the test never boots the engine, so create the
     * handle the way boot does. */
    if (!ds4_gpu_init()) { fprintf(stderr, "test_ds41_fp8: ds4_gpu_init failed\n"); return 2; }

    ds4_gpu_tensor *x_t = ds4_gpu_tensor_alloc((uint64_t)plain->cols * 12u * 4u);
    ds4_gpu_tensor *out_t = ds4_gpu_tensor_alloc((uint64_t)plain->rows * 12u * 4u);
    if (!x_t || !out_t) { fprintf(stderr, "test_ds41_fp8: tensor alloc\n"); return 2; }

    int n_fail = 0;
    uint64_t ref_off = 0;
    for (uint32_t ci = 0; ci < n_cases; ci++) {
        fp8_case *c = &cases[ci];
        const int is_grouped = strcmp(c->entry, "grouped") == 0;
        const fp8_tensor *t = is_grouped ? grouped : plain;
        const uint32_t in_dim = is_grouped ? t->groups * t->gdim : t->cols;
        const uint32_t out_dim = is_grouped ? t->groups * t->rank : t->rows;
        const uint64_t n_out = (uint64_t)c->n_tok * out_dim;
        ds4_gpu_tensor_write(x_t, 0, c->x, (uint64_t)c->n_tok * in_dim * 4);
        int rc = 0;
        if (is_grouped)
            rc = ds4_gpu_v41_grouped_matmul_fp8blk_tensor(out_t, img, img_bytes, t->off, t->groups, t->gdim, t->rank,
                                                          x_t, c->n_tok, c->round_out);
        else if (strcmp(c->entry, "round") == 0)
            rc = ds4_gpu_v41_matmul_fp8blk_round_tensor(out_t, img, img_bytes, t->off, in_dim, out_dim,
                                                        x_t, c->n_tok, c->round_out);
        else
            rc = ds4_gpu_v41_matmul_fp8blk_tensor(out_t, img, img_bytes, t->off, in_dim, out_dim, x_t, c->n_tok);
        if (!rc) { fprintf(stderr, "test_ds41_fp8: case %u: entry refused\n", ci); n_fail++; ref_off += n_out; continue; }
        float *got = (float *)malloc(n_out * 4);
        ds4_gpu_tensor_read(out_t, 0, got, n_out * 4);
        const float *want = ref + ref_off;

        int bit_exact = 1, n_diff = 0;
        float max_abs = 0.0f, max_ref = 0.0f;
        for (uint64_t i = 0; i < n_out; i++) {
            const float d = fabsf(got[i] - want[i]);
            if (got[i] != want[i]) { bit_exact = 0; n_diff++; }
            if (d > max_abs) max_abs = d;
            if (fabsf(want[i]) > max_ref) max_ref = fabsf(want[i]);
        }
        const int onehot = strcmp(c->kind, "onehot") == 0;
        /* Dense: the f64 emulation vs the device's f32 accumulation — the
         * reorder error scales with the sum's magnitude, so the criterion is
         * relative to the reference vector's own max. Round-out cases add one
         * bf16 quantum: the final rounding can flip a boundary value. */
        const int pass = onehot ? bit_exact
                                : (max_abs <= TOL_SCALE * max_ref || (c->round_out && max_abs <= bf16_ulp_at(max_ref)));
        if (!pass) n_fail++;
        printf("case %u %s entry=%s n=%u round=%d %s (bit-exact %d max rel %.3e max abs %.3e scale %.3e diffs %d)\n",
               ci, c->kind, c->entry, c->n_tok, c->round_out, pass ? "PASS" : "FAIL",
               bit_exact, max_ref > 0.0f ? max_abs / max_ref : 0.0f, max_abs, max_ref, n_diff);
        ref_off += n_out;
        free(got);
    }
    if (ref_off != ref_bytes / 4) {
        fprintf(stderr, "test_ds41_fp8: consumed %llu ref values, file has %llu\n",
                (unsigned long long)ref_off, (unsigned long long)(ref_bytes / 4));
        return 2;
    }
    printf("DS41 FP8 gate: %s (%u cases, %s)\n", n_fail ? "FAIL" : "PASS", n_cases,
           n_fail ? "see the failing case above" : "onehot bit-exact, dense within TOL_SCALE");
    return n_fail ? 1 : 0;
}
