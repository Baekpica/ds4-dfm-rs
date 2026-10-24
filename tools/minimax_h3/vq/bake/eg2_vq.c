/* eg2_vq.c — dense-matrix VQ encoder for EmbeddingGemma-2 (unit 2: encoder +
 * known-answer test).
 *
 * Nothing in the payload format is invented. Every field is copied from the
 * reference toolchain, cited with file:line:
 *
 *   /data/YoungAi @ a4e6d31
 *     gguf-tools/quantize/vq_qc.h:86      vq_encode_full — kmeans (strided
 *         subsample <= 200k vectors, 8 Lloyd rounds, codebook round-tripped
 *         through f16 before assignment, line 109), GPTQ over column groups of
 *         128 with 1% diagonal damping, channel weighting 1/diag(H^-1),
 *         per-row gains by closed-form ridge clamped to [0.25, 4.0].
 *         X = NULL is the supported "nocal" path (line 108: XbT stays NULL,
 *         feedback off, row gains skipped at line 138).
 *     gguf-tools/quantize/vq_qc.h:241     vq_pack      — payload writer
 *     gguf-tools/quantize/vq_qc.h:268     vq_unpack_dequant — payload reader
 *     gguf-tools/quantize/vq_qc_bytes.h   vq_idx_bytes / vq_payload_bytes, the
 *         single byte account shared by writer and allocator: 16 B header
 *         ('DQVQ', u16 dim, u16 nc, u32 rows, u32 cols) + nc*dim f16 codebook +
 *         rows f16 gains + index stream, index width ceil(log2 nc), 8-bit
 *         stored one byte per index and wider widths as a LE bitstream plus one
 *         guard byte.
 *     gguf-tools/quantize/go2b_qc.h:70    g2_inv (group inverse for GPTQ)
 *     gguf-tools/quantize/onebit_quant.h  go1b_fp32_to_fp16 / go1b_fp16_to_fp32
 *     src/common/ds4_st41.h:133,163,169   v41_st_open / v41_st_find /
 *         v41_st_data — the format-generic multi-shard safetensors index
 *         (locate only, no decode), so the reader is not reimplemented.
 *     src/common/ds4_float.h:34           ds4_bf16_to_f32
 *
 * /data/ds4-dfm-rs/ds41_vq_fmt.h:7     value rule:
 *         value[row][col] = codebook[idx][col % dim] * gain[row]
 *
 * Build (mirrors the reference recipe's CPU path; no CUDA, because this
 * machine's GPU path assumes unified memory and measured pageableMemoryAccess
 * = 0 — see vq_gpu.cu's header comment):
 *
 *   cc -O3 -std=c99 -Wall -D_POSIX_C_SOURCE=200809L -I/data/YoungAi/gguf-tools/quantize \
 *      -o bin/eg2_vq tools/eg2_vq.c -lm
 *
 * Usage:
 *   eg2_vq selftest
 *   eg2_vq tensor <hf-dir> <tensor-name> [dim nc] [--dump FILE]
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>

/* dq_matmul must be defined before vq_qc.h (it calls it). Semantic copied from
 * the reference: /data/YoungAi/gguf-tools/quantize/vq_shim.c:9
 *   out[S,M] = X[S,K] @ W[M,K]^T */
static void dq_matmul(const float *X, const float *W, float *out, int S, int K, int M) {
    for (int s = 0; s < S; s++)
        for (int m = 0; m < M; m++) {
            const float *x = X + (size_t)s * K, *w = W + (size_t)m * K;
            double acc = 0;
            for (int k = 0; k < K; k++) acc += (double)x[k] * w[k];
            out[(size_t)s * M + m] = (float)acc;
        }
}

#include "onebit_quant.c"   /* go1b_f16 helpers */
#include "go2b_qc.h"        /* g2_inv */
#include "vq_qc.h"          /* vq_encode_full, vq_pack, vq_unpack_dequant */
#include "ds4_st41.h"   /* local edit: was ../../YoungAi/src/common/, resolved by -I youngai/src/common */
#include "ds4_float.h"   /* local edit, same reason */

/* vq_qc.h declares these extern (line 369); the plan-table globals only feed
 * the per-layer sidecar path, so zero is the documented default. */
int g_vq_dim = 0, g_vq_nc = 0;

static double now_s(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }

/* Metrics of a reconstruction: cosine, relative L2, max abs error. */
static void metrics(const float *W, const float *Wq, size_t n, const char *tag, int dim, int nc, double secs) {
    double dot = 0, na = 0, nb = 0, se = 0, maxe = 0;
    for (size_t i = 0; i < n; i++) {
        double a = W[i], b = Wq[i];
        dot += a * b; na += a * a; nb += b * b;
        double d = a - b; se += d * d; if (fabs(d) > maxe) maxe = fabs(d);
    }
    const double cos = dot / (sqrt(na) * sqrt(nb) + 1e-30);
    const double rel = sqrt(se / (na + 1e-30));
    printf("%-52s dim=%d nc=%4d  cos=%.4f  relL2=%7.4f%%  maxabs=%.5f  n=%zu  %.2fs\n",
           tag, dim, nc, cos, 100.0 * rel, maxe, n, secs);
}

/* Encode one matrix with the reference encoder and measure the round trip. */
static int encode_and_check(const char *tag, const float *W, int rows, int cols, int dim, int nc,
                            const char *dump_path) {
    if (cols % dim) { fprintf(stderr, "%s: cols %d not divisible by dim %d\n", tag, cols, dim); return -1; }
    const size_t n = (size_t)rows * cols;
    float *Cb = malloc((size_t)nc * dim * 4);
    float *gr = malloc((size_t)rows * 4);
    int *idx = malloc((n / dim) * sizeof(int));
    float *Wq = malloc(n * 4);
    const double t0 = now_s();
    /* X = NULL, n_act = 0: the nocal path the DeepSeek recipe records
     * ("Calibration data: None; the codebooks are solved on the weights alone"). */
    vq_encode_full(W, rows, cols, dim, nc, NULL, 0, Cb, idx, gr, Wq);
    const double t1 = now_s();
    const size_t psz = vq_payload_bytes(rows, cols, dim, nc);
    uint8_t *buf = malloc(psz);
    const size_t wrote = vq_pack(Cb, idx, gr, rows, cols, dim, nc, buf);
    if (wrote != psz) { fprintf(stderr, "%s: pack wrote %zu, account says %zu\n", tag, wrote, psz); return -1; }
    float *Wback = malloc(n * 4);
    int r2 = 0, c2 = 0;
    if (vq_unpack_dequant(buf, wrote, Wback, &r2, &c2) != 0) { fprintf(stderr, "%s: unpack failed\n", tag); return -1; }
    if (r2 != rows || c2 != cols) { fprintf(stderr, "%s: unpack shape %dx%d != %dx%d\n", tag, r2, c2, rows, cols); return -1; }
    metrics(W, Wq, n, tag, dim, nc, t1 - t0);
    printf("    payload %zu B (%.4f bpw)  decode-vs-encode maxabs=%.3g\n", wrote,
           8.0 * (double)wrote / (double)n, ({
        double m = 0; for (size_t i = 0; i < n; i++) { double d = fabs(Wback[i] - Wq[i]); if (d > m) m = d; } m; }));
    if (dump_path) {
        FILE *f = fopen(dump_path, "wb");
        if (!f) { fprintf(stderr, "cannot write %s\n", dump_path); return -1; }
        fwrite(buf, 1, wrote, f);
        fclose(f);
        printf("    payload -> %s\n", dump_path);
    }
    free(Cb); free(gr); free(idx); free(Wq); free(Wback); free(buf);
    return 0;
}

int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "usage: eg2_vq selftest | eg2_vq tensor <hf-dir> <name> [dim nc]\n"); return 2; }

    if (!strcmp(argv[1], "selftest")) {
        /* Deterministic PRNG matrix so the known-answer test is reproducible. */
        const int rows = 512, cols = 512;
        const size_t n = (size_t)rows * cols;
        float *W = malloc(n * 4);
        uint64_t s = 0x123456789abcdefULL;
        for (size_t i = 0; i < n; i++) {
            s = s * 6364136223846793005ULL + 1442695040888963407ULL;
            W[i] = (float)((double)(int32_t)(s >> 33) / 2147483648.0) * 0.05f;
        }
        printf("selftest: deterministic %dx%d matrix, X=NULL (nocal)\n", rows, cols);
        if (encode_and_check("selftest/hot-4x512", W, rows, cols, 4, 512, NULL)) return 1;
        if (encode_and_check("selftest/cold-8x256", W, rows, cols, 8, 256, NULL)) return 1;
        free(W);
        return 0;
    }

    if (!strcmp(argv[1], "tensor") && argc >= 4) {
        const char *dir = argv[2], *name = argv[3];
        int dim = 4, nc = 512;
        const char *dump = NULL;
        for (int i = 4; i < argc; i++) {
            if (!strcmp(argv[i], "--dump") && i + 1 < argc) dump = argv[++i];
            else if (!dim || 1) { if (argc - i >= 2) { dim = atoi(argv[i]); nc = atoi(argv[i + 1]); i++; } }
        }
        v41_st S;
        if (v41_st_open(&S, dir)) { fprintf(stderr, "cannot open %s\n", dir); return 1; }
        const v41_st_ent *E = v41_st_find(&S, name);
        if (!E) { fprintf(stderr, "tensor not found: %s\n", name); return 1; }
        if (E->nd != 2) { fprintf(stderr, "%s: nd=%d, only 2-D matrices are VQ targets\n", name, E->nd); return 1; }
        const int rows = (int)E->shape[0], cols = (int)E->shape[1];
        const uint8_t *raw = v41_st_data(&S, E);
        if (!raw) return 1;
        const size_t n = (size_t)rows * cols;
        float *W = malloc(n * 4);
        if (!strcmp(E->dtype, "BF16")) {
            for (size_t i = 0; i < n; i++) W[i] = ds4_bf16_to_f32(((const uint16_t *)raw)[i]);
        } else if (!strcmp(E->dtype, "F32")) {
            memcpy(W, raw, n * 4);
        } else { fprintf(stderr, "%s: dtype %s not handled (BF16/F32 only)\n", name, E->dtype); return 1; }
        printf("tensor %s  %dx%d  dtype=%s  (%.1f MB bf16)\n", name, rows, cols, E->dtype, 2.0 * n / 1e6);
        int rc = encode_and_check(name, W, rows, cols, dim, nc, dump);
        v41_st_close(&S);
        free(W);
        return rc ? 1 : 0;
    }

    fprintf(stderr, "unknown mode\n");
    return 2;
}
