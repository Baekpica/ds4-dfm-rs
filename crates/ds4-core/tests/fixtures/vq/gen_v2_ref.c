/* gen_v2_ref.c — P1 parity harness for the VQ decoder.
 *
 * The engine's own host decoder (vq_fmt.h, vendored next to this file from
 * /data/YoungAi at 3946dbc) is the authority for the v2 payload: this program
 * builds deterministic blobs, decodes them with ds4vq_dequant_f32, and writes
 * both the blob bytes and the reference values. The Rust decoder must then
 * reproduce the reference byte for byte on the same bytes.
 *
 * The v3 payload has no host decoder in the engine (its geometry lives in the
 * device kernel cuda_vq_row.inc.cu), so v3 is gated by the per-layer traces of
 * the golden set instead, not here.
 *
 *   cc -O2 -I. gen_v2_ref.c -o gen_v2_ref && ./gen_v2_ref
 *
 * Writes: v2_nc256.blob / v2_nc256.ref.f32, v2_nc4096.blob / v2_nc4096.ref.f32
 */
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "vq_fmt.h"

#define BLOB_MAGIC 0x4C565144u
#define MAT_MAGIC 0x51565144u

static void put_u16(uint8_t *p, uint16_t v) { memcpy(p, &v, 2); }
static void put_u32(uint8_t *p, uint32_t v) { memcpy(p, &v, 4); }
static void put_u64(uint8_t *p, uint64_t v) { memcpy(p, &v, 8); }

/* Deterministic content: codebook entry (idx,d) is an exact half value,
 * gains cycle over four exact halves, indices walk the codebook. */
static uint16_t f16_bits(float f)
{
    uint32_t b;
    memcpy(&b, &f, 4);
    uint32_t sign = (b >> 16) & 0x8000u;
    int32_t exp = (int32_t)((b >> 23) & 0xFFu) - 127 + 15;
    uint32_t man = (b >> 13) & 0x3FFu;
    if (exp <= 0 || exp >= 31) {
        fprintf(stderr, "value %f is not a normal half\n", (double)f);
        exit(1);
    }
    return (uint16_t)(sign | ((uint32_t)exp << 10) | man);
}

static size_t build_blob(uint8_t *buf, uint32_t nc, uint32_t rows, uint32_t cols)
{
    const uint32_t dim = 8;
    const uint32_t nexp = 2;
    uint8_t *hdr = buf;
    put_u32(hdr, BLOB_MAGIC);
    put_u32(hdr + 4, 2);       /* version */
    put_u32(hdr + 8, 0);
    put_u32(hdr + 12, nexp);
    const size_t table = 16;
    uint8_t *pay = buf + table + (size_t)nexp * 3 * 8;
    put_u64(buf + table + 0 * 8, (uint64_t)(pay - buf));  /* expert 0 w1 */
    put_u64(buf + table + 1 * 8, 0);
    put_u64(buf + table + 2 * 8, 0);
    put_u64(buf + table + 3 * 8, 0);                      /* expert 1 absent */
    put_u64(buf + table + 4 * 8, 0);
    put_u64(buf + table + 5 * 8, 0);

    put_u32(pay, MAT_MAGIC);
    put_u16(pay + 4, (uint16_t)dim);
    put_u16(pay + 6, (uint16_t)nc);
    put_u32(pay + 8, rows);
    put_u32(pay + 12, cols);
    uint8_t *cb = pay + 16;
    for (uint32_t w = 0; w < nc; w++) {
        for (uint32_t d = 0; d < dim; d++) {
            float v = (float)((w * dim + d) % 251 + 1) * 0.5f;
            put_u16(cb + ((size_t)w * dim + d) * 2, f16_bits(v));
        }
    }
    uint8_t *gr = cb + (size_t)nc * dim * 2;
    const float gains[4] = { 0.25f, 0.5f, 1.0f, 2.0f };
    for (uint32_t r = 0; r < rows; r++) {
        put_u16(gr + (size_t)r * 2, f16_bits(gains[r % 4]));
    }
    uint8_t *ix = gr + (size_t)rows * 2;
    const uint32_t nidx = rows * cols / dim;
    int nbit = 0;
    while ((1u << nbit) < nc) nbit++;
    if (nbit < 1) nbit = 1;
    if (nbit == 8) {
        for (uint32_t i = 0; i < nidx; i++) ix[i] = (uint8_t)((i * 37) % nc);
        /* the engine's payloads carry one safety byte past the packed bits */
        ix[nidx] = 0;
        return (size_t)(ix - buf) + nidx + 1;
    }
    size_t bits = (size_t)nidx * nbit;
    size_t bytes = (bits + 7) / 8;
    memset(ix, 0, bytes + 3);
    for (uint32_t i = 0; i < nidx; i++) {
        uint32_t v = (i * 37) % nc;
        size_t bit = (size_t)i * nbit;
        for (int b = 0; b < nbit; b++) {
            if ((v >> b) & 1u) {
                ix[(bit + b) >> 3] |= (uint8_t)(1u << ((bit + b) & 7));
            }
        }
    }
    return (size_t)(ix - buf) + bytes + 3;
}

static int gen(const char *stem, uint32_t nc, uint32_t rows, uint32_t cols)
{
    size_t cap = (size_t)nc * 8 * 2 + rows * 2 + (size_t)rows * cols * 2 + 64;
    uint8_t *blob = calloc(1, cap);
    size_t n = build_blob(blob, nc, rows, cols);
    if (!ds4vq_blob_ok(blob, n)) {
        fprintf(stderr, "%s: blob_ok rejected the fixture\n", stem);
        return 1;
    }
    const uint8_t *pay = blob + (size_t)ds4vq_slot(blob, 0, 0);
    float *out = calloc((size_t)rows * cols, sizeof(float));
    /* the engine's decoders return 0 on success */
    int rc = ds4vq_dequant_f32(pay, out, (int)rows, (int)cols);
    if (rc != 0) {
        fprintf(stderr, "%s: dequant refused (%d)\n", stem, rc);
        return 1;
    }
    char path[512];
    snprintf(path, sizeof(path), "%s.blob", stem);
    FILE *f = fopen(path, "wb");
    if (!f || fwrite(blob, 1, n, f) != n) { fprintf(stderr, "write %s failed\n", path); return 1; }
    fclose(f);
    snprintf(path, sizeof(path), "%s.ref.f32", stem);
    f = fopen(path, "wb");
    if (!f || fwrite(out, sizeof(float), (size_t)rows * cols, f) != (size_t)rows * cols) {
        fprintf(stderr, "write %s failed\n", path);
        return 1;
    }
    fclose(f);
    printf("%s: nc=%u rows=%u cols=%u blob=%zu bytes\n", stem, nc, rows, cols, n);
    free(out);
    free(blob);
    return 0;
}

int main(void)
{
    int rc = 0;
    rc |= gen("v2_nc256", 256, 16, 64);    /* 8-bit index path */
    rc |= gen("v2_nc4096", 4096, 16, 64);  /* 12-bit bit-window path */
    return rc;
}
