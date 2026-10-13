/* ds41_vq_fmt.h — DQVL/DQVQ/DQV3 container and payload layout, vendored from the
 * C engine at /data/YoungAi (commit 3946dbc), file `vq_fmt.h`, so the V4.1 VQ
 * kernels and their tests share one copy of the on-disk contract.
 *
 * Layer blob (the raw `dql_vq_L%02d.bin` bytes, or the overlay GGUF tensor
 * `blk.L.ffn_exps_vq.blob`):
 *   [u32 'DQVL'][u32 ver][u32 L][u32 nexp][slot table nexp*3 u64 payload
 *   offsets (0 = absent)][payloads...]
 * Each matrix payload:
 *   [u32 'DQVQ'][u16 dim][u16 nc][u32 rows][u32 cols][codebook nc*dim f16]
 *   [g_r rows f16][index stream: dim8 -> 1 B/idx; dim4 -> 9-bit LE stream]
 * Value = codebook[idx][d] * g_r[row]. Slot `which`: 0 = w1 gate, 1 = w3 up,
 * 2 = w2 down.
 *
 * v3 (the layout the V4.1 artifact carries, measured: DQVL v3, DQV3 payload
 * d=8 nc=8192 rows=2304 cols=5120 flags=3 mnb=12 cb_off=9232): the codebook is
 * hoisted to one per layer right after the blob header (the payload points at
 * it with cb_off, E4M3 only), and the 13-bit index is split into a fixed 12-bit
 * main stream plus a 1-bit plane. The payload magic changed, so an old binary
 * reading v3 fails to recognize it here instead of mis-decoding it.
 * The slot table position (offset 16) is the same in both versions — moving it
 * would make old binaries read codebook bytes as offsets, which is random
 * garbage rather than a diagnosable failure.
 */
#ifndef DS41_VQ_FMT_H
#define DS41_VQ_FMT_H
#include <stdint.h>
#include <string.h>
#include <stdlib.h>

#define DS4VQ_BLOB_MAGIC 0x4C565144u
#define DS4VQ_MAT_MAGIC  0x51565144u   /* v2 payload: header + own codebook + gains + bitstream */
#define DS4VQ_MAT3_MAGIC 0x33565144u   /* 'DQV3' */
#define DS4VQ_BLOB_VER_MIN 2u
#define DS4VQ_BLOB_VER_MAX 3u

static inline float ds4vq_f16(uint16_t h) {
    uint32_t s = (uint32_t)(h & 0x8000u) << 16, e = (h >> 10) & 0x1F, m = h & 0x3FF, f;
    if (e == 0) { if (!m) f = s; else { e = 112; while (!(m & 0x400)) { m <<= 1; e--; } m &= 0x3FF; f = s | (e << 23) | (m << 13); } }
    else if (e == 31) f = s | 0x7F800000u | (m << 13);
    else f = s | ((e + 112) << 23) | (m << 13);
    float out; memcpy(&out, &f, 4); return out;
}

/* Blob check + payload slot offset (0 = absent). nexp is read from the header
 * field at offset 12; V4.1 carries 384 experts per layer. */
static inline uint32_t ds4vq_blob_nexp(const uint8_t *blob) {
    uint32_t n; memcpy(&n, blob + 12, 4); return n;
}
static inline uint32_t ds4vq_blob_ver(const uint8_t *blob) {
    uint32_t v; memcpy(&v, blob + 4, 4); return v;
}
/* Version whitelist: without it a future blob would be judged "valid" and
 * decoded with this version's layout — no error, just fake weights. With it,
 * an unknown version is a hard load-time stop. */
static inline int ds4vq_blob_ok(const uint8_t *blob, size_t sz) {
    if (!blob || sz < 16) return 0;
    uint32_t mg; memcpy(&mg, blob, 4);
    if (mg != DS4VQ_BLOB_MAGIC) return 0;
    const uint32_t ver = ds4vq_blob_ver(blob);
    if (ver < DS4VQ_BLOB_VER_MIN || ver > DS4VQ_BLOB_VER_MAX) return 0;
    const uint32_t nexp = ds4vq_blob_nexp(blob);
    return nexp >= 1 && nexp <= 4096 && sz >= 16 + (size_t)nexp * 3 * 8;
}
static inline uint64_t ds4vq_slot(const uint8_t *blob, int e, int which) {
    uint64_t off; memcpy(&off, blob + 16 + ((size_t)e * 3 + which) * 8, 8);
    return off;
}

/* Payload dequant -> f32 (the engine's CPU MoE fallback and the v2 test
 * oracle). out[rows*cols] row-major. 0 = success. v2 payloads only; v3's host
 * oracle is the Rust `vq.rs` decoder. */
static inline int ds4vq_dequant_f32(const uint8_t *pay, float *out, int exp_rows, int exp_cols) {
    uint32_t mg; memcpy(&mg, pay, 4);
    if (mg != DS4VQ_MAT_MAGIC) return -1;
    uint16_t dim, nc; memcpy(&dim, pay + 4, 2); memcpy(&nc, pay + 6, 2);
    uint32_t rows, cols; memcpy(&rows, pay + 8, 4); memcpy(&cols, pay + 12, 4);
    if ((int)rows != exp_rows || (int)cols != exp_cols) return -2;
    const uint8_t *cb = pay + 16;
    const uint8_t *gr = cb + (size_t)nc * dim * 2;
    const uint8_t *ix = gr + (size_t)rows * 2;
    /* Heap-allocated and width-generic: the index width follows the codebook
     * size (a hardcoded 9-bit once mis-decoded every nc=256 payload). */
    float *cbf = (float *)malloc((size_t)nc * dim * sizeof(float));
    if (!cbf) return -3;
    for (int i = 0; i < (int)nc * dim; i++) { uint16_t h; memcpy(&h, cb + 2 * (size_t)i, 2); cbf[i] = ds4vq_f16(h); }
    int nbit = 0; while ((1 << nbit) < (int)nc) nbit++; if (nbit < 1) nbit = 1;
    const uint32_t imsk = (nbit >= 32) ? 0xFFFFFFFFu : ((1u << nbit) - 1u);
    for (uint32_t r = 0; r < rows; r++) {
        uint16_t gh; memcpy(&gh, gr + 2 * (size_t)r, 2);
        float g = ds4vq_f16(gh);
        size_t i0 = (size_t)r * cols / dim, i1 = i0 + cols / dim;
        float *orow = out + (size_t)r * cols;
        for (size_t i = i0; i < i1; i++) {
            uint32_t v;
            if (nbit == 8) v = ix[i];
            else { size_t bit = i * (size_t)nbit; uint32_t w = (uint32_t)ix[bit >> 3] | ((uint32_t)ix[(bit >> 3) + 1] << 8) | ((uint32_t)ix[(bit >> 3) + 2] << 16); v = (w >> (bit & 7)) & imsk; }
            const float *c = cbf + (size_t)v * dim;
            float *o = orow + (i - i0) * dim;
            for (int d = 0; d < dim; d++) o[d] = c[d] * g;
        }
    }
    free(cbf);
    return 0;
}
#endif
