/* onebit_quant.h — GO1B: strict-binary (±1) 1-bit row quantizer for ds4 Go-domain.
 *
 * Product ① of go-onebit/SPEC.md §4: the 1-bit fixed model Theta_fix. Every routed
 * expert weight is stored as one sign bit; each matrix ROW carries a single fp16 scale.
 * Effective rate -> 1.0 bit/weight as ncols grows (the 16-bit scale is amortized over
 * the whole row).
 *
 * ============================================================================
 * BYTE LAYOUT  (authoritative — a GGUF writer and a Metal kernel must match this)
 * ============================================================================
 * A matrix is nrows x ncols float32, ROW-MAJOR (rows = output channels). Each row is
 * encoded independently into a fixed-size "row record".
 *
 * Per-row math (L2-optimal binary quantization):
 *   Given row weights w[0..ncols-1], pick signs s[j]=sign(w[j]) in {+1,-1} and ONE
 *   positive scale a minimizing  sum_j (w[j] - a*s[j])^2.  With the signs fixed to
 *   sign(w[j]), the unconstrained optimum has the closed form
 *       a = (sum_j w[j]*s[j]) / ncols = (sum_j |w[j]|) / ncols = mean(|w|).
 *   (d/da of the sum is 0 at a = <w,s>/<s,s>, and <s,s> = ncols, <w,s> = sum|w|.)
 *   Reconstruction:  w_hat[j] = a * (bit[j] ? +1 : -1).
 *
 * Row record (exactly go1b_row_bytes(ncols) bytes, NO padding between fields):
 *   offset 0 : uint16  scale       -- IEEE-754 half (fp16) of a = mean(|w|),
 *                                      stored LITTLE-ENDIAN (low byte first).
 *   offset 2 : uint32  sign_word[ ceil(ncols/32) ]   -- each stored LITTLE-ENDIAN.
 *              Element j's sign bit is in word (j/32) at bit (j%32), bit 0 = LSB.
 *              bit == 1 -> +scale,  bit == 0 -> -scale.  Convention on encode:
 *              w[j] >= 0 -> bit 1.  Bits of the final partial word with index
 *              >= ncols are ZERO-PADDED and are never read on dequant.
 *
 * Whole matrix = the nrows row records concatenated; row r begins at byte offset
 * r * go1b_row_bytes(ncols). Records are NOT individually 4-byte aligned, so all
 * multi-byte fields are accessed via explicit little-endian byte loads/stores
 * (see go1b_load_u32_le / go1b_store_u32_le) — never via a misaligned uint32* cast.
 * ============================================================================
 */
#ifndef GO1B_ONEBIT_QUANT_H
#define GO1B_ONEBIT_QUANT_H

#include <stddef.h>   /* size_t  */
#include <stdint.h>   /* int64_t, uint16_t, uint32_t */
#include <string.h>   /* memcpy  */

#ifdef __cplusplus
extern "C" {
#endif

/* ---- public API ---------------------------------------------------------- */

/* Bytes occupied by one row record for a row of `ncols` elements:
 * 2 (fp16 scale) + ceil(ncols/32)*4 (sign words). */
size_t go1b_row_bytes(int64_t ncols);

/* Quantize a full nrows x ncols row-major float32 matrix into `dst`.
 * `dst` must hold at least nrows * go1b_row_bytes(ncols) bytes.
 * Returns the total number of bytes written. */
size_t go1b_quantize(const float *src, void *dst, int64_t nrows, int64_t ncols);

/* Dequantize one row record into dst[0..ncols-1]; each value is exactly +scale or
 * -scale (the fp16-roundtripped row scale). */
void go1b_dequantize_row(const void *row, float *dst, int64_t ncols);

/* ============================================================================
 * BLOCK layout (block_go1b) — the RUNTIME on-disk format the Metal kernel reads.
 * ============================================================================
 * Unlike the per-row record above (one scale + a variable-length sign-word tail),
 * this is a fixed-size 256-element block so the GPU can stream uniform records:
 *
 *     struct block_go1b { uint16_t d;  uint8_t signs[32]; };   // 34 bytes, packed
 *
 * A matrix is [nrows x ncols], ROW-MAJOR, with ncols % 256 == 0. The per-ROW scale
 *     d_row = mean(|w|) over the WHOLE row (all ncols)
 * is stored as fp16 and PHYSICALLY REPLICATED into the `d` field of every one of
 * the ncols/256 blocks belonging to that row (the kernel reads `d` per block and
 * never needs the rest of the row). Sign packing within a block: element j (0..255)
 * -> byte signs[j/8], bit (j%8); bit==1 iff w[j] >= 0 (dequant gives +d), bit==0 ->
 * -d. Any bit past the data (only possible in a defensive partial last block) is 0.
 * Whole tensor = nrows * (ncols/256) * 34 bytes, blocks concatenated row-major.
 * On-disk GGUF ggml type number = 40 (see quants.h DS4Q_TYPE_GO1B_BLK).
 * ============================================================================ */
#define GO1B_BLK_QK    256    /* elements per fixed block               */
#define GO1B_BLK_BYTES 34     /* sizeof(struct block_go1b): 2 + 32      */

/* Bytes for one row of `ncols` columns (ncols a multiple of 256): (ncols/256)*34. */
size_t go1b_blk_row_bytes(int64_t ncols);

/* Quantize a full nrows x ncols row-major float32 matrix into the 256-block layout.
 * `dst` must hold at least nrows * go1b_blk_row_bytes(ncols) bytes. The per-row scale
 * mean(|w|) is replicated into each 256-block. Returns total bytes written. */
size_t go1b_blk_quantize(const float *src, void *dst, int64_t nrows, int64_t ncols);

/* Input-aware (L_fix): ew[ncols] = per-channel Go activation second moments
 * E[x²] (the per-expert imatrix slice). Sign stays sign(w); the scale re-fits
 * closed-form s* = Σ ew·|w| / Σ ew (per-row, or block-local under
 * per-block 模式(原 DS4_GO1B_PER_BLOCK env, 已随 env 大扫除拔除)). ew=NULL / zero mass → plain go1b_blk_quantize. */
/* joint-LS 输出最优 per-block scale (79% 质量关键): X[n_act×ncols] 行主序激活;
 * per-row 联合解 nblk block scale 使 Σ_b s_b·(B_b·x)≈w·x。X=NULL→退回 mean|w|。 */
size_t go1b_blk_quantize_joint(const float *src, void *dst, int64_t nrows, int64_t ncols,
                               const float *X, int64_t n_act);
size_t go1b_blk_quantize_imat(const float *src, void *dst, int64_t nrows, int64_t ncols,
                              const float *ew);

/* Dequantize one full row (all ncols) from the block layout into dst[0..ncols-1];
 * each value is exactly +d or -d (the fp16-roundtripped row scale stored per block). */
void go1b_blk_dequantize_row(const void *row, float *dst, int64_t ncols);

/* ---- minimal inline fp16 <-> fp32 helpers (no external libs, no math.h) ----
 * Branch-light round-to-nearest-even conversion (Fabian Giesen's method, the same
 * one used by ggml/PyTorch). Correct for normals, subnormals, overflow->inf, NaN. */

static inline uint32_t go1b_fp32_to_bits(float f) {
    uint32_t w; memcpy(&w, &f, sizeof w); return w;
}
static inline float go1b_bits_to_fp32(uint32_t w) {
    float f; memcpy(&f, &w, sizeof f); return f;
}

static inline uint16_t go1b_fp32_to_fp16(float f) {
    /* scale_to_inf * scale_to_zero == 1, but the intermediate over/underflows force
     * correct rounding of the low mantissa bits without explicit bit fiddling. */
    const float scale_to_inf  = 0x1.0p+112f;
    const float scale_to_zero = 0x1.0p-110f;
    uint32_t w    = go1b_fp32_to_bits(f);
    float    absf = go1b_bits_to_fp32(w & 0x7fffffffu); /* |f| via clearing sign bit */
    float    base = (absf * scale_to_inf) * scale_to_zero;

    uint32_t shl1_w = w + w;                  /* logical-left-1: drops sign, doubles */
    uint32_t sign   = w & 0x80000000u;
    uint32_t bias   = shl1_w & 0xff000000u;
    if (bias < 0x71000000u) bias = 0x71000000u;

    base = go1b_bits_to_fp32((bias >> 1) + 0x07800000u) + base;
    uint32_t bits      = go1b_fp32_to_bits(base);
    uint32_t exp_bits  = (bits >> 13) & 0x00007c00u;
    uint32_t mant_bits = bits & 0x00000fffu;
    uint32_t nonsign   = exp_bits + mant_bits;
    /* shl1_w > 0xff000000 means |f| was inf/NaN -> emit fp16 quiet NaN 0x7e00. */
    return (uint16_t)((sign >> 16) | (shl1_w > 0xff000000u ? 0x7e00u : nonsign));
}

static inline float go1b_fp16_to_fp32(uint16_t h) {
    uint32_t w     = (uint32_t)h << 16;
    uint32_t sign  = w & 0x80000000u;
    uint32_t two_w = w + w;

    uint32_t exp_offset = 0xe0u << 23;
    const float exp_scale = 0x1.0p-112f;
    float normalized = go1b_bits_to_fp32((two_w >> 4) + exp_offset) * exp_scale;

    uint32_t magic_mask = 126u << 23;
    const float magic_bias = 0.5f;
    float denormalized = go1b_bits_to_fp32((two_w >> 17) | magic_mask) - magic_bias;

    uint32_t denorm_cutoff = 1u << 27; /* below this two_w is a fp16 subnormal */
    uint32_t result = sign | (two_w < denorm_cutoff
        ? go1b_fp32_to_bits(denormalized)
        : go1b_fp32_to_bits(normalized));
    return go1b_bits_to_fp32(result);
}

/* ---- little-endian byte (de)serialization helpers ------------------------
 * Endian-independent + alignment-safe; the on-disk format is fixed little-endian
 * regardless of host byte order. */

static inline void go1b_store_u16_le(void *p, uint16_t v) {
    uint8_t *b = (uint8_t *)p;
    b[0] = (uint8_t)(v & 0xffu);
    b[1] = (uint8_t)((v >> 8) & 0xffu);
}
static inline uint16_t go1b_load_u16_le(const void *p) {
    const uint8_t *b = (const uint8_t *)p;
    return (uint16_t)((uint16_t)b[0] | ((uint16_t)b[1] << 8));
}
static inline void go1b_store_u32_le(void *p, uint32_t v) {
    uint8_t *b = (uint8_t *)p;
    b[0] = (uint8_t)(v & 0xffu);
    b[1] = (uint8_t)((v >> 8) & 0xffu);
    b[2] = (uint8_t)((v >> 16) & 0xffu);
    b[3] = (uint8_t)((v >> 24) & 0xffu);
}
static inline uint32_t go1b_load_u32_le(const void *p) {
    const uint8_t *b = (const uint8_t *)p;
    return (uint32_t)b[0]
         | ((uint32_t)b[1] << 8)
         | ((uint32_t)b[2] << 16)
         | ((uint32_t)b[3] << 24);
}

#ifdef __cplusplus
}
#endif

#endif /* GO1B_ONEBIT_QUANT_H */
