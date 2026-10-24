/* h3_gvq.h — the GVQ1 v1 matrix payload and the GVR8 residual stage, for H3.
 *
 * Rung A of docs/minimax-h3/vq-recipe.md section 2: dim 8, K 8192, 13 index
 * bits as a 12-bit stream plus a 1-bit plane, one E4M3 codebook per matrix,
 * one F16 gain per row. Rung B (K 4096, 12 bits, 32 KiB book) is the same
 * codec with GVQ_K12. Rung A+8 adds the GVR8 second stage.
 *
 * Sources of truth, in the order they were consulted:
 *
 *   /data/minimax-m3.1-vq-recipe/FORMAT.md      the field-by-field contract:
 *       "GVQ1 matrix" (header `<4sHBBIIIIIIQQQQ32s32s`, canonical offsets,
 *       the low-12 stream's 3-bytes-per-2-indices grouping, the high-1 plane's
 *       LSB-first bit order, the F16 gain rules) and "Residual stage
 *       (R13+8, container version 2)" (GVR1 header, one index byte per 8
 *       weights, the shared row gain, the two-stage value rule).
 *   crates/ds4-core/src/glm_vq.rs:117-225   the strict reader already in this
 *       repository, which is what "canonical" means operationally: offsets
 *       `[128, align(128+2R), align(main+R*C*3/16), align(...)]`, checksum
 *       placement (codebook at 64, body at 96, both SHA-256), the gain rules
 *       (non-zero, non-negative, finite), and zero padding in every gap.
 *       The GVQ1 magic/version/index-bits/dim/K/precision/cols%64 checks are
 *       the same lines.
 *   /data/YoungAi/src/common/ds4_fp8.h:118-138   ds4_e4m3fn_f32_to_byte and
 *       ds4_e4m3fn_to_f32 — E4M3FN with round-to-nearest-even and saturation
 *       at +-448; 0x7f/0xff are NaN and therefore never written.
 *   /data/YoungAi/src/common/ds4_sha256.h        ds4_sha256 (FIPS 180-4),
 *       the repository's only hash, so the checksums are the reader's own.
 *   /data/minimax-m3.1-vq-recipe/RECIPE.md and recipe.json   the encoder
 *       parameters this header's writer is fed with: equidistant init, 32
 *       Lloyd iterations, stratified pool, imatrix column weighting with
 *       col_power 0.75 / col_floor 0.02, least-squares gain refit, and for
 *       the residual stage the target `w/g - C1[i1]` against the *stored*
 *       gain and index, stage 1 unchanged ("Residual stage").
 *
 * Nothing in this file is inferred: every constant is either a field of the
 * format above or arithmetic over one (the byte accounts).
 */
#ifndef H3_GVQ_H
#define H3_GVQ_H

#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "ds4_fp8.h"       /* E4M3FN: round, value, f32 <-> byte */
#include "ds4_sha256.h"    /* the checksums the reader compares */
#include "onebit_quant.h"  /* go1b_fp16_to_fp32 / go1b_fp32_to_fp16 (the gain pair) */

/* ---- the format's constants ---------------------------------------------- */
#define GVQ_ALIGN         128u   /* FORMAT.md: "128-byte aligned" */
#define GVQ_HEADER        128u   /* `<4sHBBIIIIIIQQQQ32s32s` = 128 bytes */
#define GVQ_VECTOR          8u   /* the vector dimension of every rung here */
#define GVQ_K            8192u   /* rung A: 13 bits, 64 KiB E4M3 book */
#define GVQ_K12          4096u   /* rung B: 12 bits, 32 KiB E4M3 book */
#define GVQ_BITS13         13u
#define GVQ_BITS12         12u
#define GVQ_PREC_E4M3       1u   /* header field 20: "codebook precision 1 = E4M3FN" */
#define GVQ_PREC_F16        2u
#define GVR_BITS            8u   /* GVR8: 8 index bits, K = 256 */
#define GVR_K             256u

/* GGUF tensor type ids (FORMAT.md "Tensor types"; the same numbers the
 * readers in crates/ds4-core/src/{glm_vq,ling31_vq}.rs match on). */
#define GVQ_TYPE_VQ        0x47565138u   /* GVQ1 matrix */
#define GVQ_TYPE_RESIDUAL  0x47565238u   /* GVR8 residual index matrix */
#define GVQ_TYPE_CODEBOOK  0x47564342u   /* [K, 8] E4M3FN codebook */

/* ---- byte accounts -------------------------------------------------------
 * One writer, one reader, one account: both sides call these, so a size that
 * drifts is a compile-time-visible disagreement rather than a 5-hour surprise.
 *   main  = align128(128 + 2R)            header + F16 row gains
 *   main_end = main + R*C*3/16            the 12-bit stream, 3 bytes per 2 indices
 *   plane = align128(main_end)            the 13th-bit plane (K 8192 only)
 *   end   = plane + R*C/64                one bit per index
 *   size  = align128(end)                 the payload's total bytes
 * The reader in glm_vq.rs:140-160 computes exactly these four expressions. */
static size_t gvq_align(size_t v) { return (v + (GVQ_ALIGN - 1)) / GVQ_ALIGN * GVQ_ALIGN; }

static size_t gvq_main_off(int rows) {
    return gvq_align(GVQ_HEADER + 2u * (size_t)rows);
}
static size_t gvq_plane_off(int rows, int cols) {
    return gvq_align(gvq_main_off(rows) + (size_t)rows * (size_t)cols / 16u * 3u);
}
static size_t gvq_bytes(int rows, int cols, int k) {
    const size_t main = gvq_main_off(rows);
    const size_t main_end = main + (size_t)rows * (size_t)cols / 16u * 3u;
    const size_t end = (k == (int)GVQ_K) ? gvq_plane_off(rows, cols) + (size_t)rows * (size_t)cols / 64u
                                         : main_end;
    return gvq_align(end);
}
/* GVR8: header + one index byte per 8-weight vector, padded to 128. */
static size_t gvr_bytes(int rows, int cols) {
    return gvq_align(GVQ_HEADER + (size_t)rows * (size_t)(cols / (int)GVQ_VECTOR));
}
/* The book tensor's own bytes: K codewords of `dim` E4M3 bytes, no padding
 * (FORMAT.md "Tensor types": `[K, 8]` E4M3FN, GGUF dims `[8, K]`). */
static size_t gvq_book_bytes(int k) { return (size_t)k * GVQ_VECTOR; }

/* Every H3 VQ matrix has a column count divisible by 64, which is what makes
 * the 3-byte grouping land on whole rows (cols/8 is even) and the bit plane
 * exactly R*C/64 bytes. The reader rejects anything else (glm_vq.rs:146). */
static int gvq_shape_ok(int rows, int cols) {
    return rows > 0 && cols > 0 && cols % 64 == 0;
}

/* ---- field access ------------------------------------------------------- */
static void gvq_put16(uint8_t *p, uint16_t v) { memcpy(p, &v, 2); }
static void gvq_put32(uint8_t *p, uint32_t v) { memcpy(p, &v, 4); }
static void gvq_put64(uint8_t *p, uint64_t v) { memcpy(p, &v, 8); }
static uint16_t gvq_get16(const uint8_t *p) { uint16_t v; memcpy(&v, p, 2); return v; }
static uint32_t gvq_get32(const uint8_t *p) { uint32_t v; memcpy(&v, p, 4); return v; }
static uint64_t gvq_get64(const uint8_t *p) { uint64_t v; memcpy(&v, p, 8); return v; }

/* An E4M3 codeword byte is storable unless it is NaN: 0x7f/0xff are the only
 * NaN patterns in E4M3FN (ds4_fp8.h:26, and glm_vq.rs:262 applies the same
 * mask to a stored book). */
static int gvq_e4m3_finite(uint8_t b) { return (b & 0x7f) != 0x7f; }

/* ---- the value rule, both stages ----------------------------------------
 *   W[r, 8j + d] = f32(gain[r]) * f32(C1[i1[r,j]][d])                     (rung A)
 *   W[r, 8j + d] = f32(gain[r]) * (f32(C1[i1[r,j]][d]) + f32(C2[i2[r,j]][d]))
 * The sum of two E4M3 values is exact in F32 (FORMAT.md), which is why the
 * residual stage never needs its own gain. */
static float gvq_elem(const uint8_t *c1, const uint8_t *c2, int d) {
    float v = ds4_e4m3fn_to_f32(c1[d]);
    if (c2) { v += ds4_e4m3fn_to_f32(c2[d]); }
    return v;
}

/* ---- header ------------------------------------------------------------- */
typedef struct {
    int rows, cols, k, precision;
    uint32_t book_id;
    size_t size;
} gvq_hdr;

typedef struct {
    int rows, cols;
    uint32_t book_id;
    size_t size;
} gvr_hdr;

/* Strict GVQ1 read. Returns 0 and fills *h, or -1 and writes the first failing
 * rule into err. `book` may be NULL; when given, its SHA-256 must equal the
 * header's field. Mirrors glm_vq.rs:120-172 rule for rule. */
static int gvq_check(const uint8_t *p, size_t avail, const uint8_t *book, gvq_hdr *h,
                     char *err, size_t errn) {
    if (avail < GVQ_HEADER) { snprintf(err, errn, "truncated header"); return -1; }
    const int rows = (int)gvq_get32(p + 8), cols = (int)gvq_get32(p + 12);
    const int k = (int)gvq_get32(p + 16);
    if (memcmp(p, "GVQ1", 4)) { snprintf(err, errn, "bad magic"); return -1; }
    if (gvq_get16(p + 4) != 1) { snprintf(err, errn, "bad version"); return -1; }
    if (p[7] != GVQ_VECTOR) { snprintf(err, errn, "vector dim %u", p[7]); return -1; }
    if (k != (int)GVQ_K && k != (int)GVQ_K12) { snprintf(err, errn, "K %d", k); return -1; }
    if (p[6] != (k == (int)GVQ_K ? GVQ_BITS13 : GVQ_BITS12)) {
        snprintf(err, errn, "index bits %u for K %d", p[6], k);
        return -1;
    }
    if (gvq_get32(p + 20) != GVQ_PREC_E4M3) { snprintf(err, errn, "precision %u", gvq_get32(p + 20)); return -1; }
    if (gvq_get32(p + 28) != 0) { snprintf(err, errn, "flags %u", gvq_get32(p + 28)); return -1; }
    if (!gvq_shape_ok(rows, cols)) { snprintf(err, errn, "shape %dx%d", rows, cols); return -1; }
    const size_t size = gvq_bytes(rows, cols, k);
    if (avail < size) { snprintf(err, errn, "truncated (%zu < %zu)", avail, size); return -1; }
    const size_t main = gvq_main_off(rows);
    const size_t main_end = main + (size_t)rows * (size_t)cols / 16u * 3u;
    const size_t plane = (k == (int)GVQ_K) ? gvq_plane_off(rows, cols) : 0;
    if (gvq_get64(p + 32) != GVQ_HEADER || gvq_get64(p + 40) != main ||
        gvq_get64(p + 48) != plane || gvq_get64(p + 56) != size) {
        snprintf(err, errn, "noncanonical offsets");
        return -1;
    }
    uint8_t dig[32];
    ds4_sha256(p + GVQ_HEADER, size - GVQ_HEADER, dig);
    if (memcmp(dig, p + 96, 32)) { snprintf(err, errn, "payload checksum"); return -1; }
    if (book) {
        ds4_sha256(book, gvq_book_bytes(k), dig);
        if (memcmp(dig, p + 64, 32)) { snprintf(err, errn, "codebook checksum"); return -1; }
        for (int i = 0; i < k * (int)GVQ_VECTOR; i++)
            if (!gvq_e4m3_finite(book[i])) { snprintf(err, errn, "nonfinite codebook byte %d", i); return -1; }
    }
    /* Gains: F16, finite, non-negative. FORMAT.md (the release this rung is
     * named after) states "a zero row has gain +0 and restores exact zeros",
     * so +0 is legal here; glm_vq.rs:145 rejects it because GLM has no zero
     * rows. That one line is the only place this reader is looser than GLM's,
     * and it is looser on purpose. */
    for (int r = 0; r < rows; r++) {
        const uint16_t g = gvq_get16(p + GVQ_HEADER + 2u * (size_t)r);
        if ((g & 0x8000) || (g & 0x7c00) == 0x7c00) {
            snprintf(err, errn, "negative or nonfinite gain at row %d", r);
            return -1;
        }
    }
    /* Padding: gains -> main, main_end -> plane, end -> size. */
    for (size_t i = GVQ_HEADER + 2u * (size_t)rows; i < main; i++)
        if (p[i]) { snprintf(err, errn, "nonzero padding before main"); return -1; }
    if (plane) {
        for (size_t i = main_end; i < plane; i++)
            if (p[i]) { snprintf(err, errn, "nonzero padding before plane"); return -1; }
    }
    {
        const size_t end = (plane ? plane + (size_t)rows * (size_t)cols / 64u : main_end);
        for (size_t i = end; i < size; i++)
            if (p[i]) { snprintf(err, errn, "nonzero tail padding"); return -1; }
    }
    if (h) { h->rows = rows; h->cols = cols; h->k = k; h->precision = (int)GVQ_PREC_E4M3; h->book_id = gvq_get32(p + 24); h->size = size; }
    return 0;
}

/* Strict GVR8 read, same rule set (glm_vq.rs:186-224). */
static int gvr_check(const uint8_t *p, size_t avail, const uint8_t *rbook, gvr_hdr *h,
                     char *err, size_t errn) {
    if (avail < GVQ_HEADER) { snprintf(err, errn, "truncated residual header"); return -1; }
    const int rows = (int)gvq_get32(p + 8), cols = (int)gvq_get32(p + 12);
    if (memcmp(p, "GVR1", 4)) { snprintf(err, errn, "bad residual magic"); return -1; }
    if (gvq_get16(p + 4) != 1) { snprintf(err, errn, "bad residual version"); return -1; }
    if (p[6] != GVR_BITS || p[7] != GVQ_VECTOR) { snprintf(err, errn, "residual bits/dim"); return -1; }
    if (gvq_get32(p + 16) != GVR_K) { snprintf(err, errn, "residual K %u", gvq_get32(p + 16)); return -1; }
    if (gvq_get32(p + 20) != GVQ_PREC_E4M3) { snprintf(err, errn, "residual precision"); return -1; }
    if (gvq_get32(p + 28) != 0) { snprintf(err, errn, "residual flags"); return -1; }
    if (!gvq_shape_ok(rows, cols)) { snprintf(err, errn, "residual shape %dx%d", rows, cols); return -1; }
    const size_t size = gvr_bytes(rows, cols);
    if (avail < size) { snprintf(err, errn, "truncated residual (%zu < %zu)", avail, size); return -1; }
    if (gvq_get64(p + 32) != GVQ_HEADER || gvq_get64(p + 40) != 0 ||
        gvq_get64(p + 48) != 0 || gvq_get64(p + 56) != size) {
        snprintf(err, errn, "noncanonical residual offsets");
        return -1;
    }
    uint8_t dig[32];
    ds4_sha256(p + GVQ_HEADER, size - GVQ_HEADER, dig);
    if (memcmp(dig, p + 96, 32)) { snprintf(err, errn, "residual payload checksum"); return -1; }
    if (rbook) {
        ds4_sha256(rbook, gvq_book_bytes((int)GVR_K), dig);
        if (memcmp(dig, p + 64, 32)) { snprintf(err, errn, "residual codebook checksum"); return -1; }
        for (unsigned i = 0; i < GVR_K * GVQ_VECTOR; i++)
            if (!gvq_e4m3_finite(rbook[i])) { snprintf(err, errn, "nonfinite residual book"); return -1; }
    }
    const size_t end = GVQ_HEADER + (size_t)rows * (size_t)(cols / (int)GVQ_VECTOR);
    for (size_t i = end; i < size; i++)
        if (p[i]) { snprintf(err, errn, "nonzero residual padding"); return -1; }
    if (h) { h->rows = rows; h->cols = cols; h->book_id = gvq_get32(p + 24); h->size = size; }
    return 0;
}

/* ---- writers ------------------------------------------------------------ */
static void gvq_write_common(uint8_t *out, const char *magic, int rows, int cols, int k,
                             uint32_t book_id, const uint8_t *book, size_t book_bytes,
                             size_t main, size_t plane, size_t size) {
    memcpy(out, magic, 4);
    gvq_put16(out + 4, 1);
    out[6] = (uint8_t)(k == (int)GVR_K ? GVR_BITS : (k == (int)GVQ_K ? GVQ_BITS13 : GVQ_BITS12));
    out[7] = (uint8_t)GVQ_VECTOR;
    gvq_put32(out + 8, (uint32_t)rows);
    gvq_put32(out + 12, (uint32_t)cols);
    gvq_put32(out + 16, (uint32_t)k);
    gvq_put32(out + 20, GVQ_PREC_E4M3);
    gvq_put32(out + 24, book_id);
    gvq_put32(out + 28, 0);
    gvq_put64(out + 32, GVQ_HEADER);
    gvq_put64(out + 40, main);
    gvq_put64(out + 48, plane);
    gvq_put64(out + 56, size);
    ds4_sha256(book, book_bytes, out + 64);
}

/* GVQ1: gains, the 12-bit stream, then the 13th-bit plane. idx is
 * rows*cols/8 values, row-major vector order, each < K. Returns the size. */
static size_t gvq_write(uint8_t *out, int rows, int cols, int k, uint32_t book_id,
                        const uint8_t *book, const uint16_t *gain, const uint32_t *idx) {
    const size_t size = gvq_bytes(rows, cols, k);
    const size_t main = gvq_main_off(rows);
    const size_t plane = (k == (int)GVQ_K) ? gvq_plane_off(rows, cols) : 0;
    const size_t nidx = (size_t)rows * (size_t)cols / GVQ_VECTOR;
    memset(out, 0, size);
    gvq_write_common(out, "GVQ1", rows, cols, k, book_id, book, gvq_book_bytes(k),
                     main, plane, size);
    for (int r = 0; r < rows; r++) { gvq_put16(out + GVQ_HEADER + 2u * (size_t)r, gain[r]); }
    /* FORMAT.md: for adjacent indices a, b the stream holds
     * [a & 255, ((a >> 8) & 15) | ((b & 15) << 4), (b >> 4) & 255]. */
    uint8_t *lo = out + main;
    for (size_t j = 0; j + 1 < nidx; j += 2) {
        const uint32_t a = idx[j], b = idx[j + 1];
        lo[3 * (j / 2) + 0] = (uint8_t)(a & 0xffu);
        lo[3 * (j / 2) + 1] = (uint8_t)(((a >> 8) & 0x0fu) | ((b & 0x0fu) << 4));
        lo[3 * (j / 2) + 2] = (uint8_t)((b >> 4) & 0xffu);
    }
    if (plane) {   /* bit 12 of index j is bit j%8 of byte j/8, LSB first */
        uint8_t *hp = out + plane;
        for (size_t j = 0; j < nidx; j++)
            hp[j / 8] |= (uint8_t)(((idx[j] >> 12) & 1u) << (j % 8));
    }
    ds4_sha256(out + GVQ_HEADER, size - GVQ_HEADER, out + 96);
    return size;
}

/* GVR8: one index byte per vector, zero padding to 128.
 * Offsets: field 32 = 128 (the index region starts at the header's end, which
 * is what FORMAT.md calls the "index offset"), fields 40 and 48 = 0. glm_vq.rs
 * :217 checks exactly [128, 0, 0, size]; FORMAT.md names only the index offset
 * and the (zero) high offset, so the third field is zero by construction. */
static size_t gvr_write(uint8_t *out, int rows, int cols, uint32_t book_id,
                        const uint8_t *rbook, const uint8_t *idx8) {
    const size_t size = gvr_bytes(rows, cols);
    const size_t n = (size_t)rows * (size_t)(cols / (int)GVQ_VECTOR);
    memset(out, 0, size);
    gvq_write_common(out, "GVR1", rows, cols, (int)GVR_K, book_id, rbook,
                     gvq_book_bytes((int)GVR_K), 0, 0, size);
    memcpy(out + GVQ_HEADER, idx8, n);
    ds4_sha256(out + GVQ_HEADER, size - GVQ_HEADER, out + 96);
    return size;
}

/* ---- index recovery -----------------------------------------------------
 * The unpackers are the reader's own half of the writer above; the selftest
 * compares their output against the encoder's index arrays, so a packing bug
 * cannot hide behind a matching decode. */
static void gvq_unpack(const uint8_t *p, uint32_t *idx) {
    const int rows = (int)gvq_get32(p + 8), cols = (int)gvq_get32(p + 12);
    const size_t nidx = (size_t)rows * (size_t)cols / GVQ_VECTOR;
    const uint8_t *lo = p + gvq_main_off(rows);
    const uint8_t *hp = gvq_get64(p + 48) ? p + gvq_get64(p + 48) : NULL;
    for (size_t j = 0; j + 1 < nidx; j += 2) {
        const uint8_t *t = lo + 3 * (j / 2);
        idx[j] = (uint32_t)t[0] | ((uint32_t)(t[1] & 0x0fu) << 8);
        idx[j + 1] = (uint32_t)t[2] << 4 | ((uint32_t)t[1] >> 4);
        if (hp) {
            idx[j] |= (uint32_t)((hp[j / 8] >> (j % 8)) & 1u) << 12;
            idx[j + 1] |= (uint32_t)((hp[(j + 1) / 8] >> ((j + 1) % 8)) & 1u) << 12;
        }
    }
}
static void gvr_unpack(const uint8_t *p, uint8_t *idx8) {
    const int rows = (int)gvq_get32(p + 8), cols = (int)gvq_get32(p + 12);
    memcpy(idx8, p + GVQ_HEADER, (size_t)rows * (size_t)(cols / (int)GVQ_VECTOR));
}

/* Decode into W (rows*cols F32). Both stages when rbook is given. */
static void gvq_decode(const uint8_t *p, const uint8_t *book, const uint8_t *rp,
                       const uint8_t *rbook, float *W) {
    const int rows = (int)gvq_get32(p + 8), cols = (int)gvq_get32(p + 12);
    const size_t nvec_row = (size_t)cols / GVQ_VECTOR;
    uint32_t *idx = (uint32_t *)malloc((size_t)rows * nvec_row * sizeof(uint32_t));
    const uint8_t *rid = rp ? rp + GVQ_HEADER : NULL;
    gvq_unpack(p, idx);
    for (int r = 0; r < rows; r++) {
        const float g = go1b_fp16_to_fp32(gvq_get16(p + GVQ_HEADER + 2u * (size_t)r));
        const size_t base = (size_t)r * nvec_row;
        float *orow = W + (size_t)r * (size_t)cols;
        for (size_t j = 0; j < nvec_row; j++) {
            const uint8_t *c1 = book + (size_t)idx[base + j] * GVQ_VECTOR;
            const uint8_t *c2 = rbook ? rbook + (size_t)rid[base + j] * GVQ_VECTOR : NULL;
            for (int d = 0; d < (int)GVQ_VECTOR; d++) {
                orow[j * GVQ_VECTOR + (size_t)d] = g * gvq_elem(c1, c2, d);
            }
        }
    }
    free(idx);
}

#endif /* H3_GVQ_H */
