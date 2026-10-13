/* ds41_vq_launch.cuh — the VQ decode dispatcher, ported from the C engine at
 * /data/YoungAi (commit 3946dbc), file `src/cuda/cuda_vq_decode_launch.inc.cu`.
 *
 * One job: pick the template instance by [on-disk version + codebook word count]
 * and do the shape precheck. All numerics live in the template it calls.
 * Width = ceil(log2(words)) decides words per round, and the kernel is
 * instantiated on it. A round is 32 indices => the row's index count (cols/8)
 * must be a multiple of 32 (IN/MID multiples of 256; V4.1 Flash: 5120 -> 20
 * rounds, 2304 -> 9); anything else is a hard error, not a slow path.
 *
 * The VERSION goes into the instance too (2026-09-21): a v2 payload carries its
 * own f16 codebook and a bitstream packed at NBIT; v3 hoists one E4M3 codebook
 * per layer and always packs a 12-bit main stream plus (13-bit layers) a bit
 * plane. The layout differences live in ds41_vq_row.cuh; this file only picks
 * the instance — picking wrong produces no error, just a whole set of fake
 * weights, so the version is read from the blob header by the caller, never
 * guessed from a filename.
 * Instance table: v2 12-bit / v2 11-bit / v3 12-bit / v3 13-bit (with plane).
 *
 * Must follow ds41_vq_decode.cuh and ds41_vq_group.cuh (v41_vq_fused_moe_n is
 * instantiated here; v41_vq_grp_launch must be visible by now).
 */
#pragma once
#include <stdint.h>
#include <stdio.h>

#include "ds41_vq_group.cuh"

static int v41_vq_fused_moe(float *out, const uint8_t *blob, uint32_t IN, uint32_t MID, uint32_t OUT,
                            const int32_t *sel, const float *w, uint32_t K, float clamp, const float *x, uint32_t n_tok, uint32_t nc,
                            const float *gr, uint32_t ver) {
    uint32_t nbit = 0; while ((1u << nbit) < nc) nbit++;
    if ((IN % 256u) || (MID % 256u)) {
        fprintf(stderr, "ds4: [ds41] VQ decode kernels need IN/MID multiples of 256 (32 indices per round), got %u/%u\n", IN, MID);
        return 0;
    }
    if (ver == 3u) {
        /* v3: 12-bit layers carry no plane, 13-bit layers one. Anything wider
         * would need another plane (the converter hard-stops too). */
        if (nbit == 12u) return v41_vq_fused_moe_n<12, 1, 0>(out, blob, IN, MID, OUT, sel, w, K, clamp, x, n_tok, nc, gr);
        if (nbit == 13u) return v41_vq_fused_moe_n<13, 1, 1>(out, blob, IN, MID, OUT, sel, w, K, clamp, x, n_tok, nc, gr);
        fprintf(stderr, "ds4: [ds41] DQVL v3 codebook with %u words (%u bits) has no decode instance (12/13 only)\n", nc, nbit);
        return 0;
    }
    /* v2 instances: nc4096 = 12-bit, nc2048 = 11-bit */
    if (nbit == 12u) return v41_vq_fused_moe_n<12, 0, 0>(out, blob, IN, MID, OUT, sel, w, K, clamp, x, n_tok, nc, gr);
    if (nbit == 11u) return v41_vq_fused_moe_n<11, 0, 0>(out, blob, IN, MID, OUT, sel, w, K, clamp, x, n_tok, nc, gr);
    fprintf(stderr, "ds4: [ds41] VQ codebook with %u words (%u bits) has no decode instance\n", nc, nbit);
    return 0;
}
