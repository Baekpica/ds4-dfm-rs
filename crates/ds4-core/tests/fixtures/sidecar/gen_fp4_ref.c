/* gen_fp4_ref.c — P2 parity harness for the fp4x32 block decoder.
 *
 * `ds4_deq_fp4x32` below is the engine's function verbatim
 * (/data/YoungAi src/common/ds4_quantfmt.c:28-40), and `ds4_e8m0_to_f32` /
 * `ds4_fp4_nibble_to_f32` come from the vendored ds4_fp8.h next to this file.
 * The sidecar loader decodes gr and amp payloads with this format; the Rust
 * `deq_fp4x32` must reproduce every block bit for bit.
 *
 *   cc -O2 -I. gen_fp4_ref.c -o gen_fp4_ref && ./gen_fp4_ref
 *
 * Writes fp4_blocks.bin (raw 17-byte blocks) and fp4_ref.f32 (decoded f32,
 * little endian). The scales include e=0 (2^-127), e=127 (1.0) and e=255
 * (2^128, infinite) because the engine's format has no guard on either end.
 */
#include <stdio.h>
#include <stdint.h>
#include <string.h>

#include "ds4_fp8.h"

/* Engine: ds4_quantfmt.c:28-40, verbatim. */
void ds4_deq_fp4x32(const uint8_t *src, uint64_t nblk, float *out) {
    for (uint64_t b = 0; b < nblk; b++) {
        const uint8_t *blk = src + b * 17u;
        const float s = ds4_e8m0_to_f32(blk[16]);
        float *o = out + b * 32u;
        for (int j = 0; j < 16; j++) {
            o[2 * j]     = ds4_fp4_nibble_to_f32(blk[j] & 0x0F) * s;
            o[2 * j + 1] = ds4_fp4_nibble_to_f32(blk[j] >> 4) * s;
        }
    }
}

#define NBLK 6u

int main(void) {
    uint8_t blocks[NBLK * 17];
    const uint8_t scales[NBLK] = {127, 124, 130, 0, 255, 100};

    for (uint32_t b = 0; b < NBLK; b++) {
        uint8_t *blk = blocks + b * 17u;
        for (int j = 0; j < 16; j++) {
            /* Every nibble value appears: low = (b+j) mod 16, high = 15 - low. */
            const uint8_t lo = (uint8_t)((b + (uint32_t)j) & 15u);
            blk[j] = (uint8_t)(lo | ((15u - lo) << 4));
        }
        blk[16] = scales[b];
    }

    float out[NBLK * 32];
    ds4_deq_fp4x32(blocks, NBLK, out);

    FILE *fb = fopen("fp4_blocks.bin", "wb");
    FILE *fr = fopen("fp4_ref.f32", "wb");
    if (!fb || !fr) { perror("fixture"); return 1; }
    fwrite(blocks, 1, sizeof blocks, fb);
    fwrite(out, 4, NBLK * 32, fr);
    fclose(fb);
    fclose(fr);
    printf("wrote %u blocks, %u f32 values\n", NBLK, NBLK * 32u);
    return 0;
}
