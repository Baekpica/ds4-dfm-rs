#ifndef DS4_GEMMA4_H
#define DS4_GEMMA4_H

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "ds4_gemma4_plan.h"
#include "ds4_gemma4_vq.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Gemma-4-26B-A4B geometry and KV layout helpers. The VQ device runtime's
 * entry points live in ds4_gemma4_vq.h (Gemma's own optimization surface);
 * this header only carries the plan-derived row counts and the full-layer
 * KV format switch. ds4_glmvq.h is not included here. */

/* Sliding layers keep window + the widest block one forward writes (a ring):
 * that block may overwrite the slots it writes, and window + block keeps those
 * slots older than the oldest key the first row of the block still reads. The
 * block is the prefill chunk or one draft round's verify block, whichever is
 * wider (gemma4_token_rows); with a ring of only window + chunk, a verify
 * block wider than the chunk would clobber keys its own first rows read. A
 * full layer keeps the whole context. A context shorter than the ring is
 * clamped to ctx, the largest position the ring can ever hold, so the
 * allocator and the host quote (serving_host) price the same number. */
static inline uint32_t ds4_gemma4_kv_rows(unsigned layer, unsigned ctx, unsigned chunk) {
    if (gemma4_is_full(layer)) { return ctx; }
    const uint32_t rows = G4_WINDOW + gemma4_token_rows(chunk);
    return rows < ctx ? rows : ctx;
}

/* The five full layers may keep their KV as E4M3 instead of bf16. The
 * mechanism is the one Beam and MiMo already ship: E4M3 K and V with one f16
 * scale per KV head, full layers only (ds4_beam_bind.inc:56-67 and
 * ds4_mimo2_plan.h:38-58). Off by default; DS4_GEMMA4_FULL_KV=fp8 turns it
 * on. The 25 sliding layers keep bf16 rows, which the ring already bounds. */
static inline bool gemma4_full_kv_fp8(void) {
    static int fp8 = -1;
    if (fp8 < 0) {
        const char *value = getenv("DS4_GEMMA4_FULL_KV");
        fp8 = value && !strcmp(value, "fp8");
    }
    return fp8 != 0;
}

/* One K or V row of a bf16 layer. */
static inline uint64_t ds4_gemma4_kv_row(unsigned layer) {
    return (uint64_t)gemma4_kv_heads(layer) * gemma4_head_dim(layer) * sizeof(uint16_t);
}

/* The combined FP8 row of a full layer: E4M3 K, E4M3 V, then one f16 scale
 * per KV head for each. 2,056 bytes against 4,096 bf16 for the 2x512 full
 * geometry. */
static inline uint64_t ds4_gemma4_full_kv_row(void) {
    const uint64_t dim = (uint64_t)G4_GLOBAL_KV_HEADS * G4_GLOBAL_HEAD_DIM;
    return 2u * dim + 2u * (uint64_t)G4_GLOBAL_KV_HEADS * sizeof(uint16_t);
}

static inline uint64_t ds4_gemma4_kv_bytes(unsigned ctx, unsigned chunk) {
    const bool fp8 = gemma4_full_kv_fp8();
    uint64_t total = 0;
    for (unsigned il = 0; il < G4_LAYERS; il++) {
        const uint64_t rows = ds4_gemma4_kv_rows(il, ctx, chunk);
        if (fp8 && gemma4_is_full(il)) {
            total += rows * ds4_gemma4_full_kv_row();
        } else {
            total += 2u * rows * ds4_gemma4_kv_row(il);
        }
    }
    return total;
}

#ifdef __cplusplus
}
#endif

#endif
