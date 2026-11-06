#ifndef DS4_GEMMA4_H
#define DS4_GEMMA4_H

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "ds4_gemma4_plan.h"
#include "ds4_gemma4_knobs.h"
#include "ds4_gemma4_vq.h"
/* ds4_gvq_model / ds4_gvq_record: the SSD stream plan the bind builds. The
 * header is guarded and includes nothing back here, so this stays a leaf. */
#include "ds4_glmvq.h"

/* Host routed-expert path (DS4_GEMMA4_CPU_MOE=1): the expert records stay
 * host-cold in the model mmap and their projections run on the CPU, so a
 * discrete card does not need the SSD slot pool. Read by the engine (map
 * install), the binder (cold records) and the MoE dispatch. */
static inline int gemma4_cpu_moe_on(void) {
    return ds4_gemma4_knob_bool("DS4_GEMMA4_CPU_MOE");
}

/* Per-layer host placement (DS4_GEMMA4_MOE_CPU_LAYERS=N): layers below N
 * keep their routed experts host-cold and their top-8 on the CPU, layers at
 * or above N keep theirs device-resident. This is llama.cpp's -ncmoe shape
 * (common/arg.cpp:2778-2787 pushes blk.0 .. blk.(N-1)); llama.cpp decides it
 * once at load by tensor-name regex, and so does this: the value is read when
 * the map and the binder are planned, never per token.
 *
 * N = 0 keeps today's all-GPU placement, and DS4_GEMMA4_CPU_MOE=1 stays the
 * shorthand for N = G4_LAYERS (every expert on the host), so the two knobs
 * cannot disagree: the boolean widens the layer count, and the count never
 * narrows the boolean.
 *
 * The accepted range is 0..999 and the clamp below is deliberate, not a
 * guard: llama.cpp's -ncmoe accepts any N and lets an N past the last block
 * match nothing (common/arg.cpp:2778-2787), which is what makes "-ncmoe 99"
 * mean "every layer" there. So 30, 99 and 999 all mean the whole model. */
static inline uint32_t gemma4_moe_cpu_layers(void) {
    if (gemma4_cpu_moe_on()) { return (uint32_t)G4_LAYERS; }
    const char *v = getenv("DS4_GEMMA4_MOE_CPU_LAYERS");
    long n = v ? strtol(v, NULL, 10) : 0;
    if (n <= 0) { return 0u; }
    if (n > G4_LAYERS) { n = G4_LAYERS; }
    return (uint32_t)n;
}

/* The routing-driven GPU expert cache (DS4_GEMMA4_MOE_CACHE_MIB=N), the
 * --moe-cache-mib shape: the experts stay host-cold in the mmap and a bounded
 * pool of VRAM slots is filled from them as the router asks for them, so a
 * hot expert runs on the GPU and a cold one misses. 0 means off, which is the
 * default and what every earlier lane measured. */
static inline uint64_t gemma4_moe_cache_bytes(void) {
    enum { MAX_MIB = 1 << 20 }; /* 1 TiB: past any card, never a wrap */
    const char *v = getenv("DS4_GEMMA4_MOE_CACHE_MIB");
    long mib = v ? strtol(v, NULL, 10) : 0;
    if (mib <= 0) { return 0u; }
    if (mib > MAX_MIB) { mib = MAX_MIB; }
    return (uint64_t)mib << 20;
}

/* The pinned staging ring the pool's slot fills are copied through
 * (DS4_GEMMA4_MOE_STAGING_MIB): 0 selects the 256 MiB default, anything else
 * is clamped to 64..2048 MiB.  The ring
 * bounds how many expert records can be in flight: 256 MiB holds 655 of the
 * 0.39 MiB records, well past one layer's 128 experts. */
static inline uint64_t gemma4_moe_staging_bytes(void) {
    const char *v = getenv("DS4_GEMMA4_MOE_STAGING_MIB");
    const long n = v ? strtol(v, NULL, 10) : 0;
    uint32_t mib = n > 0 && n < 4096 ? (uint32_t)n : 0u;
    if (mib == 0u) { mib = 256u; }
    if (mib < 64u) { mib = 64u; }
    if (mib > 2048u) { mib = 2048u; }
    return (uint64_t)mib << 20;
}

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
    return ds4_gemma4_knob_full_kv_fp8() != 0;
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

#ifndef DS4_NO_GPU
/* SSD streaming on a discrete card (ds4_gemma4_map.inc): after the GVQ plan
 * binds with the cold records presented header-sized, put the truthful records
 * back and clear their source entries. `truth` is the record array exactly as
 * gemma4_gpu_bind built it; `cold` lists the indices that were presented
 * header-sized. The first ds4_gpu_g4_stream_rebind fills each of them from its
 * pread slot. Returns 0 on a failed restore (the engine aborts the open). */
int ds4_gpu_g4_stream_plan_restore(ds4_gvq_model *plan,
        const ds4_gvq_record *truth, const uint32_t *cold, uint32_t count);
#endif

#ifdef __cplusplus
}
#endif

#endif
