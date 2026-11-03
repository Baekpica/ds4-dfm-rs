#ifndef DS4_GEMMA4_PLAN_H
#define DS4_GEMMA4_PLAN_H

#include <stdbool.h>
#include <stdint.h>

/* Gemma-4 26B A4B VQ geometry, transcribed from the checkpoint's config.json
 * (revision 4d7ae498): 30 layers, 25 sliding and 5 full (layer_types 5, 11,
 * 17, 23, 29). The two attention geometries differ in four numbers, so they
 * are functions of the layer here rather than one set of constants:
 *
 *   sliding  16 query heads, 8 KV heads, head_dim 256, rope base 10000,
 *            full rotary, window 1024
 *   full     16 query heads, 2 KV heads, head_dim 512, rope base 1e6,
 *            partial_rotary_factor 0.25 (128 of 512 dims rotate)
 *
 * attention_k_eq_v: a full layer has no v_proj, its value projection is the
 * key projection. No layer in this family has an attention sink.
 */
enum {
    G4_LAYERS = 30,
    G4_EMBED = 2816,
    G4_VOCAB = 262144,
    G4_HEADS = 16,
    G4_HEAD_DIM = 256,
    G4_GLOBAL_HEAD_DIM = 512,
    G4_KV_HEADS = 8,
    G4_GLOBAL_KV_HEADS = 2,
    /* Rotary pairs (half of the rotated dims). Sliding layers use rope_type
     * "default", so every pair rotates: head_dim/2 = 128. Full layers use
     * "proportional" with partial_rotary_factor 0.25, so rope_angles =
     * int(0.25 * 512 // 2) = 64 pairs of the 256 (modeling_rope_utils.py:
     * 193-265; the reference buffers measure 128 and 64 nonzero). */
    G4_ROT = G4_HEAD_DIM / 2,
    G4_GLOBAL_ROT = 64,
    G4_WINDOW = 1024,
    G4_DENSE = 2112,
    G4_EXPERTS = 128,
    G4_USED = 8,
    G4_FF = 704,
    G4_CONTEXT = 262144,
    G4_PREFILL = 2048,
    G4_PREFILL_MAX = 8192,
    G4_VECTOR = 8,
    G4_WORDS = 8192,
    /* The drafter (`gemma4_assistant`): four layers at hidden 1024, no
     * experts, no k/v projection (it borrows the trunk's), and its third
     * layer is the full-attention one. */
    G4_MTP_LAYERS = 4,
    G4_MTP_EMBED = 1024,
    G4_MTP_FFN = 8192,
    G4_MTP_FULL = 3,
    G4_MTP_HEADS = 16,
    /* The trunk layers that publish the shared K/V. configuration_gemma4.py:
     * 236-251 designates the last non-consumer layer of each type as the
     * "store" producer, "even if no layer reuses them: assistant models
     * (e.g. Gemma4Assistant) read them from the shared_kv_states returned by
     * the target model". This trunk shares no layer (config.json
     * num_kv_shared_layers 0), so every layer is a non-consumer and the
     * producers are the last sliding layer (28) and the last full layer
     * (29). The pinned property returns early for a 0-sharing config, so the
     * file and its own comment disagree there; the port follows the stated
     * rule (docs/gemma4-reference-semantics.md, "Which trunk layers must
     * publish KV"). */
    G4_SHARED_SLIDING = 28,
    G4_SHARED_FULL = 29,
    /* Draft width the engine clamps `--mtp-draft` to: the assistant drafts
     * one token per forward at the fixed position, and four keep the verify
     * row block at the artifact's own layer count (G4_MTP_LAYERS). */
    G4_MTP_DRAFTS = G4_MTP_LAYERS
};

/* Logit softcap (3.9), the checkpoint's final_logit_softcapping = 30
 * (ds4_gemma4_ref.h:25 carries the oracle's own copy). */
#define G4_SOFTCAP 30.0f

static inline bool gemma4_is_full(unsigned layer) {
    return layer == 5u || layer == 11u || layer == 17u || layer == 23u ||
           layer == 29u;
}

static inline unsigned gemma4_head_dim(unsigned layer) {
    return gemma4_is_full(layer) ? G4_GLOBAL_HEAD_DIM : G4_HEAD_DIM;
}

static inline unsigned gemma4_kv_heads(unsigned layer) {
    return gemma4_is_full(layer) ? G4_GLOBAL_KV_HEADS : G4_KV_HEADS;
}

static inline unsigned gemma4_rot(unsigned layer) {
    return gemma4_is_full(layer) ? G4_GLOBAL_ROT : G4_ROT;
}

static inline double gemma4_rope_base(unsigned layer) {
    return gemma4_is_full(layer) ? 1000000.0 : 10000.0;
}

/* The drafter's per-layer geometry is its own config.json: layers 0-2 are
 * sliding at head_dim 256 over the trunk's 8 KV heads, layer 3 is full at
 * 512 over the trunk's 2 (text_config in the assistant checkpoint). The
 * rotary is the trunk's: sliding default theta 1e4 with every pair rotating,
 * full proportional 0.25 at theta 1e6, so the trunk's own tables apply. */
static inline bool gemma4_mtp_is_full(unsigned layer) {
    return layer == G4_MTP_FULL;
}

static inline unsigned gemma4_mtp_head_dim(unsigned layer) {
    return gemma4_mtp_is_full(layer) ? G4_GLOBAL_HEAD_DIM : G4_HEAD_DIM;
}

static inline unsigned gemma4_mtp_rot(unsigned layer) {
    return gemma4_rot(gemma4_mtp_is_full(layer) ? G4_SHARED_FULL : G4_SHARED_SLIDING);
}

/* The published K/V of one layer type: the trunk layer, its cache geometry
 * and the KV head count the drafter's 16 query heads group over. */
static inline unsigned gemma4_shared_layer(bool full) {
    return full ? G4_SHARED_FULL : G4_SHARED_SLIDING;
}

typedef struct {
    uint64_t full, sliding, scratch;
} ds4_gemma4_memory;

/* The graph's activation set for one `cap`-row chunk: five hidden rows
 * (cur, norm, branch, tmp, exp_acc), three dense-FF rows, the wide query and
 * attention buffers (allocated at the full-layer width for every layer), the
 * two KV row widths, the router scores and the top-8 selection, and the
 * tied-table logits. The graph allocates exactly this and the estimator
 * quotes the same number, so the two cannot drift.
 *
 * The token and position rows are the LARGER of the chunk and one draft
 * round's verify block (G4_MTP_DRAFTS + 1): the drafter's verify forward runs
 * the whole candidate block through the same graph. */
static inline uint32_t gemma4_token_rows(unsigned cap) {
    return cap < G4_MTP_DRAFTS + 1u ? G4_MTP_DRAFTS + 1u : cap;
}

static inline uint64_t gemma4_activation_bytes(unsigned cap) {
    const uint64_t rows = gemma4_token_rows(cap);
    return rows * ((uint64_t)5u * G4_EMBED + 3u * G4_DENSE +
        (uint64_t)G4_HEADS * G4_GLOBAL_HEAD_DIM + (uint64_t)G4_HEADS * G4_HEAD_DIM +
        2u * G4_KV_HEADS * G4_HEAD_DIM + (uint64_t)G4_GLOBAL_KV_HEADS * G4_GLOBAL_HEAD_DIM +
        G4_EXPERTS + 2u * G4_USED) * sizeof(float) +
        (uint64_t)gemma4_token_rows(cap) * (sizeof(int32_t) + sizeof(uint32_t)) +
        (uint64_t)G4_VOCAB * sizeof(float);
}

/* A sliding layer keeps window + prefill_cap rows (a ring, the same rule as
 * ds4_gemma4_kv_rows; window - 1 + cap is the minimum, window + cap leaves the
 * same margin the engine allocates); a full layer keeps the whole context. K
 * and V are counted separately here until the engine's k=v sharing is
 * measured; the plan stays the conservative bound. */
static inline unsigned gemma4_kv_capacity(unsigned layer, unsigned ctx,
                                          unsigned cap) {
    if (gemma4_is_full(layer)) { return ctx; }
    const unsigned needed = G4_WINDOW + gemma4_token_rows(cap);
    return ctx < needed ? ctx : needed;
}

static inline ds4_gemma4_memory gemma4_memory(unsigned ctx, unsigned cap) {
    ds4_gemma4_memory m = {0};
    if (!ctx || ctx > G4_CONTEXT || !cap || cap > ctx || cap > G4_PREFILL_MAX) {
        return m;
    }
    for (unsigned il = 0; il < G4_LAYERS; il++) {
        const uint64_t row = (uint64_t)gemma4_kv_heads(il) *
                             gemma4_head_dim(il) * sizeof(uint16_t) * 2u;
        const uint64_t bytes = row * gemma4_kv_capacity(il, ctx, cap);
        if (gemma4_is_full(il)) {
            m.full += bytes;
        } else {
            m.sliding += bytes;
        }
    }
    /* One row: hidden, the two attention norms, the seven layer norms, the
     * router scores over 128 experts, the eight selected experts and their
     * FF 704 rows, the dense FF 2112 rows, the query/key/value row and the
     * logits over the tied table. */
    const uint64_t row =
        (uint64_t)4u * G4_EMBED + G4_HEADS * G4_HEAD_DIM +
        G4_HEADS * G4_GLOBAL_HEAD_DIM + G4_KV_HEADS * G4_HEAD_DIM +
        G4_GLOBAL_KV_HEADS * G4_GLOBAL_HEAD_DIM + 2u * G4_DENSE +
        G4_EXPERTS + 2u * G4_USED + 3u * G4_USED * G4_FF +
        G4_USED * G4_EMBED + 8u;
    m.scratch = ((uint64_t)cap * row + G4_VOCAB) * sizeof(float);
    return m;
}

/* The drafter's device scratch, one number for the allocator and the quote:
 * the trunk hidden row and the token row it is concatenated with, the
 * drafter's residual stream and its norm, the wide query/attention rows at
 * the full layer's width, the o projection and the three MLP rows, plus the
 * drafter's own vocabulary head (one row) and the verify block's logits (one
 * row per accepted candidate, G4_MTP_DRAFTS + 1). */
static inline uint64_t gemma4_mtp_scratch_bytes(void) {
    const uint64_t floats =
        (uint64_t)G4_EMBED +            /* hidden: the trunk's final-norm row */
        (uint64_t)G4_EMBED +            /* emb: the trunk's token row */
        (uint64_t)2u * G4_EMBED +       /* cat: [emb | hidden], 2 * 2816 */
        (uint64_t)2u * G4_MTP_EMBED +   /* x, xn */
        (uint64_t)2u * G4_MTP_HEADS * G4_GLOBAL_HEAD_DIM +  /* q, attn */
        (uint64_t)G4_MTP_EMBED +        /* o projection */
        (uint64_t)3u * G4_MTP_FFN +     /* mlp gate, up, product */
        (uint64_t)G4_VOCAB +            /* the drafter's head, one row */
        (uint64_t)(G4_MTP_DRAFTS + 1u) * G4_VOCAB;  /* verify rows */
    return floats * sizeof(float) +
        (G4_MTP_DRAFTS + 1u) * sizeof(int32_t) + sizeof(uint32_t);
}

#endif
