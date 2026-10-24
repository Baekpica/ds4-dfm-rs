/* h3_pack.c — MiniMax-H3 FL2VA dense-VQ packer: classification, full encode,
 * DQVD container, and an independent read-back verification.
 *
 * Adapted from /data/embeddinggemma2/tools/eg2_pack.c (the EmbeddingGemma-2
 * dense bake's packer) at e5d9e9ce, branch feat/eg2vq-attn-perf. Four things
 * changed for H3: the class table, the kept-name list, classify(), and the
 * recipe string. Everything else is the bake's code unchanged — the DQVD
 * container, the slot layout, the plan/layout/merge/check machinery and the
 * encoder calls — which is what docs/minimax-h3-vq-recipe.md section 9 sets
 * out, and why this file exists rather than a new one.
 *
 * H3's classes, and what they are in the checkpoints's own module layout
 * (read from FL2VA/transformer/model.safetensors.index.json and the shard
 * headers, revision 42ed227e):
 *
 *   adaln     blocks.N.adaln_proj.linear.weight  [96768, 2688]  13.006B params
 *   fc1       blocks.N.mlp.fc1.weight            [28672, 5376]   7.707B
 *   qkv       blocks.N.attn.qkv_proj.weight      [21504, 5376]   5.780B
 *   fc2       blocks.N.mlp.fc2.weight            [5376, 14336]   3.854B
 *   out       blocks.N.attn.out_proj.weight      [5376, 7168]    1.927B
 *   head      condition_proj.weight, final_layer.adaln_proj.linear.weight
 *   keep      everything else: 210 norms, every bias, rope.inv_freq, and the
 *             six matrices that are F32 by design (the two patch projections,
 *             the two output heads, the two time-embedder projections)
 *
 * The token-refiner blocks carry no adaln and classify by matrix kind, so the
 * refiner's qkv/out/fc1/fc2 land in the same classes as the 50 main blocks and
 * the per-class metrics cover them.
 *
 * Every payload is produced by the reference toolchain, unmodified. Citations:
 *
 *   /data/YoungAi @ a4e6d31
 *     gguf-tools/quantize/vq_qc.h:86       vq_encode_full — kmeans on the
 *         weights alone (strided subsample <= 200k vectors, 8 Lloyd rounds,
 *         codebook round-tripped through f16 before assignment at :109), GPTQ
 *         column-group feedback (:160 onward, group 128, 1% diagonal damping),
 *         channel weighting 1/diag(H^-1) (:118-119), per-row closed-form gains
 *         clamped to [0.25, 4.0] (:224-235). With X = NULL the reference calls
 *         this the nocal path (ds4quant_run_p9.inc.c:77 passes `ncal?Xc:NULL`,
 *         i.e. X = NULL whenever no calibration corpus is configured) — feedback
 *         and the row-gain solve are skipped and gains stay 1.0 (:236).
 *     gguf-tools/quantize/vq_qc.h:241      vq_pack      — payload writer
 *     gguf-tools/quantize/vq_qc.h:268      vq_unpack_dequant — payload reader
 *     gguf-tools/quantize/vq_qc_bytes.h:15 vq_idx_bytes
 *     gguf-tools/quantize/vq_qc_bytes.h:20 vq_payload_bytes — the single byte
 *         account shared by writer and size allocator: 16 B header ('DQVQ',
 *         u16 dim, u16 nc, u32 rows, u32 cols) + nc*dim f16 codebook + rows f16
 *         gains + index stream; index width is ceil(log2 nc), stored one byte
 *         per index when it is exactly 8 bits and as a LE bitstream plus one
 *         guard byte otherwise.
 *     gguf-tools/quantize/go2b_qc.h:70     g2_inv (group inverse for GPTQ)
 *     gguf-tools/quantize/onebit_quant.h:118,141  go1b_fp32_to_fp16 / _fp16_to_fp32
 *     gguf-tools/quantize/quants.h:75,77,78  ds4q_row_size / ds4q_quantize_init /
 *         ds4q_quantize_chunk — the repository's only block-scalar encoder. Called
 *         with imatrix = NULL, the zero-corpus path the reference uses for the
 *         skeleton (gguf-tools/quantize/v41_q4k.cu:124, vq2q2k_probe.c:79), which
 *         v41_q4k.cu:13-16 explains is deliberate: the encoder stays the single CPU
 *         implementation because a second one would let the format fork silently.
 *     src/common/ds4_quantfmt.c:17         q8_0 geometry: 34 B per 32 values
 *     src/common/ds4_quantfmt.c:19         q4_K geometry: 256 elements / 144 B
 *     src/common/ds4_quantfmt.h:45         ds4_deq_q8_0 — the gold decoder used to
 *         verify the token payloads
 *     src/common/ds4_quantfmt.c:113        ds4_deq_q4_K — the q4_K gold decoder,
 *         kept because the container contract still has to accept kind 1
 *     src/common/ds4_st41.h:133,163,169    v41_st_open / v41_st_find /
 *         v41_st_data — the multi-shard safetensors index (locate only, no
 *         decode), so the reader is not reimplemented
 *     src/common/ds4_float.h:34            ds4_bf16_to_f32
 *
 *   /data/ds4-dfm-rs/ds41_vq_fmt.h:7       DQVQ value rule:
 *         value[row][col] = codebook[idx][col % dim] * gain[row]
 *
 * The container is the only new structure. It carries the payloads unchanged,
 * so a family port in ds4-dfm-rs keeps the payload contract it already knows
 * and only has to learn this index.
 *
 *   DQVD v1, little-endian, all fields unsigned:
 *
 *   header, 64 bytes
 *      0  u32 magic 'DQVD'        4  u16 version = 1      6  u16 kind = 0
 *      8  u32 n_slots            12  u32 recipe_len
 *     16  u64 recipe_off         24  u64 name_off         32  u64 name_bytes
 *     40  u64 slot_off           48  u64 payload_off      56  u64 payload_bytes
 *   recipe string    recipe_len ASCII bytes at recipe_off
 *   name blob        NUL-terminated tensor names, addressed by name_off/name_len
 *   slot table       48 bytes per slot, ascending in name order
 *      0  u64 name_hash   (FNV-1a 64 over the name)
 *      8  u64 name_off
 *     16  u32 name_len        20  u32 payload_off (relative to payload_off)
 *     24  u32 payload_len     28  u32 rows
 *     32  u32 cols            36  u16 dim        38  u16 nc
 *     40  u8  kind     0 = DQVQ payload, 1 = q4_K payload, 2 = verbatim source,
 *                      3 = q8_0 payload
 *     41  u8  src_dtype 1 = BF16, 2 = F32, 3 = F16
 *     42  u16 flags    bit0 = nocal (row gains are 1.0 by construction)
 *     44  u32 reserved = 0
 *   payload region   every payload starts on a 16-byte boundary; payload_len
 *                    excludes the padding
 *
 * Verbatim slots (kind 2) carry the source tensor's raw bytes in row-major
 * order, and they are the only slots that can be higher than rank 2: rows x cols
 * are the first two dimensions, and payload_len / dtype_size is the element
 * count. When those disagree the tensor has rank > 2 (the audio convolutions and
 * the vision position table) and its full shape comes from the family contract,
 * exactly as the reference toolchain's own shape-contract loaders do it. A rank
 * of 1 or 0 is stored with rows = the element count and cols = 1.
 *
 * Address fields are u32, so the container is limited to 4 GiB of payload; the
 * encoder refuses to write anything larger rather than truncate an offset.
 *
 * Loader contract: unknown magic, a version other than 1, an unknown kind, a dim
 * that does not divide cols, or a payload outside the region is a hard stop.
 * Never guess, never pad a decode.
 *
 * Determinism: given the same checkpoint and binary the writer is bit-exact
 * (kmeans is seeded from a strided subsample of the weights in fixed order, the
 * q4_K encoder is the reference's deterministic host implementation), which is
 * what `verify` re-establishes by re-encoding every payload and comparing bytes.
 *
 * Two builds, one source. The CPU build is the reference path and the A/B
 * baseline; the GPU build (-DEG2_CUDA) runs the encoder's two hot loops — the
 * kmeans assign (vq_qc.h:99) and the GPTQ column-group pass (:157) — on
 * vq_gpu.cu's kernels and pins every host buffer they touch (see the block above
 * the vq_qc.h include for the measurement behind that).
 *
 *   cc -O3 -std=c99 -Wall -D_POSIX_C_SOURCE=200809L \
 *      -I/data/YoungAi/gguf-tools/quantize -I/data/YoungAi/src/common \
 *      -o bin/eg2_pack_cpu tools/eg2_pack.c /data/YoungAi/gguf-tools/quantize/quants.c \
 *      /data/YoungAi/gguf-tools/quantize/onebit_quant.c \
 *      /data/YoungAi/src/common/ds4_quantfmt.c -lm -lpthread
 *
 *   nvcc -O3 -arch=native -c /data/YoungAi/gguf-tools/quantize/vq_gpu.cu -o out/vq_gpu.o
 *   cc -O3 -std=c99 -Wall -D_POSIX_C_SOURCE=200809L -DEG2_CUDA \
 *      -I/data/YoungAi/gguf-tools/quantize -I/data/YoungAi/src/common -I/usr/local/cuda/include \
 *      -o bin/eg2_pack tools/eg2_pack.c /data/YoungAi/gguf-tools/quantize/quants.c \
 *      /data/YoungAi/gguf-tools/quantize/onebit_quant.c \
 *      /data/YoungAi/src/common/ds4_quantfmt.c out/vq_gpu.o \
 *      -L/usr/local/cuda/lib64 -lcudart -lcublas -lstdc++ -lm -lpthread
 *
 * vq_gpu.o carries the reference's own flags (Makefile:197: nvcc -O3 -arch=native
 * -c, no -fmad=false). The reference does not claim byte equality between the two
 * VQ paths — it validates them by comparing quality, in its words "cblas 浮点序差
 * 异容忍, held 对拍验" (vq_gpu.cu:7) and "Hi double→f32 为唯一近似" (vq_gpu.cu:33).
 * This packer does the same: the encoder tag in the recipe string says which path
 * wrote the artifact, and every payload is measured, so the two builds are
 * compared on their reported cosines rather than on an equality that cannot hold
 * when two centroids sit within float noise of each other.
 *
 * Usage:
 *   eg2_pack plan   <hf-dir> [plan.tsv]
 *   eg2_pack encode <hf-dir> <out.bin> [log.txt] [--cold cls,...] [--limit N] [--only SUBSTR]
 *   eg2_pack verify <hf-dir> <in.bin>  [log.txt] [--cold cls,...] [--limit N] [--only SUBSTR]
 *
 * --limit and --only exist for smoke tests; either one stamps
 * "SMOKE-TEST-NOT-A-RELEASE" into the container's recipe string so a partial
 * artifact cannot be mistaken for a release.
 */
#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <time.h>

/* dq_matmul must precede vq_qc.h (it calls it). Semantic copied from the
 * reference: /data/YoungAi/gguf-tools/quantize/vq_shim.c:9
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

#include "onebit_quant.h"   /* go1b_fp16 helpers (static inline) */
#include "go2b_qc.h"        /* g2_inv */

/* ---- the GPU build: vq_qc.h's DS4QUANT_CUDA paths on this machine ----------
 *
 * vq_qc.h gates the GPU kernels on #ifdef DS4QUANT_CUDA and, when it takes them,
 * hands the kernels host pointers. On the reference's own machine (GB10, unified
 * memory) that is free: vq_gpu.cu:8 states the assumption, "malloc 指针直传, 无
 * 显式搬运". This card is a discrete RTX 4070 SUPER over PCIe and cannot
 * dereference pageable host memory — measured, not assumed: tools/eg2_gpuprobe.cu
 * reports pageableMemoryAccess = 0, and feeding vqg_assign plain malloc buffers
 * yields cudaErrorIllegalAddress with not one of the 200000 indices written
 * (vqg_last_err at vq_gpu.cu:26 prints at most four sticky errors and returns, so
 * such a run would otherwise continue on garbage).
 *
 * The mechanism for a host buffer the device reads is the reference's own: pin it.
 * vq_gpu.cu:459 vqg_host_alloc ("钉页宿主内存: DMA 满带宽(pageable 走 staging 慢
 * 3-5x)") is the reference's API for exactly this, and the reference already uses
 * that pattern for device-touched host buffers in ds4quant_fwd_p2.inc.c:221-230
 * (ATT_PIN: cudaMallocHost for the q/o activation round trips) and
 * ds4quant_moe_gpu.inc.c:46-53. Measured on this card with the same probe: pinned
 * buffers complete the same assign correctly at 146x the CPU scalar loop
 * (managed memory also works, 78x; pageable does not work at all).
 *
 * The substitution is scoped to this one include, because three of the buffers the
 * kernels touch are allocated by vq_qc.h itself: sub (vq_qc.h:91, the kmeans
 * subsample handed to vqg_assign), WkF (:145, the full-matrix working copy handed
 * to vqg_gptq_full) and sidx_f (:148, the index output the kernel writes). A
 * pinning failure is a hard stop rather than a fallback to malloc: unpinned memory
 * does not fail loudly here, it fails silently with unwritten indices. */
#ifdef EG2_CUDA
extern int  vqg_host_alloc(void **p, size_t bytes);
extern void vqg_host_free(void *p);
extern int  vqg_ready(void);
static size_t g_pinned_bytes = 0;   /* device-visible host bytes outstanding */
static void *eg2_pin(size_t n) {
    void *p = NULL;
    if (!vqg_host_alloc(&p, n ? n : 1)) {
        fprintf(stderr, "eg2_pack: cudaMallocHost(%zu) failed; refusing to hand the kernels pageable memory\n", n);
        exit(3);
    }
    g_pinned_bytes += n;
    return p;
}
#define malloc(n) eg2_pin(n)
#define free(p)   vqg_host_free(p)
#include "vq_qc.h"          /* vq_encode_full, vq_pack, vq_unpack_dequant */
#undef malloc
#undef free
/* My own buffers cross the same kernel boundary — the payload handed to
 * vqg_unpack_dequant and the matrix it decodes into (vq_qc.h:275-280) — so they go
 * through the same pinning. */
static void *dev_alloc(size_t n) { return eg2_pin(n); }
static void dev_free(void *p) { if (p) vqg_host_free(p); }
#else
#include "vq_qc.h"          /* vq_encode_full, vq_pack, vq_unpack_dequant */
static void *dev_alloc(size_t n) { return malloc(n); }
static void dev_free(void *p) { free(p); }
#endif
#include "quants.h"         /* ds4q_quantize_chunk (q4_K, zero-corpus) */
#include "ds4_st41.h"       /* v41_st_* tensor index */
#include "ds4_float.h"      /* ds4_bf16_to_f32 */
#include "ds4_quantfmt.h"   /* ds4_deq_q4_K (gold decoder) */

/* vq_qc.h declares these extern (line 369); they only feed the reference's
 * per-layer sidecar path, so zero is the documented default. */
int g_vq_dim = 0, g_vq_nc = 0;

#define DQVD_MAGIC     0x44565144u   /* 'DQVD' LE */
#define DQVD_VERSION   1
#define DQVD_HDR_BYTES 64
#define DQVD_SLOT_BYTES 48
#define DQVD_ALIGN     16

#define KIND_VQ       0
#define KIND_Q4K      1
#define KIND_VERBATIM 2
#define KIND_Q8_0     3

#define DT_BF16 1
#define DT_F32  2
#define DT_F16  3

/* The shipping tier, chosen by measurement. docs/ladder-embeddinggemma2.txt:
 * dim 4 with 1024/2048/4096 codebook entries measured cos 0.9737/0.9814/0.9869 on
 * the text tower and 0.9728/0.9804/0.9857 on the text PLE, against 0.9218/0.9111
 * for dim 8 at the same 12-bit index width (8x4096) and 0.8216 for the reference's
 * cold tier (8x256). For H3 the default is the quality end of the ladder,
 * vq4x4096, which is 12 index bits on 4-dim vectors = 1.5 bits/weight with a
 * 32 KiB F16 codebook (docs/minimax-h3-vq-recipe.md section 2, rung C). The
 * alternate tier is vq8x8192: 13 index bits on 8-dim vectors = 1.625
 * bits/weight, the rate the operator's MiniMax-M3.1 VQ release ships, but with
 * this writer it carries an F16 codebook of 128 KiB, which no GB10 kernel block
 * can hold; it is here as the encoder-side comparison, not as a runtime rung.
 * The rung that adds bits without growing the book past 64 KiB is vq4x8192
 * (13 bits at dim 4, 1.625 bits/weight, 64 KiB book) and is selected by editing
 * TIER_NC, because --cold keeps its per-class meaning.
 *
 * The codebook is F16 in every payload this writer produces (vq_qc_bytes.h:20:
 * 16 B header + nc*dim f16 + rows f16 + the index stream, index width
 * ceil(log2 nc)). E4M3 codewords — the codeword form the GB10 MMA path consumes
 * without conversion — are not available from vq_pack; that difference is
 * recorded in the recipe, not worked around here. */
#define TIER_DIM 4
#define TIER_NC  4096
#define COLD_DIM 8
#define COLD_NC  8192

/* The token table ships at q8_0, not the reference recipe's q4_K: q8_0 is 34 bytes
 * per 32 values (ds4_quantfmt.c:17), the only 8-bit weight type this toolchain can
 * both encode (quants_p3.inc.c:276) and decode with a gold reference
 * (ds4_quantfmt.h:45 ds4_deq_q8_0). q8_K is not a weight type here at all — its
 * trait entry is can_quantize=false (quants_p1.inc.c:55), there is no q8_K branch in
 * ds4q_quantize_chunk, no ds4_deq_q8_K, and both engines use Q8_K only for
 * activation blocks (core_types.h:16, ds4.c:1322,4581). */
#define TOKEN_TYPE     DS4Q_TYPE_Q8_0

#ifdef EG2_CUDA
#define ENCODER_TAG "cuda+pinned"
#else
#define ENCODER_TAG "cpu"
#endif

/* The reference's own gold decoder for the block-scalar formats
 * (ds4_quantfmt.h:44-47). The packer never decodes those bytes itself, so a token
 * payload can only pass its read-back check by going through the same decoder the
 * runtime will use. */
static void deq_block(int kind, const uint8_t *src, uint64_t nblk, float *out) {
    if (kind == KIND_Q8_0) ds4_deq_q8_0(src, nblk, out);
    else ds4_deq_q4_K(src, nblk, out);
}
static const char *kind_str(int kind) {
    return kind == KIND_VQ ? "DQVQ" : kind == KIND_Q8_0 ? "q8_0"
         : kind == KIND_Q4K ? "q4_K" : "verbatim";
}

enum { C_ADALN, C_FC1, C_QKV, C_FC2, C_OUT, C_HEAD, C_KEEP, C_NCLASS };
static const char *CLS_NAME[C_NCLASS] = {
    "adaln", "mlp-fc1", "attn-qkv", "mlp-fc2", "attn-out", "head", "keep"
};

typedef struct {
    const char *name;
    int cls;
    int kind;           /* KIND_* */
    int dim, nc;        /* VQ only */
    int rows, cols;
    int src_dtype;      /* DT_* */
    uint64_t src_bytes; /* bytes in the source checkpoint */
    uint64_t payload_len, payload_off;
    uint64_t name_off, hash;
    uint32_t name_len;
    /* measured in this run */
    double cos, rel, rt_maxabs, secs;
    int failed;
} slot_t;

static double now_s(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }

static uint64_t fnv1a64(const char *s) {
    uint64_t h = 1469598103934665603ULL;
    for (const unsigned char *p = (const unsigned char *)s; *p; p++) { h ^= *p; h *= 1099511628211ULL; }
    return h;
}

/* H3's matrices that stay at source precision even though they are rank-2 and
 * their column count divides the vector width. All six are F32 in the released
 * checkpoint, by design rather than by accident: the two patch projections are
 * the input side of the DiT, the two output heads are its output side, and the
 * two time-embedder projections feed the modulation path. The bake's equivalent
 * policy kept its embedding projections and tower frontends for the same reason. */
static int name_is_kept_matrix(const char *n) {
    static const char *KEEP[] = {
        "video_patch_proj.weight",
        "audio_patch_proj.weight",
        "final_layer.video_out.weight",
        "final_layer.audio_out.weight",
        "time_embedder.proj_in.weight",
        "time_embedder.proj_out.weight",
        NULL
    };
    for (int i = 0; KEEP[i]; i++) if (!strcmp(n, KEEP[i])) return 1;
    return 0;
}

static int starts(const char *s, const char *p) { return !strncmp(s, p, strlen(p)); }

/* Classification is structural first (rank, divisibility) and name-prefix
 * second. The prefix set is the checkpoint's own module layout, enumerated from
 * docs/tensor-inventory.tsv. */
static int classify(const v41_st_ent *E) {
    const char *n = E->name;
    if (E->nd != 2) return C_KEEP;                                 /* norms, rope.inv_freq, every bias */
    if (E->shape[1] % TIER_DIM) return C_KEEP;                     /* the vector width must divide cols */
    if (name_is_kept_matrix(n)) return C_KEEP;                     /* the six F32-by-design matrices */
    if (!strcmp(n, "condition_proj.weight")) return C_HEAD;
    if (!strcmp(n, "final_layer.adaln_proj.linear.weight")) return C_HEAD;
    /* Block matrices by suffix, so the two token-refiner blocks land in the same
     * classes as the fifty main blocks and one per-class metric covers both. */
    if (starts(n, "blocks.") || starts(n, "token_refiner.blocks.")) {
        if (strstr(n, ".adaln_proj.linear.weight")) return C_ADALN;
        if (strstr(n, ".mlp.fc1.weight")) return C_FC1;
        if (strstr(n, ".mlp.fc2.weight")) return C_FC2;
        if (strstr(n, ".attn.qkv_proj.weight")) return C_QKV;
        if (strstr(n, ".attn.out_proj.weight")) return C_OUT;
    }
    return C_KEEP;
}

/* Tier policy. Default: the shipping tier (dim 4, 4096 codebook entries) on every
 * VQ class, which is the highest quality measured in docs/ladder-embeddinggemma2.txt.
 * --cold CLASS[,CLASS...] switches the named classes to the reference's cold tier
 * (8x256, 1.0 bpw) as a size experiment — that tier measured 0.8183-0.8464 on the
 * FFN classes, so it is not a shipping configuration. */
static int cold_mask;   /* bit per class */
static int limit_slots; /* > 0: smoke-test mode, records itself in the recipe string */
static const char *only_sub;  /* non-NULL: keep only slots whose name contains this */
static int profile_mode;      /* --profile: per-stage timings of one payload, to stderr */
static int shard_i = -1, shard_n = 1;   /* --shard i/N: encode a slice of the plan */
static int merged_shards = 0;           /* merge: how many shards produced this artifact */
static double g_mp[4];
static void mark(int i) { if (profile_mode) g_mp[i] = now_s(); }
static int subset_mode(void) { return limit_slots > 0 || only_sub != NULL; }
static int dim_of(int cls) { return (cold_mask >> cls) & 1 ? COLD_DIM : TIER_DIM; }
static int nc_of(int cls)  { return (cold_mask >> cls) & 1 ? COLD_NC  : TIER_NC;  }

static uint64_t payload_len_of(const slot_t *s) {
    if (s->kind == KIND_VQ) return vq_payload_bytes(s->rows, s->cols, s->dim, s->nc);
    if (s->kind == KIND_Q4K || s->kind == KIND_Q8_0) {
        const ds4q_type t = (s->kind == KIND_Q8_0) ? DS4Q_TYPE_Q8_0 : DS4Q_TYPE_Q4_K;
        return (uint64_t)ds4q_row_size(t, s->cols) * (uint64_t)s->rows;
    }
    return s->src_bytes;
}

static void put_u32(uint8_t *p, uint32_t v) { memcpy(p, &v, 4); }
static void put_u64(uint8_t *p, uint64_t v) { memcpy(p, &v, 8); }
static void put_u16(uint8_t *p, uint16_t v) { memcpy(p, &v, 2); }

static const char *recipe_string(char *buf, size_t n) {
    char tier[64], cold[192] = "";
    snprintf(tier, sizeof tier, "vq%dx%d", TIER_DIM, TIER_NC);
    if (cold_mask) {
        for (int c = 0; c < C_NCLASS; c++)
            if ((cold_mask >> c) & 1) { strcat(cold, cold[0] ? "," : "cold="); strcat(cold, CLS_NAME[c]); }
        snprintf(cold + strlen(cold), sizeof cold - strlen(cold), ":vq%dx%d", COLD_DIM, COLD_NC);
    }
    /* No marker for a merged artifact: how the payloads were assembled is not part
     * of the container contract, and keeping the recipe identical means a sharded
     * run can be compared byte for byte against a single-process one. The merge log
     * records the shard count. */
    snprintf(buf, n, "minimax-h3-fl2va %s %s sharedcb=0 cbfp16 native=verbatim nocal encoder=%s (h3_pack) src=42ed227ee7df40d41602854ae760620d6eb651fe%s",
             tier, cold, ENCODER_TAG,
             subset_mode() ? " SMOKE-TEST-NOT-A-RELEASE" : "");
    return buf;
}

/* Which of the two builds wrote (or is re-encoding) these payloads. Recorded in
 * the container's recipe string and in the log, because the two paths are allowed
 * to differ on genuine index ties (see the build comment at the top). */
static void encoder_banner(FILE *log) {
#ifdef EG2_CUDA
    fprintf(log, "# encoder: %s, vq_gpu.cu kernels via DS4QUANT_CUDA, host buffers pinned with vqg_host_alloc\n", ENCODER_TAG);
    fprintf(log, "# encoder: vqg_ready=%d  pinned host bytes outstanding=%zu\n", vqg_ready(), g_pinned_bytes);
#else
    fprintf(log, "# encoder: %s (vq_qc.h scalar reference path, no CUDA)\n", ENCODER_TAG);
#endif
}

/* Source dtype codes recorded in the slot table for verbatim slots (and for
 * every slot, as provenance). Only the three dtypes this checkpoint uses. */
static int dtype_code(const char *d) {
    if (!strcmp(d, "BF16")) return DT_BF16;
    if (!strcmp(d, "F32")) return DT_F32;
    if (!strcmp(d, "F16")) return DT_F16;
    return 0;
}

/* Widen a source tensor to f32. Only the three dtypes the checkpoint uses. */
static int widen(const v41_st_ent *E, const uint8_t *raw, float **out, int *dt_code) {
    int64_t n = v41_st_numel(E);
    float *W = malloc((size_t)n * 4);
    if (!W) return -1;
    if (!strcmp(E->dtype, "BF16")) {
        const uint16_t *h = (const uint16_t *)raw;
        for (int64_t i = 0; i < n; i++) W[i] = ds4_bf16_to_f32(h[i]);
        *dt_code = DT_BF16;
    } else if (!strcmp(E->dtype, "F32")) {
        memcpy(W, raw, (size_t)n * 4);
        *dt_code = DT_F32;
    } else if (!strcmp(E->dtype, "F16")) {
        const uint16_t *h = (const uint16_t *)raw;
        for (int64_t i = 0; i < n; i++) W[i] = go1b_fp16_to_fp32(h[i]);
        *dt_code = DT_F16;
    } else { free(W); return -1; }
    *out = W;
    return 0;
}

/* Reconstruction metrics of a payload against its source tensor: decode through the
 * same reader the runtime will use (VQ: vq_unpack_dequant; block formats: the
 * reference's gold decoder) and compare element by element. emit_payload calls this
 * right after encoding and the merge path calls it after copying a payload out of a
 * shard, so a merged artifact is measured by exactly the same code as a
 * single-process one. W_in may be a pre-widened copy of the tensor, to avoid
 * widening the same matrix twice; NULL widens it here. */
static int payload_metrics(v41_st *S, const v41_st_ent *E, slot_t *s, const uint8_t *payload, const float *W_in) {
    const size_t n = (size_t)s->rows * s->cols;
    float *W = (float *)W_in;
    if (!W) {
        const uint8_t *raw = v41_st_data(S, E);
        int dt = 0;
        if (!raw || widen(E, raw, &W, &dt)) return -1;
    }
    float *Wq = dev_alloc(n * 4);
    if (s->kind == KIND_VQ) {
        int r2 = 0, c2 = 0;
        if (vq_unpack_dequant(payload, s->payload_len, Wq, &r2, &c2) != 0 || r2 != s->rows || c2 != s->cols) {
            dev_free(Wq); if (W != W_in) free(W); return -1;
        }
    } else {
        const ds4q_type t = (s->kind == KIND_Q8_0) ? DS4Q_TYPE_Q8_0 : DS4Q_TYPE_Q4_K;
        deq_block(s->kind, payload, (uint64_t)(n / (size_t)ds4q_block_size(t)), Wq);
    }
    double dot = 0, na = 0, nb = 0, se = 0;
    for (size_t i = 0; i < n; i++) {
        const double a = W[i], b = Wq[i];
        dot += a * b; na += a * a; nb += b * b;
        const double d = a - b; se += d * d;
    }
    s->cos = dot / (sqrt(na) * sqrt(nb) + 1e-30);
    s->rel = sqrt(se / (na + 1e-30));
    dev_free(Wq);
    if (W != W_in) free(W);
    return 0;
}

/* Produce the payload for one slot into out[]. Measures the reconstruction
 * against the source (cos, relL2) and, for VQ, the decode-versus-encode
 * bit-exactness (rt_maxabs). Returns 0 on success. */
static int emit_payload(v41_st *S, const v41_st_ent *E, slot_t *s, uint8_t *out) {
    const uint8_t *raw = v41_st_data(S, E);
    if (!raw) return -1;
    const double t0 = now_s();

    if (s->kind == KIND_VERBATIM) {
        const int dt = dtype_code(E->dtype);
        if (!dt) { fprintf(stderr, "%s: dtype %s not supported for verbatim passthrough\n", E->name, E->dtype); return -1; }
        if (s->src_dtype && s->src_dtype != dt) { fprintf(stderr, "%s: dtype changed since the plan\n", E->name); return -1; }
        s->src_dtype = dt;
        memcpy(out, raw, (size_t)s->src_bytes);
        s->cos = 1.0; s->rel = 0.0; s->rt_maxabs = 0.0; s->secs = now_s() - t0;
        return 0;
    }

    float *W = NULL;
    int dt = 0;
    if (widen(E, raw, &W, &dt)) { fprintf(stderr, "%s: dtype %s not supported\n", E->name, E->dtype); return -1; }
    s->src_dtype = dt;
    mark(0);   /* widen done */
    const size_t n = (size_t)s->rows * s->cols;

    if (s->kind == KIND_VQ) {
        const int dim = s->dim, nc = s->nc;
        /* Cb is the codebook the kernels read (vq_qc.h:157 passes it straight to
         * vqg_gptq_full), so it is device-visible; gr and idx never leave the CPU
         * on either path (the index stream reaches this process from the kernel
         * through vq_qc.h's own sidx_f). */
        float *Cb = dev_alloc((size_t)nc * dim * 4), *gr = malloc((size_t)s->rows * 4);
        int *idx = malloc((n / dim) * sizeof(int));
        float *Wq = malloc(n * 4);
        vq_encode_full(W, s->rows, s->cols, dim, nc, NULL, 0, Cb, idx, gr, Wq);
        mark(1);   /* kmeans + GPTQ done */
        const size_t want = vq_payload_bytes(s->rows, s->cols, dim, nc);
        const size_t wrote = vq_pack(Cb, idx, gr, s->rows, s->cols, dim, nc, out);
        if (wrote != want) { fprintf(stderr, "%s: pack wrote %zu, account says %zu\n", E->name, wrote, want); goto fail; }
        /* The read-back goes through vq_unpack_dequant, which in the GPU build
         * decodes on the device for matrices this size (vq_qc.h:275-280), so both
         * the payload and the destination are device-visible. */
        float *Wback = dev_alloc(n * 4);
        int r2 = 0, c2 = 0;
        if (vq_unpack_dequant(out, wrote, Wback, &r2, &c2) != 0 || r2 != s->rows || c2 != s->cols) {
            fprintf(stderr, "%s: payload did not read back as %dx%d\n", E->name, s->rows, s->cols);
            dev_free(Wback); goto fail;
        }
        s->rt_maxabs = 0;
        for (size_t i = 0; i < n; i++) { double d = fabs((double)Wback[i] - Wq[i]); if (d > s->rt_maxabs) s->rt_maxabs = d; }
        dev_free(Wback); dev_free(Cb); free(gr); free(idx); free(Wq);
        mark(2);   /* payload packed, read back and compared */
    } else { /* KIND_Q4K or KIND_Q8_0: a block-scalar format the reference encodes and decodes */
        const ds4q_type t = (s->kind == KIND_Q8_0) ? DS4Q_TYPE_Q8_0 : DS4Q_TYPE_Q4_K;
        const int64_t blk = ds4q_block_size(t);
        size_t rb = ds4q_row_size(t, s->cols);
        size_t wrote = ds4q_quantize_chunk(t, W, out, 0, s->rows, s->cols, NULL);
        if (wrote != rb * (size_t)s->rows) { fprintf(stderr, "%s: %s wrote %zu, expected %zu\n", E->name, ds4q_type_name(t), wrote, rb * (size_t)s->rows); goto fail; }
        float *Wback = malloc(n * 4);
        deq_block(s->kind, out, (uint64_t)(n / blk), Wback);   /* the reference's gold decoder */
        s->rt_maxabs = 0;
        for (size_t i = 0; i < n; i++) { double d = fabs((double)Wback[i] - W[i]); if (d > s->rt_maxabs) s->rt_maxabs = d; }
        free(Wback);
    }

    {   /* reconstruction metrics against the source */
        if (payload_metrics(S, E, s, out, W)) goto fail;
    }

    s->secs = now_s() - t0;
    if (profile_mode && s->kind == KIND_VQ)
        fprintf(stderr, "PROFILE %s widen=%.3f encode=%.3f readback=%.3f metrics=%.3f total=%.3f\n",
                E->name, g_mp[0] - t0, g_mp[1] - g_mp[0], g_mp[2] - g_mp[1], now_s() - g_mp[2], s->secs);
    free(W);
    return 0;
fail:
    free(W);
    return -1;
}

/* ---- plan ----------------------------------------------------------------- */

static int build_plan(v41_st *S, slot_t **slots_out, int *n_out) {
    slot_t *slots = calloc((size_t)S->ne, sizeof(slot_t));
    if (!slots) return -1;
    int n = 0;
    for (int i = 0; i < S->ne; i++) {
        const v41_st_ent *E = &S->e[i];
        if (only_sub && !strstr(E->name, only_sub)) continue;
        slot_t *s = &slots[n];
        s->name = E->name;
        s->cls = classify(E);
        s->src_bytes = E->nbytes;
        s->src_dtype = dtype_code(E->dtype);
        if (!s->src_dtype) { fprintf(stderr, "%s: dtype %s is not supported by this packer\n", E->name, E->dtype); free(slots); return -1; }
        s->name_len = (uint32_t)strlen(E->name);
        s->hash = fnv1a64(E->name);
        if (E->nd >= 2) { s->rows = (int)E->shape[0]; s->cols = (int)E->shape[1]; }
        else if (E->nd == 1) { s->rows = (int)E->shape[0]; s->cols = 1; }
        else { s->rows = 1; s->cols = 1; }
        /* H3 has no token table inside the DiT (the embedding lives in the text
         * encoder), so this packer only ever writes VQ payloads and verbatim
         * copies: the bake's block-scalar token branch has no H3 referent and is
         * gone rather than left unreachable. */
        if (s->cls == C_KEEP) {
            s->kind = KIND_VERBATIM;
        } else {
            s->kind = KIND_VQ;
            s->dim = dim_of(s->cls);
            s->nc = nc_of(s->cls);
        }
        s->payload_len = payload_len_of(s);
        n++;
        if (limit_slots > 0 && n >= limit_slots) break;
    }
    if (shard_n > 1) {
        /* --shard i/N: this process encodes plan slots i, i+N, i+2N, ... — applied
         * after --only/--limit, so a sharded subset differs from a sharded full run */
        int k = 0;
        for (int i = 0; i < n; i++) if (i % shard_n == shard_i) slots[k++] = slots[i];
        n = k;
    }
    *slots_out = slots;
    *n_out = n;
    return 0;
}

static int plan_report(v41_st *S, slot_t *slots, int n, FILE *out) {
    uint64_t src_total = 0, pay_total = 0;
    fprintf(out, "# DQVD v%d plan — written by h3_pack (MiniMax-H3 FL2VA). Measured and computed, never estimated.\n", DQVD_VERSION);
    fprintf(out, "# source: %s\n", S->sh[0].path);
    fprintf(out, "# class totals: tensors, VQ matrices, source MB, payload MB, tier, bpw\n");
    fprintf(out, "%-16s %8s %6s %12s %12s  %-10s %6s\n", "class", "tensors", "matrices", "src MB", "payload MB", "tier", "bpw");
    double agg_cos[C_NCLASS] = {0}, agg_rel[C_NCLASS] = {0};
    int agg_n[C_NCLASS] = {0};
    for (int c = 0; c < C_NCLASS; c++) {
        int nt = 0, nm = 0;
        uint64_t sb = 0, pb = 0;
        for (int i = 0; i < n; i++) {
            slot_t *s = &slots[i];
            if (s->cls != c) continue;
            nt++; sb += s->src_bytes; pb += s->payload_len;
            if (s->kind == KIND_VQ) nm++;
            if (s->cos > 0) { agg_cos[c] += s->cos; agg_rel[c] += s->rel; agg_n[c]++; }
            src_total += s->src_bytes; pay_total += s->payload_len;
        }
        char tier[32];
        if (c == C_KEEP) snprintf(tier, sizeof tier, "verbatim");
        else snprintf(tier, sizeof tier, "vq%dx%d", dim_of(c), nc_of(c));
        double bpw = sb ? 8.0 * (double)pb / ((double)sb / 2.0) : 0.0;
        fprintf(out, "%-16s %8d %6d %12.2f %12.2f  %-10s %8.3f\n",
                CLS_NAME[c], nt, nm, (double)sb / 1e6, (double)pb / 1e6, tier, bpw);
    }
    fprintf(out, "%-16s %8d %6s %12.2f %12.2f\n", "TOTAL", n, "-", (double)src_total / 1e6, (double)pay_total / 1e6);
    fprintf(out, "ratio: %.2fx smaller than the source checkpoint\n", (double)src_total / (double)pay_total);
    return 0;
}

static void plan_tsv(slot_t *slots, int n, FILE *f) {
    fprintf(f, "name\tclass\tkind\trows\tcols\tdim\tnc\tsrc_dtype\tsrc_bytes\tpayload_bytes\n");
    static const char *kind_name[] = { "vq", "q4_K", "verbatim", "q8_0" };
    static const char *dt_name[] = { "?", "BF16", "F32", "F16" };
    for (int i = 0; i < n; i++) {
        slot_t *s = &slots[i];
        fprintf(f, "%s\t%s\t%s\t%d\t%d\t%d\t%d\t%s\t%llu\t%llu\n", s->name, CLS_NAME[s->cls],
                kind_name[s->kind], s->rows, s->cols, s->dim, s->nc,
                dt_name[s->src_dtype < 4 ? s->src_dtype : 0],
                (unsigned long long)s->src_bytes, (unsigned long long)s->payload_len);
    }
}

/* ---- container ------------------------------------------------------------ */

static uint64_t align_up(uint64_t v, uint64_t a) { return (v + a - 1) / a * a; }

/* A slot as read back from a container, for the verify comparison. */
typedef struct {
    char name[192];
    uint32_t rows, cols, payload_len, payload_off;
    uint16_t dim, nc, flags;
    uint8_t kind, src_dtype;
} dvd_slot;

static int layout(slot_t *slots, int n, uint64_t *recipe_off, uint64_t *name_off,
                  uint64_t *name_bytes, uint64_t *slot_off, uint64_t *payload_off) {
    char recipe[256];
    recipe_string(recipe, sizeof recipe);
    uint64_t off = DQVD_HDR_BYTES;
    *recipe_off = off; off += align_up(strlen(recipe) + 1, DQVD_ALIGN);
    *name_off = off;
    for (int i = 0; i < n; i++) slots[i].name_off = 0;   /* filled below */
    uint64_t nb = 0;
    for (int i = 0; i < n; i++) { slots[i].name_off = nb; nb += slots[i].name_len + 1; }
    *name_bytes = nb; off += align_up(nb, DQVD_ALIGN);
    *slot_off = off; off += align_up((uint64_t)n * DQVD_SLOT_BYTES, DQVD_ALIGN);
    *payload_off = align_up(off, DQVD_ALIGN);
    uint64_t po = 0;
    for (int i = 0; i < n; i++) {
        slots[i].payload_off = po;
        if (po > 0xffffffffULL || slots[i].payload_len > 0xffffffffULL) {
            fprintf(stderr, "payload offsets are u32 in DQVD v1; %s would overflow them\n", slots[i].name);
            return -1;
        }
        po += align_up(slots[i].payload_len, DQVD_ALIGN);
    }
    if (po > 0xffffffffULL) { fprintf(stderr, "payload region %llu B exceeds the u32 address space of DQVD v1\n", (unsigned long long)po); return -1; }
    return 0;
}

static void write_header(FILE *f, int n, uint32_t recipe_len, uint64_t recipe_off, uint64_t name_off,
                         uint64_t name_bytes, uint64_t slot_off, uint64_t payload_off, uint64_t payload_bytes) {
    uint8_t h[DQVD_HDR_BYTES];
    memset(h, 0, sizeof h);
    put_u32(h + 0, DQVD_MAGIC);
    put_u16(h + 4, DQVD_VERSION);
    put_u16(h + 6, 0);
    put_u32(h + 8, (uint32_t)n);
    put_u32(h + 12, recipe_len);
    put_u64(h + 16, recipe_off);
    put_u64(h + 24, name_off);
    put_u64(h + 32, name_bytes);
    put_u64(h + 40, slot_off);
    put_u64(h + 48, payload_off);
    put_u64(h + 56, payload_bytes);
    fwrite(h, 1, sizeof h, f);
}

static void write_slot_table(FILE *f, slot_t *slots, int n) {
    uint8_t *tab = calloc((size_t)n, DQVD_SLOT_BYTES);
    for (int i = 0; i < n; i++) {
        slot_t *s = &slots[i];
        uint8_t *p = tab + (size_t)i * DQVD_SLOT_BYTES;
        put_u64(p + 0, s->hash);
        put_u64(p + 8, s->name_off);
        put_u32(p + 16, s->name_len);
        put_u32(p + 20, (uint32_t)s->payload_off);
        put_u32(p + 24, (uint32_t)s->payload_len);
        put_u32(p + 28, (uint32_t)s->rows);
        put_u32(p + 32, (uint32_t)s->cols);
        put_u16(p + 36, (uint16_t)s->dim);
        put_u16(p + 38, (uint16_t)s->nc);
        p[40] = (uint8_t)s->kind;
        p[41] = (uint8_t)s->src_dtype;
        put_u16(p + 42, s->kind == KIND_VQ ? 1u : 0u);   /* bit0 = nocal */
    }
    fwrite(tab, DQVD_SLOT_BYTES, (size_t)n, f);
    free(tab);
}

/* ---- encode / verify ------------------------------------------------------ */

static int run(v41_st *S, slot_t *slots, int n, const char *out_path, const char *log_path,
               int verify_mode, const dvd_slot *tab, uint32_t n_tab) {
    FILE *log = log_path ? fopen(log_path, "w") : stdout;
    if (!log) { fprintf(stderr, "cannot write %s\n", log_path); return -1; }
    if (verify_mode && n_tab != (uint32_t)n) {
        fprintf(stderr, "container has %u slots, the plan for this checkpoint has %d (hard stop)\n", n_tab, n);
        if (log != stdout) fclose(log);
        return -1;
    }

    char recipe[256];
    recipe_string(recipe, sizeof recipe);
    uint64_t recipe_off, name_off, name_bytes, slot_off, payload_off;
    if (layout(slots, n, &recipe_off, &name_off, &name_bytes, &slot_off, &payload_off)) {
        if (log != stdout) fclose(log);
        return -1;
    }
    uint64_t payload_bytes = 0;
    for (int i = 0; i < n; i++) payload_bytes = slots[i].payload_off + align_up(slots[i].payload_len, DQVD_ALIGN);

    FILE *out = NULL;
    FILE *vin = NULL;
    if (verify_mode) {
        vin = fopen(out_path, "rb");
        if (!vin) { fprintf(stderr, "cannot read %s\n", out_path); return -1; }
    } else {
        out = fopen(out_path, "wb");
        if (!out) { fprintf(stderr, "cannot write %s\n", out_path); return -1; }
        write_header(out, n, (uint32_t)strlen(recipe), recipe_off, name_off, name_bytes, slot_off, payload_off, payload_bytes);
        fwrite(recipe, 1, strlen(recipe) + 1, out);
        uint64_t pos = DQVD_HDR_BYTES + strlen(recipe) + 1;
        while (pos < name_off) { fputc(0, out); pos++; }
        for (int i = 0; i < n; i++) fwrite(slots[i].name, 1, slots[i].name_len + 1, out);
        pos = name_off + name_bytes;
        while (pos < slot_off) { fputc(0, out); pos++; }
        write_slot_table(out, slots, n);
        pos = slot_off + (uint64_t)n * DQVD_SLOT_BYTES;
        while (pos < payload_off) { fputc(0, out); pos++; }
        if (ftell(out) != (long)payload_off) { fprintf(stderr, "layout error: at %ld, expected %llu\n", ftell(out), (unsigned long long)payload_off); return -1; }
        fprintf(log, "# encoded %s\n# recipe: %s\n", out_path, recipe);
    }

    encoder_banner(log);
    ds4q_quantize_init(DS4Q_TYPE_Q4_K);
    fprintf(log, "#%-58s %-14s %-8s %9s %9s %9s %6s\n", "tensor", "class", "kind", "cos", "relL2%", "runtime", "s");
    uint64_t total_pay = 0, total_src = 0;
    double cos_sum[C_NCLASS] = {0}, rel_sum[C_NCLASS] = {0};
    int cos_n[C_NCLASS] = {0}, bad = 0, mismatch = 0;
    const double t0 = now_s();
    for (int i = 0; i < n; i++) {
        slot_t *s = &slots[i];
        const v41_st_ent *E = v41_st_find(S, s->name);
        if (!E) { fprintf(stderr, "tensor vanished: %s\n", s->name); return -1; }
        /* The payload buffer is written by the CPU and read back by the device in
         * the GPU build's reader path (vq_qc.h:275), so it is pinned there too. */
        uint8_t *buf = dev_alloc((size_t)s->payload_len);
        if (!buf) { fprintf(stderr, "out of memory for %s (%llu B)\n", s->name, (unsigned long long)s->payload_len); return -1; }
        /* emit_payload needs payload_len for its metric re-decode; it is already set */
        const uint64_t want = s->payload_len;
        const int rc = emit_payload(S, E, s, buf);
        if (rc) {
            fprintf(stderr, "FAILED %s (%s)\n", s->name, CLS_NAME[s->cls]);
            dev_free(buf); s->failed = 1; bad++;
            if (!verify_mode) { fclose(log); if (out) fclose(out); remove(out_path); fprintf(stderr, "partial artifact removed: %s\n", out_path); return -1; }
            continue;
        }
        if (payload_len_of(s) != want) { fprintf(stderr, "%s: payload length changed %llu -> %llu\n", s->name, (unsigned long long)want, (unsigned long long)s->payload_len); return -1; }
        if (verify_mode) {
            const dvd_slot *d = &tab[i];
            if (strcmp(d->name, s->name)) { fprintf(stderr, "slot %d: container holds %s, plan holds %s\n", i, d->name, s->name); return -1; }
            if (d->kind != s->kind || d->rows != (uint32_t)s->rows || d->cols != (uint32_t)s->cols ||
                d->dim != (uint16_t)s->dim || d->nc != (uint16_t)s->nc || d->payload_len != (uint32_t)s->payload_len) {
                fprintf(stderr, "slot %d (%s): container metadata disagrees with the plan (kind %u/%d rows %u/%d cols %u/%d dim %u/%d nc %u/%d len %u/%llu)\n",
                        i, s->name, d->kind, s->kind, d->rows, s->rows, d->cols, s->cols,
                        d->dim, s->dim, d->nc, s->nc, d->payload_len, (unsigned long long)s->payload_len);
                return -1;
            }
            uint8_t *got = malloc((size_t)want);
            if (fseeko(vin, (off_t)(payload_off + s->payload_off), SEEK_SET)) { perror("fseeko"); return -1; }
            if (fread(got, 1, (size_t)want, vin) != want) { fprintf(stderr, "%s: short read from container\n", s->name); return -1; }
            if (memcmp(got, buf, (size_t)want)) {
                mismatch++;
                fprintf(stderr, "MISMATCH %s: container payload differs from a fresh encode\n", s->name);
            }
            free(got);
        } else {
            fwrite(buf, 1, (size_t)want, out);
            uint64_t pad = align_up(want, DQVD_ALIGN) - want;
            for (uint64_t k = 0; k < pad; k++) fputc(0, out);
        }
        dev_free(buf);
        total_pay += want; total_src += s->src_bytes;
        if (s->cos > 0) { cos_sum[s->cls] += s->cos; rel_sum[s->cls] += s->rel; cos_n[s->cls]++; }
        fprintf(log, "%-60s %-14s %-8s %9.4f %9.4f %8.2f\n", s->name, CLS_NAME[s->cls],
                kind_str(s->kind), s->cos, 100.0 * s->rel, s->secs);
    }
    const double secs = now_s() - t0;

    if (log != stdout) {
        fprintf(log, "\n# class aggregates over %s payloads\n", verify_mode ? "re-encoded" : "encoded");
        fprintf(log, "%-16s %8s   %10s  %10s\n", "class", "tensors", "cos mean", "relL2 mean %");
        for (int c = 0; c < C_NCLASS; c++)
            if (cos_n[c]) fprintf(log, "%-16s %8d   %10.4f  %10.4f\n", CLS_NAME[c], cos_n[c],
                                   cos_sum[c] / cos_n[c], 100.0 * rel_sum[c] / cos_n[c]);
        fprintf(log, "\n# tensors %d  failed %d  payload mismatches %d\n", n, bad, mismatch);
        fprintf(log, "# payload bytes %llu  source bytes %llu  ratio %.2fx  wall %.1fs (%.1f MB/s source)\n",
                (unsigned long long)total_pay, (unsigned long long)total_src,
                (double)total_src / (double)total_pay, secs, (double)total_src / 1e6 / secs);
    }
    if (out) { fflush(out); fclose(out); }
    if (vin) fclose(vin);
    if (log != stdout) fclose(log);
    if (bad || mismatch) return -1;
    printf("%s: %d tensors, payload %llu B, source %llu B, %.2fx, %.1fs\n",
           verify_mode ? "verify OK (all payloads bit-identical to a fresh encode)" : "encode OK",
           n, (unsigned long long)total_pay, (unsigned long long)total_src,
           (double)total_src / (double)total_pay, secs);
    return 0;
}

/* ---- merge: assemble one container from shard containers --------------------
 *
 * A matrix's payload depends only on that matrix — the codebook is per matrix and
 * the kmeans is seeded from the matrix's own weights (vq_qc.h:91-95) — so encoding
 * the plan in N slices and pasting the payloads back in plan order produces the same
 * artifact as one single-process run. That equivalence is what the shard-versus-
 * single byte comparison checks; this function only has to move bytes without
 * inventing any. Every plan slot must appear in exactly one shard with the metadata
 * the plan expects; anything missing, duplicated or mismatched is a hard stop, and
 * each pasted payload is then re-measured through the same decoder path as an
 * encoded one. */
static int check_container(const char *path, dvd_slot **tab_out, uint32_t *n_out,
                           uint64_t *payload_off, uint64_t *payload_bytes);
static int merge_run(v41_st *S, slot_t *slots, int n, const char *out_path, const char *log_path,
                     char **shard_paths, int n_shards) {
    FILE *log = log_path ? fopen(log_path, "w") : stdout;
    if (!log) { fprintf(stderr, "cannot write %s\n", log_path); return -1; }
    dvd_slot **tabs = calloc((size_t)n_shards, sizeof(dvd_slot *));
    uint32_t *ns = calloc((size_t)n_shards, sizeof(uint32_t));
    uint64_t *poff = calloc((size_t)n_shards, sizeof(uint64_t));
    FILE **fs = calloc((size_t)n_shards, sizeof(FILE *));
    for (int k = 0; k < n_shards; k++) {
        uint64_t pb = 0;
        if (check_container(shard_paths[k], &tabs[k], &ns[k], &poff[k], &pb)) {
            fprintf(stderr, "%s: not a usable shard (hard stop)\n", shard_paths[k]);
            if (log != stdout) fclose(log);
            return -1;
        }
        fs[k] = fopen(shard_paths[k], "rb");
        if (!fs[k]) { perror(shard_paths[k]); if (log != stdout) fclose(log); return -1; }
    }

    int *src_shard = malloc((size_t)n * sizeof(int));
    uint64_t *abs_off = malloc((size_t)n * sizeof(uint64_t));
    int missing = 0;
    for (int i = 0; i < n; i++) {
        src_shard[i] = -1; abs_off[i] = 0;
        for (int k = 0; k < n_shards && src_shard[i] < 0; k++)
            for (uint32_t j = 0; j < ns[k]; j++) {
                if (strcmp(tabs[k][j].name, slots[i].name)) continue;
                const dvd_slot *d = &tabs[k][j];
                if (d->kind != slots[i].kind || d->rows != (uint32_t)slots[i].rows || d->cols != (uint32_t)slots[i].cols ||
                    d->dim != (uint16_t)slots[i].dim || d->nc != (uint16_t)slots[i].nc || d->payload_len != (uint32_t)slots[i].payload_len) {
                    fprintf(stderr, "%s: shard metadata for %s disagrees with the plan (kind %u/%d rows %u/%d cols %u/%d dim %u/%d nc %u/%d len %u/%llu)\n",
                            shard_paths[k], slots[i].name, d->kind, slots[i].kind, d->rows, slots[i].rows, d->cols, slots[i].cols,
                            d->dim, slots[i].dim, d->nc, slots[i].nc, d->payload_len, (unsigned long long)slots[i].payload_len);
                    if (log != stdout) fclose(log);
                    return -1;
                }
                src_shard[i] = k; abs_off[i] = poff[k] + d->payload_off;
                break;
            }
        if (src_shard[i] < 0) { fprintf(stderr, "slot %d (%s): no shard carries it\n", i, slots[i].name); missing++; }
    }
    if (missing) {
        fprintf(stderr, "%d plan slot(s) are not present in any shard: refusing to write a partial container\n", missing);
        if (log != stdout) fclose(log);
        return -1;
    }

    char recipe[256];
    recipe_string(recipe, sizeof recipe);
    uint64_t recipe_off, name_off, name_bytes, slot_off, payload_off;
    if (layout(slots, n, &recipe_off, &name_off, &name_bytes, &slot_off, &payload_off)) { if (log != stdout) fclose(log); return -1; }
    uint64_t payload_bytes = 0;
    for (int i = 0; i < n; i++) payload_bytes = slots[i].payload_off + align_up(slots[i].payload_len, DQVD_ALIGN);

    FILE *out = fopen(out_path, "wb");
    if (!out) { perror(out_path); if (log != stdout) fclose(log); return -1; }
    write_header(out, n, (uint32_t)strlen(recipe), recipe_off, name_off, name_bytes, slot_off, payload_off, payload_bytes);
    fwrite(recipe, 1, strlen(recipe) + 1, out);
    uint64_t pos = DQVD_HDR_BYTES + strlen(recipe) + 1;
    while (pos < name_off) { fputc(0, out); pos++; }
    for (int i = 0; i < n; i++) fwrite(slots[i].name, 1, slots[i].name_len + 1, out);
    pos = name_off + name_bytes;
    while (pos < slot_off) { fputc(0, out); pos++; }
    write_slot_table(out, slots, n);
    pos = slot_off + (uint64_t)n * DQVD_SLOT_BYTES;
    while (pos < payload_off) { fputc(0, out); pos++; }
    if (ftell(out) != (long)payload_off) { fprintf(stderr, "layout error: at %ld, expected %llu\n", ftell(out), (unsigned long long)payload_off); return -1; }

    fprintf(log, "# merged %s from %d shards\n# recipe: %s\n", out_path, n_shards, recipe);
    encoder_banner(log);
    fprintf(log, "#%-58s %-14s %-8s %9s %9s %9s %6s\n", "tensor", "class", "kind", "cos", "relL2%", "runtime", "s");
    size_t cap = 0;
    for (int i = 0; i < n; i++) if (slots[i].payload_len > cap) cap = (size_t)slots[i].payload_len;
    uint8_t *buf = dev_alloc(cap ? cap : 1);
    uint64_t total_pay = 0, total_src = 0;
    double cos_sum[C_NCLASS] = {0}, rel_sum[C_NCLASS] = {0};
    int cos_n[C_NCLASS] = {0}, bad = 0;
    const double t0 = now_s();
    for (int i = 0; i < n; i++) {
        slot_t *s = &slots[i];
        const v41_st_ent *E = v41_st_find(S, s->name);
        if (!E) { fprintf(stderr, "tensor vanished: %s\n", s->name); return -1; }
        if (fseeko(fs[src_shard[i]], (off_t)abs_off[i], SEEK_SET)) { perror("fseeko"); return -1; }
        if (fread(buf, 1, (size_t)s->payload_len, fs[src_shard[i]]) != s->payload_len) { fprintf(stderr, "%s: short read from shard\n", s->name); return -1; }
        fwrite(buf, 1, (size_t)s->payload_len, out);
        for (uint64_t k = align_up(s->payload_len, DQVD_ALIGN) - s->payload_len; k; k--) fputc(0, out);
        const double t1 = now_s();
        if (payload_metrics(S, E, s, buf, NULL)) { fprintf(stderr, "FAILED to measure %s\n", s->name); bad++; }
        s->secs = now_s() - t1;
        total_pay += s->payload_len; total_src += s->src_bytes;
        if (s->cos > 0) { cos_sum[s->cls] += s->cos; rel_sum[s->cls] += s->rel; cos_n[s->cls]++; }
        fprintf(log, "%-60s %-14s %-8s %9.4f %9.4f %8.2f\n", s->name, CLS_NAME[s->cls], kind_str(s->kind), s->cos, 100.0 * s->rel, s->secs);
    }
    const double secs = now_s() - t0;
    dev_free(buf);
    for (int k = 0; k < n_shards; k++) fclose(fs[k]);
    fflush(out); fclose(out);
    if (log != stdout) {
        fprintf(log, "\n# class aggregates over merged payloads\n");
        fprintf(log, "%-16s %8s   %10s  %10s\n", "class", "tensors", "cos mean", "relL2 mean %");
        for (int c = 0; c < C_NCLASS; c++)
            if (cos_n[c]) fprintf(log, "%-16s %8d   %10.4f  %10.4f\n", CLS_NAME[c], cos_n[c], cos_sum[c] / cos_n[c], 100.0 * rel_sum[c] / cos_n[c]);
        fprintf(log, "\n# tensors %d  unmeasurable %d\n", n, bad);
        fprintf(log, "# payload bytes %llu  source bytes %llu  ratio %.2fx  merge wall %.1fs\n",
                (unsigned long long)total_pay, (unsigned long long)total_src, (double)total_src / (double)total_pay, secs);
        fclose(log);
    }
    printf("merge OK: %d tensors from %d shards, payload %llu B, source %llu B, %.2fx, %.1fs\n",
           n, n_shards, (unsigned long long)total_pay, (unsigned long long)total_src, (double)total_src / (double)total_pay, secs);
    return bad ? -1 : 0;
}

/* ---- header checks for verify -------------------------------------------- */

/* Reads back the container and checks every structural rule the loader depends
 * on. The parsed slot table is returned so verify can compare it field by field
 * against a fresh plan. Returns 0 when the container is well formed. */
static int check_container(const char *path, dvd_slot **tab_out, uint32_t *n_out,
                           uint64_t *payload_off, uint64_t *payload_bytes) {
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "cannot read %s\n", path); return -1; }
    uint8_t h[DQVD_HDR_BYTES];
    if (fread(h, 1, sizeof h, f) != sizeof h) { fprintf(stderr, "%s: short header\n", path); fclose(f); return -1; }
    uint32_t magic, n_slots, recipe_len; memcpy(&magic, h, 4); memcpy(&n_slots, h + 8, 4); memcpy(&recipe_len, h + 12, 4);
    uint16_t ver, kind; memcpy(&ver, h + 4, 2); memcpy(&kind, h + 6, 2);
    if (magic != DQVD_MAGIC) { fprintf(stderr, "%s: bad magic %08x (hard stop)\n", path, magic); fclose(f); return -1; }
    if (ver != DQVD_VERSION) { fprintf(stderr, "%s: version %u not supported (hard stop)\n", path, ver); fclose(f); return -1; }
    if (kind != 0) { fprintf(stderr, "%s: kind %u not supported (hard stop)\n", path, kind); fclose(f); return -1; }
    uint64_t recipe_off, name_off, name_bytes, slot_off;
    memcpy(&recipe_off, h + 16, 8); memcpy(&name_off, h + 24, 8); memcpy(&name_bytes, h + 32, 8);
    memcpy(&slot_off, h + 40, 8); memcpy(payload_off, h + 48, 8); memcpy(payload_bytes, h + 56, 8);
    char recipe[512] = {0};
    if (recipe_len && recipe_len < sizeof recipe - 1) {
        fseeko(f, (off_t)recipe_off, SEEK_SET);
        if (fread(recipe, 1, recipe_len, f) != recipe_len) fprintf(stderr, "%s: recipe short read\n", path);
        recipe[recipe_len] = 0;
    }
    printf("container %s: kind=%u slots=%u\nrecipe: %s\n", path, kind, n_slots, recipe);
    printf("layout: recipe=%llu names=%llu(+%llu) slots=%llu payloads=%llu+%llu\n",
           (unsigned long long)recipe_off, (unsigned long long)name_off, (unsigned long long)name_bytes,
           (unsigned long long)slot_off, (unsigned long long)*payload_off, (unsigned long long)*payload_bytes);
    fseeko(f, 0, SEEK_END);
    const off_t fsz = ftello(f);
    if ((uint64_t)fsz != *payload_off + *payload_bytes) {
        fprintf(stderr, "%s: file is %lld B, header accounts for %llu B (hard stop)\n", path,
                (long long)fsz, (unsigned long long)(*payload_off + *payload_bytes));
        fclose(f); return -1;
    }
    char *blob = malloc((size_t)name_bytes);
    fseeko(f, (off_t)name_off, SEEK_SET);
    if (!blob || fread(blob, 1, (size_t)name_bytes, f) != name_bytes) { fprintf(stderr, "%s: name blob short read\n", path); free(blob); fclose(f); return -1; }
    uint8_t *raw = malloc((size_t)n_slots * DQVD_SLOT_BYTES);
    fseeko(f, (off_t)slot_off, SEEK_SET);
    if (!raw || fread(raw, DQVD_SLOT_BYTES, n_slots, f) != n_slots) { fprintf(stderr, "%s: slot table short read\n", path); free(blob); free(raw); fclose(f); return -1; }
    dvd_slot *tab = calloc(n_slots, sizeof(dvd_slot));
    uint64_t prev_end = 0;
    int bad = 0;
    for (uint32_t i = 0; i < n_slots; i++) {
        const uint8_t *p = raw + (size_t)i * DQVD_SLOT_BYTES;
        dvd_slot *d = &tab[i];
        uint64_t hash, noff; uint32_t nlen;
        memcpy(&hash, p, 8); memcpy(&noff, p + 8, 8); memcpy(&nlen, p + 16, 4);
        memcpy(&d->payload_off, p + 20, 4); memcpy(&d->payload_len, p + 24, 4);
        memcpy(&d->rows, p + 28, 4); memcpy(&d->cols, p + 32, 4);
        memcpy(&d->dim, p + 36, 2); memcpy(&d->nc, p + 38, 2);
        memcpy(&d->flags, p + 42, 2);
        d->kind = p[40]; d->src_dtype = p[41];
        if ((uint64_t)noff + nlen + 1 > name_bytes) { fprintf(stderr, "slot %u: name range out of blob\n", i); bad++; continue; }
        snprintf(d->name, sizeof d->name, "%s", blob + noff);
        if (strlen(blob + noff) != nlen || fnv1a64(d->name) != hash) { fprintf(stderr, "slot %u (%s): name/hash mismatch\n", i, d->name); bad++; }
        if (d->payload_off < prev_end) { fprintf(stderr, "slot %u (%s): payload overlaps the previous one\n", i, d->name); bad++; }
        if ((uint64_t)d->payload_off + d->payload_len > *payload_bytes) { fprintf(stderr, "slot %u (%s): payload out of region\n", i, d->name); bad++; }
        if (d->payload_off % DQVD_ALIGN) { fprintf(stderr, "slot %u (%s): payload offset %u is not %d-aligned\n", i, d->name, d->payload_off, DQVD_ALIGN); bad++; }
        prev_end = align_up((uint64_t)d->payload_off + d->payload_len, DQVD_ALIGN);
        if (d->kind > KIND_Q8_0) { fprintf(stderr, "slot %u (%s): unknown kind %u (hard stop)\n", i, d->name, d->kind); bad++; }
        /* metadata must reproduce the payload length through the reference's own
         * byte accounts, so a mistyped dim/nc cannot slip through as garbage */
        if (d->kind == KIND_VQ) {
            if (!d->dim || !d->nc || d->cols % d->dim || d->nc > 65535) {
                fprintf(stderr, "slot %u (%s): VQ metadata dim=%u nc=%u cols=%u is not usable\n", i, d->name, d->dim, d->nc, d->cols); bad++;
            } else {
                const size_t want = vq_payload_bytes((int)d->rows, (int)d->cols, (int)d->dim, (int)d->nc);
                if (want != d->payload_len) { fprintf(stderr, "slot %u (%s): payload_len %u, the VQ byte account says %zu\n", i, d->name, d->payload_len, want); bad++; }
            }
        } else if (d->kind == KIND_Q4K || d->kind == KIND_Q8_0) {
            /* geometry straight from the reference's trait table (quants.h:75-78):
             * q4_K is 144 B per 256 values, q8_0 is 34 B per 32 (ds4_quantfmt.c:17) */
            const ds4q_type t = (d->kind == KIND_Q8_0) ? DS4Q_TYPE_Q8_0 : DS4Q_TYPE_Q4_K;
            const int64_t blk = ds4q_block_size(t);
            const size_t rb = ds4q_row_size(t, (int64_t)d->cols);
            if (d->cols % blk) { fprintf(stderr, "slot %u (%s): %s needs cols divisible by %lld, got %u\n", i, d->name, ds4q_type_name(t), (long long)blk, d->cols); bad++; }
            else if (rb * (size_t)d->rows != d->payload_len) {
                fprintf(stderr, "slot %u (%s): %s payload_len %u, the reference's byte account says %llu (%zu B per %lld values)\n", i, d->name, ds4q_type_name(t),
                        d->payload_len, (unsigned long long)(rb * (size_t)d->rows),
                        rb / (size_t)(d->cols / blk), (long long)blk); bad++;
            }
        } else {
            const uint64_t dts = (d->src_dtype == DT_BF16 || d->src_dtype == DT_F16) ? 2 : (d->src_dtype == DT_F32 ? 4 : 0);
            if (!dts) { fprintf(stderr, "slot %u (%s): unknown source dtype code %u\n", i, d->name, d->src_dtype); bad++; }
            else {
                if (d->payload_len % dts) { fprintf(stderr, "slot %u (%s): %u bytes is not a whole number of %llu-byte elements\n", i, d->name, d->payload_len, (unsigned long long)dts); bad++; }
                else if (d->payload_len / dts < (uint64_t)d->rows * d->cols) {
                    fprintf(stderr, "slot %u (%s): %llu elements is fewer than the recorded %ux%u shape\n", i, d->name,
                            (unsigned long long)(d->payload_len / dts), d->rows, d->cols); bad++;
                }
            }
        }
        if (i && strcmp(tab[i - 1].name, d->name) >= 0) { fprintf(stderr, "slot %u: names are not in ascending order\n", i); bad++; }
    }
    if (prev_end > *payload_bytes) { fprintf(stderr, "payload region is padded past its end\n"); bad++; }
    free(blob); free(raw); fclose(f);
    if (bad) { fprintf(stderr, "%s: %d structural problems\n", path, bad); free(tab); return -1; }
    printf("structure OK: %u slots, names, hashes, alignment, payload ranges\n", n_slots);
    *tab_out = tab; *n_out = n_slots;
    return 0;
}

int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: h3_pack plan <hf-dir> [plan.tsv]\n"
                        "       eg2_pack encode <hf-dir> <out.bin> [log.txt] [--cold cls,...] [--limit N] [--only SUBSTR] [--shard i/N] [--profile]\n"
                        "       eg2_pack verify <hf-dir> <in.bin>  [log.txt] [--cold cls,...] [--limit N] [--only SUBSTR] [--shard i/N]\n"
                        "       eg2_pack check  <in.bin>\n"
                        "       eg2_pack merge  <hf-dir> <out.bin> <log.txt> <shard.bin>...\n");
        return 2;
    }
    const char *mode = argv[1], *dir = argv[2];
    /* --cold <class>[,<class>...] switches the named classes to the reference's
     * cold tier (8x256). Everything else stays on the shipping tier (4x4096, the
     * highest quality measured in docs/ladder-embeddinggemma2.txt). */
    const char *cold = NULL;
    for (int i = 3; i < argc; i++) {
        if (!strcmp(argv[i], "--cold") && i + 1 < argc) cold = argv[++i];
        else if (!strcmp(argv[i], "--limit") && i + 1 < argc) limit_slots = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--only") && i + 1 < argc) only_sub = argv[++i];
        else if (!strcmp(argv[i], "--profile")) profile_mode = 1;
        else if (!strcmp(argv[i], "--shard") && i + 1 < argc) {
            const char *sp = argv[++i];
            if (sscanf(sp, "%d/%d", &shard_i, &shard_n) != 2 || shard_n < 1 || shard_i < 0 || shard_i >= shard_n) {
                fprintf(stderr, "--shard wants i/N with 0 <= i < N, got %s\n", sp);
                return 2;
            }
        }
    }
    if (cold) {
        char *tmp = strdup(cold), *tok = strtok(tmp, ",");
        while (tok) {
            for (int c = 0; c < C_NCLASS; c++) if (!strcmp(tok, CLS_NAME[c])) cold_mask |= 1 << c;
            tok = strtok(NULL, ",");
        }
        free(tmp);
    }

    if (!strcmp(mode, "plan")) {
        v41_st S;
        if (v41_st_open(&S, dir)) return 1;
        slot_t *slots; int n;
        if (build_plan(&S, &slots, &n)) return 1;
        plan_report(&S, slots, n, stdout);
        if (argc > 3 && argv[3][0] != '-') {
            FILE *f = fopen(argv[3], "w");
            if (!f) { fprintf(stderr, "cannot write %s\n", argv[3]); return 1; }
            plan_tsv(slots, n, f);
            fclose(f);
            printf("per-tensor plan -> %s\n", argv[3]);
        }
        v41_st_close(&S);
        return 0;
    }

    if (!strcmp(mode, "check")) {
        /* check <in.bin> — the container's structure only: magic, version, kind
         * range, name blob, hashes, ordering, alignment, payload ranges and every
         * geometry the reference's byte accounts imply. No re-encode, so this is the
         * cheap gate that a merged artifact is run through; the determinism proof
         * (re-encode and compare bytes) is verify, which is run per shard because a
         * payload only depends on its own matrix. */
        dvd_slot *tab = NULL; uint32_t n_tab = 0; uint64_t po = 0, pb = 0;
        const int rc = check_container(argv[2], &tab, &n_tab, &po, &pb);
        free(tab);
        return rc ? 1 : 0;
    }

    if (!strcmp(mode, "merge")) {
        /* merge <hf-dir> <out.bin> <log.txt> <shard.bin>... — --only/--limit are
         * accepted so the shard path can be checked against a single-process subset;
         * recipe_string stamps any subset subset SMOKE-TEST-NOT-A-RELEASE, so a
         * partial container cannot be mistaken for the release artifact. --shard is
         * refused: the plan a merge assembles is the whole plan. */
        if (argc < 6) { fprintf(stderr, "merge needs an output path, a log path and at least one shard\n"); return 2; }
        if (shard_n > 1) { fprintf(stderr, "merge builds the plan given by --only/--limit; --shard is not accepted here\n"); return 2; }
        /* the shard list is every remaining argument that is not a flag (the global
         * parser above already consumed the flags, so they may appear anywhere) */
        char **shards = malloc((size_t)argc * sizeof(char *));
        int n_shards = 0;
        for (int i = 5; i < argc; i++) {
            if (!strncmp(argv[i], "--", 2)) { i++; continue; }   /* flag and its value */
            shards[n_shards++] = argv[i];
        }
        if (!n_shards) { fprintf(stderr, "merge needs at least one shard container\n"); return 2; }
        merged_shards = n_shards;
        v41_st S;
        if (v41_st_open(&S, dir)) return 1;
        slot_t *slots; int n;
        if (build_plan(&S, &slots, &n)) return 1;
        const int rc = merge_run(&S, slots, n, argv[3], argv[4], shards, n_shards);
        v41_st_close(&S);
        return rc ? 1 : 0;
    }

    if (!strcmp(mode, "encode") || !strcmp(mode, "verify")) {
        const int verify_mode = !strcmp(mode, "verify");
        if (argc < 4) { fprintf(stderr, "%s needs an output/input path\n", mode); return 2; }
        const char *path = argv[3];
        const char *log = (argc > 4 && argv[4][0] != '-') ? argv[4] : NULL;
        dvd_slot *tab = NULL; uint32_t n_tab = 0;
        if (verify_mode) {
            uint64_t po, pb;
            if (check_container(path, &tab, &n_tab, &po, &pb)) return 1;
        }
        v41_st S;
        if (v41_st_open(&S, dir)) return 1;
        slot_t *slots; int n;
        if (build_plan(&S, &slots, &n)) return 1;
        const int rc = run(&S, slots, n, path, log, verify_mode, tab, n_tab);
        v41_st_close(&S);
        free(tab);
        return rc ? 1 : 0;
    }

    fprintf(stderr, "unknown mode %s\n", mode);
    return 2;
}
