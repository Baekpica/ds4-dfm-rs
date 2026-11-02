/* gemma4_gpu.h - device residency and the fused VQ row dot for the
 * Gemma-4-26B-A4B release, behind a plain C ABI.
 *
 * The caller (crates/ds4-core, example gemma4_a4b_gpu_gate) opens a
 * vq-release/1 artifact, lists every encoded container with the shape the
 * release records, and this module maps, checks and uploads each one once.
 * A matvec then runs wholly on the device: one fused gather-and-dot per
 * payload, stages summed in the payload's own gain, nothing decoded into a
 * weight matrix. A row-list gather (gm4_gpu_rows) reads only the rows a
 * mixture layer routed to, from the same resident containers.
 *
 * The payload is the TOOLS' DQV3 (24-byte header, inline f16 codebook,
 * min(nbit, 12)-bit row stream, no plane), which is what this artifact
 * carries. It is not the engine's DQV3 in /data/YoungAi (32-byte header,
 * layer-shared codebook at cb_off, fixed 12 bits); the discriminator is the
 * word at payload offset 20, zero here and the engine's mnb there.
 *
 * Failures are returned as -1 with a message in the caller's buffer; nothing
 * falls back to host streaming, because a silent fallback would hide exactly
 * the residency failure this module exists to report.
 */
#ifndef GEMMA4_GPU_H
#define GEMMA4_GPU_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    const char *path;   /* an RQVQ container file */
    uint32_t rows;      /* the release's shape, checked against the header */
    uint32_t cols;
} gm4_spec;

typedef struct {
    uint64_t files_bytes;       /* mmapped file size, all containers */
    uint64_t payload_bytes;     /* sum of the container payload region */
    uint64_t device_bytes;      /* cudaMalloc'd, including slack and padding */
    uint64_t device_free_before;
    uint64_t device_free_after;
    int tensors;
    int payloads;
} gm4_account;

typedef struct gm4_gpu gm4_gpu;

/* Upload every container in `specs`. Returns NULL on failure, with the reason
 * in `err`. The module is a singleton: cudaSetDevice(0) is fixed at open. */
gm4_gpu *gm4_gpu_open(const gm4_spec *specs, int n, char *err, size_t errlen);
void gm4_gpu_close(gm4_gpu *g);

/* What was allocated, and the device's free memory before and after. */
void gm4_gpu_account_of(const gm4_gpu *g, gm4_account *out);

/* The device name the residency was measured on ("NVIDIA GeForce RTX 4070
 * SUPER" here), owned by the handle. */
const char *gm4_gpu_device_name(const gm4_gpu *g);

/* y[rows] = sum over the container's payloads of payload_dot(row) * gain,
 * with x[cols] read from the host. `threads` is the launch width of the
 * warp-per-row kernels (a multiple of 32 up to 1024); it changes how many
 * warps share a block, never the arithmetic. Returns 0, or -1 with `err`. */
int gm4_gpu_matvec(gm4_gpu *g, int tensor, const float *x, float *y, int threads,
                   char *err, size_t errlen);

/* y[i] = the dot of x with row rows[i], in the order given, summed over the
 * container's payloads with each stage's own gain. The list is host row
 * numbers of the whole tensor; it is uploaded once per call and every entry
 * must be below the tensor's row count. This is the routed-expert gather:
 * eight chosen experts touch 11264 of the 180224 rows of
 * `experts.gate_up_proj` and 22528 of the 360448 rows of
 * `experts.down_proj`, so the read is the selected rows and their gains, not
 * the stack. A repeated row produces a repeated output. */
int gm4_gpu_rows(gm4_gpu *g, int tensor, const uint32_t *rows, int nrows,
                 const float *x, float *y, int threads, char *err, size_t errlen);

/* Which kernel a tensor's payloads run: 3 = the DQV3 block kernel, 2 = the
 * dim-4 fast path, 1 = the generic warp kernel. Per payload; they agree for
 * a real container because the geometry is constant across its payloads. */
int gm4_gpu_variant(const gm4_gpu *g, int tensor);

/* Tiled prefill, the VQ tools' f16 tensor-core path
 * (/data/VQ_tools/c/gpu/decode-gpu.cu:320-520). xT is the transposed
 * activation matrix [cols][ntok] and y is [ntok][rows], both host f32; y is
 * cleared and then filled with every payload's contribution, so the caller
 * reads the complete result and need not zero it.
 *
 * bn is the token tile (64, 128 or 256) and cb the codebook placement (0 =
 * global/L1, 1 = shared behind the tiles). The column round is 64: the
 * A-tile fill reads one 64-bit window per thread and BK 128 would silently
 * truncate, which is why this build offers only BK 64. bn 0 is the auto
 * choice: the tiled kernel at 128/0 when the geometry fits, the legacy
 * one-thread-per-(row, token) kernel otherwise (a dim-4 8-bit codebook, for
 * instance). */
int gm4_gpu_prefill(gm4_gpu *g, int tensor, const float *xT, int ntok, float *y, int bn, int cb,
                    char *err, size_t errlen);

/* The same prefill over a contiguous row slice, written into the full-row
 * layout y[t * rows + r], so a slice produces exactly the addresses a full
 * launch writes for those rows. This is a routed expert's prefill: its rows
 * are one contiguous block of the stack. */
int gm4_gpu_prefill_rows(gm4_gpu *g, int tensor, int row0, int nrows, const float *xT, int ntok,
                         float *y, int bn, int cb, char *err, size_t errlen);

/* Whether the tiled prefill can run this tensor with this tile and placement:
 * 1 when it can, 0 when it cannot, with the reason in `err`. */
int gm4_gpu_prefill_ready(const gm4_gpu *g, int tensor, int bn, int cb, char *err, size_t errlen);

/* The layer's elementwise arithmetic, the same definitions the CPU forward
 * uses (crates/ds4-core/src/gemma4_a4b/forward.rs): RMSNorm with `eps` and an
 * optional weight, gelu_pytorch_tanh times a second vector, a residual add
 * with an optional addend and a scale, and the logit softcap. Host in, host
 * out; every kernel is gated against the CPU forward, so the chain has one
 * definition of each op. */
int gm4_gpu_rmsnorm(gm4_gpu *g, const float *x, int rows, int width, const float *weight, float *y,
                    float eps, char *err, size_t errlen);
int gm4_gpu_gelu_mul(gm4_gpu *g, const float *gate, const float *up, float *act, size_t n,
                     char *err, size_t errlen);
int gm4_gpu_residual_scale(gm4_gpu *g, const float *x, const float *add, float *out, size_t n,
                           float scale, char *err, size_t errlen);
int gm4_gpu_softcap(gm4_gpu *g, float *y, size_t n, float cap, char *err, size_t errlen);

/* Rotary, in place, on [heads][head_dim] at position `pos`. `inv` is the layer
 * type's table (Config::rope_inv): head_dim/2 entries, zero past the rotated
 * pairs, so the unrotated half is the identity (the reference's
 * cat(freqs, freqs) fold). */
int gm4_gpu_rope(gm4_gpu *g, float *x, int heads, int head_dim, int pos, const float *inv,
                 char *err, size_t errlen);

/* One query token's attention over `positions` keys and values: the last
 * `window` of them when window is positive (the sliding layers), all of them
 * when it is zero (the full layers). q is [heads][head_dim]; k and v are
 * [positions][kv_heads][head_dim], rounded to f16 on upload, which is the
 * cache's format. Softmax is fp32 and the scaling is 1.0
 * (modeling_gemma4.py:846, :1200). */
int gm4_gpu_attention(gm4_gpu *g, const float *q, const float *k, const float *v, int heads,
                      int kv_heads, int head_dim, int positions, int window, float *out,
                      char *err, size_t errlen);

/* A layer's key/value cache on the device, the structure the CPU session
 * keeps (crates/ds4-core/src/gemma4_a4b/forward.rs:239-269): `per =
 * kv_heads * head_dim` f16 values per position, `cap` positions, and
 * `slot(pos) = pos % cap` for a ring layer, `pos` for an appending one.
 * `window > 0` makes a ring of that many positions (a sliding layer's
 * config.sliding_window); `window == 0` makes an appending cache of
 * `max_pos` (the caller's session bound for a full layer, which the CPU
 * grows lazily instead). The handle is opaque and owns its two buffers. */
typedef struct gm4_kv gm4_kv;
int gm4_gpu_kv_new(gm4_gpu *g, int kv_heads, int head_dim, int window, int max_pos, gm4_kv **out,
                   char *err, size_t errlen);
void gm4_gpu_kv_free(gm4_kv *kv);
size_t gm4_gpu_kv_bytes(const gm4_kv *kv);
int gm4_gpu_kv_reset(gm4_gpu *g, gm4_kv *kv, char *err, size_t errlen);

/* Write position `pos` into its slot, rounding to f16, which is the cache's
 * format. A ring accepts any non-negative position; an appending cache
 * refuses pos >= cap. */
int gm4_gpu_kv_put(gm4_gpu *g, gm4_kv *kv, int pos, const float *k, const float *v, char *err,
                   size_t errlen);

/* Read a position's slot back as f32, for checking the slot walk. */
int gm4_gpu_kv_get(gm4_gpu *g, gm4_kv *kv, int pos, float *k, float *v, char *err, size_t errlen);

/* One query token's attention over the cache at `pos`: the ring's last `cap`
 * positions (a ring created with the layer's window), or every position of an
 * appending cache. gm4_gpu_kv_put(pos) must come first, so the query sees its
 * own key and value. q is [heads][head_dim]; the output is the same width.
 * fp32 softmax, scaling 1.0, exactly gm4_gpu_attention's arithmetic. */
int gm4_gpu_attention_ring(gm4_gpu *g, gm4_kv *kv, const float *q, int heads, int pos, float *out,
                           char *err, size_t errlen);

/* One layer's wiring: the resident tensors by index (`v` is -1 where the layer
 * has no v_proj, the `k_eq_v` full layers), the geometry, and the small
 * vectors the CPU layer keeps beside its matrices. The vectors are uploaded
 * once at gm4_gpu_layer_new, so a step uploads only the input activation.
 * `router_scale` is folded with 1/sqrt(hidden) here, exactly as the CPU folds
 * it (forward.rs:362-365). */
typedef struct {
    int q, k, v, o, dense_gate, dense_up, dense_down, router, expert_gate_up, expert_down;
    int heads, kv_heads, head_dim, hidden, dense_ff, moe_ff, experts, top_k;
    int threads; /* launch width; 0 means 256 */
    float eps;
    float layer_scalar;
    const float *input_norm, *post_attn_norm, *pre_ffn_norm, *post_ffn_norm;
    const float *pre_ffn_norm2, *post_ffn_norm1, *post_ffn_norm2;
    const float *q_norm, *k_norm, *router_scale, *per_expert_scale, *rope_inv;
} gm4_layer_spec;

typedef struct gm4_layer gm4_layer;
int gm4_gpu_layer_new(gm4_gpu *g, const gm4_layer_spec *spec, gm4_layer **out, char *err,
                      size_t errlen);
void gm4_gpu_layer_free(gm4_layer *L);

/* What the layer holds on the device: one arena for every intermediate, the
 * small vectors, the picks and the two row lists. Allocated once here, never
 * grown. */
size_t gm4_gpu_layer_bytes(const gm4_layer *L);

/* One layer, start to finish, on the device: the CPU layer's order
 * (forward.rs:335-398), including the o_proj that ends its attention (:462),
 * the dense feed-forward, the router's selection, the routed experts and the
 * two closing norms. The input activation is uploaded once and the output
 * downloaded once; nothing in between leaves the device, and the router's
 * picks stay there. `pos` must be the cache's next position: the key and value
 * are written to it before attention reads it. */
int gm4_gpu_layer_forward(gm4_gpu *g, gm4_layer *L, gm4_kv *kv, const float *x, int pos,
                          float *out, char *err, size_t errlen);

/* Diagnostic: copy one intermediate of the last forward back as f32, for the
 * gate to locate a wiring error. Not on any inference path, and the arena
 * reuses buffers, so a name is the state the forward leaves there. `which` is
 * 0 xa, 1 hb, 2 qa, 3 qb, 4 ka, 5 kb, 6 va, 7 vb, 8 attn, 9 attn2, 10 xb,
 * 11 dg, 12 du, 13 dact, 14 dd, 15 dense, 16 r, 17 scores, 18 moe, 19 moe2,
 * 20 comb, 21 comb2, 22 outd, 23 gu, 24 act, 25 d. `n` is the caller's width
 * for the value and must fit the buffer. */
int gm4_gpu_layer_probe(gm4_layer *L, int which, float *out, int n, char *err, size_t errlen);

/* Diagnostic: turn the chain's nine section records on, then read the last
 * forward's spans in milliseconds with gm4_gpu_layer_sections. The records are
 * stream-ordered timestamps, not waits, so turning them on does not serialise
 * the chain; the arithmetic and the launch order are unchanged either way. */
int gm4_gpu_layer_timing(gm4_layer *L, int on);
int gm4_gpu_layer_sections(gm4_layer *L, double *ms, const char **names, int n);

/* Rows per warp in the span kernel (the v3 geometry too short for the block
 * kernel's whole 32-index groups: the expert and dense down projections).
 * Default 4. Returns the previous value; a non-positive `span` leaves it
 * unchanged. Diagnostic: it changes how much independent work one warp holds,
 * never a row's arithmetic. */
int gm4_gpu_span(int span);

/* out = x * w over n values: the router's per-dimension scale. */
int gm4_gpu_mul(gm4_gpu *g, const float *x, const float *w, float *out, size_t n, char *err,
                size_t errlen);

/* The router's selection over `experts` scores: the fp32 softmax, the k
 * indices highest first with ties to the lower index, and their renormalised
 * weights with the per-expert scale folded in (`per_expert_scale` may be
 * NULL). The CPU forward's rule is forward.rs:80-88. */
int gm4_gpu_router_topk(gm4_gpu *g, const float *scores, const float *per_expert_scale, int experts,
                        int k, float *probs, int *indices, float *weights, char *err,
                        size_t errlen);

#ifdef __cplusplus
}
#endif

#endif /* GEMMA4_GPU_H */
