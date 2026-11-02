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

/* Read the first `positions` positions of a cache back as f32, `k` and `v`
 * each [positions][kv_heads * head_dim] in position order through the slot
 * map. This is the writer a calibration capture needs (the toolkit's vq-kv/1
 * format) and it is not on any inference path. */
int gm4_gpu_kv_dump(gm4_gpu *g, gm4_kv *kv, int positions, float *k, float *v, char *err,
                    size_t errlen);

/* ---- the toolkit's VQ KV container --------------------------------------
 *
 * Decode a whole RQVQ container of DQVQ stages into row-major f32 on the
 * host: the same walk as the toolkit's Container::decode, stage 0 assigning
 * and each later stage adding. This is the reader a calibration codebook and
 * the KV gate use; a cache that lives in this form is read in the attention,
 * never through a decoded matrix. The returned buffer is released with
 * gm4_gpu_vq_free. */
int gm4_gpu_vq_decode(const char *path, float **out, int *rows, int *cols, char *err,
                      size_t errlen);
void gm4_gpu_vq_free(float *p);

/* ---- the VQ encoder ------------------------------------------------------
 *
 * A calibration container's codebooks, resident: one f16 codebook per residual
 * stage, as the toolkit's `vq-kv` wrote them. `gm4_gpu_cb_open` refuses a
 * container whose per-row gains are not 1.0, because a new row has no gain for
 * the runtime to apply (the toolkit's KV path leaves every gain at 1.0,
 * c/vendor/vq_qc.h:236 with X == NULL).
 *
 * `gm4_gpu_vq_encode` runs the reference's residual chain against them: each
 * stage assigns every `dim`-wide segment of a row its nearest codeword
 * (c2 - 2*(v . C[z]), first minimum), and the next stage sees what the
 * previous ones left. `idx` is [stages][nrows][cols/dim] u16; `rec` is the
 * f32 reconstruction the stages add up to and may be NULL. */
typedef struct gm4_cb gm4_cb;
int gm4_gpu_cb_open(const char *path, gm4_cb **out, char *err, size_t errlen);
void gm4_gpu_cb_free(gm4_cb *cb);
int gm4_gpu_cb_geometry(const gm4_cb *cb, int *nc, int *dim, int *stages, int *cols);
int gm4_gpu_vq_encode(gm4_cb *cb, const float *x, int nrows, int cols, uint16_t *idx, float *rec,
                      char *err, size_t errlen);

/* ---- a container read as a cache, for attention -------------------------
 *
 * A container opened as a cache keeps its per-stage codebooks and its packed
 * index stream on the device, so attention reads the index and the codeword
 * and never a decoded matrix - the format's point. The stream is the
 * toolkit's own layout, one bit stream over the whole matrix with a
 * three-byte guard read (kv_parse_payload above), so the bytes read here are
 * the bytes vq-kv wrote.
 *
 * `cap` is the number of rows the cache holds (0 takes the container's own);
 * `window` is the sliding window, and a positive window makes the cache a
 * ring of `cap` slots (position p lives at p % cap, the rule of
 * forward.rs:248-254 and of gm4_kv). Zero means an appending cache, which is
 * how a capture of `cap` positions is read. */
typedef struct gm4_vqkv gm4_vqkv;
int gm4_gpu_vqkv_open(gm4_gpu *g, const char *path, int cap, int window, gm4_vqkv **out,
                      char *err, size_t errlen);
void gm4_gpu_vqkv_free(gm4_vqkv *kv);
int gm4_gpu_vqkv_geometry(const gm4_vqkv *kv, int *rows, int *cols, int *dim, int *nc, int *stages);

/* One query token's attention over a VQ cache, the same call and schedule as
 * gm4_gpu_attention_ring (`pos` is the query's own position, so the cache
 * must hold it). K and V are separate containers; their geometry must agree.
 * The arithmetic is attn_split_kernel's, with each key and value read as
 * C0[idx0] + C1[idx1] - the sum gm4_gpu_vq_encode reproduces bit for bit -
 * and the container's per-row gain is 1.0 by construction (the encoder
 * refuses any other). */
int gm4_gpu_attention_vq(gm4_gpu *g, gm4_vqkv *k, gm4_vqkv *v, const float *q, int heads,
                         int head_dim, int pos, float *out, char *err, size_t errlen);

/* Encode one position's row against the calibration codebooks, on the device,
 * and pack it into the cache at its slot (`pos % cap` for a ring). `xd` is
 * [cols] f32 on the device, which is the layer's already normed and roped key
 * or value. This is the runtime's only write to a VQ cache. */
int gm4_gpu_vqkv_encode_row(gm4_gpu *g, gm4_vqkv *kv, int pos, const float *xd, char *err,
                            size_t errlen);

/* Encode `nrows` host rows into the cache, row r at position r. This is the
 * gate's capture-shaped call: it is the same per-row path the layer takes, so
 * a stream written here and one written by a step are the same bytes. */
int gm4_gpu_vqkv_encode_host(gm4_gpu *g, gm4_vqkv *kv, const float *x, int nrows, char *err,
                             size_t errlen);

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
    /* Optional: when both are set, this layer's key and value are written to
     * and read from VQ containers (the KV unit) instead of the f16 gm4_kv the
     * caller also passes. NULL keeps the f16 path. The caches are the
     * caller's and are not freed by gm4_gpu_layer_free. */
    gm4_vqkv *kvq_k, *kvq_v;
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

/* Rows per warp in the v3 cross-row stream, and its switch: a positive `span`
 * turns the stream on, 0 turns it off. Returns the previous span (0 = off).
 * Diagnostic, like gm4_gpu_span: the stream changes the pipeline over a span
 * of consecutive rows - the reference engine's mechanism at /data/ds4-dfm-rs/
 * cuda/ds41_vq_persist.cuh:46-56 and :86-116 - never a row's arithmetic, so
 * the gate's layer bands are the check that the two walks agree to the bit. */
int gm4_gpu_stream(int span);

/* Force the eight-launch expert path (legacy = 1) or the persistent walk
 * (legacy = 0); returns the previous value. Diagnostic only: both paths run
 * the same row arithmetic, so the A/B changes the launches, never a value.
 * The shape is the reference engine's, /data/ds4-dfm-rs/cuda/
 * ds41_vq_persist.cuh:157-238, and its own switch is at
 * cuda/ds41_vq_decode.cuh:329-335. */
int gm4_gpu_expert_legacy(gm4_gpu *g, int legacy);

/* ---- one token, the whole model ------------------------------------------
 *
 * The layers above chained into 30 and bounded by the two ends the CPU's step
 * runs around them (forward.rs:301-336): the tied table read as one row and
 * scaled by embed_scale, the final norm, the tied head over the whole
 * vocabulary, and the softcap. A step uploads nothing and downloads only the
 * logits; the activation moves between two device buffers, so the 30 layers
 * synchronise the host once, at the end.
 *
 * The layers and their caches stay the caller's: build them with
 * gm4_gpu_layer_new and gm4_gpu_kv_new, sizing every cache the way the CPU
 * session does (the window for a sliding layer, the context bound for a full
 * one), and pass them here. `embed` and `head` are tensor indices of the
 * residency and may be the same tensor: the checkpoint ties them
 * (`Gemma4ForCausalLM._tied_weights_keys`). */
typedef struct {
    gm4_layer *const *layers;
    gm4_kv *const *kv;
    int n;
    int embed, head;
    const float *final_norm;
    int hidden, vocab, threads; /* threads 0 means 256 */
    float eps, embed_scale, softcap;
} gm4_model_spec;

typedef struct gm4_model gm4_model;
int gm4_gpu_model_new(gm4_gpu *g, const gm4_model_spec *spec, gm4_model **out, char *err,
                      size_t errlen);
void gm4_gpu_model_free(gm4_model *m);

/* The whole device footprint: every chain's arena, every cache and the
 * model's own activation, norm and logits buffers. */
size_t gm4_gpu_model_bytes(const gm4_model *m);

/* One token at `pos`, which must be the caches' next position: the logits of
 * the position after it, softcapped, in `logits[vocab]`. */
int gm4_gpu_model_step(gm4_gpu *g, gm4_model *m, int token, int pos, float *logits, char *err,
                       size_t errlen);

/* A block of n tokens at pos0..pos0+n-1, in one submission per layer, and the
 * softcapped logits of the block's last position. This is the prefill the
 * step-per-token walk lacks: the caches take every position, so a prompt is
 * one call (or a few blocks), and a second block at pos0 + n reads the first
 * block's keys. The dense projections run the tiled tensor-core kernel over
 * the block and the routed experts run the reference engine's work items -
 * one expert's rows decoded once for the ~8 tokens that chose it - which is
 * where a block's speed over a token walk comes from.
 *
 * The block is limited to 1024 tokens; the scratch is grown on first use to
 * the longest block the session asks for and reused after that. */
int gm4_gpu_model_prefill(gm4_gpu *g, gm4_model *m, const int32_t *tokens, int n, int pos0,
                          float *logits, char *err, size_t errlen);

/* Diagnostic: copy one scratch buffer of the block path back as f32, so the
 * bisect can name the first wrong stage. The indices are gm4_gpu_layer_probe's
 * names where the buffer is single-use: 3 qb, 5 kb, 7 vb, 8 attn, 9 attn2,
 * 15 dense, 16 r, 17 scores, 18 moe, 19 moe2, 20 comb, 21 comb2, 22 outd.
 * `floats` is the caller's buffer size. Not on any inference path. */
int gm4_gpu_prefill_probe(gm4_model *m, int which, float *out, size_t floats, char *err,
                          size_t errlen);

/* Diagnostic: the block's expert lists, so a gate can compare the expert pass
 * against the step's values in the right order instead of by rms. which 0 is
 * the pairs' token (perm), 1 the slot each pair's down projection writes (dst),
 * 2 the (expert, first pair, count) items. `n` is the caller's capacity and is
 * the pair count for 0 and 1; the item count comes back through *count. */
int gm4_gpu_prefill_list(gm4_model *m, int which, int32_t *out, int n, int *count, char *err,
                         size_t errlen);

/* Diagnostic: this layer's router selection (which 0) or its renormalised
 * weights as f32 (which 1), so a gate can compare a block's routing against
 * the step's. `n` must be the layer's top_k. */
int gm4_gpu_layer_list(gm4_layer *L, int which, void *out, int n, char *err, size_t errlen);

/* The block path's device footprint, 0 before its first block. */
size_t gm4_gpu_model_prefill_bytes(const gm4_model *m);

/* Diagnostic: the tiled kernel's token tile and codebook placement for the
 * block path's dense projections (gm4_gpu_prefill's six combinations). Returns
 * the previous tile; a non-positive `bn` leaves it alone, and cb < 0 leaves the
 * placement alone. */
int gm4_gpu_prefill_cfg(int bn, int cb);

/* The last step's two halves in milliseconds, the reference engine's account
 * of a decode step (/data/ds4-dfm-rs/ds4_ds41_graph.inc:106-113): `submit` is
 * the host time from entry to the last launch's return, `wait` is the sync and
 * the logits readback after it. A single wall number cannot say which half the
 * next change has to move. Either pointer may be NULL. */
int gm4_gpu_model_split(const gm4_model *m, double *submit, double *wait);

/* How the steps ran: the number of position buckets captured, and whether the
 * graph path is off (a capture failed, so every step walks direct, which is
 * correct but pays the per-launch submit cost). */
int gm4_gpu_model_graphs(const gm4_model *m, int *captures, int *off);

/* Turn the graph path on or off, for a gate that compares a replayed step with
 * the direct walk of the same step: the same kernels in the same order, so the
 * two must agree bit for bit. */
int gm4_gpu_model_set_graph(gm4_model *m, int on);

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
