/* cuda/ds41_head.cuh — DeepSeek V4.1 (ds41) coarse-to-fine output head.
 *
 * The head is `output.weight` (q4_K, 129,280 x 5,120 = 372.3 MB) and the MTP
 * step reads it twice: once for the verify rows and once for the draft tower
 * (`ds4_ds41_forward.inc:1202`, `ds4_ds41_draft.inc:348`), 744.6 MB of a
 * ~6.8 GB step.  The coarse-to-fine head scores a 1024-column Q8 sketch of the
 * head (132.4 MB), keeps the top 3,584 vocab ids above the 512 anchors that
 * are always candidates, and rescores those 4,096 rows exactly from the
 * artifact's own q4_K bytes (11.8 MB): 144.2 MB per call instead of 372.3 MB.
 *
 * Idea source: Flash1017 `predict_coarse_to_fine` / `build_and_save` (see
 * docs/ds41-head-gate-2026-11-09.md for the reference lines).  That tree is
 * source-available, not MIT, so only the idea is used; this is a fresh
 * implementation against this artifact's q4_K head.
 *
 * Greedy only, by contract: the device sampler and the sampled-speculation
 * verify read the whole logits row, so a truncated row would change the
 * top-p/top-k support.  The caller sets the flag only for a bare-argmax
 * request.  The truncated row keeps the logits interface — every id outside
 * the candidate set holds -INFINITY — so the argmax kernel, the markov bias
 * add and the per-row pick stay unchanged.
 *
 * The sketch is a derived view, built lazily on the first coarse call (read
 * 372 MB, quantize; ~0.5 s once per process) and held in device memory
 * (132.9 MB).  No file is written and the artifact is untouched.  Column
 * leverage is `col_energy[c] = sum_r (w[r][c]*norm[c])^2` over a 512-row
 * stride sample; column set = the top half by energy plus a stride sample of
 * the remaining columns, ascending; per-row symmetric Q8 over that set.
 *
 * Every step is a kernel launch in the current stream, so a captured decode
 * graph replays it: gather the sketch columns of each token row -> coarse
 * score -> radix top-k -> -inf fill -> exact q4_K rescore at the candidate
 * ids.  The build is the only host-synchronizing part and refuses to run
 * inside a capture.
 */
#pragma once
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <vector>

#define V41_HEAD_ANCHOR 512u    /* ids [0, anchor) are always candidates */
#define V41_HEAD_PICK 3584u     /* sketch-picked ids; candidates = anchor + pick = 4,096 */
#define V41_HEAD_COLS 1024u     /* sketch columns (gate: 100% recall at 2K and 8K) */
#define V41_HEAD_SAMPLE 512u    /* rows in the column-leverage stride sample */
#define V41_HEAD_TOK 8u         /* the largest verify/draft width the head takes */
#define V41_HEAD_TOPK_THREADS 1024u

typedef struct {
    int state;              /* 0 = not built, 1 = ready, -1 = refused (the full head runs) */
    uint64_t w_off;         /* the head tensor this sketch was built from */
    uint32_t vocab, dim;
    int8_t *q8;             /* [vocab][cols] per-row symmetric Q8 over the selected columns */
    float *scale;           /* [vocab] Q8 row scale (max_abs/127, 1 when the row is ~0) */
    uint16_t *cols;         /* [cols] selected column ids, ascending */
    float *coarse;          /* [V41_HEAD_TOK][vocab] coarse scores */
    float *xsel;            /* [V41_HEAD_TOK][cols] activation gathered at the selected columns */
    int32_t *cand;          /* [V41_HEAD_TOK][anchor+pick] candidate vocab ids */
} v41_head_c2f;

static v41_head_c2f g_v41_hc = {0};

/* Order-preserving float key for the radix select, the indexer's mapping
 * (cuda/ds41_indexer.cuh v41_topk_key): -inf and NaN map to 0 and are never
 * selected; otherwise the sign bit is flipped so the unsigned compare matches
 * the float order. */
__device__ __forceinline__ static uint32_t v41_head_key(float f) {
    if (!(f > -INFINITY)) return 0u;
    uint32_t u; memcpy(&u, &f, 4);
    return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}

/* One q4_K row -> f32: the lane layout is the engine's embed decoder
 * (cuda/ds41_q4k.cuh v41_q4k_embed_kernel) so both paths write the same value
 * bits.  `b` is the 256-element block inside the row. */
__device__ __forceinline__ static void v41_head_deq_blk(float *o, const uint8_t *blk) {
    const uint32_t lane = threadIdx.x & 31u, gidx = lane >> 3, q0 = (lane & 7u) * 4u;
    const float d = __half2float(__ushort_as_half((unsigned short)((uint16_t)blk[0] | ((uint16_t)blk[1] << 8))));
    const float dmin = __half2float(__ushort_as_half((unsigned short)((uint16_t)blk[2] | ((uint16_t)blk[3] << 8))));
    float s_lo, m_lo, s_hi, m_hi;
    v41_q4k_sm(blk + 4, (int)(gidx * 2u), &s_lo, &m_lo);
    v41_q4k_sm(blk + 4, (int)(gidx * 2u + 1u), &s_hi, &m_hi);
    const uint32_t qw = *(const uint32_t *)(blk + 16u + 4u * lane);
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        const uint32_t byte = (qw >> (8 * i)) & 0xFFu;
        o[gidx * 64u + q0 + i]       = d * s_lo * (float)(byte & 0xFu) - dmin * m_lo;
        o[gidx * 64u + 32u + q0 + i] = d * s_hi * (float)(byte >> 4)  - dmin * m_hi;
    }
}

/* Column leverage over the sampled rows: energy[c] = sum_r (row[r][c]*norm[c])^2.
 * One thread per column, the sample is row-major in `rows` (10.5 MB, L2-fed).
 * Accumulated in double to match the gate harness that measured the column
 * set (misc/scratch/ds41_head_gate.rs); the cost is boot-only. */
__global__ static void v41_head_energy_kernel(float *energy, const float *rows, const float *norm,
                                              uint32_t ns, uint32_t dim) {
    const uint32_t c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= dim) return;
    const double nc = (double)norm[c];
    double e = 0.0;
    for (uint32_t r = 0; r < ns; r++) {
        const double p = (double)rows[(uint64_t)r * dim + c] * nc;
        e += p * p;
    }
    energy[c] = (float)e;
}

/* One head row -> one sketch row.  Grid.x = vocab, 256 threads (8 warps); the
 * row decodes to shared (dim*4 bytes, 20 KB at 5,120) once, then the max over
 * the selected columns and the Q8 quantization read it back. */
__global__ static void __launch_bounds__(256) v41_head_q8_kernel(int8_t *q8, float *scale, const uint8_t *w,
                                                                 const uint16_t *cols, uint32_t vocab, uint32_t dim,
                                                                 uint32_t ncols) {
    extern __shared__ float srow[];
    const uint32_t v = blockIdx.x, tid = threadIdx.x, nblk = dim / V41_Q4K_BLK;
    for (uint32_t b = tid >> 5; b < nblk; b += blockDim.x >> 5) {
        v41_head_deq_blk(srow + b * V41_Q4K_BLK, w + ((uint64_t)v * nblk + b) * V41_Q4K_BYTES);
    }
    __syncthreads();
    __shared__ float red[256];
    float m = 0.f;
    for (uint32_t i = tid; i < ncols; i += blockDim.x) {
        const float a = fabsf(srow[cols[i]]);
        if (a > m) m = a;
    }
    red[tid] = m;
    __syncthreads();
    for (uint32_t s = blockDim.x >> 1; s > 0u; s >>= 1) {
        if (tid < s) red[tid] = fmaxf(red[tid], red[tid + s]);
        __syncthreads();
    }
    /* Per-row symmetric Q8 (the gate's bytes: max/127, round half away from
     * zero, clamp to +-127); a ~zero row keeps scale 1 and quantizes to 0. */
    const float mx = red[0];
    const bool zero = !(mx > 1e-12f);
    const float inv = zero ? 0.f : 127.f / mx;
    if (tid == 0u) scale[v] = zero ? 1.f : mx / 127.f;
    int8_t *dst = q8 + (uint64_t)v * ncols;
    for (uint32_t i = tid; i < ncols; i += blockDim.x) {
        const float qv = srow[cols[i]] * inv;
        const float r = qv >= 0.f ? floorf(qv + 0.5f) : ceilf(qv - 0.5f);
        dst[i] = (int8_t)(r > 127.f ? 127.f : (r < -127.f ? -127.f : r));
    }
}

/* Gather each token row's selected columns once, so the coarse dot reads the
 * sketch and a contiguous activation. */
__global__ static void v41_head_xsel_kernel(float *xs, const float *x, uint32_t x_stride,
                                            const uint16_t *cols, uint32_t ncols) {
    const uint32_t t = blockIdx.x;
    for (uint32_t i = threadIdx.x; i < ncols; i += blockDim.x) {
        xs[(uint64_t)t * ncols + i] = x[(uint64_t)t * x_stride + cols[i]];
    }
}

/* Coarse score: out[t][v] = scale[v] * sum_i q8[v][i] * xs[t][i].  One warp per
 * vocab row, one 8-byte sketch span and one 8-float activation span per lane
 * per iteration (8 bytes keeps every load 8-byte aligned: ncols % 8 == 0). */
__global__ static void v41_head_coarse_kernel(float *out, const int8_t *q8, const float *scale, const float *xs,
                                              uint32_t ncols, uint32_t vocab, uint32_t n_tok) {
    const uint32_t lane = threadIdx.x & 31u, warp = threadIdx.x >> 5u;
    const uint32_t row = blockIdx.x * (blockDim.x >> 5u) + warp;
    if (row >= vocab) return;
    const int8_t *q = q8 + (uint64_t)row * ncols;
    for (uint32_t t = 0; t < n_tok; t++) {
        const float *xt = xs + (uint64_t)t * ncols;
        float acc = 0.f;
        for (uint32_t i = lane * 8u; i < ncols; i += 32u * 8u) {
            const uchar4 qa = *(const uchar4 *)(q + i), qb = *(const uchar4 *)(q + i + 4u);
            const float4 xa = *(const float4 *)(xt + i), xb = *(const float4 *)(xt + i + 4u);
            acc += (float)(int8_t)qa.x * xa.x + (float)(int8_t)qa.y * xa.y
                 + (float)(int8_t)qa.z * xa.z + (float)(int8_t)qa.w * xa.w
                 + (float)(int8_t)qb.x * xb.x + (float)(int8_t)qb.y * xb.y
                 + (float)(int8_t)qb.z * xb.z + (float)(int8_t)qb.w * xb.w;
        }
        #pragma unroll
        for (int o = 16; o; o >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, o);
        if (lane == 0u) out[(uint64_t)t * vocab + row] = acc * scale[row];
    }
}

/* Radix select, the indexer's method (cuda/ds41_indexer.cuh:335-489): 8 bits a
 * round over an order-preserving key, then a segmented ascending write-out.
 * The set, the order and the tie rule (larger value, lower id) are that
 * kernel's; the domain is the vocab ids at or above `anchor`.  Threads 0..511
 * also write the anchors, which are candidates unconditionally. */
__global__ static void __launch_bounds__(V41_HEAD_TOPK_THREADS) v41_head_topk_kernel(
        int32_t *cand, const float *score, uint32_t vocab, uint32_t anchor, uint32_t pick) {
    const uint32_t t = blockIdx.x, tid = threadIdx.x, nt = blockDim.x;
    int32_t *cl = cand + (uint64_t)t * (anchor + pick);
    for (uint32_t i = tid; i < anchor + pick; i += nt) cl[i] = i < anchor ? (int32_t)i : -1;
    const float *s = score + (uint64_t)t * vocab + anchor;
    const uint32_t ns = vocab - anchor;
    __shared__ uint32_t hist[256], suf[257];
    __shared__ uint32_t sh_bucket;
    uint32_t prefix = 0, kk = pick < ns ? pick : ns;
    for (int shift = 24; shift >= 0; shift -= 8) {
        for (uint32_t b = tid; b < 256u; b += nt) hist[b] = 0u;
        __syncthreads();
        for (uint32_t g = tid; g < ns; g += nt) {
            const uint32_t key = v41_head_key(s[g]);
            if (key == 0u || (shift < 24 && (key >> (shift + 8)) != (prefix >> (shift + 8)))) continue;
            atomicAdd(&hist[(key >> shift) & 0xFFu], 1u);
        }
        __syncthreads();
        if (tid < 32u) {   /* suffix sums suf[b] = sum_{b' >= b} hist[b'] over the lane's 8 buckets */
            uint32_t loc = 0;
            for (uint32_t b = 0; b < 8u; b++) loc += hist[tid * 8u + b];
            uint32_t inc = loc;
            for (int o = 1; o < 32; o <<= 1) {
                const uint32_t v = __shfl_down_sync(0xffffffffu, inc, o);
                if (tid + (uint32_t)o < 32u) inc += v;
            }
            uint32_t run = inc - loc;
            for (int b = 7; b >= 0; b--) {
                run += hist[tid * 8u + (uint32_t)b];
                suf[tid * 8u + (uint32_t)b] = run;
            }
            if (tid == 0u) { suf[256] = 0u; sh_bucket = 0u; }
        }
        __syncthreads();
        if (shift == 24) {   /* round 1 covers every valid key, so suf[0] is the valid count */
            const uint32_t nval = suf[0];
            if (kk > nval) kk = nval;
            if (kk == 0u) break;   /* no candidate is selectable: every slot keeps -1 */
        }
        for (uint32_t b = 1u + tid; b < 256u; b += nt) if (suf[b] >= kk) atomicMax(&sh_bucket, b);
        __syncthreads();
        const uint32_t bs = sh_bucket;
        prefix |= bs << shift;
        kk -= suf[bs + 1u];   /* still wanted inside the bucket */
        __syncthreads();
    }
    /* Segmented write-out, ascending (deterministic: the same request selects
     * the same ids in the same order).  Each thread counts its span's > and
     * == threshold entries; two prefix sums give the span starts. */
    __shared__ uint32_t cgt[V41_HEAD_TOPK_THREADS], ceq[V41_HEAD_TOPK_THREADS];
    const uint32_t chunk = (ns + nt - 1u) / nt;
    const uint32_t g0 = tid * chunk, g1 = (g0 + chunk) < ns ? (g0 + chunk) : ns;
    uint32_t ngt = 0, neq = 0;
    for (uint32_t g = g0; g < g1; g++) {
        const uint32_t key = v41_head_key(s[g]);
        if (key == 0u) continue;
        if (key > prefix) ngt++;
        else if (key == prefix) neq++;
    }
    cgt[tid] = ngt;
    ceq[tid] = neq;
    __syncthreads();
    {   /* shift right one, then Hillis-Steele: cgt/ceq become exclusive prefix sums */
        const uint32_t vg = tid ? cgt[tid - 1u] : 0u;
        const uint32_t ve = tid ? ceq[tid - 1u] : 0u;
        __syncthreads();
        cgt[tid] = vg;
        ceq[tid] = ve;
        __syncthreads();
    }
    for (uint32_t off = 1u; off < nt; off <<= 1) {
        const uint32_t ag = tid >= off ? cgt[tid - off] : 0u;
        const uint32_t ae = tid >= off ? ceq[tid - off] : 0u;
        __syncthreads();
        cgt[tid] += ag;
        ceq[tid] += ae;
        __syncthreads();
    }
    const uint32_t eq = kk;   /* how many == threshold the radix tail still wants */
    uint32_t wgt = cgt[tid], weq = ceq[tid];
    for (uint32_t g = g0; g < g1; g++) {
        const uint32_t key = v41_head_key(s[g]);
        if (key == 0u) continue;
        if (key > prefix) {
            const uint32_t pos = wgt + (weq < eq ? weq : eq);
            if (pos < pick) cl[anchor + pos] = (int32_t)(anchor + g);
            wgt++;
        } else if (key == prefix) {
            if (weq < eq) {
                const uint32_t pos = wgt + weq;
                if (pos < pick) cl[anchor + pos] = (int32_t)(anchor + g);
            }
            weq++;
        }
    }
}

/* Every id outside the candidate set holds -inf: the truncated row is still a
 * logits row for the argmax and the markov add. */
__global__ static void v41_head_fill_kernel(float *out, uint64_t n4) {
    const uint64_t i = ((uint64_t)blockIdx.x * blockDim.x + threadIdx.x) * 4u;
    if (i < n4 * 4u) *(float4 *)(out + i) = make_float4(-INFINITY, -INFINITY, -INFINITY, -INFINITY);
}

/* Exact rescore of the candidate rows with the artifact's own q4_K head: one
 * warp per (token row, candidate) pair — every token row has its own
 * candidate set, so no weight read is shared across rows.  out[t][id] =
 * <row id, x row t>. */
__global__ static void v41_head_rescore_kernel(float *out, uint32_t out_stride, const uint8_t *w,
                                               const int32_t *cand, uint32_t ncand, uint32_t n_tok,
                                               uint32_t nblk, const float *x, uint32_t x_stride) {
    const uint32_t warp = threadIdx.x >> 5u, lane = threadIdx.x & 31u;
    const uint64_t f = (uint64_t)blockIdx.x * (blockDim.x >> 5u) + warp;
    if (f >= (uint64_t)n_tok * ncand) return;
    const int32_t id = cand[f];
    if (id < 0) return;
    const uint64_t t = f / ncand;
    float acc[1] = {0.f};
    v41_q4k_rown_smem<1u>(w + (uint64_t)id * nblk * V41_Q4K_BYTES, nblk, 1u, 0u,
                          x + t * x_stride, x_stride, acc);
    #pragma unroll
    for (int o = 16; o; o >>= 1) acc[0] += __shfl_xor_sync(0xffffffffu, acc[0], o);
    if (lane == 0u) out[t * out_stride + (uint32_t)id] = acc[0];
}

/* Build the sketch once per head tensor.  Host side (device sync is fine: this
 * runs on the first coarse call, outside any capture). */
static int v41_head_build(const void *model_map, uint64_t model_size, uint64_t w_off, uint64_t n_off,
                          uint32_t vocab, uint32_t dim) {
    const uint8_t *w = (const uint8_t *)cuda_model_range_ptr(model_map, w_off, (uint64_t)vocab * (dim / V41_Q4K_BLK) * V41_Q4K_BYTES, "v41 head q4_K");
    const float *norm = (const float *)cuda_model_range_ptr(model_map, n_off, (uint64_t)dim * 4u, "v41 head norm");
    if (!w || !norm) return 0;
    const uint32_t nblk = dim / V41_Q4K_BLK;
    const uint32_t step = vocab / V41_HEAD_SAMPLE ? vocab / V41_HEAD_SAMPLE : 1u;
    const uint32_t ns = (vocab + step - 1u) / step;

    if (cudaMalloc((void **)&g_v41_hc.q8, (size_t)vocab * V41_HEAD_COLS) != cudaSuccess ||
        cudaMalloc((void **)&g_v41_hc.scale, (size_t)vocab * 4u) != cudaSuccess ||
        cudaMalloc((void **)&g_v41_hc.cols, V41_HEAD_COLS * 2u) != cudaSuccess ||
        cudaMalloc((void **)&g_v41_hc.coarse, (size_t)V41_HEAD_TOK * vocab * 4u) != cudaSuccess ||
        cudaMalloc((void **)&g_v41_hc.xsel, (size_t)V41_HEAD_TOK * V41_HEAD_COLS * 4u) != cudaSuccess ||
        cudaMalloc((void **)&g_v41_hc.cand, (size_t)V41_HEAD_TOK * (V41_HEAD_ANCHOR + V41_HEAD_PICK) * 4u) != cudaSuccess) {
        (void)cudaGetLastError();
        fprintf(stderr, "ds4: [v41] head sketch allocation failed (%.1f MB)\n", (double)vocab * V41_HEAD_COLS / 1048576.0);
        return 0;
    }

    /* 1. Column leverage over the stride sample: decode the sample rows once
     * (the embed kernel's demand decode, one row id per blockIdx.y). */
    float *rows = NULL;
    int32_t *ids = NULL;
    float *energy = NULL;
    float *h_energy = (float *)malloc((size_t)dim * 4u);
    int ok = h_energy != NULL &&
             cudaMalloc((void **)&rows, (size_t)ns * dim * 4u) == cudaSuccess &&
             cudaMalloc((void **)&ids, (size_t)ns * 4u) == cudaSuccess &&
             cudaMalloc((void **)&energy, (size_t)dim * 4u) == cudaSuccess;
    if (ok) {
        int32_t *h_ids = (int32_t *)malloc((size_t)ns * 4u);
        ok = h_ids != NULL;
        if (ok) {
            for (uint32_t i = 0; i < ns; i++) h_ids[i] = (int32_t)(i * step);
            ok = cudaMemcpy(ids, h_ids, (size_t)ns * 4u, cudaMemcpyHostToDevice) == cudaSuccess;
            free(h_ids);
        }
    }
    if (!ok) {
        (void)cudaGetLastError();
        fprintf(stderr, "ds4: [v41] head sketch sample allocation failed\n");
        goto fail;
    }
    v41_q4k_embed_kernel<<<dim3((nblk + 7u) / 8u, ns), dim3(32, 8), 0, ds4_current_stream()>>>(rows, ids, w, vocab, dim);
    v41_head_energy_kernel<<<(dim + 255u) / 256u, 256, 0, ds4_current_stream()>>>(energy, rows, norm, ns, dim);
    if (cudaMemcpy(h_energy, energy, (size_t)dim * 4u, cudaMemcpyDeviceToHost) != cudaSuccess) {
        (void)cudaGetLastError();
        fprintf(stderr, "ds4: [v41] head sketch leverage readback failed\n");
        goto fail;
    }
    cudaFree(rows); cudaFree(ids); cudaFree(energy);
    rows = NULL; ids = NULL; energy = NULL;

    /* 2. Column set: the top half by energy, then a stride sample of the rest
     * (the gate's construction, docs/ds41-head-gate-2026-11-09.md). */
    {
        struct col_e { float e; uint16_t c; };
        std::vector<col_e> order((size_t)dim);
        for (uint32_t c = 0; c < dim; c++) order[c] = col_e{h_energy[c], (uint16_t)c};
        std::sort(order.begin(), order.end(), [](const col_e &a, const col_e &b) { return a.e > b.e; });
        const uint32_t half = V41_HEAD_COLS / 2u;
        std::vector<uint8_t> taken((size_t)dim, 0);
        std::vector<uint16_t> cols;
        cols.reserve(V41_HEAD_COLS);
        for (uint32_t i = 0; i < half; i++) {
            taken[order[i].c] = 1;
            cols.push_back(order[i].c);
        }
        uint32_t rem = 0;
        for (uint32_t c = 0; c < dim; c++) rem += taken[c] ? 0u : 1u;
        const uint32_t stride = (rem / half) > 0u ? rem / half : 1u;
        uint32_t seen = 0;
        for (uint32_t c = 0; c < dim && cols.size() < V41_HEAD_COLS; c++) {
            if (taken[c]) continue;
            if ((seen++ % stride) == 0u) cols.push_back((uint16_t)c);
        }
        std::sort(cols.begin(), cols.end());
        if (cudaMemcpy(g_v41_hc.cols, cols.data(), cols.size() * 2u, cudaMemcpyHostToDevice) != cudaSuccess) {
            (void)cudaGetLastError();
            fprintf(stderr, "ds4: [v41] head sketch column upload failed\n");
            goto fail;
        }
    }
    free(h_energy);
    h_energy = NULL;

    /* 3. Quantize every head row over the selected columns. */
    v41_head_q8_kernel<<<vocab, 256, (size_t)dim * 4u, ds4_current_stream()>>>(g_v41_hc.q8, g_v41_hc.scale, w,
                                                                               g_v41_hc.cols, vocab, dim, V41_HEAD_COLS);
    if (cudaDeviceSynchronize() != cudaSuccess) {
        (void)cudaGetLastError();
        fprintf(stderr, "ds4: [v41] head sketch quantization failed\n");
        goto fail;
    }
    g_v41_hc.w_off = w_off;
    g_v41_hc.vocab = vocab;
    g_v41_hc.dim = dim;
    g_v41_hc.state = 1;
    fprintf(stderr, "ds4: [v41] head sketch: %u cols over %u rows (%.1f MB), candidates %u\n",
            V41_HEAD_COLS, vocab, (double)vocab * V41_HEAD_COLS / 1048576.0, V41_HEAD_ANCHOR + V41_HEAD_PICK);
    return 1;
fail:
    free(h_energy);
    if (rows) cudaFree(rows);
    if (ids) cudaFree(ids);
    if (energy) cudaFree(energy);
    g_v41_hc.state = -1;   /* refuse once; the full head serves the rest of the process */
    return 0;
}

/* Coarse-to-fine head: fills out[t][0..out_dim) for t < n_tok, non-candidates
 * at -INFINITY.  Returns 1 on success; 0 refuses and the caller falls back to
 * the full head (a capture, a skipped build, a non-q4_K head). */
extern "C" int ds4_gpu_v41_head_c2f_tensor(ds4_gpu_tensor *out, const void *model_map, uint64_t model_size,
                                           uint64_t weight_offset, uint64_t norm_offset,
                                           uint64_t in_dim, uint64_t out_dim,
                                           const ds4_gpu_tensor *x, uint32_t n_tok) {
    if (!out || !x || n_tok == 0u) return 0;
    if (n_tok > V41_HEAD_TOK) return 0;   /* the full head's own width rule (the GEMV takes <= 8 as well) */
    if (out->bytes < (uint64_t)n_tok * out_dim * 4u || x->bytes < (uint64_t)n_tok * in_dim * 4u) return 0;
    if (g_v41_hc.state == 0) {
        if (in_dim % V41_Q4K_BLK != 0u || in_dim > 8192u || out_dim <= V41_HEAD_ANCHOR || out_dim % 4u != 0u) {
            fprintf(stderr, "ds4: [v41] head: %llux%llu is outside the coarse path's shape\n",
                    (unsigned long long)out_dim, (unsigned long long)in_dim);
            g_v41_hc.state = -1;
            return 0;
        }
        cudaStreamCaptureStatus cs = cudaStreamCaptureStatusNone;
        if (cudaStreamIsCapturing(ds4_current_stream(), &cs) != cudaSuccess || cs != cudaStreamCaptureStatusNone) {
            (void)cudaGetLastError();
            fprintf(stderr, "ds4: [v41] head: a capture started before the sketch was built; using the full head\n");
            g_v41_hc.state = -1;
            return 0;
        }
        if (!v41_head_build(model_map, model_size, weight_offset, norm_offset, (uint32_t)out_dim, (uint32_t)in_dim)) return 0;
    }
    if (g_v41_hc.state < 0 || g_v41_hc.w_off != weight_offset) return 0;

    const uint32_t vocab = (uint32_t)out_dim, dim = (uint32_t)in_dim;
    const uint32_t ncand = V41_HEAD_ANCHOR + V41_HEAD_PICK;
    v41_head_xsel_kernel<<<n_tok, 256, 0, ds4_current_stream()>>>(g_v41_hc.xsel, (const float *)x->ptr, dim, g_v41_hc.cols, V41_HEAD_COLS);
    v41_head_coarse_kernel<<<(vocab + 7u) / 8u, 256, 0, ds4_current_stream()>>>(g_v41_hc.coarse, g_v41_hc.q8, g_v41_hc.scale,
                                                                                g_v41_hc.xsel, V41_HEAD_COLS, vocab, n_tok);
    v41_head_topk_kernel<<<n_tok, V41_HEAD_TOPK_THREADS, 0, ds4_current_stream()>>>(g_v41_hc.cand, g_v41_hc.coarse, vocab,
                                                                                    V41_HEAD_ANCHOR, V41_HEAD_PICK);
    const uint64_t n4 = (uint64_t)n_tok * vocab / 4u;
    v41_head_fill_kernel<<<(unsigned)((n4 + 255u) / 256u), 256, 0, ds4_current_stream()>>>((float *)out->ptr, n4);
    const uint32_t nblk = dim / V41_Q4K_BLK;
    const uint64_t pairs = (uint64_t)n_tok * ncand;
    v41_head_rescore_kernel<<<(unsigned)((pairs + 7u) / 8u), 256, 0, ds4_current_stream()>>>(
        (float *)out->ptr, vocab,
        (const uint8_t *)cuda_model_range_ptr(model_map, weight_offset, (uint64_t)vocab * nblk * V41_Q4K_BYTES, "v41 head q4_K"),
        g_v41_hc.cand, ncand, n_tok, nblk, (const float *)x->ptr, dim);
    return cuda_ok(cudaGetLastError(), "v41 head coarse-to-fine");
}
