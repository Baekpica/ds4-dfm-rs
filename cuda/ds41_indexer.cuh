/* ds41_indexer.cuh — DeepSeek V4.1 (ds41) indexer three-piece family, ported
 * from the C engine at /data/YoungAi (commit 3946dbc), file
 * `src/cuda/cuda_v41_indexer.inc.cu`:
 *   - the score kernel and its entry (:25-122, official Indexer.forward:
 *     score[i][g] = bf16(sum_h bf16(relu(bf16(q_h·k_g))·w[i][h])) with
 *     invisible groups at -inf; V41_IDX_G groups per warp);
 *   - the candidate-block selector (:132-238, official
 *     select_candidate_blocks: block score = in-block max, the block holding
 *     the query's newest position pinned +inf, topk_blocks blocks kept;
 *     radix select when kk < nb) and its [nb] scratch (:145-149);
 *   - the topk selector (:260-406, official topk over min(index_topk, visible)
 *     then position-ascending, unreachable slots -1; 1024-thread radix select
 *     + segmented write-out).
 *
 * The C2 compact candidate list is the contract (ds4_gpu_v41.h:126-152): the
 * candidate kernel writes list[i][0] = block count nc followed by nc
 * ascending block numbers; when a list is present the score kernel writes a
 * compact row [ns] with ns = v41_cand_ns(ng, bs, cap) and the topk kernel maps
 * compact index c back to group cl[1 + c/bs]·bs + c%bs.  The list is
 * ascending, so compact order == position order and the "ties take the smaller
 * index" rule holds unchanged.
 *
 * ★Named port deviations (P4-4)★:
 *   - the s8 mma score variant (cuda_v41_indexer_mma.inc.cu) is NOT ported;
 *     ds4_gpu_v41_set_indexer_mma records the flag and the score entry refuses
 *     by name while it is on (the engine keeps the switch for A/B judgment
 *     only, default off);
 *   - the engine's candidate scratch is per decode lane (DS4_GPU_MAX_LANES,
 *     concurrent captured decode graphs); the port has no lane concept yet —
 *     one static slot, named here because the day lanes land this must split.
 *
 * posd (the captured-graph device-position convention, ds4_gpu_v41.h:116-124)
 * is kept in the kernels and entries with the engine's refusals (n_tok > 8);
 * the port's eager forward passes NULL.
 */
#pragma once

/* Groups per warp in the score kernel.  The engine's archived negative result
 * (2026-10-07) kept 4: 8 groups took registers 64 -> 96 (half the blocks per
 * SM), A/B 46.68 -> 46.48 t/s within noise, bit-identical output. */
#define V41_IDX_G 4u

/* Compact-row width (cuda_v41_indexer.inc.cu:25-28): ns = min(cap·bs, ng).
 * The score kernel and the topk kernel each compute it with this one copy or
 * their row strides drift apart. */
__host__ __device__ __forceinline__ static uint32_t v41_cand_ns(uint32_t ng, uint32_t cand_bs, uint32_t cand_cap) {
    const uint64_t full = (uint64_t)cand_cap * cand_bs;
    return full < ng ? (uint32_t)full : ng;
}

/* ---- indexer score (cuda_v41_indexer.inc.cu:29-92) ----
 * One warp advances V41_IDX_G groups at once (2026-09-23): the per-(group,
 * head) chain "4 MACs -> 5-level shfl reduce -> round" is a serial dependency;
 * G chains interleave and each head's q is read once for G groups.
 * Bit-identical to one-group-per-warp: the MAC order (e ascending), the xor
 * reduction tree, the bf16 rounding point and the head accumulation order are
 * all unchanged.  grid.y covers the group dimension so a decode step is not
 * one block; the kernel wraps by grid stride when grid.y runs out (65535 cap).
 * The key is unpacked to registers once and reused across all 64 heads. */
__global__ static void v41_indexer_score_kernel(float *score, const float *q, const uint8_t *k, const float *w, const int32_t *cand,
                                                uint32_t pos0, uint32_t ng, uint32_t n_head, uint32_t dk, uint32_t ratio,
                                                const int32_t *posd, uint32_t cand_bs, uint32_t cand_cap) {
    v41_pdl_wait();   /* PDL: first instruction waits for the upstream; no-op when not launched through PDL */
    const uint32_t i = blockIdx.x, lane = threadIdx.x & 31u, warp = threadIdx.x >> 5, nwarp = blockDim.x >> 5;
    if (posd) { pos0 = (uint32_t)posd[0]; ng = (pos0 + gridDim.x) / ratio; }   /* graph: device position; host ng is the bucket cap */
    const uint32_t vis = (pos0 + i + 1u) / ratio;   /* visible groups at this absolute position */
    const uint32_t per = dk / 32u;                 /* dk=128 => 4 dims per lane */
    /* C2: with a list only the compact row [ns] is scored, item c maps to group g */
    const int32_t *cl = cand ? cand + (uint64_t)i * (1u + cand_cap) : NULL;
    const uint32_t nc = cl ? (uint32_t)cl[0] : 0u;
    const uint32_t ns = cl ? v41_cand_ns(ng, cand_bs, cand_cap) : ng;
    const uint32_t gstride = nwarp * gridDim.y * V41_IDX_G;
    for (uint32_t gb = (blockIdx.y * nwarp + warp) * V41_IDX_G; gb < ns; gb += gstride) {
        float kv[V41_IDX_G][4], acc[V41_IDX_G];
        bool live[V41_IDX_G];
        #pragma unroll
        for (uint32_t j = 0; j < V41_IDX_G; j++) {
            const uint32_t c = gb + j;
            uint32_t g = c;
            if (cl) { const uint32_t b = c / cand_bs; g = b < nc ? (uint32_t)cl[1u + b] * cand_bs + c % cand_bs : ng; }   /* off-list item = dead group */
            live[j] = c < ns && g < ng && g < vis;
            acc[j] = 0.f;
            /* Unpack the packed MXFP4 index key into registers once, reuse across all heads */
            const uint8_t *kg = k + (uint64_t)(live[j] ? g : 0u) * DS4_V41_IDXK_BYTES;
            #pragma unroll
            for (uint32_t e = 0; e < 4u; e++) kv[j][e] = (live[j] && e < per) ? v41_idxk_get(kg, lane * per + e) : 0.f;
        }
        for (uint32_t h = 0; h < n_head; h++) {
            const float *qh = q + ((uint64_t)i * n_head + h) * dk;
            float qv[4];
            #pragma unroll
            for (uint32_t e = 0; e < 4u; e++) qv[e] = e < per ? qh[lane * per + e] : 0.f;
            const float wh = w[(uint64_t)i * n_head + h];
            float d[V41_IDX_G];
            #pragma unroll
            for (uint32_t j = 0; j < V41_IDX_G; j++) {
                d[j] = 0.f;
                #pragma unroll
                for (uint32_t e = 0; e < 4u; e++) if (e < per) d[j] += qv[e] * kv[j][e];
            }
            #pragma unroll
            for (int o = 16; o > 0; o >>= 1) {
                #pragma unroll
                for (uint32_t j = 0; j < V41_IDX_G; j++) d[j] += __shfl_xor_sync(0xffffffffu, d[j], o);
            }
            #pragma unroll
            for (uint32_t j = 0; j < V41_IDX_G; j++) {
                float dd = v41_bf16r(d[j]);          /* einsum lands in bf16 */
                dd = fmaxf(dd, 0.0f);                /* relu_ */
                acc[j] += v41_bf16r(dd * wh);        /* x weights (bf16), then sum */
            }
        }
        if (lane == 0) {
            #pragma unroll
            for (uint32_t j = 0; j < V41_IDX_G; j++) if (gb + j < ns) score[(uint64_t)i * ns + gb + j] = live[j] ? v41_bf16r(acc[j]) : -INFINITY;
        }
    }
}

/* The mma score switch (cuda_v41_indexer.inc.cu:95-99).  Kept so the flag has
 * one owner; the port has no mma variant, and the score entry below refuses by
 * name while the flag is on. */
static int g_v41_idx_mma = 0;
void ds4_gpu_v41_set_indexer_mma(int on) { g_v41_idx_mma = on ? 1 : 0; }

extern "C" int ds4_gpu_v41_indexer_score_tensor(ds4_gpu_tensor *score, const ds4_gpu_tensor *q, const ds4_gpu_tensor *k,
                                                const ds4_gpu_tensor *weights, const ds4_gpu_tensor *cand_list, uint32_t cand_bs, uint32_t cand_cap,
                                                uint32_t n_tok, uint32_t pos0, uint32_t ng, uint32_t n_head, uint32_t dk, uint32_t ratio,
                                                const ds4_gpu_tensor *posd) {
    if (!score || !q || !k || !weights || (dk % 32u) || ratio == 0) return 0;
    if (dk / 32u > 4u) { fprintf(stderr, "ds4: [ds41] indexer score supports dk <= 128 (<= 4 dims per lane)\n"); return 0; }
    if (posd && n_tok > 8u) return 0;   /* graph path: pure decode 1 row or a verify batch <= 8 rows */
    if (ng == 0) return 1;
    if (cand_list && (cand_bs == 0u || cand_cap == 0u)) return 0;
    if (g_v41_idx_mma) {
        fprintf(stderr, "ds4: [ds41] the indexer s8 mma score variant is not ported (scalar kernel only)\n");
        return 0;
    }
    /* One block = 8 warps x V41_IDX_G groups; more groups = more blocks
     * (65535 is CUDA's grid.y hard cap; the kernel wraps by grid stride).
     * C2: with a list the grid is sized by the compact width. */
    const uint32_t ns = cand_list ? v41_cand_ns(ng, cand_bs, cand_cap) : ng;
    uint32_t gblocks = (ns + 8u * V41_IDX_G - 1u) / (8u * V41_IDX_G);
    if (gblocks > 65535u) gblocks = 65535u;
    v41_indexer_score_kernel<<<dim3(n_tok, gblocks), 256, 0, ds4_current_stream()>>>((float *)score->ptr, (const float *)q->ptr, (const uint8_t *)k->ptr,
        (const float *)weights->ptr, cand_list ? (const int32_t *)cand_list->ptr : NULL, pos0, ng, n_head, dk, ratio,
        posd ? (const int32_t *)posd->ptr : NULL, cand_bs, cand_cap);
    return cuda_ok(cudaGetLastError(), "v41 indexer score");
}

/* ---- candidate blocks (cuda_v41_indexer.inc.cu:132-238) ----
 * The engine's original was one serial O(kk x nb) thread-0 loop: at 12k
 * context (nb = 1558, kk = 2048 > nb) that was 2.4M rounds = 46.4 ms/step.
 * The rewrite keeps the selection set bit-identical:
 *   (1) kk >= nb (the live config up to ~32k): the original selects every
 *       block with score > -inf — written directly, no selection needed;
 *   (2) kk < nb: parallel radix select for the kk-th key; keys > threshold
 *       all selected, keys == threshold filled in ascending block order
 *       (the original scanned with strict > so ties take the smaller block).
 * The [nb] block scores and [nb] flags live in global scratch (not shared):
 * the old static shared 48 KB capped nb at 9830 blocks, which was the wall
 * that pinned ctx to 32768; the row stride is the host's bucket cap nb_cap so
 * two rows can never overlap when the kernel's own nb is smaller.
 * float -> order-preserving u32 key, shared with topk (:137-141):
 * -inf and NaN -> 0 (both originals scan with `x > bv`, bv = -inf, so those
 * never win; key 0 reproduces that).  Positives set the top bit; negatives are
 * complemented — IEEE float order becomes unsigned key order. */
__device__ __forceinline__ static uint32_t v41_topk_key(float f) {
    if (!(f > -INFINITY)) return 0u;
    uint32_t u; memcpy(&u, &f, 4);
    return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}

/* Grow-only scratch for the candidate kernel, 5 bytes per block per row
 * (float score + u8 flag).  Named deviation: the engine splits this per
 * decode lane; the port has no lanes yet. */
static v41_scratch g_v41_cand_blk;

extern "C" int ds4_gpu_v41_candidate_scratch_prepare(uint32_t n_tok, uint32_t nb) {
    if (n_tok == 0u || nb == 0u) return 1;
    return v41_grow(&g_v41_cand_blk, (uint64_t)n_tok * nb * 5u, "v41 candidate blocks") ? 1 : 0;
}

__global__ static void v41_candidate_kernel(int32_t *list, const float *score, uint32_t pos0, uint32_t ng, uint32_t ratio,
                                            uint32_t topk_blocks, uint32_t bs, const int32_t *posd,
                                            float *blk, uint32_t nb_cap) {
    v41_pdl_wait();   /* PDL: first instruction waits for the upstream; no-op when not launched through PDL */
    const uint32_t i = blockIdx.x;
    if (posd) { pos0 = (uint32_t)posd[0]; ng = (pos0 + gridDim.x) / ratio; }   /* graph: device position; scratch sized by the bucket cap */
    const uint32_t nb = (ng + bs - 1u) / bs;
    float *bsc = blk + (uint64_t)i * nb_cap;
    uint8_t *sel = (uint8_t *)(blk + (uint64_t)gridDim.x * nb_cap) + (uint64_t)i * nb_cap;
    __shared__ uint32_t hist[256], sh_bucket, sh_k;
    const uint32_t vis = (pos0 + i + 1u) / ratio;
    for (uint32_t b = threadIdx.x; b < nb; b += blockDim.x) {
        float m = -INFINITY;
        for (uint32_t j = b * bs; j < (b + 1u) * bs && j < ng; j++) m = fmaxf(m, score[(uint64_t)i * ng + j]);
        if (vis > 0u && b == (vis - 1u) / bs) m = INFINITY;   /* pin the block holding the newest position */
        bsc[b] = m; sel[b] = 0;
    }
    __syncthreads();
    const uint32_t kk = topk_blocks < nb ? topk_blocks : nb;
    if (kk >= nb) {                               /* (1) select all except fully invisible blocks */
        for (uint32_t b = threadIdx.x; b < nb; b += blockDim.x) sel[b] = v41_topk_key(bsc[b]) != 0u;
    } else {                                      /* (2) parallel radix select for the threshold */
        uint32_t prefix = 0, want = kk;
        for (int shift = 24; shift >= 0; shift -= 8) {
            for (uint32_t t = threadIdx.x; t < 256u; t += blockDim.x) hist[t] = 0;
            __syncthreads();
            for (uint32_t b = threadIdx.x; b < nb; b += blockDim.x) {
                const uint32_t key = v41_topk_key(bsc[b]);
                if (key == 0u || (shift < 24 && (key >> (shift + 8)) != (prefix >> (shift + 8)))) continue;
                atomicAdd(&hist[(key >> shift) & 0xffu], 1u);
            }
            __syncthreads();
            if (threadIdx.x == 0) {               /* accumulate from the top bucket down to the one holding the want-th key */
                uint32_t acc = 0; int t = 255;
                for (; t > 0; t--) { if (acc + hist[t] >= want) break; acc += hist[t]; }
                sh_bucket = (uint32_t)t; sh_k = want - acc;
            }
            __syncthreads();
            prefix |= sh_bucket << shift;
            want = sh_k;
            __syncthreads();
        }
        for (uint32_t b = threadIdx.x; b < nb; b += blockDim.x) sel[b] = v41_topk_key(bsc[b]) > prefix;
        __syncthreads();
        if (threadIdx.x == 0) {                   /* keys == threshold fill in ascending block order (ties take the smaller block) */
            uint32_t eq = want;
            for (uint32_t b = 0; b < nb && eq; b++) if (v41_topk_key(bsc[b]) == prefix) { sel[b] = 1; eq--; }
        }
    }
    __syncthreads();
    /* Compact the selection into an ascending list (2026-09-30 C2, contract in
     * ds4_gpu_v41.h): list[i][0] = block count, list[i][1..] = ascending block
     * numbers.  The old form wrote an [n][nb] byte mask (268 MB for a 1M
     * context prefill) and downstream still ran the head loop over the masked
     * million groups.  Each thread owns one contiguous block span (ascending
     * within the span); span counts -> exclusive prefix -> spans written in
     * order, so the list is ascending exactly like the original mask scan. */
    __shared__ uint32_t cnt[256];   /* same origin as the launched blockDim(256) */
    const uint32_t chunk = (nb + blockDim.x - 1u) / blockDim.x;
    const uint32_t c0 = threadIdx.x * chunk, c1 = (c0 + chunk) < nb ? (c0 + chunk) : nb;
    uint32_t mine = 0;
    for (uint32_t b = c0; b < c1; b++) mine += sel[b];
    cnt[threadIdx.x] = mine;
    __syncthreads();
    if (threadIdx.x == 0) {
        uint32_t run = 0;
        for (uint32_t t = 0; t < blockDim.x; t++) { const uint32_t v = cnt[t]; cnt[t] = run; run += v; }
        list[(uint64_t)i * (1u + topk_blocks)] = (int32_t)run;   /* <= kk <= topk_blocks: the list always fits */
    }
    __syncthreads();
    uint32_t wr = cnt[threadIdx.x];
    for (uint32_t b = c0; b < c1; b++) if (sel[b]) list[(uint64_t)i * (1u + topk_blocks) + 1u + wr++] = (int32_t)b;
}

extern "C" int ds4_gpu_v41_candidate_blocks_tensor(ds4_gpu_tensor *cand_list, const ds4_gpu_tensor *score, uint32_t n_tok, uint32_t pos0,
                                                   uint32_t ng, uint32_t ratio, uint32_t topk_blocks, uint32_t block_size,
                                                   const ds4_gpu_tensor *posd) {
    if (!cand_list || !score || block_size == 0 || ratio == 0 || topk_blocks == 0) return 0;
    if (posd && n_tok > 8u) return 0;
    if (ng == 0) return 1;
    const uint32_t nb = (ng + block_size - 1u) / block_size;
    /* Eager path grows the scratch here; on the graph path the prepare has
     * already grown it to the bucket cap and this call is a no-op. */
    float *blk = (float *)v41_grow(&g_v41_cand_blk, (uint64_t)n_tok * nb * 5u, "v41 candidate blocks");
    if (!blk) return 0;
    v41_candidate_kernel<<<n_tok, 256, 0, ds4_current_stream()>>>((int32_t *)cand_list->ptr, (const float *)score->ptr, pos0, ng, ratio,
                                                                   topk_blocks, block_size, posd ? (const int32_t *)posd->ptr : NULL,
                                                                   blk, nb);
    return cuda_ok(cudaGetLastError(), "v41 candidate blocks");
}

/* ---- topk (cuda_v41_indexer.inc.cu:260-406) ----
 * Official semantics: topk over min(index_topk, visible groups), output in
 * position order, unreachable slots -1.  The engine's original was a serial
 * O(topk·ng) thread-0 loop (44 ms per call at 3300 context, 62% of the GPU
 * time in long-context decode).  The rewrite is bit-equivalent to it:
 *   (1) parallel radix select finds the kk-th largest key (float mapped to an
 *       order-preserving u32, 8 bits per round, 4 rounds; 1024 threads, reads
 *       batched V41_TOPK_B at a time);
 *   (2) thread-0's ascending scan is replaced by segmented write-out: position
 *       = (count of > threshold before g) + min(count of == threshold before
 *       g, eq), segment counts + two exclusive prefix sums, segments written
 *       in order — the set, the order and the tie rule are all unchanged.
 * Equivalence points preserved verbatim: `s[g] > bv` (strict) so ties take
 * the earlier (smaller) index; -inf and NaN (key 0) are never selected.
 * With a candidate list the scan runs over the compact row [ns] only and
 * writes map back through v41_cand_map; the topk cap still uses ng (official
 * min(index_topk, ng)). */
#define V41_TOPK_THREADS 1024u
#define V41_TOPK_B 8u
/* C2: compact item c -> real group number (ascending list => monotone); no list => c is the group */
__device__ __forceinline__ static int32_t v41_cand_map(const int32_t *cl, uint32_t c, uint32_t bs) {
    return cl ? cl[1u + c / bs] * (int32_t)bs + (int32_t)(c % bs) : (int32_t)c;
}
__global__ static void __launch_bounds__(V41_TOPK_THREADS) v41_topk_kernel(int32_t *idx, const float *score, uint32_t ng, uint32_t topk,
                                                                           uint32_t ratio, const int32_t *posd,
                                                                           const int32_t *cand, uint32_t cand_bs, uint32_t cand_cap) {
    v41_pdl_wait();   /* PDL: first instruction waits for the upstream; no-op when not launched through PDL */
    const uint32_t i = blockIdx.x, tid = threadIdx.x, nt = blockDim.x;
    if (posd) {   /* graph: ng and topk recomputed from the device position (same formula as the host's min(index_topk, ng)) */
        ng = ((uint32_t)posd[0] + gridDim.x) / ratio;
        if (ng < topk) topk = ng;
    }
    /* C2: with a list score is the compact row [ns] the score kernel wrote; the scan covers ns items and writes map back; the topk cap still uses ng */
    const int32_t *cl = cand ? cand + (uint64_t)i * (1u + cand_cap) : NULL;
    const uint32_t ns = cl ? v41_cand_ns(ng, cand_bs, cand_cap) : ng;
    const float *s = score + (uint64_t)i * ns;
    /* Archived negative result (2026-09-16): per-warp histograms (hist[8][256])
     * to remove shared-atomic contention measured 5.91 -> 7.15 ms (21%
     * slower); contention is not this kernel's bottleneck. */
    __shared__ uint32_t hist[256], suf[257], sh_bucket;
    uint32_t prefix = 0, kk = 0;
    for (int shift = 24; shift >= 0; shift -= 8) {
        for (uint32_t b = tid; b < 256u; b += nt) hist[b] = 0;
        __syncthreads();
        for (uint32_t gb = tid; gb < ns; gb += V41_TOPK_B * nt) {
            float v[V41_TOPK_B];
            #pragma unroll
            for (uint32_t j = 0; j < V41_TOPK_B; j++) { const uint32_t g = gb + j * nt; v[j] = g < ns ? s[g] : -INFINITY; }
            #pragma unroll
            for (uint32_t j = 0; j < V41_TOPK_B; j++) {
                const uint32_t key = v41_topk_key(v[j]);   /* out-of-range -inf => key 0, same as "not counted" */
                /* shift=24 has no higher bits to compare (a 32-bit right shift is UB); everything counts */
                if (key == 0u || (shift < 24 && (key >> (shift + 8)) != (prefix >> (shift + 8)))) continue;
                atomicAdd(&hist[(key >> shift) & 0xffu], 1u);
            }
        }
        __syncthreads();
        if (tid < 32u) {   /* suffix sums suf[b] = sum_{b' >= b} hist[b']: lane l owns buckets [8l, 8l+8), shfl for "lanes above me" */
            uint32_t loc = 0;
            for (uint32_t b = 0; b < 8u; b++) loc += hist[tid * 8u + b];
            uint32_t inc = loc;
            for (int o = 1; o < 32; o <<= 1) { const uint32_t t = __shfl_down_sync(0xffffffffu, inc, o); if (tid + (uint32_t)o < 32u) inc += t; }
            uint32_t run = inc - loc;   /* buckets above this lane */
            for (int b = 7; b >= 0; b--) { run += hist[tid * 8u + (uint32_t)b]; suf[tid * 8u + (uint32_t)b] = run; }
            if (tid == 0u) { suf[256] = 0u; sh_bucket = 0u; }
        }
        __syncthreads();
        if (shift == 24) {   /* (3) round 1's histogram covers every valid key => suf[0] = nvalid */
            const uint32_t nval = suf[0];
            kk = topk < nval ? topk : nval;
            if (kk == 0u) break;   /* original: kk == 0 never enters radix, prefix stays 0 (block-uniform branch) */
        }
        for (uint32_t b = 1u + tid; b < 256u; b += nt) if (suf[b] >= kk) atomicMax(&sh_bucket, b);
        __syncthreads();
        const uint32_t bs = sh_bucket;
        prefix |= bs << shift;
        kk -= suf[bs + 1u];   /* how many more to take inside this bucket */
        __syncthreads();      /* everyone must finish reading hist/suf before the next round clears them */
    }
    /* Write-out segments (parallel since 2026-09-16): position = (count of >
     * threshold before g) + min(count of == threshold before g, eq).  ng is
     * cut into nt spans, each counts (gt, eq), two prefix sums give the span
     * starts, spans written ascending.  Set, order and tie rule identical to
     * the original. */
    __shared__ uint32_t cgt[V41_TOPK_THREADS], ceq[V41_TOPK_THREADS], sh_tgt, sh_teq;
    if (tid == 0) { sh_tgt = 0; sh_teq = 0; }
    __syncthreads();
    const uint32_t chunk = (ns + nt - 1u) / nt;
    const uint32_t g0 = tid * chunk, g1 = (g0 + chunk) < ns ? (g0 + chunk) : ns;
    {   /* (1) each span counts its own > threshold / == threshold (reads batched) */
        uint32_t ngt = 0, neq = 0;
        for (uint32_t gb = g0; gb < g1; gb += V41_TOPK_B) {
            float v[V41_TOPK_B];
            #pragma unroll
            for (uint32_t j = 0; j < V41_TOPK_B; j++) v[j] = gb + j < g1 ? s[gb + j] : -INFINITY;
            #pragma unroll
            for (uint32_t j = 0; j < V41_TOPK_B; j++) {
                const uint32_t key = v41_topk_key(v[j]);
                if (key == 0u) continue;
                if (key > prefix) ngt++;
                else if (key == prefix) neq++;
            }
        }
        cgt[tid] = ngt; ceq[tid] = neq;
        atomicAdd(&sh_tgt, ngt); atomicAdd(&sh_teq, neq);   /* totals: (4) uses them to fill the tail */
    }
    __syncthreads();
    /* (2) two exclusive prefix sums (nt entries, Hillis-Steele in place: shift
     * right one, then add level by level; every level needs its __syncthreads,
     * all outside branches) */
    {
        uint32_t vg = tid ? cgt[tid - 1u] : 0u;
        uint32_t ve = tid ? ceq[tid - 1u] : 0u;
        __syncthreads();
        cgt[tid] = vg; ceq[tid] = ve;
        __syncthreads();
        for (uint32_t off = 1u; off < nt; off <<= 1) {
            const uint32_t ag = tid >= off ? cgt[tid - off] : 0u;
            const uint32_t ae = tid >= off ? ceq[tid - off] : 0u;
            __syncthreads();
            cgt[tid] += ag; ceq[tid] += ae;
            __syncthreads();
        }
    }
    {   /* (3) spans written ascending; eq = "how many == threshold still to take" from the radix tail */
        const uint32_t eq = kk;
        uint32_t wgt = cgt[tid], weq = ceq[tid];
        for (uint32_t gb = g0; gb < g1; gb += V41_TOPK_B) {
            float v[V41_TOPK_B];
            #pragma unroll
            for (uint32_t j = 0; j < V41_TOPK_B; j++) v[j] = gb + j < g1 ? s[gb + j] : -INFINITY;
            #pragma unroll
            for (uint32_t j = 0; j < V41_TOPK_B; j++) {
                const uint32_t g = gb + j;
                const uint32_t key = v41_topk_key(v[j]);
                if (key == 0u) continue;
                if (key > prefix) {
                    const uint32_t pos = wgt + (weq < eq ? weq : eq);
                    if (pos < topk) idx[(uint64_t)i * topk + pos] = v41_cand_map(cl, g, cand_bs);
                    wgt++;
                } else if (key == prefix) {
                    if (weq < eq) {
                        const uint32_t pos = wgt + weq;
                        if (pos < topk) idx[(uint64_t)i * topk + pos] = v41_cand_map(cl, g, cand_bs);
                    }
                    weq++;
                }
            }
        }
    }
    __syncthreads();
    {   /* (4) fill the tail with -1 (unreachable).  Total = all > threshold + min(all == threshold, eq), same as the original's w carry. */
        uint32_t total = sh_tgt + (sh_teq < kk ? sh_teq : kk);
        if (total > topk) total = topk;
        for (uint32_t w = total + tid; w < topk; w += nt) idx[(uint64_t)i * topk + w] = -1;
    }
}

extern "C" int ds4_gpu_v41_indexer_topk_tensor(ds4_gpu_tensor *idx, const ds4_gpu_tensor *score, uint32_t n_tok, uint32_t ng,
                                               uint32_t topk, uint32_t ratio, const ds4_gpu_tensor *posd,
                                               const ds4_gpu_tensor *cand_list, uint32_t cand_bs, uint32_t cand_cap) {
    if (!idx || !score || topk == 0) return 0;
    if (posd && (n_tok > 8u || ratio == 0u)) return 0;
    if (cand_list && (cand_bs == 0u || cand_cap == 0u)) return 0;
    if (ng == 0) return 1;
    /* shared is down to the fixed 256 buckets (1 KB), so the old "ng > 48K refuses" gate is gone */
    v41_topk_kernel<<<n_tok, V41_TOPK_THREADS, 0, ds4_current_stream()>>>((int32_t *)idx->ptr, (const float *)score->ptr, ng, topk, ratio,
                                                     posd ? (const int32_t *)posd->ptr : NULL,
                                                     cand_list ? (const int32_t *)cand_list->ptr : NULL, cand_bs, cand_cap);
    return cuda_ok(cudaGetLastError(), "v41 topk");
}
