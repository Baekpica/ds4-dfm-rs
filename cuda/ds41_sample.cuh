/* cuda/ds41_sample.cuh — the V4.1 device sampling kernel, ported from the C
 * engine at /data/YoungAi (commit 3946dbc): src/cuda/cuda_v41_sample.inc.cu
 * (2026-09-28).  Temperature / top-k / top-p / min-p all run on the GPU, one
 * block per row, each row's result lands in a 16-byte device slot (the SAME
 * slot the argmax kernel uses, zero-copy readback); the speculative rejection
 * sampling comes out of the same launch.
 *
 * Why it exists (engine header comment): after 2026-09-28 made sampling the
 * default (temperature 1.0, the model-card recipe), a request without a
 * temperature took the sampling path — which used to read all 129280 logits
 * back to the host per step, run two expf passes there, and turn speculation
 * off.  Moving the pick into the decode-step graph's tail (replacing the
 * argmax launch) leaves the host reading back 16 bytes; speculation stays on.
 *
 * The draw is Gumbel-max: argmax_i (l_i/T + g_i), g_i = -ln(-ln u_i), u_i iid
 * uniform, so i is picked with probability exactly softmax(l/T)_i (an exact
 * equivalence, not an approximation); restricting to the kept set K is the
 * truncated-then-renormalized distribution.  u_i is hashed from (seed,
 * position, i, salt) with splitmix64's final mix — no dependence on thread
 * scheduling, so same seed + same input always gives the same result, and the
 * graph and direct routes (same kernel) emit the same token stream.
 * Kept set K = {l_i >= L_cut}, L_cut = max(min_p gate, top_k gate, top_p
 * gate): min_p: l >= M + T*ln(min_p) (relative to the max probability);
 * top_k: the k-th largest logit (radix select over 4 x 8-bit histogram
 * passes, integer counts -> deterministic); top_p: the largest-to-smallest
 * cumulative probability first reaching top_p (same radix select, masses
 * accumulated as 2^40 fixed-point integers -> deterministic, floating-point
 * atomic-add order cannot move the threshold).  Semantics match the host
 * ds4_sample_logits: top_k first, top_p relative to the post-cut total mass,
 * min_p relative to the max probability; ties at a boundary are all kept
 * (the host's qsort cuts ties in an arbitrary order of its own).
 * Speculation (rejection sampling, Leviathan 2023 / Chen 2023): row i's draft
 * d = tok[i+1] (the next row's input in the verify batch).  Point-mass draft
 * (qlogits == NULL): accept probability = p(d), reject draws from the
 * "p minus d, renormalized" residual; distribution draft (qlogits = the
 * draft tower's row logits, 2026-09-29): accept = min(1, p(d)/q(d)), reject
 * draws from max(0, p - q) renormalized — both emit tokens whose marginal
 * distribution is exactly p; the latter's accept-rate ceiling is sum min(p,q),
 * far above the point mass's p(argmax q).  The accept coin and the Gumbel
 * noise use different salts, and the draft draw (stream 1) and the verify
 * (stream 0) use different streams, so everything is independent.  The
 * residual is empty (p == q, or K subset of {d}) -> accept outright.  The
 * last row has no draft and only emits its full-distribution sample.
 * Error behavior: an all-non-finite row writes four zero ints (same fallback
 * as the host sample_argmax: index 0); temperature <= 0 must not enter this
 * kernel (the caller runs the argmax kernel — that is the greedy gate). */
#ifndef DS41_SAMPLE_CUH
#define DS41_SAMPLE_CUH

#define V41_SAMPLE_THREADS 1024u
#define V41_SAMPLE_BINS 256u
/* 2^40: fixed-point resolution relative to the max probability.  Every vocab
 * entry <= 1, so the total stays < 2^58 and cannot overflow u64; entries
 * below 2^-40 relative probability count as 0 mass — they cannot move a
 * top_p threshold anyway (host float accumulation absorbs no such terms
 * either). */
#define V41_SAMPLE_FIX 1099511627776.0

__device__ __forceinline__ static uint64_t v41_mix64(uint64_t x) {
    x ^= x >> 30; x *= 0xBF58476D1CE4E5B9ull; x ^= x >> 27; x *= 0x94D049BB133111EBull; x ^= x >> 31; return x;
}
/* A full-precision uniform on (0,1): the binary exponent comes from the
 * leading-zero count of the remaining 41 bits (geometric, P(n) = 2^-(n+1)),
 * the mantissa takes the top 23 bits -> value = (1 + m*2^-23) * 2^-(n+1),
 * 24-bit resolution in every binary bucket, finer toward 0 (down to 2^-42) —
 * matching float's own resolution structure; upper bound 1-2^-24, never 0 or 1.
 * Why this fuss (the engine's three crashes, judge = tests/cuda_sample_selftest.c
 * counting low-p tokens):
 *   1. 24-bit integer + 0.5 times 2^-24: at k >= 2^23, k+0.5 exceeds fp32's
 *      24 significant bits, rounds to even, k = 2^24-1 rounds to 2^24 => u =
 *      1.0f => -ln(-ln 1) = +inf => that token wins the argmax unconditionally.
 *      With top_p=1 the kept set is the whole vocab; each slot hit with
 *      probability 1-(1-2^-24)^129280 = 0.77%/stream — a uniformly random
 *      word every ~130 tokens.  In a 200k-token report, 1,880 (846 verify +
 *      1,034 draft); the body degraded into multilingual noise and the model
 *      could not close.  Gate read 1,720 vs 558 expected.
 *   2. 23 bits + half-cell (exact, below 1) still red at 1,363: fast-math's
 *      __logf has absolute error 3.6e-7 near 1, which scrambles the |ln u|
 *      ~1e-7 upper tail — fixed by log1pf (see the kernel).
 *   3. After that the gate read 347, too few: the uniform near 0 only had
 *      2^-23 absolute resolution, flattening the Gumbel upper tail, and
 *      probability-1e-7 tokens were sampled ~40% short.  This version solves
 *      3: the uniform is bucketed the way float is, upper tail fine to 2^-42. */
__device__ __forceinline__ static float v41_u01(uint64_t seed, int32_t pos, uint32_t i, uint32_t salt) {
    const uint64_t x = v41_mix64(seed ^ (0x9E3779B97F4A7C15ull * (uint64_t)(uint32_t)pos + 0xD1B54A32D192ED03ull * (uint64_t)i +
                                          0x8CB92BA72F3D8DD7ull * (uint64_t)(salt + 1u)));
    const uint32_t m = (uint32_t)(x >> 41);                       /* 23-bit mantissa */
    const uint64_t r = x << 23;                                   /* the remaining 41 bits moved high, count leading zeros */
    const int n = r ? __clzll((long long)r) : 41;                 /* all-zero caps at 41 => smallest bucket 2^-42, still a normal float */
    return __uint_as_float(((uint32_t)(126 - n) << 23) | m);     /* exponent field 127-(n+1), mantissa field m */
}
/* float <-> monotonic u32 key (ascending key = ascending float), finite values only */
__device__ __forceinline__ static uint32_t v41_fkey(float f) { const uint32_t u = __float_as_uint(f); return (u & 0x80000000u) ? ~u : (u | 0x80000000u); }

/* Block argmax (ties take the lower index; index -1 = empty, always loses) */
__device__ static void v41_blk_argmax(float *sf, int32_t *si, float v, int32_t i, float *ov, int32_t *oi) {
    sf[threadIdx.x] = v; si[threadIdx.x] = i; __syncthreads();
    for (uint32_t k = blockDim.x / 2u; k > 0u; k >>= 1) {
        if (threadIdx.x < k) {
            const float bv = sf[threadIdx.x + k]; const int32_t bi = si[threadIdx.x + k];
            const float av = sf[threadIdx.x]; const int32_t ai = si[threadIdx.x];
            if (bi >= 0 && (ai < 0 || bv > av || (bv == av && bi < ai))) { sf[threadIdx.x] = bv; si[threadIdx.x] = bi; }
        }
        __syncthreads();
    }
    *ov = sf[0]; *oi = si[0]; __syncthreads();
}
/* Block float sum: fixed tree order => deterministic (no atomic adds) */
__device__ static float v41_blk_sumf(float *sf, float v) {
    sf[threadIdx.x] = v; __syncthreads();
    for (uint32_t k = blockDim.x / 2u; k > 0u; k >>= 1) { if (threadIdx.x < k) sf[threadIdx.x] += sf[threadIdx.x + k]; __syncthreads(); }
    const float s = sf[0]; __syncthreads(); return s;
}

/* Radix select: over {finite and key >= floor_key}, accumulate weight from the
 * largest key down, return the key of the first item reaching target.
 * mass=0: weight = 1 (top_k, target = k); mass=1: weight = fixed-point
 * probability (top_p, target = ceil(top_p*Z)).  Four passes x 256 buckets
 * (shared integer atomic adds, order-independent) -> scan from the high
 * bucket down to the one holding the threshold -> next pass looks only inside
 * that bucket.  Returns the threshold item's 32-bit key (everything >= it is
 * kept, ties together).  When target exceeds the total (only top_k > finite
 * count) it returns 0 = keep everything. */
__device__ static uint32_t v41_radix_select(const float *l, uint32_t V, uint32_t floor_key, int mass, float M, float inv_T,
                                            unsigned long long target, unsigned long long *hist, uint32_t *s_prefix, unsigned long long *s_target) {
    uint32_t prefix = 0u;
    for (int shift = 24; shift >= 0; shift -= 8) {
        const uint32_t mask = shift == 24 ? 0u : (0xFFFFFFFFu << (shift + 8));
        for (uint32_t b = threadIdx.x; b < V41_SAMPLE_BINS; b += blockDim.x) hist[b] = 0ull;
        __syncthreads();
        for (uint32_t i = threadIdx.x; i < V; i += blockDim.x) {
            const float v = l[i];
            if (!isfinite(v)) continue;
            const uint32_t k = v41_fkey(v);
            if (k < floor_key || (k & mask) != prefix) continue;
            unsigned long long w = 1ull;
            if (mass) w = (unsigned long long)((double)expf((v - M) * inv_T) * V41_SAMPLE_FIX + 0.5);
            atomicAdd(&hist[(k >> shift) & 0xFFu], w);
        }
        __syncthreads();
        if (threadIdx.x == 0) {
            unsigned long long cum = 0ull; uint32_t sel = 0u; unsigned long long rest = target;
            for (int b = (int)V41_SAMPLE_BINS - 1; b >= 0; b--) {
                cum += hist[b];
                if (cum >= target) { sel = (uint32_t)b; rest = target - (cum - hist[b]); break; }
            }
            *s_prefix = prefix | (sel << shift); *s_target = rest;
        }
        __syncthreads();
        prefix = *s_prefix; target = *s_target;
        __syncthreads();
        if (target == 0ull) return 0u;   /* scanned every bucket without reaching target => keep everything; otherwise the remainder is >= 1 */
    }
    return prefix;
}

/* One row's three gates: returns the key floor cut (0 = keep everything),
 * *M = the finite maximum (*Mi < 0 = the whole row is non-finite, caller
 * falls back). */
__device__ static uint32_t v41_sample_gate(const float *l, uint32_t V, float inv_T, float min_p, uint32_t top_k, float top_p,
                                           float *sf, int32_t *si, unsigned long long *hist, uint32_t *s_prefix, unsigned long long *s_u64,
                                           float *M, int32_t *Mi) {
    float best = -INFINITY; int32_t bi = -1;
    for (uint32_t i = threadIdx.x; i < V; i += blockDim.x) { const float v = l[i]; if (isfinite(v) && v > best) { best = v; bi = (int32_t)i; } }
    v41_blk_argmax(sf, si, best, bi, M, Mi);
    if (*Mi < 0) return 0u;
    uint32_t cut = 0u;
    if (min_p > 0.0f) cut = v41_fkey(*M + logf(min_p) / inv_T);
    uint32_t kfloor = 0u;
    if (top_k > 0u && top_k < V) {
        kfloor = v41_radix_select(l, V, 0u, 0, *M, inv_T, (unsigned long long)top_k, hist, s_prefix, s_u64);
        if (kfloor > cut) cut = kfloor;
    }
    if (top_p < 1.0f) {
        if (threadIdx.x == 0) *s_u64 = 0ull;
        __syncthreads();
        unsigned long long z = 0ull;
        for (uint32_t i = threadIdx.x; i < V; i += blockDim.x) {
            const float v = l[i];
            if (!isfinite(v) || v41_fkey(v) < kfloor) continue;
            z += (unsigned long long)((double)expf((v - *M) * inv_T) * V41_SAMPLE_FIX + 0.5);
        }
        atomicAdd(s_u64, z);
        __syncthreads();
        const unsigned long long Z = *s_u64;
        __syncthreads();
        unsigned long long target = (unsigned long long)ceil((double)top_p * (double)Z);
        if (target == 0ull) target = 1ull;
        const uint32_t pfloor = v41_radix_select(l, V, kfloor, 1, *M, inv_T, target, hist, s_prefix, s_u64);
        if (pfloor > cut) cut = pfloor;
    }
    return cut;
}

__global__ static void v41_sample_kernel(int32_t *out, const float *logits, const float *qlogits, uint32_t V, const int32_t *pos, const int32_t *tok,
                                         uint32_t row0, uint32_t n_rows, float inv_T, float min_p, uint32_t top_k, float top_p,
                                         uint64_t seed, uint32_t stream) {
    v41_pdl_wait();   /* upstream = the head writing logits; returns immediately when not launched with PDL */
    __shared__ float sf[V41_SAMPLE_THREADS]; __shared__ int32_t si[V41_SAMPLE_THREADS];
    __shared__ unsigned long long hist[V41_SAMPLE_BINS];
    __shared__ uint32_t s_prefix; __shared__ unsigned long long s_u64;
    const uint32_t r = row0 + blockIdx.x;
    const float *l = logits + (size_t)r * V;
    const int32_t p = pos[r];
    const int32_t d = blockIdx.x + 1u < n_rows ? tok[r + 1u] : -1;
    const float *lq = (d >= 0 && qlogits) ? qlogits + (size_t)blockIdx.x * V : NULL;
    const uint32_t salt_g = 2u * stream, salt_a = 2u * stream + 1u;   /* Gumbel noise / accept coin, split by stream */
    int32_t *o = out + 4u * blockIdx.x;
    float M; int32_t Mi;
    const uint32_t cut = v41_sample_gate(l, V, inv_T, min_p, top_k, top_p, sf, si, hist, &s_prefix, &s_u64, &M, &Mi);
    if (Mi < 0) { if (threadIdx.x == 0) { o[0] = 0; o[1] = 0; o[2] = 0; o[3] = 0; } return; }
    /* The draft distribution q's gates and both normalization constants (a
     * point-mass draft needs none: the residual is p minus d, the ratio needs
     * no normalization). */
    float Mq = 0.0f; int32_t Mqi = -1; uint32_t cutq = 0u; float Zp = 1.0f, Zq = 1.0f;
    if (lq) {
        cutq = v41_sample_gate(lq, V, inv_T, min_p, top_k, top_p, sf, si, hist, &s_prefix, &s_u64, &Mq, &Mqi);
        if (Mqi < 0) lq = NULL;   /* the q row is all non-finite: treat it as a point-mass draft */
    }
    if (lq) {
        float zp = 0.0f, zq = 0.0f;
        for (uint32_t i = threadIdx.x; i < V; i += blockDim.x) {
            const float v = l[i], vq = lq[i];
            if (isfinite(v) && v41_fkey(v) >= cut) zp += expf((v - M) * inv_T);
            if (isfinite(vq) && v41_fkey(vq) >= cutq) zq += expf((vq - Mq) * inv_T);
        }
        Zp = v41_blk_sumf(sf, zp); Zq = v41_blk_sumf(sf, zq);
    }
    /* Gumbel-max over the kept set: the full-distribution sample; the residual
     * sample (q: over max(0, p-q) / point mass: p minus d); and along the way
     * Z_K, and the numerators of p(d) and q(d). */
    float gb = -INFINITY, gxb = -INFINITY, zk = 0.0f, md = 0.0f, mqd = 0.0f; int32_t gi = -1, gxi = -1;
    for (uint32_t i = threadIdx.x; i < V; i += blockDim.x) {
        const float v = l[i];
        /* q(d) must be computed even outside p's kept set (the 09-29 selftest
         * conviction: the draft fell outside K_p, this used to skip it, q(d)
         * stayed 0 and was read as "a draft that cannot be drawn" and
         * accepted, emitting a token outside the kept set) — a draft outside
         * K_p means p(d) = 0 => reject. */
        if (lq && (int32_t)i == d) { const float vq = lq[i]; mqd = (isfinite(vq) && v41_fkey(vq) >= cutq) ? expf((vq - Mq) * inv_T) : 0.0f; }
        if (!isfinite(v) || v41_fkey(v) < cut) continue;
        const float m = expf((v - M) * inv_T);
        zk += m;
        if ((int32_t)i == d) md = m;
        /* Gumbel g = -ln E, E = -ln(1-u) ~ Exp(1) (u uniform => 1-u uniform).
         * Why log1pf(-u) and not logf(u) (the 09-29 distribution-gate
         * conviction): nvcc --use_fast_math turns logf into __logf, which has
         * absolute error 2^-21.4 ~ 3.6e-7 on [0.5, 2], while the u values
         * closest to 1 have |ln u| of only 0.6~3e-7 — the computed E could be
         * 0 (=> g = +inf, always wins), negative (=> NaN, never wins) or off
         * by a factor, and this is exactly the tail that decides whether an
         * extremely low-probability word wins.  With u == 1.0f fixed the gate
         * stayed red (low-p tokens 1,363 vs 558 expected, 32.7 sigma) because
         * of it.  log1pf has no fast-math replacement and keeps relative
         * accuracy across small arguments; the outer -logf(E) has E in
         * [2^-24, 16.7] where __logf's error is ulp-level, ~1e-5 on g,
         * harmless. */
        const float u = v41_u01(seed, p, i, salt_g);
        const float g = -logf(-log1pf(-u));
        const float key = v * inv_T + g;
        if (key > gb) { gb = key; gi = (int32_t)i; }
        if (lq) {
            const float vq = lq[i];
            const float qm = (isfinite(vq) && v41_fkey(vq) >= cutq) ? expf((vq - Mq) * inv_T) : 0.0f;
            const float rres = m / Zp - qm / Zq;
            if (rres > 0.0f) { const float rk = logf(rres) + g; if (rk > gxb) { gxb = rk; gxi = (int32_t)i; } }
        } else if ((int32_t)i != d && key > gxb) { gxb = key; gxi = (int32_t)i; }
    }
    float tv; int32_t full, resid;
    v41_blk_argmax(sf, si, gb, gi, &tv, &full);
    v41_blk_argmax(sf, si, gxb, gxi, &tv, &resid);
    const float ZK = v41_blk_sumf(sf, zk);
    const float MD = v41_blk_sumf(sf, md);     /* only one thread holds d's term, the sum is the value */
    const float MQD = v41_blk_sumf(sf, mqd);
    if (threadIdx.x == 0) {
        int32_t acc = 0;
        if (d >= 0) {
            const float pd = MD / ZK;                                  /* d not in K_p => MD = 0 => reject (residual non-empty; should it ever be empty [2] falls back to the full sample, still a sample of p) */
            if (pd <= 0.0f) acc = 0;
            else if (resid < 0) acc = 1;                              /* residual empty (K_p subset of {d}, or p == q): ratio = 1, accept */
            else if (lq) {
                const float qd = MQD / Zq;                             /* q(d) = 0 only at a numerical edge (the draft was drawn from q): ratio inf => accept */
                acc = (qd <= 0.0f || v41_u01(seed, p, (uint32_t)d, salt_a) < pd / qd) ? 1 : 0;
            } else acc = v41_u01(seed, p, (uint32_t)d, salt_a) < pd ? 1 : 0;
        }
        o[0] = full; o[1] = acc; o[2] = resid >= 0 ? resid : full; o[3] = Mi;   // [3] = the raw argmax (the tool-syntax greedy hook, c6e2706)
    }
}

extern "C" int ds4_gpu_v41_sample_tensor(ds4_gpu_tensor *out, const ds4_gpu_tensor *logits, uint32_t row0, uint32_t n_rows, uint32_t n_vocab,
                                         const ds4_gpu_tensor *pos, const ds4_gpu_tensor *tok, const ds4_gpu_sample_params *sp, const ds4_gpu_tensor *qlogits) {
    if (!out || !logits || !pos || !tok || !sp || n_rows == 0u || n_vocab == 0u) return 0;
    if (!(sp->temperature > 0.0f)) return 0;   /* greedy is the argmax kernel's job, this one does not pretend */
    const uint64_t last = (uint64_t)row0 + n_rows;
    if (out->bytes < (uint64_t)n_rows * 16u || logits->bytes < last * n_vocab * 4u || pos->bytes < last * 4u || tok->bytes < last * 4u) return 0;
    if (qlogits && (row0 != 0u || (n_rows > 1u && qlogits->bytes < (uint64_t)(n_rows - 1u) * n_vocab * 4u))) return 0;
    /* Parameter conventions match the host ds4_sample_logits: top_p outside
     * (0,1] counts as 1; negative min_p as 0; top_k <= 0 = no cut (the device
     * has no host's 1024 stack limit). */
    float top_p = sp->top_p; if (!(top_p > 0.0f) || top_p > 1.0f) top_p = 1.0f;
    const float min_p = sp->min_p > 0.0f ? sp->min_p : 0.0f;
    const uint32_t top_k = sp->top_k > 0 ? (uint32_t)sp->top_k : 0u;
    v41_sample_kernel<<<n_rows, V41_SAMPLE_THREADS, 0, ds4_current_stream()>>>((int32_t *)out->ptr, (const float *)logits->ptr,
                                                                              qlogits ? (const float *)qlogits->ptr : NULL, n_vocab,
                                                                              (const int32_t *)pos->ptr, (const int32_t *)tok->ptr, row0, n_rows,
                                                                              1.0f / sp->temperature, min_p, top_k, top_p, sp->seed, sp->stream);
    return cuda_ok(cudaGetLastError(), "v41 sample launch");
}

#endif /* DS41_SAMPLE_CUH */
