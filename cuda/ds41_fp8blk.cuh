/* ds41_fp8blk.cuh — DeepSeek V4.1 (ds41) fp8_32x32 decode family, ported from
 * the C engine at /data/YoungAi (commit 3946dbc):
 *   - primitives: src/common/ds4_fp8.h:33-50 (E4M3FN bit decode, E8M0 scale);
 *   - the GEMV family: src/cuda/cuda_v41_gemv_highprec.inc.cu:194-341 (the
 *     kernel, the grouped launcher) and :397-434 (the two tensor entries);
 *   - the single-group entry and the >8 arm: src/cuda/cuda_v41_1.inc.cu:154-167
 *     (x->bf16, round-bf16) and :338-364 (wkv entry, to-bf16 + cuBLAS);
 *   - the engram row dequant: src/cuda/cuda_v41_4.inc.cu:302-315.
 *
 * The on-disk format (ds4_quantfmt.h:36-38): a rows*cols E4M3 plane followed
 * by ceil(rows/32)*ceil(cols/32) E8M0 scale bytes, one scale per 32x32 tile.
 * Value = e4m3 * 2^(e-127); e=0 decodes to the 2^-127 bit pattern (ds4_fp8.h:46).
 * This is NOT a per-row block format, so it never enters the gguf block table:
 * the geometry here reads the two planes directly.
 *
 * Consumers (the engine's): the engram wkv [6144][25600] (ds4_gpu_v41_matmul_
 * fp8blk_tensor, core_v41_engram.c:468) and the 25 tower dense tensors
 * (ds4_gpu_v41_matmul_fp8blk_round_tensor / _grouped_, core_v41_attn.c:208,224).
 *
 * Numerics: the GEMV decodes each weight element as (e4m3 * scale) in f32 and
 * accumulates f32 per token; n>1 tokens read activations already rounded to
 * bf16 (XB=1, the engine's same knife as the skeleton GEMV). The >8 arm
 * expands the weight to bf16 and runs cuBLAS — a different numeric path by
 * design (prefill), named in the engine at cuda_v41_1.inc.cu:350-364. */
#pragma once

/* E4M3FN bit decode (ds4_fp8.h:33-42): abs==0x7f is NaN (E4M3FN has no Inf);
 * exp==0 is the subnormal man*2^-9; else (1+man/8)*2^(exp-7). */
__device__ __forceinline__ static float ds4_e4m3fn_to_f32(uint8_t x) {
    const uint8_t abs = x & 0x7f;
    const bool sign = (x & 0x80) != 0;
    if (abs == 0) return sign ? -0.0f : 0.0f;
    if (abs == 0x7f) return NAN;
    const int exp = (x >> 3) & 0x0f;
    const int man = x & 0x07;
    float value = exp == 0 ? ldexpf((float)man, -9)
                           : ldexpf(1.0f + (float)man / 8.0f, exp - 7);
    return sign ? -value : value;
}

/* E8M0 scale byte -> f32 (ds4_fp8.h:46-50): e=0 is the 0x00400000 bit pattern
 * (2^-127), else 2^(e-127). */
__device__ __forceinline__ static float ds4_e8m0_to_f32(uint8_t e) {
    const uint32_t bits = e == 0 ? 0x00400000u : ((uint32_t)e << 23);
    float result;
    memcpy(&result, &bits, sizeof(result));
    return result;
}

/* ---- the encode-side primitives (ds4_fp8.h:52-164) ----
 * P4-4's act_quant / KV pack need the round and encode halves, not just the
 * decoders: the FP4/E4M3 magnitude tables, nearest-rounding (ties to even
 * mantissa), the nibble map and the byte encoders.  One copy each, shared by
 * act_quant and the pack kernels (a second copy would drift, and the packed
 * bytes must equal what act_quant writes). */
__device__ __forceinline__ static float ds4_e4m3fn_value(int i) {
    static const float exp_scale[16] = {
        0.0f, 0.015625f, 0.03125f, 0.0625f,
        0.125f, 0.25f, 0.5f, 1.0f,
        2.0f, 4.0f, 8.0f, 16.0f,
        32.0f, 64.0f, 128.0f, 256.0f,
    };
    const int exp = (i >> 3) & 0x0f;
    const int mant = i & 0x07;
    return exp == 0
        ? (float)mant * 0.001953125f
        : (1.0f + (float)mant * 0.125f) * exp_scale[exp];
}
__device__ __forceinline__ static float ds4_e4m3fn_round(float x) {
    const float sign = x < 0.0f ? -1.0f : 1.0f;
    const float ax = fminf(fabsf(x), 448.0f);
    int lo = 0;
    int hi = 126;
    while (lo < hi) {
        const int mid = (lo + hi + 1) >> 1;
        if (ds4_e4m3fn_value(mid) <= ax) lo = mid;
        else hi = mid - 1;
    }
    int best = lo;
    if (best < 126) {
        const float best_diff = fabsf(ax - ds4_e4m3fn_value(best));
        const float next_diff = fabsf(ax - ds4_e4m3fn_value(best + 1));
        if (next_diff < best_diff ||
            (next_diff == best_diff && ((best + 1) & 1) == 0 && (best & 1) != 0)) {
            best++;
        }
    }
    return sign * ds4_e4m3fn_value(best);
}
__device__ __forceinline__ static float ds4_e2m1fn_value(int i) {
    static const float values[8] = {
        0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
    };
    return values[i & 7];
}
__device__ __forceinline__ static float ds4_e2m1fn_round(float x) {
    const float sign = x < 0.0f ? -1.0f : 1.0f;
    const float ax = fminf(fabsf(x), 6.0f);
    int best = 0;
    float best_diff = fabsf(ax - ds4_e2m1fn_value(0));
    for (int i = 1; i < 8; i++) {
        const float diff = fabsf(ax - ds4_e2m1fn_value(i));
        if (diff < best_diff || (diff == best_diff && (i & 1) == 0 && (best & 1) != 0)) {
            best = i;
            best_diff = diff;
        }
    }
    return sign * ds4_e2m1fn_value(best);
}
/* FP4 nibble (with sign, 0..15) -> f32 (ds4_fp8.h:122-127; slot 8 is -0.0f).
 * The engine's measured verdict stands: the table beats a bit-trick version
 * (the decode kernels are memory-latency bound and the LDC replays are hidden;
 * the bit version measured 8% slower decode, 2026-09-15). */
__device__ __forceinline__ static float ds4_fp4_nibble_to_f32(uint8_t n) {
    static const float t[16] = {
        0.0f,  0.5f,  1.0f,  1.5f,  2.0f,  3.0f,  4.0f,  6.0f,
       -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f,
    };
    return t[n & 15];
}
__device__ __forceinline__ static uint8_t ds4_fp4_f32_to_nibble(float x) {
    const float v = ds4_e2m1fn_round(x);
    const float av = fabsf(v);
    uint8_t n = 0;
    for (int i = 1; i < 8; i++) {
        if (av == ds4_e2m1fn_value(i)) { n = (uint8_t)i; break; }   /* exact table values, equality is safe */
    }
    return n == 0 ? (uint8_t)0 : (uint8_t)(v < 0.0f ? (n | 8u) : n);
}
__device__ __forceinline__ static uint8_t ds4_e8m0_f32_to_byte(float s) {
    uint32_t bits;
    memcpy(&bits, &s, sizeof(bits));
    if (bits == 0x00400000u) return 0;
    return (uint8_t)((bits >> 23) & 0xffu);
}
__device__ __forceinline__ static uint8_t ds4_e4m3fn_f32_to_byte(float x) {
    const float v = ds4_e4m3fn_round(x);
    const float av = fabsf(v);
    int lo = 0, hi = 126;
    while (lo < hi) {
        const int mid = (lo + hi + 1) >> 1;
        if (ds4_e4m3fn_value(mid) <= av) lo = mid;
        else hi = mid - 1;
    }
    return (uint8_t)((v < 0.0f && av != 0.0f) ? (lo | 0x80) : lo);
}

/* f32 -> bf16 (cuda_v41_1.inc.cu:154-157) and the RNE round-back kernel
 * (:158-162; the port's v41_bf16r lives in ds41_primitives.cuh:53). */
__global__ static void v41_x_to_bf16_kernel(__nv_bfloat16 *out, const float *x, uint64_t n) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = __float2bfloat16(x[i]);
}
__global__ static void v41_round_bf16_kernel(float *x, uint64_t n) {
    v41_pdl_wait();   /* PDL: waits for the upstream when launched through PDL; a no-op otherwise */
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] = v41_bf16r(x[i]);
}
extern "C" int ds4_gpu_v41_round_bf16_tensor(ds4_gpu_tensor *x, uint64_t n) {
    if (!x || x->bytes < n * 4) return 0;
    v41_round_bf16_kernel<<<(unsigned)((n + 255) / 256), 256, 0, ds4_current_stream()>>>((float *)x->ptr, n);
    return cuda_ok(cudaGetLastError(), "v41 round bf16");
}

/* ---- FP8(e4m3 + 32x32 block ue8m0) weight GEMV: engram wkv + the towers ----
 * Layout: w[0..rows*cols) is e4m3; then ceil(rows/32)*ceil(cols/32) ue8m0
 * bytes, tile (r/32, c/32).  One lane eats 16 consecutive elements per step
 * (one 128-bit read), so its 16 elements always sit in one column tile and
 * the scale is looked up once.  in_dim % 32 == 0 is the only shape rule
 * (relaxed 2026-09-17 from 512; the 512 gate now lives only on the wkv
 * entry's GEMV-vs-cuBLAS choice, cuda_v41_1.inc.cu:348). */
/* XB=1: activations are read from the bf16 buffer — the same knife and the
 * same account as the skeleton GEMV (see the engine's kernel comment). */
/* grid.y = the block-diagonal group (2026-09-17): the towers' wo_a is 8 block
 * groups and its row segment [g*out_dim, (g+1)*out_dim) is contiguous in BOTH
 * planes (out_dim is a multiple of 32), so the group is just a row offset.
 * n_groups=1 for every non-grouped call (offset is always 0, byte-for-byte
 * the pre-grouping instruction stream). */
template <uint32_t NT, uint32_t XB>
__global__ static void v41_fp8blk_gemv_kernel(float *out, const uint8_t *w, const uint8_t *sc, const float *x,
                                              const __nv_bfloat16 *x16,
                                              uint32_t in_dim, uint32_t out_dim, uint32_t sbc, uint32_t ksplit,
                                              uint32_t x_stride, uint32_t out_stride,
                                              uint32_t x_gstride, uint32_t out_gstride) {
    v41_pdl_wait();   /* PDL: first instruction waits for the upstream; no-op when not launched through PDL */
    const uint32_t g = blockIdx.y;
    w += (uint64_t)g * in_dim * out_dim;
    sc += (uint64_t)g * ((out_dim + 31u) / 32u) * sbc;
    x += (uint64_t)g * x_gstride; out += (uint64_t)g * out_gstride;
    if (XB) x16 += (uint64_t)g * x_gstride;
    const uint32_t warp = threadIdx.x >> 5, lane = threadIdx.x & 31u;
    const uint32_t rows_per_block = 8u / ksplit, rloc = warp / ksplit, kpart = warp % ksplit;
    const uint32_t r = blockIdx.x * rows_per_block + rloc;
    __shared__ float red[8][NT];
    float acc[NT];
    #pragma unroll
    for (uint32_t t = 0; t < NT; t++) acc[t] = 0.f;
    if (r < out_dim) {
        const uint8_t *wr = w + (uint64_t)r * in_dim;
        const uint8_t *scr = sc + (uint64_t)(r >> 5) * sbc;
        for (uint32_t c = kpart * 512u + lane * 16u; c < in_dim; c += ksplit * 512u) {
            const uint4 raw = *(const uint4 *)(wr + c);
            uint8_t b[16];
            memcpy(b, &raw, 16);
            const float s = ds4_e8m0_to_f32(scr[c >> 5]);
            float wv[16];
            #pragma unroll
            for (uint32_t j = 0; j < 16u; j++) wv[j] = ds4_e4m3fn_to_f32(b[j]) * s;
            #pragma unroll
            for (uint32_t t = 0; t < NT; t++) {
                /* The two arms write their own loop bodies on purpose: a shared
                 * float xv[16] intermediate changes the whole kernel's register
                 * allocation (measured 5% slower decode on the skeleton GEMV);
                 * the XB=0 arm is the pre-change instruction stream verbatim. */
                if (XB) {   /* 16 bf16 = 32 B contiguous, two uint4; the shift restores f32 exactly */
                    const uint4 *xp = (const uint4 *)(x16 + (uint64_t)t * x_stride + c);
                    const uint4 x0 = xp[0], x1 = xp[1];
                    const uint32_t xu[8] = { x0.x, x0.y, x0.z, x0.w, x1.x, x1.y, x1.z, x1.w };
                    #pragma unroll
                    for (uint32_t j = 0; j < 8u; j++) {
                        acc[t] += wv[2u * j]      * __uint_as_float(xu[j] << 16);
                        acc[t] += wv[2u * j + 1u] * __uint_as_float(xu[j] & 0xffff0000u);
                    }
                } else {
                    const float *xt = x + (uint64_t)t * x_stride + c;
                    #pragma unroll
                    for (uint32_t j = 0; j < 16u; j++) acc[t] += wv[j] * xt[j];
                }
            }
        }
        #pragma unroll
        for (uint32_t t = 0; t < NT; t++) {
            float v = acc[t];
            for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
            acc[t] = v;
        }
    }
    if (ksplit > 1u) {
        if (lane == 0) {
            #pragma unroll
            for (uint32_t t = 0; t < NT; t++) red[warp][t] = acc[t];
        }
        __syncthreads();
        if (kpart == 0 && lane == 0 && r < out_dim) {
            #pragma unroll
            for (uint32_t t = 0; t < NT; t++) {
                float v = 0.f;
                for (uint32_t k = 0; k < ksplit; k++) v += red[rloc * ksplit + k][t];
                out[(uint64_t)t * out_stride + r] = v;
            }
        }
    } else if (lane == 0 && r < out_dim) {
        #pragma unroll
        for (uint32_t t = 0; t < NT; t++) out[(uint64_t)t * out_stride + r] = acc[t];
    }
}

/* Prefill (n > 8): expand this table to bf16 once, then one cuBLAS call.
 * Only allowed to spill here: e4m3 with 2-D block scales has no cuBLASLt FP8
 * algorithm (it eats one scale per tensor), and the table is 0.157 GB/layer
 * with only two layers, so the expansion amortizes over the whole token
 * block.  The decode path above spills nothing. */
__global__ static void v41_fp8blk_to_bf16_kernel(__nv_bfloat16 *o, const uint8_t *w, const uint8_t *sc,
                                                 uint32_t in_dim, uint32_t sbc, uint64_t n) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const uint32_t r = (uint32_t)(i / in_dim), c = (uint32_t)(i % in_dim);
    o[i] = __float2bfloat16(ds4_e4m3fn_to_f32(w[i]) * ds4_e8m0_to_f32(sc[(uint64_t)(r >> 5) * sbc + (c >> 5)]));
}

/* in_dim % 32 is the shape rule (a lane's 16 elements and the 32-wide scale
 * tile need it); a ragged last warp is fine, only idle lanes. */
static v41_scratch g_v41_fp8_xb;
static v41_scratch g_v41_fp8_wbf, g_v41_fp8_xbf;
static int v41_fp8blk_gemv_g(const uint8_t *w, const uint8_t *sc, uint64_t in_dim, uint64_t out_dim,
                             const float *x, float *out, uint32_t n_tok, uint32_t n_groups,
                             uint32_t x_stride, uint32_t out_stride, uint32_t x_gstride, uint32_t out_gstride,
                             const char *what) {
    if ((in_dim % 32u) != 0u || n_tok == 0 || n_groups == 0) return 0;
    if (n_tok > V41_GEMV_MAX_TOK) {
        /* Bigger batches loop in 8-row segments (the engine's 2026-09-18 fix:
         * the draft tower's 128-position window rebuild needs one call; the
         * weight is re-read per segment, decode keeps single-issue rows). */
        for (uint32_t off = 0; off < n_tok; off += V41_GEMV_MAX_TOK) {
            const uint32_t nn = n_tok - off < V41_GEMV_MAX_TOK ? n_tok - off : V41_GEMV_MAX_TOK;
            if (!v41_fp8blk_gemv_g(w, sc, in_dim, out_dim, x + (uint64_t)off * x_stride, out + (uint64_t)off * out_stride,
                                   nn, n_groups, x_stride, out_stride, x_gstride, out_gstride, what)) return 0;
        }
        return 1;
    }
    uint32_t ksplit = 1;
    while (ksplit < 8u && out_dim * ksplit * n_groups < 32768u) ksplit <<= 1;
    const uint32_t nseg = (uint32_t)((in_dim + 511u) / 512u);
    while (ksplit > 1u && ksplit > nseg) ksplit >>= 1;
    const uint32_t rpb = 8u / ksplit, sbc = (uint32_t)((in_dim + 31u) / 32u);
    const dim3 grid((unsigned)((out_dim + rpb - 1u) / rpb), n_groups);
    /* n>1 converts the activations to bf16 (the same knife as the skeleton
     * GEMV); n=1 does not (that path is byte-bound). */
    const __nv_bfloat16 *x16 = NULL;
    if (n_tok > 1u) {
        const uint64_t xn = (uint64_t)n_tok * x_stride;
        __nv_bfloat16 *xbuf = (__nv_bfloat16 *)v41_grow(&g_v41_fp8_xb, xn * sizeof(__nv_bfloat16), "v41 fp8blk x bf16");
        if (!xbuf) return 0;
        v41_x_to_bf16_kernel<<<(unsigned)((xn + 255) / 256), 256, 0, ds4_current_stream()>>>(xbuf, x, xn);
        if (!cuda_ok(cudaGetLastError(), "v41 fp8blk x bf16")) return 0;
        x16 = xbuf;
    }
    #define V41_FP8GEMV_LAUNCH(NT, XB) v41_fp8blk_gemv_kernel<NT, XB><<<grid, 256, 0, ds4_current_stream()>>>( \
        out, w, sc, x, x16, (uint32_t)in_dim, (uint32_t)out_dim, sbc, ksplit, \
        x_stride, out_stride, x_gstride, out_gstride)
    switch (n_tok) {
        case 1: V41_FP8GEMV_LAUNCH(1u, 0u); break;  case 2: V41_FP8GEMV_LAUNCH(2u, 1u); break;
        case 3: V41_FP8GEMV_LAUNCH(3u, 1u); break;  case 4: V41_FP8GEMV_LAUNCH(4u, 1u); break;
        case 5: V41_FP8GEMV_LAUNCH(5u, 1u); break;  case 6: V41_FP8GEMV_LAUNCH(6u, 1u); break;
        case 7: V41_FP8GEMV_LAUNCH(7u, 1u); break;  default: V41_FP8GEMV_LAUNCH(8u, 1u); break;
    }
    #undef V41_FP8GEMV_LAUNCH
    return cuda_ok(cudaGetLastError(), what);
}

/* Single-group entry (the engine's v41_fp8blk_gemv): groups 1, strides = dims. */
static int v41_fp8blk_gemv(const uint8_t *w, const uint8_t *sc, uint64_t in_dim, uint64_t out_dim,
                           const float *x, float *out, uint32_t n_tok, const char *what) {
    return v41_fp8blk_gemv_g(w, sc, in_dim, out_dim, x, out, n_tok, 1u,
                             (uint32_t)in_dim, (uint32_t)out_dim, 0u, 0u, what);
}

/* Expand one whole fp8 tensor to bf16 (cuda_v41_gemv_highprec.inc.cu:344-348). */
static int v41_fp8blk_to_bf16(__nv_bfloat16 *o, const uint8_t *w, const uint8_t *sc, uint64_t in_dim, uint64_t rows) {
    const uint64_t n = in_dim * rows;
    const uint32_t sbc = (uint32_t)((in_dim + 31u) / 32u);
    v41_fp8blk_to_bf16_kernel<<<(unsigned)((n + 255) / 256), 256, 0, ds4_current_stream()>>>(o, w, sc, (uint32_t)in_dim, sbc, n);
    return cuda_ok(cudaGetLastError(), "v41 fp8blk->bf16");
}

/* engram wkv entry (cuda_v41_1.inc.cu:338-364): n <= 8 and in_dim % 512 == 0
 * take the fused GEMV; otherwise expand to bf16 + one cuBLAS call. */
extern "C" int ds4_gpu_v41_matmul_fp8blk_tensor(ds4_gpu_tensor *out, const void *model_map, uint64_t model_size,
                                     uint64_t weight_offset, uint64_t in_dim, uint64_t out_dim,
                                     const ds4_gpu_tensor *x, uint32_t n_tok) {
    if (!out || !x || n_tok == 0) return 0;
    const uint64_t sbc = (in_dim + 31u) / 32u, sbr = (out_dim + 31u) / 32u;
    const uint64_t wbytes = in_dim * out_dim + sbr * sbc;
    if (weight_offset > model_size || wbytes > model_size - weight_offset) return 0;
    const uint8_t *W = (const uint8_t *)cuda_model_range_ptr(model_map, weight_offset, wbytes, "v41 fp8blk w");
    if (!W) return 0;
    const uint8_t *SC = W + in_dim * out_dim;
    if (n_tok <= V41_GEMV_MAX_TOK && (in_dim % 512u) == 0u)
        return v41_fp8blk_gemv(W, SC, in_dim, out_dim, (const float *)x->ptr, (float *)out->ptr, n_tok, "v41 fp8blk gemv");
    if (!g_cublas_ready) return 0;
    const uint64_t wn = in_dim * out_dim;
    __nv_bfloat16 *wb = (__nv_bfloat16 *)v41_grow(&g_v41_fp8_wbf, wn * sizeof(__nv_bfloat16), "v41 wkv bf16");
    if (!wb || !v41_fp8blk_to_bf16(wb, W, SC, in_dim, out_dim)) return 0;
    const uint64_t xn = (uint64_t)n_tok * in_dim;
    __nv_bfloat16 *xb = (__nv_bfloat16 *)v41_grow(&g_v41_fp8_xbf, xn * sizeof(__nv_bfloat16), "v41 wkv x bf16");
    if (!xb) return 0;
    v41_x_to_bf16_kernel<<<(unsigned)((xn + 255) / 256), 256, 0, ds4_current_stream()>>>(xb, (const float *)x->ptr, xn);
    if (!cuda_ok(cudaGetLastError(), "v41 wkv x->bf16")) return 0;
    /* The engine calls cublasSetStream before every V4.1 GEMM
     * (cuda_v41_1.inc.cu:360 here), and per the cuBLAS docs (2.4.7) that call
     * unconditionally resets the bound workspace to the default pool — so the
     * engine's GEMMs actually run on the default pool, not on its 32 MB user
     * buffer.  Mirror the call (the port's stream is the legacy default, so
     * ordering is unchanged); cuda_cublas_ws_prep (the port's S1.1a rule,
     * ds4_cuda.cu) then only touches the now-unbound user buffer. */
    (void)cublasSetStream(g_cublas, ds4_current_stream());
    cuda_cublas_ws_prep(ds4_current_stream());
    const float alpha = 1.0f, beta = 0.0f;
    cublasStatus_t st = cublasGemmEx(g_cublas, CUBLAS_OP_T, CUBLAS_OP_N, (int)out_dim, (int)n_tok, (int)in_dim, &alpha,
                                     wb, CUDA_R_16BF, (int)in_dim, xb, CUDA_R_16BF, (int)in_dim, &beta,
                                     (float *)out->ptr, CUDA_R_32F, (int)out_dim, CUDA_R_32F, CUBLAS_GEMM_DEFAULT);
    return cublas_ok(st, "v41 wkv bf16 gemm");
}

/* ---- the towers' fp8 entries (mtp-1.md M6 root fix, 2026-09-17) ----
 * DSpark's towers are FP8 in the original; the converter used to compress
 * them to FP4 with the skeleton and halved the drafter's precision (first-
 * position agreement 0.50).  Both entries round the output to bf16 once
 * when round_out is set — the fp4 arm folds that rounding into its kernel,
 * and the two must share one numeric contract. */
extern "C" int ds4_gpu_v41_matmul_fp8blk_round_tensor(ds4_gpu_tensor *out, const void *model_map, uint64_t model_size,
                                           uint64_t weight_offset, uint64_t in_dim, uint64_t out_dim,
                                           const ds4_gpu_tensor *x, uint32_t n_tok, int round_out) {
    if (!out || !x || n_tok == 0) return 0;
    const uint64_t sbc = (in_dim + 31u) / 32u, sbr = (out_dim + 31u) / 32u;
    const uint64_t wbytes = in_dim * out_dim + sbr * sbc;
    if (weight_offset > model_size || wbytes > model_size - weight_offset) return 0;
    const uint8_t *W = (const uint8_t *)cuda_model_range_ptr(model_map, weight_offset, wbytes, "v41 mtp fp8 w");
    if (!W) return 0;
    if (!v41_fp8blk_gemv_g(W, W + in_dim * out_dim, in_dim, out_dim, (const float *)x->ptr, (float *)out->ptr,
                           n_tok, 1u, (uint32_t)in_dim, (uint32_t)out_dim, 0u, 0u, "v41 mtp fp8 gemv")) return 0;
    return round_out ? ds4_gpu_v41_round_bf16_tensor(out, (uint64_t)n_tok * out_dim) : 1;
}
extern "C" int ds4_gpu_v41_grouped_matmul_fp8blk_tensor(ds4_gpu_tensor *low, const void *model_map, uint64_t model_size,
                                             uint64_t weight_offset, uint32_t n_groups, uint64_t group_dim,
                                             uint64_t rank, const ds4_gpu_tensor *heads, uint32_t n_tok, int round_out) {
    if (!low || !heads || n_tok == 0 || n_groups == 0) return 0;
    const uint64_t in_all = (uint64_t)n_groups * group_dim, out_all = (uint64_t)n_groups * rank;
    const uint64_t sbc = (group_dim + 31u) / 32u, sbr = (rank + 31u) / 32u;
    /* On disk: the whole [n_groups*rank][group_dim] e4m3 plane plus the same
     * shape's scale plane — group g's row segment is contiguous, so the
     * kernel only needs the grid.y row offset (see the kernel comment). */
    const uint64_t wbytes = out_all * group_dim + n_groups * sbr * sbc;
    if (weight_offset > model_size || wbytes > model_size - weight_offset) return 0;
    const uint8_t *W = (const uint8_t *)cuda_model_range_ptr(model_map, weight_offset, wbytes, "v41 mtp wo_a fp8");
    if (!W) return 0;
    if (!v41_fp8blk_gemv_g(W, W + out_all * group_dim, group_dim, rank, (const float *)heads->ptr,
                           (float *)low->ptr, n_tok, n_groups, (uint32_t)in_all, (uint32_t)out_all,
                           (uint32_t)group_dim, (uint32_t)rank, "v41 mtp wo_a fp8 gemv")) return 0;
    return round_out ? ds4_gpu_v41_round_bf16_tensor(low, (uint64_t)n_tok * out_all) : 1;
}

/* ---- engram table rows: fp8 (e4m3 x ue8m0/32) -> bf16 grid ----
 * One thread per element (cuda_v41_4.inc.cu:302-308): row stride is
 * head_dim + head_dim/32 (the scale plane of one row is its own tail). */
__global__ static void v41_engram_rows_kernel(float *out, const uint8_t *raw, uint32_t n_rows, uint32_t hd) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (uint64_t)n_rows * hd) return;
    const uint32_t r = (uint32_t)(i / hd), d = (uint32_t)(i % hd), stride = hd + hd / 32u;
    const uint8_t *row = raw + (uint64_t)r * stride;
    out[i] = v41_bf16r(ds4_e4m3fn_to_f32(row[d]) * ds4_e8m0_to_f32(row[hd + d / 32u]));
}
extern "C" int ds4_gpu_v41_engram_rows_tensor(ds4_gpu_tensor *out, const ds4_gpu_tensor *raw, uint32_t n_rows, uint32_t head_dim) {
    if (!out || !raw || (head_dim % 32u)) return 0;
    const uint64_t n = (uint64_t)n_rows * head_dim;
    if (out->bytes < n * 4 || raw->bytes < (uint64_t)n_rows * (head_dim + head_dim / 32u)) return 0;
    v41_engram_rows_kernel<<<(unsigned)((n + 255) / 256), 256, 0, ds4_current_stream()>>>((float *)out->ptr, (const uint8_t *)raw->ptr, n_rows, head_dim);
    return cuda_ok(cudaGetLastError(), "v41 engram rows");
}
