# Gemma-4 Phase 3 CUDA Optimization Review (`sm_89` / RTX 4070 SUPER)

### 1. Vocabulary Head (`262144 × 2816` BF16 GEMV, 1.476 GB)

#### Why [`glmvq_bf16_dot`](file:///data/ds4-gemma4/cuda/glm_vq_gemv.cuh#L37-L80) leaves bandwidth on the table on `sm_89`
When called from [`ds4_gemma4_graph.inc`](file:///data/ds4-gemma4/ds4_gemma4_graph.inc#L860-L863) with `rows = 262144, cols = 2816, tokens = 1`, [`glmvq_bf16_decode`](file:///data/ds4-gemma4/cuda/glm_vq_gemv.cuh#L102) selects `split = 1` (`rows >= 4096`) and launches `32,768` blocks of `8` warps (`1` warp/row):
1. **Unnecessary barrier & shared store on `split == 1`:** Every block executes `partial[warp][0] = sum; __syncthreads();` ([lines 71–73](file:///data/ds4-gemma4/cuda/glm_vq_gemv.cuh#L71-L73)) even though `split == 1` has no cross-warp reduction.
2. **Redundant `x` loads and BF16 conversions:** Every warp re-loads `input[0..2815]` (11.25 KiB) via two `float4` loads per `uint4` weight load and executes 8 `glmvq_round` (`__float2bfloat16_rn` + `__bfloat162float`) calls per 8 FMAs ([lines 22–25](file:///data/ds4-gemma4/cuda/glm_vq_gemv.cuh#L22-L25))—`262,144 × 2,816 = 7.38 × 10^8` redundant conversions per token—even though `g->norm` is already BF16-rounded by [`g4_rms_kernel`](file:///data/ds4-gemma4/cuda/gemma4_primitives.cuh#L67).
3. **Insufficient per-warp MLP & L2 thrashing:** The `#pragma unroll 4` loop over `11` steps (`2816 / 256 = 11`) has a partial remainder and uses default `.ca` caching on a 1.476 GB stream, thrashing the 4070 SUPER's 36 MB L2 ~41× per token. (`cp.async.bulk.prefetch` on [line 47](file:///data/ds4-gemma4/cuda/glm_vq_gemv.cuh#L47) is gated on `__CUDA_ARCH__ >= 900` and is a no-op on `sm_89`.)
4. **Separate softcap pass:** [`ds4_gpu_g4_softcap`](file:///data/ds4-gemma4/ds4_gemma4_graph.inc#L862) launches a second kernel that re-reads and re-writes 1.05 MB of logits.

#### Optimal `sm_89` Kernel Shape
Put a Gemma-only kernel in [`cuda/gemma_vq_decode.cuh`](file:///data/ds4-gemma4/cuda/gemma_vq_decode.cuh) (and wrapper in [`ds4_gemma4_gpu.cuh`](file:///data/ds4-gemma4/ds4_gemma4_gpu.cuh)) specialized for `COLS = G4_EMBED (2816)`:

* **Grid & Block:** Persistent (or fixed multi-row) grid of **`192` blocks × `256` threads (`8` warps/block)** (`4` blocks/SM × `48` SMs = `32` warps/SM = 66.7% occupancy).
* **Shared memory:** `__shared__ __align__(16) float sx[G4_EMBED];` (`11,264` bytes/block = `44 KiB`/SM across 4 blocks/SM, fitting within the `48 KiB` default static limit without `cudaFuncSetAttribute`).
  * At CTA startup, the 256 threads cooperatively load `input[0..2815]`, apply `g4_round` **once**, store to `sx`, and run **one `__syncthreads()`** for the entire CTA lifetime.
* **Row assignment:** Grid-stride warp loop (`const uint32_t total_warps = gridDim.x * 8; for (uint32_t row = blockIdx.x * 8 + warp; row < rows; row += total_warps)`). Inside the loop, warps never synchronize (`__syncthreads()` count is **zero**), so warps stall and issue independently while streaming weights.
* **Per-lane load width & L2 hint:** Keep `uint4` (16-byte = 8 BF16) per lane at stride `256` columns (`col = lane * 8 + k * 256` for `k = 0..10`), fully unrolled (`#pragma unroll 11`). Load weights with **`__ldcs((const uint4 *)(w + col))`** (`ld.global.cs.v4.u32`, evict-first streaming L2 policy) and prefetch two `k`-steps (`32` bytes/lane = `1 KiB`/warp) into registers ahead of the FMA chain.
  * *Why not contiguous 32-byte (`lane * 16`) loads?* `sm_89`'s widest hardware load instruction is `LDG.E.128` (`uint4`); two stride-256 `uint4` loads in flight give the same 32-byte per-lane MLP while preserving each lane's exact column subset `{lane*8 + k*256}`.
* **Reduction & fused epilogue:** 5-step butterfly `sum += __shfl_xor_sync(0xffffffffu, sum, stride)` (`stride = 16, 8, 4, 2, 1`). Lane 0 rounds `float logit = g4_round(sum);` and—if softcap is fused—applies [`g4_softcap_kernel`](file:///data/ds4-gemma4/cuda/gemma4_primitives.cuh#L140-L146)'s exact 3-rounding expression before writing `output[row]`.

```cuda
template <bool SOFTCAP>
__global__ void __launch_bounds__(256, 4) g4_vocab_gemv_kernel(
        const __nv_bfloat16 * __restrict__ weight,
        const float * __restrict__ input,
        float * __restrict__ output,
        uint32_t rows, float cap) {
    __shared__ __align__(16) float sx[G4_EMBED];
    for (uint32_t c = threadIdx.x; c < G4_EMBED; c += 256u) {
        sx[c] = g4_round(input[c]);
    }
    __syncthreads();

    const uint32_t warp = threadIdx.x >> 5, lane = threadIdx.x & 31u;
    const uint32_t base_col = lane * 8u;
    const uint32_t stride_rows = gridDim.x * 8u;

    for (uint32_t row = blockIdx.x * 8u + warp; row < rows; row += stride_rows) {
        const __nv_bfloat16 *w = weight + (uint64_t)row * G4_EMBED + base_col;
        float sum = 0.0f;
        #pragma unroll 11
        for (uint32_t k = 0; k < 11u; k++) {
            const uint4 wv = __ldcs((const uint4 *)(w + k * 256u));
            const float4 a = *(const float4 *)(sx + base_col + k * 256u);
            const float4 b = *(const float4 *)(sx + base_col + k * 256u + 4u);
            sum = fmaf(glmvq_bf16_lo(wv.x), a.x, sum);
            sum = fmaf(glmvq_bf16_hi(wv.x), a.y, sum);
            sum = fmaf(glmvq_bf16_lo(wv.y), a.z, sum);
            sum = fmaf(glmvq_bf16_hi(wv.y), a.w, sum);
            sum = fmaf(glmvq_bf16_lo(wv.z), b.x, sum);
            sum = fmaf(glmvq_bf16_hi(wv.z), b.y, sum);
            sum = fmaf(glmvq_bf16_lo(wv.w), b.z, sum);
            sum = fmaf(glmvq_bf16_hi(wv.w), b.w, sum);
        }
        #pragma unroll
        for (uint32_t s = 16u; s; s >>= 1u) {
            sum += __shfl_xor_sync(0xffffffffu, sum, s);
        }
        if (lane == 0u) {
            float y = g4_round(sum);
            if constexpr (SOFTCAP) {
                const float div = g4_round(y / cap);
                const float sq  = g4_round((float)tanh((double)div));
                y = g4_round(sq * cap);
            }
            output[row] = y;
        }
    }
}
```

* **Exactness risk:** **Zero (bit-identical).** Every lane accumulates the exact same 11×8 products in the exact same order, reduces via the same `__shfl_xor_sync` tree, and rounds at the same boundaries.
* **Expected bandwidth & gain:** **450–470 GB/s** on RTX 4070 SUPER (~89–93% of 504 GB/s peak, **~3.15–3.28 ms** per token, plus saving ~0.04 ms from fusing `g4_softcap_kernel`).

---

### 2. Fused RMSNorm + RoPE ([`cuda/gemma4_primitives.cuh`](file:///data/ds4-gemma4/cuda/gemma4_primitives.cuh#L44-L122))

#### Why [`g4_rope_kernel`](file:///data/ds4-gemma4/cuda/gemma4_primitives.cuh#L106-L122) costs 18.1 ms at 256-token prefill
`g4_rope_kernel` assigns **one thread per `(row, head)`** and runs a serial loop of `pairs` iterations (`128` on sliding layers, `64` on full layers). Each thread executes `128` serial **FP64** `cos(double)` and `sin(double)` calls (on `sm_89`, FP64 throughput is 1/64 of FP32: only 2 FP64 units/SM) while reading/writing `v[i]` at stride `head_dim` (`1024` or `2048` bytes between adjacent threads in a warp—32-way uncoalesced global memory traffic).

#### Evaluation of the Proposed Shape & Better Structure
[`ds4_gpu_g4_rms`](file:///data/ds4-gemma4/ds4_gemma4_graph.inc#L654-L657) already launches `rows * heads` blocks of `256` threads (**one block per `(row, head)`**). Fusing RoPE into that norm block so **thread `i = threadIdx.x` (`i < pairs`) handles pair `i` in parallel** eliminates both the 128-step serial FP64 loop per thread and the uncoalesced global round-trip.

**Two improvements over the baseline fused plan:**
1. **Use `sincos((double)angle, &sd, &cd)` instead of separate `cos` and `sin`:** Evaluates the double-precision Payne-Hanek range reduction once per angle (~1.8× faster on `sm_89`'s 2 FP64 units) and is bit-identical to separate `cos((double)angle)` / `sin((double)angle)`.
2. **Use a 2D grid `dim3(heads, rows)` (or share `(c, s)` across heads in prefill):**
   * For **decode (`rows = 1`)**, `dim3(heads, 1)` (`16` blocks for Q, `8` or `2` for K) keeps multiple SMs active.
   * For **prefill (`rows = 256`)**, note that all `16` Q heads and `8` K heads in the same `row` share identical `positions[row]` and `inv[0..pairs-1]`. Even with `dim3(heads, rows)` (one block per `(row, head)`), parallelizing across `i = threadIdx.x < pairs` drops the per-thread serial FP64 calls from `256` to `1` (`sincos`), cutting the 256-token prefill cost from **18.1 ms to ~0.25 ms**. (If you want to cut the remaining FP64 work by another 16× in prefill, a `rows`-block kernel that computes `c[0..pairs-1], s[0..pairs-1]` once per `row` in shared memory and loops over `h = 0..heads-1` does `sincos` once per token instead of 24×; dispatch that when `rows >= 16` and `dim3(heads, rows)` when `rows < 16`.)

#### Six Correctness Traps to Watch
1. **Pre-RoPE BF16 rounding (`g4_round(y)`):** [`g4_rms_kernel`](file:///data/ds4-gemma4/cuda/gemma4_primitives.cuh#L67) rounds `y` to BF16 *before* writing `o[d]`. `a` and `b` entering RoPE **must** be `g4_round(y)`, not raw FP32 `y`.
2. **Partial rotary (`dim = 512, half = 256, pairs = 64`):** On the 5 full layers (`5, 11, 17, 23, 29`), only `[0, 64)` and `[256, 320)` rotate; indices `[64, 256)` and `[320, 512)` must still write their normalized BF16 values `g4_round(y)` to `out` unmodified.
3. **Actually zero shared-memory staging is needed for `dim = 512`, and `scratch[256]` can be reused for `dim = 256`:**
   * When `dim == 512` and `BLOCK == 256`, `half = 256 = BLOCK`. Thread `t` (`0..255`) already computes **both** `d = t` (`a`) and `d = t + 256` (`b`)! Threads `t < 64` can rotate `(a, b)` directly in registers; threads `t >= 64` write `a` and `b` directly.
   * When `dim == 256` (`half = 128, pairs = 128`), thread `t` writes `scratch[t] = g4_round(y)` after reading `scratch[0]` for `inv` (guarded by the existing `__syncthreads()`), synchronizes once, and threads `t < 128` read `a = scratch[t], b = scratch[t + 128]`.
   * If you prefer one unified path, declare `__shared__ float s_row[512]` (2 KiB), write all `d < dim` to `s_row[d]`, `__syncthreads()`, have `t < pairs` overwrite `s_row[t]` and `s_row[t + half]`, `__syncthreads()`, and flush `s_row[0..dim-1]` coalesced to `out`.
4. **Sign placement inside `-b * s`:** Keep `g4_round(g4_round(a * c) + g4_round(-b * s))` exactly as in [`g4_rope_kernel:119`](file:///data/ds4-gemma4/cuda/gemma4_primitives.cuh#L119).
5. **`positions` indexing:** In `ds4_gpu_g4_rms`, `rows` was `rows * heads`. The fused kernel must index `positions[row]` with `row = blockIdx.y` (in `dim3(heads, rows)`) or `blockIdx.x / heads`, **not** `blockIdx.x`.
6. **`attention_k_eq_v` copy order ([`ds4_gemma4_graph.inc:650-653`](file:///data/ds4-gemma4/ds4_gemma4_graph.inc#L650-L653)):** Full layers copy raw `g->k` to `g->v` *before* `key norm`. Keep that copy before the fused `key norm + rope` call (and don't fuse RoPE into `value norm`). Also update the drafter's `query norm + query rope` in [`ds4_gemma4_mtp.inc:259-262`](file:///data/ds4-gemma4/ds4_gemma4_mtp.inc#L259-L262).

```cuda
template <unsigned BLOCK>
__global__ static void g4_rms_rope_kernel(float *out, const float *in,
        const __nv_bfloat16 *weight, const uint32_t *positions, const float *inv_freq,
        uint32_t rows, uint32_t heads, uint32_t dim, uint32_t pairs, float eps) {
    const uint32_t head = blockIdx.x, row = blockIdx.y;
    if (head >= heads || row >= rows) { return; }
    const uint64_t offset = ((uint64_t)row * heads + head) * dim;
    const float *x = in + offset;
    float sum = 0.0f;
    for (uint32_t d = threadIdx.x; d < dim; d += BLOCK) {
        const float v = x[d];
        sum += v * v;
    }
    __shared__ float scratch[512]; // covers BLOCK <= 256 reduction and dim <= 512 staging
    scratch[threadIdx.x] = sum;
    __syncthreads();
    for (unsigned stride = BLOCK / 2u; stride; stride >>= 1u) {
        if (threadIdx.x < stride) { scratch[threadIdx.x] += scratch[threadIdx.x + stride]; }
        __syncthreads();
    }
    const float inv = (float)(1.0 / sqrt((double)(scratch[0] / (float)dim + eps)));
    __syncthreads();

    for (uint32_t d = threadIdx.x; d < dim; d += BLOCK) {
        float y = x[d] * inv;
        if (weight) { y *= __bfloat162float(weight[d]); }
        scratch[d] = g4_round(y);
    }
    __syncthreads();

    const uint32_t t = threadIdx.x;
    const uint32_t half = dim / 2u;
    float *o = out + offset;
    if (t < pairs) {
        const double angle = (double)(float)((double)positions[row] * (double)inv_freq[t]);
        double sd, cd;
        sincos(angle, &sd, &cd);
        const float c = g4_round((float)cd);
        const float s = g4_round((float)sd);
        const float a = scratch[t], b = scratch[t + half];
        o[t]        = g4_round(g4_round(a * c) + g4_round(-b * s));
        o[t + half] = g4_round(g4_round(b * c) + g4_round(a * s));
    }
    for (uint32_t d = t; d < dim; d += BLOCK) {
        if (d >= pairs && (d < half || d >= half + pairs)) {
            o[d] = scratch[d];
        }
    }
}
```

* **Expected gain:** Saves **~17.8 ms** on 256-token prefill (18.1 ms $\to$ ~0.25 ms) and **~0.25–0.35 ms/token** on batch-1 decode (60 fewer kernel launches + eliminating serial FP64 trig).

---

### 3. The Router (`g4_router_input_kernel` + `128 × 2816` BF16 dot + `g4_router_kernel`)

#### Critical Exactness Trap: `glmvq_bf16_decode` Uses `split = 8`, NOT `split = 1`
Check [`cuda/glm_vq_gemv.cuh:102`](file:///data/ds4-gemma4/cuda/glm_vq_gemv.cuh#L102):
```cuda
const uint32_t split = rows < GLMVQ_DOT_WIDE_ROWS && cols >= GLMVQ_DOT_SPLIT_COLS ? GLMVQ_DOT_SPLIT : 1;
```
For the router (`rows = 128 < 4096`, `cols = 2816 >= 2048`), **`split` is `8`**, not `1`!
That means in the current decode path (`rows <= 4` or `DS4_GEMMA4_EXACT_PREFILL=1`), each expert row `e` is computed by a **full 256-thread (`8`-warp) block**, not a single warp:
1. Thread `tid = part * 32 + lane` (`0..255`) accumulates `col = tid * 8` (`0..2047`) in iteration 0, and **only threads `tid < 96` (`part = 0, 1, 2`)** accumulate a second vector at `col = 2048 + tid * 8` (`2048..2815`) in iteration 1 (`step = 8 * 32 * 8 = 2048`).
2. Each warp `part` (`0..7`) reduces its 32 lanes via `__shfl_xor_sync(0xffffffffu, sum, stride)` (`stride = 16, 8, 4, 2, 1`) into `partial[part]`.
3. Thread 0 sums `partial[0..7]` **sequentially in left-to-right warp order** (`for (i = 0; i < 8; i++) value += partial[i];`) and rounds `g4_round(value)`.

If a fused kernel assigned **1 warp per expert** (each lane summing 11 vectors at stride 256 before one 32-lane shuffle), the FP32 association order would differ on every logit and flip router ties!

#### Performance Trap in the Plan ("One Block Per Row")
For batch-1 decode (`rows = 1`), launching **one block per token row (`grid = 1`)** restricts the entire `128 × 2816` (`704 KiB`) weight matrix read to **1 SM out of 48 SMs (2.1% of the GPU)**. A single SM on `sm_89` can only sustain ~25–35 GB/s from DRAM (~20–28 µs/layer = **600–850 µs/token** across 30 layers), which will **regress** batch-1 decode compared to the current 128-block launch!

#### Correct Design for `sm_89`
To get the full benefit of the 3.4% router share (~1.36 ms/token across 30 layers) **without** collapsing decode to 1 SM and while keeping **100% bitwise-identical logits**:

1. **Fuse `g4_router_input_kernel` into the 128-block dot launch (`grid = dim3(G4_EXPERTS, rows)`, `block = 256` threads), modeled on [`glmvq_hc_mix`](file:///data/ds4-gemma4/cuda/glm_vq_gemv.cuh#L119-L172):**
   * Block `(e, row)` (`256` threads = `8` warps) owns expert `e` (`0..127`) for token `row`.
   * All 256 threads compute the RMS norm and 3-stage BF16 rounding of `in + row * 2816` (`11.25 KiB`, which stays hot in L2 across all 128 blocks on the 48 SMs) in the **exact same 256-thread reduction order** as [`g4_router_input_kernel<256>`](file:///data/ds4-gemma4/cuda/gemma4_primitives.cuh#L75-L100), staging the rounded input into `__shared__ float sx[2816]` instead of writing/re-reading `g->tmp` in global memory.
   * Then the 256 threads in block `(e, row)` already have the exact `split = 8` thread-to-column mapping (`tid = 0..255` at `col = tid * 8` and `tid < 96` at `col = 2048 + tid * 8`), reducing `partial[0..7]` in exact `i = 0..7` order!
   * *(Note on how to reproduce `split = 8` if you ever did run 1 warp per expert:* a single warp can still reproduce `split = 8` bit-for-bit by looping `for (uint32_t part = 0; part < 8; part++)`, computing `s = dot8(col = (part*32 + lane)*8)` plus `if (part < 3) s = dot8(2048 + (part*32 + lane)*8, s)`, doing `__shfl_xor_sync` inside each `part`, and accumulating `value += s` across `part = 0..7`! That gives bit-identity, and would allow e.g. `16` blocks of `8` warps = `128` warps per token row so 16 SMs share the 704 KiB read.)*

2. **Fix the real culprit in [`g4_router_kernel`](file:///data/ds4-gemma4/cuda/gemma4_primitives.cuh#L186-L204) (or fuse it via a last-block atomic retirement counter):**
   * Look at [`g4_router_kernel:186-204`](file:///data/ds4-gemma4/cuda/gemma4_primitives.cuh#L186-L204): after the 128-thread softmax, **`if (t == 0)` runs a single-thread serial top-8 selection loop (`8 × 128 = 1,024` iterations with inner `j < s` scans)**. Across 30 layers, that single-thread serial loop accounts for over half of the router's wall time!
   * Replace the `if (t == 0)` loop with an 8-iteration parallel warp/block argmax that preserves lower-index tie-breaking (`p[x] > best_p || (p[x] == best_p && x < best_idx)`), setting `p[best] = -1.0f` in shared memory after each pick:

```cuda
// Inside g4_router_kernel (blockDim.x == 128, t = threadIdx.x):
// After const float e = (float)exp((double)(local - maximum)); and sum reduction in scratch[0]:
const float prob = e / scratch[0];
p[t] = prob;
__shared__ float picked_p[G4_REF_USED];
__shared__ uint32_t s_idx[G4_REF_EXPERTS];
__syncthreads();
for (uint32_t s = 0; s < k; s++) {
    scratch[t] = p[t];
    s_idx[t] = t;
    __syncthreads();
    for (unsigned stride = 64u; stride; stride >>= 1u) {
        if (t < stride) {
            const float a = scratch[t], b = scratch[t + stride];
            const uint32_t ia = s_idx[t], ib = s_idx[t + stride];
            if (b > a || (b == a && ib < ia)) {
                scratch[t] = b;
                s_idx[t] = ib;
            }
        }
        __syncthreads();
    }
    if (t == 0u) {
        picked[s] = s_idx[0];
        picked_p[s] = scratch[0];
        p[s_idx[0]] = -1.0f;
    }
    __syncthreads();
}
if (t == 0u) {
    float sum = 0.0f;
    for (uint32_t s = 0; s < k; s++) { sum += picked_p[s]; }
    uint32_t *out_index = selected + (uint64_t)row * k;
    float *out_weight = weights + (uint64_t)row * k;
    for (uint32_t s = 0; s < k; s++) {
        const uint32_t best = picked[s];
        out_index[s] = best;
        out_weight[s] = (picked_p[s] / sum) * g4_round(__bfloat162float(per_expert[best]));
    }
}
```

#### Is the Router Optimization Worth It?
* **Yes, with the multi-block shape above (not 1-block-per-row):**
  * Eliminating the single-thread `8 × 128` top-8 loop in `g4_router_kernel` + fusing `g4_router_input_kernel` into the multi-block router dot cuts the router from ~45 µs/layer (1.36 ms/token, 3.4%) to **~12–15 µs/layer (~0.4 ms/token)**—recovering **~0.9–1.0 ms/token (~2.4% end-to-end decode speedup)** with **zero numerical risk** (100% bit-identical `router_logits`, `selected`, and `weights`).
  * Keep the fused dot path gated on `rows == 1 || gemma4_exact_prefill()` (or `rows <= 4`) so `rows > 4` prefill continues to use the batched `cublasGemmEx` path in `gemma4_projection_rows`.
