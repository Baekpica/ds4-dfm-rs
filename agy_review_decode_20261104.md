# Gemma-4 Batch-1 Decode Architecture & Roofline Review (`2026-11-04`, `feat/gemma4-family`)

Independent read-only audit of the Gemma-4 26B A4B (`G426`) batch-1 decode path in `/data/ds4-gemma4` (`feat/gemma4-family`) on the Spark GB10 (`sm_121`, 48 SMs, 2.4 GHz, measured `240 GB/s` DRAM bandwidth, `ctx = 262,144`, BF16 KV, 8 fast options + tensor-core prefill MMA + split-K decode, no MTP).

---

## 0. GB10 Decode Roofline Decomposition (`ctx = 256` vs `ctx = 13,447`)

At `240 GB/s`, streaming the **2.33 GB** of active weights touched by one token (`1.476 GB` BF16 LM head + `472 MB` R13+8 routed experts + `369 MB` Q4_K projections + `12 MB` Q8_0 dense down) has a pure bandwidth floor of **9.71 ms/step (`103.0 tok/s`)**. Measured decode is **38.59 ms (`25.9 tok/s`)** at `ctx = 256` and **51.21 ms (`19.5 tok/s`)** at `ctx = 13,447`—**4.0× to 5.3× above the weight-streaming floor**:

| Bucket | Active Kernel(s) | Launches / step | `ctx 256` (ms) | `ctx 13,447` (ms) | Bytes / FLOPs per Step | GB10 Floor (`240 GB/s` / Peak) | Effective BW / Efficiency | Root Bottleneck |
|---|---|---:|---:|---:|---|---:|---|---|
| **1. Attention** | [`g4_attn_split_kernel<16>`](file:///data/ds4-gemma4/cuda/gemma4_attn_split.cuh#L49-L133) + [`g4_attn_split_combine_kernel`](file:///data/ds4-gemma4/cuda/gemma4_attn_split.cuh#L136-L202) | `60` | `9.291` | `19.275` | `256`: `53 MB` KV<br>`13.4K`: `485 MB` KV (`210 MB` SWA + `275 MB` full), `1.31G FMA` (`2.62 GFLOP`) | `0.22 ms` (`256` BW)<br>`2.02 ms` (`13.4K` BW)<br>`16.5 ms` (`13.4K` scalar FMA) | `5.7 GB/s` (`256`)<br>`25.2 GB/s` (`13.4K`) | **Launch/occupancy + scalar FMA bound:**<br>- At `ctx 256`, `splits = 1`, yet [`ds4_gemma4_graph.inc:810`](file:///data/ds4-gemma4/ds4_gemma4_graph.inc#L810-L814) still launches `split + combine` (`60` launches of `2` or `8` CTAs).<br>- All 25 sliding layers (`window = 1024`) **always** have `splits = ceil(1024/512) = 2` (`16 CTAs` of `2 warps` = `32 warps` across 48 SMs = **4.2% occupancy**) + 25 combine launches (`~8.0 ms` fixed overhead!).<br>- The 5 full layers at `13.4K` (`kv_heads = 2`, `group = 8`, `D = 512`) run scalar per-lane `fmaf` + 6 warp shuffles + 2 `expf` + 16 rescalings **per key** (`16.5 ms` scalar FMA floor). |
| **2. MoE Gate/Up** | [`g4_vq_gemv_block_kernel<256>`](file:///data/ds4-gemma4/cuda/gemma_vq_decode.cuh#L105-L158) | `480` | `8.587` | `8.710` | `240 × 2 × 656.4 KB = 315.1 MB` weights + `30 × 64 KiB` books | `1.31 ms` | `36.2 GB/s` (**15.1%**) | **Launch + SMEM/L1 thrash + tiny per-thread work:**<br>- `704 blocks × 256 threads` per matrix: each thread decodes **1.375 vectors** (`352 / 256`), then pays `__syncthreads()` + SMEM warp reduction.<br>- Uncached `64 KiB` codebook gathered via 1-byte scalar loads; `x[2816]` re-read in FP32 and converted to BF16 per vector (`480×` per token). |
| **3. MoE Down + Loop** | [`g4_vq_gemv_warp_kernel<8>`](file:///data/ds4-gemma4/cuda/gemma_vq_decode.cuh#L50-L102) + gather/gelu/scatter + CPU sort | `960` + `30` D2H syncs | `4.119` (+ `~1.7` loop) | `4.195` (+ `~1.7` loop) | `240 × 656.4 KB = 157.5 MB` weights | `0.66 ms` | `37.5 GB/s` (**15.6%**) | **`1,440` launches/token + `30` D2H stream syncs:**<br>- [`g4_moe_run_routes`](file:///data/ds4-gemma4/ds4_gemma4_gpu.cuh#L887-L923) launches `gather, gate, up, gelu, down, scatter` per expert (`48/layer × 30 = 1,440`).<br>- [`ds4_gpu_g4_moe_run`](file:///data/ds4-gemma4/ds4_gemma4_gpu.cuh#L1118-L1142) calls `ds4_gpu_tensor_read(selected)` (`cudaMemcpy` D2H) per layer, draining the stream 30 times/token. |
| **4. LM Head** | [`g4_head_gemv_kernel<2816, true>`](file:///data/ds4-gemma4/cuda/gemma4_head.cuh#L12-L50) | `1` | `6.077` | `6.049` | `262,144 × 2,816 × 2B = 1,476.4 MB` | `6.15 ms` | **`244.1 GB/s` (~100%)** | **At DRAM bandwidth floor.** Cannot be accelerated in batch-1 without quantizing `output.weight` or amortizing via MTP. |
| **5. Q4_K Projections** | [`glmvq_q4_small`](file:///data/ds4-gemma4/cuda/glm_vq_q4.cuh#L27-L69) + [`glmvq_q4_stream`](file:///data/ds4-gemma4/cuda/glm_vq_q4.cuh#L127-L180) | `145` | `5.155` | `5.219` | `369.0 MB` (`Q,K,V,O` + `dense gate,up`) | `1.54 ms` (`240 GB/s`)<br>`1.71 ms` (`216 GB/s` Q4_K peak) | `70.7 GB/s` (**32.7%**) | **Trapped in scalar `glmvq_q4_small` fallback (`<= 4 MiB`):**<br>- In [`glmvq_q4_select`](file:///data/ds4-gemma4/cuda/glm_vq_q4.cuh#L278-L280), `small_weight_bytes = 4 MiB`.<br>- Gemma's `k_proj` (`3.09 MiB` SWA, `1.55 MiB` full), `v_proj` (`3.09 MiB`), `dense_gate` (`3.19 MiB`), and `dense_up` (`3.19 MiB`) **all sit under `4 MiB`** (`115 of 175` matrices = `191 MB`), so they fall back to scalar per-element `ds4_q4_k_value` in `glmvq_q4_small`! |
| **6. Norms & Rest** | [`g4_rms_kernel<256>`](file:///data/ds4-gemma4/cuda/gemma4_primitives.cuh#L39-L73) (`1.88 ms`), `dense down Q8` (`0.94 ms`), `gelu` (`0.93 ms`), rest (`1.70 ms`) | `~390` | `5.359` | `5.767` | `12 MB` Q8_0 + activations | `~0.90 ms` | Latency-bound | `g4_rms_kernel` uses unbatched scalar loads + FP64 `sqrt((double)...)` across `181` launches/token; `gelu` and `add` are unfused. |
| **Total** | | **`~2,036`** | **`38.59 ms` (`25.9 tok/s`)** | **`51.21 ms` (`19.5 tok/s`)** | **`2.38–2.81 GB`** | **`9.9–11.7 ms`** | **`61.7 / 54.9 GB/s`** | **29–39 ms/step of recoverable overhead** |

---

## 1. Question 1: Top Three Decode Actions Ranked by Expected `tok/s` per Effort

### Rank 1: Persistent Layer-Book MoE Decode (`g4_vq_decode_gu` + `g4_vq_decode_down` + `g4_vq_decode_reduce` + Device Route Sort) in [`cuda/gemma_vq_decode.cuh`](file:///data/ds4-gemma4/cuda/gemma_vq_decode.cuh)
- **Target Bottleneck:** `8.59–8.71 ms` (`gate/up`) + `4.12–4.20 ms` (`down`) + `~0.44 ms` (routed `gelu`) + `~1.20 ms` (`30` D2H `selected` syncs + `720` gather/scatter launches) = **`14.35–14.55 ms/step`** (`37.2%` of `ctx 256`, `28.4%` of `ctx 13,447`) vs a **`1.97 ms`** DRAM floor (`472.6 MB` at `240 GB/s`).
- **Mechanism:**
  1. **Eliminate the 30 per-layer D2H `selected` syncs and CPU sort ([`ds4_gemma4_gpu.cuh:1118-1142`](file:///data/ds4-gemma4/ds4_gemma4_gpu.cuh#L1118-L1142)):** When all experts are resident in VRAM (`!m->host_Stream`), launch a 1-warp device kernel (`g4_vq_order_kernel<<<1, 32>>>`, modeled on [`v41_vq_order_kernel`](file:///data/ds4-gemma4/cuda/ds41_vq_decode.cuh#L189-L196)) that reads `selected[8]` on device, sorts the 8 active expert IDs ascending in registers, and populates `gu_items[8]` and `down_items[8]` from a static device descriptor table uploaded once at model load.
  2. **Collapse `1,440` MoE kernel launches/token into `120` (`4` launches/layer):**
     - `g4_vq_decode_gu<<<48, 512>>>` (`48 CTAs = 1 CTA/SM`, `16 warps/CTA`): stages the layer's `64 KiB` main codebook (`m.book`), **both** `2 KiB` residual books (`gate_res_book` + `up_res_book` = `4 KiB`), and the `2,816`-element input row converted once to `BF16` (`5.5 KiB`) in shared memory (**`73.5 KiB` total `< 99 KiB` SMEM limit**). Each warp claims `(expert_slot, row)` tasks across all `8 × 704 = 5,632` rows, decodes `gate` and `up` weights in registers using `uint2` shared-book loads + `__nv_cvt_fp8x2_to_halfraw2` + `__funnelshift_r`, applies `g4_gelu(gate) * up` in registers, and writes `mid[8][704]` (`BF16`) directly.
     - `g4_vq_decode_down<<<48, 512>>>`: stages the `64 KiB` main book + `2 KiB` `down` residual book + all 8 experts' `mid[8][704]` (`11.0 KiB`, **`77.0 KiB` total SMEM**) once per layer, and computes `partial[8][2816]` across `8 × 2,816 = 22,528` row tasks.
     - `g4_vq_decode_reduce<<<11, 256>>>` (modeled on [`glmvq_decode_reduce`](file:///data/ds4-gemma4/cuda/glm_vq_decode.cuh#L388-L401)): accumulates the 8 expert outputs in ascending expert order with `g4_round(acc + partial * weight)`.
- **Why this beats tuning single-matrix GEMV:**
  - Codebook reads drop from `240 × 3 = 720` uncached global/L2 scatter-gather passes (`45 MiB` of random byte gathers) to **60 coalesced `uint4` shared-memory loads per token (`3.84 MB` total)**.
  - Input `x` reads drop from `480` global FP32 reads (`5.4 MB`) to **30 shared-memory loads (`165 KB`)**, and `gather`/`gelu`/`scatter` (`720` launches) disappear completely.
- **Expected Gain:** Drops total MoE step time from `~14.5 ms` to **`~2.5–2.9 ms`** (`472 MB` at `~195–210 GB/s` + `120` launches), saving **`~11.6–12.0 ms/step`**:
  - Alone lifts `ctx 256` from `25.9 -> ~37.3 tok/s` (**`+44%`**) and `ctx 13,447` from `19.5 -> ~25.4 tok/s` (**`+30%`**).

---

### Rank 2: Hybrid Split-K Dispatch, Single-Launch Multi-Warp Sliding Decode, Finer Full-Layer Splits (`128` keys/split), and Full-Layer Tensor-Core `m16n8k16` GQA-8 Decode in [`cuda/gemma4_attn_split.cuh`](file:///data/ds4-gemma4/cuda/gemma4_attn_split.cuh) & [`cuda/gemma4_attn_fast.cuh`](file:///data/ds4-gemma4/cuda/gemma4_attn_fast.cuh)
- **Target Bottleneck:** **`9.291 ms`** at `ctx 256` (`24.1%` of step) and **`19.275 ms`** at `ctx 13,447` (`37.6%` of step) vs a bandwidth floor of **`0.22 ms`** (`ctx 256`) and **`2.02 ms`** (`ctx 13,447`).
- **Mechanism (Three Coordinated Fixes in Gemma Attention):**
  1. **Stop launching `split + combine` on the 25 sliding layers (`window = 1024`, `kv_heads = 8`, `group = 2`, `D = 256`) and at short contexts (`visible_keys <= 1024`):**
     - Currently [`ds4_gemma4_graph.inc:810`](file:///data/ds4-gemma4/ds4_gemma4_graph.inc#L810-L814) calls [`ds4_gpu_g4_attn_split`](file:///data/ds4-gemma4/cuda/gemma4_attn_split.cuh#L204-L237) unconditionally whenever `rows == 1`.
     - Because `G4_SPLIT_STEP = 512`, all 25 sliding layers have `splits = 1` (`ctx <= 512`, **8 CTAs** of 2 warps) or `splits = 2` (`ctx > 512`, **16 CTAs** of 2 warps = `32 warps` on a `2,304`-warp GPU = **1.4% occupancy**), followed by 25 separate `g4_attn_split_combine_kernel` launches! That costs **`~8.0 ms`** across the 25 sliding layers even though their total KV footprint at `window = 1024` is only `25 × 8.39 MB = 209.7 MB` (**`0.87 ms` at `240 GB/s`**, **`0.14 ms` FMA floor**).
     - **Fix:** For sliding layers (`window > 0` or `group == 2`), launch **one single CTA per KV head (`8 CTAs/layer`) with `8 warps/CTA` (`4` key-splits × `2` query heads)** inside [`g4_attn_fast_kernel`](file:///data/ds4-gemma4/cuda/gemma4_attn_fast.cuh) (or an in-CTA split-K specialization `g4_attn_swa_decode_kernel`), reducing the 4 warps' `(m, l, o[256])` in shared memory (`4 × 2 × 256 × 4B = 8 KiB` SMEM) at the end of the CTA with **zero combine kernel and zero global workspace writes**.
  2. **Use finer splits (`128` keys/split, up to `64` splits) on the 5 full layers (`kv_heads = 2`, `group = 8`, `D = 512`):**
     - With `kv_heads = 2` and `G4_SPLIT_STEP = 512`, full layers launch only `2 × splits` CTAs (`2..48 CTAs` for `ctx = 512..12,288`), leaving many of GB10's 48 SMs idle at medium context and capping at `2 × 32 = 64 CTAs` (`1.33 CTAs/SM`, `10.6 warps/SM = 22% occupancy`) above `16K`.
     - Lowering the full-layer split step to `128` keys (`splits = clamp(ceil(keys / 128), 1, 96)`) produces `48..192 CTAs` from `ctx = 3,072` upward, filling all 48 SMs evenly.
  3. **Replace scalar per-key FMA + warp shuffle on the 5 full layers with Tensor-Core `mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32` (`group = 8` query heads packed into `M = 0..7`):**
     - In [`g4_attn_split_kernel<16>`](file:///data/ds4-gemma4/cuda/gemma4_attn_split.cuh#L94-L113), every single key executes `16` scalar FMAs (`QK`), `6` dependent `__shfl_down_sync`/`__shfl_sync` reductions, `2` `expf` calls, `16` scalar multiplies (`acc[i] *= alpha`), and `16` scalar FMAs (`acc[i] += p * V[i]`), with a `__syncthreads()` every `16` keys. At `ctx = 13,447`, the 5 full layers execute `1.10G` useful FMAs (`2.20 GFLOP`) plus `1.10G` rescaling multiplies and `8.6M` warp shuffles—taking **`~11.3 ms`** out of the `19.28 ms` total.
     - Packing the `group = 8` query heads of one KV head into rows `0..7` of `m16n8k16` (`50%` M-utilization) and tiling `TK = 32` keys per step executes both $Q K^T$ (`[8, 512] × [32, 512]^T`) and $P V$ (`[8, 32] × [32, 512]`) on tensor cores (**`112×` fewer warp instructions issued**, online softmax rescaling `o *= alpha` **once per 32 keys** instead of once per key). Full-layer compute time at `13.4K` drops from `~11.3 ms` to `< 0.15 ms`, leaving the 5 full layers purely bandwidth-bound on their `275.4 MB` of KV cache (**`1.15 ms` floor -> `~1.4–1.7 ms` measured**).
- **Expected Gain:**
  - At `ctx 256`: drops attention from `9.29 ms` to **`~1.4–1.8 ms`** (saving **`~7.5–7.9 ms/step`**).
  - At `ctx 13,447`: drops attention from `19.28 ms` (`~8.0 ms` SWA + `~11.3 ms` full) to **`~2.8–3.5 ms`** (`~1.4 ms` SWA + `~1.6 ms` full MMA), saving **`~15.8–16.5 ms/step`**.

---

### Rank 3: Bypass the `<= 4 MiB` `glmvq_q4_small` Fallback and Fuse Multi-Matrix `Q4_K` Decode (`Q+K+V` and `dense_gate+up`) in a Gemma-Local Wrapper
- **Target Bottleneck:** **`5.155–5.219 ms/step`** across `145` `Q4_K` launches (`369.0 MB` of weights, running at only **`70.7 GB/s`** vs GB10's measured **`216 GB/s`** `Q4_K` streaming peak in [`AGENTS.md`](file:///data/ds4-gemma4/AGENTS.md#L234) = `1.71 ms` floor).
- **Why It Happens (Exact Code Root Cause):**
  1. In [`gemma4_projection_rows`](file:///data/ds4-gemma4/ds4_gemma4_graph.inc#L574-L578), `ds4_gpu_gvq_linear_many` is called per matrix (`count = 1`), and at `rows == 1` [`glmvq_q4_select`](file:///data/ds4-gemma4/cuda/glm_vq_q4.cuh#L278-L280) routes any matrix with `weight_bytes <= 4 MiB` to `GLMVQ_Q4_SCALAR` ([`glmvq_q4_small`](file:///data/ds4-gemma4/cuda/glm_vq_q4.cuh#L27-L69)):
     ```cuda
     constexpr uint64_t small_weight_bytes = 4ull << 20;
     if (config->mode == GLMVQ_Q4_AUTO && tokens == 1 && weight_bytes <= small_weight_bytes) {
         return GLMVQ_Q4_SCALAR;
     }
     ```
  2. In Gemma-4 (`cols = 2816 = 11 × 256` = `1,584 B/row`), **5 of the 7 `Q4_K` matrix shapes sit just under `4.00 MiB`**:
     - Sliding `k_proj` (`2048 × 2816`): **`3.09 MiB <= 4 MiB`** $\rightarrow$ `glmvq_q4_small`!
     - Sliding `v_proj` (`2048 × 2816`): **`3.09 MiB <= 4 MiB`** $\rightarrow$ `glmvq_q4_small`!
     - Full `k_proj` (`1024 × 2816`): **`1.55 MiB <= 4 MiB`** $\rightarrow$ `glmvq_q4_small`!
     - `dense_gate` (`2112 × 2816`): **`3.19 MiB <= 4 MiB`** $\rightarrow$ `glmvq_q4_small`!
     - `dense_up` (`2112 × 2816`): **`3.19 MiB <= 4 MiB`** $\rightarrow$ `glmvq_q4_small`!
     - **115 of the 145 `Q4_K` matrices per token (`191 MB` of `369 MB`) execute [`glmvq_q4_small`](file:///data/ds4-gemma4/cuda/glm_vq_q4.cuh#L27-L69)**, which calls scalar `ds4_q4_k_value(packed, j + lane)` in a loop (re-reading and bit-unpacking the 12-byte scale header from global/L1 on every single element with no shared-memory `x` staging)!
  3. Furthermore, `(q_proj, k_proj, v_proj)` and `(dense_gate, dense_up)` share the exact same normalized input `norm`, yet [`gemma4_run_attn_block`](file:///data/ds4-gemma4/ds4_gemma4_graph.inc#L774-L785) and [`gemma4_run_ffn_block`](file:///data/ds4-gemma4/ds4_gemma4_graph.inc#L890-L894) launch them as separate kernels (`145` launches/token).
- **Mechanism (Strictly in Gemma-Only Files):**
  - Create `cuda/gemma4_q4_decode.cuh` (copied from [`glmvq_q4_stream_many`](file:///data/ds4-gemma4/cuda/glm_vq_q4.cuh#L183-L243) + [`glmvq_q4_acc`](file:///data/ds4-gemma4/cuda/glm_vq_q4.cuh#L73-L125)) and wire it at `rows == 1`:
    1. Always use the vectorized `uint4`-header + `uint32_t`-quants + shared-memory `x[2816]` staged kernel (`8 warps/block = 8 rows/block`, `0` scalar `glmvq_q4_small` fallbacks).
    2. Fuse `(q_proj, k_proj, v_proj)` into a single multi-matrix launch per layer (`30` launches instead of `85`), and fuse `(dense_gate, dense_up)` + `g4_gelu(gate) * up` into a single launch per layer (`30` launches instead of `90` launches for `gate + up + gelu`).
- **Expected Gain:**
  - Moves all `369 MB` of `Q4_K` weights onto the `~200–216 GB/s` vectorized streaming path (`~1.75–1.95 ms`) and eliminates `85` kernel launches + `30` dense `gelu` launches (`~0.44 ms`), saving **`~3.6–3.8 ms/step`** for `< 150` lines of Gemma-local code.

---

### Summary of All Candidate Levers & Combined Impact

| Rank | Action | Target Bucket | Effort | Saved at `ctx 256` | Saved at `ctx 13,447` | Numerical Class |
|---:|---|---|---|---:|---:|---|
| **1** | **Persistent Layer-Book MoE Decode + Device Route Sort** (`cuda/gemma_vq_decode.cuh`) | MoE `gate/up` (`8.7 ms`) + `down` (`4.2 ms`) + `gelu`/syncs (`1.7 ms`) | Medium (`~250 LOC`) | **`-11.6 ms`** | **`-11.8 ms`** | **0 ULP** (if `GLMVQ_GAIN_WEIGHT` + F32 dot order kept) or **Class B** (reorder K blocks) |
| **2** | **Attention Dispatch Guard + Single-CTA SWA Split-K + Finer Full Splits + Full-Layer `m16n8k16` MMA** (`cuda/gemma4_attn_split.cuh`, `cuda/gemma4_attn_fast.cuh`) | Split-K attention (`9.29 ms` / `19.28 ms`) | Medium (`~220 LOC`) | **`-7.6 ms`** | **`-16.0 ms`** | **0 ULP** (SWA dispatch) / **Class B** (`m16n8k16` BF16 on full layers) |
| **3** | **Vectorized Multi-Matrix `Q4_K` Decode (`Q+K+V` & `dense_gate+up+gelu`) Bypassing `glmvq_q4_small`** (`cuda/gemma4_q4_decode.cuh`) | `Q4_K` (`5.22 ms`) + dense `gelu` (`0.44 ms`) | Low (`~120 LOC`) | **`-3.7 ms`** | **`-3.7 ms`** | **0 ULP** (`glmvq_q4_acc` matches `ds4_q4_k_value` algebraically) |
| *4* | **Vectorized `float4` + `rsqrtf` + Fused Residual-Add RMSNorm** ([`cuda/gemma4_primitives.cuh:39-73`](file:///data/ds4-gemma4/cuda/gemma4_primitives.cuh#L39-L73)) | `g4_rms_kernel<256>` (`1.65–1.88 ms`) + `g4_add_kernel` (`~0.35 ms`) | Trivial (`~50 LOC`) | `-1.3 ms` | `-1.5 ms` | **Class A** (`rsqrtf` vs `sqrt((double))`) |
| *—* | **LM Head (`g4_head_gemv_kernel<2816, true>`)** | `6.05–6.08 ms` (`1.476 GB` BF16) | — | `0.0 ms` (at `244 GB/s` floor) | `0.0 ms` (at `244 GB/s` floor) | Leave untouched for batch-1 |
| **Total** | **Top 3 Actions Combined** | | | **`38.59 -> ~15.7 ms` (`25.9 -> ~63.7 tok/s`, `+146%`)** | **`51.21 -> ~19.7 ms` (`19.5 -> ~50.8 tok/s`, `+160%`)** | |

---

## 2. Question 2: The Right Batch-1 Expert GEMV Shape on GB10 (`sm_121`, 48 SMs)

### 2.1 Quantitative Comparison of the Three Candidate Shapes

For each layer, batch-1 MoE executes `top_k = 8` routed experts out of `128`, where each expert has:
- `gate` and `up`: `rows = 704`, `cols = 2,816` (`352` vectors of 8 weights = `11` groups of `32` vectors; `656.4 KB`/matrix).
- `down`: `rows = 2,816`, `cols = 704` (`88` vectors of 8 weights = `2` groups of `32` vectors + `24`-vector tail; `656.4 KB`/matrix).
- Layer-shared main codebook: `8,192 × 8 B = 65,536 B` (`64 KiB` E4M3).
- Residual codebooks (`GVR8`): `256 × 8 B = 2,048 B` (`2 KiB` E4M3) per projection (`gate`, `up`, `down`).

| Metric | Option A: Current Block-per-Row (`g4_vq_gemv_block_kernel<256>`) | Option B: One Warp per Row, `x` in Registers (`G4_VQ_WARP` / `ds41_vq_row`) | Option C (Winner): **Persistent CTA Over All 8 Experts**, 1 Warp per `(expert, row)`, Books + `BF16 x` in Shared Memory |
|---|---|---|---|
| **Grid / Launches per Layer** | `704 blocks × 256 threads` per matrix $\rightarrow$ **`48` launches/layer (`1,440`/token)** | `88` or `352 blocks × 256 threads` per matrix $\rightarrow$ **`48` launches/layer (`1,440`/token)** | **`48 CTAs × 512 threads` (`1 CTA/SM`, `16 warps/CTA`) $\rightarrow$ `3` launches/layer (`90`/token)** (`gu`, `down`, `reduce`) |
| **Vectors Decoded per Thread** | `352 / 256 =` **`1.375` vectors/thread** (then `__syncthreads()` + SMEM reduction!) | `352 / 32 = 11` (`gate/up`) or `88 / 32 = 2.75` (`down`) vectors/lane | `11` (`gate/up`, fused for **both** matrices = `22` vectors/lane) and `2.75` (`down`) across `7.3` / `29.3` rows per warp |
| **64 KiB Main Codebook Traffic** | Re-gathered from L1/L2 in 1-byte scalar loads across `480 + 240 = 720` launches (`45 MiB` demand) | Re-gathered from L1/L2 across `720` launches; `64 KiB` book + `x` + L1 lines evict each other | **Loaded once per CTA per kernel (`2 × 48 × 64 KiB = 6 MiB/layer` from L2, `128 KB/layer` from DRAM)**; read via `uint2` LDS (`128 B/cycle/SM`) with **100% hit rate** |
| **Input `x` Storage & Traffic** | Global FP32 re-read `704 × 16 = 11,264` times/layer (`g4_round(x[col])` per vector) | Holding `x[2816]` in registers takes **`44` `bf16x2` regs (`88` FP32 regs)/lane**, blowing the `64-reg` limit and spilling to local memory! | Staged once in shared memory as `BF16`: **`5,632 B` (`5.5 KiB`)** for `gate+up`, **`11,264 B` (`11.0 KiB`)** (`mid[8][704]`) for `down` |
| **Gate + Up Fusion** | Separate launches; `x` read twice; `gelu` is a 3rd launch | Separate or paired per-expert launches; still requires `gather` and `scatter` launches | **Fused in-register:** same warp loads `x_s[col]` once, decodes `gate` and `up` row `r`, writes `bf16(g4_gelu(gate) * up)` directly to `mid[e][r]` |

### 2.2 Exact Shared-Memory, Register, and Occupancy Numbers on GB10 (`sm_121`)

GB10 (`sm_121`) provides **`101,376 B` (`99 KiB`)** opt-in shared memory per block, **`65,536` 32-bit registers per SM**, and **`48` warps (`1,536` threads) max per SM** ([`docs/gb10-hardware.md`](file:///data/ds4-gemma4/docs/gb10-hardware.md#L12-L21)). Because a `64 KiB` codebook requires `> 49.5 KiB` of shared memory, **every codebook-staging kernel runs at exactly `1 CTA per SM` (`48 CTAs` across the GPU)**.

1. **Shared Memory Budget per CTA (`1 CTA/SM`):**
   - **Fused `g4_vq_decode_gu` (`gate + up + gelu` across all 8 active experts):**
     $$\underbrace{65,536\text{ B}}_{\text{main book } (8192 \times 8\text{B})} + \underbrace{2,048\text{ B}}_{\text{gate res book}} + \underbrace{2,048\text{ B}}_{\text{up res book}} + \underbrace{5,632\text{ B}}_{x_{\text{bf16}}[2816]} = \mathbf{75,264\text{ B}\ (73.50\text{ KiB})} \le 101,376\text{ B}$$
     *(Note: In [`ds4_gemma4_gpu.cuh:667-708`](file:///data/ds4-gemma4/ds4_gemma4_gpu.cuh#L667-L708), `gate` and `up` share the layer's `64 KiB` main book `m->book` and either share a stack residual book or have individual `2 KiB` residual books; staging both takes only `4 KiB`.)*
   - **Persistent `g4_vq_decode_down` (`down` across all 8 active experts):**
     $$\underbrace{65,536\text{ B}}_{\text{main book } (8192 \times 8\text{B})} + \underbrace{2,048\text{ B}}_{\text{down res book}} + \underbrace{11,264\text{ B}}_{\text{mid}_{\text{bf16}}[8][704]} = \mathbf{78,848\text{ B}\ (77.00\text{ KiB})} \le 101,376\text{ B}$$

2. **CTA Thread Count, Register Budget, and Occupancy (`16 Warps = 512 Threads` vs `24 Warps = 768 Threads`):**
   - With `16 warps/CTA` (`__launch_bounds__(512, 1)`):
     - **Active warps/SM:** `16 / 48 =` **`33.3%` occupancy** (`768` warps active across 48 SMs).
     - **Register ceiling per thread:** `65,536 / 512 =` **`128 registers/thread`** (zero register spills; compare [`cuda/minimax_decode.cuh:66`](file:///data/ds4-gemma4/cuda/minimax_decode.cuh#L66): *"One-row R13+8 uses 16 warps to remove the 64-register launch cap's spills"*).
     - **Work balance across `768` warps (`48 CTAs × 16 warps`):**
       - `g4_vq_decode_gu`: `8 experts × 704 rows = 5,632` fused `(gate, up)` row-pairs $\rightarrow$ **`7.33` row-pairs per warp** (`7` warps waves + `256` warps on wave 8).
       - `g4_vq_decode_down`: `8 experts × 2,816 rows = 22,528` rows $\rightarrow$ **`29.33` rows per warp**.
   - With `24 warps/CTA` (`__launch_bounds__(768, 1)`):
     - **Active warps/SM:** `24 / 48 =` **`50.0%` occupancy** (`1,152` warps across 48 SMs), **`84 registers/thread`** max (`4.89` `gu` row-pairs/warp, `19.56` `down` rows/warp). Both `16` and `24` warps are template instantiations (`template <int WARPS>`) gated by an env knob; `16 warps` is the proven default in [`cuda/minimax_decode.cuh`](file:///data/ds4-gemma4/cuda/minimax_decode.cuh#L66).

3. **Coalesced 32-Lane Block Unpacking for Gemma's `2816` (`gate/up`) and `704` (`down`) Widths:**
   - **`gate/up` (`cols = 2,816 = 352 vectors = 11 × 32 vectors`):**
     - Each row has `352 × 12 bits = 528 B` (`132` `uint32_t` words) of `main`, `44 B` (`11` `uint32_t` words) of `plane`, and `352 B` (`88` `uint32_t` words) of `res`.
     - Every row is **4-byte `uint32_t`-aligned** across all three streams (`132`, `11`, `88` words/row)!
     - Decompose `352` vectors into **1 full 8-step block (`2,048 cols`: `96` main words, `8` plane words, `64` res words)** + **1 tail 3-step block (`768 cols`: `36` main words, `3` plane words, `24` res words)**—the exact `head + tail` pattern used by [`mm_down_dot`](file:///data/ds4-gemma4/cuda/minimax_decode.cuh#L20-L58) for MiniMax's `3,072 = 2,048 + 1,024` width.
   - **`down` (`cols = 704 = 88 vectors = 2 × 32 + 24 vectors`):**
     - Each row has `132 B` (`33` `uint32_t` words) of `main`, `88 B` (`22` `uint32_t` words) of `res`, and `11 B` (`88 bits`) of `plane`.
     - Because `11 B` is not a multiple of 4 bytes, `plane` is read via `uint32_t` unaligned load (`__ldg` / byte-offset shift `bit_offset = (row * 88 + j)`) while `main` (`33` words/row) and `res` (`22` words/row) are 4-byte aligned! Each warp runs `3` iterations (`step = 0, 1, 2` for vectors `0..31`, `32..63`, and `64..87` with `lane < 24` predicated on `step = 2`).

---

## 3. Question 3: Tensor-Core Decode Attention Efficiency Math (`group = 2` Sliding vs `group = 8` Full)

On GB10 (`sm_121`), the native BF16 tensor-core instruction is `mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32` (`M = 16, N = 8, K = 16`). At batch-1 decode (`rows = 1`), the only queries that share a single KV head and can occupy distinct rows of the `M = 16` MMA tile are the `group = heads / kv_heads` query heads of that KV group:

| Property | **25 Sliding Layers** (`window = 1024`) | **5 Full Layers** (`window = 0`, `ctx = 13,447`) |
|---|---|---|
| **Geometry** | `heads = 16, kv_heads = 8`, **`group = 2`**, `D = 256` | `heads = 16, kv_heads = 2`, **`group = 8`**, `D = 512` |
| **Max Visible Keys per Layer** | `min(ctx, 1024) = 1,024` | `ctx = 13,447` (up to `262,144`) |
| **KV Bytes Read per Layer / Total** | `8 × 1024 × 2 × 256 × 2B = 8.39 MB/layer` (**`209.7 MB` across 25 layers**) | `2 × 13,447 × 2 × 512 × 2B = 55.08 MB/layer` (**`275.4 MB` across 5 layers**) |
| **DRAM Bandwidth Floor (`240 GB/s`)** | **`0.87 ms` total** (`35 µs/layer`) | **`1.15 ms` total** (`230 µs/layer`) |
| **Useful FMAs (`QK^T + PV`)** | `25 × 16 × 2 × 1024 × 256 =` **`209.7M FMA`** (`0.42 GFLOP`) | `5 × 16 × 2 × 13,447 × 512 =` **`1,101.6M FMA`** (`2.20 GFLOP`) |
| **Arithmetic Intensity (FLOP / KV Byte)** | **`2.0 FLOP/B`** (`1.0 FMA/B`) | **`8.0 FLOP/B`** (`4.0 FMA/B`) |
| **MMA `M = 16` Tile Utilization (`group / 16`)** | **`2 / 16 = 12.5%`** (`87.5%` wasted tensor-core lanes!) | **`8 / 16 = 50.0%`** (rows `0..7` active, `8..15` zero) |
| **Does Tensor-Core MMA Pay Here?** | **NO — Do NOT use MMA on sliding decode.** | **YES — Huge win (`~9.5 ms` saved at `13.4K`, scales to `262K`).** |

### 3.1 Why Full Layers (`group = 8`, `D = 512`) Pay Heavily for Tensor Cores at `>= 2K` Context
1. **Why the current scalar full-layer kernel takes `~11.3 ms` at `13,447` keys:**
   - Look at the inner key loop of [`g4_attn_split_kernel<16>`](file:///data/ds4-gemma4/cuda/gemma4_attn_split.cuh#L94-L113):
     - For **each individual key** `t = 0..15`, each warp (1 warp per query head, `8 warps/CTA`) executes:
       1. `16` scalar `fmaf` instructions for $q \cdot k_t$ (`512 / 32 = 16`),
       2. **5 dependent `__shfl_down_sync` + 1 `__shfl_sync`** in [`g4_split_warp_sum`](file:///data/ds4-gemma4/cuda/gemma4_attn_split.cuh#L12-L21) (`~24–30 cycles` of serial warp-shuffle latency per key!),
       3. `2` `expf` calls (`alpha` and `p`),
       4. **`16` scalar multiplies** to rescale `acc[0..15] *= alpha` **on every key**, plus **`16` scalar `fmaf`** for `acc[0..15] += p * v_t[0..15]`,
       5. `__syncthreads()` every `16` keys!
     - Across `32` keys, the `8` warps of one CTA issue **`8 warps × 32 keys × (16 FMA + 6 shfl + 2 expf + 16 mul + 16 FMA) = 14,336` warp instructions**, and at `54 CTAs` (`1.125 CTAs/SM = 9 warps/SM = 18.75%` occupancy) the serial shuffle + barrier chain stalls the FP32 pipelines near the ~16.5 ms scalar floor.
2. **What happens with `mma.m16n8k16` on the 5 full layers (`group = 8`, `TK = 32` keys per tile):**
   - Pack the `8` query heads of one KV head into rows `m = 0..7` of an `m16n8k16` fragment (with rows `8..15` unused = **50% M-utilization**).
   - **Step A ($S = Q_{8 \times 512} K_{32 \times 512}^T$):** Across `TK = 32` keys (`4` `n8` tiles) and `D = 512` (`32` `k16` steps), `32 × 4 = 128` `mma.m16n8k16` instructions compute all `8 × 32 = 256` attention scores without a single warp-shuffle reduction!
   - **Step B (Online Softmax Once per 32 Keys):** Each row `m = 0..7` updates its running max $m_{\text{new}} = \max(m_{\text{old}}, \max_{j < 32} S_{m,j})$, rescales the running output accumulator `O[8, 512]` by $\alpha = \exp(m_{\text{old}} - m_{\text{new}})$ **once per 32 keys** (**32× fewer rescalings** than `g4_attn_split_kernel`), and packs $P_{8 \times 32}$ to `BF16`.
   - **Step C ($O \mathrel{+}= P_{8 \times 32} V_{32 \times 512}$):** `2` `k16` steps × `64` `n8` tiles = `128` `mma.m16n8k16` instructions update all `8 × 512` output elements.
   - **Instruction & Time Math:** `256` MMA instructions per 32-key tile across 4 warps/CTA (`64` MMAs/warp) replaces `14,336` scalar/shuffle warp instructions (**`56×` fewer instructions**). At 50% M-utilization (`~54 TFLOP/s` effective on GB10), the `2.20 GFLOP` across all 5 full layers at `13.4K` takes **`0.04 ms`** of tensor-core pipe time, so the 5 full layers drop from `~11.3 ms` to **`~1.4–1.7 ms`** (purely DRAM-bandwidth bound on the `275.4 MB` of KV cache).

### 3.2 Why Sliding Layers (`group = 2`, `D = 256`, `window = 1024`) Should Use Single-CTA Multi-Warp Scalar Split-K Instead
- On the 25 sliding layers, `group = 2` wastes `14 / 16 = 87.5%` of an `m16n8k16` tile, and the total FMA work across all 25 layers is only `209.7M FMA` (**`0.14 ms`** on scalar FP32 cores).
- The entire reason sliding layers cost `~8.0 ms` today is that [`g4_attn_split_kernel`](file:///data/ds4-gemma4/cuda/gemma4_attn_split.cuh#L49-L133) launches **only `16` CTAs of `2` warps (`32` warps total across 48 SMs = `1.4%` occupancy)** plus a second `g4_attn_split_combine_kernel` launch on every layer (`50` launches total).
- **Optimal Sliding Kernel (`g4_attn_swa_decode_kernel` in [`cuda/gemma4_attn_fast.cuh`](file:///data/ds4-gemma4/cuda/gemma4_attn_fast.cuh)):**
  - Launch **`8` CTAs (`1` per KV head) × `8` warps (`256` threads) per CTA**: `4` key-split groups × `2` query heads (`group = 2`), where each split group covers at most `1024 / 4 = 256` keys (`16` tiles of `16` keys).
  - At the end of the 256-key loop, the 4 split warps for each query head merge their `(m_s, l_s, o_s[256])` states directly in `8 KiB` of shared memory and write the final BF16 output row—**1 launch per layer (`25` total instead of `50`), zero global `split_out`/`split_lse` traffic, and `~1.1–1.4 ms` total across all 25 layers**.

---

## 4. Question 4: Sibling Kernel Reference Map — What to Copy and What to Change

| Sibling File | Verdict for Gemma-4 Decode | Exactly What to Copy (with `file:line`) | Exactly What to Adapt for Gemma-4 (`G426`) |
|---|---|---|---|
| **[`cuda/minimax_decode.cuh`](file:///data/ds4-gemma4/cuda/minimax_decode.cuh)** + **[`cuda/glm_vq_decode.cuh`](file:///data/ds4-gemma4/cuda/glm_vq_decode.cuh)** | **PRIMARY REFERENCE** for Rank 1 (Persistent MoE Decode) | 1. Persistent 16-warp CTA skeleton from [`mm_decode_gu`](file:///data/ds4-gemma4/cuda/minimax_decode.cuh#L67-L104) (`__launch_bounds__(16 * 32, 1)`, `48` CTAs, `13`-bit + `RES = true`).<br>2. Multi-part `head + tail` row dot from [`mm_down_dot`](file:///data/ds4-gemma4/cuda/minimax_decode.cuh#L20-L58).<br>3. Shared-book loader [`glmvq_decode_book<13, true>`](file:///data/ds4-gemma4/cuda/glm_vq_decode.cuh#L270-L281) and `uint2` + `__nv_cvt_fp8x2_to_halfraw2` + `__funnelshift_r` unpacker [`glmvq_decode_word8_res<GLMVQ_GAIN_WEIGHT>`](file:///data/ds4-gemma4/cuda/glm_vq_decode.cuh#L118-L134) & [`glmvq_decode_block_res`](file:///data/ds4-gemma4/cuda/glm_vq_decode.cuh#L205-L267).<br>4. Expert reduction [`glmvq_decode_reduce`](file:///data/ds4-gemma4/cuda/glm_vq_decode.cuh#L388-L401). | 1. **Dimensions:** `gate/up` has `cols = 2816 = 2048 (8-step head) + 768 (3-step tail)` and `rows = 704`; `down` has `cols = 704 = 512 (2-step head) + 192 (24-lane tail)` and `rows = 2816`.<br>2. **Activation:** Replace `glmvq_silu(gate) * up` with [`g4_gelu(gate) * up`](file:///data/ds4-gemma4/cuda/gemma4_primitives.cuh#L23-L31).<br>3. **Dual Residual Books in `gu`:** Stage both `gate_res_book[256]` (`2 KiB`) and `up_res_book[256]` (`2 KiB`) in SMEM.<br>4. **Down `plane` offset:** In `down` (`88` vectors/row), `main` (`33` words/row) and `res` (`22` words/row) are 4-byte aligned, while `plane` is `11 B/row`; read `plane` via byte-addressed `uint32_t` shift `(row * 11 + step * 4)`. |
| **[`cuda/ds41_vq_decode.cuh`](file:///data/ds4-gemma4/cuda/ds41_vq_decode.cuh)** & **[`cuda/ds41_vq_persist.cuh`](file:///data/ds4-gemma4/cuda/ds41_vq_persist.cuh)** | **USE FOR DEVICE ROUTE SORT ONLY** (do NOT copy `v41_vq_stream`) | Copy the 1-CTA device route sort pattern from [`v41_vq_order_kernel`](file:///data/ds4-gemma4/cuda/ds41_vq_decode.cuh#L189-L196) to sort `selected[8]` on the GPU and emit `g4_decode_item[8]` without `ds4_gpu_tensor_read(selected)` ([`ds4_gemma4_gpu.cuh:1118`](file:///data/ds4-gemma4/ds4_gemma4_gpu.cuh#L1118)). | **Do NOT copy [`v41_vq_stream`](file:///data/ds4-gemma4/cuda/ds41_vq_decode.cuh#L36-L86):** `ds41` has no `GVR8` residual stream, applies `gain` after the dot (`GLMVQ_GAIN_DOT`), and its cross-row bitstream (`COLS % 256 != 0`) assumes `plane` is `uint32_t`-streamable without residual alignment. |
| **[`cuda/glm_vq_q4.cuh`](file:///data/ds4-gemma4/cuda/glm_vq_q4.cuh)** | **PRIMARY REFERENCE** for Rank 3 (`Q4_K` Decode Projections) | Copy [`glmvq_q4_acc`](file:///data/ds4-gemma4/cuda/glm_vq_q4.cuh#L73-L125) and [`glmvq_q4_stream_many`](file:///data/ds4-gemma4/cuda/glm_vq_q4.cuh#L183-L243) into a new Gemma-owned `cuda/gemma4_q4_decode.cuh`. | Remove the `weight_bytes <= 4 MiB` scalar fallback ([`cuda/glm_vq_q4.cuh:278`](file:///data/ds4-gemma4/cuda/glm_vq_q4.cuh#L278-L280)) so Gemma's `k_proj`, `v_proj`, `dense_gate`, and `dense_up` (`1.55–3.19 MiB`) all run the staged `uint4` kernel, and add an optional fused `g4_gelu(gate) * up` epilogue for `dense_gate + dense_up`. |
| **[`cuda/gemma4_attn_mma.cuh`](file:///data/ds4-gemma4/cuda/gemma4_attn_mma.cuh)** | **PRIMARY REFERENCE** for Rank 2 (Full-Layer MMA Split-K Decode) | Reuse [`g4_mma_swz`](file:///data/ds4-gemma4/cuda/gemma4_attn_mma.cuh#L39-L41), [`g4_mma_cp16`](file:///data/ds4-gemma4/cuda/gemma4_attn_mma.cuh#L43-L49), [`g4_mma_ldsm4`](file:///data/ds4-gemma4/cuda/gemma4_attn_mma.cuh#L51-L59), [`g4_mma_mma`](file:///data/ds4-gemma4/cuda/gemma4_attn_mma.cuh#L61-L68), and [`g4_mma_pack`](file:///data/ds4-gemma4/cuda/gemma4_attn_mma.cuh#L70-L72). | Specialize a `rows == 1, group == 8, D == 512` split-K kernel (`g4_attn_split_mma_full_kernel`) that packs the `8` query heads of one KV head into `m = 0..7` of `m16n8k16`, loops over `[start, end)` key splits in `TK = 32` tiles, and writes partial `(split_out, split_lse)` for [`g4_attn_split_combine_kernel`](file:///data/ds4-gemma4/cuda/gemma4_attn_split.cuh#L136-L202). |
| **[`cuda/iquest_decode.cuh`](file:///data/ds4-gemma4/cuda/iquest_decode.cuh)** | **Skip** (minor `cp.async` reference only) | [`iq_cp_async_8`](file:///data/ds4-gemma4/cuda/iquest_decode.cuh#L12-L20) shows 8-byte `cp.async` double-buffering in a single-warp attention loop. | iQuest uses `Q8_0` KV + attention sinks (`head_dim = 128`). Gemma's [`g4_mma_cp16`](file:///data/ds4-gemma4/cuda/gemma4_attn_mma.cuh#L43-L49) already provides 16-byte `cp.async.cg`. |
| **`/data/VQ_tools`** (`be8c7e6`, `feature/gvq1-gemv-profile`) | **Do NOT use to solve decode now** (Defer `--vq-profile`) | [`docs/profile-optimizer.md`](file:///data/VQ_tools/docs/profile-optimizer.md) and [`c/gpu/decode-gpu.cu`](file:///data/VQ_tools/c/gpu/decode-gpu.cu) sweep single-matrix `(variant ∈ {WARP, BLOCK}, threads ∈ {128, 256, 512})`. | `vq-tune` only tunes isolated single-expert GEMV calls (`g4_vq_gemv`). It **cannot** fix the `1,440` launches/token, the `30` D2H `selected` syncs, the uncached `64 KiB` codebook re-reads across experts, or the `glmvq_q4_small` fallback. Land Rank 1–3 first. |

---

## 5. Implementation Rules, Env Gates, and Parity Verification

To comply with [`AGENTS.md`](file:///data/ds4-gemma4/AGENTS.md) and the family isolation constraint:
1. **Gemma-Only Files:** Keep all changes inside `cuda/gemma_vq_decode.cuh`, `cuda/gemma4_attn_split.cuh`, `cuda/gemma4_attn_fast.cuh`, `cuda/gemma4_attn_mma.cuh`, `cuda/gemma4_q4_decode.cuh` (new), `ds4_gemma4_gpu.cuh`, and `ds4_gemma4_graph.inc`. Do not touch `cuda/glm_vq_*.cuh` or sibling families.
2. **Default-Off Kill Switches Until Measured:**
   - `DS4_GEMMA4_MOE_PERSIST=1` (Rank 1: persistent `gu` + `down` + `reduce` + device route sort).
   - `DS4_GEMMA4_ATTN_SPLIT_MMA=1` (Rank 2: single-CTA multi-warp SWA decode + 128-key full splits + `m16n8k16` full-layer split-K decode).
   - `DS4_GEMMA4_Q4_STREAM=1` (Rank 3: staged multi-matrix `Q4_K` decode bypassing `glmvq_q4_small`).
3. **Parity & Numerical Gates Before Flipping Defaults:**
   - **Rank 3 (`DS4_GEMMA4_Q4_STREAM=1`):** Pure scheduling/staging change with identical `ds4_q4_k` scale algebra $\rightarrow$ verify `relL2 < 1e-6` per projection and 0 top-1 flips on 64-token greedy generation.
   - **Rank 1 (`DS4_GEMMA4_MOE_PERSIST=1`):** Preserves `GLMVQ_GAIN_WEIGHT` (`w = bf16(gain * (C1 + C2))`) and ascending expert reduction order; changing the 32-lane K-accumulation tree from `block<256>` to 1-warp 32-lane reduction is **Class B** (`relL2 < 1e-4` per layer, frontier KL well below the `512 vs 1024` chunk noise floor).
   - **Rank 2 (`DS4_GEMMA4_ATTN_SPLIT_MMA=1`):** SWA single-CTA split-K is 0-ULP identical to `g4_attn_split` at `splits = 4`; full-layer `m16n8k16` is **Class B** (already verified in prefill Phase 8 with `logit RMS = 0.044`, `KL = 1.98e-5` at `13.5K`). Gate with `nsys` (`gen 8 minus gen 0`) at `ctx = 256` and `ctx = 13,447` on GB10.
