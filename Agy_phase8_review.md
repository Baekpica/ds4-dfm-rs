# Phase 8 Review — Gemma-4 26B A4B VQ Tensor-Core Prefill Attention (`cuda/gemma4_attn_mma.cuh`)

**Date:** 2026-11-03  
**Repository / Branch:** `/data/ds4-gemma4` (`feat/gemma4-family` at `b87664cb`)  
**Reference Commit Inspected:** [`acd8aa49288ef5d1b5f2f4f13b632133980c482e`](https://github.com/JLFN/ds4-dfm-rs/commit/acd8aa49288ef5d1b5f2f4f13b632133980c482e) (`/tmp/g4-acd8aa49.diff`: *"Run GLM-5.3 mixed prefill attention on tensor cores"*)  
**Files Reviewed:**
- `/tmp/g4-acd8aa49.diff`, `/tmp/g4-glm53_flash.cuh`, `/tmp/g4-glm_vq_flash.cuh`, [`cuda/glm_vq_flash.cuh`](file:///data/ds4-gemma4/cuda/glm_vq_flash.cuh)
- [`cuda/gemma4_attn_tile.cuh`](file:///data/ds4-gemma4/cuda/gemma4_attn_tile.cuh), [`cuda/gemma4_attn_fast.cuh`](file:///data/ds4-gemma4/cuda/gemma4_attn_fast.cuh), [`cuda/gemma4_attn.cuh`](file:///data/ds4-gemma4/cuda/gemma4_attn.cuh), [`cuda/gemma4_attn_fp8.cuh`](file:///data/ds4-gemma4/cuda/gemma4_attn_fp8.cuh)
- [`ds4_gemma4_gpu.cuh`](file:///data/ds4-gemma4/ds4_gemma4_gpu.cuh), [`ds4_gemma4_graph.inc`](file:///data/ds4-gemma4/ds4_gemma4_graph.inc), [`ds4_gemma4_mtp.inc`](file:///data/ds4-gemma4/ds4_gemma4_mtp.inc), [`ds4_gemma4_ref.h`](file:///data/ds4-gemma4/ds4_gemma4_ref.h)
- [`docs/gemma4-reference-semantics.md`](file:///data/ds4-gemma4/docs/gemma4-reference-semantics.md), [`docs/gemma4-attention-round-2026-11-03.md`](file:///data/ds4-gemma4/docs/gemma4-attention-round-2026-11-03.md), [`docs/gb10-hardware.md`](file:///data/ds4-gemma4/docs/gb10-hardware.md)
- [`cuda/mimo2_prefill.cuh`](file:///data/ds4-gemma4/cuda/mimo2_prefill.cuh), [`cuda/axk2_flash.cuh`](file:///data/ds4-gemma4/cuda/axk2_flash.cuh), [`cuda/ds41_attn_mma.cuh`](file:///data/ds4-gemma4/cuda/ds41_attn_mma.cuh)

---

## Executive Summary

1. **Subagent Assessment & Decomposition (Q1):** The subagent is right that `acd8aa49` (`glm53_flash.cuh` over `glmvq_fa_kernel_t<__half>`) cannot be used as-is because GLM-5.3 is an MLA absorbed-latent kernel (`64` query heads along $M$ sharing a single `512`-dim latent row per key, $K == V$, no `kv_head` axis, no sliding window, no ring wrap, 96.25 KiB smem, and out-of-place `F32<->F16` conversion kernels that clobber `low_q` as scratch). However, **four parts of the subagent's proposed Gemma decomposition should be changed**:
   - **Do not implement sliding-only first:** At 8K+ prefill (the only regime where tensor-core attention moves end-to-end wall time), sliding layers are capped at `window = 1024` keys while full layers (`dim = 512, kv_heads = 2, group = 8, window = 0`) scan up to `8192+` keys at $2\times$ the head dimension ($8\times$ the work per query head at 8K, $>80\%$ of attention FLOPs at 32K+). Both `D = 256` (sliding) and `D = 512` (full) must be covered by a single templated family `g4_attn_mma_kernel<D, GROUP>` from the start.
   - **Pack the GQA `group` into the warp/MMA tile instead of pure token-major $M = 16$:** Pure token-major $M = 16$ per query head either re-reads KV tiles across heads in the group or under-utilizes warps on short chunks. Folding `(token_subtile, head_in_group)` into the CTA's 8 warps lets every staged `K`/`V` tile feed **all** query heads in the GQA group (`group = 2` on sliding, `group = 8` on full) in a single pass.
   - **Eliminate GLM's 32 KiB shared-memory score exchange on `D = 256`, and shrink it to a 2-half (`half = 0, 1`) exchange on `D = 512`:** GLM split `512` across 4 quarters for `QK^T` and 2 halves for `PV`. On Gemma `D = 256`, a single warp reduces all `256` dims in registers (`16` `k16` steps) with **zero** score-exchange shared memory and **zero** inter-warp score barriers. On `D = 512`, splitting `D` across 2 warps (`256` dims each) prevents register spilling (`o[32][4]` = 128 regs instead of `o[64][4]` = 256 regs) and requires only a `16 KiB` score-exchange buffer.
   - **Fuse `F32 -> BF16` query conversion and `F32` output write in-kernel:** Do not copy `glm53_flash.cuh`'s `glm53_f32_f16_kernel` / `glm53_f16_f32_kernel` launch wrapper, and never overwrite `query` in place.
2. **Warp/CTA Mapping & Budgets (Q2):** Put `kv_head` on `blockIdx.x` (`grid = dim3(kv_heads, cdiv(rows, RTILE))`). Use `8 warps` (`256 threads`) and `TK = 32` keys per tile:
   - **Sliding (`D = 256, kv_heads = 8, group = 2`):** `RTILE = 64` tokens (`4` row-groups of `16` tokens $\times$ `2` query heads = `8` warps), 1 warp per `(16-token, 1-head)` tile, full `D = 256` in registers (`o[32][4]` = 128 regs, `s[4][4]` = 16 regs, `Q` staged in smem or registers), **32–48 KiB smem**, zero score-exchange smem.
   - **Full (`D = 512, kv_heads = 2, group = 8`):** `RTILE = 8` tokens (`8 tokens × 8 query heads = 64` query rows per CTA, packed as `4` M-tiles of `16 = 2 tokens × 8 heads` across `4` warp pairs $\times$ `2` column-halves of `256` dims = `8` warps), `o[32][4]` = 128 regs per warp (zero local spills), **48 KiB smem** (`32 KiB` reused for `K`/`V` tile + `16 KiB` 2-half score exchange), fitting inside the standard **48 KiB static shared-memory limit without requiring dynamic-smem opt-in**.
3. **Honest Exactness Gate (Q3):** Because [`g4_rms_rope_kernel`](file:///data/ds4-gemma4/cuda/gemma4_primitives.cuh#L139-L176) and [`g4_ref_rope`](file:///data/ds4-gemma4/ds4_gemma4_ref.h#L101-L102) already round `Q` and `K` to exact BF16 values stored in `F32`/`BF16`, converting `Q` to `BF16` for MMA is **bit-lossless (0 ULP)**. The only arithmetic differences vs [`g4_attn_tile_kernel`](file:///data/ds4-gemma4/cuda/gemma4_attn_tile.cuh) are MMA tree order, `TK = 32` block-online softmax rescaling, and rounding $P = \exp(S - m)$ to `BF16` for $P V$. Aligning the KV loop start to `base0 = (begin_all / TK) * TK` makes `g4_attn_mma` **100% batch-size and chunk-size invariant** (`0.0` diff regardless of chunking). Gate with: (a) unit test `max_abs <= 2.5e-3`, `relL2 <= 1.5e-3`, and `0.0` chunk-invariance diff; (b) `ctx 256` 8-token greedy parity `[6113, 1698, 236761, 28333, 506, 877, 6910, 3136]`; (c) all 4 reference probes (`A=28, B=1243, C=1024, D=1025`) matching argmax `[818, 1509, 1509, 1509]` with `KL(tile || mma) < 5e-3`; (d) GB10 8K frontier `KL < 2e-2` and $|\Delta\text{NLL}| < 0.01$; (e) `sha256` repeat determinism.
4. **Sequencing vs P8/P9 (Q4):** **Land model integration after P8 and P9 close.** Do not wire `ds4_gpu_g4_attn_mma` into [`ds4_gemma4_graph.inc`](file:///data/ds4-gemma4/ds4_gemma4_graph.inc) before P8/P9, and set `G4_ATTN_MMA_MIN_ROWS = 16` (never `2`) so 5-row/7-row MTP verify passes permanently stay on the F32-exact [`g4_attn_tile_kernel`](file:///data/ds4-gemma4/cuda/gemma4_attn_tile.cuh). If a smaller slice is desired right now while P8/P9 run, write **only** the self-contained header `cuda/gemma4_attn_mma.cuh` and standalone unit test `tests/test_gemma4_attn_mma.cu` without touching `ds4_gemma4_graph.inc`.
5. **Failure Modes & Rollback (Q5):** Seven concrete failure modes (MTP verify drift at `rows = 5`, `0 * NaN = NaN` on out-of-bounds KV slots, `expf(-INF - (-INF)) = NaN` on fully masked sliding-window tiles, mid-tile `k % ring` wrap, `D = 512` register spills, dynamic smem opt-in / latched CUDA errors, and CUDA graph / FP8 full-KV collisions) have explicit hardware/software guards detailed in §5, backed by the three-valued (`1` / `0` / `-1`) fallback contract and `DS4_GEMMA4_ATTN_MMA=0` (default off).

---

## 1. Assessment of the Subagent's Analysis & Kernel Decomposition for Gemma's GQA + Window + Ring + $K \neq V$

### 1.1 Where the Subagent Is Right

Inspection of `/tmp/g4-acd8aa49.diff` (`cuda/glm53_flash.cuh` + `cuda/glm_vq_flash.cuh`) confirms why `acd8aa49` cannot be used directly for Gemma-4:

| Axis | GLM-5.3 (`acd8aa49`: `glm53_flash.cuh` + `glm_vq_flash.cuh`) | Gemma-4 26B A4B (`gemma4_attn_tile.cuh` / `ds4_gemma4_graph.inc`) |
|---|---|---|
| **Attention type** | Absorbed MLA (`q_low = q * W_k`, `low16 = flash(q_low, kv_cache)`, `out = low16 * W_v`) | Standard GQA with per-head `Q`, `K`, `V` (`heads = 16`) |
| **KV heads & GQA group** | `1` shared latent vector per token (`no kv_head axis`, `heads = 64`) | **Sliding (25 layers):** `kv_heads = 8, group = 2`<br>**Full (5 layers):** `kv_heads = 2, group = 8` |
| **Head dim (`D`)** | Fixed `GLMVQ_KV_LORA = 512` | **Sliding:** `D = 256`<br>**Full:** `D = 512` |
| **$K$ vs $V$** | $K == V$ (single 512-dim latent row per key) | $K \neq V$ (`kcache[il]` and `vcache[il]` are distinct BF16 buffers; even on `attention_k_eq_v` full layers, `k = rope(k_norm(proj))` and `v = v_norm(proj)` differ) |
| **MMA $M=16$ axis** | `16` query heads of **1 token row** (`grid.x = rows`, `4` head groups $\times 16 = 64$ heads) | Only `2` or `8` query heads share a KV head; pure head-major $M=16$ wastes $87.5\%$ (sliding) or $50\%$ (full) of every MMA |
| **Causal + Window + Ring** | Pure causal `key <= pos`, linear slot `k`, no window, no ring | Causal `k <= pos`, `k < keys` (MTP drafter), sliding `k + window > pos` (`window = 1024`), ring slot `k % ring` (`ring = 1024 + max(cap, drafts + 1)`) |
| **Attention scale** | Arbitrary `float scale` (`scale * log2(e)` via `exp2f`) | Exact `1.0f` ([`docs/gemma4-reference-semantics.md`](file:///data/ds4-gemma4/docs/gemma4-reference-semantics.md#L97-L98)) |
| **Input / Output types** | Launches extra `glm53_f32_f16_kernel` and `glm53_f16_f32_kernel`, overwriting `low_q` (`F32`) in place as scratch (`glm53_flash.cuh:62-71`) | `query` is `const float *` (must not be mutated), `key`/`value` are `const __nv_bfloat16 *`, `out` is `float *` |
| **Shared memory & ISA** | `96.25 KiB` (`64 KiB` double-buffered latent + `32 KiB` 4-quarter score exchange), `glmvq_pdl_open/wait`, `cp.async.bulk.prefetch.L2` | Needs clean `sm_89` (RTX 4070) + `sm_121a` (GB10) compatibility without PDL or `cp.async.bulk` |

### 1.2 Four Critical Changes Needed in the Subagent's Proposed Decomposition

While the subagent's diagnosis of the blockers is accurate, four parts of its proposed kernel plan should be revised before writing code:

#### Change 1: Do NOT restrict Phase 8 to Sliding Layers (`dim 256`) First — Cover Both `D = 256` and `D = 512` from Day One
The subagent proposes implementing sliding layers (`dim 256, window 1024`) first and deferring full layers (`dim 512`). **At 8K+ prefill, that targets the cold path and leaves the actual bottleneck unaccelerated:**
- For any token at position $p \ge 1024$, a sliding layer visits **at most `1024` keys** (`[p - 1023, p]`). Its key loop stops growing after token 1024.
- A full layer (`window = 0, ring = 0, dim = 512, kv_heads = 2, group = 8`) visits **all $p + 1$ keys** at **$2\times$ the head dimension (`512` vs `256`)**.
- Let us compare total attention MACs across all 25 sliding layers vs all 5 full layers as context length $L$ grows:
  $$\text{MACs}_{\text{sliding}}(L) \approx 25 \times 16 \times 256 \times \min(L, 1024) \cdot L$$
  $$\text{MACs}_{\text{full}}(L) = 5 \times 16 \times 512 \times \frac{L^2}{2} = 20{,}480 \, L^2$$
  - At $L = 256$ (RTX 4070 benchmark): [`docs/gemma4-attention-round-2026-11-03.md`](file:///data/ds4-gemma4/docs/gemma4-attention-round-2026-11-03.md#L73-L75) already measured each full layer at **`0.50–0.70 ms`** vs **`0.28 ms`** per sliding layer (`2.5 ms` across 5 full layers vs `7.1 ms` across 25 sliding layers).
  - At $L = 8{,}192$ (GB10 8K prefill): a full layer scans $4\times$ as many keys on average ($4096$ vs $1024$) at $2\times$ the dimension ($512$ vs $256$) = **$8\times$ the work per layer**, so the 5 full layers account for **$61.5\%$ of total attention FLOPs** (and over **$75\%$ of attention runtime** because `g4_attn_tile_kernel` uses `rtile = 1` on full layers vs `rtile = 4` on sliding layers, [`ds4_gemma4_gpu.cuh:481`](file:///data/ds4-gemma4/ds4_gemma4_gpu.cuh#L481)).
  - At $L = 32{,}768$: the 5 full layers account for **$86.5\%$ of total attention FLOPs**.
- **Fix:** Design `cuda/gemma4_attn_mma.cuh` around a single template `g4_attn_mma_kernel<D, GROUP>` instantiated for **both** `<256, 2>` (sliding) and `<512, 8>` (full).

#### Change 2: Fold `(token_rows, group_heads)` into the CTA / MMA $M=16$ Tile so Every KV Tile Feeds the Entire GQA Group
If $M = 16$ is purely 16 token rows of a single query head and each CTA only handles 1 query head, `K` and `V` are re-read from global/L2 memory `group` times (`2x` on sliding, `8x` on full). Instead:
- Pin one CTA to **one `kv_head` (`blockIdx.x = kv_head`) and a tile of `RTILE` token rows (`blockIdx.y = row_tile`)**, covering **all `GROUP` query heads** that share `kv_head`.
- **On Sliding (`D = 256, GROUP = 2`):**
  - Use `RTILE = 64` token rows (or `RTILE = 32` when `rows <= 32`).
  - Within the CTA's 8 warps, each warp owns a `M = 16` tile consisting of **`8` contiguous token rows $\times$ `2` query heads (`GROUP = 2`)** (`m_lane = r_in_warp * 2 + head_in_group`, $r_{\text{in\_warp}} \in [0, 7]$, $h \in [0, 1]$).
  - Why pack `8 tokens × 2 heads = 16` inside each warp's `m16n8k16` fragment instead of `16 tokens × 1 head`?
    1. At `rows = 16` (or tail chunks), 2 warps already have 100% full `M = 16` fragments (`16 tokens × 2 heads = 32` rows), whereas `16 tokens × 1 head` across 8 warps requires `64` token rows to fill all warps.
    2. Across 8 warps (`RTILE = 64` tokens $\times$ `2` heads = `128` query rows), every shared-memory `K[32][256]` and `V[32][256]` tile is reused **`128x`** (`64 tokens × 2 heads`).
- **On Full (`D = 512, GROUP = 8`):**
  - Pack `M = 16` as **`2` contiguous token rows $\times$ `8` query heads (`GROUP = 8`)** (`m_lane = r_in_pair * 8 + head_in_group`, $r_{\text{in\_pair}} \in [0, 1]$, $h \in [0, 7]$).
  - With `4` such M-tiles per CTA (`RTILE = 8` token rows $\times$ `8` query heads = `64` query rows per CTA) and `2` warps per M-tile splitting `D = 512` into two 256-dim halves (`8` warps total):
    - Even at `rows = 16`, `grid = (2 kv_heads, 2 row_tiles) = 4 CTAs` (or with `RTILE = 16` across 2 passes, `grid = (2, 1)`), every warp's `M = 16` MMA tile is **100% utilized** once `rows >= 8`!
    - Every shared-memory `K[32][512]` and `V[32][512]` tile is reused across all **8 query heads** of the GQA group simultaneously.

#### Change 3: Eliminate Cross-Warp Score Exchange on `D = 256` and Use a 2-Way (Not 4-Way) Split on `D = 512`
In `cuda/glm_vq_flash.cuh:169-205`, GLM splits `D = 512` across `4` warps (`quarter = 0..3`) for $Q K^T$ and `2` warps (`half = 0..1`) for $P V$, requiring `32 KiB` of shared memory (`GLMVQ_FA_EXCHANGE`) and two `bar.sync` barriers per tile just to sum partial scores.
- **For `D = 256` (sliding):** Each warp owns all `256` columns (`16` `k16` MMA steps for $Q K^T$, `32` `n8` output tiles `o[32][4]` = `128` registers for $P V$). Every warp computes its own complete dot products `s[4][4]`, runs online softmax in registers (`__shfl_xor_sync` across the 4 lanes of an MMA row), and accumulates `o[32][4]` with **0 bytes of score-exchange shared memory** and **no inter-warp score reduction**.
- **For `D = 512` (full):** A single thread cannot hold all `512` output columns in registers (`o[64][4]` = `256` FP32 registers, exceeding the hardware limit of `255` registers/thread on `sm_89` and `sm_121a`). Splitting `D = 512` into **2 halves of `256` dims** (`half = warp >> 2`, `0` for dims `0..255` and `1` for dims `256..511`):
  - Gives each warp the **exact same `256`-column inner loop** as the `D = 256` sliding kernel (`qa[16][4]` = 64 regs, `o[32][4]` = 128 regs, `s[4][4]` = 16 regs)!
  - Requires exchanging only **two** partial score matrices per M-tile (`4 M-tiles × 2 halves × 16 × 32 × 4B = 16 KiB` of shared memory), which can even alias the KV tile buffer after $Q K^T$ completes!

#### Change 4: Keep Conversion Inside `g4_attn_mma_kernel` and Reuse One `32 KiB` Shared Tile Buffer for `K` then `V` on `D = 512`
- Do not launch separate `f32_to_bf16` / `bf16_to_f32` kernels (`glm53_flash.cuh:62-71`). Each warp loads its `query` rows once as `float2`, converts via `__float22bfloat162_rn` into register fragments `qa[16][4]`, and writes `float2` (`o / l`) directly to `out` at CTA exit.
- Because $K \neq V$, staging both `K[32][D]` and `V[32][D]` simultaneously would take `32 KiB` (`D = 256`) or `64 KiB` (`D = 512`) single-buffered, and `64 KiB` / `128 KiB` double-buffered. Instead:
  - Within each `TK = 32` key tile, stage `K[32][D]` into `smem_kv` (`16 KiB` for `D = 256`, `32 KiB` for `D = 512`), overlap the `cp.async` of `V[32][D]` into a second `16 KiB` buffer (for `D = 256`) or pipeline `K` then `V` into the same `32 KiB` buffer (for `D = 512`).
  - This keeps total shared memory at **`32 KiB` (`D = 256`)** and **`48 KiB` (`D = 512`)** — **both $\le 48\text{ KiB}$**, so the kernel runs within the default static shared-memory carve-out on every CUDA device without dynamic-smem opt-in failures!

---

## 2. Warp/CTA Mapping, Register & Shared-Memory Budget, and `kv_head` Axis

### 2.1 Where the `kv_head` Axis Must Live

```
gridDim = dim3(kv_heads, (rows + RTILE - 1) / RTILE, 1)
blockDim = dim3(256, 1, 1)   // 8 warps
```

- **`blockIdx.x = kv_head`** (`0..7` on sliding, `0..1` on full), matching [`ds4_gpu_g4_attn_tile`](file:///data/ds4-gemma4/ds4_gemma4_gpu.cuh#L477-L483).
- **`blockIdx.y = row_block`** (`row0 = blockIdx.y * RTILE`).
- Putting `kv_head` on `blockIdx.x` guarantees that all 8 warps in a CTA read the **same** `kv_head` slice of `kcache` and `vcache`, so every `cp.async` load into shared memory serves all `GROUP` query heads (`2` or `8`) and all `RTILE` token rows in the CTA.

### 2.2 Unified Sub-256-Dim Inner Engine (`DHALF = 256`, `TK = 32`)

Both `D = 256` (sliding) and `D = 512` (full) share the **exact same per-warp inner tile shape**:
- **Per-warp M-tile:** `M = 16` query rows (`m16n8k16`), where lane `l = 0..31` owns `gid = l >> 2` ($0..7$) and `tid = l & 3` ($0..3$), covering MMA rows `r0 = gid` and `r1 = gid + 8`.
- **Per-warp K/V column slice:** `DHALF = 256` columns (`16` `k16` chunks for $Q K^T$; `32` `n8` tiles for $P V$).
- **Key tile width:** `TK = 32` keys (`4` `n8` tiles for $Q K^T$; `2` `k16` steps for $P V$).

| Parameter | Sliding (`D = 256, kv_heads = 8, group = 2`) | Full (`D = 512, kv_heads = 2, group = 8`) |
|---|---|---|
| **Template instantiation** | `g4_attn_mma_kernel<256, 2>` | `g4_attn_mma_kernel<512, 8>` |
| **Threads / Warps per CTA** | `256` threads (`8` warps) | `256` threads (`8` warps) |
| **Token rows per CTA (`RTILE`)** | **`64` tokens** (`8` warps $\times$ `8` tokens/warp) | **`8` tokens** (`4` M-groups $\times$ `2` tokens/M-group) |
| **How `M = 16` is packed in each warp** | `8 tokens × 2 query heads = 16`<br>`tok_in_warp = m_row >> 1` ($0..7$)<br>`head_in_group = m_row & 1` ($0..1$) | `2 tokens × 8 query heads = 16`<br>`tok_in_group = m_row >> 3` ($0..1$)<br>`head_in_group = m_row & 7` ($0..7$) |
| **Warp assignment (`warp = 0..7`)** | `w_mgroup = warp` ($0..7$), `w_half = 0`<br>(1 warp owns full `D = 256` for 8 tokens $\times$ 2 heads) | `w_mgroup = warp & 3` ($0..3$), `w_half = warp >> 2` ($0..1$)<br>(2 warps `w_half=0,1` split `D = 512` into `2 × 256` for 2 tokens $\times$ 8 heads) |
| **KV byte reuse per CTA** | `64 tokens × 2 heads = 128x` | `8 tokens × 8 heads = 64x` |
| **CTAs launched at `rows = 256`** | `grid = (8, 4) = 32 CTAs` | `grid = (2, 32) = 64 CTAs` |
| **CTAs launched at `rows = 512`** | `grid = (8, 8) = 64 CTAs` | `grid = (2, 64) = 128 CTAs` |

> **Why this mapping works well on both GPUs (`sm_89` = 46 SMs, `sm_121a` = 48 SMs):**
> On full layers (`kv_heads = 2`), a naive `RTILE = 64` mapping would launch only `grid = (2, 4) = 8 CTAs` at `rows = 256`, leaving **83% of the GPU's SMs idle**! By packing `2 tokens × 8 query heads = 16` into each M-tile (`RTILE = 8` tokens per CTA), `rows = 256` launches `2 × 32 = 64 CTAs` on full layers (filling all 48 SMs on GB10 and 46 SMs on RTX 4070) while achieving 100% M-tile occupancy and $64\times$ KV shared-memory reuse.

### 2.3 Register Budget per Thread (`__launch_bounds__(256, 1)`)

Because both specializations operate on `DHALF = 256` columns per warp, their per-thread register allocation is identical and stays comfortably below the 255-register hardware ceiling with **zero local-memory spills**:

| Register Array / Variable | Shape per Thread | FP32 Regs | Purpose |
|---|---|---:|---|
| `qa[16][4]` | `16` `k16` chunks $\times$ `4` `uint32_t` (`bf16x2`) | **64** | Resident `Q` fragment for this warp's `16` M-rows $\times$ `256` dims |
| `o[32][4]` | `32` `n8` tiles $\times$ `4` `float` | **128** | FP32 output accumulator for `16` M-rows $\times$ `256` dims |
| `s[4][4]` / `p_frag[2][4]` | `4` `n8` tiles $\times$ `4` `float` (reused for `p_frag`) | **16** | FP32 attention scores for `TK = 32` keys / BF16 `P` fragments |
| `m[2]`, `l[2]`, `alpha[2]` | `2` MMA rows (`gid`, `gid+8`) | **6** | Per-row running max $m$, normalizer $\ell$, and rescale factor $\alpha$ |
| `kb[4]`, loop/address temps | `ldmatrix` operand + pointers/predicates | **18–22** | Temporary `K`/`V` B-fragment and loop bookkeeping |
| **Total per thread** | | **~232–236** | **$\le 255$ limit (`0B` spill stores, `0B` spill loads)** |

*(Optional register-pressure relief if `ptxas` ever approaches 250 regs on a future CUDA toolkit: staging `Q` in `16 KiB` of shared memory and loading `qa` per `k16` step via `ldmatrix.sync.aligned.m8n8.x4.b16` drops `qa` from `64` regs to `4` regs, reducing total register usage to **~176 regs/thread**.)*

### 2.4 Shared-Memory Budget & Swizzle (`<= 48 KiB` Static Limit)

To eliminate shared-memory bank conflicts for both row-major $Q K^T$ (`ldmatrix.sync.aligned.m8n8.x4.shared.b16`) and transposed $P V$ (`ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16`), apply the standard 16-byte XOR swizzle across the 16-byte chunks of each key row:
```cuda
// k_idx in [0, 31], chunk16 in [0, D/8 - 1] (each chunk16 is 8 bf16 = 16 bytes)
__device__ __forceinline__ uint32_t g4_fa_swizzle_offset(uint32_t k_idx, uint32_t chunk16) {
    return (k_idx * (D / 8u) + (chunk16 ^ (k_idx & 7u))) * 16u;
}
```
- **Why `chunk16 ^ (k_idx & 7u)` is conflict-free:**
  - Each `chunk16` is 16 bytes (4 banks). Across 8 consecutive keys (`k_idx = 0..7`), the 8 lanes of an `m8n8` matrix read the same logical `chunk16` across 8 rows; XORing with `k_idx & 7u` permutes the 8 rows across all 8 distinct 16-byte bank groups (32 banks) with **0 bank conflicts** on both `ldmatrix.x4` (for $K$) and `ldmatrix.x4.trans` (for $V$).

| Kernel Specialization | `K`/`V` Tile Smem (`TK = 32`) | Score Exchange Smem | Total Smem | Static `<= 48 KiB`? |
|---|---|---|---|---|
| **Sliding `<256, 2>`** | `K[32][256]` (`16 KiB`) + `V[32][256]` (`16 KiB`) | `0 KiB` (each warp owns full `D = 256`) | **`32.0 KiB`** | **Yes** (`32,768 B`) |
| **Full `<512, 8>`** | `KV[32][512]` (`32 KiB`, reused for `K` then `V`) | `exchange[4][2][16][32]` floats (`16 KiB`, or union-aliased with `KV`) | **`32.0–48.0 KiB`** | **Yes** (`32,768–49,152 B`) |

Because both kernels fit in **`<= 48 KiB`**, neither kernel depends on `cudaFuncSetAttribute(..., cudaFuncAttributeMaxDynamicSharedMemorySize, ...)` to launch, eliminating dynamic shared-memory opt-in failures across architectures and CUDA Graph captures.

---

## 3. Concrete Kernel Structure for `cuda/gemma4_attn_mma.cuh`

Below is the exact per-tile dataflow that handles GQA (`group = 2` or `8`), sliding window (`window = 1024`), ring buffer wrap (`k % ring`), `keys` cap (`keys`), and $K \neq V$:

```cuda
// Per-CTA bounds over the token rows [row0, row0 + RTILE - 1] covered by this CTA:
const uint32_t row0      = blockIdx.y * RTILE;
const uint32_t row_end   = (row0 + RTILE < rows) ? (row0 + RTILE) : rows;
const uint32_t pos_first = first + row0;
const uint32_t pos_last  = first + row_end - 1u;

// Earliest key needed by ANY row in this CTA (row0 has the smallest pos):
const uint32_t begin_all = (window > 0u && pos_first + 1u > window)
                         ? (pos_first + 1u - window) : 0u;
// Latest key needed by ANY row in this CTA:
const uint32_t last_all  = (keys > 0u && keys <= pos_last)
                         ? (keys - 1u) : pos_last;
if (keys > 0u && begin_all >= keys) return; // Entire CTA masked

// CRITICAL FOR CHUNK INVARIANCE: align tile loop start to a multiple of TK (32)
// so key tile boundaries [base, base + 31] depend ONLY on absolute key indices,
// never on chunk size `rows` or `row0`!
const uint32_t base_start = (begin_all / 32u) * 32u;

// Fast-path unmasked bounds for thisCTA / warp:
// Every row in this CTA is unmasked on [begin_max, last_min] when row_end - row0 == RTILE:
const uint32_t begin_max = (window > 0u && pos_last + 1u > window)
                         ? (pos_last + 1u - window) : 0u;
const uint32_t last_min  = (keys > 0u && keys <= pos_first)
                         ? (keys - 1u) : pos_first;
```

### Key Correctness Invariants Inside the Tile Loop (`for (uint32_t base = base_start; base <= last_all; base += 32u)`):

1. **Per-Key Ring Modulo & Zero-Fill on Out-of-Bounds Keys:**
   When staging `K[t]` or `V[t]` ($t \in [0, 31]$) with `cp.async.cg.shared.global`:
   ```cuda
   const uint32_t k_abs = base + t;
   const bool k_valid   = (k_abs >= begin_all) && (k_abs <= last_all);
   const uint32_t slot  = k_valid ? (ring > 0u ? (k_abs % ring) : k_abs) : 0u;
   // Issue cp.async with src_size = k_valid ? 16 : 0 so invalid keys hardware-zero smem!
   ```
   - Computing `k_abs % ring` per key $t$ handles ring wrap (`k_abs = m * ring - 1 -> m * ring`) mid-tile without out-of-bounds reads.
   - Setting `src_size = 0` on `k_abs > last_all` or `k_abs < begin_all` hardware-zeros the shared-memory slot so uninitialized cache memory (`NaN`/`Inf` beyond `last_all`) can never enter `mma.sync` and cause `0.0f * NaN = NaN`.

2. **Exact Per-Row Masking & All-Masked Tile NaN Guard:**
   For each of the two MMA rows `h = 0, 1` (`m_row = gid + 8 * h`) owned by a lane:
   - Decode `tok_idx = row0 + (w_mgroup * TOK_PER_MGROUP) + (m_row / GROUP)` and `pos = first + tok_idx`.
   - `row_valid = (tok_idx < rows)`.
   - `row_begin = (window > 0u && pos + 1u > window) ? (pos + 1u - window) : 0u`.
   - `row_last  = (keys > 0u && keys <= pos) ? (keys - 1u) : pos`.
   - If the tile is not inside the unmasked interior `(row_end == row0 + RTILE && base >= begin_max && base + 31u <= last_min)`, mask score `s[nt][h*2 + c]` to `-INFINITY` whenever `!row_valid || k_abs < row_begin || k_abs > row_last`.
   - **Online softmax NaN guard when a tile is 100% masked for row `h`** (which happens on early tiles `[base, base+31]` for later rows in the CTA whose sliding window starts at `row_begin > base + 31`, or on late tiles where `base > row_last`):
     ```cuda
     // After butterfly max reduction of tile_max[h] across the 4 lanes of row h:
     if (tile_max[h] > -INFINITY) {
         const float m_new = fmaxf(m[h], tile_max[h]);
         const float alpha = (m[h] == -INFINITY) ? 0.0f : __expf(m[h] - m_new);
         l[h] *= alpha;
         m[h]  = m_new;
         // Rescale o[d][h*2 + 0..1] *= alpha, compute p = (s == -INFINITY) ? 0.0f : __expf(s - m_new)
     } else {
         // Entire 32-key tile is masked for row h: keep m[h], l[h], o[][h] unchanged, and set p_frag = 0!
     }
     ```
   - This `if (tile_max[h] > -INFINITY)` branch guarantees:
     1. Zero `NaN`s (`-INFINITY - (-INFINITY)` never executes).
     2. Masked tiles before `row_begin` or after `row_last` perform **no rescaling** of `m[h]`, `l[h]`, or `o`, which makes the output of every row `r` **100% independent of `RTILE`, `row0`, and `rows`**!

---

## 4. Honest Exactness Gate, Tolerances, and Required Probes (Q3)

### 4.1 Why `Q` Conversion Is 0-ULP Exact and Where the Real Numerical Delta Comes From

A subtle property of the Gemma-4 port makes `g4_attn_mma` even closer to `g4_attn_tile` than `glm53_flash` is to `glm53_decode`:
- Look at [`ds4_gemma4_ref.h:101-102`](file:///data/ds4-gemma4/ds4_gemma4_ref.h#L101-L102) and [`cuda/gemma4_primitives.cuh:169-174`](file:///data/ds4-gemma4/cuda/gemma4_primitives.cuh#L169-L174): **`g4_rms_rope_kernel` already rounds every element of `query` and `key` through `g4_cuda_bf16` (`__bfloat162float(__float2bfloat16_rn(x))`) before storing `query` to the `F32` buffer!**
- Likewise, `kcache` and `vcache` are stored in `BF16`.
- Therefore, **converting `query` from `F32` to `BF16` at the start of `g4_attn_mma_kernel` is bit-lossless (0 ULP error)**! Both `Q` and `K` entering `mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32` have the exact same bit values as in [`g4_attn_tile_kernel`](file:///data/ds4-gemma4/cuda/gemma4_attn_tile.cuh).
- The **only** numerical differences between `g4_attn_mma_kernel` and `g4_attn_tile_kernel` are:
  1. **Dot-product reduction tree for $S = Q K^T$:** `m16n8k16` tensor-core accumulation order (plus `__expf` vs `expf`) instead of 32-lane strided FMA + warp butterfly shuffle.
  2. **Block-online softmax rescaling (`TK = 32` keys at a time):** Rescaling `l` and `o` once per 32-key tile instead of per individual key.
  3. **Rounding $P = \exp(S - m)$ from `F32` to `BF16`** (`__floats2bfloat162_rn`) to feed the $O \mathrel{+}= P V$ tensor-core MMA, plus its `m16n8k16` accumulation order.

### 4.2 Why `base_start = (begin_all / 32) * 32` Preserves Exact Chunk Invariance

Because:
1. Key tiles always start on global multiples of 32 (`k = 0, 32, 64, ...`),
2. Fully masked tiles (`tile_max == -INFINITY`) leave `m`, `l`, and `o` untouched, and
3. `mma.sync` has zero cross-row mixing along the $M$ dimension,

a query token at absolute position `pos` processes the **exact same sequence of 32-key tiles with the exact same `tile_max` and `alpha` rescalings regardless of whether `rows` is `16`, `64`, `256`, or `512`, and regardless of which `m_row` lane (`0..15`) the token occupies!**

### 4.3 Required 5-Stage Exactness Gate & Concrete Tolerances

| Gate Stage | Workload / Command | Required Metric & Threshold |
|---|---|---|
| **1. Standalone CUDA Primitive Gate** (`tests/test_gemma4_attn_mma.cu`) | Synthetic shapes covering:<ul><li>Sliding (`D=256, kv=8, group=2, win=1024, ring=0 & 1029`)</li><li>Full (`D=512, kv=2, group=8, win=0, ring=0`)</li><li>`rows = 16, 17, 32, 33, 64, 128, 300`</li><li>`first = 0, 500, 1020, 1025` (window + ring wrap)</li><li>Poisoned cache slots beyond `last_all` (`NaN` pattern)</li></ul> | <ul><li>vs `g4_ref_attn` (FP64/FP32 oracle) & `g4_attn_tile`:<br>**`max_abs <= 2.5e-3`**, **`relL2 <= 1.5e-3`** (tighter than GLM-5.3's `4.0e-3` in `test_glm53_flash.cu:38` because `Q` has no F16 rounding loss and `O` writes F32 directly)</li><li>**Chunk-invariance check:** Token at `pos = 1300` evaluated inside `rows = 16` (`first = 1295`), `rows = 64` (`first = 1260`), and `rows = 256` (`first = 1100`) must match with **`max_abs == 0.0f` (bit-identical)**</li></ul> |
| **2. Short-Context Live Model Gate (`ctx 256`, RTX 4070 or GB10)** | `/data/vq-artifacts/gguf-family-it-r8` on `/tmp/g4-bench-256.txt` (`DS4_GEMMA4_ATTN_MMA=1` vs `DS4_GEMMA4_ATTN_PREFILL=1`) | <ul><li>**8-token greedy sequence:** exact match `[6113, 1698, 236761, 28333, 506, 877, 6910, 3136]`</li><li>**Frontier logit drift (`tile` vs `mma`):** `KL(tile \|\| mma) <= 5.0e-3`, `relL2 <= 6.0e-2` (inside the accepted `R5`/`R6` batching band `3.89e-2..5.10e-2` in [`docs/gemma4-attention-round-2026-11-03.md`](file:///data/ds4-gemma4/docs/gemma4-attention-round-2026-11-03.md#L121-L128))</li><li>**Repeat determinism:** two `--dump-logits` runs with `DS4_GEMMA4_ATTN_MMA=1` must have identical `sha256` (`relL2 = 0.0`)</li></ul> |
| **3. Four Pinned Reference Probes (GB10)** | `tools/gemma4-ref/` on `/data/vq-artifacts/gguf-family-it-r8`:<ul><li>Probe A (`28` tok, inside window)</li><li>Probe B (`1243` tok, past 1024 window)</li><li>Probe C (`1024` tok, exact window edge)</li><li>Probe D (`1025` tok, first eviction)</li></ul> | <ul><li>All 4 probes preserve the pinned HF reference argmax: **`[818, 1509, 1509, 1509]`**</li><li>`KL(tile \|\| mma) <= 5.0e-3` on all 4 probes</li><li>Cosine vs HF BF16 oracle stays within `±0.001` of the R13+8 baseline (`0.9906` on A, `0.9901` on B)</li></ul> |
| **4. Long-Context 8K Frontier & NLL Gate (GB10)** | 8K prompt prefill + 512-token teacher-forced NLL (`DS4_GEMMA4_ATTN_MMA=0` vs `1`) | <ul><li>8K frontier `KL(tile \|\| mma) <= 2.0e-2` (matching `acd8aa49`'s `1.6e-2` gate)</li><li>Teacher-forced $|\Delta\text{NLL}| \le 0.01$</li></ul> |
| **5. Live Serving Gate** | `./ds4 --serving-check` with `DS4_GEMMA4_ATTN_MMA=1` | **`works 10, FAIL 0, WRONG 0`** |

---

## 5. Sequencing: After P8/P9, or a Smaller Slice Now? (Q4)

### 5.1 Recommendation: Land Graph Integration **After** P8 and P9 Close

The subagent is right that wiring `ds4_gpu_g4_attn_mma` into [`ds4_gemma4_graph.inc`](file:///data/ds4-gemma4/ds4_gemma4_graph.inc) should wait until **after P8 (MTP verification & wrapped-ring gates) and P9 (HTTP serving gate) are closed**, for three concrete technical reasons:

1. **MTP Verify Width (`rows = 5` or `7`) Would Collide If `MIN_ROWS = 2`:**
   - In [`ds4_gemma4_mtp.inc:325`](file:///data/ds4-gemma4/ds4_gemma4_mtp.inc#L325), `gemma4_mtp_verify` calls `gemma4_forward(g, m, w, ids, n, pos, false)` with `n = drafts + 1` (`5` at 4 drafts, `7` at 6 drafts).
   - In [`ds4_gemma4_graph.inc:723`](file:///data/ds4-gemma4/ds4_gemma4_graph.inc#L723), any `rows > 1u` enters the prefill attention branch.
   - If `g4_attn_mma` used `GLM53_FLASH_MIN_ROWS = 2`, a 5-row MTP verify pass would run `g4_attn_mma` (which rounds $P$ to BF16) while 1-row plain decode runs `g4_attn_fast` (which keeps $P$ in F32), introducing unnecessary verify logit drift during P8's `drafting on == drafting off` greedy equivalence gate.
   - **Permanent Rule (even after P8/P9):** Set **`G4_ATTN_MMA_MIN_ROWS = 16`** (not `2`)! For `rows = 2..8` (all MTP verify widths), [`g4_attn_tile_kernel`](file:///data/ds4-gemma4/cuda/gemma4_attn_tile.cuh) (`rtile = 4` sliding, `rtile = 1` full) has zero wasted MMA rows and is bit-identical to 1-row `g4_attn_fast_kernel`. `g4_attn_mma` should only engage when `rows >= 16` and `keys == 0`.
2. **Amdahl's Law on the RTX 4070 (`ctx 256`):**
   - At `ctx 256` on the RTX 4070, [`g4_attn_tile_kernel`](file:///data/ds4-gemma4/cuda/gemma4_attn_tile.cuh) is only **`9.5 ms` of `288.0 ms` (`3.3%`)** of prefill GPU time (`7.1 ms` sliding + `2.5 ms` full), whereas VQ expert GEMM is `142.2 ms` (`49.4%`) and dense down is `51.8 ms` (`18.0%`). Even a $3\times$ speedup on attention at `ctx 256` saves only `~6 ms` (`~2%`).
   - Because the 12 GiB RTX 4070 cannot hold `ctx >= 2048` device-resident alongside the `9.37 GiB` model weight allocation ([`docs/gemma4-attention-round-2026-11-03.md:148-157`](file:///data/ds4-gemma4/docs/gemma4-attention-round-2026-11-03.md#L148-L157)), the regime where `g4_attn_mma` yields a large end-to-end gain (`ctx 4096..32768+`, where attention grows to `20–50%+` of prefill) is the **GB10 long-context campaign after P8 and P9 close**.
3. **Zero Interference with Active P8/P9 Binary Gates:**
   - Editing `ds4_gemma4_graph.inc` or `ds4_gemma4_gpu.cuh` forces a rebuild of `ds4_cuda.o` and `ds4`/`ds4-server` while P8 and P9 are being gated.

### 5.2 What Smaller Slice *Is* Worth Doing Now (Zero Risk to P8/P9)

If you want to prepare Phase 8 in parallel without touching any file compiled into `ds4` or `ds4-server`:
- Write **only** the standalone header `cuda/gemma4_attn_mma.cuh` and a standalone unit test binary `tests/test_gemma4_attn_mma.cu` (compiled on demand via `nvcc -O3 -arch=sm_89 tests/test_gemma4_attn_mma.cu -o /tmp/test_gemma4_attn_mma`).
- That validates both `<256, 2>` (sliding) and `<512, 8>` (full), `ptxas` register counts (`0` spills), `max_abs <= 2.5e-3`, and `0.0` chunk-invariance against `g4_ref_attn` and `g4_attn_tile_kernel` in ~1 second of GPU time, leaving `ds4_gemma4_graph.inc` untouched until P8 and P9 finish.

---

## 6. Failure Modes and Rollback (Q5)

### 6.1 Seven Specific Failure Modes & Required Guards

| # | Failure Mode | Root Cause | Required Guard in `cuda/gemma4_attn_mma.cuh` |
|---|---|---|---|
| **1** | **MTP verify / drafter logit mismatch (`rows = 2..7` or `keys != 0`)** | `gemma4_mtp_verify` calls `gemma4_forward` with `rows = drafts + 1` (`5` or `7`); drafter passes `keys = start`. Engaging MMA changes verify rounding vs 1-row `g4_attn_fast`. | Gate entry with `if (rows < 16u \|\| keys != 0u) return 0;` so MTP verify (`rows <= 8`) and drafter (`keys != 0`) always fall back to `ds4_gpu_g4_attn_tile` / `ds4_gpu_g4_attn`. |
| **2** | **`0.0f * NaN = NaN` from unwritten KV cache slots past `last_all`** | For tail keys `base + t > last_all`, `kcache`/`vcache` slots have not been written yet and may contain garbage/`NaN`. In `mma.sync` for $P V$, multiplying $P = 0$ by $V = \text{NaN}$ produces `NaN` in `o`. | In `cp.async.cg.shared.global [dst], [src], 16, src_size`, pass `src_size = (k_abs >= begin_all && k_abs <= last_all) ? 16 : 0` so hardware zeros shared memory for out-of-range keys. |
| **3** | **`expf(-INF - (-INF)) = NaN` on all-masked sliding-window tiles** | In a multi-row CTA (`RTILE = 64`), early tiles near `begin_all` (for `row0`) lie completely before `row_begin` of `row0 + 63`. All 32 scores for `row0 + 63` are `-INFINITY`, so `tile_max = -INFINITY`. | Guard the online softmax update with `if (tile_max[h] > -INFINITY)` before computing `m_new = fmaxf(m[h], tile_max[h])` and `__expf(m[h] - m_new)`. When `tile_max[h] == -INFINITY`, set `p_frag = 0` and leave `m[h], l[h], o` untouched. |
| **4** | **Illegal memory access / wrong keys on ring wrap mid-tile** | A 32-key tile `[base, base + 31]` can straddle `k_abs = m * ring - 1 -> m * ring` (`ring = 1029` or `1280`). Computing `base % ring` once per tile reads out of bounds past `ring - 1`. | Compute `slot = (ring > 0u) ? (k_abs % ring) : k_abs` per key row `t = 0..31` inside the `cp.async` staging helper. |
| **5** | **Severe local-memory register spilling on `D = 512` full layers** | Holding all 512 dims of `o[64][4]` (`256` regs) + `qa[32][4]` (`128` regs) in one thread exceeds 255 regs/thread (`glm_vq_flash.cuh:17` measured 34 GB of spill traffic). | Split `D = 512` across 2 warps (`w_half = warp >> 2`, `DHALF = 256`, `o[32][4]` = `128` regs) and enforce `-Xptxas -v` verification of `0 bytes spill stores, 0 bytes spill loads`. |
| **6** | **Dynamic shared-memory opt-in failure or latched prior CUDA error** | Requesting `> 48 KiB` dynamic smem fails if `cudaFuncSetAttribute` is rejected or called during stream capture (`cudaErrorStreamCaptureUnsupported`). | Keep static shared memory `<= 48 KiB` (`32 KiB` on `D=256`, `48 KiB` on `D=512`) so no `cudaFuncSetAttribute` call is needed. |
| **7** | **Full-layer FP8 KV cache (`DS4_GEMMA4_FULL_KV=fp8`) collision** | When `g->full_fp8` is enabled, `g->kcache[il]` and `g->vcache[il]` are `NULL` on full layers (`g->fkcache[il]` is allocated instead). | In `ds4_gemma4_graph.inc:709`, keep the `if (full && g->full_fp8)` branch ahead of the BF16 attention dispatch, and check `if (!query \|\| !key \|\| !value \|\| !out) return 0;` at the top of `ds4_gpu_g4_attn_mma`. |

### 6.2 Three-Valued Dispatch & Instant Rollback Contract

In [`ds4_gemma4_gpu.cuh`](file:///data/ds4-gemma4/ds4_gemma4_gpu.cuh#L446-L491) and [`ds4_gemma4_graph.inc`](file:///data/ds4-gemma4/ds4_gemma4_graph.inc#L723-L735):

1. **Default-Off Environment Gate (`DS4_GEMMA4_ATTN_MMA`):**
   - `ds4_g4_attn_mma_enabled()` reads `getenv("DS4_GEMMA4_ATTN_MMA")` once and defaults to **`0` (disabled)**.
2. **Three-Valued Return from `ds4_gpu_g4_attn_mma(...)`:**
   - **`1` (Handled):** Kernel launched and `cudaGetLastError() == cudaSuccess`.
   - **`0` (Not covered / disabled):** Returned **before any GPU launch or output mutation** whenever `!ds4_g4_attn_mma_enabled()`, `rows < 16`, `keys != 0`, unsupported `(dim, heads, kv_heads)`, or invalid tensor capacities. Caller immediately falls through to `ds4_gpu_g4_attn_tile` (when `DS4_GEMMA4_ATTN_PREFILL=1`) or `ds4_gpu_g4_attn`.
   - **`-1` (Launch failure):** Kernel was selected and launched, but `cudaGetLastError()` returned an error. Caller propagates `-1` to `G4_STEP` and aborts the forward cleanly instead of silently running a second kernel over a faulted stream.
3. **Rollback Levels:**
   - **Runtime (0 seconds, no rebuild):** Unset `DS4_GEMMA4_ATTN_MMA` or set `DS4_GEMMA4_ATTN_MMA=0` — every prefill call takes `ds4_gpu_g4_attn_tile` bit-identically.
   - **Compile-time / Git:** All MMA code lives in one new Gemma-only header `cuda/gemma4_attn_mma.cuh` plus a 6-line dispatch hook in `ds4_gemma4_graph.inc`; reverting the commit or removing the hook leaves zero footprint in GLM/GLM53/Ling or the existing Gemma tile/fast kernels.
