# Phase 8 Attention & P8 Verify-Head Review (`1bd4afe5`)

**Date:** 2026-11-04  
**Repository / Commit:** `/data/ds4-gemma4` (`feat/gemma4-family` at `1bd4afe5`, plus working-tree P8 trial gate edits)  
**Files Inspected:**
- [`ds4_gemma4_mtp.inc`](file:///data/ds4-gemma4/ds4_gemma4_mtp.inc), [`ds4_gemma4_graph.inc`](file:///data/ds4-gemma4/ds4_gemma4_graph.inc), [`ds4_gemma4_gpu.cuh`](file:///data/ds4-gemma4/ds4_gemma4_gpu.cuh)
- [`cuda/gemma4_head.cuh`](file:///data/ds4-gemma4/cuda/gemma4_head.cuh), [`cuda/glm_vq_gemv.cuh`](file:///data/ds4-gemma4/cuda/glm_vq_gemv.cuh), [`ds4_glmvq_gpu.cuh`](file:///data/ds4-gemma4/ds4_glmvq_gpu.cuh)
- [`ds4_cuda.cu`](file:///data/ds4-gemma4/ds4_cuda.cu), [`ds4.c`](file:///data/ds4-gemma4/ds4.c), [`ds4_ling31_mtp.inc`](file:///data/ds4-gemma4/ds4_ling31_mtp.inc), [`ds4_ling31_session.inc`](file:///data/ds4-gemma4/ds4_ling31_session.inc), [`ds4_glm53_mtp.inc`](file:///data/ds4-gemma4/ds4_glm53_mtp.inc)
- [`cuda/ds41_attn_mma.cuh`](file:///data/ds4-gemma4/cuda/ds41_attn_mma.cuh), [`Agy_phase8_review.md`](file:///data/ds4-gemma4/Agy_phase8_review.md), [`crates/ds4-core/src/serving_check.rs`](file:///data/ds4-gemma4/crates/ds4-core/src/serving_check.rs)
- `/tmp/g4-offload.sh` and `/tmp/g4-offload-*.log`

---

## 1. Is the Multi-Row Verify Head + Device Argmax Worth Doing Before Phase 8, and What Gain Should Be Expected on `18.42 tok/s` (`draft=6`)?

**Yes — do the multi-row verify head and device `argmax_rows` before Phase 8.** Phase 8 (`cuda/gemma4_attn_mma.cuh` with `G4_ATTN_MMA_MIN_ROWS = 16`) will contribute **0% speedup** to `draft=6` (`n = 7`) decode, whereas the multi-row verify head + device `argmax_rows` eliminates **`8.86 GB` of redundant DRAM weight reads per verify step** and is expected to lift GB10 `draft=6` decode from **`18.42 tok/s` to `~20.8–21.2 tok/s` (`+12.5% to +15%`)**, crossing the 20 tok/s usable decode floor in [`AGENTS.md`](file:///data/ds4-gemma4/AGENTS.md#L204).

### Roofline & Bottleneck Attribution at `draft=6` (`n = 7` verify rows)

In [`gemma4_mtp_verify`](file:///data/ds4-gemma4/ds4_gemma4_mtp.inc#L329-L356):
```c
for (uint32_t row = 0; row < n; row++) {
    ...
    const bool ok = in && out &&
        ds4_gpu_gvq_linear(g->gvq, out, gemma4_index(m, w->token_embd), in, 1u,
            DS4_GVQ_BF16) &&
        ds4_gpu_g4_softcap(out, G4_VOCAB);
    ...
}
if (!ds4_gpu_tensor_read(r->verify, 0, r->host,
        (uint64_t)n * G4_VOCAB * sizeof(float))) { ... }
for (uint32_t i = 0; i < n; i++) {
    target[i] = sample_argmax(r->host + (uint64_t)i * G4_VOCAB, G4_VOCAB);
}
```

1. **Redundant DRAM weight traffic in the `row = 0..n-1` head loop:**
   - The trunk tied embedding table `w->token_embd` is `262,144 × 2,816` `BF16` = **`1,476,395,008 B` (`1.4764 GB`, `1.375 GiB`)**.
   - At `draft=6` (`n = 7`), looping `ds4_gpu_gvq_linear(..., 1u, DS4_GVQ_BF16)` + `ds4_gpu_g4_softcap` **streams the entire `1.4764 GB` table 7 separate times (`10.335 GB` per verify step)** across **14 kernel launches** (`7` `glmvq_bf16_dot` + `7` `g4_softcap_kernel`).
   - On **GB10 (Spark)** (`~245 GB/s` achieved BF16 streaming bandwidth, [`AGENTS.md`](file:///data/ds4-gemma4/AGENTS.md#L221)):
     - 1 pass over `1.4764 GB`: $\frac{1.4764\text{ GB}}{245\text{ GB/s}} \approx \mathbf{6.03\text{ ms}}$ (`+ ~0.12 ms` for `g4_softcap_kernel` over `262,144` FP64 `tanh` evaluations) $\approx \mathbf{6.15\text{ ms}}$.
     - 7 single-row passes: $7 \times 6.15\text{ ms} \approx \mathbf{43.0\text{ ms}}$ per verify round.
     - 1 multi-row fused pass (`g4_head_rows_kernel` reading `1.4764 GB` once and writing `7 × 1.05 MB = 7.34 MB` with fused softcap): **`~6.2–6.4 ms`**, saving **`~36.6–36.8 ms` per verify round**.
   - On **RTX 4070 SUPER**:
     - Device-resident (`~468 GB/s`, [`gemma4-port-handoff.md:1567`](file:///data/ds4-gemma4/gemma4-port-handoff.md#L1567)): `7 × 3.18 ms = 22.3 ms` $\to$ `3.25 ms` (saving **`~19.0 ms` per verify round**).
     - Host-pinned zero-copy (`/tmp/g4-offload-resident-1024.log` at `~24 GB/s` PCIe 4.0): `7 × 61.5 ms = 430.5 ms` $\to$ `61.5 ms` (saving **`~369 ms` per verify round**).

2. **Host readback (`7.34 MB`) + CPU `sample_argmax` (`1.835M` floats):**
   - `ds4_gpu_tensor_read` of `7.34 MB` plus 7 sequential single-threaded CPU scans of `262,144` floats costs **`~2.5–3.5 ms` per verify round** on GB10.
   - Replacing this on greedy verify with `ds4_gpu_argmax_rows_tensor(r->next, r->verify, G4_VOCAB, n)` (`28 B` readback of `target[0..7)`) plus a single-row (`1.05 MB`) readback of committed row `keep - 1` into `s->logits` inside `ds4_session_gemma4_commit` takes **`~0.4–0.6 ms`**, saving an additional **`~2.0–2.9 ms` per verify round**.

3. **End-to-end impact on the `18.42 tok/s` GB10 `draft=6` benchmark:**
   - At `18.42 tok/s` (`54.29 ms/token`, average `~6.1–6.4` accepted tokens/round on the 32-token benchmark in [`gemma4-port-handoff.md:1625-1629`](file:///data/ds4-gemma4/gemma4-port-handoff.md#L1625-L1629)), one `draft=6` round takes **`~330–347 ms`**.
   - Removing **`~38.6–39.7 ms` per round** (`~36.7 ms` from the 6 redundant head passes + `~2.4 ms` from the 6 redundant row readbacks and CPU argmax scans) reduces round time from `~338 ms` to `~299 ms`:
     $$\text{Expected GB10 draft=6 rate} \approx 18.42 \times \frac{338}{299} \approx \mathbf{20.8\text{–}21.2\text{ tok/s}} \quad (\mathbf{+12.5\%\text{ to }+15\%})$$

### How to Extend [`cuda/gemma4_head.cuh`](file:///data/ds4-gemma4/cuda/gemma4_head.cuh) to `1..8` Rows with **0.0 ULP Drift** and **`<= 44 KiB` Static Shared Memory**

Notice in [`g4_head_gemv_kernel`](file:///data/ds4-gemma4/cuda/gemma4_head.cuh#L32-L33):
```cuda
__shared__ __align__(16) float sx[COLS];
for (uint32_t c = threadIdx.x; c < COLS; c += 256u) { sx[c] = g4_round(input[c]); }
```
Because `g4_round(x)` rounds every input element to an exact `BF16` value (`__bfloat162float(__float2bfloat16_rn(x))`), storing the staged inputs in shared memory as `__nv_bfloat16 sx[NT][COLS]` (`uint16_t` bits, widened back via `g4_bf16_lo` / `g4_bf16_hi` when loaded as `uint4`) is **bit-for-bit identical (`0 ULP`)** to storing `float sx[NT][COLS]`, while using half the shared memory:
- For `NT = 8` rows and `COLS = 2816`: `sizeof(sx) = 8 × 2816 × 2 B = 45,056 B (44.0 KiB)`, fitting inside the **`48 KiB` static shared-memory limit** with zero dynamic-smem opt-in!
- Each warp loads `const uint4 wv = __ldcs((const uint4 *)(w + k * 256u));` **once** per column step and accumulates `float sum[NT]` across `t = 0..tokens-1` in the **exact same 8-FMA sequence and warp XOR reduction** as `rows = 1`, producing logits that are **byte-for-byte identical** to the current `row = 0..n-1` loop.

---

## 2. Is the Readback Removable for Greedy Verify of `n` Rows Without a Device-Side Multi-Row Head?

**Yes — device `argmax_rows` and the multi-row head are completely orthogonal, and `ds4_gpu_argmax_rows_tensor` can be wired in immediately over the existing `r->verify` and `r->next` tensors.**

1. **The tensors are already allocated for it:**
   - Look at [`ds4_gemma4_mtp.inc:198-201`](file:///data/ds4-gemma4/ds4_gemma4_mtp.inc#L198-L201):
     ```c
     r->verify = ds4_gpu_tensor_alloc((uint64_t)rows * G4_VOCAB * sizeof(float));
     ...
     r->next = ds4_gpu_tensor_alloc((uint64_t)rows * sizeof(int32_t));
     ```
     `r->next` was **already** allocated with size `(drafts + 1) * sizeof(int32_t)` (`rows * 4 B`), not `1 * sizeof(int32_t)`.
2. **Exact wiring in `ds4_gemma4_mtp.inc` (works with or without the multi-row head):**
   - **In [`gemma4_mtp_verify`](file:///data/ds4-gemma4/ds4_gemma4_mtp.inc#L348-L356):** replace the `n`-row `r->host` readback and `sample_argmax` loop with:
     ```c
     if (!ds4_gpu_argmax_rows_tensor(r->next, r->verify, G4_VOCAB, n) ||
         !ds4_gpu_tensor_read(r->next, 0, target, (uint64_t)n * sizeof(int32_t))) {
         fprintf(stderr, "ds4: Gemma-4 verify argmax failed\n");
         return false;
     }
     for (uint32_t i = 0; i < n; i++) {
         if (target[i] < 0 || (uint32_t)target[i] >= G4_VOCAB) { return false; }
     }
     ```
   - **In [`ds4_session_gemma4_trial`](file:///data/ds4-gemma4/ds4_gemma4_mtp.inc#L424-L426) (`temp > 0` path):** read `r->verify` into `r->host` **only** when `spec_sampling(&s->spec)` is active (matching `Ling-3.1` at [`ds4_ling31_session.inc:134-141`](file:///data/ds4-gemma4/ds4_ling31_session.inc#L134-L141) and `GLM-5.3` at [`ds4_glm53_mtp.inc:545-548`](file:///data/ds4-gemma4/ds4_glm53_mtp.inc#L545-L548)):
     ```c
     if (spec_sampling(&s->spec)) {
         if (!ds4_gpu_tensor_read(r->verify, 0, r->host,
                 (uint64_t)n * G4_VOCAB * sizeof(float))) {
             gemma4_session_fail(s, err, errlen);
             return -1;
         }
         spec_sample_rows(&s->spec, r->host, G4_VOCAB, G4_VOCAB, tokens, n, target);
     }
     ```
   - **Critical detail in [`ds4_session_gemma4_commit`](file:///data/ds4-gemma4/ds4_gemma4_mtp.inc#L456-L457):**
     Currently, line 456 does:
     ```c
     memcpy(s->logits, r->host + (size_t)(keep - 1) * G4_VOCAB,
         (size_t)G4_VOCAB * sizeof(float));
     ```
     If `gemma4_mtp_verify` no longer populates `r->host` on greedy runs, `r->host` holds stale data at commit time, which would break callers that inspect `s->logits` after a speculative step (including `s.argmax()` on the next step and `s.copy_logits(vocab)` in [`serving_check.rs:1281`](file:///data/ds4-gemma4/crates/ds4-core/src/serving_check.rs#L1281)!). Instead, read **only the single committed frontier row `(keep - 1)`** (`1.05 MB` instead of `7.34 MB`) directly from `r->verify` into `s->logits` (matching [`l31_mtp_land` in `ds4_ling31_mtp.inc:117`](file:///data/ds4-gemma4/ds4_ling31_mtp.inc#L117)):
     ```c
     if (!ds4_gpu_tensor_read(r->verify,
             (uint64_t)(keep - 1) * G4_VOCAB * sizeof(float),
             s->logits, (uint64_t)G4_VOCAB * sizeof(float))) {
         return gemma4_session_fail(s, err, errlen);
     }
     ```
3. **Standalone gain of `argmax_rows_tensor` vs paired gain with the multi-row head:**
   - `argmax_rows_tensor` alone (without a multi-row head) saves **`~2.0–2.9 ms` per round** (`~0.7%` of a `draft=6` round), because the `6.29 MB` extra PCIe/C2C readback and CPU scan are small next to the **`10.335 GB`** of weight reads in the 7-row `ds4_gpu_gvq_linear` loop (`~43 ms` on GB10).
   - Pairing `argmax_rows_tensor` with `g4_head_rows_kernel` saves the full **`~39 ms` per round** (`+12.5–15%`).

---

## 3. Corrections & Clarifications to Claims 5 and 6

### Corrections to Claim 5
1. **`ds41`'s per-row draft tower is the sequential *drafter*, not the *verify* head:**
   - Claim 5 notes that *"ds41's own draft tower is per-row too; the batched argmax_rows_kernel serves GLM53, Ling, MiniMax and Beam, not ds41."*
   - In `ds41` (`ds4.c:64856`), the **verify** head over the `n` candidate rows *does* use `ds4_gpu_argmax_rows_tensor(msl->draft_ids, msl->draft_logits, ...)`. Only the autoregressive draft generation steps (`step 0..k-1`, where draft token $i+1$ depends on draft token $i$) are per-row — just as Gemma's `gemma4_mtp_step` (`ds4_gemma4_mtp.inc:303-304`) is per-row and already uses device `ds4_gpu_argmax_tensor`. Every MTP family in this repository batches its **verify** head except `ds4_gemma4_mtp.inc:329-340`.
2. **Why `ds4_gpu_gvq_linear` cannot simply be called with `rows = n` (`n = 5` or `7`) on `w->token_embd`:**
   - In [`ds4_glmvq_gpu.cuh:785`](file:///data/ds4-gemma4/ds4_glmvq_gpu.cuh#L785), the direct BF16 dot kernel `glmvq_bf16_decode` is gated on `rows <= GLMVQ_DOT_TOKENS`, where **`GLMVQ_DOT_TOKENS = 4`** ([`cuda/glm_vq_gemv.cuh:15`](file:///data/ds4-gemma4/cuda/glm_vq_gemv.cuh#L15)).
   - At `draft=2` (`n = 3 <= 4`), `ds4_gpu_gvq_linear(..., n, DS4_GVQ_BF16)` would run `glmvq_bf16_dot`.
   - At `draft=4` (`n = 5`) and `draft=6` (`n = 7`), `rows > 4` falls off `glmvq_bf16_dot` into `glmvq_gemm` (F32$\to$BF16 convert + `cublasGemmEx` + BF16$\to$F32 convert), which both runs slower (`147–178 GB/s`, `cuda/glm_vq_gemv.cuh:8`) and changes the FP32 reduction order relative to 1-row decode!
   - That is why extending Gemma's own `g4_head_gemv_kernel` (`cuda/gemma4_head.cuh`) to `NT <= 8` (`drafts <= 6`) is required for `draft=4` and `draft=6`.
3. **Why "cheapest lever" *is* supported by existing measurements:**
   - While multi-row verify head has not been timed yet on Gemma, single-row `g4_head_gemv` was already timed in Phase 3 (`3.15 ms` for `1.476 GB` on 4070 = `468 GB/s`, [`gemma4-port-handoff.md:1567`](file:///data/ds4-gemma4/gemma4-port-handoff.md#L1567)). Because a single-row head is 93% DRAM bandwidth-bound on streaming `1.476 GB` of weights, calling it 7 times in a loop is guaranteed by DRAM byte counting (`10.34 GB` vs `1.48 GB`) to cost $7\times$ a single pass.

### Clarifications to Claim 6 (and a Hidden Default-Off Trap Affecting `18.42 tok/s`)
1. **Claim 6 is accurate on `ds41_attn_mma.cuh`, `acd8aa49`, and `Agy_phase8_review.md`:**
   - `cuda/ds41_attn_mma.cuh` is DeepSeek V4.1's sparse MLA kernel (`window + top-k FP4 ckv`, $K == V$, `dim = 512`), not GQA.
   - `acd8aa49` is GLM-5.3's absorbed-latent MLA kernel.
   - Because [`Agy_phase8_review.md`](file:///data/ds4-gemma4/Agy_phase8_review.md) sets `G4_ATTN_MMA_MIN_ROWS = 16`, Phase 8 (`cuda/gemma4_attn_mma.cuh`) only accelerates prefill (`rows >= 16`, i.e., `first_token_sec`), never 7-row verify steps.
2. **Crucial consequence of Claim 6's observation that `DS4_GEMMA4_ATTN_PREFILL` / `_FAST` (and Phase 3 kernels) are default-off:**
   - In [`ds4_gemma4_graph.inc:533-585`](file:///data/ds4-gemma4/ds4_gemma4_graph.inc#L533-L585) and [`ds4_gemma4_gpu.cuh:257-264`](file:///data/ds4-gemma4/ds4_gemma4_gpu.cuh#L257-L264), **all six bit-identical (`0.0 ULP`) kernels** (`DS4_GEMMA4_ATTN_FAST`, `DS4_GEMMA4_ATTN_PREFILL`, `DS4_GEMMA4_FUSE_ROPE`, `DS4_GEMMA4_ROUTER_PAR`, `DS4_GEMMA4_HEAD_GEMV`, `DS4_GEMMA4_ROUTER_DOT`) are still **default-off**!
   - That means the `18.42 tok/s` `draft=6` Spark benchmark (`/tmp/g4-spark-bench.sh`) and `/tmp/g4-offload.sh` ran:
     - the unoptimized `g4_attn_kernel` on both `7`-row verify and `1`-row drafter steps (re-reading K/V `group` times without shared-memory tiling),
     - separate `g4_rms` + `g4_rope` launches (`5.5 ms/step` on 4070), and
     - the **single-thread serial top-8 router scan** (`g4_router_kernel<false>`, `2.88 ms/step` on 4070)!
   - Flipping those six **bit-identical (`relL2 = 0.0`)** kernels to default-on (with `DS4_GEMMA4_*=0` kill switches) is zero-risk for exactness gates and will immediately speed up both plain decode and `draft=6` verify!

---

## 4. Recommended Order of Work (Including Findings from the Running 4070 Offload / FP8-KV Script)

### Status of `/tmp/g4-offload.sh` on the RTX 4070 SUPER
Inspection of `/tmp/g4-offload-*.log` shows why the 4070 offload / FP8-KV runs behave as they do:
1. **`resident-1024` (`/tmp/g4-offload-resident-1024.log` & `.csv`):**
   - Because `9.37 GiB` trunk + `0.58 GiB` MTP + KV cannot fit in 12 GiB VRAM with `DS4_CUDA_COPY_MODEL=1`, it ran **host-pinned zero-copy over PCIe 4.0** (`prefill_tps = 67.34`, `gen_tps = 3.81 tok/s`, `first_token_sec = 1.66 s`). Over PCIe (`~24 GB/s`), the 7-row verify head loop alone (`10.34 GB`) costs `~430 ms` per verify round!
2. **`ssd-1024` & `ssd-2048` (`/tmp/g4-offload-ssd-1024.log`):**
   - Exited immediately with `ds4: Gemma-4 SSD streaming requires coherent CUDA host page tables (the GB10 shape); this card keeps the resident path`. Discrete GPUs (`sm_89`) are intentionally refused by `ds4_gemma4_stream.inc`; SSD streaming must be benchmarked on **GB10**.
3. **`fp8-1024` & `fp8-1536` (`/tmp/g4-offload-fp8-1024.log`):**
   - Exited immediately with `ds4: DS4_GEMMA4_FULL_KV=fp8 cannot pair with --mtp: the drafter reads the full layer's bf16 KV`.
   - *(Note: if you want `DS4_GEMMA4_FULL_KV=fp8` to work with `--mtp`, it only takes a 6-line branch in [`gemma4_mtp_step` (`ds4_gemma4_mtp.inc:265`)](file:///data/ds4-gemma4/ds4_gemma4_mtp.inc#L265) to call `ds4_gpu_g4_attn_fp8` on `g->fkcache[shared]` when `full && g->full_fp8`, then removing the refusal in `ds4.c`. However, at `ctx 1024..1536`, FP8 full-KV only saves `10–15 MB` across the 5 full layers; its real value is `32K–256K` context on GB10.)*

### Concrete Execution Order

1. **First (Now — P8 Verify Head + Device Argmax + Bit-Identical Defaults):**
   - **1a.** Extend [`cuda/gemma4_head.cuh`](file:///data/ds4-gemma4/cuda/gemma4_head.cuh) with `g4_head_rows_kernel<COLS, SOFTCAP, NT>` (`NT = 1, 4, 8`, `__nv_bfloat16 sx[NT][COLS]` in `<= 44 KiB` static smem) and expose `ds4_gpu_g4_head_rows` in [`ds4_gemma4_gpu.cuh`](file:///data/ds4-gemma4/ds4_gemma4_gpu.cuh).
   - **1b.** In [`gemma4_mtp_verify`](file:///data/ds4-gemma4/ds4_gemma4_mtp.inc#L329-L356), replace the `row = 0..n-1` loop with one call to `ds4_gpu_g4_head_rows` (with fallback to the loop if disabled), replace the `n`-row `r->host` readback + host `sample_argmax` with `ds4_gpu_argmax_rows_tensor(r->next, r->verify, G4_VOCAB, n)` + `28 B` readback, read `r->verify` into `r->host` only inside `if (spec_sampling(&s->spec))` ([`ds4_gemma4_mtp.inc:424`](file:///data/ds4-gemma4/ds4_gemma4_mtp.inc#L424)), and read only row `(keep - 1)` into `s->logits` in [`ds4_session_gemma4_commit`](file:///data/ds4-gemma4/ds4_gemma4_mtp.inc#L456).
   - **1c.** Flip the 6 bit-identical (`0.0 ULP`) kernels (`ATTN_FAST`, `ATTN_PREFILL`, `FUSE_ROPE`, `ROUTER_PAR`, `HEAD_GEMV`, `ROUTER_DOT`) to default-on (disabled only when `DS4_GEMMA4_*=0`).
2. **Second (Close P8 Trial Gate & Re-Measure `draft=6` on GB10):**
   - Sync to GB10, run the new `gemma4_mtp_gate` (`--serving-check` with `--mtp`, first with `DS4_GEMMA4_EXACT_PREFILL=1` and then default), and re-run `/tmp/g4-spark-bench.sh` to record the new `draft=6` serving-default speed (`> 20 tok/s`).
   - *(Watch item for `gemma4_mtp_gate` without `DS4_GEMMA4_EXACT_PREFILL=1`: in [`serving_check.rs:1263, 1275, 1294, 1388`](file:///data/ds4-gemma4/crates/ds4-core/src/serving_check.rs#L1263-L1294), `mtp_plain_greedy` runs `rows = 1` while `mtp_capture` runs `rows = 7` and `mtp_replay`/`stop` run `rows = tokens.len()`. If Q4_K MMA (`glmvq_q4m_decode`) or router cuBLAS (`rows > 4`) flips a near-tie between `rows = 1`, `rows = 3`, and `rows = 7` when `!exact`, either gate `gemma4_mtp_verify`'s `gemma4_forward` on the 1-row projections for small `n <= 7` or apply the same noise-floor tolerance used by `reuse/prefix` when `!exact`.)*
3. **Third (Close P9 OpenAI Serving Gate + Optional GB10 SSD / FP8-KV Runs):**
   - Run `tests/gemma4_serving_http_gate.sh` (P9).
   - Run `--ssd-streaming` and long-context `DS4_GEMMA4_FULL_KV=fp8` on **GB10** (where coherent unified memory and 128 GB capacity exist).
4. **Fourth (Phase 8 Tensor-Core Prefill Attention `cuda/gemma4_attn_mma.cuh`):**
   - After P8 and P9 are closed, land `cuda/gemma4_attn_mma.cuh` (`<256, 2>` and `<512, 8>`, `G4_ATTN_MMA_MIN_ROWS = 16`) per [`Agy_phase8_review.md`](file:///data/ds4-gemma4/Agy_phase8_review.md) and benchmark 8K–32K prefill on GB10.
