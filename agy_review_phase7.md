# Gemma-4 26B A4B VQ — Phase 7 (P8: MTP Drafter) Technical Review

**Repository**: `/data/ds4-gemma4` (`feat/gemma4-family`, after `8fae06b0` and `f0c2b92a`)  
**Scope**: Phase 7 P8 qualification, wrapped-ring gate, `G4_MTP_DRAFTS` selection, `--serving-check --mtp`, speed measurement plan, and pre-closure re-gating.

---

## Executive Summary of Critical Findings

Before running the 6 steps below, note four code-level findings uncovered during this review that directly affect the wrapped-ring gate and P8 closure:

1. **Why the 1300-token wrapped-ring pair took 12.7 minutes (and when it fails to wrap at all)**:
   - In `crates/ds4-core/src/serving_host.rs:219`, `GEMMA4_NATIVE_DEFAULT` is still `1`, and `fill_quote_facts` (`serving_host.rs:365`) exports `DS4_GEMMA4_PREFILL_CHUNK=1` into `plan.env_overrides()`. When `DS4_CUDA_COPY_MODEL=1` is not set on the 12 GiB RTX 4070 SUPER, `ds4` runs 1,300 serial 1-row forwards over PCIe zero-copy (~3.9 tok/s prefill).
   - Conversely, when `DS4_GEMMA4_PREFILL_CHUNK` is **not** forced to `1` (e.g., direct C harness or `DS4_GEMMA4_PREFILL_CHUNK=512`, matching `gemma4_prefill_cap` in `ds4_gemma4_bind.inc:268`), `ds4_gemma4_kv_rows(il, 2048, 512)` allocates a ring of `1024 + max(512, 5) = 1536` rows (`ds4_gemma4.h:61-65`). **A 1300-token prompt does not wrap a 1536-row ring (`1300 < 1536`)!** To wrap the sliding ring **and** exercise the `G4_MTP_DRAFTS + 1` ring margin (`window + max(chunk, drafts + 1)`), the run must have `chunk <= drafts + 1` (e.g., `DS4_GEMMA4_PREFILL_CHUNK=1`, giving `ring = 1029`) and `prompt_len >= ring` (e.g., `1032` tokens).

2. **Stale bit-identity assumption in `gemma4_mtp_verify` (`ds4_gemma4_mtp.inc:314-325`)**:
   - `ds4_gemma4_mtp.inc:314-316` states: *"the graph keeps one-row call shapes per row, so the block forward is bit-identical to decoding the rows one at a time"*.
   - **Commit `8fae06b0` invalidated that assumption when `DS4_GEMMA4_EXACT_PREFILL` is unset.** `gemma4_mtp_verify` calls `gemma4_forward(s->model, s, r->tokens, start, n, start + n)` with `n = drafts + 1 = 5`. For `rows = 5` (`rows > 1`) with `DS4_GEMMA4_EXACT_PREFILL=0`:
     - `gemma4_projection_rows` (`ds4_gemma4_graph.inc:509-515`) dispatches `ds4_gpu_gvq_linear_many` (`q4m` 2..8-row MMA kernel) instead of 1-row `ds4_gpu_gvq_linear`.
     - `l->dense_down` (`ds4_gemma4_graph.inc:763-774`) dispatches multi-row `ds4_gpu_matmul_q8_0_tensor` (cuBLAS) instead of 1-row `warp8` and ignores `DS4_GEMMA4_VQ_GEMV`.
     - `g4_moe_run` (`ds4_gemma4_gpu.cuh:802-835`) dispatches the expert-grouped `glmvq_restore` + cuBLAS path (`glmvq_gemm`) instead of the 1-row direct VQ GEMV (`g4_moe_linear_one` $\rightarrow$ `g4_vq_gemv`).
   - Consequently, even with a 100% bug-free KV ring, a width-5 verify forward (`--mtp`) and a width-1 decode forward (no `--mtp`) execute different floating-point reduction orders across all 30 layers. At 1,032+ tokens, a ~2 ULP shift can cross a tie in the discontinuous top-8 router (`ds4_gemma4_graph.inc:818`) and diverge from plain decode unless verify uses the 1-row path (or both sides use the same path).

3. **Missing `spec_sample_rows` hook in `gemma4_mtp_verify` (`ds4_gemma4_mtp.inc:353-356`)**:
   - Per `docs/speculative-sampling.md` ("New model families") and `AGENTS.md`, a family's trial must fill `target` with `spec_sample_rows(&s->spec, ...)` when `spec_sampling(&s->spec)` is true, and `sample_argmax` otherwise. `gemma4_mtp_verify` currently hardcodes `sample_argmax` for all verify rows.

4. **Missing post-wrap guard in `gemma4_sync_keep` (`ds4_gemma4_session.inc:39-46`) and `gemma4_bank_fork` (`ds4_gemma4_batch.inc:124-135`)**:
   - Once a session or bank advances past `ring` (`position > ring`), slots `(k % ring)` for $k < \text{position} - \text{ring}$ have been overwritten by newer tokens. Rewinding to an earlier prefix `start` without checking `start >= old_len - G4_WINDOW` (or `pos >= sg->position - G4_WINDOW`) leaves corrupted keys inside `[start - G4_WINDOW + 1, start]`.

---

## 1. Making the Wrapped-Ring Gate Decisive and Cheaper

### 1.1 Why 1300 tokens / ctx 2048 / 12 tokens was slow and potentially non-wrapping

The sliding-layer ring size in `ds4_gemma4.h:61-65` is:
$$\text{ring} = \text{G4\_WINDOW} + \max(\text{cap}, \text{G4\_MTP\_DRAFTS} + 1) = 1024 + \max(\text{cap}, 5)$$

- **Case A (`cap = 512`, the C default in `ds4_gemma4_bind.inc:268`)**: $\text{ring} = 1024 + 512 = 1536$. A 1,300-token prompt writes positions $0 \dots 1299$ into slots $0 \dots 1299$. **It never wraps the 1536-row ring** and never tests the `G4_MTP_DRAFTS + 1` ring margin because `cap = 512 >> 5`.
- **Case B (`cap = 1`, forced by `serving_host.rs:219` `GEMMA4_NATIVE_DEFAULT = 1`)**: $\text{ring} = 1024 + \max(1, 5) = 1029$. It wraps at position $1029$, **but** prefilling 1,300 tokens 1 row at a time without `DS4_CUDA_COPY_MODEL=1` streams ~14 GB of weights over PCIe 1,300 times (12.7 minutes per run).

### 1.2 Shorter, Targeted Variant That Exercises Both Wrap and the Verify Block

To exercise **both** (a) the `G4_MTP_DRAFTS + 1` ring margin (`ring = 1024 + 5 = 1029`) and (b) the wrap boundary inside a 5-row verify block and subsequent drafter steps:

1. **Pin `DS4_GEMMA4_PREFILL_CHUNK=1` for `ring = 1029`**, **OR** temporarily decouple the prefill batch size from the ring allocation in a test knob, **OR** run with `DS4_CUDA_COPY_MODEL=1` and a **1032-token** prompt at `-c 1056 -n 12`:
   - Wait: if we want **fast prefill** (chunked) **and** a **tight ring** (`ring = 1029`, testing `window + drafts + 1`), why is `ring` sized by `cap`? Because batched prefill of `cap` rows writes `cap` rows before attention runs (`ds4_gemma4_graph.inc:618-625`).
   - With `DS4_CUDA_COPY_MODEL=1` and `DS4_GEMMA4_PREFILL_CHUNK=64`, $\text{ring} = 1024 + 64 = 1088$:
     - Prefill of **1,092 tokens** takes **18 chunk launches (~3.2 seconds total on resident VRAM)** instead of 12.7 minutes!
     - At position $1092$, slots $0, 1, 2, 3$ have wrapped (overwritten by positions $1088, 1089, 1090, 1091$).
   - Even better, to test the exact `window + drafts + 1 = 1029` ring margin where `cap <= drafts + 1`:
     - Set `DS4_CUDA_COPY_MODEL=1 DS4_GEMMA4_PREFILL_CHUNK=4` (`cap = 4 < 5`, so $\max(4, 5) = 5 \implies \text{ring} = 1029$!).
     - Prefill **1,032 tokens** (`258` launches of 4 rows with resident weights $\approx 9\text{ s}$, vs 762 s before) and decode `-n 8 --temp 0`.
     - Trace of positions $1027 \dots 1039$ on $\text{ring} = 1029$:
       - Prompt positions $0 \dots 1028$ fill slots $0 \dots 1028$.
       - Prompt positions $1029, 1030, 1031$ wrap to slots $0, 1, 2$.
       - Token 0 is sampled at position $1031$; first MTP trial starts at `start = 1032` (slot $3$):
         - Drafter runs 4 steps held at `position = 1032, keys = 1032`, reading keys $[1032 - 1023, 1031] = [9, 1031]$, which spans slots $9 \dots 1028$ **and wrapped slots $0, 1, 2$**.
         - Verify block (`n = 5`) writes positions $1032 \dots 1036$ into wrapped slots $3 \dots 7$.
         -Verify row $r = 0$ (pos $1032$) reads keys $[9, 1032]$ (slots $9 \dots 1028, 0 \dots 3$) while slots $4 \dots 7$ hold verify rows $r = 1 \dots 4$ (positions $1033 \dots 1036$, which used to hold positions $4 \dots 7$ — just outside $[9, 1032]$ by the exact `drafts + 1` margin!).

### 1.3 Exact Command Pair for the Fast Wrapped-Ring Gate (~10s per run)

Note: Include `DS4_GEMMA4_EXACT_PREFILL=1` on both runs first so the width-5 verify block uses the exact same 1-row kernel sequence as width-1 decode (eliminating cross-width FP reduction noise when testing ring correctness):

```bash
PROMPT_1032=$(python3 -c 'print(" ".join(["the"] * 1030))')

# Run 1: Drafting OFF (ring = 1029 because cap = 4 and max(4, 5) = 5)
DS4_CUDA_COPY_MODEL=1 \
DS4_GEMMA4_PREFILL_CHUNK=4 \
DS4_GEMMA4_EXACT_PREFILL=1 \
DS4_GEMMA4_ATTN_PREFILL=1 DS4_GEMMA4_ATTN_FAST=1 DS4_GEMMA4_FUSE_ROPE=1 \
DS4_GEMMA4_ROUTER_PAR=1 DS4_GEMMA4_HEAD_GEMV=1 DS4_GEMMA4_ROUTER_DOT=1 \
./ds4 -m /data/models/gemma-4-26B-A4B-it-vq-release \
  -c 1056 -n 12 --temp 0 -p "$PROMPT_1032" > /tmp/g4_wrap_off.txt

# Run 2: Drafting ON (--mtp-draft 4, same ring = 1029)
DS4_CUDA_COPY_MODEL=1 \
DS4_GEMMA4_PREFILL_CHUNK=4 \
DS4_GEMMA4_EXACT_PREFILL=1 \
DS4_GEMMA4_ATTN_PREFILL=1 DS4_GEMMA4_ATTN_FAST=1 DS4_GEMMA4_FUSE_ROPE=1 \
DS4_GEMMA4_ROUTER_PAR=1 DS4_GEMMA4_HEAD_GEMV=1 DS4_GEMMA4_ROUTER_DOT=1 \
./ds4 -m /data/models/gemma-4-26B-A4B-it-vq-release \
  --mtp /data/models/gemma-4-26B-A4B-it-vq-release/gemma4-26b-a4b-it-mtp.sft \
  --mtp-draft 4 \
  -c 1056 -n 12 --temp 0 -p "$PROMPT_1032" > /tmp/g4_wrap_on.txt

cmp /tmp/g4_wrap_off.txt /tmp/g4_wrap_on.txt
```

### 1.4 Direct Debug Probe to Assert Ring Invariants Without a Long Run

Yes — we can assert the ring geometry and slot non-collision directly on the host in `gemma4_layer` (`ds4_gemma4_graph.inc:618-634`) and on the device in `g4_attn_kernel` (`cuda/gemma4_attn.cuh:66-76`) behind `DS4_GEMMA4_DEBUG=1` (or a dedicated `DS4_GEMMA4_RING_ASSERT=1`):

```c
/* In gemma4_layer (ds4_gemma4_graph.inc:618), before ds4_gpu_g4_kv_store: */
if (gemma4_debug() && window > 0) {
    /* 1. Every written row [first, first + rows) has a unique slot in [0, ring). */
    assert(rows <= ring);
    /* 2. The earliest key needed by row r=0 is begin_0 = max(0, first + 1 - window).
     *    When keys > first (normal prefill / verify), row r=rows-1 writes slot
     *    (first + rows - 1) % ring. That write must not clobber any key in
     *    [begin_0, min(first, keys - 1)] still needed by row r=0:
     *    distance = (first + rows - 1) - begin_0 <= rows + window - 2 < ring. */
    int begin_0 = first + 1 - window;
    if (begin_0 < 0) begin_0 = 0;
    int max_span = (first + rows - 1) - begin_0 + 1;
    if (max_span > ring) {
        fprintf(stderr,
                "GEMMA4 RING OVERWRITE: layer=%d first=%d rows=%d keys=%d "
                "window=%d ring=%d span=%d\n",
                il, first, rows, keys, window, ring, max_span);
        abort();
    }
    /* 3. For MTP drafter steps (keys == first, rows == 1):
     *    last = keys - 1 = first - 1; span is (first) - begin_0 + 1 <= window <= ring. */
    if (keys == first) {
        assert(rows == 1);
        assert(first - begin_0 < window);
        assert(window < ring);
    }
}
```

With this assertion compiled in, even a synthetic check or a 5-second run with `DS4_GEMMA4_PREFILL_CHUNK=4` across the wrap boundary (`first = 1028..1036`) proves mechanically that no verify block or drafter step can ever overwrite an active sliding-window key.

---

## 2. If the Two Outputs Differ: Inspection Order and Invariants to Assert

If `/tmp/g4_wrap_off.txt` and `/tmp/g4_wrap_on.txt` differ, inspect in this exact order:

### Step 1: Isolate Cross-Width Kernel Drift vs. Ring / Logic Bug

Look at `ds4_gemma4_mtp.inc:325`:
```c
if (gemma4_forward(s->model, s, r->tokens, start, n, start + n) != 0) return -1;
```
When `n = drafts + 1` (`5`), `gemma4_forward` runs with `rows = 5`. As noted in the Executive Summary, commit `8fae06b0` changed `rows > 1` (`!gemma4_exact_prefill()`) to dispatch:
- `ds4_gpu_gvq_linear_many` (`q4m` multi-row MMA) instead of 1-row `ds4_gpu_gvq_linear` (`ds4_gemma4_graph.inc:509`),
- `ds4_gpu_matmul_q8_0_tensor` instead of 1-row `warp8` (`ds4_gemma4_graph.inc:763`),
- `glmvq_restore` + cuBLAS (`g4_moe_run` multi-row branch, `ds4_gemma4_gpu.cuh:812`) instead of 1-row `g4_vq_gemv` (`ds4_gemma4_gpu.cuh:616`).

**Immediate test**: Run both off and on with `DS4_GEMMA4_EXACT_PREFILL=1 DS4_GEMMA4_VQ_GEMV=0`.
- If they match with `DS4_GEMMA4_EXACT_PREFILL=1 DS4_GEMMA4_VQ_GEMV=0` and diverge without it, **the ring is clean** and the divergence is purely the `rows = 5` vs `rows = 1` kernel path split in `gemma4_forward`.
- **Fix (also required for verify speed in Section 5)**: Either loop `r = 0..n` one row at a time inside `gemma4_mtp_verify` (or when `rows <= G4_MTP_DRAFTS + 1` in the MoE/dense-down/VQ dispatch if 1-row GEMV is faster or required for exactness), **and** make `DS4_GEMMA4_EXACT_PREFILL=1` disable `g4_vq_gemv()` at `rows == 1` (`ds4_gemma4_gpu.cuh:616` and `ds4_gemma4_graph.inc:763`) so `rows == 1` and `rows > 1` call identical kernels under `DS4_GEMMA4_EXACT_PREFILL=1`.

### Step 2: Check Position-0 / Post-Drop Hidden State Priming

In `ds4_session_gemma4_trial` (`ds4_gemma4_mtp.inc:390-394`):
```c
if (s->gemma4_mtp->hidden_pos != start - 1 || max_drafts <= 0) {
    if (gemma4_mtp_verify(s, &token, 1, target) != 0) return -1;
    s->gemma4_mtp->last_verify = 1;
    return 0;
}
```
Assert that at the first decode step after prefill (`start = prompt_len`), `s->gemma4_mtp->hidden_pos == prompt_len - 1` (published by `gemma4_mtp_publish_hidden(s, first + rows - 1, rows - 1)` at the end of `gemma4_forward` in `ds4_gemma4_graph.inc:894`).

### Step 3: Assert the Five Drafter & Verify Invariants

If divergence persists under `DS4_GEMMA4_EXACT_PREFILL=1 DS4_GEMMA4_VQ_GEMV=0`, assert these 5 invariants in `ds4_gemma4_mtp.inc` and `cuda/gemma4_attn.cuh`:

| # | Invariant | Formula / Condition | Where to Assert |
|---|---|---|---|
| **I1** | **Ring capacity** | `ring == 1024 + max(g->cap, G4_MTP_DRAFTS + 1)` on sliding layers (`il % 6 != 5`); `ring == g->ctx` on full layers (`il % 6 == 5`). Also `max_drafts + 1 <= ring - 1024`. | `gemma4_layer` (`ds4_gemma4_graph.inc:614`) and `ds4_session_gemma4_trial` (`ds4_gemma4_mtp.inc:388`) |
| **I2** | **Verify block rows & causality** | In `gemma4_mtp_verify(s, tokens, n, target)`: `start = g->position`, `first = start`, `rows = n`, `keys = start + n`. For each verify row $r \in [0, n)$, `position = start + r`, `last = min(keys - 1, position) = start + r`, and `begin = (window > 0) ? max(0, start + r + 1 - 1024) : 0`. Row $r$ reads $[ \text{begin}, \text{start} + r ]$ and writes slot $(\text{start} + r) \bmod \text{ring}$. | `g4_attn_kernel` (`cuda/gemma4_attn.cuh:66-76`) |
| **I3** | **No verify block row overlap** | For any two rows $r_1 < r_2$ in the verify block ($0 \le r_1 < r_2 < n$) and any key $k \in [\text{begin}(r_1), \text{start} + r_1]$ read by $r_1$: $(\text{start} + r_2) \bmod \text{ring} \neq k \bmod \text{ring}$. Since $(\text{start} + r_2) - \text{begin}(r_1) \le (n - 1) + (1024 - 1) = 1023 + n - 1 < 1024 + n \le \text{ring}$, this holds iff `n <= ring - 1024 + 1`. | `ds4_gemma4_graph.inc:618` |
| **I4** | **Drafter `keys = start` window** | In `gemma4_mtp_step(s, token, hidden, start)` (`ds4_gemma4_mtp.inc:265-267`): `first = start`, `rows = 1`, `keys = start`. In `g4_attn_kernel`: `position = start`, `last = min(start - 1, start) = start - 1`, `begin = max(0, start + 1 - 1024) = max(0, start - 1023)`. The drafter attends to **at most 1023 past trunk keys** $[ \max(0, \text{start} - 1023), \text{start} - 1 ]$ and **never** reads slot $\text{start} \bmod \text{ring}$ (nor any rejected trial slot $\ge \text{start}$). | `gemma4_mtp_step` (`ds4_gemma4_mtp.inc:265`) & `cuda/gemma4_attn.cuh:68` |
| **I5** | **Position held at `start = input_ids.shape[1] - 1` & Commit Rollback** | Across all $i \in [0, \text{max\_drafts})$ calls to `gemma4_mtp_step`, `start` is unchanged (`start == g->position` before verify), `r->positions[0] == start`, and `g->position` is not incremented by `gemma4_mtp_step`. After `gemma4_mtp_verify` advances `g->position = start + n`, `ds4_session_gemma4_commit(s, keep)` sets `g->position = start + keep` (`1 <= keep <= n`) and copies row `keep - 1` of `r->verify_hidden` (`position = start + keep - 1`) to `r->last_hidden` with `r->hidden_pos = start + keep - 1`. | `ds4_session_gemma4_trial` & `ds4_session_gemma4_commit` (`ds4_gemma4_mtp.inc:408-456`) |

---

## 3. Deciding `G4_MTP_DRAFTS` (2 vs. 4 vs. 6) with Measurement

### 3.1 Architecture vs. Rollout Horizon

In `docs/gemma4-reference-semantics.md` §4 (`SinglePositionMultiTokenCandidateGenerator`):
- `G4_MTP_LAYERS = 4` is the assistant sub-model's **depth** (layers 30..33: 3 sliding + 1 full). This is fixed by the checkpoint tensor layout.
- `G4_MTP_DRAFTS` is the autoregressive **rollout count** $D$ (`num_assistant_tokens = 6` in HF `generation_config.json`, currently `#define G4_MTP_DRAFTS 4` in `ds4_gemma4_plan.h:35`).
- Because every draft step re-runs the same 4 assistant layers at `position = start` without writing KV (`ds4_gemma4_mtp.inc:245-307`), raising the compile-time ceiling `G4_MTP_DRAFTS` from `4` to `6` only increases:
  1. Sliding KV ring by **2 rows** (`1029 -> 1031` when `cap <= 7`), adding `2 * 25 * 2 * 2 * 512 * 2 = 409,600` bytes (400 KiB) to a session KV buffer.
  2. `gemma4_mtp_scratch_bytes()` (`ds4_gemma4_plan.h:158-174`) by 2 verify rows (`n = 5 -> 7`): device scratch grows from `7,584,792` B (7.23 MiB) to `9,688,096` B (9.24 MiB), and host logits buffer grows from `5.24 MiB` to `7.34 MiB`.

**Recommendation on compile-time ceiling**: Set `#define G4_MTP_DRAFTS 6` in `ds4_gemma4_plan.h:35` (and `GEMMA4_MTP_DRAFTS = 6` in `serving_host.rs:225`) so a single binary can benchmark `--mtp-draft 2`, `--mtp-draft 4`, and `--mtp-draft 6` at runtime without recompiling, then set the default runtime `mtp_drafts` to the winner of the matrix below.

### 3.2 Why `D = 2` Often Beats `D = 4` or `D = 6` on a 12 GiB Card

Inspect the per-trial cost in `ds4_gemma4_mtp.inc`:
1. **Per draft step ($D$ times)**:
   - 4 assistant layers (3 sliding + 1 full + 4 MoE blocks at `d_model = 1024`) + `ds4_gpu_gvq_linear` on the tied `262,144 x 1024` BF16 embedding table (`536.87 MB` weight read **per draft step**!) + synchronous `cudaMemcpy` of `262,144 * 4 = 1.05 MB` (`ds4_gpu_tensor_read(&r->logits, ...)` at `ds4_gemma4_mtp.inc:300`) + CPU `sample_argmax` over 262,144 floats.
2. **Per verify step ($n = D + 1$ rows)**:
   - `gemma4_forward` for $n = D + 1$ rows: currently falls off the 1-row `g4_vq_gemv` path onto `glmvq_restore` + cuBLAS!
   - Vocabulary head loop (`ds4_gemma4_mtp.inc:335-346`): calls `ds4_gpu_gvq_linear` on the `262,144 x 2816` BF16 embedding table **$n = D + 1$ separate times** (`1.476 GB` read $\times (D + 1)$ = **4.43 GB** at $D=2$, **7.38 GB** at $D=4$, **10.33 GB** at $D=6$!), followed by a `ds4_gpu_tensor_read` of $(D + 1) \times 1.05\text{ MB}$ to host RAM!
3. **Diminishing marginal acceptance at positions $k \ge 3$**:
   - Because the Gemma-4 drafter holds `position = start` constant and masks out prior draft tokens (`keys = start`), draft token $k$ attends only to the trunk prompt $[0, \text{start}-1]$ plus the point-wise sum of embeddings `embed(tok_k) + proj(hidden_k)`. Conditional accuracy drops sharply after $k = 2$.

### 3.3 Experiment Matrix over $D \in \{2, 4, 6\}$

Run with `DS4_CUDA_COPY_MODEL=1` and all Phase 1–5 default flags enabled, `--temp 0`, generating `N = 128` tokens across 3 representative prompt workloads (1: deterministic code/JSON, 2: reasoning `<|channel>thought`, 3: factual prose):

| Context Regime | Prompt Length | `-c` | Draft Depths $D$ (`--mtp-draft`) | Recorded Counters & Timings |
|---|---|---|---|---|
| **Short (`ctx 256`)** | 64 tokens | 256 | `0` (off), `2`, `4`, `6` | `trials`, `spec_drafts`, `spec_hits`, per-position conditional acceptance $\alpha_k$ ($k=1\dots D$), mean accepted tokens/verify $\tau = 1 + \frac{\text{spec\_hits}}{\text{trials}}$, mean $t_{\text{draft}}(D)$ (ms), mean $t_{\text{verify}}(D+1)$ (ms), net `gen_tps` (tok/s) |
| **Long / Wrapped (`ctx 2048`)** | 1,280 tokens | 2048 | `0` (off), `2`, `4`, `6` | Same metrics as above (exercises wrapped sliding ring at `pos > 1029` and longer full-attention KV in layers 5, 11, 17, 23, 29, and MTP layer 3) |

### 3.4 Decision Rule for Default Draft Count $D^*$

Let:
- $t_1$ = plain 1-token decode latency (`--mtp` off),
- $t_{\text{trial}}(D) = D \cdot t_{\text{step}} + t_{\text{verify}}(D+1)$ = wall time of one speculative trial at depth $D$,
- $\tau(D) = 1 + \frac{\text{spec\_hits}(D)}{\text{trials}(D)}$ = average tokens committed per trial.

The net speedup factor is:
$$S(D) = \frac{\text{gen\_tps}(D)}{\text{gen\_tps}(0)} = \frac{\tau(D) \cdot t_1}{t_{\text{trial}}(D)}$$

**Rule**:
1. **Hard correctness prerequisite**: $D$ must pass the wrapped-ring parity gate (`cmp` identical to $D=0$ under greedy decoding with matched kernel path).
2. **Marginal step efficiency**: Increment $D \to D + 2$ only if the marginal accepted tokens $\Delta \tau = \tau(D+2) - \tau(D)$ more than pays for the incremental trial time $\Delta t_{\text{trial}} = t_{\text{trial}}(D+2) - t_{\text{trial}}(D)$:
   $$\frac{\Delta \tau}{\Delta t_{\text{trial}}} > \frac{\tau(D)}{t_{\text{trial}}(D)}$$
   at **both** `ctx 256` and `ctx 2048` averaged across the 3 prompt classes, with at least a **+3% net `gen_tps` gain** at `ctx 2048` to justify the higher verify latency variance. Otherwise choose the smaller $D$ (e.g., $D^* = 2$).

---

## 4. What `--serving-check --mtp` Must Prove and Whether `DS4_GEMMA4_EXACT_PREFILL` Masks Ring Bugs

### 4.1 What `./ds4 --serving-check --mtp` Actually Tests Today (and What It Misses)

Inspecting `crates/ds4-core/src/serving_check.rs`:
- `session_items()` (`serving_check.rs:285-626`) and `bank_items()` (`serving_check.rs:628-887`) call `s.eval(token)` (plain 1-token forward), **not** speculative trials!
- In `unchecked(c)` (`serving_check.rs:918-921`), `mtp / lossless` is explicitly listed as `"no check yet"`!
- Passing `--mtp <path>` to `./ds4 --serving-check` only populates `opts.mtp_path` (`serving_check.rs:1012`), which causes `Model::load` to bind the MTP sidecar (`gemma4_mtp_create`) and prints `Sidecar qualified Serial` in the report table (`serving_check.rs:1164-1168`), **without executing a single speculative trial (`ds4_session_gemma4_trial`) during the 10 checks!**

Therefore, after the ring change (`window + max(chunk, drafts + 1)`), qualifying `--mtp` requires proving **four additional invariants** beyond `./ds4 --serving-check --mtp`:

1. **Post-Reuse / Post-Rewind Hidden-State Drop (`gemma4_mtp_drop_hidden`)**:
   - When a session reuses a prefix (`reuse/extend`, `reuse/prefix`, `reuse/identical`), restores a snapshot (`snapshot/session`), or loads disk KV (`disk/session`), `gemma4_mtp_drop_hidden(s)` sets `hidden_pos = -1`.
   - Prove that on the very first token decoded after any prefix reuse / snapshot restore / disk restore with `--mtp` active, `ds4_session_gemma4_trial` detects `hidden_pos != start - 1` (`ds4_gemma4_mtp.inc:390`), runs a 1-token prime verify (`last_verify = 1`), publishes `last_hidden` at `start`, and resumes full drafting on the second decoded token — producing bytes identical to `--mtp` off.
2. **Uncommitted Reject Tail Does Not Corrupt Snapshot or Prefix Reuse**:
   - When a trial at `start` drafts $D = 4$ tokens ($n = 5$) and only `keep < 5` tokens are accepted, `gemma4_mtp_verify` wrote KV slots $(\text{start} + \text{keep}) \bmod \text{ring} \dots (\text{start} + 4) \bmod \text{ring}$, and `ds4_session_gemma4_commit` rewound `g->position = start + keep`.
   - In a **wrapped** ring ($\text{start} \ge \text{ring}$), those $n - \text{keep}$ rejected verify rows overwrote slots that previously held positions $\text{start} + \text{keep} - \text{ring} \dots \text{start} + 4 - \text{ring}$.
   - Why is that safe for normal forward decoding? Because at position $\text{start} + \text{keep}$, the sliding window only reads back to $\text{start} + \text{keep} + 1 - 1024 > \text{start} + 4 - \text{ring}$ (since $\text{ring} \ge 1024 + 5 \implies \text{ring} - 1023 \ge 6 > 4$).
   - **Where it CAN fail**: If the caller **rewinds** the session (`reuse/prefix`) back to a position $P < \text{start} + \text{keep}$ whose sliding window $[P - 1023, P]$ includes those clobbered slots $\text{start} + 4 - \text{ring}$! That requires tightening the ring-resume guard in `gemma4_sync_keep` (see Section 6.3).
3. **Finishing Inside an Accepted Speculative Prefix (`max_tokens` Mid-Trial)**:
   - Per `AGENTS.md` ("New model-family serving contract"), test a multi-turn conversation with `--mtp` where Turn 1 stops (`-n` / `max_tokens`) in the middle of an accepted multi-token verify batch, and Turn 2 extends Turn 1 (`cached_tokens > 0`) and matches `--mtp` off.
4. **Sampled Speculation (`temp > 0`)**:
   - Once `spec_sample_rows` is wired into `gemma4_mtp_verify` (`ds4_gemma4_mtp.inc:353`), verify that seeded `temp > 0` runs are deterministic across identical seeds.

### 4.2 Can `DS4_GEMMA4_EXACT_PREFILL=1` Mask a Ring Bug?

**Directly: No. Indirectly (if confused with `DS4_GEMMA4_PREFILL_CHUNK=1`): Yes.**

- **What `DS4_GEMMA4_EXACT_PREFILL=1` actually does (`ds4_gemma4_graph.inc:500-515, 763, 802` and `ds4_gemma4_gpu.cuh:804`)**:
  - It only switches the linear/MoE compute kernels (`gemma4_projection_rows`, `dense_down`, `g4_moe_run`) to loop 1 row at a time.
  - It **does not** change `g->cap`, `ring = ds4_gemma4_kv_rows(il, ctx, cap)`, `ds4_gpu_g4_kv_store` (which still writes all `rows` in the chunk at once), or `ds4_gpu_g4_attn` (which still runs all `rows` in the chunk against `ring`).
  - Therefore, `DS4_GEMMA4_EXACT_PREFILL=1` **cannot** mask a KV ring indexing or wrap bug during prefill or verify.
- **What DOES mask ring bugs (and must be avoided in the ring gate)**:
  1. **`DS4_GEMMA4_PREFILL_CHUNK=512` (large `cap`)**: Sets `ring = 1024 + max(512, 5) = 1536`. This masks:
     - Any wrap bug on prompts $\le 1536$ tokens, and
     - Any bug in the `G4_MTP_DRAFTS + 1` term of `max(cap, G4_MTP_DRAFTS + 1)`, because `cap = 512` dominates `G4_MTP_DRAFTS + 1 = 5`, giving the verify block a massive 512-row safety cushion instead of a tight 5-row cushion!
  2. **`DS4_GEMMA4_PREFILL_CHUNK=1` without `--mtp`**: Sets `rows = 1` on every prefill call and `rows = 1` on every decode call, which completely masks **intra-batch** ring overwrite bugs (where row $r > 0$ of a multi-row call overwrites a slot still needed by row $0$ of the *same* call).
  - **Conclusion**: To test the tight `window + drafts + 1` ring cushion without masking anything, you **must** run with `DS4_GEMMA4_PREFILL_CHUNK=1` (or `4`, i.e. $\text{cap} \le \text{drafts} + 1$ so $\text{ring} = 1024 + \text{drafts} + 1 = 1029$) **and** `--mtp` enabled (`verify` runs `rows = drafts + 1 = 5`) across a prompt $> 1029$ tokens.

---

## 5. Speed Measurement Plan (`ds4-bench`) and Separating Draft Overhead from Acceptance Gains

### 5.1 Pre-Measurement Fix: Stop Verify From Falling Back to Slow Multi-Row / Host Paths

Before benchmarking `--mtp` speed, look at why a 5-row verify step (`gemma4_mtp_verify`, `ds4_gemma4_mtp.inc:325-350`) is currently penalized on the RTX 4070 SUPER:
1. **MoE verify fallback**: `gemma4_forward(..., rows = 5)` calls `g4_moe_run(..., rows = 5)` (`ds4_gemma4_gpu.cuh:802`), which skips the fast 1-row `g4_vq_gemv` decode kernel (`rows == 1`) and instead runs `ds4_gpu_g4_moe_plan` + 60 `glmvq_restore` launches + cuBLAS! For $n = 3\dots 5$ rows, calling the 1-row `g4_vq_gemv` path in a loop (or allowing `g4_moe_linear_one` for $r = 0\dots n-1$) avoids materializing restored BF16 expert weights.
2. **Vocabulary head fallback**: In `ds4_gemma4_mtp.inc:338` and `293`:
   - `gemma4_mtp_verify` calls `ds4_gpu_gvq_linear` instead of `ds4_gpu_g4_head` (`g4_head_gemv()`), missing the Phase 3 fast BF16 head GEMV (`cuda/gemma4_RMS_head.cuh`).
   - Worse, `gemma4_mtp_verify` copies `n * 262,144 * 4` bytes (`5.24 MB` for $n=5$) from GPU to CPU on every verify step (`ds4_gpu_tensor_read` at `ds4_gemma4_mtp.inc:350`) and runs a serial CPU `sample_argmax` over `5 * 262,144 = 1.31M` floats (`ds4_gemma4_mtp.inc:355`), and `gemma4_mtp_step` copies `1.05 MB` D2H per draft step (`ds4_gemma4_mtp.inc:300`)! When greedy (`!spec_sampling(&s->spec)`), a device-side argmax (or `g4_head_gemv`) eliminates `~9.4 MB` of D2H transfers and `2.36M` CPU float comparisons per 4-draft trial.

### 5.2 `ds4-bench` Matrix

Run with a single resident GPU process (`DS4_CUDA_COPY_MODEL=1`, checking `nvidia-smi` first so no competing job is resident):

```bash
MODEL=/data/models/gemma-4-26B-A4B-it-vq-release
MTP=$MODEL/gemma4-26b-a4b-it-mtp.sft
ENV_FAST="DS4_CUDA_COPY_MODEL=1 DS4_GEMMA4_ATTN_PREFILL=1 DS4_GEMMA4_ATTN_FAST=1 DS4_GEMMA4_FUSE_ROPE=1 DS4_GEMMA4_ROUTER_PAR=1 DS4_GEMMA4_HEAD_GEMV=1 DS4_GEMMA4_ROUTER_DOT=1"

# 1. Baseline (MTP off) at ctx 256 (p=64, n=64) and ctx 2048 (p=1024, n=64)
env $ENV_FAST ./target/release/ds4-bench -m $MODEL -p 64 -n 64 -r 3 --warmup 1
env $ENV_FAST ./target/release/ds4-bench -m $MODEL -c 2048 -p 1024 -n 64 -r 3 --warmup 1

# 2. MTP ON with D in {2, 4, 6} at ctx 256 and ctx 2048
for D in 2 4 6; do
  env $ENV_FAST ./target/release/ds4-bench -m $MODEL --mtp $MTP --mtp-draft $D -p 64 -n 64 -r 3 --warmup 1
  env $ENV_FAST ./target/release/ds4-bench -m $MODEL --mtp $MTP --mtp-draft $D -c 2048 -p 1024 -n 64 -r 3 --warmup 1
done
```

Record for each configuration:
- `prefill_tps` (tok/s) and `ttft_ms` (ms): verifies that binding `--mtp` and publishing `last_hidden` on the last prefill chunk does not regress prefill or TTFT.
- `gen_tps` (tok/s): end-to-end decode throughput.

### 5.3 How to Separate Draft Overhead from Acceptance Gains

Net speculative decode throughput is governed by three independent factors:
$$\text{gen\_tps}(D) = \frac{\tau(D)}{D \cdot t_{\text{draft}} + t_{\text{verify}}(D + 1)}$$
where:
- $\tau(D) = 1 + \sum_{k=1}^{D} \text{Acc}(1 \dots k) \in [1, D+1]$ is the **acceptance multiplier** (tokens committed per trial),
- $D \cdot t_{\text{draft}}$ is the **drafter overhead** ($D$ rollouts of the 4-layer assistant model),
- $t_{\text{verify}}(D + 1)$ is the **verify forward cost** for $D + 1$ trunk tokens (compared to $t_1 = t_{\text{verify}}(1)$ for plain 1-token decode).

To isolate all three terms cleanly without guessing:

1. **Measure $t_1$ (plain decode baseline)**:
   - Run with `--mtp` off: $t_1 = \frac{1000}{\text{gen\_tps}(0)}\text{ ms/token}$.
2. **Separate $D \cdot t_{\text{draft}}$ from $t_{\text{verify}}(D + 1)$ using two timers in `ds4_session_gemma4_trial` (`ds4_gemma4_mtp.inc:408-420`)** (under `DS4_GEMMA4_PROFILE=1`):
   - `t_draft_us`: wall time across the `for (int i = 0; i < max_drafts; ++i)` loop calling `gemma4_mtp_step` (with `cudaStreamSynchronize` already happening inside `ds4_gpu_tensor_read` at the end of every `gemma4_mtp_step`!).
   - `t_verify_us`: wall time across `gemma4_mtp_verify(s, tokens, max_drafts + 1, target)` (which also synchronizes inside `ds4_gpu_tensor_read` at `ds4_gemma4_mtp.inc:350`!).
   - Because both `gemma4_mtp_step` and `gemma4_mtp_verify` already end with a synchronous D2H `ds4_gpu_tensor_read`, wrapping those two blocks in `clock_gettime(CLOCK_MONOTONIC, ...)` has **zero extra synchronization overhead** and gives exact per-trial `t_draft_ms` and `t_verify_ms`!
3. **Compute the break-even acceptance $\tau_{\text{breakeven}}(D)$**:
   $$\tau_{\text{breakeven}}(D) = \frac{D \cdot t_{\text{draft}} + t_{\text{verify}}(D + 1)}{t_1}$$
   - **Verify scaling efficiency**: $\eta_{\text{verify}}(D) = \frac{t_{\text{verify}}(D + 1)}{t_1}$. On a bandwidth-bound streaming GPU where weights are read once per forward, an ideal multi-row verify has $\eta_{\text{verify}}(D) \approx 1.1\text{–}1.4\times$. If $\eta_{\text{verify}}(D) > 2.0\times$, the verify block is falling back to `glmvq_restore` + cuBLAS or repeating the 1.47 GB vocabulary head GEMV $D + 1$ times instead of sharing weight reads!
   - **Drafter tax**: $\eta_{\text{draft}}(D) = \frac{D \cdot t_{\text{draft}}}{t_1}$. Expected theoretical floor is $\approx D \times \frac{\text{bytes}_{\text{mtp}}}{\text{bytes}_{\text{trunk}}}$.

---

## 6. What Else Must Be Re-Gated Before P8 Closes

Because the sliding-ring capacity changed from `window + 1` (`1025`) to `window + max(chunk, drafts + 1)` (`1029` at `chunk = 1`, `1536` at `chunk = 512`), the following 6 areas must be checked and re-gated before closing Phase 7 P8:

### 6.1 Snapshots & Session Disk KV (`ds4_gemma4_payload.inc`)
- In `ds4_gemma4_payload.inc:20-23`, `gemma4_payload_rows(g, il)` returns `ds4_gemma4_kv_rows(il, g->ctx, g->cap)`.
- Header slot `h[9]` (`ds4_gemma4_payload.inc:43`) stores `gemma4_payload_rows(g, 0) = 1024 + max(g->cap, G4_MTP_DRAFTS + 1)`, and `gemma4_payload_validate` (`ds4_gemma4_payload.inc:73`) rejects any snapshot or disk KV file whose `h[9]` does not match the live session's `gemma4_payload_rows(g, 0)`.
- **Consequence**:
  1. Any on-disk KV cache files written before the ring change (or written with a different `G4_MTP_DRAFTS` or a different `DS4_GEMMA4_PREFILL_CHUNK`) will fail `gemma4_payload_validate` (`h[9]` mismatch).
  2. Notice that `h[9]` depends on `g->cap` (`DS4_GEMMA4_PREFILL_CHUNK`)! If a user saves a disk KV cache with `DS4_GEMMA4_PREFILL_CHUNK=512` (`h[9] = 1536`) and restores it with `DS4_GEMMA4_PREFILL_CHUNK=1` (`h[9] = 1029`), `gemma4_payload_validate` rejects it.
  3. **Action**: Re-run `disk/session` and `snapshot/session` in `./ds4 --serving-check` **after** locking `G4_MTP_DRAFTS` and `GEMMA4_NATIVE_DEFAULT`.

### 6.2 Continuous Banks, Bank Fork, and Bank Disk KV (`ds4_gemma4_batch.inc` & `ds4_gemma4_bank_payload.inc`)
- In `ds4_gemma4_batch.inc:24`, each bank slot allocates `rows = ds4_gemma4_kv_rows(il, b->max_seq_len, 1)`, which is `1024 + max(1, G4_MTP_DRAFTS + 1) = 1029` rows for sliding layers (`b->cap` is hardcoded to `1` in banks).
- In `ds4_gemma4_bank_payload.inc:11-26`, `gemma4_bank_layer_bytes` serializes `used = (pos < rows) ? pos : rows` rows (`1029` max for sliding layers).
- **Critical bug to fix in `gemma4_bank_fork` (`ds4_gemma4_batch.inc:124-135`)**:
  - `gemma4_bank_fork` currently assumes any prefix `pos <= sg->position` of a bank is valid in-place (`if (dst == src) { dg->position = pos; return 0; }`).
  - Once `sg->position > rows` (`> 1029`), slots `(k % 1029)` for $k \in [0, \text{sg->position} - 1029)$ have been overwritten! If a bank at `sg->position = 1200` is rewound/forked to `pos = 500`, slots `0..170` hold tokens `1029..1199` instead of `0..170`!
  - **Fix**: In `gemma4_bank_fork`, if `pos < sg->position` and `sg->position > ds4_gemma4_kv_rows(0, b->max_seq_len, 1)` and `pos - G4_WINDOW < sg->position - ds4_gemma4_kv_rows(0, b->max_seq_len, 1)`, return `-1` (cannot fork an overwritten sliding prefix).

### 6.3 Serial Session Ring-Resume Guard (`ds4_gemma4_session.inc:39-46`)
- Similarly, in `gemma4_sync_keep` (`ds4_gemma4_session.inc:39-46`), when `s->checkpoint.len > ring` (or when a previous speculative verify wrote up to `s->checkpoint.len + G4_MTP_DRAFTS` before committing `s->checkpoint.len`), rewinding to `start < s->checkpoint.len` is only valid if every key in the sliding window $[ \max(0, \text{start} - \text{G4\_WINDOW}), \text{start} - 1 ]$ is still intact in the ring:
  ```c
  int ring = ds4_gemma4_kv_rows(0, g->ctx, g->cap);
  int high_water = g->high_water; /* track max position written, including uncommitted verify rows */
  if (start < s->checkpoint.len && high_water > ring) {
      int oldest_intact = high_water - ring;
      int earliest_needed = (start > G4_WINDOW) ? (start - G4_WINDOW) : 0;
      if (earliest_needed < oldest_intact) start = 0;
  }
  ```
- Without tracking `high_water` (including rejected MTP verify rows `start_prev + n`), rewinding after an MTP trial in a wrapped ring can read a slot overwritten by a rejected draft token.

### 6.4 Rust Quote & Plan Constants (`crates/ds4-core/src/serving_host.rs`)
Three discrepancies in `crates/ds4-core/src/serving_host.rs` must be reconciled before P8 closes:
1. **`GEMMA4_MTP_DRAFTS` (`serving_host.rs:225`) and unit test (`serving_host.rs:7613-7627`)**:
   - Currently `GEMMA4_MTP_DRAFTS = 4` and `gemma4_bank_kv_prices_the_hybrid_layout` asserts `252_682_240` bytes at `ctx = 2048, chunk = 1`. If `G4_MTP_DRAFTS` in `ds4_gemma4_plan.h` changes (e.g., to `2` or `6`), `GEMMA4_MTP_DRAFTS` and the test assertion (`252_682_240 + (D - 4) * 204_800`) must update in lockstep.
2. **`GEMMA4_NATIVE_DEFAULT` (`serving_host.rs:219`) vs `gemma4_prefill_cap` (`ds4_gemma4_bind.inc:268`)**:
   - `serving_host.rs:219` still defines `GEMMA4_NATIVE_DEFAULT: usize = 1` (with a stale Phase 0 comment), which causes `fill_quote_facts` (`serving_host.rs:365`) to inject `DS4_GEMMA4_PREFILL_CHUNK=1` into `plan.env_overrides()` whenever `ds4` or `ds4-server` builds a `ServingPlan`! Meanwhile `ds4_gemma4_bind.inc:268` expects `512` (`cap = 512`). Update `GEMMA4_NATIVE_DEFAULT` (or align both C and Rust on the qualified prefill chunk width) and update the quote test accordingly.
3. **Missing `ModelFamily::Gemma4Vq` arm in `fill_quote_facts` (`serving_host.rs:369-717`) and `gguf_span_bytes` (`serving_host.rs:1130-1156`)**:
   - `ModelFamily::Gemma4Vq` currently hits `_ => (kv, 0, 0, 0)` in `fill_quote_facts`, reporting `0` bytes for `scratch` (actually `gemma4_activation_bytes(cap) + ds4_gpu_g4_moe_bytes(cap)` $\approx 60\text{–}380\text{ MiB}$) and `0` bytes for `mtp_state` (actually `gemma4_mtp_scratch_bytes()` $\approx 7.23\text{ MiB}$ + `mtp.sft` weight bytes when `--mtp` is enabled).

### 6.5 Sampled Speculation (`spec_sample_rows`) in `ds4_gemma4_mtp.inc`
- Wire `spec_sample_rows(&s->spec, r->host, G4_VOCAB, tokens, n, target)` when `spec_sampling(&s->spec)` is true in `gemma4_mtp_verify` (`ds4_gemma4_mtp.inc:353-356`), matching `docs/speculative-sampling.md`.

### 6.6 Complete P8 Closure Checklist (Ordered)
1. [ ] Fix `gemma4_mtp_verify` cross-width consistency (`ds4_gemma4_mtp.inc:325, 338`) so verify rows use the same numerical path (and fast head GEMV) as 1-row decode, and wire `spec_sample_rows` (`ds4_gemma4_mtp.inc:353`).
2. [ ] Add the `high_water` / `ring_holds` guard in `gemma4_sync_keep` (`ds4_gemma4_session.inc:39`) and `gemma4_bank_fork` (`ds4_gemma4_batch.inc:124`).
3. [ ] Run the fast wrapped-ring gate (Section 1.3: 1,032-token prompt, `DS4_GEMMA4_PREFILL_CHUNK=4`, `-c 1056 -n 12 --temp 0`, off vs `--mtp-draft 4`) and verify `cmp` is byte-identical.
4. [ ] Run the $D \in \{2, 4, 6\}$ experiment matrix (Section 3.3 & Section 5.2) at `ctx 256` and `ctx 2048`, record $\alpha_k$, $\tau(D)$, $t_{\text{draft}}$, $t_{\text{verify}}$, and `gen_tps`, and lock `G4_MTP_DRAFTS` / default draft count.
5. [ ] Sync `crates/ds4-core/src/serving_host.rs` (`GEMMA4_MTP_DRAFTS`, `GEMMA4_NATIVE_DEFAULT`, `fill_quote_facts`, `gguf_span_bytes`) and run `cargo test -p ds4-core` + `make test-gemma4-primitives`.
6. [ ] Run `./ds4 --serving-check -m /data/models/gemma-4-26B-A4B-it-vq-release` (both without and with `--mtp .../gemma4-26b-a4b-it-mtp.sft`) plus a live `--mtp` prefix-reuse + snapshot + mid-trial stop check.
