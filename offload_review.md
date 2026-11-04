# Gemma-4 MoE Offload Review: `llama.cpp` (`-ot "exps=CPU"` / `--n-cpu-moe`) vs. `ds4`

**Reference Guide Reviewed:** [DocShotgun — *Performant local mixture-of-experts CPU inference with GPU acceleration in llama.cpp*](https://gist.github.com/DocShotgun/a02a4c0c0a57e43ff4f038b46ca66ae0)  
**Repository:** `/data/ds4-gemma4` (branch `feat/gemma4-family` at `1bd4afe5` + working-tree `ds4_gemma4_map.inc`)  
**Target Artifact:** Gemma-4 26B-A4B VQ R13+8 (`/data/vq-artifacts/gguf-family-it-r8`, `9.37 GiB` trunk + `0.58 GiB` MTP)

---

## 1. Executive Summary: What DocShotgun's Guide Does & How It Maps to `ds4`

In [DocShotgun's `llama.cpp` MoE Offload Guide](https://gist.github.com/DocShotgun/a02a4c0c0a57e43ff4f038b46ca66ae0), high-performance MoE offloading on a GPU that cannot hold the entire model in VRAM is built on **three rules**:

1. **Rule 1 — Put 100% of "Always-Active" Weights in GPU VRAM (`-ngl 999 -ot "exps=CPU"` / `--cpu-moe`):**
   - Attention (`q/k/v/o`), Dense FFN (`gate/up/down`), Shared Experts / VQ Codebooks, Norms, Routers, `token_embd` (tied output head), KV cache, and the MTP drafter are used on **every single token**.
   - For Gemma-4 26B-A4B VQ, all "always-active" trunk weights total only **`2.35 GiB`** (`25.1%` of the `9.37 GiB` model), plus **`0.58 GiB`** when `--mtp` is enabled (**`2.93 GiB` total**).
   - Routed experts (`30 layers × 128 experts = 3,840 experts`) make up the remaining **`7.01 GiB` (`74.9%`)** (`239.4 MiB` per layer), but each token only activates **`8` of `128` experts per layer (`6.25%` = `14.96 MiB/layer`)**.

2. **Rule 2 — Fill Remaining Free VRAM with Whole Layers of Routed Experts, and Keep Only the Overflow Expert Layers in Host RAM (`-ot "blk\.(0..K)\.=CUDA0,exps=CPU"` / `--n-cpu-moe N`):**
   - Instead of offloading *all* 30 layers of routed experts when you have a 12 GiB GPU, you keep **layers `0 .. (29 - N)` of routed experts permanently in VRAM** (`239.4 MiB` per layer) and offload **only the last `N` layers (`--n-cpu-moe N`) to Host RAM**.
   - For example, on a **12 GiB RTX 4070 SUPER**:
     - Always-active trunk (`2.35 GiB`) + MTP drafter (`0.58 GiB`) = **`2.93 GiB` VRAM**.
     - Keeping **`22` of `30` layers of routed experts in VRAM (`5.14 GiB`)** brings total weight VRAM to **`8.07 GiB`**, leaving **`~3.9 GiB` of VRAM** for KV cache, MTP verify scratch, cuBLAS workspace, and desktop!
     - Only **`N = 8` layers (`1.87 GiB`)** of routed experts live in Host RAM — and during decode, only their **8 active experts per layer (`8 × 14.96 MiB = 119.7 MiB/token`)** cross PCIe!

3. **Rule 3 — Prompt Processing "Op Offload" (`GGML_OP_OFFLOAD_MIN_BATCH = 32`):**
   - During prefill (`rows >= 32`), almost all `128` experts per layer are activated. Reading an offloaded layer over PCIe tile-by-tile thrash-fetches the same expert weights multiple times.
   - Instead, when `rows >= 32`, `llama.cpp` copies one offloaded layer's experts (`239.4 MiB`) into a reusable GPU VRAM bounce buffer, runs the batched GEMM at full GPU VRAM bandwidth (`504 GB/s`), and reuses that same `239.4 MiB` buffer for the next offloaded layer.

---

## 2. Critical Finding: Why `ds4`'s Built-In Unit Planner Didn't Do This Automatically on RTX 4070 SUPER

**`ds4` actually already has the data structures for DocShotgun-style hybrid residency built into [`ds4_model_catalog.h`](file:///data/ds4-gemma4/ds4_model_catalog.h) and [`ds4_cuda.cu`](file:///data/ds4-gemma4/ds4_cuda.cu) (`DS4_TCAT_ROUTED_EXPERT`, `DS4_UPOL_DEVICE_PROMOTE`, `DS4_UPOL_EXPERT_COLD`, and two-pass `cuda_model_range_resolve`), BUT three bugs/gates currently prevent it from working on Gemma-4 on a discrete GPU:**

### Root Cause 1: [`ds4_tensor_catalog_classify`](file:///data/ds4-gemma4/ds4_model_catalog.h#L89-L168) Does Not Recognize Gemma-4's Routed Expert Tensor Names
Look at line 5 of `/tmp/g4-offload-resident-1024.log`:
```text
ds4: model catalog base: 23757 tensors, 0 routed-expert, 0 artifact-replaced, 0 artifact-additive, 0 optional
```
Why did the catalog report **`0 routed-expert`** out of `23,757` tensors?
- In [`ds4_model_catalog.h:104-111`](file:///data/ds4-gemma4/ds4_model_catalog.h#L104-L111), the 2D VQ expert rule only checks for HuggingFace GLM/Ling names (`model.language_model.layers.*.mlp.experts.*.{gate,up,down}_proj.weight`).
- Gemma-4 uses per-expert GGUF names:
  - `blk.<L>.ffn_{gate,up,down}.<E>.weight` (`GVQ1`, `0x47565138`)
  - `blk.<L>.ffn_{gate,up,down}.<E>.weight.gvr` (`GVR1`, `0x47565238`)
- Because neither pattern is in `ds4_tensor_catalog_classify`, all `23,040` routed expert tensors (`7.01 GiB`) are misclassified as `0` (`ALWAYS-HOT`) instead of `DS4_TCAT_ROUTED_EXPERT`!

### Root Cause 2: [`ds4_gpu_materialize_model_plan`](file:///data/ds4-gemma4/ds4_cuda.cu#L7298-L7308) Has an Early `if (!integrated) return 1;`
Look at [`ds4_cuda.cu:7303-7308`](file:///data/ds4-gemma4/ds4_cuda.cu#L7303-L7308):
```cpp
    {
        int dev = 0, integrated = 0;
        if (cudaGetDevice(&dev) == cudaSuccess)
            (void)cudaDeviceGetAttribute(&integrated, cudaDevAttrIntegrated, dev);
        if (!integrated) return 1;
    }
```
And look at what happened in `/tmp/g4-offload-resident-1024.log` (lines 12–13) when running on the discrete RTX 4070 SUPER (`integrated == 0`) without `DS4_CUDA_COPY_MODEL_CHUNKED=1`:
```text
ds4: model source base [primary]: map 9.37 GiB, device 0.00 GiB, host-pinned 9.37 GiB
ds4: model source mtp [auxiliary]: map 0.58 GiB, device 0.00 GiB, host-pinned 0.58 GiB
```
- Because `!integrated` returned early, **`0.00 GiB` was copied to VRAM**!
- Every single tensor — including the `1.38 GiB` `token_embd` LM head (read **7 times** per MTP draft-6 step!), all 30 Attention layers, all 30 Dense FFN layers, all 120 VQ codebooks, and the `0.58 GiB` MTP drafter — was read over PCIe 4.0 from `host-pinned` RAM on every token, resulting in **`3.81 tok/s` decode and `67.34 tok/s` prefill** (`/tmp/g4-offload-resident-1024.csv`)!
- Conversely, if you set `DS4_CUDA_COPY_MODEL_CHUNKED=1`, `cuda_model_copy_chunked` tries to copy **all `9.37 GiB + 0.58 GiB = 9.95 GiB`** into VRAM at once, which OOMs a 12 GiB GPU once KV cache + MTP scratch + desktop VRAM are added.

### Root Cause 3: [`cuda_model_range_resolve`](file:///data/ds4-gemma4/ds4_cuda.cu#L2049-L2076) Already Supports Hybrid VRAM + Host-Pinned Lookup!
Look at [`cuda_model_range_resolve`](file:///data/ds4-gemma4/ds4_cuda.cu#L2049-L2076):
```cpp
    // Pass 1: Device-resident VRAM ranges (!r.host_registered) win first!
    for (const cuda_model_range &r : g_model_ranges) {
        if (r.host_base == model_map && !r.host_registered &&
            offset >= r.offset && end >= offset && end <= r.offset + r.bytes) {
            return r.device_ptr + (offset - r.offset);
        }
    }
    // Pass 2: Fall back to host-registered UVA mapping (r.host_registered)
    for (const cuda_model_range &r : g_model_ranges) { ... }
```
And in [`cuda_model_range_ptr`](file:///data/ds4-gemma4/ds4_cuda.cu#L2092-L2106):
- `cuda_model_range_resolve` runs **before** `cuda_model_current_direct_available(model_map)`!
- That means if `m->map` is `cudaHostRegister`-mapped AND we promote selected units (`DS4_UPOL_DEVICE_PROMOTE`) into VRAM via `cuda_unit_materialize_copy`:
  1. **Every VRAM-promoted tensor** (the `2.35 GiB` always-active trunk + `0.58 GiB` MTP + the first `30 - N` layers of routed experts) automatically resolves in Pass 1 to its **GDDR6X VRAM pointer (`504 GB/s`)**!
  2. **Every remaining `DS4_UPOL_EXPERT_COLD` tensor** (the `N` offloaded layers of routed experts) automatically falls through to `cuda_model_current_direct_available` / Pass 2 and resolves to its **zero-copy `cudaHostRegister` UVA pointer in Host RAM (`0 B` of VRAM)**!
  3. During decode (`rows = 1`) and MTP verify (`rows = 7`), for those `N` offloaded layers, the GPU kernel (`g4_vq_gemv` / `g4_vq_gemm_items`) **only dereferences the 8 active experts (`14.96 MiB` per layer)** over PCIe 4.0 x16 (`~24 GB/s`), while their 64 KiB VQ codebooks sit in GPU VRAM!

---

## 3. Side-by-Side Comparison: DocShotgun's `llama.cpp` Flags vs. `ds4` Modes

| Offload Strategy | `llama.cpp` Command (DocShotgun Gist) | How It Works on Gemma-4 26B-A4B VQ (`9.37 GiB`) | VRAM Used (Weights) | PCIe Traffic / Decode Token | Expected RTX 4070S Speed |
|---|---|---|---:|---:|---:|
| **A. Pure Host-Mapped (What `resident-1024` accidentally ran)** | `-ngl 0` (or UVA all) | `0 GiB` in VRAM; trunk + MTP + experts all read over PCIe | `0.00 GiB` | `~2.8 GiB/tok` (`~14.4 GiB` per MTP round!) | `3.8 tok/s` (measured) |
| **B. All Always-Active in VRAM, All Routed Experts in RAM (`--cpu-moe`)** | `-ngl 999 -ot "exps=CPU"` (or `--cpu-moe`) | **`2.35 GiB` trunk (`+ 0.58 GiB` MTP) in VRAM**; all 30 expert layers (`7.01 GiB`) in Host RAM (`cudaHostRegister`); only **8 active experts/layer** read over PCIe | **`2.35 GiB`** (`2.93 GiB` w/ MTP) | `448.9 MiB/tok` (`30 × 14.96 MiB`) | **`~42–48 tok/s`** (`~65+` w/ MTP) |
| **C. Hybrid Layer Split (`--n-cpu-moe N`, DocShotgun's #1 recommendation)** | `-ngl 999 --n-cpu-moe 8` (or `-ot "blk\.([0-9]\|1[0-9]\|2[0-1])\.=CUDA0,exps=CPU"`) | **`2.35 GiB` trunk + `0.58 GiB` MTP + 22 expert layers (`5.14 GiB`) in VRAM**; only **8 expert layers (`1.87 GiB`) in Host RAM**; only `8 × 8 = 64` active experts cross PCIe | **`8.07 GiB`** (leaves `~3.9 GiB` VRAM free!) | **`119.7 MiB/tok`** (`8 × 14.96 MiB`) | **`~60–65 tok/s`** (`~95–110` w/ MTP!) |
| **D. Bounded VRAM Cache + Cold SSD (`--ssd-streaming`)** | *(Not in `llama.cpp`; unique to `ds4`)* `--ssd-streaming --ssd-streaming-cache-experts 8 [--ssd-streaming-cold]` | **`2.37 GiB` VRAM** (`2.35 GiB` trunk + `15 MiB` 8-slot pool); experts fetched via `pread` + `cudaMemcpy` + `rebind` (can run with **`0.39 MiB` Host RAM**!) | **`2.37 GiB`** | `448.9 MiB/tok` | `~20–40 tok/s` (RAM) / `~10–20 tok/s` (SSD) |

---

## 4. Concrete Code Suggestions to Implement DocShotgun-Style Offloading in `ds4`

### Suggestion 1: Teach `ds4_tensor_catalog_classify` to Recognize Gemma-4's Routed Experts (`ds4_model_catalog.h`)

In [`ds4_model_catalog.h:87-112`](file:///data/ds4-gemma4/ds4_model_catalog.h#L87-L112), add `DS4_TCAT_GLM_RES` (`0x47565238u`) and match Gemma-4's per-expert GGUF tensor names (`blk.<L>.ffn_{gate,up,down}.<E>.weight` and `.weight.gvr`) while keeping shared codebooks (`.codebook`, `.codebook.gvr`), dense FFN (`blk.<L>.ffn_{gate,up,down}.weight`), and routers (`ffn_gate_inp`) marked as `0` (`ALWAYS-HOT`):

```c
/* In ds4_model_catalog.h:87 */
#define DS4_TCAT_GLM_FP8      0x47514638u
#define DS4_TCAT_GLM_VQ       0x47565138u
#define DS4_TCAT_GLM_RES      0x47565238u

/* Inside ds4_tensor_catalog_classify (ds4_model_catalog.h:112):
 * Gemma-4 per-expert GVQ1 / GVR1 matrices:
 *   blk.<L>.ffn_{gate,up,down}.<E>.weight
 *   blk.<L>.ffn_{gate,up,down}.<E>.weight.gvr
 * Note: Dense FFN is blk.<L>.ffn_{gate,up,down}.weight (no ".<E>." before .weight),
 * and codebooks end in .codebook / .codebook.gvr, so checking for ffn_{gate,up,down}.
 * without the dense suffix matches only the 23,040 routed expert matrices. */
if (ndim == 2 && (ggml_type == DS4_TCAT_GLM_VQ || ggml_type == DS4_TCAT_GLM_RES) &&
    ds4_tcat_contains(name, name_len, "blk.") &&
    (ds4_tcat_contains(name, name_len, ".ffn_gate.") ||
     ds4_tcat_contains(name, name_len, ".ffn_up.") ||
     ds4_tcat_contains(name, name_len, ".ffn_down.")) &&
    !ds4_tcat_has_suffix(name, name_len, ".ffn_gate.weight") &&
    !ds4_tcat_has_suffix(name, name_len, ".ffn_up.weight") &&
    !ds4_tcat_has_suffix(name, name_len, ".ffn_down.weight") &&
    (ds4_tcat_has_suffix(name, name_len, ".weight") ||
     ds4_tcat_has_suffix(name, name_len, ".weight.gvr"))) {
    traits |= DS4_TCAT_ROUTED_EXPERT;
}
```
- **Effect:** `ds4: model catalog base` immediately reports **`23040 routed-expert`** tensors (`7.01 GiB`) and **`717` always-active** tensors (`2.35 GiB`)!

---

### Suggestion 2: Support `--cpu-moe` and `--n-cpu-moe N` (or `DS4_CUDA_CPU_MOE_LAYERS=N` / Auto-Fit) in `ds4.c` and `ds4_cuda.cu`

To replicate DocShotgun's `-ot "blk\.(0..K)\.=CUDA0,exps=CPU"` / `--n-cpu-moe N` on discrete GPUs:

1. **In [`ds4.c:74701-74754`](file:///data/ds4-gemma4/ds4.c#L74701-L74754) (Unit Compilation):**
   - Add an option or env knob `DS4_CUDA_CPU_MOE_LAYERS` (or CLI `--n-cpu-moe <N>` / `--cpu-moe` where `--cpu-moe` sets `N = 30`).
   - For layers `il < (G4_LAYERS - N)` (the GPU-resident expert layers), keep `up.promote_experts = 1` (or clear `DS4_TCAT_ROUTED_EXPERT` on those layers' tensors so they compile to `DS4_UPOL_DEVICE_PROMOTE`).
   - For layers `il >= (G4_LAYERS - N)` (the last `N` layers offloaded to Host RAM), keep `DS4_TCAT_ROUTED_EXPERT` with `promote_experts = 0` so they compile to **`DS4_UPOL_EXPERT_COLD`**.
   ```c
   /* Helper to parse layer index from "blk.<il>." */
   static int ds4_parse_blk_layer(const char *name, uint64_t len) {
       if (len < 6 || memcmp(name, "blk.", 4) != 0) return -1;
       int il = 0;
       uint64_t pos = 4;
       while (pos < len && name[pos] >= '0' && name[pos] <= '9') {
           il = il * 10 + (name[pos] - '0');
           pos++;
       }
       return (pos > 4 && pos < len && name[pos] == '.') ? il : -1;
   }
   ```
   When building `tin[i]` in [`ds4.c:74701-74706`](file:///data/ds4-gemma4/ds4.c#L74701-L74706):
   - If `n_cpu_moe > 0` (e.g. `DS4_CUDA_CPU_MOE_LAYERS=8`), then for any tensor with `DS4_TCAT_ROUTED_EXPERT` in layer `il < (30 - n_cpu_moe)`, clear `DS4_TCAT_ROUTED_EXPERT` on `tin[i].traits` (so layers `0 .. 29 - n_cpu_moe` become `DS4_UPOL_DEVICE_PROMOTE` in VRAM) and leave `up.promote_experts = 0` (so layers `30 - n_cpu_moe .. 29` become `DS4_UPOL_EXPERT_COLD` in `cudaHostRegister` Host RAM!).

2. **In [`ds4_cuda.cu:7303-7308`](file:///data/ds4-gemma4/ds4_cuda.cu#L7303-L7308) (`ds4_gpu_materialize_model_plan`):**
   - Allow `ds4_gpu_materialize_model_plan` to run on discrete GPUs (`!integrated`) whenever the model is not already `g_model_device_owned`, and promote **non-expert (`!DS4_TCAT_ROUTED_EXPERT`) units first**, followed by funded `DEVICE_PROMOTE` expert units up to the available VRAM budget (leaving a `~1.5–2.0 GiB` reserve for KV cache + MTP scratch):
   - Because `m->map` is registered with `cudaHostRegister`, any `DS4_UPOL_EXPERT_COLD` unit (or any expert unit that doesn't fit the remaining VRAM budget) stays in `cudaHostRegister` Host RAM at **`0 B` VRAM cost**, and [`cuda_model_range_resolve`](file:///data/ds4-gemma4/ds4_cuda.cu#L2049-L2076) automatically serves the promoted units from VRAM and the cold expert units from Host RAM over PCIe!
   - Furthermore, for every unit that *is* promoted to VRAM (`DS4_RESUNIT_POPULATED`), call `cudaHostUnregister` / `madvise(MADV_DONTNEED)` on its host pages (or only `cudaHostRegister` the `DS4_UPOL_EXPERT_COLD` spans!) so Host RAM only pins the `N` offloaded expert layers (`N × 239.4 MiB`), **not** the VRAM-resident layers!

---

### Suggestion 3: Add `llama.cpp`-Style Prompt Processing "Op Offload" (`GGML_OP_OFFLOAD_MIN_BATCH`) for Offloaded Expert Layers

In Section 2 of [DocShotgun's Guide](https://gist.github.com/DocShotgun/a02a4c0c0a57e43ff4f038b46ca66ae0#2-prompt-processing-optimizations), `llama.cpp` speeds up prompt processing (prefill) on CPU-offloaded expert layers (`exps=CPU`) by temporarily copying an offloaded layer's experts to the GPU when the prompt batch size `rows >= 32` (`GGML_OP_OFFLOAD_MIN_BATCH`):

- **Why this matters in `ds4`:**
  - During decode (`rows = 1`), an offloaded layer in `cudaHostRegister` Host RAM only touches `8` experts (`14.96 MiB`), and `g4_vq_gemv` reads each of those 8 experts **once** over PCIe (`0.62 ms` per offloaded layer).
  - During prefill (`rows = 512`), `g4_vq_gemm_items` ([`ds4_gemma4_gpu.cuh:740-799`](file:///data/ds4-gemma4/ds4_gemma4_gpu.cuh#L740-L799)) tiles tokens by `G4_VQP_BN = 8` and rows by `G4_VQP_BM = 64`. If an expert's weights live in `cudaHostRegister` Host RAM across PCIe, every `BN = 8` token tile re-fetches that expert's weights over PCIe!
- **How to implement Op Offload in `ds4` (`~30` lines in `ds4_gemma4_gpu.cuh`):**
  - Allocate a single **`239.4 MiB`** (`128 × 1.87 MiB`) reusable device bounce buffer `m->op_offload_buf` in `ds4_g4_moe` (or reuse `e->gemma4_stream->pool` when `--ssd-streaming` is active).
  - In `g4_moe_run_prefill` ([`ds4_gemma4_gpu.cuh:740`](file:///data/ds4-gemma4/ds4_gemma4_gpu.cuh#L740)), when `rows >= 32` (`min_batch`) and layer `il`'s expert pointers point to `host_registered` memory rather than VRAM:
    1. Copy the active experts (or the contiguous `239.4 MiB` layer span) from `cudaHostRegister` Host RAM into `m->op_offload_buf` in VRAM **once** via `cudaMemcpyAsync` (`~9.5 ms` at `25 GB/s` PCIe 4.0).
    2. Point `m->hqmat[e]` at `m->op_offload_buf` so `g4_vq_gemm_items` executes all tiles from **`504 GB/s` GDDR6X VRAM** (`~1.2 ms` instead of `~35 ms` of repeated PCIe reads!).

---

### Suggestion 4: Also Fix the 3 Issues in the Existing `--ssd-streaming` Path (For Low-RAM / Cold-SSD Machines)

When the machine does **not** have enough Host RAM to pin the offloaded experts (so `--ssd-streaming --ssd-streaming-cache-experts 8 [--ssd-streaming-cold]` is used instead of `cudaHostRegister`), apply these three fixes to [`ds4.c`](file:///data/ds4-gemma4/ds4.c) and [`ds4_gemma4_stream.inc`](file:///data/ds4-gemma4/ds4_gemma4_stream.inc):

1. **Fix `--ssd-streaming` + `--mtp` Startup Failure ([`ds4.c:74554`](file:///data/ds4-gemma4/ds4.c#L74554) & [`ds4.c:74693`](file:///data/ds4-gemma4/ds4.c#L74693)):**
   - At [`ds4.c:74554`](file:///data/ds4-gemma4/ds4.c#L74554), map `e->mtp_model` via `ds4_gpu_set_aux_model_map_range` when `opt->ssd_streaming` is on so it does not overwrite `g_model_host_base = e->mtp_model.map` and break [`ds4_gpu_g4_stream_plan_cold`](file:///data/ds4-gemma4/ds4_gemma4_map.inc#L95).
   - At [`ds4.c:74693`](file:///data/ds4-gemma4/ds4.c#L74693), call `gemma ? gemma4_stream_spans(&e->weights, &e->model, &sl_spans) : glm53_stream_spans(&e->weights, &sl_spans)` so the unit table only marks the `2.35 GiB` mandatory spans as active under `--ssd-streaming`.
2. **Use Pinned Host Memory (`cudaHostAlloc`) for `s->staging` ([`ds4_gemma4_stream.inc:247`](file:///data/ds4-gemma4/ds4_gemma4_stream.inc#L247)):**
   - Replace `s->staging = xmalloc(s->staging_bytes)` with a `1.87 MiB` (`s->slot_bytes`) `cudaHostAlloc` buffer and upload all 6 tensors of an expert in **1 contiguous H2D DMA transfer** (`240` DMAs/token at `25 GB/s` instead of `1,440` pageable copies/token at `11 GB/s`).
3. **Bypass `pread()` When `!s->cold` ([`ds4_gemma4_stream.inc:353`](file:///data/ds4-gemma4/ds4_gemma4_stream.inc#L353)):**
   - When `!s->cold`, `memcpy` directly from `(const char *)s->model->map + t->abs_offset` into pinned `s->staging` instead of issuing `1,440` `pread()` syscalls per token.
