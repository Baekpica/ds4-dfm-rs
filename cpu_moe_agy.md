# Gemma-4 26B A4B Dense-VQ: CPU MoE (`--cpu-moe`) vs Active-Expert VRAM Cache

**Verdict: STOP. Do not port the host CPU GVQ1+GVR1 decoder. Keep the active-expert VRAM cache (and/or partial layer VRAM residency via `--n-cpu-moe`).**

The benchmark in [`tests/gemma4_cpu_expert_gemv.c`](file:///data/ds4-gemma4/tests/gemma4_cpu_expert_gemv.c) is **not a fair AVX2 ceiling** (it is a scalar C loop with per-matrix `pthread_create`/`join` overhead and warm L3 cache hits), so an AVX2 port **could** beat `278.4 ms/token` and reach **~38–55 ms/token** on the i7-14700F for batch-1 decode. However, the negative conclusion **is still decisive** for three physical reasons:

1. **The ~94 ms/token (~10.6 tok/s) device number is the uncached PCIe streaming baseline, not the VRAM cache path.** When active experts hit in VRAM on the RTX 4070 SUPER (504 GB/s GDDR6X), all 240 expert GEMVs (720 matrices) execute in **9.29 ms/token** (`6.24 ms` warp + `3.05 ms` block in [`docs/gemma4-phase3-2026-11-03.md`](file:///data/ds4-gemma4/docs/gemma4-phase3-2026-11-03.md#L26-L30)), and full decode runs at **31.1–34.8 tok/s** (`28.7–32.1 ms/token`). Even a perfect AVX2 CPU MoE decoder (~38–55 ms/token for experts alone) is **4–6× slower** than GPU expert execution and caps end-to-end batch-1 decode at ~15–20 tok/s while pegging all 28 CPU threads.
2. **CPU experts cannot overlap with GPU attention.** In Gemma-4's layer topology ([`/data/llama.cpp/src/models/gemma4.cpp`](file:///data/llama.cpp/src/models/gemma4.cpp#L295-L352) and [`ds4_gemma4_graph.inc`](file:///data/ds4-gemma4/ds4_gemma4_graph.inc#L805-L916)), Layer `l` Routed MoE consumes the post-attention residual `cur` and feeds Layer `l+1` Attention. The only GPU work independent of Routed MoE within Layer `l` is the shared Dense FFN (`2816 → 2112 → 2816`), which takes **~0.08 ms/layer (~2.4 ms/token)** on the GPU.
3. **CPU MoE collapses under MTP verify (`rows = 7`).** At draft-6 (`rows = 7`), 7 tokens route to ~45–56 unique experts per layer (~1,350–1,680 experts/step = **2.65–3.30 GiB** of expert weights per verify step). On dual-channel DDR5, streaming 2.8 GiB from host RAM takes **~40–60 ms** in DRAM wire time alone and **~180–250 ms/step** with L2 codebook gathers, destroying MTP speedup.

---

## 1. Is [`tests/gemma4_cpu_expert_gemv.c`](file:///data/ds4-gemma4/tests/gemma4_cpu_expert_gemv.c) Fair and Conclusive?

### Why the microbenchmark is NOT a fair AVX2 ceiling (4 benchmark flaws pushing time up, 1 flaw pulling time down)

Despite compiling with `-O3 -mavx2 -mf16c` ([`tests/gemma4_cpu_expert_gemv.c`](file:///data/ds4-gemma4/tests/gemma4_cpu_expert_gemv.c#L10)), [`gemv_rows`](file:///data/ds4-gemma4/tests/gemma4_cpu_expert_gemv.c#L152-L194) is a scalar loop with severe measurement artifacts:

1. **`pthread_create` + `pthread_join` inside the timed loop ([`lines 241–255`](file:///data/ds4-gemma4/tests/gemma4_cpu_expert_gemv.c#L241-L255)):**
   Every timed matrix call spawns and joins 28 OS threads for 101 rows per thread. On Linux, 28× `clone3` + futex join costs **~0.10–0.15 ms per matrix** (**72–108 ms** across 720 matrices)—nearly a third of the measured `0.387 ms/matrix`! That is also why scaling stalls between 16 threads (`285.3 ms`) and 28 threads (`278.4 ms`).
2. **Scalar `bf16_rn(gain * word)` via `memcpy` per weight ([`lines 72–77, 186`](file:///data/ds4-gemma4/tests/gemma4_cpu_expert_gemv.c#L72-L77)):**
   Inside the innermost `d = 0..7` loop, every single weight executes `bf16_rn(gain * word)` (FP32 multiply + integer rounding bias + shift + `memcpy` back to `float`), costing **1.427 billion** scalar conversions per token. Because the dot product is linear, `gain` can be multiplied **once per row** outside the `K` loop (`88×` fewer multiplies, zero inner-loop BF16 roundings), matching the GPU playbook rule in [`AGENTS.md`](file:///data/ds4-gemma4/AGENTS.md#L317).
3. **Single serial FP32 `fmaf` dependency chain ([`line 186`](file:///data/ds4-gemma4/tests/gemma4_cpu_expert_gemv.c#L186)):**
   `acc = fmaf(w, x[col + d], acc)` serializes all 704 (or 2,816) FMAs onto one accumulator register. On Raptor Lake P-cores (`vfmadd231ps` latency = 4 cycles, throughput = 2/cycle), a single scalar FMA chain is latency-bound at **0.25 floats/cycle** instead of **16 floats/cycle** (2× 8-wide AVX2 FMA units).
4. **Scalar bit-unpack + scalar `e4m3[256]` lookup per byte ([`lines 79–106, 185`](file:///data/ds4-gemma4/tests/gemma4_cpu_expert_gemv.c#L79-L106)):**
   `gvq_index` unpacks one 13-bit index at a time, then loads 8 bytes from `main_book` and 8 bytes from `res_book`, doing 16 scalar table lookups into `e4m3[256]` (`16 × 4 B = 64 B` gather) per 8-weight codeword.
5. **Contrariwise, the benchmark is artificially optimistic on memory bandwidth ([`lines 239–256`](file:///data/ds4-gemma4/tests/gemma4_cpu_expert_gemv.c#L239-L256)):**
   It loops `iters = 10` times over a **single 656 KiB matrix + 66 KiB codebook**, which fits entirely inside the i7-14700F's **33 MiB L3 cache** (and 28 MiB aggregate L2 cache). A real token decode touches `30 layers × 8 experts = 240 experts × 1,965,568 B = 471.7 MiB` of expert records plus `30 × 66 KiB = 2.0 MiB` of codebooks (**473.7 MiB/token**), which is **14.3× larger than L3** and must stream cold from dual-channel DDR5 DRAM.

---

## 2. What to Read in `/data/llama.cpp`

To see how `llama.cpp` places MoE tensors on the CPU, schedules GPU↔CPU splits, caches active experts in VRAM, and vectorizes codebook/grid quant dot products on AVX2, read these exact locations in `/data/llama.cpp`:

| Subsystem | File & Lines | Key Symbols & Mechanism |
|---|---|---|
| **CLI flags (`-ot`, `--cpu-moe`, `--n-cpu-moe`, `--moe-cache-mib`)** | [`common/common.h:1249-1268`](file:///data/llama.cpp/common/common.h#L1249-L1268)<br>[`common/arg.cpp:2765-2808`](file:///data/llama.cpp/common/arg.cpp#L2765-L2808) | `LLM_FFN_EXPS_REGEX = "\\.ffn_(up\|down\|gate)_(ch)?exps"` matches only merged expert 3D tensors (`blk.N.ffn_{gate,up,down}_exps.weight`). `llm_ffn_exps_cpu_override()` pushes `{pattern, ggml_backend_cpu_buffer_type()}` into `params.tensor_buft_overrides`. Keep dense FFN (`ffn_gate.weight`, etc.), router (`ffn_gate_inp.weight`), and norms on GPU. |
| **Loader buffer placement** | [`src/llama-model-loader.cpp:1244-1278`](file:///data/llama.cpp/src/llama-model-loader.cpp#L1244-L1278) | Iterates `tensor_buft_overrides` via `std::regex_search`. When matched to CPU buffer type, calls `select_weight_buft(..., buft_list_cpu)` which picks host memory (or pinned/extra buffer type) instead of `CUDA0`. |
| **Backend scheduler & GPU↔CPU split handoff** | [`ggml/src/ggml-backend.cpp:1109-1130`](file:///data/llama.cpp/ggml/src/ggml-backend.cpp#L1109-L1130)<br>[`ggml/src/ggml-backend.cpp:1823-1900`](file:///data/llama.cpp/ggml/src/ggml-backend.cpp#L1823-L1900) | `ggml_backend_sched_backend_id_from_cur` assigns `GGML_OP_MUL_MAT_ID` to the backend holding `src0` (the expert weights). Note lines `1115–1120`: if `src0` is in pinned host memory (`CPU_Mapped`) and batch size hits `op_offload_min_batch`, `op_offload` runs the op on the GPU over PCIe zero-copy; otherwise it splits the graph. In `ggml_backend_sched_compute_splits` (`1865–1900`), every split copies inputs (`ggml_backend_tensor_copy_async`) and **synchronizes** (`ggml_backend_synchronize`) before launching the CPU split. |
| **Gemma-4 layer graph & MoE call site** | [`src/models/gemma4.cpp:295-352`](file:///data/llama.cpp/src/models/gemma4.cpp#L295-L352)<br>[`src/llama-graph.cpp:1710-1739`](file:///data/llama.cpp/src/llama-graph.cpp#L1710-L1739)<br>[`src/llama-graph.cpp:2362-2430`](file:///data/llama.cpp/src/llama-graph.cpp#L2362-L2430) | In `llm_build_gemma4`, Attention produces `attn_out` (`line 295`). Both the shared Dense FFN (`cur_mlp`, `lines 299–308`) and Routed MoE (`cur_moe`, `lines 310–345`) branch from `attn_out` and merge at `cb(ggml_add(ctx0, cur_mlp, cur_moe), "ffn_out", il)` (`line 348`). `build_moe_ffn` issues three `build_lora_mm_id` (`ggml_mul_mat_id`) ops: `up`, `gate`, and `down`. |
| **Active-expert VRAM SLRU cache (`--moe-cache-mib`)** | [`src/llama-moe-cache.cpp:15-635`](file:///data/llama.cpp/src/llama-moe-cache.cpp#L15-L635)<br>[`src/llama-graph.cpp:2593-2633`](file:///data/llama.cpp/src/llama-graph.cpp#L2593-L2633) | `llama_moe_cache` allocates an SLRU (probationary + protected segment) slot pool in GPU VRAM for CPU-resident `*_exps` tensors, packs `(gate, up, down)` per expert, copies missing experts H2D via `cudaMemcpyAsync` on a prefetch stream, and runs `ggml_mul_mat_id` **on the GPU** using remapped slot IDs (`build_moe_cache_slots`). |
| **CPU `GGML_OP_MUL_MAT_ID` dispatch & Q8 activation quant** | [`ggml/src/ggml-cpu/ggml-cpu.c:1483-1741`](file:///data/llama.cpp/ggml/src/ggml-cpu/ggml-cpu.c#L1483-L1741)<br>[`ggml/src/ggml-cpu/tiled/tiled.cpp:1202-1245`](file:///data/llama.cpp/ggml/src/ggml-cpu/tiled/tiled.cpp#L1202-L1245) | Note first that `ggml_compute_forward_mul_mat_id_tiled` requires `cne1 >= 8` (`tiled.cpp:1219`), so batch-1 decode (`cne1 == 1`) always takes `ggml_compute_forward_mul_mat_id` in `ggml-cpu.c`. Threads cooperatively quantize the FP32 input vector `src1` to `q8_K`/`q8_0` **once** before `ggml_barrier` (`lines 1609–1646`), build a compact `(expert, token)` matrix table (`lines 1546–1606`), and work-steal 64-row chunks across all 8 active experts via `atomic_fetch_add_explicit(& params->current_chunk, 64, ...)` (`lines 1695–1739`). |
| **AVX2 codebook/grid & K-quant `vec_dot` kernels** | [`ggml/src/ggml-cpu/arch/x86/quants.c:68-135`](file:///data/llama.cpp/ggml/src/ggml-cpu/arch/x86/quants.c#L68-L135)<br>[`ggml/src/ggml-cpu/arch/x86/quants.c:2102-2185`](file:///data/llama.cpp/ggml/src/ggml-cpu/arch/x86/quants.c#L2102-L2185)<br>[`ggml/src/ggml-cpu/arch/x86/quants.c:2724-3719`](file:///data/llama.cpp/ggml/src/ggml-cpu/arch/x86/quants.c#L2724-L3719) | Compare `ggml_vec_dot_q4_K_q8_K` (`line 2102`) and the codebook/grid quants `ggml_vec_dot_iq2_xxs_q8_K` (`line 2724`), `ggml_vec_dot_iq3_xxs_q8_K` (`line 3229`), and `ggml_vec_dot_iq3_s_q8_K` (`line 3466`). Notice why `iq2`/`iq3` are fast on AVX2: their codebooks (`iq2xxs_grid`, `iq3s_grid`) store **8-byte `uint64_t` entries of `int8`/`uint8` grid values**; they load 4× 8-byte codewords via scalar `uint64_t` loads into `_mm256_set_epi64x` (note `line 3504`: `_mm256_i32gather_epi32` is explicitly commented out because AVX2 `vpgatherdd` is slower than 4 scalar 64-bit loads!), apply signs, and compute 32 int8×int8 products per instruction via `mul_add_epi8` (`_mm256_maddubs_epi16` + `_mm256_madd_epi16`), scaling only once per 32- or 256-weight block. |

---

## 3. Roofline Analysis: Can an AVX2 GVQ1+GVR1 Decoder Reach ~94 ms on i7-14700F?

**Yes, a `llama.cpp`-grade AVX2 host decoder can beat 94 ms (~38–55 ms/token for batch-1), because 278.4 ms is dominated by scalar serial FMA latency, per-weight BF16 bit-twiddling, and `pthread_create` inside the loop. No, it cannot compete with the GPU VRAM cache (9.29 ms/token), nor can it survive MTP verify.**

Let us price each proposed optimization on the **i7-14700F** (8 Raptor Cove P-cores @ ~5.2 GHz + 12 Gracemont E-cores @ ~4.1 GHz ≈ **90 GHz aggregate core-cycles/s**, 48 KiB L1D + 2 MiB L2 per P-core, 32 KiB L1D per E-core + 2 MiB L2 per 4-E-core cluster, 33 MiB shared L3, dual-channel DDR5):

### Lever 1: Persistent Threadpool + Batching All 8 Active Experts (`ggml-cpu.c:1695-1739`)
- **Savings: ~80–105 ms/token immediately.**
- Replacing the 720 per-matrix `pthread_create`/`pthread_join` calls with a persistent spin/futex threadpool (like `ggml_compute_forward_mul_mat_id` atomic chunking over `8 experts × rows`) removes the OS thread-spawn tax and balances P-cores and E-cores dynamically via atomic 32- or 64-row chunks.
- Furthermore, in each layer, `gate_proj` (`704 × 2816`) and `up_proj` (`704 × 2816`) share the same input vector `norm` (`2816` floats) and can be dispatched together (`16 × 704 = 11,264 rows` per layer before `gelu`, then `8 × 2816 = 22,528 rows` for `down_proj`, so **60 barrier syncs/token** instead of 720).

### Lever 2: Hoist Row `gain` + Decode 4–8 Vectors per Iteration with 4× `__m256` Accumulators
- **Savings: ~110–130 ms/token.**
- In [`tests/gemma4_cpu_expert_gemv.c:186`](file:///data/ds4-gemma4/tests/gemma4_cpu_expert_gemv.c#L186):
  $$\sum_{b=0}^{B-1} \sum_{d=0}^{7} \operatorname{bf16\_rn}\!\bigl(\text{gain}_r \cdot (\text{main}[i_b, d] + \text{res}[r_b, d])\bigr) \cdot x_{8b+d} \;\approx\; \text{gain}_r \sum_{b=0}^{B-1} \sum_{d=0}^{7} (\text{main}[i_b, d] + \text{res}[r_b, d]) \cdot x_{8b+d}$$
  Hoisting `gain_r` outside the `B` loop (an accepted Class-B numerical change under [`AGENTS.md`](file:///data/ds4-gemma4/AGENTS.md#L317) that preserves the exact FP32 dot accumulation order when reduced) eliminates 1.427B scalar `bf16_rn` conversions per token.
- Each 8-weight codeword (`d = 0..7`) maps naturally to **one 8-lane `__m256` FP32 vector**! Unrolling by 4 vectors (`32` weights = 4 codewords per iteration) with 4 independent `__m256` accumulators (`acc0..acc3`) hides the 4-cycle `vfmadd231ps` latency and uses both 256-bit FMA pipes.

### Lever 3: Eliminate the `e4m3[256]` Table
- There are two clean ways to eliminate the 16 scalar `e4m3[byte]` table lookups per codeword on AVX2+F16C:
  1. **Pre-expand codebooks to FP16 (`uint16_t[8]` = 16 B/entry) once at model load (Best):**
     Every E4M3 finite value (including subnormals) is **exact** in IEEE FP16 (`1 sign, 5 exp, 10 mantissa` vs E4M3 `1 sign, 4 exp, 3 mantissa`). Pre-converting each layer's `8192 × 8` main book from `64 KiB` (E4M3) to **`128 KiB` (FP16)** and `256 × 8` residual book from `2 KiB` to **`4 KiB` (FP16)** adds only `30 × 66 KiB = 1.98 MiB` of RAM across the entire model. Then each codeword lookup is a single aligned 128-bit load (`_mm_load_si128((const __m128i *)&book_fp16[idx][0])`) followed by a single hardware F16C instruction `_mm256_cvtph_ps` (`vcvtph2ps ymm, xmm`, 1-cycle throughput)!
  2. **In-register bitwise E4M3 → FP16 (`_mm_loadl_epi64` + `_mm_cvtepu8_epi16` + shift/mask + `_mm256_cvtph_ps`):**
     Keeps the main book at `64 KiB` in L2 cache. Unpacks 8 E4M3 bytes to 8× `uint16_t` via `_mm_cvtepu8_epi16`, separates sign `(v & 0x80) << 8`, shifts exponent+mantissa `((v & 0x7f) << 7) + (8 << 10)` (with a zero/subnormal blend via `_mm_cmpeq_epi16`), and feeds `_mm256_cvtph_ps`.
- Either way, **1 codeword (8 weights) + residual takes ~6–8 instructions instead of ~85 scalar instructions**.

### Lever 4: Prefetching vs Cold DDR5 Bandwidth and L2 Codebook Gather Limits
Even with Levers 1–3, two hardware bottlenecks set a hard floor of **~38–55 ms/token** on i7-14700F:

1. **Cold DDR5 DRAM Streaming Floor (~8–12 ms/token pure bus time):**
   Per token, 240 active experts stream **471.7 MiB** of `GVQ1` + `GVR1` records from host RAM (14× larger than the 33 MiB L3 cache). On dual-channel DDR5-5600 (~65–72 GB/s sustained read), moving 471.7 MiB takes **6.5–7.3 ms** at 100% bus utilization (or **~10–12 ms** on DDR5-4800 / DDR4-3600). Hardware L2 stream prefetchers already track the sequential `plane12`, `plane13`, and `res_idx` streams per row; software `_mm_prefetch` on the random `main_book[idx]` lookups helps hide L2 latency (by prefetching `idx` for `b + 4`), because the `64 KiB` (E4M3) or `128 KiB` (FP16) main codebook overflows the `48 KiB` P-core L1D (`32 KiB` E-core L1D) and resides in **L2 cache** (~14-cycle latency).
2. **L2 Cache Port & Instruction Throughput Floor (~28–40 ms/token):**
   Per token, the CPU decodes $1.427 \times 10^9 / 8 = 178.43 \times 10^6$ codewords:
   - Each codeword requires:
     - Bit-unpacking 12-bit + 1-bit index (`~3–4 uops` when unrolled 2-at-a-time per 3 bytes),
     - One 8-B (E4M3) or 16-B (FP16) random gather from `main_book` in **L2** + one load from `res_book` in **L1D**,
     - 2× `vcvtph2ps` (or E4M3 shift+`vcvtph2ps`) + `vaddps` + `vfmadd231ps` (or 2× `vfmadd231ps` if accumulating main and residual separately).
   - At ~14–18 core cycles per codeword across 8 P-cores + 12 E-cores (accounting for L2 bank conflicts and E-cores sharing one 2 MiB L2 per 4 cores), $178.43 \times 10^6$ codewords take **~28–40 ms/token** of compute/L2 time.
   - Combined with DRAM streaming and 60 GPU↔CPU PCIe split synchronizations (`30 layers × (D2H 11 KiB + cudaStreamSynchronize + H2D 11 KiB)` ≈ **2–3 ms/token**), realistic end-to-end CPU MoE expert time on i7-14700F is **~38–55 ms/token**.

### Lever 5: Can CPU Experts Overlap with GPU Attention?
**No.** Inspecting both [`/data/llama.cpp/src/models/gemma4.cpp:295-352`](file:///data/llama.cpp/src/models/gemma4.cpp#L295-L352) and [`/data/ds4-gemma4/ds4_gemma4_graph.inc:801-916`](file:///data/ds4-gemma4/ds4_gemma4_graph.inc#L801-L916):

```
Layer l Attention (GPU)
        │
        ▼
  cur (post-attn residual) ──────────────────────────┐
        │                                            │
        ├─► Dense FFN: norm[2] → gate/up → gelu      ├─► Router: router_input → logits → top-8
        │              → dense_down → norm[5] (GPU)  │           │
        │              [~0.08 ms/layer]              │           ▼
        │                                            └─► Routed MoE: norm[4] → 8 experts → norm[6]
        │                                                        │
        ▼                                                        ▼
  branch = norm[3](dense_out + moe_out) ◄────────────────────────┘
        │
        ▼
  cur = layer_scale(round(cur + branch))
        │
        ▼
Layer l+1 Attention (GPU)
```

- Layer `l` Routed MoE **cannot start** until Layer `l` Attention and Router finish on the GPU.
- Layer `l+1` Attention **cannot start** until Layer `l` Routed MoE finishes and merges into `cur`.
- In batch-1 decode, the **only** GPU work that can run concurrently with Layer `l` CPU Routed MoE is Layer `l`'s **shared Dense FFN** (`cur_mlp`), which is only `2.4 ms/token` total across all 30 layers (~0.08 ms/layer). Overlapping 2.4 ms of GPU Dense FFN with 45 ms of CPU MoE saves at most 2.4 ms.

---

## 4. Why the Active-Expert VRAM Cache Wins Decisively

Compare the three architectures across batch-1 decode, MTP draft-6 verify (`rows = 7`), and prefill:

| Metric | All-VRAM / Active-Expert VRAM Cache (Hit) | Uncached PCIe On-Demand GPU Stream | Best-Case AVX2 `--cpu-moe` (i7-14700F) | Scalar C Benchmark ([`gemma4_cpu_expert_gemv.c`](file:///data/ds4-gemma4/tests/gemma4_cpu_expert_gemv.c)) |
|---|---|---|---|---|
| **Where expert GEMV runs** | RTX 4070 SUPER (504 GB/s GDDR6X) | RTX 4070 SUPER (stalled on PCIe 4.0) | i7-14700F (28T, ~70 GB/s DDR5) | i7-14700F (28T scalar, warm L3) |
| **Batch-1 expert time (30L × 8 exp)** | **9.29 ms/tok** (`6.24 + 3.05 ms`) | ~75 ms/tok (~94 ms total) | **~38–55 ms/tok** | 278.4 ms/tok |
| **Batch-1 total decode rate** | **31.1–34.8 tok/s** (`28.7–32.1 ms`) | ~10.6 tok/s (`~94 ms`) | **~15–20 tok/s** (`~50–67 ms`) | ~3.3 tok/s (`~300 ms`) |
| **MTP draft-6 verify (`rows=7`, ~1,500 exp)** | **~24–30 ms/step** (GPU warp/block batch) | ~350+ ms/step (PCIe bound) | **~180–250 ms/step** (2.8 GiB DDR5 read) | >1,500 ms/step |
| **Prefill (`chunk=512`, all 128 exp/layer)** | **1,040 tok/s** ([`g4_vq_moe_mma_kernel`](file:///data/ds4-gemma4/ds4_gemma4_gpu.cuh#L612-L775)) | PCIe bound (~7.5 GB/chunk) | **<80 tok/s** (no FP8 tensor cores on CPU) | Unusable |

Even `llama.cpp` itself recognized that CPU MoE compute is a bottleneck when GPU VRAM is available: [`src/llama-moe-cache.cpp`](file:///data/llama.cpp/src/llama-moe-cache.cpp#L15-L635) (`--moe-cache-mib`) keeps the master `*_exps` tensors in host RAM but caches active expert slots in GPU VRAM via an SLRU policy and runs `ggml_mul_mat_id` **on the GPU**, not on the CPU. That is the exact same architectural pattern as `ds4`'s active-expert VRAM cache (`--ssd-streaming` with the 4 GiB VRAM slot cache, which already achieves a **99% slot hit rate** after prompt warmup, [`gemma4-port-handoff.md`](file:///data/ds4-gemma4/gemma4-port-handoff.md#L1818-L1821)) and the pinned-RAM / hybrid layer residency plan in [`offload_review.md`](file:///data/ds4-gemma4/offload_review.md#L156-L189).

---

## 5. What to Measure Next

### A. If you want to close out the CPU microbenchmark with a true AVX2 + cold-DRAM number (~30 minutes of C edits in [`tests/gemma4_cpu_expert_gemv.c`](file:///data/ds4-gemma4/tests/gemma4_cpu_expert_gemv.c))
Make four targeted edits to [`tests/gemma4_cpu_expert_gemv.c`](file:///data/ds4-gemma4/tests/gemma4_cpu_expert_gemv.c) to turn it from a warm-L3 scalar test into a fair AVX2 + cold-DRAM ceiling test:
1. **Persistent barrier threadpool:** Spawn the `threads` pthreads **once** outside the `iters` loop using `pthread_barrier_wait(&bar_start)` / `pthread_barrier_wait(&bar_done)` and atomic 32-row work-stealing (`__atomic_fetch_add(&next_row, 32, __ATOMIC_RELAXED)`).
2. **Pre-expand codebooks to FP16:** At startup, convert `main_book` (`8192 × 8 B` E4M3) to `uint16_t main_fp16[8192][8]` (`128 KiB`) and `res_book` (`256 × 8 B` E4M3) to `uint16_t res_fp16[256][8]` (`4 KiB`).
3. **AVX2 + F16C inner loop with hoisted `gain` and 4× `__m256` accumulators:**
   In each row, initialize `__m256 acc0 = _mm256_setzero_ps(), acc1 = ..., acc2 = ..., acc3 = ...;`. Unroll `b` by 4 codewords (32 weights = 6 bytes of `plane12` = 3 pairs of 12-bit indices):
   ```c
   __m256 xm = _mm256_cvtph_ps(_mm_load_si128((const __m128i *)main_fp16[idx0]));
   __m256 xr = _mm256_cvtph_ps(_mm_load_si128((const __m128i *)res_fp16[r0]));
   __m256 xv = _mm256_loadu_ps(x + b * 8);
   acc0 = _mm256_fmadd_ps(_mm256_add_ps(xm, xr), xv, acc0);
   ```
   Horizontal-sum `acc0 + acc1 + acc2 + acc3` at the end of the row and multiply by `gain` **once**.
4. **Cold DRAM ring buffer (720 matrices = 472.6 MiB):** Allocate a 472.6 MiB buffer holding 720 copies of the matrix so consecutive iterations stream cold from DDR5 DRAM rather than hitting the 33 MiB L3 cache.
   - *Expected result:* Warm-L3 time drops from `0.387 ms` to `~0.035–0.050 ms/matrix` (`~25–36 ms` extrapolated), while the cold-DRAM 720-matrix sweep lands at **`~38–55 ms/token`**—confirming both why `278.4 ms` was unfair and why CPU MoE still loses by **4–6×** to the GPU's `9.29 ms/token`.

### B. Where to spend engineering time instead (GPU Active-Expert VRAM Cache & Hybrid Residency)
Follow the priority order in [`offload_review.md`](file:///data/ds4-gemma4/offload_review.md#L156-L189):
1. **Batch-1 `g4_vq_moe_warp_kernel` shared-memory input staging (`x_sh`):** Stage the 2,816-float (`11 KiB`) input vector `x` in shared memory alongside the `66 KiB` codebooks (`77 KiB` total < `99 KiB/block`) so all 8 warps read `x` from shared memory instead of L1/L2 global loads.
2. **Eliminate cold-miss `pread` + unpinned `cudaMemcpy` overhead:** Keep cold expert records in a host RAM cache (or `cudaHostRegister` pinned pages when sufficient host RAM is free) and batch layer misses into a single async H2D DMA per layer on a dedicated copy stream.
3. **Partial layer VRAM residency (`--n-cpu-moe` equivalent):** When VRAM allows (e.g., 8–10 GiB budget), pin the first $K$ layers' experts permanently in VRAM and run the SLRU active-expert VRAM cache only on the remaining $30 - K$ layers.
