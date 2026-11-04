# CPU experts and host-sourced slots in the runtime — 2026-11-04

Two lanes were added to answer one question: can Gemma-4 run on the 12 GiB
4070 SUPER at 128K without the expert set in VRAM? Both were measured on the
same request (23-token chat prompt, 40 greedy tokens, ctx 131072).

## What was built

`cuda/gemma4_cpu_moe.cuh` — the host expert path. `DS4_GEMMA4_CPU_MOE=1`
makes `g4_moe_run_routes` skip every GPU launch and compute the routed
experts on a persistent 20-thread pool: the R13+8 GEMV with the GPU's own
arithmetic (`w = bf16(gain*word)`, `x = bf16(x)`, f32 fma), the tanh GELU
with its bf16 roundings, and `out = bf16(out + down*weight)` accumulated in
ascending expert id, which is what `g4_moe_scatter_ordered_kernel` does. The
record parse and book expansion (E4M3 to FP16, exact) are the ones verified
in `tests/gemma4_cpu_expert_gemv.c`.

The engine (`ds4.c`) installs the discrete host-cold map when the option is
set, then skips the slot pool: the expert records stay in the model map and
nothing is streamed. `ds4_gemma4_graph.inc` treats that map as the cold
record plan, and `run-gemma4.sh` no longer sets `DS4_CUDA_COPY_MODEL` in this
lane.

`ds4_gemma4_stream.inc` — `DS4_GEMMA4_MOE_HOST=1` makes the slot fill read
the record from the model mapping instead of `pread` on the shard fd. The
VRAM slot cache and the GPU expert kernels are untouched; only the source of
the bytes changes.

## Equivalence

- One-token context, decode path: the raw MoE accumulator at layer 0 is
  bit-identical between the host path and the GPU path
  (`-0.0164795 -0.0166016 -0.148438 0.00390625` in both).
- Longer contexts: layers 0-4 match bit-for-bit; from layer 5 the two paths
  differ by 1-2 bf16 ULPs per element, which is enough to flip the
  discontinuous top-8 route selection. The GPU's own two prefill
  arithmetics (grouped fused vs `DS4_GEMMA4_EXACT_PREFILL`) differ by the
  same amount: on the same 16-token frontier, exact-vs-default is maxabs
  10.66, the host path is 10.47 from exact and 11.73 from default.
- The earlier four-cycle token sequences were a prompt truncated
  mid-sentence at `--ctx 16`; all three arithmetics loop there, with
  different periods.

## The three lanes at 128K (same request)

| Lane | first request (cold) | second request (prefix + slots warm) |
|---|---|---|
| host experts, CPU compute | prefill 11.3, decode 13.6 tok/s | not measured |
| GPU experts, slots filled by O_DIRECT pread | prefill 15.7, decode 25.8 | prefill 41.4, decode 37.8 |
| GPU experts, slots filled from host pages | prefill 11.6, decode 21.8 | prefill 42.5, decode 34.8 |

All three answer the same question and produce the same content.

## The miss path (DS4_GEMMA4_MOE_MISS=cpu)

`llama.cpp --n-cpu-moe` shape: a route whose expert is absent while the slot
pool is full is computed on the host instead of evicting a resident expert.
The GPU chunk sees a zero-weight stand-in for that route, which the ordered
scatter turns into a no-op, and the host contribution is added right after
the chunk with the same rounding (`g4_moe_add_one_kernel`). The host kernel
is `g4_cpu_moe_expert` in cuda/gemma4_cpu_moe.cuh: one expert, one row, the
same arithmetic as the decode path.

Measured with a deliberately small cache (128 experts, so both lanes miss
constantly), same request, decode tok/s cold/warm:

| 128-expert cache | cold | warm |
|---|---|---|
| evict + copy from host pages | 10.8 | 13.0 |
| miss -> CPU compute | 10.5 | 12.6 |

The miss policy is neutral here; the cache size is what matters (4 GB gives
25.8 cold / 37.8 warm). With 128 slots the per-token working set cannot fit,
so every lane pays either a refill or a host compute for almost every route.

## What it means

The active sixteen experts per token must be in VRAM for GPU compute; the
cold experts only need storage. That is the slot cache, and it is 1.9-2.8x
faster than computing everything on the CPU. Sourcing the slot fill from
host pages rather than O_DIRECT does not separate them on this machine
(34.8 against 37.8 warm, inside run noise): the per-token refill is small
next to the GPU work, and the first request is dominated by page faults
either way.

The CPU path stays as the fallback it was built to be: the lane for a card
whose VRAM cannot hold the working set at the requested context, and the
measurement that prices what host compute would cost.

## Metal

The Metal backend (`ds4_metal.m`, `metal/*.metal`) has no Gemma-4 support
today: 0 occurrences of gemma4 and 0 of glmvq/vq_decode/gvq in
`ds4_metal.m`, and no gemma or VQ kernel under `metal/`. The host path is
also x86 AVX2 plus CUDA copies, so it is not portable as written. On Apple
Silicon the question does not arise in the same form: memory is unified, so
the mapped expert records are already GPU-readable and neither a CPU
decoder nor a slot cache is needed.
