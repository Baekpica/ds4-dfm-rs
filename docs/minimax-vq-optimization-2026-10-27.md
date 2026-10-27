# MiniMax-M3.1 VQ optimization, 2026-10-27

Campaign target: at least three adopted prefill rounds, emphasizing long
contexts, and three decode rounds. Current adoption: **prefill 2, decode 1**.
HF publication remains pending the completed campaign. The
[October 25 report](minimax-vq-optimization-2026-10-25.md) is historical;
its rounds do not count toward this campaign.

One GB10, CUDA 13.4.92, KMD 615.71.10, busy clocks 2,177–2,190 MHz under
the unchanged 300–2,200 MHz policy. Artifact: the unchanged
[MiniMax-M3.1-Flash VQ release](minimax-vq.md#model-and-format), manifest
SHA-256 `2110c1e5cba8d59987e370d50dab6dde5d5f7fc82009cdc27d8131452536f4b0`.
CUDA runtime 13.4.92 and cuBLAS/cuBLASLt 13.8.0.4 were loaded from the
system CUDA installation; no framework libraries were loaded.

## Protocol

Rust `ds4-bench-perf`, base source `be372776` plus P1's recorded patch;
binary SHA-256 `8e70bfaa965e05d7ba1d87622d8286819c39bbf1002b56d28309d43a00a37250`.
Each sample uses a fresh process, cleared freed-driver/page cache, one GPU
job, and the host memory guard with 2 GiB reserve/trip. PSI is recorded.

```sh
DS4_MINIMAX_CUR_BF16=0 ./ds4-bench-perf --cuda -m "$MODEL" \
  --prompt-file speed-bench/promessi_sposi.txt \
  --ctx-start 8192 --ctx-max 65536 --step-incr 57344 \
  --ctx-alloc 131072 --gen-tokens 64 --warmup-prefill 4096 \
  --dump-frontier-logits-dir "$PROOF"
```

Repeat with `DS4_MINIMAX_CUR_BF16=1`, order off/on/on/off/off/on.
The 4,096-row warmup is untimed and the session restarts at position zero.
8K computes 8,192 prompt tokens; 64K computes a 57,344-token suffix after
restoring the 8K prompt snapshot. K/V stays resident. MTP is off; decode
forces 64 greedy tokens excluding EOS. This is distinct from the older
512K-allocation, SSD-tier, 2,048-row incremental DSpark sweep.

## P1: store text-prefill residuals as BF16 — adopted

The baseline 64K suffix took 52.03 s under Nsight: experts 41.35%, Q4
projections 27.54%, attention 10.33%, residual/norm 5.85%, expert reduce
2.05%. The residual's 1.812 ms per 4,096 rows already approached its
streaming-byte floor (~222 GB/s against ~240): reduce bytes rather than
arithmetic. Decode uses PDL, so overlapping kernel durations must not be
added as wall time.

Residual values already have BF16 rounding. P1 stores them as BF16 during
text prefill, carrying the format through embedding, RMS, residual adds,
head slicing and DSpark taps. Each forward selects its format: decode,
verify, media spans and traces keep F32. Tensor allocation sizes are
unchanged. `DS4_MINIMAX_CUR_BF16=0` restores the prior storage path.

Numerical class: **bit-identical**. Addition, RMS reduction order and q4p
input preparation are unchanged. The F32 RMS and both F32-residual kernel
instruction streams match the original binary's SASS exactly after
removing dump addresses/comments.

Matched model-free NCU (`--clock-control none`, 2.20 GHz, 4,096 rows,
BF16 branch) measured residual **1.883040 → 1.422912 ms**, a 24.4% time
reduction. Global load/store requests fell from 14,159,872 to 11,014,144
32-byte sectors, saving 100,663,296 bytes per call. Both use 64 registers
per thread with no spills. NCU's default clock control was excluded from
the comparison because it selected 2.42 GHz.

Unprofiled medians of three, tok/s:

| Frontier | Prefill off | Prefill on | Change | Decode off | Decode on |
|---:|---:|---:|---:|---:|---:|
| 8K | 1,175.68 | 1,197.63 | +1.87% | 21.27 | 20.97 |
| 64K | 1,100.65 | 1,121.21 | +1.87% | 20.91 | 20.61 |

All three prefill pairs improved. The short decode windows varied by
about 3%; their marginal medians show -1.4%. The final matched pair was
21.27 → 21.27 at 8K and 20.91 → 20.90 at 64K. An additional fresh off/on
pair, cold single 64K with **256 tokens**, measured **20.42 → 20.40**
decode (-0.10%) and 1,114.49 → 1,132.09 prefill (+1.58%). Together with
unchanged F32 device instructions, this supports adopting the reproducible
prefill gain; no decode gain is claimed.

Validation: the model-free fixture passed 20 width/branch/storage cases
(1, 8, 9, 128, 4,096 rows), checking CPU BF16 residual sums, exact norm,
q4p input/scales, RMS, embedding and tap rows. All 200,064 frontier logits
and all 64 greedy tokens matched the original binary at 8K and 64K in
every full-model sample. The 256-token follow-up also matched exactly.

Raw build, guard, clocks, NCU, Nsight, proofs and commands remain in ignored
`scratch/minimax-m31-20261027/`. The scoped [P1 receipt](benchmarks/minimax-m31-2026-10-27/p1.json)
retains the samples. These gates cover text performance and the changed
storage path; they do not extend serving, media or 1M qualification.

## P2: emit the indexer's MMQ input from the residual — rejected

After P1, the 64K suffix profile was 50.94 s. Its 840 separate MMQ input
quantizers cost 0.468 s and read a 100.7 MB F32 norm per 4,096-row call.
The trial emitted the exact D4 Q8 input from the residual, omitted that
F32 norm, and consumed existing context scratch without new tensors.
Independent byte parity and integrated producer/consumer probes passed;
all frontier logits and 256 greedy tokens matched at both frontiers.

The isolated residual plus quantizer fell from 1.977248 to 1.107136 ms,
with 64 registers and no spills. Three fresh unpinned pairs improved
64K prefill from a median 1,122.01 to 1,137.31 tok/s (+1.36%), but decode
fell from 20.84 to 20.36 tok/s (-2.30%). A further non-profiled pair on
the operator's performance CPUs (`taskset -c 5-9,15-19`) reproduced the
tradeoff: prefill 1,119.82 → 1,138.45; decode 20.68 → 20.29 (-1.89%).
Steady-state rates excluding the first token also fell.

Full Nsight traces preserving snapshots and proof dumps showed the
opposite decode sign (~2% faster), identical kernel counts and ~99.6%
GPU coverage. They do not explain or invalidate the repeatable unprofiled
regression. The GPU-state cause remains unresolved. The trial was removed;
the prefill gain does not outweigh the decode cost for the joint target.
The [rejected receipt](benchmarks/minimax-m31-2026-10-27/p2-rejected.json)
retains both CPU protocols. P2 contributes no adopted round.

## D1: resolve decode projection kernels before prefill — adopted

Two different prefill-only trials exposed sensitivity of later decode
measurements to initialization conditions. Broad `CUDA_MODULE_LOADING=EAGER`
reversed a decode comparison's sign but lowered minimum available host RAM
by 1.3–1.4 GiB. It was not retained. The precise cache/driver mechanism is
unresolved; no instruction-cache diagnosis is claimed.

D1 calls `cudaFuncGetAttributes` for 22 existing Q4 and joint projection
functions when allocating the MiniMax graph, after expert setup and before
prefill. [CUDA documents this as loading a kernel before execution](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/lazy-loading.html).
The diagnostic `DS4_MINIMAX_PRELOAD_DECODE=0` restores the previous setup.
No GPU arithmetic or tensor allocation changes. The four hot one-row
projection instruction streams match the prior binary exactly.

These measurements explicitly use lazy module loading, P1 enabled and the
P3 reduction trial disabled. Cold loading can use CPUs `5-9,15-19`; during
the untimed 4K warmup, the main thread is fixed to CPU 5 and helpers retain
that performance-core set. Samples exclude serving, media and MTP, and
use the same 8K → 64K snapshot/proof workload with **256 generated tokens**.
This CPU protocol differs from P1's unpinned samples and is not counted as
an optimization round. Resolving this setup issue precedes more prefill
work so later comparisons share a controlled baseline.

Three fresh-process pairs, off/on/on/off/off/on; medians in tok/s:

| Frontier | Decode off | Decode on | Change | Prefill off | Prefill on |
|---:|---:|---:|---:|---:|---:|
| 8K | 20.69 | 21.00 | +1.50% | 1,201.81 | 1,195.72 |
| 64K | 20.29 | 20.64 | +1.72% | 1,121.87 | 1,118.65 |

Every pair improved decode at both frontiers, including rates excluding
the first token. The smallest 64K paired gain was 0.79%, versus a 0.59%
control run range. Prefill marginal medians are -0.51% / -0.29%, within
the control ranges of 0.70% / 0.46%; one pair improves, so a repeatable
prefill regression was not established. No measurable peak-memory increase
or PSI occurred. The preload-only model-free timer measured 0.069 / 0.164 ms
(off/on), after CUDA initialization; this is separate from model load time.

All 200,064 frontier logits and all 256 generated IDs matched at both
frontiers in every sample. The [D1 receipt](benchmarks/minimax-m31-2026-10-27/d1.json)
retains raw samples, affinity checks, resource bounds and the measured
binary identity. P3 is compiled but inactive for this D1 comparison.

## P3: consume expert reduction in the residual — adopted

A fresh retained P1+D1 whole-workload profile measured the 64K suffix at
51.125 s: experts 41.98%, Q4 projections 27.89%, attention 10.52%,
residual/norm 4.47% and expert reduction 2.08% (1.061 s). Reduction was
already close to its streaming floor. P3 removes its intermediate BF16
branch store and read: 100,663,296 bytes per 4,096-row call, with no new
tensor allocation. Four expert offsets/scales are loaded once per row.
The fused kernel retains ascending expert F32 multiply/add order, shared
BF16 rounding, branch BF16 rounding and the residual/norm/q4p order.
`DS4_MINIMAX_REDUCE_RESIDUAL=0` restores the separate launches. Decode,
verify, media and tracing retain their existing paths.

Fourteen model-free cases cover F32/BF16 residual and shared rows, widths
9/128/4,096, the final layer and norm without q4p. Independent CPU sums and
the separate GPU path match exactly. A warmed seven-sample median improved
the isolated pair from **2.697024 to 2.331488 ms (-13.55%)**. The minimum
row traffic is about 503 MB, a 2.14 ms floor at 235 GB/s; this isolates
synthetic skewed routes and does not reproduce every producer-cache state.
NCU with clock control disabled measured 2.352352 ms versus the separate
1.368192 + 1.422912 ms, 64 registers and no spills. Issued instructions
fell 50.22% and global request sectors 40.78%; those sector counts include
cached metadata and are not DRAM byte counts. Retained F32 RMS/residual
instruction streams match P1 exactly.

Three fresh-process pairs use the same frozen binary and fixed-CPU protocol
as D1, with P1 and D1 enabled and only reduction fusion varied. All runs
explicitly use lazy loading, 256 generated tokens and resident KV:

| Frontier | Prefill off | Prefill on | Change | Decode off | Decode on |
|---:|---:|---:|---:|---:|---:|
| 8K | 1,199.88 | 1,204.84 | +0.41% | 20.85 | 21.14 |
| 64K | 1,122.69 | 1,128.05 | +0.48% | 20.45 | 20.77 |

Every pair improved prefill at both frontiers. The smallest 64K paired gain
is 0.42%, above the control range of 0.25%; all on samples exceed all off
samples. Decode did not regress (+1.39% / +1.56% marginal medians), but this
prefill change contributes only one prefill round. All 200,064 logits and
all 256 token IDs match at both frontiers in all six samples. Minimum host
availability was 7.33–7.61 GiB with PSI zero. The
[P3 receipt](benchmarks/minimax-m31-2026-10-27/p3.json) records the complete
cohort, resources, affinity and target counters. The earlier pool-only and
broad-eager diagnostic pairs remain excluded from adoption evidence.
