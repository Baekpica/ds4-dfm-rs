# MiniMax-M3.1 VQ optimization, 2026-10-27

Campaign target: at least three adopted prefill rounds, emphasizing long
contexts, and three decode rounds. Current adoption: **prefill 1, decode 0**.
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
retains both CPU protocols. Adopted counts remain **prefill 1, decode 0**.
