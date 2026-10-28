# A.X-K2 prefill/decode campaign — 2026-10-29

Optimization rounds start from `daa0901f861eb19d3d4301123b90666fe032533e`
on `feat/perf-ax-k2`. This record covers the 62-part
[Baekpica/A.X-K2-VQ-GGUF](https://huggingface.co/Baekpica/A.X-K2-VQ-GGUF)
release, revision `a0c3b02e9b45dc67ef8dd1d48eb48034c93d6607`;
`release.json` SHA256
`3b4da5db6502611afe3ea06d83ef7cb77c2c205a9f3e93c75a2293579735e14c`.

One GB10, 48 SM, CUDA 13.4.92 / `sm_121a`, driver 615.71.10.
Boot `f5d9ee6b-7f0f-443c-b629-4f39b4519994`; the enabled clock service
keeps 300–2200 MHz. Observed busy clocks accompany each round. Actual native
libraries: system libcuda 615.71.10, cuBLAS/cuBLASLt 13.8.0.4 and
CUDA runtime 13.4.92. One model process at a time, CPU affinity 5–9,15–19,
page cache cleared between processes, 2 GiB host-memory reserve and trip.
No MTP, no drafter, no CPU weight offload.

## Round protocol

Fresh processes alternate off/on with the same executable, artifact and
prompt. A 2,048-token warmup/reset allocates lazy workspace, followed by a
cold 8,192-token prefill and 64 forced greedy tokens. This differs from the
published incremental 2K–64K sweep; its numbers are not interchangeable.
Full-vocabulary frontier logits and generated token IDs are retained.

```sh
DS4_AXK2_FLASH_BF16=0 ./ds4-bench --cuda -m /mnt/p31/data/models/A.X-K2-VQ-GGUF \
  --prompt-file speed-bench/promessi_sposi.txt --ctx-start 8192 --ctx-max 8192 \
  --warmup-prefill 2048 --gen-tokens 64 --dump-frontier-logits-dir proof-off
# Repeat with DS4_AXK2_FLASH_BF16=1 in a fresh process.
```

Prompt SHA256:
`f53e0d80cb2d4492d24ebd63c7000c397b16ae70f9bf09b3763e5d8323ec209f`.
PDL is enabled for rates and disabled only for Nsight attribution. Kernel
sums use launch correlation IDs inside `ds4.prefill` / `ds4.decode` NVTX
ranges. Isolated NCU uses `--clock-control none`, never the full model.

## P1 — direct BF16 flash output (adopted)

Baseline prefill: 15.883 s wall / 15.842 s kernels. Flash attention is
3.624 s (22.9% of kernel time); `ax_bf16_rows` is 0.553 s (3.5%), including
both query packing and latent-output packing. The 48 MiB latent copy
alone reaches 240–252 GB/s on finite random values: remove the traffic.

Single-split prefill rows >=48 write the value GEMM's BF16 operand directly.
This removes the F32 intermediate write/read and its conversion launch.
The existing consumer rounded at exactly this boundary. Class 5.1: storage
and epilogue fusion, no new arithmetic or allocation; decode and short rows
keep their F32 path. `DS4_AXK2_FLASH_BF16=0` retains the old path.

`tests/test_axk2_flash` checks dense/sparse selection, masked fillers,
causal and partial tiles, row counts 1/48/49/64/255/256 and allocation
canaries. Direct BF16 matches CPU rounding of the original F32 output.
All 2,296 original F32 SASS instructions match after normalizing constant
bank relocation offsets. Synthetic 256-row pipeline timing (flash plus
conversion) is 2.429–2.467 ms off versus 2.030–2.075 ms on. Synthetic
selection/cache distribution differs from a real model; only whole-model
measurements decide adoption.

| Fresh process | Mode | Prefill tok/s | Decode tok/s |
|---|---|---:|---:|
| 1 | off | 516.23 | 15.41 |
| 2 | on | 533.92 | 15.34 |
| 3 | on | 533.66 | 15.33 |
| 4 | off | 514.55 | 15.46 |
| 5 | off | 513.54 | 15.35 |
| 6 | on | 534.63 | 15.22 |
| Median | off | 514.55 | 15.41 |
| Median | on | 533.92 | 15.33 |

Prefill improves **3.76%**, with nonoverlapping repeat ranges. All six
processes exit 0, all 163,840 logits are finite, and full logits and all
64 generated tokens agree exactly. Busy SM clocks are 2,184–2,190 MHz;
minimum available host memory is 15.608 GiB, full PSI avg10 stays zero.

The short decode median is 0.52% lower. A fresh **512-token** follow-up
pair (on then off, otherwise identical) reverses that direction:

| Mode | Prefill tok/s | Decode tok/s |
|---|---:|---:|
| on | 534.68 | 15.29 |
| off | 516.76 | 15.05 |

Both exit 0 with identical finite logits and all 512 tokens. Prefill again
improves 3.47%; decode differs by +1.59%. Clock policy, busy SM clocks,
CPU affinity (the ten 3.9 GHz cores), model, tokens and PSI match. The
remaining process-to-process decode variation is unresolved; neither this
pair nor the short median establishes a decode gain or a consistent loss.
The unchanged decode kernel's SASS and reversed sign support retaining the
repeatable prefill gain with no identified resource/numerical cost. **This
counts only as one prefill round.** It is not sustained serving qualification.

The on Nsight trace gives 15.375 s prefill wall versus 15.883 s before.
Copies fall from 3,904 to 1,952 calls, 552.6 to 128.5 ms; flash falls from
3,624.2 to 3,488.5 ms. The value cuBLAS kernel now starts before flash ends:
its 790.5 ms raw duration contains 535.6 ms of overlap, leaving 254.9 ms
beyond earlier kernels. Kernel interval union falls 15.840 to 15.332 s.
Do not sum those overlapping durations as extra compute. cuBLAS supports
its own [dependent launches](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html#cublas-release-13-0-update-1);
`DS4_GLMVQ_PDL=0` controls this project's launch wrapper, not cuBLAS.
The profiled decode wall is 1,061.1 versus 1,065.1 ms for 16 tokens.

Measured binary SHA256 (source above plus this round's native diff):

- `ds4-bench`: `3fe81cb61dcde3b7337cc0d58fbf2da4baabe84e8920fdb6821c5c1d3b81d0ea`
- `ds4-bench-perf`: `4cfdcd9d1527adfaa37937a7e19c9416cde795ab49d9e41145c3fc232ce671ad`

Raw receipts, logs, clocks, source digests, full logits/tokens and profiles
remain in ignored `scratch/axk2-perf/`: `p1-{0-0,1-1,2-1,3-0,4-0,5-1}`,
`p1-long-{0-1,1-0}`, `baseline-trace`, `p1-trace`, `p1-timeline.txt` and
`p1-fixture.log`. The final publication sweep and further rounds are pending.

## P2 — cooperative sparse-key metadata (adopted)

Retained P1 attention is 22.69% of prefill wall. Its 256-row NCU probe
spends 64.7% of long-scoreboard samples on the selected-key/exponent chain;
its 73 GFLOP compute floor is 0.677 ms versus 2.096 ms observed. Four
warp lanes now gather four independent keys and exponents together before
the warp copies their latent rows. Class 5.1 scheduling; floating-point
operations and allocations are unchanged. F32 decode stays serial.
`DS4_AXK2_FLASH_KEYS=0` restores P1; this path requires P1's BF16 output.

All six fixture cases pass against CPU-rounded F32, including canaries.
The retained native F32 path keeps all 2,296 SASS instructions. The isolated
BF16 serial times are 2.046923/2.015478/2.043101 ms; cooperative times are
1.985931/1.947278/1.973226 ms (3.42% median reduction). Full NCU reports
2.096 → 2.038 ms, 166.22 → 148.78 million executed instructions, and
400.5 → 400.0 MB of L2-to-crossbar traffic. Both use 255 registers and an
80-byte stack. No additional allocation or floating-point work pays for
the gain. Synthetic selections differ from the full model.

The full-model protocol extends generation to **256 tokens** to reduce
short timing noise. Both arms use 8K cold prefill after the same 2K reset;
P1 stays enabled and only `DS4_AXK2_FLASH_KEYS` changes.

| Fresh process | Mode | Prefill tok/s | Decode tok/s |
|---|---|---:|---:|
| 1 | off | 534.11 | 15.37 |
| 2 | on | 540.63 | 15.36 |
| 3 | on | 539.42 | 15.51 |
| 4 | off | 534.19 | 15.14 |
| 5 | off | 532.90 | 15.44 |
| 6 | on | 540.46 | 15.37 |
| Median | off | 534.11 | 15.37 |
| Median | on | 540.46 | 15.37 |

Prefill gains **1.19%**, with nonoverlapping repeat ranges and positive
paired gains in all three pairs. Decode medians agree; retain its observed
15.14–15.51 tok/s process spread, without claiming a decode improvement.
All six processes exit 0 with identical finite full logits and all 256
tokens. Busy SM clocks are 2,184–2,190 MHz, CPU frequency is 3.9 GHz,
full memory PSI stays zero, and minimum available memory is 15.687 GiB.
Timed prefill/decode samples show no power/thermal cap. One idle startup
sample reports a power cap at 208 MHz and 0% GPU utilization, 179 seconds
before the frontier dump. NVML does not expose a GPU memory clock here.

The new whole-workload trace reduces prefill wall 15.375 → 15.185 s and
flash 3,488.5 → 3,318.3 ms (4.88%). Kernel interval union is 15.144 s;
512.8 ms of overlapping durations must not be added as extra wall time.
The profiled decode wall is 1,057.2 ms for 16 tokens. These are attribution
measurements, separate from the unprofiled rates above.

Measured `ds4-bench` SHA256:
`7e90986eccb52ba1fbc5b115bee824a126436ceb7b65ac487990075e81e4c26b`;
`ds4-bench-perf`:
`a2bbbd79dcf09bc7dfc5522f75365e08fd4b51498a50d6235b2650d61ed4896f`.
Source is `430b711f` plus this round's native diff. Raw receipts are
`scratch/axk2-perf/p2-*`, including clocks/host telemetry, full outputs,
fixture timing, source/SASS receipts and the new trace. Prefill now has
two adopted rounds; decode rounds and the final publication sweep remain.
