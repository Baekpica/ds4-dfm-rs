# MiMo prefill optimization — 2026-09-26

GB10, CUDA 13.3, sm_121a; MiMo V2.6 Flash RL MQ-IQ2-XXS-XS-Q8.
Fixture: `speed-bench/promessi_sposi.txt`, 8192 prompt tokens, 128 greedy
output tokens, 4096-token prefill chunks, MTP/DFlash off. Each sample uses
its own process after a separate warmup process, sharing one VMM owner.
GPU clock range stays 300–2200 MHz; observed busy clocks are 2190 MHz.
[Workload and artifact hashes](sum-residual/workload.json).

## Routed sum and residual fusion

**Accepted: round 1 of 3.** The final build improves prefill in all three
fresh pairs, with exact 152,576 logits and all 128 tokens in every run.

| Final build | Prefill tok/s | Decode tok/s |
|---|---:|---:|
| Original median | 1192.96 | 24.44 |
| Fused median | 1199.38 | 24.38 |
| Change | +0.54% | −0.25% |

Prefill pair gains: +0.49%, +0.74%, +0.51%. The first two final pairs
overlapped CPU SASS analysis; the last pair was uncontended and decode
improved 24.44→24.47. All samples remain recorded. Decode ranges overlap;
there is no claim of a decode speedup. Retain the consistent prefill gain
below 1% because resource costs decrease and no repeatable decode penalty
was established. [Final samples and proofs](sum-residual/final-ab.json),
[build receipt](sum-residual/receipt.json).

Fresh whole-workload nsys measured 6.90556 s prefill. Expert sum consumed
226.85 ms in 94 calls; each was followed by a separate residual add.
The earlier adjacent-call trace measured 300.51 ms combined (4.37% of
prefill). Full isolated NCU matched production geometry: 4096 rows,
4096 channels, eight route slots, grid 65536, block 256.

NCU showed 2.39 ms sum plus 0.777 ms add, no spills, 0.23% L2 hit rate,
and dominant long-scoreboard waits. Fusion removes a 64 MiB temporary
write and its 64 MiB reread per call. The dense-FFN scratch allocation
remains; no additional VRAM or arithmetic is introduced.

| Isolated path | Median ms | Change |
|---|---:|---:|
| Separate sum + in-place add | 3.175731 | — |
| Scalar fusion | 2.694980 | −15.14% |
| float4 fusion | 2.924834 | −7.90% |

Three fresh processes per path, rotated order, 16 warmups and 100 measured
iterations. All outputs after 116 accumulated passes were byte-exact.
Scalar fusion was selected. It uses 28 registers and no spills; full NCU
executed instructions fell from 60,817,408 across two kernels to 30,932,992.
See [isolated samples](sum-residual/isolated.json),
[baseline NCU](sum-residual/ncu-baseline.txt), and
[fused NCU](sum-residual/ncu-fused.txt).

### Numerical and dispatch contract

Sum slots 0…7 from zero with separate FP32 multiply/add rounding, then add
the residual. Do not seed the route accumulation with the residual or
contract the operations into FMA. The CUDA path covers 32–8192 rows;
narrow decode retains the existing sum/add path. `DS4_MIMO2_SUM_RESIDUAL=0`
restores the separate path. Unsupported dispatch returns before GPU work;
execution errors abort instead of replaying the fallback.

[Kernel tests](../../../tests/mimo2_sum_residual.cu) compare byte parity
at 1, 31, 32, 33 and 4096 rows. [Wrapper tests](../../../tests/mimo2_sum_dispatch.cu)
cover default/diagnostic dispatch, invalid buffers, alignment, refusal
without mutation, and 32/33-row parity. Build either with:

```sh
nvcc -O3 -g -lineinfo --use_fast_math -std=c++17 -arch=sm_121a \
  tests/mimo2_sum_residual.cu -o /tmp/mimo2-sum-test
/tmp/mimo2-sum-test
```

### Build control

The initial candidate used `--split-compile 8`. SASS comparison found seven
changed existing attention/router kernels, including new local loads/stores
in HMMA prefill. This candidate's OFF control was slower than the original.
Its three-arm A/B remains [recorded](sum-residual/candidate-ab.json), but is
not the final build qualification.

The final build uses the original compiler flags. All 36 existing project
kernels executed by the fixture have byte-identical SASS to the original
binary. The three external cuBLAS kernels were outside this comparison.
See the [per-kernel comparison](sum-residual/codegen.json).

Raw reports, binary snapshots, full logits, guards, scripts and rejected
patches remain under `scratch/mimo-prefill-20260926/`; this directory keeps
compact measurements and proof receipts. Dated evidence qualifies this
artifact/workload, not all serving modes.
