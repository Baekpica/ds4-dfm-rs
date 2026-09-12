# Solar Open2 long-context prefill, round 5

Round 5 adds a warp-specialized K-FP8/V-FP4 prefill attention kernel
(`ds4_fattn_hmma_solar_ws_kernel`, `cuda/mmq/ds4_fattn.cu`) whose output
is byte-identical to the GQA-pair kernel and which runs the 64K tail
component in 114 ms per layer instead of 292 ms. It ships **opt-in**
(`DS4_SOLAR_FATTN_WS=1`): its 64K full-model run hard-froze the host, and
the cause is the GB10 platform's power trip, not a kernel defect. The 8K
A/B is exact with prefill +1.8 %; the 64K candidate has no full-model
sample. The [four earlier rounds](solar-open2-optimization-2026-09-12.md)
retained nothing; this report covers only round 5.

## Why attention

The retained path's 64K profile (round 1 control, 84.3 s of GPU prefill)
put 28.9 s in `ds4_fattn_hmma_gqa2_kernel`: 192 launches, one per GQA layer
(12 of 48) per 4,096-token chunk (16). Every other phase scales linearly:
the 8K profile's non-attention time, 6.88 s, times eight is 55.0 s, the
64K non-attention time. Attention is the only term that grows with the
square of the context, so it is the whole reason prefill throughput falls
from 1,119 tok/s at 8K to 779 tok/s at 64K.

The tensor floor sets the target. A microbenchmark on this GB10 measures
128 TFLOPS for `mma.sync.m16n8k16` f16 with fp32 accumulation (the same as
f16 accumulation; no consumer-class halving, 48 SMs, 3.0 GHz maximum SM
clock). The 64K tail component, 4,096 queries against 65,536 keys for 64
heads, is 8.8 TFLOP, so 69 ms at peak. The pair kernel needed 292 ms: 131
registers hold it to one 256-thread CTA per SM, and each 64-key tile is
filled synchronously (scale loads, barrier, packed K/V loads, decode,
barrier) before the eight warps touch a tensor core. Nothing overlapped the
global round trip or the decode with HMMA work.

## The kernel

Twelve warps per CTA, one CTA per SM, 95 KB of dynamic shared memory:

```
warps 0-7   consumers  two Q heads x 64 queries, the pair kernel's HMMA
                       layout and 16-key online-softmax steps, unchanged
warps 8-11  producers  cp.async raw K/V rows and scale words into a
                       two-stage raw ring (two tiles in flight), then
                       decode into double-buffered half tiles

producers   raw[0] raw[1] raw[0] ...          cp.async, tile t+2 issued
            decode->half[0]  decode->half[1]   after tile t is read
consumers   FULL[0] wait, walk half[0], EMPTY[0] arrive, FULL[1] wait ...
```

Named barriers 1 to 4 (count 384) pair one producer arrive with one
consumer sync (`FULL[b]`) or the reverse (`EMPTY[b]`). Consumers never wait
for a decode and producers never wait for HMMA work. The kernel uses 168
registers, no local memory, and the sanitizer's memcheck, racecheck and
synccheck report nothing on the ragged, windowed and ring shapes.

Three further changes are exact by construction:

- **Paired conversions.** Producers convert two elements per instruction
  (`e4m3x2` and `e2m1x2` to `half2`, both exact) and pack both fp32
  products with one `cvt.rn.f16x2.f32`. Every element still sees the same
  fp32 multiply by the row scale and one round-to-nearest to half, so the
  staged tile matches `solar_fattn_fill_kv_tile` byte for byte at half the
  instruction count.
- **Interior steps.** When no causal or window edge falls inside a warp's
  64 keys (row 0 has the warp's smallest position; `qfirst` grows with
  position), the mask chain cannot change any finite score and is skipped.
  `__fmul_rn` keeps the score scaling a plain multiply; in the masked path
  the select already blocks FMA contraction.
- **Identity rescale.** The accumulator rescale runs only when some row in
  the warp has a factor other than exactly `1.0f`. Deep in a walk the
  running maximum rarely moves, so most steps save 64 multiplies per lane.

Numerical contract: each query row consumes the same 16-key steps in the
same order with the same operations and the same operands as the pair
kernel. `tests/test_solar_fattn.c` (`make test-solar-fattn`) compares the
two kernels byte for byte at widths 1 to 257, windows 0/33/127, 2/4/8 KV
heads (8-byte and 16-byte copy paths), a 4,353-row ring and the 4,096-query
chunk at positions 0, 4,096 and 61,440 with capacity 65,601.

## Component results

Synthetic K-FP8/V-FP4 rows, 8 KV heads, 64 Q heads, head dim 128, median
of three launches on the idle GPU. All variants below match the pair
kernel byte for byte unless marked.

| Shape | Pair kernel | Round 5 | Ratio |
|---|---:|---:|---:|
| 4,096 queries at position 0 (chunk 0) | 6.91 ms | 3.91 ms | 1.77 |
| 4,096 queries at position 4,096 (8K tail) | 19.9 ms | 10.9 ms | 1.83 |
| 4,096 queries at position 61,440 (64K tail) | 292 ms | 114 ms | 2.57 |

Intermediate probes (`scratch/solar-longctx/r5/probe`) explain the design:

| Variant (64K tail) | ms | Note |
|---|---:|---|
| cp.async single raw stage, all warps decode, 2 CTAs/SM (128 regs) | 133 | first pipeline |
| + identity rescale skip | 124 | |
| + paired conversions | 118 | |
| warp-specialized (retained) | 114 | no spills, 1 CTA/SM |
| ablation: no softmax (wrong output) | 108 | softmax costs 16-25 ms |
| ablation: HMMA + ldmatrix + barriers only (wrong output) | 75 | 92 % of the tensor floor |

Rejected: intra-warp software pipelining (issuing step s+1's Q.K^T before
step s's softmax gained nothing; in-order issue stalls on the busy tensor
pipe before the softmax instructions can issue), head-group ping-pong
through named barriers (168 ms and 223 ms; the barrier round trips cost
more than the overlap gained), a forced 2-CTA/SM pair kernel (128
registers with spills, no gain over 1 CTA), and 8-byte copies where 16-byte
copies are aligned (4 % slower). ncu on the retained kernel: tensor pipe
63 % of peak, 0.53 eligible warps per scheduler; the two consumer warps per
scheduler enter their softmax phases together, so that latency is still
exposed. That is the remaining exact lever.

## Full-model results

`DS4_SOLAR_FATTN_WS=0` (pair kernel) against `1` (round 5), three fresh
unprofiled processes per arm after a separate warmup, medians. All
1,179,648 frontier logits and 64 generated IDs per 8K sample match byte for
byte across both arms; the 64K control's proofs match the round 1 control.

| Prompt tokens | Metric | OFF | ON | Change |
|---:|---|---:|---:|---:|
| 8,192 | prefill tok/s | 1,118.5 | 1,138.5 | +1.8 % |
| 8,192 | decode tok/s | 18.07 | 18.03 | -0.2 % |
| 8,192 | first decode s | 0.1492 | 0.1527 | +2.3 % |
| 65,536 | prefill tok/s | 782.2 | none | |
| 65,536 | decode tok/s | 13.70 | none | |
| 65,536 | first decode s | 0.1666 | none | |

The 8K comparison's recorded verdict is `Inconclusive`: the first-decode
sample-extrema envelope reached +5.89 %, past the unchanged 3 % ceiling,
as it did for round 4. That metric times the first single-token decode
after prefill, proof output and snapshot creation; the candidate kernel
runs only for chunks of at least 64 tokens and never in decode, so the
envelope reflects the ~5 ms jitter of a 150 ms measurement. The prefill
gain is small at 8K because attention is 4.6 % of 8K prefill; the 8K trace
shows the candidate's 24 attention launches at 186 ms against 333 ms.

The 64K OFF arm completed (guard minimum 12.18 GiB available). The 64K ON
arm did not: see below. No candidate 64K sample, proof or profile exists,
so this round claims no full-model 64K speedup. The component ratio
(2.57x on the tail chunk) and the OFF profile (attention 28.95 s of 84.10 s
of prefill kernels) bound the expectation at roughly 84 s to 67 s, or
782 to about 980 tok/s, if the platform sustained it.

## The second host freeze and its cause

At 23:14 KST the 64K ON warmup entered its first chunk; the host stopped
answering about a minute later and needed a forced reboot, the same
sequence as round 1's prefetch incident. `last -Fx` records the boot as
`crash` with no shutdown entry, the journal ends on routine lines, and the
guard's last sample (23:15:05) shows 14.5 GiB available with zero memory
pressure. No OOM, Xid, hung-task or thermal message was written.

A three-second power probe on the 64K tail shape separates the two kernels
(`nvidia-smi --query-gpu=power.draw,clocks.sm,temperature.gpu -lms 200`):

| Kernel (64K tail, idle host) | GPU power | SM clock | GPU temp |
|---|---:|---:|---:|
| Pair kernel | 64.3 W | 2,554 MHz | 52 C |
| Round 5 kernel | 105.6 W | 2,418 MHz | 66 C after 3 s |

The tensor pipe goes from 25 % busy to 63 % and the draw rises by two
thirds. GB10 systems are reported to power off or freeze without any log
line under sustained draw above roughly 90 W, and the documented
workaround is an SM clock cap (`sudo nvidia-smi -lgc 300,2200`, about 5 %
throughput on LLM inference), persisted with a systemd unit; some units
needed a power-brick reset or replacement. This host is a Lenovo
ThinkStation PGX (product 30KLS02X00, SBIOS S0QKT0EA of 2026-07-13, driver
610.43.02) with a 3,003 MHz maximum SM clock, so the cap value for it is
untested. Round 1's register lookahead raised the tensor duty in the same
way, which explains why two independent kernels failed only at 64K, only
in the full model (the isolated component runs in short bursts), and only
after twenty minutes of 64K control load.

The user chose to keep the kernel opt-in rather than apply the cap and
re-measure. Until a clock cap is in place and a full 64K run under it
completes, `DS4_SOLAR_FATTN_WS=1` must not be used for long-context
prefill on this class of host. The round 1 lookahead is not exonerated of
other faults, but this evidence makes the platform power trip the
sufficient explanation for both incidents.

References: [hard power-off under sustained GPU load at ~90 W](https://forums.developer.nvidia.com/t/hard-power-off-under-sustained-gpu-load-at-90w-persists-after-full-platform-firmware-update/378315),
[DGX Spark GB10 hard freeze under sustained load](https://forums.developer.nvidia.com/t/dgx-spark-gb10-hard-freeze-under-sustained-load-rcu-stall-on-cpu-11-watchdog-kdump-both-fail-working-pstore-only-crash-capture-recipe-eviden/381655),
[clock-cap recipe](https://github.com/tonyd2wild/dgx-spark-hard-poweroff-fix).

## Protocol and evidence

Unchanged from the four-round report: one GB10 host, CUDA 13.3,
`sm_121a`, Solar MXQ-v1 (11 shards), `speed-bench/promessi_sposi.txt`,
independent 8,192- and 65,536-token prompts, 64 greedy tokens, MTP off,
K-FP8/V-FP4, both prefill chunk controls 4,096, one resident VMM/base
owner (`ds4_weight_server --backend vmm --scope base --reserve-gb 16`),
guarded workers (12 GiB reserve and trip), three fresh unprofiled processes
per arm after a separate warmup, one Nsight process per arm, `ds4-perf
compare --logit-atol 0 --logit-rtol 0 --regression`.

- [Measured samples](solar-open2-2026-09-12-r5-rounds.csv): the nine
  completed samples in execution order.
- [Evidence summary](solar-open2-2026-09-12-r5-evidence.json): medians,
  proof hashes, guard minima, the 8K compare metrics, the power probe and
  the host identity.
- Raw scouts, guard logs, the probe (`probe/fattn_probe.cu`, `build.sh`)
  and the power samples stay under `scratch/solar-longctx/r5`.
