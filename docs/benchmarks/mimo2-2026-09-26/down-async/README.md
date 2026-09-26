# Down activation async copy — rejected

Round 4 retained `a3fd2a5c` (round 3). The fresh whole-workload trace
measured 6.71637 s prefill; IQ2_XS Down used 1.44041 s in 94 calls,
21.45% of prefill. [Whole-workload evidence](nsys-before.json).

Full baseline NCU identified activation-copy consumers waiting on global
loads as one contributor to long-scoreboard stalls. The candidate used
16-byte `cp.async` for activation tiles, retaining the upstream direct
weight loader, MMA order, tile dimensions and shared allocation. The first
activation half could overlap weight expansion; the second reused the same
buffer and still waited after the first MMA. The diagnostic specialization
was limited to the MiMo entry and measured GB10 shape. The
[archived patch](rejected-async.patch.gz) is not retained production code.

## Isolated result

| Fresh-process sample | Original ms | New OFF ms | New ON ms |
|---|---:|---:|---:|
| 0 | 17.506239 | 17.522352 | 17.737633 |
| 1 | 17.617218 | 17.601515 | 17.804127 |
| 2 | 17.615929 | 17.562212 | 17.738012 |
| Median | 17.615929 | 17.562212 | 17.738012 |

ON increased median latency by 0.69% against the original and 1.00% against
same-build OFF. It lost every pair: +0.69–1.32% against original and
+1.00–1.23% against OFF. The three arms used rotated order, 16 warmups and
100 CUDA-event iterations per process through `ds4_mmq_mimo2_down`, including
activation quantization and routing work. Synthetic weights/activations,
uniform distinct-eight routing and isolated cache residency differ from
the full model. [All samples and proof hashes](isolated.json).

The first sample from each arm dumped all 134,217,728 output floats.
Both byte comparisons passed and all three SHA256 hashes match. All nine
guard and payload exits were zero. Busy clock samples were 2184–2190 MHz;
one-second sampling does not resolve each iteration. [Clock summary](clocks.json).

## NCU explanation

| Counter | Baseline | Candidate | Change |
|---|---:|---:|---:|
| Profiled kernel duration | 14.440608 ms | 14.910848 ms | +3.26% |
| Registers/thread | 255 | 255 | unchanged |
| Dynamic shared bytes/block | 80,128 | 80,128 | unchanged |
| Local spill requests | 4,224 | 72,448 | +1,615.15% |
| Local-load sectors | 0 | 262,144 | introduced |
| Total L2 sectors | 166,162,644 | 188,161,377 | +13.24% |
| Executed instructions | 2,242,017,152 | 2,267,476,992 | +1.14% |
| INT8 tensor operations | 577,807,319,040 | 577,807,319,040 | unchanged |
| Eligible warps/scheduler | 0.505668 | 0.479424 | −5.19% |

Explicit global-load and shared-store instruction counts fell, but local
traffic, total L2 traffic and long-scoreboard/barrier/MIO stalls increased.
No extra tensor or shared allocation was introduced. Unchanged tensor
operations do not imply unchanged scalar/address work. These counters
explain why fewer explicit copy instructions were insufficient; they do
not uniquely identify the compiler scheduling cause. See exact keys and
units in [counter comparison](counters.json), full
[baseline details](ncu-baseline-details.txt) / [raw counters](ncu-baseline-raw.csv)
and [candidate details](ncu-candidate-details.txt) / [raw counters](ncu-candidate-raw.csv).

Reject this implementation after repeated isolated regression. Production
source and binaries were restored to round 3. No model A/B was run; no
end-to-end or decode benefit is claimed. This result does not reject all
async-copy designs. The [receipt](receipt.json) records standard MMQ build
flags, binary/fixture hashes and the source patch against its base. The
original probe's exact historical link command was not retained.
