# RoPE row mapping — rejected

Round 5 retained the round-3 kernel baseline. After rejecting round 4, a
fresh whole-workload trace measured 6.70559 s prefill. QKV split/RoPE used
184.07888 ms in 96 calls, 2.745% of prefill.
[Whole-workload evidence](nsys-before.json).

The baseline maps a linear element index to row/column using a runtime
64-bit divide/remainder. The candidate obtains the row from a second grid
dimension. Both supported strides contain whole 256-thread blocks, so
element mapping, block count, loads/stores and floating-point expressions
remain unchanged. The experimental dispatcher covered 32–8192 rows and
KV heads 4/8, with `DS4_MIMO2_ROPE_ROWS=0` restoring the linear path.
The [archived patch](rejected-rows.patch.gz) is not retained production code.

## Isolated result

| KV heads | Sample | Original ms | New OFF ms | New ON ms |
|---|---|---:|---:|---:|
| 4 | 0 | 1.817917 | 1.810679 | 1.873421 |
| 4 | 1 | 1.807725 | 1.804948 | 1.839965 |
| 4 | 2 | 1.810486 | 1.806689 | 1.833821 |
| 4 | Median | 1.810486 | 1.806689 | 1.839965 |
| 8 | 0 | 1.971623 | 1.976453 | 2.018554 |
| 8 | 1 | 1.980881 | 1.986745 | 2.042656 |
| 8 | 2 | 1.986047 | 1.985957 | 2.025319 |
| 8 | Median | 1.980881 | 1.985957 | 2.025319 |

ON lost every pair against both controls. Median latency increased
1.63%/2.24% against original and 1.84%/1.98% against OFF for heads 4/8.
Each fresh process used 4096 rows, positions 4096–8191, 16 warmups and
100 CUDA-event iterations. Three-arm order rotated between samples.
Synthetic finite QKV, absent neighboring kernels and repeated immutable
inputs differ from model residency. [All samples and hashes](isolated.json).

Complete Q/K/V dumps from the first sample of every arm compare byte-exact;
their SHA256 hashes match. All 18 guard/payload exits were zero. The
[focused parity test](parity.log) also passed both head counts, rows
1/31/32/33/4096, long positions through 1048575, padded-grid bounds and
host dispatch boundaries. No model dispatch qualification was performed.
Busy [clock samples](clocks.json) were 2190 MHz within the preserved
300–2200 MHz cap; one-second sampling does not resolve each iteration.

## NCU explanation

| Counter | Heads 4 baseline → candidate | Heads 8 baseline → candidate |
|---|---:|---:|
| Profiled duration | 1.781664 → 1.840640 ms | 1.953376 → 1.964384 ms |
| Executed instructions | 133,398,528 → 74,907,648 | 146,210,816 → 82,182,144 |
| Active warps/scheduler | 9.533 → 8.732 | 9.484 → 9.017 |
| Eligible warps/scheduler | 0.271 → 0.152 | 0.267 → 0.147 |
| Global load sectors | 15,859,712 → unchanged | 17,039,360 → unchanged |
| Global store sectors | 6,946,816 → unchanged | 7,602,176 → unchanged |
| Registers/thread | 22 → unchanged | 22 → unchanged |
| Spills | 0 → unchanged | 0 → unchanged |

Baseline NCU already showed long-scoreboard latency, not saturated integer
throughput. Its largest sampled waits consume loaded QKV/table operands.
Removing about 44% of executed instructions leaves memory requests intact;
eligible warps fall and timing worsens. Per-issued-instruction stall ratios
also change their denominator, so their increase alone does not measure
additional total stall time. These observations do not uniquely identify
the scheduling cause. See [exact counters](counters.json), full baseline
[heads 4](baseline-heads4-details.txt) / [heads 8](baseline-heads8-details.txt)
and candidate [heads 4](candidate-heads4-details.txt) /
[heads 8](candidate-heads8-details.txt) reports; all four raw CSV exports
are retained alongside them.

Reject after repeated isolated regression. No model A/B was run, and no
end-to-end or decode benefit is claimed. Source, standard build commands,
binary hashes, full NCU guards and profiler receipts are in
[receipt.json](receipt.json). Archived probe sources reproduce the experiment
only with their recorded source state and original include paths.
