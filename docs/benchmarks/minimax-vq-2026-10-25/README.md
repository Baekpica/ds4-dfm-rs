# MiniMax VQ: 64K optimization and serving qualification

Work in progress. The final goal is reliable serving at the native
1,048,576-token context. At least 1,000 tok/s incremental prefill and
22 tok/s decode throughout the 2K–64K sweep are intermediate targets.
The baseline below does not meet those targets.

## Fresh baseline

`70481513faa174758b284841d86c69f89a375738`, GB10, CUDA 13.4,
driver 615.71.10; no clock-policy changes. Observed busy SM clock median
2,151 MHz (1,982–2,470 MHz); historical 2.4 GHz results are not a matched
code-only comparison. One process, in-process VMM, no competing model.

| Metric | Result |
|---|---:|
| 64K incremental prefill | 793.28 tok/s |
| 64K DSpark decode | 14.13 tok/s |
| Mean prefill over 32 frontiers | 821.97 tok/s |
| Mean prefill excluding first lazy allocation | 844.54 tok/s |
| Mean decode over 32 frontiers | 18.46 tok/s |
| Minimum host MemAvailable | 2.53 GiB |

[Raw samples](baseline.csv). The first 2K prefill includes lazy graph
allocation under memory pressure: 122.32 tok/s. It remains in the raw data
and overall mean. Each later point adds 2,048 prompt tokens, generates 128
forced greedy tokens, then restores the prompt snapshot. Allocation is
524,288 tokens; the prefill chunk ceiling is 4,096 rows; DSpark proposes
three tokens, with its measured-rate gate enabled.

The recipe remains index keys resident, main K/V on SSD. At this allocation,
the K/V file is 16.88 GiB and the pinned page pool 0.66 GiB. The benchmark's
explicit RAM snapshot separately grows to about 2.4 GiB at 64K. Ordinary
serial server disk checkpoints stream payloads to files instead.

```sh
python3 tools/host_memory_guard.py \
  --max-gib 117 --high-gib 116 --reserve-gib 2 --trip-gib 2 \
  --psi-full-limit 0 --timeout 1800 --log baseline.memory.jsonl -- \
  ./ds4-bench --cuda -m /models/MiniMax-M3.1-Flash-VQ-GGUF \
  --prompt-file speed-bench/promessi_sposi.txt \
  --ctx-start 2048 --ctx-max 65536 --step-incr 2048 \
  --ctx-alloc 524288 --gen-tokens 128 --mtp-draft 3 --csv baseline.csv
```

The guard applied a dynamic 116.20/115.20 GiB max/high ceiling and completed
with status 0. A separate full-sweep Nsight capture crossed the 2 GiB host
floor at about 46K and was stopped (exit 75); its partial samples are not
performance evidence. Subsequent captures must bound trace/snapshot memory.

Identities:

- Binary SHA256: `4cbb06c1bd15a24676e06a8d14eb3ff48a63cbc8895da344341c2a2219f93efd`.
- Release manifest SHA256: `2110c1e5cba8d59987e370d50dab6dde5d5f7fc82009cdc27d8131452536f4b0`.
- Prompt SHA256: `f53e0d80cb2d4492d24ebd63c7000c397b16ae70f9bf09b3763e5d8323ec209f`.

## Open qualification

Final matched speed/correctness, Vision preprocessing and HTTP media,
MSA-selected SSD reads with eviction, partial reuse, fresh-process disk
restore, MTP accepted-prefix/restored-context state, and measured TTFT remain
open. A configured context or old process log does not qualify a serving
shape. Preserve weight precision and the bounded SSD page pool; price
weights, derived artifacts, scratch, snapshots and profiler memory separately.

Hardware ceilings and their measurement conditions are in
[GB10 hardware limits](../../gb10-hardware.md).

## Legacy DSpark restore regression

The MMX1 payload stores target KV but no draft context. Before the fix,
all four 8K restore cases (same-session contaminated history and fresh
session; RAM snapshot and disk payload) generated three proposals from
stale/uninitialized draft rings. The target still committed the correct
anchor with identical logits; token-only tests missed the invalid draft
state.

`minimax_payload_restore` now sets the first valid draft position to the
restored frontier. The first ordinary step refills valid context. The same
four cases passed: zero draft proposals on the first step, one committed
anchor, and byte-identical full-vocabulary logits against ordinary decode.
`tests/minimax_state --legacy-expect-fallback` is the regression gate.
Both processes used a 114 GiB guard with a 4 GiB host floor. This proves
legacy restore safety, not preservation of the draft window or 1M serving.

The user reset the graphics clock range to 300–2200 MHz after this baseline
(on 2026-10-25 at about 00:26 KST). Subsequent comparisons must use fresh
matched processes under that setting and retain observed clocks. The older
2.4–2.6 GHz results do not establish a code-only speed difference.

## Bounded snapshots and draft context

The new MMX2 payload appends at most 4,096 committed BF16 K/V rows for
each of the five draft layers: 40 MiB total. It excludes pending trial rows,
preserves the ring's chronological order across wrap, and accepts MMX1
with an empty draft history. Target-only sessions consume the MMX2 tail
without allocating a drafter.

An SSD-tier `SessionSnapshot` now uses a private, unlinked file in the
configured KV directory. It reserves space before saving and flushes and
discards completed layer spans; it no longer allocates a whole payload
on the heap (about 37.97 GiB at 1M). Final Rust KV-envelope staging also
flushes and discards completed 64 MiB destination windows. Cache discard
is advisory, so bounded host pressure still needs live measurement.

Model-free codec tests passed for ring wrap, legacy/target-only restore,
invalid headers and lengths, interrupted I/O and nonseekable streams.
File ownership/error tests and exact large-envelope copy/cleanup tests
passed. The combined 8K live MMX2 gate did **not** reach inference; the
loading incident below leaves live state preservation unqualified.

## Loading pressure on 2026-10-25

After the native build, the next load stalled around 78–84 GiB of copied
weights, before graph/KV allocation. Host available memory remained
30–36 GiB while PSI full reached 81%. The scope's high/max/OOM counters
stayed zero. `systemd-oomd` killed a separate tmux scope and repeatedly
killed Hermes; the task's model was then stopped, exit 75. Hermes restarted
automatically. This is a failed load, not a state or performance result.

The run used `--psi-full-limit 0`; recording pressure without acting on it
did not protect other user scopes. Subsequent MiniMax qualification uses
`--psi-full-limit 20` with the existing 4 GiB reserve/trip floor. The saved
trace crosses 20% about 26 seconds before the first collateral kill.
The historical baseline command above is retained as provenance, not the
current launch policy.

Current boot has `kho=off`, `CmaTotal=0` and `CmaFree=0`. The driver caches
freed GPU pages as kernel-reclaimable memory; allocation/compaction remains
under investigation. CUDA reports both minimum and recommended VMM
granularity as 2 MiB, so changing that query alone cannot fix this load.
No model precision, context limit, clock, kernel or service policy was changed.

An allocation-only VMM reproducer subsequently isolated the stall from
model validation, file reads and inference. With 1 GiB allocations, creation
took 20.6 ms at 100 GiB, 862 ms at 102 GiB and 1,330 ms at 103 GiB.
At the last point, 15.21 GiB remained available but the Normal zone had only
one free 2 MiB block and none larger. The final allocation added 592
compaction stalls and 325 failed compactions. The probe stopped at its
one-second allocation limit and released every allocation; maximum PSI
was 7.97%, with no new collateral service kills.

This reproduces contiguous-block exhaustion and compaction before total
RAM exhaustion. The probe began with 87.08 GiB of cached driver pages;
it does not establish a reliable cold load or justify changing the recipe.
