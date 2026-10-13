# DeepSeek V4.1 Flash on one DGX Spark: 2K–64K

Three fresh Rust-server sweeps of `vq8sh14-q4k-mtpnative`, the original Engram
tables and `grrb-code_fit_n15360-engine`. All 96 requests completed 128 generated
tokens with zero reused prompt tokens. These are full-prompt prefill rates.
Rates use the server's prefill/decode timers; TTFT and wall time are separate
CSV fields.

![DeepSeek V4.1 Flash prefill and decode on one DGX Spark](throughput.png)

[JLFN](https://github.com/JLFN) ported DeepSeek V4.1 Flash serving from the
[YoungAi engine](https://github.com/fodelf/YoungAi/tree/7fb26cc332d740416792204a29c7d13d2986caf5)
to `ds4-dfm-rs` in [PR #80](https://github.com/Baekpica/ds4-dfm-rs/pull/80).
The VQ GGUF and code GR/RB sidecars come from
[YoungAi's model repository](https://huggingface.co/wenzhouwu/YoungAi-DeepSeek-V4.1-Flash/tree/84b86701456f4e4b357e41340b3b88914f51834d).
The original model and Engram weights are from
[DeepSeek](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/tree/5254448103b6254fc646dacbca1cbbf20e9216c1).

| Metric | Median of three run means |
|---|---:|
| Full prompt prefill | 1,039.65 tok/s |
| Decode, automatic embedded MTP | 38.17 tok/s |

The [Prefill and Decode tables](tables.md) retain every sample and the actual
input token counts. Curves show the median at each frontier; shading is the
observed minimum–maximum of three runs, not a confidence interval.
The summary above takes each run's arithmetic mean across 32 frontiers and
then the median of those three means, following the
[Qwen MXQ card's presentation](https://huggingface.co/Baekpica/Qwen3.8-Flash-Next-Mixed-Quant-SSD-PLE-GGUF/blob/f92da0134498acfefe55aac7bd29e4dcdbbdbfe7/README.md).
Qwen's card measures incremental prefill; this report measures full prompt
prefill. The two figures do not establish a matched cross-model comparison.

| Frontier | Actual input | Prefill median | Decode median |
|---:|---:|---:|---:|
| 2K | 2,149 | 775.7 | 42.6 |
| 8K | 8,449 | 1,059.3 | 41.9 |
| 16K | 16,546 | 1,120.0 | 41.3 |
| 32K | 32,749 | 1,030.2 | 40.0 |
| 48K | 49,440 | 1,042.5 | 40.0 |
| 64K | 66,608 | 969.5 | 39.8 |

## Conditions

- One NVIDIA DGX Spark / GB10, driver 615.71.09; CUDA compiler 13.3.73,
  `sm_121a`. Clock range remains 300–2200 MHz. Observed request-clock medians:
  2,184–2,190 MHz; all raw clock samples are retained locally.
- Merged source `cd5c2c69`; runtime `64503131`, server SHA256
  `e43f03de0e0a675e0f04eed11d96c24c168640300c5ff10ea6b2033499ab750a`.
  No inference changes or rebuild for this campaign.
- Same binary, model and prompt requests in all three fresh server processes.
  Single serial sequence, automatic embedded MTP, default CUDA graphs/VMM,
  temperature 0, reasoning off. No profiler, emitted token trace or logits dump.
- Pinned `speed-bench/promessi_sposi.txt`, SHA256
  `f53e0d80cb2d4492d24ebd63c7000c397b16ae70f9bf09b3763e5d8323ec209f`.
  The 8,192-byte calibration yields 2,512 tokens. Measured text starts at byte
  262,144, separate from the calibration. Nominal frontiers use the measured
  3.2611 bytes/token ratio; the tables preserve actual rendered prompt lengths.
- Every request prefills its entire prompt; the appended chat suffix prevents
  the next longer user message from extending the saved prompt checkpoint.
  Prefix reuse stays enabled by default and every measured cached-token count
  is zero. Disk KV is off. Engram file caching can warm within each process;
  `clear_cache` runs between processes.
- The family uses its 1,048,576-token metadata boundary and allocates for the
  actual request. No `--ctx` override is used. The measured endpoint has
  66,608 input tokens plus 128 generated tokens; metadata is not a 1M test.
- External reserve/trip and common server memory floor: 2 GiB. PSI is recorded
  without terminating on pressure alone. Swap remains disabled in the scope.
  Minimum available host memory: 6.57 GiB; maximum PSI full avg10:
  0. Cgroup high/max/OOM events: 0 / 0 / 0.

The 113,556,639,424-byte GGUF and 203,073,077,576-byte Engram files retain the
[review campaign's artifact hashes](../ds41-spark-review-2026-10-13/README.md#model-and-source).
Engram file identities still match their completed SHA256 verification. The
source/port numerical, graph, MTP and API checks in that report apply to the
unchanged runtime paths; this campaign adds throughput at larger frontiers.

## Validation and limits

All 96 requests return 128 generated tokens, `finish_reason=length`,
finite positive rates and zero cached prompt tokens. Request hashes and
actual input counts match at all 32 frontiers across the three runs.
Response choices match across all three runs at 32/32 frontiers;
the receipt retains each choice hash. All 96 periodic stats snapshots
report zero census, observation and governor faults. The guard never trips,
and every observed cgroup event counter remains zero. The kernel journal
covering the campaign has no matched GPU or OOM errors. Stats snapshots
are periodic and do not claim a final post-request snapshot at every frontier.

This is one serial text workload. It does not establish broad long-context
answer quality, concurrent banks, session snapshots, other sidecars or 1M
throughput. The earlier 40K screenshot comparison remains a separate dated
[review result](../ds41-spark-review-2026-10-13/README.md).

## Reproduce

With the verified model paths assigned to `MODEL`, `ENGRAM` and `CODE_SIDECAR`,
repeat the existing sweep three times, stopping the task server and clearing
filesystem caches between runs:

```sh
clear_cache
DS41_SWEEP_OUT="$OUT/run-1" \
DS41_SWEEP_CSV="$OUT/run-1.csv" \
DS41_SWEEP_NO_GGUF_SHA=1 DS41_SWEEP_NO_SVG=1 \
DS41_SWEEP_PORT_ARGS="--mem-floor-gb 2 --no-update-check" \
python3 tools/host_memory_guard.py \
  --max-gib 128 --high-gib 128 --reserve-gib 2 --trip-gib 2 \
  --psi-full-limit 0 --timeout 2700 --log "$OUT/run-1.guard.jsonl" \
  -- speed-bench/ds41_spark_sweep.sh ./ds4-server \
     "$MODEL" "$ENGRAM" "$CODE_SIDECAR" 8099
```

Use unique `run-2` and `run-3` outputs for the other processes. The sweep's
defaults are 2,048 through 65,536 in 2,048 steps and 128 generated tokens.
`DS41_SWEEP_NO_GGUF_SHA=1` reuses the prior completed artifact verification;
it does not replace verification for a different model file.

Regenerate the tables, summary and PNG with
[the plot script](../plot-ds41-64k.py) and Matplotlib 3.10.6:

```sh
python3 docs/benchmarks/plot-ds41-64k.py
```

Published data: [run 1](run-1.csv), [run 2](run-2.csv), [run 3](run-3.csv),
[summary](summary.json), [identity and request receipts](evidence.json).
Raw requests, responses, clocks, server logs, guards and memory telemetry are
kept under `scratch/ds41-64k-2026-10-13/`; the receipt binds them by hash.
