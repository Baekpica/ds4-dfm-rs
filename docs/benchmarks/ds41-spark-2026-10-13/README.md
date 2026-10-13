# DeepSeek V4.1 Flash (ds41) five-case benchmark

Recorded decode, cold prefill and warm-prefix TTFT for the ds41 port on the
DGX Spark, on the recorded `ds41` port checkout `2fb144b`. This is the port's
own side of a five-case HTTP benchmark; the harness talks plain HTTP to the
running server and both sides were fed identical case bytes.

Not live state. It retains the recorded host, checkout and workload, and it does
not qualify a new build. The engine's side of the same run is not carried here.

## Protocol

- Host: DGX Spark GB10. These outputs do not record the driver, CUDA version
  or clocks; another campaign's environment cannot fill those gaps.
- Same base GGUF and the same coding sidecar on both sides of the run,
  temperature 0, greedy, single stream, 3 samples per case.
- Five cases across three content types at two prompt lengths. `decode t/s` is
  the streaming throughput after the first token; the warm rows reuse the
  prefix KV, so their TTFT is the reuse path; cold prefill is the first (cold)
  sample of the two long cases.
- The harness (`suite.py`) is not carried in this record; the outputs below are
  the recorded artefacts.

## Results (median of 3)

| case | prompt tok | completion tok | decode t/s | cold prefill t/s | warm TTFT ms | e2e t/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| prose_short | 54 | 512 | 36.26 | 82.1 | 190.9 | 35.75 |
| code_short | 54 | 512 | 54.33 | 91.4 | 196.6 | 53.03 |
| table_short | 43 | 512 | 65.41 | 79.1 | 194.1 | 63.88 |
| prose_long | 10356 | 32 | 31.08 | 1286.0 | 265.7 | 25.15 |
| table_long | 10377 | 256 | 59.14 | 1283.4 | 273.4 | 55.77 |

Decode spans 31.1-65.4 t/s across the five cases. Draft acceptance may explain
the content-dependent spread, but these CSVs contain no acceptance counters.
Prefill at ~10.4k tokens reads 1286 and 1283
t/s; the short-case cold prefill (82-91 t/s) is a 43-54-token prompt and is a
latency number, not a throughput one.

The CSV prefill values match prompt tokens divided by TTFT after rounding.
They include first-token latency; the frontier sweep instead reports the
server's prefill timer. These rates are different measurements.

## Files

- `bench-compare.csv`: per case, median of 3 (the table above).
- `results-raw.csv`: every sample (3 per case, cold and warm).

Case bytes are proven identical across samples by the `prompt_sha` column in
`results-raw.csv`:
prose_short `c55f11e5ab6295a5`, code_short `e791a3e572806a31`,
table_short `9e2049a005deec0c`, prose_long `1a220de99d8928d5`,
table_long `8a03aff2d1070268`.

## Limits

- One run per case, median of 3 single-stream samples; not a concurrency or
  sustained-throughput measurement.
- Decode is content-dependent; read it per case, not as one family rate.
- The warm rows measure the prefix-reuse path, so their TTFT is not a cold
  prefill.
- No broad task-quality claim follows from these inference checks.
