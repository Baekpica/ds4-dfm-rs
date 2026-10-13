# DeepSeek V4.1 (ds41) warm-path A/B - the sidecar/engram open cache, 2026-10-13

Both sides of one five-case HTTP run: the port with the per-process asset
caches (KU-1) against the reference engine on the same artifacts, the same box
and the same harness. This record carries BOTH engines, which the previous
five-case record (docs/benchmarks/ds41-spark-2026-10-13/) did not.

## Result

Warm-prefix TTFT (the metric the unit targets: the second and later requests
of a session, which reuse the checkpoint), median of two warm samples:

| case | prompt tok | engine warm | port warm | engine/port | cold port | cold engine | decode port | decode engine |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| prose_short | 50 | 86.0 ms | 16.9 ms | 5.09x | 1.206 s | 0.843 s | 47.04 | 46.82 |
| code_short | 68 | 90.8 ms | 12.3 ms | 7.35x | 0.376 s | 0.452 s | 41.03 | 41.24 |
| table_short | 88 | 80.9 ms | 17.9 ms | 4.52x | 0.357 s | 0.441 s | 47.90 | 47.87 |
| prose_long | 8607 | 130.3 ms | 85.0 ms | 1.53x | 6.929 s | 7.364 s | 43.34 | 42.74 |
| table_long | 17614 | 131.0 ms | 80.7 ms | 1.62x | 14.403 s | 16.107 s | 59.66 | 59.43 |

Decode is at parity on every case; cold prefill is faster on the port for four
of the five (the exception is prose_short, where the port pays the one-time
sidecar mount inside its first request - see Limits).

## What was measured

The unit found the port doing, on every request, the file work the engine does
once when it opens its state: the ②/③ sidecar mount (a 40 MB `gr/rb` directory
read, its device upload and its digest fold) and the engram open (GGUF,
inventory, wire, hash tables, shards, io_uring ring). The instrumented split
(control arm, `DS41_NO_SETUP_CACHE=1 DS41_PHASE_LOG=1`):

    host phase sidecar mount:        244.1 / 254.7 / 207.4 ms
    host phase engram feed open:       7.9 /   4.9 /   4.5 ms
    host phase host prelude:         252.0 / 259.6 / 211.9 ms

With the cache on, the same three samples read `sidecar mount 0.0 ms`,
`engram open 0.0 ms`, `host prelude 0.1 ms`, and the server's own `ttft_ms`
fell from 996.1 / 268.2 / 220.4 (cold / warm / warm) to 1072.2 / 13.4 / 9.7.

## Protocol

- Host: DGX Spark GB10. The runs did not sample clocks (the GB10 query returns
  N/A through this shell); treat absolute wall numbers as this box, this
  window, and compare only within the record.
- Same GGUF, same `-grrb-code_fit_n15360-engine` sidecar directory, same
  `--engram-dir`, temperature 0, greedy, one stream, three samples per case.
- Harness: `tests/ds41_http_suite.py` on `/v1/chat/completions` with
  `stream_options.include_usage`; rep 0 of each case is cold, reps 1-2 repeat
  the identical bytes, and the prompt `prompt_sha` column proves the bytes are
  the same on both sides. Case sizes are character budgets, so the token
  counts are the server's own (50 / 68 / 88 / 8607 / 17614).
- The engine side ran first, then the port on the same case bytes
  (sequential, see Limits).

Command lines (the engine first, then the port, one at a time on this box):

    /home/leandro/youngai-src-restructure/ds4-server --cuda -m <gguf> \
        --zchain <engine dir> --engram-dir <engram> --host 0.0.0.0 --port 8098
    /home/leandro/ds4-dfm-rs/ds4-server --cuda -m <gguf> \
        --zchain <engine dir> --engram-dir <engram> --mem-budget-mb 110000 \
        --host 0.0.0.0 --port 8000
    python3 tests/ds41_http_suite.py --url http://127.0.0.1:8098 \
        --engine youngai --out raw.csv
    python3 tests/ds41_http_suite.py --url http://127.0.0.1:8000 \
        --engine ds4-dfm-rs --out raw.csv --append

Builds: port `ds4-server` sha256
`f0b3d1c0ba9058c7660e9a5d89144c61045682a921cfc487cbb299680299d7c9`, built on
the Spark from `2fb144b` plus the KU-1 patch (commit `9bf2847`); engine
`ds4-server` sha256
`48ecd223ddc4ec7ef8e851d6b2877dd7a52d883e73de50e1e6abfffb07b336f5`, built from
`/home/leandro/youngai-src-restructure` at `fc54d1e1e4feb056a3dc4fd7204876bc435c8e4b`.

## Correctness

The same port build passes the reuse gate on the Spark
(`tests/ds41_kv_reuse_gate.sh`, `GATE-RC=0`): B restored 3725/3726 tokens with
prefill suffix 1 and reproduced A's 47 emitted tokens and 47 whole logits rows;
C restored the identical prompt whole (3726/3726) and matched B bit for bit.
The cache re-notes the same mount digest, so the KV checkpoint's identity is
unchanged. `cargo test -p ds4-core --lib v41_run` is 4/4.

## Limits

- Sequential, not interleaved: both servers hold the model, so they ran one
  after the other. Cross-window wall comparisons are not valid on this host,
  so read the ratios per case, and note the engine side ran first.
- Three samples per case, one run; single stream, no concurrency.
- Cases are character budgets, not the archived case bytes (only their
  `prompt_sha` survived), so this record does not reproduce the archived table
  case for case - it measures both engines on identical bytes in one window.
- The port's cold prose_short (1.206 s against the engine's 0.843 s) carries
  the one-time mount: the engine pays it at boot, the port on its first
  request. Boot-to-first-token is correspondingly cheaper for the port. Moving
  that mount to load time is a possible follow-up, not part of this unit.
- No clock sampling; the earlier records' clock medians do not transfer.

## Files

- `results-raw.csv`: all 30 samples, both engines (the harness's own format,
  canonical fifteen columns first, then its four extra).
- `bench-compare.csv`: the per-case summary above, in the layout
  `/data/YoungAi/bench/summarize.py` emits.
