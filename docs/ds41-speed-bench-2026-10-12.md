# DeepSeek V4.1 Flash (ds41) speed-bench sweep, 2026-10-12

The ds41 frontier sweep in the speed-bench CSV standard, recorded as the
baseline for the porting campaign: when the remaining port units land, the
same sweep re-runs on the new build and the keep/drop call reads from the
CSV pair.  Host: NVIDIA GB10, driver 580.142, CUDA 13.0
(`cuda_13.0.r13.0/compiler.36424714_0`).  Build: the deployed
`ds4-server`, sha256
`862fd9a6732a2ade032c455e807330e0b29947d51f69e41db43189b253ff2fa9`,
built from the campaign checkout at `321293c` plus local edits (the memgov
union-coverage fix carried in the working tree; server semantics otherwise
equal to the branch).  Artifact:
`DeepSeek-V4.1-Flash-vq8sh14-q4k-mtpnative.gguf`, 113,556,639,424 bytes,
sha256 `9469cfa9ff47c5b9f1e2bbf31b4226ef11b4de9ddb92a0c7641c4a9b623e4654`.
Corpus: `speed-bench/promessi_sposi.txt`, SHA-256
`f53e0d80cb2d4492d24ebd63c7000c397b16ae70f9bf09b3763e5d8323ec209f`, the
same fixture the qwen38 campaign used.  Boot line: `--cuda -m <gguf>
--zchain <grrb-code_fit_n15360-engine sidecar> --engram-dir
~/youngai/deepseek-engram --mem-budget-mb 110000 --host 0.0.0.0 --port
8000`.

Why not `ds4-bench`, the folder's default recipe: the V4.1 family has no
`ds4_session` (the engine's own note, quoted at
`crates/ds4-core/src/serving.rs:659`), so the session-based
`--ctx-start/--ctx-max` sweep cannot drive it, and the CLI's V4.1 route
refuses a text prompt outright ("runs on `--gen-ids`",
`crates/ds4-cli/src/lib.rs:680-692`).  The deployed server route is the
only surface that reports prefill/decode timings for this family
(`crates/ds4-server/src/stream.rs:363-380`), so the harness
(`speed-bench/ds41_spark_sweep.sh`) drives it exactly the way `ds4-bench`
drives the other families.

## Protocol

- One greedy request per frontier, contexts 2048..65536 step 2048 by
  default; the 22 frontiers 2048..45056 ran in this pass (stopped on
  request; 47104..65536 remain open).  128 generated tokens per request,
  temperature 0, `reasoning_effort: none`.
- Rates are the server's own timings, never a wall-clock computation:
  `prefill_tps` from the `[v41] prefill` line, `gen_tps` from the response
  timings.  The graphics clock is sampled per request (median 2405 MHz
  over 521 samples, range 2346..2457).
- Every row is a full prefill: a single-turn frontier render never
  extends the previous prompt, so the KV checkpoint never matches --
  `prefill_cached_tokens` is 0 and `prefill_tokens` reads the full
  context.  (The turn-2 suffix reuse path is measured separately in
  `docs/ds41-perf-2026-10-11.md`, "Port KV prefix reuse, gated".)
- `gen_tokens` is `usage.completion_tokens`; the port's timings block
  carries no `decode_tokens`.
- Attach mode: the sweep measured the running server and never stopped,
  started, or reconfigured it.  The window is recorded in the meta
  (`speed-bench/gb10_spark_ds41_862fd9a6.csv.meta.txt`); external traffic
  during it would show as an outlier, and the two decode dips below are
  attributed against that test.

## Measurements

| context | prefill tokens | prefill t/s | gen | gen t/s | ttft s | tok/step | wall s | clock MHz |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 2048 | 2149 | 852.6 | 128 | 43.7 | 2.523 | 2.91 | 5.447 | 2447 |
| 4096 | 4268 | 1038.3 | 128 | 41.5 | 4.116 | 2.78 | 7.197 | 2450 |
| 6144 | 6398 | 1105.1 | 128 | 41.3 | 5.797 | 2.78 | 8.876 | 2444 |
| 8192 | 8449 | 1146.9 | 128 | 42.6 | 7.377 | 2.78 | 10.372 | 2437 |
| 10240 | 10487 | 1171.7 | 128 | 42.4 | 8.962 | 2.84 | 11.964 | 2434 |
| 12288 | 12541 | 1184.4 | 128 | 41.4 | 10.602 | 2.72 | 13.682 | 2431 |
| 14336 | 14516 | 1197.8 | 128 | 41.6 | 12.135 | 2.78 | 15.203 | 2424 |
| 16384 | 16546 | 1201.0 | 128 | 41.6 | 13.794 | 2.78 | 16.855 | 2418 |
| 18432 | 18537 | 1201.9 | 128 | 41.5 | 15.442 | 2.84 | 18.519 | 2418 |
| 20480 | 20573 | 1198.8 | 128 | 41.6 | 17.182 | 2.78 | 20.242 | 2411 |
| 22528 | 22609 | 1193.8 | 128 | 41.5 | 18.960 | 2.78 | 22.033 | 2411 |
| 24576 | 24651 | 1190.1 | 128 | 41.1 | 20.737 | 2.78 | 23.834 | 2411 |
| 26624 | 26665 | 1184.5 | 128 | 40.5 | 22.536 | 2.67 | 25.679 | 2411 |
| 28672 | 28691 | 1181.0 | 128 | 29.9 | 24.322 | 1.91 | 28.581 | 2408 |
| 30720 | 30712 | 1097.1 | 128 | 40.4 | 28.024 | 2.84 | 31.175 | 2405 |
| 32768 | 32749 | 1095.8 | 128 | 40.0 | 29.919 | 2.78 | 33.104 | 2405 |
| 34816 | 34846 | 1158.6 | 128 | 28.7 | 30.111 | 2.33 | 34.548 | 2405 |
| 36864 | 36948 | 1152.3 | 128 | 41.0 | 32.100 | 2.78 | 35.208 | 2398 |
| 38912 | 38975 | 1144.0 | 128 | 40.0 | 34.105 | 2.78 | 37.286 | 2398 |
| 40960 | 41121 | 1135.3 | 128 | 40.2 | 36.258 | 2.72 | 39.432 | 2398 |
| 43008 | 43208 | 1127.2 | 128 | 40.4 | 38.374 | 2.78 | 41.537 | 2398 |
| 45056 | 45315 | 1117.9 | 128 | 40.8 | 40.577 | 2.84 | 43.706 | 2398 |

## Observations

- Prefill ramps from 852.6 t/s at 2149 tokens -- the process's first
  measured request, cold -- to a 1201.9 t/s peak at 18537, holds
  ~1170-1200 through 26624, dips to ~1096 at 30720-32768 (attribution
  open; it recovers to 1158.6 at 34816), and ends at 1117.9 t/s at 45315.
  The 8192 frontier reads 1146.9 t/s.
- Decode sits at 40.0-43.7 t/s (median ~41.5) with two isolated dips:
  29.9 t/s at 28672 and 28.7 at 34816.  Both track MTP draft-acceptance
  variance -- `tok_per_step` 1.91 and 2.33 against the 2.78 typical --
  and prefill/ttft are in band at both rows, so they are not interleave
  contamination; the dips stay in their rows.  28672 is pure acceptance
  (implied step rate 15.6/s, in band); 34816 also shows a slower step
  rate (12.3/s against ~14.4-15.3).
- Speculation is live throughout: `tok_per_step` 2.67-2.91, i.e. ~1.8
  accepted draft tokens per round.
- Reference points, not a paired A/B: at 8192 the engine oracle measured
  1042.0 t/s prefill (pin 7fb26cc rebuild, 2026-10-12); this build reads
  1146.9 t/s at that frontier.  No-prof decode campaign cells:
  port 42.2-42.6 t/s against engine 43.9-44.4 (`docs/ds41-perf-2026-10-11.md`);
  the sweep's decode spread is wider because it carries per-frontier
  acceptance variance.

## Re-running and the comparison contract

- Boot mode (fresh process; the box must be free):
  `speed-bench/ds41_spark_sweep.sh <server> <gguf> <engram_dir> [sidecar] [port]`.
- Attach mode (measures a listening server; never kills anything):
  `DS41_SWEEP_URL=http://127.0.0.1:8000 speed-bench/ds41_spark_sweep.sh`.
- The keep/drop call for porting work reads from CSV against CSV:
  re-run this sweep on the new build and diff against
  `speed-bench/gb10_spark_ds41_862fd9a6.csv`.

## The r6 comparison (2026-10-12)

The first porting unit through this contract is the side-stream forks
(commit `cb970ef`, build `1e7d8f24`): the same 22 frontiers, boot mode,
same corpus, same 128 greedy tokens.  Against the baseline: prefill at or
above on every frontier (+1.4% at 2048, +0.3% median), decode median
41.20 -> 41.70 t/s, the dips unchanged in character (acceptance
variance).  The mechanism's real effect sits in the verify phase of the
decode round, which this sweep does not isolate: 57.3 -> 56.3 ms medians
at the 2048 cell (alternating A/B against `5666002e`).  Full record:
`docs/ds41-perf-2026-10-11.md`, section "The side-stream forks ported
(r6)".  Keep/drop call: keep -- no regression on any frontier, and the
port's decode graph is now structurally the engine's.

## Provenance

- `speed-bench/gb10_spark_ds41_862fd9a6.csv` (the 22 rows above),
  `.csv.meta.txt` (window, cmdline, clocks, notes), `_ts.svg`
  (`plot_speed.py` rendering).  Commits `af63e9d` (harness + record) and
  `33b3468` (the dips note) on `feat/ds41-optimize-2`.
- Window: started 2026-10-12T14:26:32+02:00, stopped 14:35:20+02:00;
  524 s of requests.  Attach mode on the live server.
- The r6 build's sweep: `speed-bench/gb10_spark_ds41_1e7d8f24.csv`
  (+ `.csv.meta.txt`, `_ts.svg`), boot mode on the same server shape,
  window finished 2026-10-12T16:01:44+02:00, boot_s 76.2, clock median
  2405 MHz.
