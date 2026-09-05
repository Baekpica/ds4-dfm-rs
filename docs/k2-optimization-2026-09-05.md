# K2 MQ87: three prefill and three decode rounds

Base: `43ee428` (the existing HMMA prefill change is already included).
These are six **new** rounds; earlier experiments are not counted again.

## Protocol

GB10 / sm_121a, CUDA 13.3, Rust `ds4-bench`, MQ87 four-shard GGUF.
Raw `speed-bench/promessi_sposi.txt`, first 8192 tokens, 64 greedy decode
tokens, allocated context 8257. Fixture SHA256:
`f53e0d80cb2d4492d24ebd63c7000c397b16ae70f9bf09b3763e5d8323ec209f`.
Default memgov, prefill chunk 512, no MTP or prefix reuse. Load/repack time
is excluded from throughput. Every full-model cell uses a fresh process.

```sh
mkdir -p scratch/k2-check/logits
./ds4-bench --cuda -m ../models/K2-Horizon-375B-A23B-Mixed-Quant-GGUF/K2-Horizon-375B-A23B-MQ87-00001-of-00004.gguf \
  --prompt-file speed-bench/promessi_sposi.txt \
  --ctx-start 8192 --ctx-max 8192 --gen-tokens 64 \
  --dump-frontier-logits-dir scratch/k2-check/logits \
  --csv scratch/k2-check/result.csv
```

Baseline A/B: **207.02 / 207.07 prefill**, **5.54 / 5.54 decode tok/s**.
Diagnostic-only commit `76d057e` writes greedy-ID files outside the timed
region; its repeat measured 206.39 / 5.54. All three frontier dumps were
bit-identical. Promotion gate: finite full-vocabulary logits, relative
RMS <= 1e-4 and the same 64 greedy IDs; microbench parity alone is not
sufficient. Retained changes match all 250624 frontier logits exactly.

## Six rounds

Throughput pairs are prefill/decode tok/s. P3 includes retained P2;
rejected P1 and D1 were measured separately against the baseline.

| Round | Hypothesis | Measured result | Verdict |
|---|---|---|---|
| Prefill 1 | Batch IQ1_M MMVQ launches (26.18% of prefill GPU time). | 218.04 prefill / 5.53 decode; frontier relative RMS 0.08057 and first token changed despite isolated parity passing. | Rejected; production patch reverted. |
| Prefill 2 | Reuse compact worklists for raw IQ2_XXS down MMQ. | 233.95 / 5.54; all logits and 64 IDs exact. NT4096 real-weight kernel 12.901 -> 7.009 ms. | Retained: `3052a06`. |
| Prefill 3 | Reuse compact worklists for raw IQ1_S gate/up MMQ (30.69% of prefill GPU time). | 288.85 / 5.53; all logits and 64 IDs exact. NT512 real-weight kernel 9.791 -> 5.035 ms. | Retained: `69f01ef`. |
| Decode 1 | Reuse grouped split-K full attention (64.88% of decode GPU time). | 206.57 / 9.88; frontier exact but token 18 changed. Synthetic 8K attention 1.888 -> 0.585 ms, 32K 7.580 -> 2.278 ms. | Rejected; no release promotion with unexplained token drift. |
| Decode 2 | Group 2/4/8 independent row warps in aligned Q8 CTAs (21.93% of decode GPU time). | All variants 1.8-3.4% slower across attention, shared FFN, ragged and head shapes; bit parity passed. | Rejected at prototype gate; no full-model candidate. |
| Decode 3 | Cover K=1792 shared-down with the aligned Q8 path. | With GPU input quantization: 0.0508 -> 0.0477 ms in two repeats; double-reference parity passed. Projected whole-decode saving about 0.1%, for 0.632 GiB extra artifacts. | Rejected at cost gate; no full-model speedup claimed. |

Parameter variants and repeat measurements belong to the same hypothesis,
not extra optimization rounds. Rejected paths and their switches are not
left in production.

## Verification and memory

Real-block MoE checks cover narrow/wide widths, skewed routing, invalid
slots and a ragged 129-row tile. The original MMQ parity suite passes.
`ds4`, `ds4-server` and `ds4-bench` build successfully; all 16 Rust bench
tests and `cargo fmt --all --check` pass. P3 used NVCC's
`--split-compile=8` with the existing release optimization/architecture flags.
A proposed extra CPU-reference case also failed on the legacy schedule
at exactly the same element; it was replaced with same-quantization
schedule parity, without weakening existing tolerances.

One model at a time in detached tmux; compiler/GPU work is serialized
during timed measurements. An 8 GiB MemAvailable guard monitors each
full-model run. Retained P2: 95.78 GiB device model/artifacts, including
9.00 GiB additive Q8 artifacts; KV 1.92 GiB plus a 1.91 GiB benchmark
snapshot, workspace 0.37 GiB. Minimum sampled available memory was
15.85 GiB for P2 and 15.62 GiB for P3.
No memory-governor override or extra resident bank was used.
This pass verifies the fixed 8K optimization workload, not a new 32K
HTTP/serving certification. Existing context and bank limits are unchanged.

Host-local, ignored evidence:
`scratch/k2-six-rounds-20260905.lsMont/`. Full-model cells contain source,
binary and fixture IDs, CSV, frontier logits, memory samples, GPU snapshots
and exit status. From `baseline-trace` onward, all 64 greedy IDs are also
recorded. Prototype sources,
rejected patches, build logs and `compare.py` are preserved there.

## Final repeat

Same committed binary (`69f01ef`): **288.85 / 287.78 prefill tok/s**,
mean 288.315 versus baseline mean 207.045 (**+39.3%**).
Decode: 5.53 / 5.52 versus the initial 5.54; no decode speedup is claimed.
Both frontier JSON files and both 64-ID files also pass byte-for-byte
`cmp` against the baseline, not just the RMS gate.
The repeat's minimum available memory was 15.86 GiB.

Same-binary rollback control (`DS4_MMQ_IQ1S_WORKLIST=0` and
`DS4_MMQ_IQ2XXS_WORKLIST=0`): **201.89 prefill / 5.51 decode tok/s**,
again byte-identical logits and IDs. Disabling the changes restores the
slower prefill path, but does not recover the initial higher decode rate;
this check shows no decode regression attributable to the new dispatch.
The initial-vs-final run/build variation is not a claimed decode gain.

All model/test PIDs exited, GPU compute-process list is empty, and
`clear_cache` completed afterward. Final available host memory: **118 GiB**.
Only this campaign's monitor tmux session was closed; existing user sessions
and the two pre-existing untracked artifacts were preserved.
