# Solar Open2 long-context prefill, September 12

Four candidates were evaluated on Solar Open2 250B MXQ-v1. None was
retained: a host freeze, excessive numerical drift, a component regression,
and an inconclusive timing gate prevented adoption. The native inference
path remains the one at `847ab69`; this campaign claims no new speedup.
Commit `4341ffd` records Solar rollback controls and the graph prefill width
in `ds4-perf` comparisons. The campaign ends at round 4.

## Protocol and baseline

- One DGX Spark GB10, CUDA 13.3, `sm_121a`, one resident VMM/base weight
  owner and sequential guarded workers. A reboot separates round 1 from
  the later rounds; percentages compare only matched arms within a round.
- [Solar MXQ-v1](https://huggingface.co/Baekpica/Solar-Open2-250B-Mixed-Quant-GGUF):
  all 11 shards, 95,533,532,160 bytes. Each scout records shard hashes,
  executable hash, GPU identity and effective performance controls.
- `speed-bench/promessi_sposi.txt`, SHA256
  `f53e0d80cb2d4492d24ebd63c7000c397b16ae70f9bf09b3763e5d8323ec209f`.
  Independent 8,192- and 65,536-token prompts; 64 greedy output tokens,
  MTP off, K-FP8/V-FP4 KV, both prefill chunk controls fixed at 4,096.
- Three fresh unprofiled processes per arm, each preceded by a separate
  warmup process. A separate Nsight process measures GPU phases. Profiled
  timings and incomplete runs never enter the throughput table.
- All 196,608 frontier logits and 64 generated IDs are saved per sample.
  Exact rounds use zero logit tolerances. Round 2 uses the predeclared
  context-scaled contract below. Timing uses the unchanged 3% sample-extrema
  slowdown ceiling; this is observed variation, not a confidence interval.

The retained path's completed controls measured:

| Control | Prompt tokens | Prefill tok/s | Decode tok/s |
|---|---:|---:|---:|
| Round 4 OFF, median of 3 | 8,192 | 1,118.83 | 18.01 |
| Round 1 OFF, median of 3 | 65,536 | 779.45 | 13.56 |

These are baseline measurements, not an optimization comparison. The
[September 7 cold interleaved results](solar-open2-optimization-2026-09-07.md)
and older HTTP card measurements retain their different protocols.

The initial 64K profile attributed 28.875 seconds to attention out of
84.219 seconds of GPU prefill (34.29%). The round 1 OFF control reproduced
28.943 seconds attention and 84.883 seconds prefill wall time. Its 8K
attention time was 0.333 seconds. This motivated the attention experiments;
none changes the model's attention complexity or KV capacity.

## Four rounds

Component timings below measure a 4,096-query attention block ending at the
listed depth. They are not full-model prefill times.

| Round | Independent candidate | 64K component ms, OFF → ON | Full-model 8K prefill tok/s, OFF → ON | Disposition |
|---|---|---:|---:|---|
| 1 | Register lookahead for packed KV | 296.165 → 154.343 | 1,114.79 → 1,129.86 (+1.35%) | Removed after host freeze during 64K ON warmup |
| 2 | 64-key softmax steps on direct loads | 294.671 → 202.162 | 1,119.77 → 1,132.82 (+1.17%) | `Incorrect`: frontier RMS exceeds declared bound |
| 3 | 32-query GQA-pair blocks | 296.073 → 444.025 | Not run | Removed after component regression |
| 4 | Direct current-row scale loads | 295.062 → 198.085 | 1,118.83 → 1,132.06 (+1.18%) | `Inconclusive`: first-decode timing ceiling unresolved |

### Round 1: packed KV lookahead

The candidate overlapped the next tile's packed K/V and scale loads with
the current tile's HMMA work. Conversion and 16-key online softmax order
were unchanged. Component outputs and the complete 8K model proofs matched
byte for byte. The 8K comparison was `Improved`; decode was 17.81 tok/s
in both arms. Registers rose from 131 to 168 without local-memory spills.

The 64K OFF control completed. ON did not finish its first warmup before
the user reported loss of SSH and IDE connectivity and forced a reboot.
There is no 64K ON sample, proof or Nsight result. The final persisted guard
sample showed 14.40 GiB available and zero memory PSI; the journal contains
no matching OOM or new Xid report. These incomplete logs cannot establish
whether the cause was OOM, a driver lockup, or something else.

The candidate was archived and removed. Prefetch is optional and is not
needed to establish a comparable baseline. No later round includes it;
no fix for the host freeze is claimed.

### Round 2: fewer softmax normalization steps

Grouping 64 keys instead of 16 changes online normalization and half
probability rounding. Before measurement, the component relative RMS limit
was 0.002, with independent sampled FP64 checks over half-rounded Q/K/V.
The full-model bound was `0.01 * log2(max(ctx, 2)) / 10`:
0.01 at 1,024 tokens, 0.013 at 8K and 0.016 at 64K.
Frontier argmax must match; later greedy differences are reported.

Every 8K candidate frontier measured relative RMS 0.04022757, exceeding
0.013. Argmax 4475 stayed fixed, but 57 of 64 generated IDs differed in
each sample. The comparison checked 1,179,648 values and returned
`Incorrect`. Decode medians were 18.03 → 18.04 tok/s. The bound was not
raised after seeing the result. No 64K or state gate was run; restoring
16-key steps reproduced the saved baseline proofs exactly.

### Round 3: smaller query blocks

The candidate reduced each GQA-pair block from 64 to 32 queries on the
original 16-key direct-load path. Outputs remained byte-exact, including
31/32/33-query boundaries and deep unaligned partitions. Both paths used
131 registers without spills. However, the 8K component grew from 20.093
to 26.608 ms, and the 64K component grew from 296.073 to 444.025 ms.
A 65-query deep-tail gain did not offset the production chunk regression.
The candidate was removed before full-model testing.

### Round 4: direct current-tile scales

Each fill warp expands one KV row. Uniform loads of that row's scales
removed shared scale staging and one block barrier without future-tile
lookahead. Query geometry, conversion, 16-key softmax, and the barriers
protecting shared K/V remained unchanged.

The 8K component fell from 20.015 to 14.886 ms. All output bytes, sampled
FP64 probes and unaligned partitions passed. Registers rose 131 → 147;
shared memory fell 36,368 → 35,856 bytes, without local/stack spills.
Solar KV tests and shared BF16 kernel checks passed; the model-dependent
EXAONE tier was not run. `ds4-perf` tests and formatting also passed.

The full 8K comparison checked 1,179,648 values with zero differences,
zero argmax mismatches and zero generated-ID mismatches. Decode medians
were 18.01 → 18.02 tok/s. First-decode time was 0.1474 → 0.1524 seconds;
its envelope extended to +5.67%, so the original verdict is `Inconclusive`.
This benchmark field times the first decode evaluation after prefill,
proof output and snapshot creation; it is not HTTP time to first token.

A supplementary order was declared before collecting more data:
OFF/ON/ON/OFF, then ON/OFF/OFF/ON, four fresh samples per arm, each with a
fresh warmup. It used the same fixture, binary, owner and effective
controls, with no profiler, and retained the same 3% extrema ceiling.
All proof bytes matched. CSV counts, finite positive timings, child
controls, executable identity and successful exits were independently
audited. The first-decode envelope was -2.03% to +4.51%, still crossing
the ceiling. This separate evidence does not replace the original verdict.

Round 4 was archived and removed. No candidate 64K, state or serving gate
was promoted after this failed acceptance condition. The final validation
below uses the restored baseline.

## Guarded serving validation

These checks rebuilt the restored native path at `4341ffd`; they did not
use any of the four candidates.

| Check | Result |
|---|---|
| Direct forward integration, 257 corpus tokens | Failed with CUDA illegal memory access; exit 134 |
| Direct forward integration, Makefile's four-token fixture | Failed with CUDA illegal memory access; exit 134 |
| Public session lifecycle | Failed: cold/warm snapshot decode max absolute logit difference 0.331638336 exceeds the existing 0.25 bound; argmax 4360 matches |
| 131,072-context HTTP worker | Guard terminated initialization at 11.52 GiB available; exit 75, before listening |

The 257-token forward failure occurred with approximately 18.7 GiB
available and zero memory PSI. Inspection found that this direct harness
omits the VMM import performed by public engine initialization, while
manifest presence already disables host registration. These forward runs
therefore do not validate the public owner-import path; the precise
illegal-access source was not localized.

The session test did import the owner correctly. Its four-token fixture
used context 128 and prefill cap 3. Snapshot bytes and restored logits
matched exactly before the subsequent decode exceeded the drift bound.
This does not establish serialization corruption, but it prevents a full
lifecycle pass: later replay, batching and continuation checks did not run.
No test threshold was changed.

The attempted HTTP configuration was one bank, 2,048-token prefill chunks,
MTP off, and native/HTTP memory floors of 13 GiB. Admission capped the
worker's requested cgroup max/high to 8.20/7.38 GiB. The watchdog observed
11.52 GiB available, below its 12 GiB termination floor, and stopped the
worker. Memory recovered and the host remained responsive. Polling can
overshoot the floor; it is not a reserved allocation.

The prepared prompt contained 124,006 raw tokens, but no HTTP request was
processed. There is no new 128K prefill, retrieval, cached-continuation or
tool-result result. This campaign does not qualify stable long-context
agent serving. The task's worker and resident owner were stopped after
validation; no test endpoint is left running.

The worker guard used 12 GiB admission and termination floors, with a
requested 16/14 GiB cgroup max/high capped to available headroom. The owner
had its own guard and lower emergency floor. Successful 8K scouts retained
at least 14.00 GiB available; the completed 64K control reached 12.08 GiB.
A completed guarded run validates only that workload. It does not establish
that the unresolved host freeze is fixed or guarantee recovery from a
kernel/driver lockup.

## Evidence and reproduction

- [Measured samples](solar-open2-2026-09-12-rounds.csv): all 21 samples from
  seven completed scouts, plus the eight separately labelled supplementary
  CLI samples, numbered in execution order. Round 2's numerically rejected
  samples remain visible.
- [Evidence summary](solar-open2-2026-09-12-evidence.json): binary/proof
  hashes, comparison verdicts, phase timings and guard minima.
- Raw scouts, commands, proofs, guard logs and rejected candidate sources
  are retained locally under `scratch/solar-longctx`; rejected controls
  are not release runtime options.

Use [the ds4-perf workflow](ds4-perf.md) with the exact eleven-shard artifact
and prompt above. Create a fresh VMM/base owner and workload identity;
record its live manifest, rather than reusing a historical PID or path.
Set `DS4_CONT_PREFILL_CHUNK=4096`, `DS4_METAL_PREFILL_CHUNK=4096`,
`DS4_SOLAR_KV_FORMAT=kfp8-vfp4`, `DS4_SOLAR_MOE_RESIDUAL=1` and
`DS4_MEMGOV=observe`. Run separate 8K/64K scouts with `--proof --repeats 3
--cache-policy warmup-then-fresh --collector nsys`, and compare only
complete matched artifacts. The branch's inference source equals the
entry baseline; only performance control recording and this evidence are
published from the four-round campaign.
