# Step 3.7 Flash: BASE artifact coverage

This experiment targets local MQ83 BASE weights with a raw MTP-only VMM
owner. Previously, any IPC manifest disabled local aligned artifacts,
including a manifest that supplied only MTP. BASE then used raw-layout
projections. The candidate defers artifact production only when BASE itself
is imported. It retains `DS4_CUDA_BUILD_ARTIFACTS=0` and
`DS4_CUDA_NO_DERIVED_WEIGHTS` as raw-path controls.

Replacement completeness is attached to the local BASE mapping. Imported
MTP ranges cannot inherit that status. Mixed local/imported artifact counts
have the `mixed` source label in CUDA logs and both server renderers.

## Fixed workload

- One GB10, the nine-shard MQ83 artifact and official Q8 MTP sidecar.
- `speed-bench/promessi_sposi.txt`, SHA-256
  `f53e0d80cb2d4492d24ebd63c7000c397b16ae70f9bf09b3763e5d8323ec209f`.
- 2048 prompt tokens, 64 greedy generated tokens, allocated context 2120,
  Step prefill chunk 512 and `--mtp-draft 3`.
- Fresh processes, a separate warmup before every measured run, three
  unprofiled samples, the same resident raw MTP owner, no serialized prefix KV.
- `ds4-perf scout --collector nsys --proof --repeats 3
  --cache-policy warmup-then-fresh`; all nine shards, MTP, prompt and consumed
  owner manifest are hashed in `scratch/step37/perf-r1/workload.json`.
  Executable identities and the candidate patch are in `builds.json`.
- Baseline binary: `01b27a0`; candidate: `766778a` plus the recorded native
  artifact patch. Scout's working-tree SHA is not the baseline binary's SHA.
  Both use CUDA 13.3, `sm_121a`, driver 610.43.02 and owner PID 382693.

Initial profiling isolates 3.766 seconds of Prefill and 3.491 seconds of Decode.
IQ2 MMQ takes 1.242 seconds of Prefill, Q4 routed MMQ 0.704 seconds, Q8 MMQ
0.555 seconds and attention 0.741 seconds. Decode Q8 vector projections take
1.695 seconds across widths 1/3/4; Q4 routed MMQ takes 0.480 seconds, IQ2 MMQ
0.451 seconds and attention 0.357 seconds. These phase times include profiler
overhead and are not the unprofiled speed baseline.

The candidate opens direct aligned IQ2 gate/up pairs and eligible BASE Q8
vector/D2R projections. Q4 down and the raw MTP projections retain their
existing paths. The acceptance and KV commit algorithms are unchanged, while
different arithmetic kernels can change logits, proposals and acceptance counts.
Output and state comparisons therefore remain necessary.

## Correctness and numerical acceptance

The 10-case isolated producer test covers local/empty manifests, MTP-only,
BASE/both/default/invalid scopes and both kill switches. Every projected Q8
output matches an exact CPU fixture. The dummy manifest tests producer
selection only; local raw storage is registered before projection. C/Rust
metrics parity includes mixed-source labels. The model-family CUDA and MMQ
operator suites pass. Actual Rust benchmark/server startup builds 384 BASE
artifacts (30.70 GiB: 24.65 GiB replacement IQ2, 6.04 GiB additive Q8) and
imports the raw 3.45 GiB MTP sidecar.

With artifacts enabled, the 1086-token image/text MTP state gate passes all
trial vocabularies and all 45 layers' committed KV byte-for-byte against the
width-matched independent graph. All 32 accepted tokens match its width-one
control. Changed pixels force refill; injected encoder failure invalidates
both frontiers. Quoted and committed session allocation agree exactly at
511,132,288 bytes (context 1122, chunk 64). This structural gate is distinct
from the chunk-512 performance workload.

Each build repeats its own full-vocabulary frontier and 64-token stream
exactly across all three measured processes. Between builds, frontier
relative RMS is **4.38%**, maximum absolute difference 0.958314; all 128896
values are finite. The first token difference is at index 2, and 60/64 token
positions differ. These are free-form Italian continuations. Accumulation and
activation-quantization paths differ, so this is not a strict token/logit
parity pass. Under the owner's Mixed Quant policy, fixed numerical thresholds
alone do not reject a path; structural correctness and material answer quality
are assessed separately. This bounded suite does not prove broad quality
equivalence or MQ83-versus-BF16 fidelity.

The candidate Rust server passes 36 arithmetic, follow-up and tool requests
with reasoning disabled/high across Chat Completions, Messages and Responses.
Both fresh old/new servers pass the same nine image checks: unchanged
256/1024/1920-pixel screenshots, changed counts, invoice total/tax, an Earth
photograph, follow-ups and four-image streaming in the correct order.

## Measured throughput

| Metric | Raw baseline samples | Aligned candidate samples | Median change |
|---|---|---|---:|
| Prefill tok/s | 545.37 / 544.91 / 546.55 | 702.09 / 702.87 / 703.01 | +28.9% |
| Decode tok/s, MTP draft 3 | 18.70 / 18.74 / 18.69 | 21.58 / 21.59 / 21.69 | +15.5% |
| First decode call, seconds | 0.1399 / 0.1354 / 0.1396 | 0.2378 / 0.2394 / 0.2285 | slower |

These are manually reviewed measurements. `ds4-perf compare` retains
**Incomparable**: the optional CUDA identity helper was absent, so scout's
structured `gpu` field is null. Both scouts retain matching GB10/driver/bus
facts and the same sole owner in their original process snapshots. No scout
metadata was rewritten to obtain a passing verdict. Numerical comparison is
recorded independently in `manual-comparison.json`.

The separate Nsight runs measure Prefill wall time 3.768 → 2.879 seconds;
GPU busy time is 3.717 → 2.829 seconds. Raw MMQ falls from 1.799 to 0.380
seconds while aligned IQ2 gate/up uses 0.458 seconds and Q8 D2R 0.148 seconds.
Attention remains about 0.743 seconds and Q4 worklist MMQ about 0.67–0.71
seconds. This establishes increased fast-path coverage, rather than a change
to attention or the mixed weight recipe.

Profiled Decode wall time is 3.540 → 3.083 seconds, but main target forwards
also fall from 32 to 28 for 64 output tokens (router calls / 42 MoE layers).
The reported MTP Decode gain therefore includes changed draft acceptance and
generation trajectory. Aligned Q8 NC/vector kernels account for 1.031 seconds
of candidate Decode. Do not attribute the full 15.5% to projection kernels.

Startup is separate: artifact construction takes 12.6–40.5 seconds in these
runs, followed by raw-cache preparation. Candidate boot prewarm itself takes
about 1.9 seconds; the baseline spends much more of startup there. Work moved
earlier, and these runs do not establish a readiness-time improvement. The
candidate's first speculative call also has a larger launch/capture gap.

A separate pair keeps the same owner and loaded sidecar but sets
`DS4_MTP_SPEC_DISABLE=1`. With 2048 input and 64 ordinary decode steps,
Prefill is 545.67 → 711.13 tok/s and Decode 17.99 → 19.54 tok/s (+8.6%).
Steady Decode is 18.00 → 20.20 tok/s; first-call latency is 0.0584 → 0.1561
seconds. Each is one fresh process with normal boot prewarm, rather than the
three-repeat scout protocol. This removes speculative acceptance as a cause
of the improvement; frontier relative RMS remains 4.38% and 61/64 generated
token positions differ between builds.

The actual candidate engine also completes the same 2048+64 MTP workload
with `DS4_CUDA_BUILD_ARTIFACTS=0`: all 128896 frontier logits and all 64 tokens
exactly match the old raw baseline. Its 544.74/18.64 tok/s Prefill/Decode are
a single fallback check, not another three-sample baseline.

An earlier exploratory old-binary run omitted `--mtp` while retaining the
MTP-only manifest and aborted in raw dispatch during boot. Its log remains in
`nomtp-baseline/`; it supplies no performance result. The successful ordinary
controls retain the sidecar and use the supported speculation kill switch.

## Image request observations

One fresh server per build, default boot prewarm, context 4096, output cap 128,
vision and MTP draft 3. Each case runs once in the fixed fixture order.
Responses can differ in wording/length; these are whole-request observations,
not isolated encoder throughput or repeated image-performance estimates.

| Request | Raw seconds | Aligned seconds |
|---|---:|---:|
| Small screenshot | 3.640 | 3.486 |
| Screen | 5.900 | 5.477 |
| Changed screen | 5.783 | 5.485 |
| Large screenshot | 8.786 | 8.232 |
| Invoice | 7.278 | 6.808 |
| Invoice follow-up | 6.806 | 6.355 |
| Photograph | 2.546 | 2.396 |
| Photo follow-up | 2.580 | 2.418 |
| Four-image Messages SSE | 20.047 | 17.871 |

## Longer-context control

At 16384 input tokens plus 1024 generated tokens (allocated context 17416,
chunk 512, MTP draft 3), the default path and a separate eager process with
`DS4_CUDA_MOE_GRAPHS=0 DS4_CUDA_LAYER_GRAPHS=0` have byte-identical complete
frontier logits and the same 1024-token stream. Both exit normally under the
100 GiB guard. Step's attention executes eagerly; this also checks the shared
captured projection/MoE paths used by its forward graph.

Default Prefill/Decode is 669.97/13.73 tok/s; eager is 671.27/13.90 tok/s.
These are one-run functional controls at a different context/output length,
not the 2K optimization comparison or evidence that graph capture is faster.
They qualify this bounded text workload, not the source's 262144-token limit,
long image conversations or multiple sequence banks.

## Evidence

Ignored local evidence is under `scratch/step37/`: `artifact-scope-green2.log`,
`artifact-metrics-green.log`, `artifact-operators.log`, `artifact-spec.log`
and the matching guard logs. `perf-r1/` contains pinned binaries, the hashed
workload, complete `baseline/` and `candidate/` scouts and traces, the original
comparison verdict, and `image-{baseline,candidate}/` requests/responses.
`ordinary-{baseline,candidate}/`, `fallback-candidate/` and
`long-{candidate,eager}/` retain each control's exact command, binary digest,
GPU UUID/driver, environment, CSV, complete frontier logits, tokens and guard
log. `controls2.log` records both exact comparisons. The required host checks
are in `scratch/step37/final-host-checks.log` and its per-command directory.
