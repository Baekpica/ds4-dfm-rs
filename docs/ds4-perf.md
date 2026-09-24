# ds4-perf

`inspect`, `scout`, `compare`, and `optimize` form the Rust performance workflow.
`serving` collects ordered HTTP conversation workloads against a local server.
The inference ABI remains `ds4-cli → ds4-core → ds4-sys → native CUDA/MMQ/VMM`.
NVIDIA's optional Rust `nvtx` SDK annotates the measured host operations.
CUDA calibration and CUPTI bindings live in the separate, optional
`ds4-perf-gpu` helper; inference packages do not depend on it.

## Build and inspect

```sh
make ds4-perf
make ds4-perf-gpu                  # Linux, CUDA 13.3 driver/NVRTC/CUPTI
make ds4-bench-perf                # after make cuda-spark / native CUDA build
./ds4-perf inspect --calibrate --out scratch/perf/machine
```

The control executable builds without CUDA. The helper uses
[cudarc](https://github.com/chelsea0x3b/cudarc) with driver, NVRTC, and CUPTI
features. This release's direct collector checks the CUDA 13.3 CUPTI activity
ABI (`130301`) before reading records; another ABI fails explicitly. Nsight
collection remains available independently. NVTX remains NVIDIA's official
`nvtx` 2 SDK, default features disabled, `std` only. Its build requires libclang;
an extracted LLVM installation may also need `BINDGEN_EXTRA_CLANG_ARGS` to set
Clang's resource directory.

Inspect records UUID/driver identity, SM and warp counts, block/thread limits,
register/shared-memory capacity, L2, available memory, and tool/process probes.
Missing device properties produce a partial artifact with reasons.
`doctor --bench ./ds4-bench-perf` remains a compatibility command.

Calibration runs bounded CUDA copy, FP32 SIMT FMA, and host-launch experiments.
It preserves warmups/repeats, CUDA event times, workload geometry, source/PTX
hashes, and helper identity. Copy results and FMA results are checked, including
a CPU FMA reference. The measured envelope is not a tensor-core peak or a model
roofline. Run calibration without competing GPU work.

## Scout and fit

```sh
./ds4-perf scout --out scratch/perf/baseline --collector nsys \
  --fit --calibration scratch/perf/machine/calibration.json \
  --proof --repeats 3 --cache-policy warmup-then-fresh \
  --workload workload.json -- \
  ./ds4-bench-perf --cuda -m /models/model.gguf \
  --prompt-file speed-bench/promessi_sposi.txt \
  --ctx-start 2048 --ctx-max 2048 --gen-tokens 8
```

Keep an intended VMM owner resident and the serving worker stopped during an
experiment. Use the [memory guard](host-memory-guard.md) for large-model runs.
Scout does not stop unrelated processes. ds4-bench uses visible CUDA device 0;
set `CUDA_VISIBLE_DEVICES` on the parent process to select another GPU. Omit `--csv FILE`: benchmark CSV belongs
on stdout. The executable is resolved and pinned before child execution.
Unix argv bytes, commands, reviewed controls, outputs, status, process probes,
and profiler artifacts are retained.

`--collector cupti` injects `libds4_perf_gpu.so` into a fresh benchmark and uses
NVIDIA's `libcupti.so` for NVTX collection. `--cupti-library`, `--cupti-sdk`, and
`--gpu-helper` override discovery. The raw JSONL retains owned activity records,
launch geometry, marker timestamps, drops, and errors. Missing footer, dropped
records, and unsupported record ABI cannot become successful phase evidence.
Marker pairing uses timestamps rather than callback completion order.

Only unprofiled fresh-process samples feed throughput comparisons. Default
`--cache-policy inherited` leaves cache state unspecified. The explicit
`warmup-then-fresh` protocol completes a separate warmup before every measured
process and before the trace. It does not claim that all OS pages remain cached
or that process-local KV/PLE caches survive. Warmup outputs remain evidence.

`--proof` requests ds4-bench's full-vocabulary frontier logits and greedy-token
files outside measured NVTX operations. Requested proof or repeated samples
that are missing make the scout incomplete. Unsupported opaque commands may
still be structurally profiled without proof or a workload contract.

Inkling still has no serialized session checkpoint. Step now has `STP3`
disk payloads and opt-in bank checkpoints; see
[step37-serving-2026-09-13.md](step37-serving-2026-09-13.md). Between sweep
frontiers a family without a usable snapshot still replays the prompt prefix
outside both measured ranges. CSV `kvcache_bytes` is zero when no serialized
snapshot is used, not a KV allocation measurement.
`tests/test_inkling_bench.py` compares sweep logits and tokens with
independent cold frontiers; run it under the memory guard with the same owner.

`--fit` joins device properties, calibration, workload metadata, and measured
grid/block/register/shared-memory geometry. Nsight exports use native units
(`csv:noconv`) so rounded memory sizes cannot alter resource bounds. Fit reports
SM residency/occupancy upper bounds, wave lower bounds, and tail fill under those
bounds. Register allocation granularity, shared-memory carveout, clusters, and
barriers are not modeled. Operand dimensions are explicitly unknown: launch
geometry and a kernel name do not establish M/N/K or tensor-core work.
Missing calibration, workload shape, or launch geometry leaves a partial fit
and a failed final scout. Auto repeats fit when its source provides calibration;
phase-only sources can still drive explicit controlled experiments.

`--ncu` performs a separate real capture of the first matching instance of each
phase's top aggregate kernel, limited to one launch in the application process.
Application replay avoids backing up accessible model allocations; strict
matching checks kernel name, grid/block, context and stream across passes. See
[NVIDIA replay semantics](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#application-replay).
It verifies the returned kernel and launch identity, preserves `.ncu-rep`/CSV,
and normalizes supported counters with units. Missing counters remain unknown.
The instance may differ from the kernel's dominant geometry; do not interpret
it as all launches. Replay/cache controls affect counters, and NCU throughput
never enters the speed comparison. Capture failure writes incomplete `ncu.json`
and returns failure. Without `--ncu`, unavailable NCU does not block scout.

## Workload identity and comparison

The workload contract uses paths relative to its JSON file and SHA-256 values:

```json
{
  "protocol": "ds4-bench-v1",
  "name": "qwen-2k-g8",
  "family": "qwen4exp",
  "files": {
    "model": { "path": "/models/model.gguf", "sha256": "<64 hex digits>" },
    "prompt": { "path": "prompt.txt", "sha256": "<64 hex digits>" }
  },
  "shape": { "ctx_tokens": 2048, "gen_tokens": 8 },
  "cache_state": "Warmup before each fresh process; same resident owner"
}
```

Include every GGUF shard, explicit MTP model, PLE manifest and physical payload,
and consumed IPC owner manifest as additional named file entries. Scout checks
the ds4-bench arguments against these files and hashes the inputs before and
after the experiment. A first-shard-only manifest is insufficient. Hashing large
sidecars is outside the measured operation. The protocol supports direct
ds4-bench inference sweeps, not arbitrary shell wrappers or output-head probes.

```sh
./ds4-perf compare --baseline scratch/perf/baseline/scout.json \
  --candidate scratch/perf/candidate/scout.json --out scratch/perf/comparison \
  --regression
```

Comparison requires matching device/driver, command arguments, working directory,
hashed workload, explicit cache protocol, memory policy, and equal complete
sample counts (at least three). Reviewed tuning controls may differ; executable
changes are reported and still require correctness proof. Unknown DS4/CUDA/LD
controls prevent automatic acceptance. GPU process snapshots must match the
intended owners before and after each run; transient contention between these
snapshots is not detected. Historical comparisons read preserved
proofs and hashes; they do not reload the model.
Inkling's `DS4_INKLING_NO_LINEAR=1` is a reviewed diagnostic control for
comparing its ordinary BF16 projection with the prior implementation.
Unset the variable for the optimized path; comparisons reject other values.
`DS4_INKLING_NO_MOE_BATCH=1` similarly restores per-token expert projections;
the optimized path groups prefill assignments while retaining MMVQ reductions.
`DS4_INKLING_NO_Q8_BATCH=1` restores separate rows for the two dense MLP layers;
the optimized path uses existing aligned-Q8 column tiles with identical reductions.
`DS4_INKLING_NO_MOE_TILE=1` restores the four-column expert batch kernel;
the optimized path decodes each weight fragment once for a warp-owned tile.
`DS4_INKLING_NO_LINEAR_TILE=1` restores the token-grouped BF16 projection;
the optimized path shares a shared-memory token slab across a row tile.
`DS4_INKLING_NO_LOGIT_TILE=1` restores the one-warp FP32 router GEMM;
the optimized path covers the aligned output rows with that slab and the
raw warp sum, then the same reduction for the leftover rows.
`DS4_INKLING_NO_LINEAR_PANEL=1` restores the original BF16 tile job order;
above 4096 rows, the candidate reuses inputs within internal 512-token panels
for 4096-input q/k/v/r/o shapes. The configured prefill chunk is unchanged.
`DS4_INKLING_NO_ATTN_GROUP=1` restores per-head prefill attention CTAs;
the optimized path scores the four query heads of a KV head in one CTA
with prefetched keys. Widths below 16 rows keep the per-head kernel.
`DS4_INKLING_NO_Q8_TILE=1` restores the eight-column dense Q8 loop; the
optimized prefill path keeps each aligned weight row in registers while
eight-token groups stream through shared memory.
`DS4_INKLING_NO_SHARED_Q8=1` restores warp-owned shared-Q8 up tiles;
the optimized prefill path reuses each payload across eight assignments
with the original four-partition reduction. This switch affects only shared up.
`DS4_INKLING_NO_SHARED_TILE=1` restores the eight-column shared Q8 up kernel;
from 64 prompt rows, the optimized path holds weights in registers across
all routed columns and stages inputs for a CTA in shared memory. Unaligned
Q8_1 input pointers and insufficient block shared memory keep the prior path.
`DS4_INKLING_NO_SHARED_DOWN_TILE=1` restores warp-owned shared Q8 down tiles;
from 64 prompt tokens (128 assignment rows), weights stay in registers while
input groups stream through shared memory with the original one-warp sum.
`DS4_INKLING_NO_IQ2_SLAB=1` keeps the per-warp lean IQ2 tiles; the optimized
path stages each eight-column tile's activations once per CTA in shared
memory and sweeps its row groups with four warps.
`DS4_INKLING_ATTN_HMMA=1` opts into the tensor-core prefill attention, which
scores 64-query tiles on the bf16 tensor cores and is not byte-identical to
the default grouped kernel (summation order); `compare --regression` reports
it as `Incorrect` because the model turns that into different greedy tokens.
`DS4_MIMO2_NO_PREFILL_HMMA=1` restores MiMo's walking full-attention prefill.
Unset, windowless prefill of 32 or more rows uses the tensor-core tile.
That path changes summation order, so compare it with `--logit-rel-rms`.
This switch does not change SWA or one-row decode.
`DS4_MIMO2_NO_PREFILL_ASYNC=1` restores scalar KV loads inside that
tensor-core prefill. Unset, the loads are asynchronous and byte-identical.
`DS4_MIMO2_NO_SWA_HMMA=1` restores the walking sliding-window prefill.
Unset, window-128 prefill of 32 or more rows uses the tensor-core tile.
Compare that path with `--logit-rel-rms`.
`DS4_MIMO2_SWA_DECODE=0` restores the one-row SWA walk. The default shares
the window across eight query heads with vector KV loads;
`DS4_MIMO2_SWA_VEC=0` selects scalar copies within that shared tile.
`DS4_MIMO2_ROUTER_WARP=0` restores serial expert selection for widths 1–8.
The default warp selection preserves IDs and weights exactly.
`DS4_MIMO2_DFLASH_CPU=1` restores host RMSNorm, RoPE and attention in the
external DFlash drafter. Unset, those operations stay on the GPU. This
changes draft arithmetic; compare verified output and long-window cases.
The shared tile changes FP32 rounding. See the
[fresh performance and numerical evidence](benchmarks/mimo2-2026-09-24/README.md).
`DS4_INKLING_NO_SHARED_PIPE=1` restores the staged-slab resident Q8 tiles;
the optimized path streams float-scale SoA rows through a cp.async column
ring (two rows per warp for up, four rows per four-warp CTA for down).
`DS4_INKLING_NO_ATTN_PAIR=1` scores one key per warp iteration in grouped
prefill attention; the optimized path loads and scores keys i and i+4
together and applies their softmax updates in order.
`DS4_INKLING_NO_Q4_LEAN=1` restores two-row branched Q4_K expert tiles; the
optimized path owns four rows per warp, hoists the activation row and loads
every column (row 0 for padding).
`DS4_INKLING_NO_SHARED_COLUMN=1` sums all eight slab columns in one K loop of
the resident Q8 tiles; the optimized path runs the K loop once per column
with float scales staged once, which removes the register spills.
`DS4_INKLING_NO_ATTN_TRANSPOSE=1` restores the all-lane head reduction in
grouped prefill attention; the optimized path finishes each head's XOR tree
in one 8-lane group and broadcasts its softmax scalars.
`DS4_INKLING_NO_Q3_TILE=1` restores four-column Q3_K expert batches; from
256 prompt tokens the optimized path decodes each fragment once for eight
routed columns with the same lane products and ordered sums.
`DS4_INKLING_NO_IQ2_LEAN=1` restores branched column loops and table-driven
signs in the IQ2 expert tiles; from 256 prompt tokens the optimized path
loads every column (row 0 for padding), hoists the row index and spreads
signs arithmetically at three CTAs per SM.
`DS4_INKLING_NO_SHARED_SOA=1` restores canonical-row slabs and float deltas in
the resident Q8 tiles; the optimized path stages the relayout SoA and keeps
half deltas so two CTAs (up) or four (down) share an SM.
`DS4_INKLING_NO_Q8_ROUTED_TILE=1` restores warp-owned routed Q8 tiles;
from 64 prompt rows, the optimized path keeps weights in registers across
all routed columns with the shared-expert tile schedule.
`DS4_INKLING_NO_IQ2_XS_ALIGNED=1` restores 74-byte IQ2_XS down tiles;
the optimized path reads owner SoA artifacts with the same MMVQ reduction.
`DS4_INKLING_NO_IQ2_ALIGNED=1` restores 66-byte IQ2_XXS expert tiles;
the optimized path reads owner SoA artifacts with the same MMVQ reduction.
`DS4_INKLING_NO_Q4_TILE=1` restores four-column Q4_K expert batches;
from 3072 routed assignments, eight-column tiles reuse payload loads and
unpacked scales with the same MMVQ partial sums and skip the unused relayout.
`DS4_INKLING_PREFILL_CHUNK=N` (1–8192) sets the maximum Inkling prefill chunk width;
shorter prompts remain valid. The default is 2048, capped by context.
`DS4_INKLING_PREFILL_CHUNK=1024` restores the previous cap.
Scout consumers also reparse each referenced unprofiled benchmark CSV and
require its rows to match the serialized samples. Hashing and parsing use the
same bytes; benchmark stdout is limited to 64 MiB per sample on load.

Every token sequence and full-vocabulary frontier is checked, including within-run
repeat consistency. Default logit tolerances are `atol=0.0001`, `rtol=0.0001`;
`--logit-atol` and `--logit-rtol` define an explicit alternative contract.
`--logit-rel-rms X` (off by default) switches to the relaxed contract for
summation-order changes that a chaotic model amplifies: each frontier must
stay within a context-dependent relative RMS bound of the baseline and keep
its argmax, greedy sequences may diverge (`token_mismatches` and
`argmax_mismatches` are still reported), and per-logit statistics remain
informational. The bound is `X` at 1,024 tokens and grows with
`log2(ctx) / 10` (1.1X at 2K, 1.3X at 8K, 1.6X at 64K): reordering noise
grows with the attended length while the model's amplification saturates,
so the allowance follows the length slowly and never linearly.
`correctness.rel_rms` is the largest observed ratio and `rel_rms_scaled`
its largest fraction of the bound. Inkling MQ85GB turns a one-ulp
reordering in one kernel into relative RMS 0.063 at 16 tokens and 0.105 at
515 with a different greedy continuation, so exact rounds keep the default
contract and only reordering-class changes use `--logit-rel-rms`, with the
observed floor recorded in the round's report.
Prefill/decode inverse TPS and first-token seconds are compared separately.
The min/max sample envelope is conservative observed variation, not a statistical
confidence interval. The default slowdown limit is 3%; overlapping evidence is
inconclusive. `--regression` fails for regression, incorrectness, inconclusive
results, or incomparable inputs. This gate does not replace model/state,
capture/eager, long-context, or serving correctness gates.

## Serving workloads

```sh
./ds4-perf serving --url http://127.0.0.1:8002 \
  --workload serving-workload.json --out scratch/perf/conversation \
  --timeout-seconds 1800 --max-output-mib 2048
```

This Linux collector sends the listed streaming Chat requests in order. Run it
against an idle server with the intended model, banks and cache policy. It
requires `curl` and a loopback HTTP origin. It does not start or restart a server,
change a serving option, clear KV, or invent conversation history. A cold case
must encounter cold KV; otherwise its trace check fails. Each append/branch
request must contain the complete intended conversation, including the earlier
assistant output. Use separate cases for short/long prompts, media, and sampler
settings; their timings remain separate.
`--repeats N` replays the whole workload with KV preserved and retains every
sample. A cold-only fixture therefore needs an externally reset server and a
separate output directory for each cold sample; replaying it as warm fails its
reuse expectation. No aggregate averages short/long or text/media cases.

```json
{
  "protocol": "ds4-serving-v1",
  "name": "qwen-cold-append-branch",
  "family": "qwen4exp",
  "cases": [{
    "name": "cold",
    "scenario": "kv_cold_prefill",
    "request": {
      "model": "qwen", "stream": true, "temperature": 0, "max_tokens": 32,
      "messages": [{"role": "user", "content": "Reply with exactly OK"}]
    },
    "expect": {
      "content": "OK", "finish_reason": "stop", "reuse_kind": "cold", "effective_lane": "continuous",
      "speculation_active": false, "fallback_reason": null
    },
    "limits": {
      "ttft_ms": 10000, "total_ms": 20000,
      "min_host_available_bytes": 4294967296
    }
  }]
}
```

The example's output and limits are illustrative; set them from the intended
accuracy and latency contract before collection. `warm_append` describes the
conversation workload; its declared `reuse_kind` can be `"exact"`, `"partial"`,
or `"fork"`, matching the actual admission path. `partial_branch` still requires
`"partial"`/`"fork"`; an append that reports `"cold"` does not pass. `media`
requires image/audio content in the request; `mtp` requires speculative decode
in its observed trace. `family` must equal `/v1/stats`'s `serving.family` exactly.
The full requested/effective/qualified plan is retained from the server, so the
collector does not maintain a separate family capability table.

Optional `--server-pid PID` checks ownership of the listening socket and pins
the process start time, boot ID, executable SHA-256, raw argv, working directory,
reviewed runtime environment and source checkout digest before/after collection.
Unknown runtime environment keys are recorded by name and cannot qualify a
profile. `localhost` resolves to IPv4; use `[::1]` for an IPv6 listener.
Optional workload `inputs` use named `{ "path": "...", "sha256": "..." }`
entries, relative to the workload file. Hashes are checked before and after the
whole run, outside HTTP timing; declare every model shard, template, sidecar and
owner manifest used. The collector records that manifest coverage is supplied
by the operator, and does not identify loaded tensors from a server PID.

`nvidia-smi` records GPU UUID, driver, SM clock and temperature before/after
each repeat and on a concurrent 500 ms polling loop during the workload
(`gpu-window.json`; `--device` selects its physical ordinal). Each sample keeps
raw stdout/stderr. Unavailable telemetry
stays explicit. Workload `expected_clock_range_mhz: {"min":300,"max":2200}`
additionally requires observed clocks within that range. This declared range
does not prove the driver's configured lock. Polling includes metadata-query
overhead and can miss short peaks. Qualification requires chronological query
timestamps, gaps and boundary slack at most 1500 ms, and a recorded workload
window covering the measured requests. Short workloads may have only a sample
near their boundaries. These checks bound missing observations; they do not
prove continuous clock compliance. No clock setting is changed.

`ttft_ms` measures client dispatch to the first nonempty content or reasoning
delta; `first_content_ms` separately measures visible answer content. Role-only
events are excluded. `total_ms` ends at `[DONE]`. These times include curl
startup and use a 1 ms polling observer of unbuffered SSE output. NVIDIA metadata
queries run concurrently; no GPU profiler is attached. These are HTTP
measurements, separate from ds4-bench's post-prefill `first_token_sec`. They do
not measure exact server queue residence or token-level decode throughput.

Each case retains its request, raw SSE, timestamped data events, headers,
stderr, result, and before/after `/v1/stats`. `serving.json` hashes these inputs.
The resolved plan must stay identical, route counters must advance by exactly
one, and the queue/other clients must be idle at the boundaries. The collector
checks exact visible output and finish reason, reuse/lane/speculation/fallback expectations,
latency limits, and host `MemAvailable` at both boundaries and through a 10 ms
sampler (`memory.json`). Qualification rejects sampling gaps or uncovered
request boundaries over 100 ms, allowing scheduling jitter. Sampling can miss
shorter memory peaks. Invalid or interrupted streams produce
incomplete evidence; collected contract failures produce `passed: false` and a
nonzero exit. An existing output directory is never overwritten.

For decode-vs-long-prefill, provide `overlaps` alongside `cases` (either array
may be empty). Each overlap has a unique `name`, `decode` and `prefill` request
objects, `min_prefill_tokens`, `min_decode_events_during_prefill`, and
`max_decode_gap_ms`. Both request objects contain `request`, `limits` as above,
and `expect` with only exact `content` and `finish_reason`. Both requests must
set `stream_options: {"include_usage":true}`. For example, require 4,096
computed peer prefill tokens, at least two decoder output events and a 1,000 ms
maximum observed gap; set these bounds from the release workload contract.

The collector starts the peer after the decoder's first generated output and
checks decoder progress until the peer's first generated output. The decoder
must remain live across that entire interval; increase its output length if it
finishes too early. Peer usage must prove `prompt_tokens - cached_tokens`
meets `min_prefill_tokens`; a hot or short prompt cannot pass that gate. Exact
responses, committed decode-token usage, both HTTP latencies and sampled
memory remain separate. Gap limits apply to received SSE output events, which
may contain multiple tokens. The interval includes peer queue/render time as
well as prefill and does not isolate GPU execution.

Overlap requires at least two effective banks, stable resolved plans, and
exactly two new route counts. Raw request/SSE/events and group boundary stats
are retained. `/v1/stats` supplies only `last_request`, so overlap deliberately
makes no per-request lane/reuse/speculation claim from that shared field.

For generated tool calls, use `scenario: "tool"`, declare the functions in
`request.tools`, and set `expect.finish_reason: "tool_calls"` with
`expect.tool_calls: [{"name":"weather","arguments":{"city":"Seoul"}}]`.
`expect.content` may be empty. The collector reconstructs fragmented arguments,
checks ordered names and JSON-object arguments, and rejects missing/duplicate
call IDs, invalid indexes and incomplete arguments. TTFT includes the first
function-name or argument output; ID-only metadata does not count. Functions
are not executed. Exact text still applies when a response contains both text
and tool calls.

For restart restore, add workload `restart_from: "seed/serving.json"` and make
the first case `scenario: "restart_restore"`. Set positive
`expect.min_cached_tokens` and `request.stream_options.include_usage: true`,
with expected `reuse_kind: "exact"`, `"partial"` or `"fork"` (a restored prefix
placed into a free bank). The seed must have passed
with disk enabled; the restored conversation must extend a seeded request.
Stop the seed process and restart the same argv/environment/binary/inputs
before collecting with the new `--server-pid`. The collector requires a newer
process, confirms the seed process stopped, compares plans, and requires zero
previous route requests plus actual cached-token usage on the first request.
Use `--repeats 1`; three fresh restart captures supply profile repetition.
The previous evidence is pinned transitively. Lifecycle orchestration remains
external so the collector cannot stop an unrelated service.
The seed path is part of the exact workload. A seed also binds its serving
plan, so different native settings need separate restart profiles. Compare
candidate settings on the same non-restart workload, then qualify restart on
the selected setting. Preserve or restore the intended task-owned disk-cache
state between fresh captures; a declared cold case must remain cold.

This collector establishes the recorded cases only. A `serving.json` pass is not
a deployment profile or a replacement for numerical/state gates. Sampling
cannot prove instantaneous clock or memory extrema. Live release gates remain in the
[v0.1.3 ledger](releases/v0.1.3.md).

## Serving controls and profiles

```sh
./ds4-perf serving-controls --plan server-plan.json --out scratch/controls
./ds4-perf serving-profile --plan selection.json --out scratch/profile
./ds4-perf apply-profile --profile scratch/profile/profile.json --out scratch/check
# Start only after reviewing launch.json; this replaces the ds4-perf process.
./ds4-perf apply-profile --profile scratch/profile/profile.json --out scratch/start --execute
```

`serving-controls` consumes the resolved plan JSON (or `/v1/stats` JSON). Its
`controls` come from `ds4_core::serving_caps`; ds4-perf has no serving family
table. It proposes one common scheduler option per candidate, excluding the
current width and respecting the native cap. Reuse, bank, MTP and disk support
remain visible capability fields. Every proposal is explicitly unqualified;
capability presence alone cannot justify changing correctness or memory policy.

The selection file names one exact workload and its candidate captures. Paths
are relative to this file:

```json
{"workload":"serving-workload.json","candidates":[
  {"name":"baseline","runs":["baseline/serving.json"]},
  {"name":"chunk-512","runs":["chunk-512/serving.json"]}
]}
```

Every named case and overlap needs at least three measured samples per
candidate, either `--repeats 3` or separate complete captures. Duplicate evidence
cannot count as additional repeats: each named request stream must have distinct
timestamped SSE observations, even if artifact metadata was repackaged. KV state still follows each declared
scenario: use fresh server processes for repeated cold captures. The selector
rechecks hashed requests, raw SSE and event times, output/finish reason, route
counters and traces, sampled memory, and boundary/in-workload raw GPU
clock/temperature records.
Incomplete, incorrect, slower-than-contract or memory-violating candidates are
rejected before ranking. Ranking minimizes the worst latency fraction of its
declared bound across all cases and overlaps; workload times are never averaged.
Different device/driver identities cannot compete in one campaign.

Qualification requires `--server-pid`, complete input hashes, an observed clock
range contract and a source checkout. It checks all model/MTP/vision GGUF shards,
adjacent tokenizer/template files, and automatically selected PLE manifests,
payloads and FP8 scales against the manifest. Media must be embedded in the
hashed request. The direct server argv must use explicit `--max-seqs` matching
the effective count; unknown/duplicate options, unreviewed environment controls,
preloaded libraries and external weight-owner launches fail closed. Old captures
without these records remain measurements and cannot acquire qualification.

Selection and application each hash every distinct input once per invocation.
Repeated references within that operation reuse the computed digest only while
device, inode, size, modification time and change time remain unchanged.
The verifier compares the opened file and its path before/after reading,
checks every reuse against the expected digest, and rechecks all inputs before
publication or launch. It also pins parsed root evidence and tracks consumed
path aliases; symlink retargeting, replacement or mutation fails the operation. No hash
cache survives between commands; collection still hashes inputs before and
after the HTTP workload.

The profile binds the selected argv, runtime environment, working directory,
executable, source digest, inputs, device and effective plan. Source digests
include untracked runtime files; changing a native include or Rust module
invalidates the profile even before it is committed. `apply-profile` rechecks
the complete evidence and current source/binary/input identities, then
writes `launch.json` and `expected-plan.json`. This default check starts no server and does not establish
current GPU availability. `--execute` additionally invokes the pinned server's
model-free `--check-config`, checks its effective/qualified plan and current GPU
identity/clock, then executes the recorded launch with a controlled
`--expect-plan` argument. The server compares family, backend, effective options,
qualification and controls again after native initialization; any change fails
before listening. Captured candidates and the final post-open plan also compare
the memory quote with only transient `available` excluded. Preflight skips quote
comparison because pre-open and post-open residency credits differ; this
exclusion is recorded in the profile. A prior recorded
guard must itself be hashed in the workload and match the measured plan; the
wrapper replaces its path for the new launch. Extra CLI options are rejected;
runtime environment overrides are cleared. It does not stop an
existing server. A profile qualifies only the named workload on the recorded
configuration; it does not qualify unmeasured restart, tool, media or MTP paths,
nor mark P4 or the release complete.

## Bounded optimization

```sh
./ds4-perf optimize --scout scratch/perf/baseline/scout.json --out scratch/perf/plan
./ds4-perf optimize --auto --scout scratch/perf/baseline/scout.json \
  --out scratch/perf/campaign --rounds 3 --repeats 3 \
  --timeout-seconds 1800 --max-output-mib 2048
```

Evidence orders concrete runtime experiments: phase diagnosis first, then
complete fit/calibration and NCU evidence when available. Initial controls cover
Qwen prefill chunks/PLE workers, Dots3 prefill chunks, and Solar grouped GQA
chunks. These proposals require measured prefill evidence; decode diagnosis,
geometry and NCU counters cannot select or reorder prefill controls. Decode
performance remains part of every acceptance comparison.
Unsupported families need a supported control before automatic execution;
the tool does not generate CUDA edits. `--plan FILE` supplies up to 32 explicit
one-variable candidates using the same reviewed controls:

```json
{"candidates":[{"name":"qwen-chunk-512","environment":{"DS4_QWEN_PREFILL_CHUNK":"512"},"reason":"Test wider work after launch fragmentation"}]}
```

Auto replays the captured controls and collector/helper identity. Each candidate
is bracketed by fresh control measurements. It is retained only when both
comparisons prove correctness, exceed the 1% improvement floor in prefill or
decode, and satisfy the slowdown limit for every metric. Failed, tied, noisy,
incorrect, and regressing trials remain recorded. A retained candidate becomes
the next control. No source code or production launcher is changed.

Per-process deadlines, process-group termination, interrupt handling, and polled
output budgets bound execution. Auto also accounts for cumulative campaign
output. Raw profiler writes can overshoot the byte limit between polls; these
limits do not replace the host memory guard or a filesystem quota.

## Evidence contract

`machine.json → calibration.json → collection.json → fit.json → ncu.json → scout.json → compare.json → decision.json`

Each typed artifact has a kind-specific schema version, producer version,
completion status, warnings, and SHA-256 input references. `collection.json`
retains the base scout evidence; final `scout.json` is published after requested
fit/NCU stages and final output accounting. A failed stage cannot leave a
complete final scout. Unknown schemas fail
closed. Publication is atomic and refuses overwrite. Loading verifies the
transitive input chain. Machine/calibration imports copy their relative evidence
bundle; local raw references survive moving the complete directory. Cross-run
comparison/decision references may be absolute, so retain their input runs too.
Raw data lives in ignored `scratch/`; publish scoped receipts rather than model
files or environment secrets. See the [v0.1.1 ledger](releases/v0.1.1.md) for
measured validation and remaining limits.
