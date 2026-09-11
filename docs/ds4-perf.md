# ds4-perf

`inspect`, `scout`, `compare`, and `optimize` form the Rust performance workflow.
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

Inkling has no serialized session checkpoint. Between sweep frontiers the
benchmark replays the prompt prefix outside both measured ranges, restoring
its KV and convolution history before measuring the next suffix. Its CSV
`kvcache_bytes` is therefore zero (no serialized snapshot), not a KV allocation
measurement. `tests/test_inkling_bench.py` compares sweep logits and tokens with
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
`DS4_INKLING_NO_Q8_ROUTED_TILE=1` restores warp-owned routed Q8 tiles;
from 64 prompt rows, the optimized path keeps weights in registers across
all routed columns with the shared-expert tile schedule.
`DS4_INKLING_NO_Q4_TILE=1` restores four-column Q4_K expert batches;
from 3072 routed assignments, eight-column tiles reuse payload loads and
unpacked scales with the same MMVQ partial sums and skip the unused relayout.
`DS4_INKLING_PREFILL_CHUNK=N` (1–8192) sets the maximum Inkling prefill chunk width;
shorter prompts remain valid. The default is 512, capped by context.
Increasing it to 8192 alone regressed the matched 8K Inkling workload.
Scout consumers also reparse each referenced unprofiled benchmark CSV and
require its rows to match the serialized samples. Hashing and parsing use the
same bytes; benchmark stdout is limited to 64 MiB per sample on load.

Every token sequence and full-vocabulary frontier is checked, including within-run
repeat consistency. Default logit tolerances are `atol=0.0001`, `rtol=0.0001`;
`--logit-atol` and `--logit-rtol` define an explicit alternative contract.
Prefill/decode inverse TPS and first-token seconds are compared separately.
The min/max sample envelope is conservative observed variation, not a statistical
confidence interval. The default slowdown limit is 3%; overlapping evidence is
inconclusive. `--regression` fails for regression, incorrectness, inconclusive
results, or incomparable inputs. This gate does not replace model/state,
capture/eager, long-context, or serving correctness gates.

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
