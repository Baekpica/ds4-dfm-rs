# IQuest-Q1

The Rust host implements the pinned mixed-quant artifact, official Jinja,
NFC tokenizer and IQuest output protocol. The native CUDA path implements
hybrid Q8_0 KV, persistent banks, partial prefix reuse, snapshots, disk KV
and embedded recursive MTP. Bounded DGX Spark serial/MTP and bank-state
checks pass. Thinking-mode HTTP reuse, restart, serial/continuous streaming
and two-request concurrency pass their functional checks, with remaining
warm/cold reasoning-text differences. The 524288-token configured ceiling
is not a qualified limit.

## Artifact contract

- Source: [IQuestLab/IQuest-Q1](https://huggingface.co/IQuestLab/IQuest-Q1/tree/5c21b0630586ef77d38cff1094b8cf37a417fcd8),
  revision `5c21b0630586ef77d38cff1094b8cf37a417fcd8`.
- Mixed artifact: [Baekpica/IQuest-Q1-Mixed-Quant-GGUF](https://huggingface.co/Baekpica/IQuest-Q1-Mixed-Quant-GGUF/tree/6f08e63aa9dcd29cd29966180846266c0ca80639),
  initial weight revision `6f08e63aa9dcd29cd29966180846266c0ca80639`.
  Six shards, 1254 tensors, 87,995,249,696 file bytes; all six local shard
  SHA-256 hashes match the release manifest.
- Architecture: `iquest_q1`; 88 main layers, hidden 3072, vocabulary 160000,
  one dense FFN followed by 87 MoE layers with 256 experts and top-8 routing.
- Attention: 25 full-attention layers and 63 SWA layers with window 4096;
  48 query heads, eight KV heads, width 128. Q/K have per-head RMSNorm;
  split-half RoPE rotates 32 dimensions, with full/SWA bases 1000000/10000.
  Learned post-RoPE key sinks have zero values. Main layers 1–87 use
  normalized-input residuals.
- Main experts use IQ2_XXS with 15 IQ2_XS projection overrides. Q/O and
  dense FFN use Q5_K; K/V and embedding use Q6_K; output uses Q8_0.
  MTP experts use Q4_K and its remaining matrices Q8_0. Norms, routers and
  sinks stay F32. Gate and up projections may have different types.

The validator requires the pinned metadata, tensor names, shapes and types.
The [source license](https://huggingface.co/IQuestLab/IQuest-Q1/blob/5c21b0630586ef77d38cff1094b8cf37a417fcd8/LICENSE)
is Modified MIT, including an IQuest-Q1 display condition for commercial use.

## Input and output protocol

The source `chat_template.jinja` is preserved byte-for-byte. `developer`
keeps its own delimiter. Thinking opens `<think>`; disabled thinking renders
`<think></think>`. NFC normalization and the three ordered pre-tokenizer
splits precede byte-level BPE. EOS/pad is `<|iquest_end|>` (ID 0); chat adds
no extra BOS or EOS wrapper.

Keep thinking enabled for normal use; do not force `reasoning_effort=none`.
The recorded HTTP checks use `reasoning_effort=high` with a 256-token budget.
At runtime revision `f94697c5`, IQuest honors explicit sampling parameters
in both serial and continuous lanes and retains reasoning-bearing history
for cache retirement. Temperature 0 enables greedy MTP when requested;
positive temperatures use ordinary target sampling.

The output parser separates reasoning and extracts
`<iquest_tool_call>NAME<arg_key>KEY</arg_key><arg_value>VALUE</arg_value>…`
calls. Properties declared as strings preserve their text; other arguments
decode as JSON when valid, retaining booleans, arrays, objects and large
integers. Invalid JSON falls back to text. Malformed calls remain visible
text. Completed calls feed Chat Completions, Responses and Anthropic Messages;
tool continuation re-renders structured history through the same Jinja.
See [template normalization and checks](chat-templates.md).

Use HTTP tool clients for IQuest agent workflows; the built-in `ds4-agent`
executor requires DeepSeek DSML and rejects IQuest before loading weights.

Tool/history integers preserve exact decimal display, `tojson` and
integer/number classification at any size. Template integer arithmetic is
limited to signed 128-bit values; larger operands reject arithmetic.
Comparisons between two larger integers are exact, but mixed wide/native
ordering is unsupported. See the [adapter limits](../vendor/hf-chat-template/VENDOR.md).

## Embedded MTP

One recursive draft layer shares the target embedding, head and final norm.
It has a 512-token window and up to seven draft slots; no DFlash or other
separate drafter is supplied. Use the common `--mtp-mode`, `--mtp-draft` and
`--mtp-margin` controls. Explicit drafting accepts 2–7 draft tokens.
Start serving with `--mtp-mode off`; enable MTP separately for correctness
checks with `--mtp-mode on --mtp-draft 3 --mtp-margin 0`.

`ds4-bench` uses ordinary target decode with `--mtp-draft 1` and embedded
speculation with `--mtp-draft 2` through `7`. Use `--mtp-margin 0` for exact
greedy verification. External drafters and draft counts above seven reject
before model loading. The common `DS4_MTP_SPEC_DISABLE` kill switch also
disables benchmark speculation.

Greedy verification runs ordinary one-token target steps sequentially.
Accepted-prefix commit restores rejected target and MTP ring writes before
publishing the frontier. Sampled generation uses ordinary target decoding.
The fresh 2048-prompt/32-output ABBA comparison measured 3.39 tokens/s for
ordinary decode and 2.08–2.09 tokens/s with draft3. MTP is slower on this
workload; keep it off by default. See the [dated report](benchmarks/2026-10-01-iquest-q1-gb10.md).

## Memory contract

The six shards remain mmap-backed. Workers import canonical VMM ranges from
one owner. IQuest disables derived expert layouts and expanded Q8/F16/F32
weight caches; manifests with matching-model derived records fail before
any import. Disable all three owner repacks as shown below.

The canonical tensor payload is 81.934 GiB. The recorded Spark owner allocated
82.05 GiB across 91 VMM ranges. The actual-weight workers import those ranges
with zero owned canonical, derived or registered-host weight bytes and zero
census faults. File metadata/alignment accounts for the
difference between the 81.934 GiB tensor payload and 81.952 GiB GGUF files;
VMM allocation rounding is additional.

Each Q8_0 K/V row occupies 2176 bytes. Full layers reserve the requested
context; SWA layers reserve `min(context, 4096 + chunk - 1)` rows. MTP and
its rollback copy each reserve 519 rows. The target retains one 3072-element
F32 carry, avoiding a context-sized hidden-state history.

| Planned device allocation | 8K, chunk 128 | 512K, chunk 128 |
| --- | ---: | ---: |
| First bank and shared session workspace | 1.026 GiB | 27.174 GiB |
| Each additional bank, KV and carry | 0.955 GiB | 27.103 GiB |
| Eight lazy partial-reuse checkpoints | 4.192 GiB | 4.192 GiB |

These are allocator quotes, excluding common CUDA/MMQ scratch, allocator
overhead and host/driver headroom. The 512K quote does not prove Spark fit
or long-context correctness. Disk KV persists checkpoints; active banks
still require GPU memory.

## Serving

Build with `make cuda-spark` and `make CUDA_ARCH=sm_121 ds4_weight_server`.
Preserve the user-managed GPU clock range. Keep all six shards, the official
template and tokenizer configuration in one directory. Start one owner in
a separate terminal; wait for both `broker listening` and `ready manifest=`.

```sh
IQUEST_MODEL=/absolute/path/IQuest-Q1-MQ-IQ2_XXS-00001-of-00006.gguf
IQUEST_RUN="$PWD/scratch/iquest-serving"
mkdir -p "$IQUEST_RUN"
./ds4_weight_server --base "$IQUEST_MODEL" --backend vmm --scope base \
  --reserve-gb 28 --no-repack-iq2-aligned --no-repack-q8-aligned \
  --no-repack-q2k-aligned --manifest "$IQUEST_RUN/weights.ipc"
```

Set the same variables in the worker terminal. Qualification of this shape
remains bounded by the results and limitations below:

```sh
DS4_CUDA_WEIGHT_IPC_MANIFEST="$IQUEST_RUN/weights.ipc" \
DS4_CUDA_WEIGHT_IPC_SCOPE=base \
./ds4-server --cuda -m "$IQUEST_MODEL" --model-id iquest-q1 \
  --host 127.0.0.1 --port 8002 -c 8192 --max-seqs 2 \
  --native-chunk 128 --prefill-chunk 128 --prefill-chunk-live 128 \
  --prefix-reuse partial --mtp-mode off \
  --kv-disk-dir "$IQUEST_RUN/disk-kv" --kv-disk-space 16G \
  --kv-cache-min-tokens 1024 --print-plan --no-update-check
```

Context, banks, prefill, reuse and disk options use the
[common serving contract](serving-contract.md). Native prefill defaults to
128 and accepts 1–8192 rows. Inspect requested/effective/qualified values
in the plan and `/v1/models`, then verify `/v1/stats` and a real completion.
Use the [memory guard](host-memory-guard.md) for bounded validation.

## Qualification

Model-free checks cover artifact metadata/layout, official template renders,
developer roles, typed JSON tool history, reasoning/tool output and numeric
continuation across Chat, Responses and Messages. The real vocabulary check
passes 534 tokenizer cases and four rendered chat vectors against the pinned
source tokenizer. Canonical-only manifest tests exercise the actual importer
with mocked GPU calls and reject derived records before any publication.

Fresh GB10 serial checks compare each accepted MTP prefix with ordinary
one-token target execution from the same snapshot:

| Fixture | Context / chunk | Prompt + output tokens | Draft | Accepted / proposed | Rounds |
| --- | --- | ---: | ---: | ---: | ---: |
| Short text | 512 / 4 | 13 + 12 | 3 | 6 / 15 | 6 |
| MTP ring wrap | 8192 / 128 | 517 + 32 | 7 | 19 / 74 | 13 |
| Main SWA ring wrap | 8192 / 128 | 4220 + 32 | 7 | 18 / 92 | 14 |

Every round has bit-exact full-vocabulary logits and byte-exact committed
target/MTP payloads. The second fixture wraps the physical 519-row MTP ring
during decode. The third crosses the 4096-token SWA window and wraps its
4223-row physical ring during decode; MTP wraps eight times during prefill.
All three retain zero census faults. A truncated native payload, with its
valid host prefix intact, is rejected and invalidates the checkpoint;
restoring the valid payload
recovers exact logits, frontier and target/MTP state. This checks in-process
payload recovery, not an HTTP worker restart. Diagnostic timings include
readback and establish no MTP speed gain.

A separate three-token, two-layer actual-weight prefix has 21 finite float
stages, exact embeddings and exact expert selection against the archived
decoded-weight reference. Final hidden mean cosine is 0.999956309, with an
error profile comparable to the archived native run. Other float stages and
source-reference Q8 KV bytes differ; this bounded structural check does not
qualify whole-model quantization quality.

The diagnostic executables use the same canonical owner. For example:

```sh
make CUDA_ARCH=sm_121 tests/iquest_verify tests/test_iquest_prefix
DS4_CUDA_WEIGHT_IPC_MANIFEST="$IQUEST_RUN/weights.ipc" \
DS4_CUDA_WEIGHT_IPC_SCOPE=base DS4_IQUEST_PREFILL_CHUNK=4 \
  ./tests/iquest_verify "$IQUEST_MODEL" "$IQUEST_RUN/verify-short" \
  --ctx 512 --draft 3 --steps 12
```

Six native bank checks pass beyond the 4223-row physical ring: source
preservation, full/partial fork, matched-schedule replay, in-place rewind
and disk restore preserve exact payloads. Cold prefill with different
batch widths is not bit-exact: the audited KV difference starts at token
4352, and one long continuation changes EOS selection. These are separate
from the exact state-copy checks; cross-width output parity is unqualified.

With `f94697c5`, the v6 thinking-mode HTTP campaign passes seed, append,
edit, branch and worker-restart checks at context8192/two banks/chunk128,
MTP draft3/margin0. Warm requests reuse 564/530/615 tokens; restart restores
668 tokens from disk. All five warm/cold answers are the strict expected
`4/5/6/8/9` with `stop` and nonempty reasoning. Four full messages match;
append reasoning differs, so the strict whole-message cold gate fails.
MTP-off seed/warm/restart checks also pass with the same five strict
answers; three whole messages match cold, with reasoning wording changes
on append and branch. Neither mode has deterministic full-message parity.
Earlier disabled-thinking format failures and the initial thinking-mode
host failures remain recorded in the [dated report](benchmarks/2026-10-01-iquest-q1-gb10.md).

Chat, Responses and Anthropic Messages pass buffered/SSE reasoning checks
in both serial and continuous lanes. Both lanes pass Chat/Responses
generated-tool-call continuation and temperature0.7 thinking through
ordinary sampling with zero speculative counter increase.
Two concurrent SSE requests interleave generation in the continuous lane
and return exact `1..24` sequences with nonempty reasoning and unchanged
fault counters. Eight disabled-thinking API checks fail their strict answer
format despite zero fault-counter changes.

Serving qualification fields remain unset: the current fields cannot
express the thinking-mode and prefill-width limitations. The bounded
checks above do not establish a general context/bank limit, cross-width
output parity or whole-model source quality. See the [dated report and
compact evidence](benchmarks/2026-10-01-iquest-q1-gb10.md). Downloaded
reference reports retain their own hardware and workload scope; they do
not qualify 512K Spark serving.
