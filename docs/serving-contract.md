# Serving contract

Common option names. Per-family implementation, memory, and verification.
See the [v0.1.3 ledger](releases/v0.1.3.md).

The host resolves one [`ServingRequest`](../crates/ds4-core/src/serving.rs)
into a [`ResolvedPlan`](../crates/ds4-core/src/serving.rs): requested,
effective, and qualified. `--print-plan` prints it. `--check-config`
exits 2 if any requested feature is unsupported. `GET /v1/stats` includes
the same object plus `last_request`.

## Flags

| Flag | Meaning | Aliases |
|---|---|---|
| `--ctx` / `-c` | Per-sequence context limit, capped per family | |
| `--max-seqs N\|auto` | Concurrent banks/sequences | `--cont-width`, `DS4_SERVER_COALESCE_MAX` |
| `--prefix-reuse off\|exact\|partial\|auto` | Conversation reuse policy | `DS4_SERVER_FORK`, `DS4_SERVER_FORK_PARTIAL` |
| `--mtp-mode off\|auto\|on` | Speculation policy | `DS4_MTP_SPEC_DISABLE` for off-with-weights |
| `--mtp PATH`, `--mtp-draft N` | Sidecar and draft length | |
| `--kv-disk-dir`, `--kv-disk-space-mb` | Persistent checkpoint store | `--kv-disk-space 32G` |
| `--prefill-chunk`, `--prefill-chunk-live` | Scheduler yield sizes, capped at 8,192 | `DS4_CONT_PREFILL_CHUNK`, `DS4_CONT_PREFILL_CHUNK_LIVE`, `DS4_CONT_PREFILL_NOFENCE=1` lifts the cap |
| `--mem-floor-gb` | Single host floor | `DS4_MEM_FLOOR_GB` (published for native) |
| `--print-plan` | Print resolved JSON and continue | |
| `--check-config` | Print resolved JSON and exit | |

`auto` reuse is the best *qualified* path. Forced `partial` on a family
that only has exact-frontier reuse is an error, and so is forcing it when
this process cannot run it: checkpoint replay lives in the bank driver and
needs the opened runtime's checkpoint store, so serial-only serving or a
runtime without that store rejects `partial` and downgrades `auto` to
`exact` with a warning rather than reporting reuse it will not perform.

`--max-seqs` is not context length. Keeping N banks is not the same as
batching N requests in one kernel. Step banks each own KV and prefill
scratch; more banks are not a linear tok/s gain. `auto` may fit fewer
banks than it asked for; an explicit `--max-seqs N` the native fit
cannot honour is an error, not a narrower start.

The quote's `floor` includes native fit headroom where required: normally
the host floor plus a 2 GiB burst reserve. `DS4_BATCH_FIT_HEADROOM_MB`
overrides that reserve; `DS4_BATCH_FIT_HEADROOM_DERIVED=0` selects 6 GiB,
otherwise `DS4_BATCH_FIT_BURST_MB` changes the burst. The quote always
preserves at least `--mem-floor-gb`.
CUDA serial graph fitting also reserves 1 GiB by default, configurable
with `DS4_SESSION_GRAPH_HEADROOM_MB` (disabled by `DS4_SESSION_GRAPH_FIT=0`).
The quote keeps the larger of the applicable serial and batch reserves.
DeepSeek shared graph costs include its initial caches and native workspace;
packed cache mirrors are conservatively included even when native VMM
support may disable them. The same native chunk sizes serial and batch graphs.
DeepSeek bank costs use full-depth compressed-cache capacity plus raw rings
and rollback states. Loaded DSpark runtime costs count even with `DS4_CONT_DSPARK=0`.
A manifest with drafter ranges defers the pre-open quote: import can soft-fail.
After open, a successful import excludes shared drafter weights; fallback
retains the local weight cost. Runtime allocations remain charged in both cases.

`--cont-width 0` and `DS4_SERVER_COALESCE_MAX=0` keep the legacy serial
meaning: no bank lane. `DS4_SERVER_CONTINUOUS=0` is narrower — it forces
the static/serial route, so the banks stay for the static lane to coalesce
over while no request enters the bank driver. Qwen and DeepSeek speculate only inside that
lane, so the combination rejects `--mtp-mode on` instead of reporting
MTP enabled. Inkling and Step also speculate on the serial engine and
are unaffected.

Disk KV is not active-bank offload. Resident bank state, partial
checkpoint memory, and disk budget are separate. A directory does not
persist conversations below the bank persist threshold (default 8,192
tokens on the continuous lane, `DS4_SERVER_PERSIST_MIN_TOKENS`). That is a
different number from the disk store's record minimum
(`--kv-cache-min-tokens`, default 512); the plan reports both.

HTTP disk records require a matching `local-file-stat-v1` identity: all
GGUF/sidecar file metadata, template contents, runtime files and effective
inference settings. This is local file identity, not full weight-content
attestation. Inputs must remain unchanged during model open. Legacy records
without this identity miss safely; all identities share the directory budget.
Native restore holds the validated file open through the payload read.

MTP weights loaded is not "this request speculated". Sampled Step
requests keep predictor state and use ordinary decode.

## Inspect

```sh
./ds4-server --check-config --cuda -m "$MODEL" \
  --prefix-reuse auto --max-seqs 2 --mem-floor-gb 12 \
  --kv-disk-dir /tmp/ds4-kv --kv-disk-space 32G --mtp-mode auto
```

The JSON has `requested`, `effective`, `qualified`, `issues`, and when
the host supplied a memory ceiling, `quote`. `effective.native_chunk` is
the allocated workspace/graph max, not the scheduler yield.
`qualified.prompt` is the verified request length when it is smaller
than configured `--ctx`. A family's session cap is separate from its
qualified context: GLM sessions refuse anything above 2,048, so the
shared default is an error there, not a warning. `--native-chunk` sets
the allocated native capacity so `--check-config` can refuse a yield
the process cannot run.

## Request trace

`GET /v1/stats` field `last_request`:

| Field | Meaning |
|---|---|
| `effective_lane` | `serial`, `continuous`, or `static` |
| `reuse_kind` | `cold`, `exact`, `partial`, or `fork` |
| `speculation_active` | This request used speculative decode |
| `reuse_miss` | Why a candidate was refused, when one was |
| `fallback_reason` | Why a requested path was not used |

The same fields may appear next to HTTP `timings`.

Each field is recorded where the decision is made, not inferred from
counters. `exact` reuses a state that ends at this prompt's common
prefix and prefills only the appended turn, so cached and computed
tokens are both positive; `partial` restores a checkpoint below that
prefix and replays the gap, including a partial copy into another bank;
`fork` copies a complete retained frontier into another bank and preserves
the source. `speculation_active` follows the executed path: the serial
engine's speculative eval, or a native sequence that ran draft rows.

## Miss reasons

Restore miss is not "disk broken". `reuse_miss` carries the reason from the
decision that produced it, and is absent when nothing was refused:

- `rendered prefix changed` (template dropped an empty thinking block).
  Step and Motif official follow-up renders omit a generation-only empty
  `<think>` pair. Their history checkpoints capture matching native KV,
  tokens and logits before that suffix; the completed bank retains its
  original token sequence. A Motif append can therefore require `partial`
  rollback even when the visible messages only grow. Without a compatible
  checkpoint, the request stays cold: shortening a text key cannot change KV.
- `below minimum token threshold`
- `payload family/layout mismatch`. Also the answer when the chosen
  payload cannot be read back — a truncated or corrupt record is refused
  by its payload, whatever wrote it.
- `no checkpoint at or below LCP`
- `session state requires prefix replay`. Cached tokens count only the
  prefix preserved by the native sync plan. Dots3 MTP replays an unaligned
  append from zero (`cold`); plain Dots3 replays its final partial chunk
  (`partial`). An identical prompt or aligned append retains its full hit.

## Capability table

The [generated capability table](serving-capabilities.md) reads
`ds4_core::serving_caps` and the same resolved controls consumed by
`ds4-perf serving-controls`. Its model-free check runs with the core tests;
the page includes the regeneration command. Dated reports stay historical.
