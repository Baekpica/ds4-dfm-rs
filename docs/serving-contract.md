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
| `--ctx` / `-c` | Per-sequence context limit | |
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

`--cont-width 0` and `DS4_SERVER_COALESCE_MAX=0` keep the legacy serial
meaning: no bank lane. Qwen and DeepSeek speculate only inside that
lane, so the combination rejects `--mtp-mode on` instead of reporting
MTP enabled. Inkling and Step also speculate on the serial engine and
are unaffected.

Disk KV is not active-bank offload. Resident bank state, partial
checkpoint memory, and disk budget are separate. A directory does not
persist conversations below the bank persist threshold (default 8,192
tokens on the continuous lane, `DS4_SERVER_PERSIST_MIN_TOKENS`). That is a
different number from the disk store's record minimum
(`--kv-cache-min-tokens`, default 512); the plan reports both.

MTP weights loaded is not "this request speculated". Sampled Step
requests keep predictor state and use ordinary decode.

## Inspect

```sh
./ds4-server --check-config --cuda -m "$MODEL" \
  --prefix-reuse auto --max-seqs 2 --mem-floor-gb 12 \
  --kv-disk-dir /tmp/ds4-kv --kv-disk-space 32G --mtp-mode auto
```

The JSON has `requested`, `effective`, `qualified`, and `issues`.
`qualified.prompt` is the verified request length when it is smaller
than configured `--ctx`.

## Request trace

`GET /v1/stats` field `last_request`:

| Field | Meaning |
|---|---|
| `effective_lane` | `serial`, `continuous`, or `static` |
| `reuse_kind` | `cold`, `exact`, `partial`, or `fork` |
| `speculation_active` | This request used speculative decode |
| `fallback_reason` | Why a requested path was not used |

The same fields may appear next to HTTP `timings`.

Each field is recorded where the decision is made, not inferred from
counters. `exact` reuses a state that ends at this prompt's common
prefix and prefills only the appended turn, so cached and computed
tokens are both positive; `partial` restores a checkpoint below that
prefix and replays the gap; `fork` copies another bank and preserves
the source. `speculation_active` follows the executed path: the serial
engine's speculative eval, or a native sequence that ran draft rows.

## Miss reasons

Restore miss is not "disk broken". Examples:

- `rendered prefix changed` (template dropped an empty thinking block).
  Step's official follow-up render drops the empty `<think>` pair that the
  stored KV still holds, so the restart is a cold prefill until a
  checkpoint exists at that history frontier (P1); reusing across it would
  continue from a token sequence the client never sent.
- `below minimum token threshold`
- `payload family/layout mismatch`
- `no checkpoint at or below LCP`

## Capability table

`ds4_core::serving_caps` is the living table. Family docs and this page
must not contradict it. Dated reports stay historical.
