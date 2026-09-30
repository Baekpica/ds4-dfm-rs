# Naive HTTP and arithmetic gates

DGX Spark/GB10, MQ87 revision `65b235a4`, shared VMM weight owner,
GPU clock range 300–2200 MHz. The server executable SHA-256 is
`62a5e9d8935704077564427b128838721a23265d93f1f4b254d66014eafebb63`.
The tests ran at configured context 8192, two continuous banks and native,
scheduler and live chunks 64. Actual prompts contain 624–702 tokens.
This qualifies the recorded short-request features, not full 8K requests
or 256K/512K/1M serving. Receipts: [serving evidence](serving-evidence.json).

## Ordinary decode

Fresh seed, append, edited-prefix reuse, retained branch and first-generation
disk restart pass `tests/serving_reuse_live.py`. A third fresh process with
reuse disabled produces identical assistant messages, completion counts and
stop reasons for all five saved requests. Each arithmetic answer is checked
against predeclared literal forms, separately from cold/warm agreement.

| Request | Input tokens | Cached tokens | Observed reuse |
| --- | ---: | ---: | --- |
| Seed | 624 | 0 | Cold |
| Append | 650 | 626 | Fork |
| Edit | 650 | 624 | Partial |
| Retained branch | 676 | 652 | Exact |
| Fresh-process restart | 702 | 678 | Fork |

Disk capacity is 2 GiB with a one-token persistence threshold. The minimum
observed host available memory is 33.27 GiB; all task workers exit cleanly.
The native quote includes 236614656 bytes per bank, 48591616 scratch bytes,
204472320 checkpoint-pool bytes and the existing 4-GiB floor. Shared owner
weights are resident credit, not an independent worker copy.

Two overlapping requests both finish correctly; the native log records
`continuous rolling path=cont served=2 fallback=0`. Automatic XML tool
selection emits `get_weather` with city `Seoul`; its tool-result continuation
returns the supplied temperature. Closing a live stream after eight SSE
chunks leaves the next arithmetic request working, without governor faults
or lane fallback. These are functional gates, not throughput measurements.

## Explicit one-bank reuse

A fresh one-bank 8K campaign passes ten requests: seed, append, edited
prefix, retained branch, disk restart and a third cold process. Every
message, stop reason and completion count matches the cold result. Append
keeps 626 tokens, edit and branch restore 624 through partial checkpoints,
and disk restart keeps 678. A single bank can restore itself; it cannot
perform a cross-bank fork. Receipts: [one-bank evidence](single-bank-evidence.json).

The host previously rejected this native lane as `partial_lane`. Its
regression fails before the scoped explicit-width fix, then all 86 serving
policy tests pass. A second regression exposes a zero checkpoint-pool
quote versus 204472320 required bytes; all 82 host quote tests pass after
that fix. The earlier functional receipts retain their original zero-pool
stats. A separately relinked live server confirms the corrected pool and
32K total quote of 7175570688 bytes; the native payload format is unchanged.

A one-bank 32K pilot processes all 32512 input tokens, retrieves the exact
access phrase at token offset 5547, and stops normally. Continuation keeps
32525 of 32548 tokens; a fresh-process disk continuation keeps 32561 of
32584. All three answers match, with no lane fallback, speculation or
fault. This is a recorded near-capacity functional fixture, not a broad
long-context quality score or a two-bank 32K gate.

## Explicit DSpark

The same ten-request seed/warm/restart/cold campaign passes with the pinned
Q8 DSpark artifact, `--mtp-mode on --mtp-draft 2 --mtp-margin 0`.
Every request reports actual speculation on the continuous lane; cold and
reused assistant messages, counts and stop reasons match exactly. An append
performs an actual fork, while an edited prefix performs partial reuse.

A verified stop row can be committed by speculation while ordinary decode
leaves it pending. Canonical chat history may therefore restore a partial
checkpoint for the retained branch or restart. Runner v4 permits this
specific Naive-on transition while requiring a real warm fork, positive
cached tokens, answer correctness and cold parity. The earlier v3 trace
failures remain in the raw evidence. No native cache behavior was changed
to satisfy the runner.

The default margin 3 admits no proposals for the short seed. That earlier
speculation assertion failed; a longer request did speculate. Explicit
margin 0 is the qualified fixture setting. Draft depth, margin and context
combinations outside this fixture remain unqualified.

The draft-aware native quote adds 10547200 bytes per bank, takes
214640760 scratch bytes and 288358400 checkpoint-pool bytes. DSpark is
slower in the separate serial fixtures. The [8K diagnostic profile](dspark-profile-evidence.json)
also records low acceptance and repeated target work. No acceleration is
qualified and automatic speculation remains off.

## Mixed-weight arithmetic reference

Six pinned holdout cases compare 32 full-vocabulary positions each:
192 rows and 29294592 F32 native logit values. All are finite. Reference
weights are canonical decoded GGUF rounded to BF16; native MMQ uses
activation quantization and F32 weight reconstruction. This is arithmetic
agreement with that reference, not original-source BF16 model parity or
a generation-quality score. Full receipts: [quality evidence](quality-evidence.json).

| Case suffix / category | Input | Top-1 agreement / 32 | Mean KL | Reference / native NLL |
| --- | ---: | ---: | ---: | ---: |
| 00000 / chat | 2048 | 26 | 0.368040 | 3.700894 / 3.499704 |
| 00008 / reasoning | 1193 | 29 | 0.008627 | 1.465939 / 1.413077 |
| 00016 / multilingual | 2048 | 28 | 0.015506 | 1.649449 / 1.619761 |
| 00024 / agentic code | 1930 | 28 | 0.052529 | 1.358297 / 1.355146 |
| 00032 / long finance | 4096 | 23 | 0.239989 | 5.127774 / 5.011060 |
| 00040 / Chinese science | 1952 | 24 | 0.407656 | 6.546400 / 6.396510 |

Aggregate top-1 agreement is 158/192. Maximum absolute logit difference
reaches 15.5 and maximum KL reaches 4.164576. Lower sampled native NLL is
not evidence of superior quality. These differences are disclosed; they
must not be described as full-model reference parity. The optimization
rounds separately require byte-exact agreement with the retained native
baseline, including target/indexer/draft state.

## Original BF16 source comparison

The same six fresh DGX outputs also compare against prerecorded original
BF16 logits from the pinned source revision, generated on eight A100s.
This is not a fresh DGX BF16 model run. Positions, targets and the input
manifest match exactly; the reference file SHA-256 is
`1ea2999f01a6c1e2020aab790b5766ebdf421de49fe13bfc8e0906b5f12ea487`.
Receipts: [source quality evidence](source-quality-evidence.json).

| Case suffix | Top-1 agreement / 32 | Mean KL | Original / native NLL |
| --- | ---: | ---: | ---: |
| 00000 | 25 | 0.687460 | 3.854552 / 3.499704 |
| 00008 | 24 | 0.082664 | 1.576411 / 1.413077 |
| 00016 | 26 | 0.121342 | 1.720866 / 1.619761 |
| 00024 | 27 | 0.096054 | 1.247757 / 1.355146 |
| 00032 | 20 | 0.847195 | 5.637621 / 5.011060 |
| 00040 | 18 | 0.796585 | 6.317648 / 6.396510 |

Aggregate agreement is 140/192, mean KL 0.438550, and mean sampled NLL
3.392476 / 3.215876. Quantization and native arithmetic both contribute;
these samples cannot isolate their effects or establish generation-quality
parity. Neither lower sampled NLL nor the exact optimization A/B proofs
remove this limitation.

Raw evidence remains under ignored `scratch/naive/api-reuse-main`,
`api-reuse-draft-v4`, `api-surface-main`, their launch/guard logs and
`quality-q6-6`. Private holdout prompts and handoff files are not published.
