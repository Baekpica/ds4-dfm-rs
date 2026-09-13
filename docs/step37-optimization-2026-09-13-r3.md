# Step 3.7: capped-clock serving and optimization

This continuation adds partial-prefix checkpoints and banked greedy MTP to
Step's continuous lane. Rust selects accepted prefixes; native trial/commit
retains the target and predictor state. See the [serving contract](step37-serving-2026-09-13.md).

Performance comparisons use the same 300–2200 MHz GPU clock lock in both
arms. Earlier reports use different clock conditions and are not the A/B
baseline for this campaign.

## Protocol

One NVIDIA GB10, MQ83's nine shards, the official Q8 MTP sidecar, draft 3,
greedy 64-token generation and the unchanged `speed-bench/promessi_sposi.txt`
fixture. The 2048-input workload allocates context 2120; the separate
16384-input workload allocates 16456. No prefix or disk KV is reused.

Every arm runs in a fresh process after a separate, identical warmup.
Three A/B pairs alternate order (A/B, B/A, A/B). The same VMM owner serves
BASE aligned artifacts and raw MTP to both arms. Model/prompt identities,
binary SHA-256, exact commands, environment switches and 200ms clock/power
telemetry are retained with the results. Host memory guards remain active.

Before candidate measurements, retention required at least 1% median gain in
the target phase, positive gains in every pair, and no greater than 1%
median regression in the other phase. Correctness is independent of speed.
Nsight timings diagnose the path; the table uses unprofiled wall time.

## Measured rounds

| Round | Change | Target phase, tok/s A → B | Gain | Other phase |
|---|---|---:|---:|---:|
| Decode 1 | Skip the redundant last-row head before verifying every row | 21.87 → 22.30 | +1.97% | Prefill −0.04% |
| Decode 2 | Reuse the accepted row's full trial logits at commit | 22.32 → 22.77 | +2.02% | Prefill −0.22% |
| Prefill 1 | Share exact post-norm Q8 quantization across target consumers | 1183.19 → 1201.20 | +1.52% | Decode +0.66% |

Each round holds the other candidate switches constant. These medians are
separate matched comparisons; their percentages are not added together.

The initial capped profile found 362.25ms in 166 target vocabulary heads
within decode and 122.77ms in 471 MMQ quantization launches within prefill.
The two decode changes remove repeated output projection without changing
the verification width or accepted-token policy. The retained trial buffer
uses about 1.97MiB of host memory per MTP session/bank and is invalidated at
commit/reset. Device frontier logits are preserved too.

## Correctness

- `tests/test_step37_checkpoint`: wrapped target/predictor windows, full KV,
  all vocabulary logits, same-bank rollback, shared fork lineage, and refusal
  to rewind below restored live rows.
- `tests/test_step37_cont`: two-bank MTP matches serial tokens, full logits,
  target/predictor KV and held hidden rows. Full and partial forks preserve
  state. Disk snapshots round-trip after clearing KV. Cancellation, EOS and
  forced protocol tokens commit no unreported drafts.
- `tests/test_step37_norm`: the original F32 norm and Q8 consumer outputs
  match byte-for-byte at scalar, verify and prefill widths.
- `tests/test_step37_spec`: owner-imported BASE/MTP, wrapped 832-token input,
  32 generated tokens, all four commit lengths, every trial vocabulary row
  and committed live KV match independent controls. Prefill's control disables
  shared quantization. Width-one greedy choices also match; disk restore and
  invalid pending operations pass.
- Both decode A/Bs and Prefill 1 preserve all 128,896 frontier logits and all
  64 generated tokens across each round's six measured processes.

These tests compare this MQ83 artifact across execution paths. They do not
extend the earlier MQ83-versus-BF16 or broad model-quality qualification.

## Evidence

Local artifacts are under `scratch/step37/perf-r3/`: `PROTOCOL.md`,
`baseline-profile.json`, `bank-spec.log`, `checkpoint-mtp2.log`, `norm.log`,
`opt-spec.log`, and per-round `identity.json`, `results.json`, `analysis.json`
and full-vocabulary/token proofs. The initial diagnostic scout uses a raw
MTP-only owner; it is not numerically compared with the later full-owner A/Bs.
