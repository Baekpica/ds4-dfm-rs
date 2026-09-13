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
| Prefill 2 | Increase 16K prefill chunk 2048 → 4096 | 1215.83 → 1288.83 | +6.00% | Ordinary Decode −0.05%; MTP trajectory changes |

Each round holds the other candidate switches constant. These medians are
separate matched comparisons; their percentages are not added together.

The initial capped profile found 362.25ms in 166 target vocabulary heads
within decode and 122.77ms in 471 MMQ quantization launches within prefill.
The two decode changes remove repeated output projection without changing
the verification width or accepted-token policy. The retained trial buffer
uses about 1.97MiB of host memory per MTP session/bank and is invalidated at
commit/reset. Device frontier logits are preserved too.


Matched full-owner profiles confirm the removed work:

| Path | Before | After |
|---|---:|---:|
| Decode 1 target head | 166 launches, 363.67ms | 138 launches, 301.03ms |
| Decode 2 target head | 138 launches, 301.03ms | 110 launches, 241.02ms |
| Prefill 1 MMQ quantization | 471 launches, 123.20ms | 279 launches, 89.62ms |

The norm reduction and quantizer arithmetic stay unchanged. The producer
cache is keyed by buffer, row count and width; overwriting a norm invalidates
its old entry. Decode/verify widths keep their existing quantization paths.
`DS4_STEP37_LEGACY_TRIAL_HEAD=1`, `DS4_STEP37_LEGACY_COMMIT_HEAD=1` and
`DS4_STEP37_NO_Q8_REUSE=1` independently restore the measured controls.

### Prefill 2: wider chunks

The default chunk becomes 4096. `DS4_STEP37_PREFILL_CHUNK=2048` restores
the control and leaves more memory for concurrent banks; the tested mixed
image/bank configuration uses 512. Context fitting still applies.

On the separate 16384-input workload, chunk 2048 → 4096 raises median
Prefill from 1215.83 to 1288.83 tok/s (+6.00%). Every pair improves
(+6.04%, +5.60%, +6.08%). The ordinary-decode control independently measures
1214.18 → 1290.74 tok/s (+6.31%), with Decode 18.42 → 18.41 (−0.05%).
Here ordinary decode means MTP loaded with `DS4_MTP_SPEC_DISABLE=1`;
predictor state is still maintained.

Matched profiles halve Q4 worklist launches (464 → 232, 2755.41 →
2109.34ms) and IQ2 gate/up launches (272 → 136, 2343.66 → 2012.81ms).
Target graph allocation at context 16401 grows from 2.209 to 3.602GiB,
before predictor state. Wider chunks trade workspace for fewer launches
and larger worklists.

This round changes floating-point execution. Frontier logits differ by
4.56% relative RMS (maximum absolute delta 1.405); top-1 agrees, top-10
overlap is 9/10, top-50 overlap 46/50, and KL(A||B) is 0.01057. The MTP
continuation first differs at output index 24. Each arm is repeatable.
Tracing row 2047 finds identical layer-0 Q/K/V and the first observed
difference in attention output (maximum 3.29e-5, relative RMS 8.83e-7).
This supports width-dependent attention arithmetic amplified by later
mixed-quant execution. It is not bitwise parity.

Both widths independently pass all 45 layers' live ring/full KV,
full-vocabulary and rewind comparisons against a same-width full-history
control on a 16384-row wrapped fixture. That cyclic correctness fixture is
separate from the unchanged performance essay.

Four 17.8K-token retrieval/arithmetic requests in English, Korean, Chinese
and Python wording return the correct three marked codes and sum in both
arms. Strict JSON value/type equality is 4/4 at chunk 2048 and 3/4 at 4096:
the Korean answer returns `"42"` instead of `42`. The original strict gate
failure is retained; semantic scoring was added during review, not declared
in advance. The explicit Python integer request passes unchanged. This
bounded evidence supports retaining the speed change under the repository's
mixed-quant contract; it does not establish broad quality equivalence or
strict-schema parity.

The first quality run was stopped by the host memory watchdog while host
tests were also running. The isolated rerun passes with the same reserve.
No clock violation was observed. Functional GPU gates and host tests are
serialized thereafter.

MTP Decode rises 14.77 → 16.35 tok/s (+10.70%), but that includes changed
continuation and draft acceptance. It is not an isolated decode-kernel gain.
Ordinary decode is faster on this particular long-context workload; MTP is
not universally faster.

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

[Raw samples, identities and numerical summaries](step37-optimization-2026-09-13-r3.json)
make the round medians and paired gains reviewable without local logs.

Local artifacts are under `scratch/step37/perf-r3/`: `PROTOCOL.md`,
`baseline-profile.json`, `bank-spec.log`, `checkpoint-mtp2.log`, `norm.log`,
`opt-spec.log`, and per-round `identity.json`, `results.json`, `analysis.json`
and full-vocabulary/token proofs. The initial diagnostic scout uses a raw
MTP-only owner; it is not numerically compared with the later full-owner A/Bs.
