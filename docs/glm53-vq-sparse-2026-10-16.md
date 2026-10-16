# GLM VQ sparse attention through 8K

The eager VQ graph now builds source-compatible index pools beyond the
previous 2,048-key limit. Context admission is bounded at 8,192. This is a
text execution boundary; HTTP serving, MTP, graphs, vision, banks and saved
state combinations remain separate qualification work.

The active artifact still changes only 136 nonexpert KDA Q/K/V/O matrices
to Q4_K. Experts, codebooks, shared experts and the other source weights
are unchanged. The user's accepted Q4 quality tradeoff remains in force.

## Execution and arithmetic

Each DSA layer applies its BF16 key projection and affine LayerNorm, then
pools four keys using the source gate/APE softmax. Probabilities, products
and pool sums cross the source BF16 boundaries. Index dots and weighted
head reduction use FP32. Each query selects 512 complete pools and up to
three incomplete-tail tokens, with an independent causal attention mask.

Pools are built during the dense prefix when the allocated context can
exceed 2K. A leading carry finishes before a later partial pool writes the
four-row tail. The old 2K allocation skips this extra indexer work entirely.
BF16 pool rows and FP32 partial tails use existing per-layer state buffers;
this does not import DeepSeek compressed-KV or bounded-replay rules.

Single-row latent attention now accepts that mask through 8K. It preserves
the previously adopted factorization and its explicit BF16 rounding change.
Wider queries retain expanded BF16 attention. `DS4_GLMVQ_LATENT_ATTN=0`
provides the matched expanded control; this is not a new permanent semantic
mode. Capacity, shape, format and alignment checks retain fallback behavior.

## Measurements

One fresh matched pair uses 7,936 input tokens and 128 forced non-EOS
greedy tokens, context 8,192, chunk 1,024, MTP/capture off. Both arms use
the same binary and Q4 artifact. A synthetic raw code-audit fixture repeats
the established prompt three times; routing comes from the actual GLM.
Only `DS4_GLMVQ_LATENT_ATTN` differs.

| Measurement | Expanded control | Latent candidate |
|---|---:|---:|
| Prefill tok/s | 398.23 | 393.26 |
| Decode tok/s | 7.07 | 16.58 |
| Inferred prefill + decode seconds | 38.033 | 27.900 |
| First decode step seconds | 0.1525 | 0.0732 |
| Inferred compute TTFT seconds | 20.081 | 20.253 |
| Entire guarded process seconds | 301.498 | 295.489 |

Decode improves 134.51% (2.35x) and the inference phase falls 26.64%.
Prefill is 1.25% slower in this pair; there is no prefill improvement claim.
The process wall includes model loading and cleanup. Phase and compute TTFT
are inferred from rounded CSV rates; TTFT here excludes HTTP/admission.
All 154,880 initial frontier logits and 128 generated IDs match. Exact IDs
are an observation, not a quality acceptance threshold.

This is one 8K pair. Prior repeated 2K evidence establishes repeatability
only in its recorded range. The expanded control makes 1,496 full-prefix
KV projections; the candidate retains 88 prefill projections and replaces
1,408 single-row calls with the previously assessed latent factorization.
Allocated workspace is identical between the arms; actual peak is measured
only for the retained candidate below.

Busy SM clocks were 2,184–2,190 MHz under the unchanged 300–2,200 MHz
policy. Each fresh process used one in-process VMM owner with inherited OS
file cache. The guard capped scope at 104 GiB, high at 100 GiB, reserve/trip
at 12 GiB, timeout 1,800 seconds; PSI was recorded without a kill threshold.
Minimum system available memory was 38.151/38.263 GiB (off/on).

Bench SHA256 is
`4423f128faaeb2fa29dc4b34845395128e2ee3be64b687720090b7b15994c42d`;
manifest SHA256 is
`6ea4054b04cfb1bd2e26db318e022a5918b49c8002d7d5e1cf17e502a074a044`.
The compiled snapshot is based on `ea0dedda` plus the recorded patch/source
hashes; later test-only Rust, quality-harness, dependency and documentation
edits are recorded separately. Actual mapped libraries match across arms:
driver 615.71.09, cuBLAS/cuBLASLt 13.8.0.4, CUDA runtime 13.4.92.

A separate fresh 2K compatibility sample measured 420.75 prefill / 17.16
decode tok/s. Its 154,880 frontier logits and 32 generated IDs exactly match
the closed prior baseline. This is numerical compatibility with historical
speed context, not a new matched 2K performance claim.

The retained Nsight trace records 18.904136 seconds prefill and 7.762537
seconds decode. Its build, artifact, settings, actual library maps, frontier
and generated IDs match the plain candidate. Profiled throughput is
419.79/16.49 tok/s; this is attribution, not another unprofiled A/B pair.

| GPU region | Prefill seconds | Decode seconds | Decode wall share |
|---|---:|---:|---:|
| Dense/shared/control projections | 3.984022 | 3.547443 | 45.70% |
| Routed VQ | 6.033128 | 1.285962 | 16.57% |
| Output head | 0.016735 | 0.920613 | 11.86% |
| mHC/norm | 1.629432 | 0.702055 | 9.04% |
| Attention/KDA core | 6.288569 | 0.657143 | 8.47% |
| Router selection | 0.072171 | 0.301756 | 3.89% |
| Index pooling/selection | 0.087141 | 0.058853 | 0.76% |

Kernel regions follow the eager launch order; attention/KDA and indexer
projection GEMMs are included in dense/shared/control. GPU idle gaps total
0.666611/0.241777 seconds (prefill/decode). These bound host/dispatch gaps;
they are not isolated CPU time. Synchronous CUDA API durations overlap GPU
work and are not added to the table.

The largest remaining decode kernels are cooperative Q4 projections,
1.523313 seconds (19.62%), and direct FP8 projections, 1.431921 seconds
(18.45%). Prefill KDA recurrence alone takes 4.510760 seconds (23.86%).
No further Q4 transport variant is adopted here: the prior actual-matrix
async/prefetch screen was slower and remains closed.

Peak live Nsight DEVICE allocations are 78.492320 GiB, with all allocations
matched to releases. This excludes mapped files/driver/OS storage; GB10
UMA is not the sum of CPU RAM and VRAM. Minimum system available memory is
36.781971 GiB during tracing. The 8K off arm has no separate peak trace.
Decode makes no DEVICE allocations. The candidate replaces full-prefix
KV expansion without adding scratch to the matched 8K graph.

Two frozen near-8K candidate cases also passed: a 7,377-token Korean record
correctly answers Friday and Seoyeon; a 7,390-token code prompt returns the
requested coupon subtraction with unchanged parameters. Both end at EOS.
The harness checks all position transitions and finite full-vocabulary
logits at 45 teacher-forced positions. Mean target NLL is 0.348806/0.039575
nat, informational without a paired 8K quality control.

This bounded scope was selected before any 8K answer results. The prior
2K paired output assessment and the new 8K component arithmetic/mask proofs
remain reusable; the expanded quality-control reload was not run. Template
wording in the raw quality receipt still describes that earlier plan;
`quality_assessment` records the actual scope. These cases do not establish
broad quality, tool, image or multi-turn behavior.

## Focused evidence

- Indexer: independent BF16/FP64 reference, chunk/carry, causal masks,
  including the 2K boundary and 8K component geometry; no sanitizer errors.
- Masked attention: actual layer3 KV_B, synthetic BF16 cache/query,
  2,049/8,192 keys, single-row factorization, width-two fallback and
  deliberately selected future keys. Per-operation propagated bounds pass;
  unselected/future probabilities are exactly zero. Cache/positions stay
  unchanged. Source-relative output differences are informational.
- Native binding: actual norm/bias/APE, partial-tail completion, sampled
  score reference, 32 rows of top512/expanded selections, selected-pointer
  capacity checks and masked DSA handoff. No sanitizer errors or leaks.
- Scratch quote: all 37,976 active source matrices across nine context/width
  cases. Focused Rust admission test accepts 2,049/8,192, rejects 8,193 and
  keeps public qualified limits unset.

Initial metadata quote used stale AllQ4 input and failed on the head quote;
using the active artifact's immutable metadata passed. An admission-test
integer type was corrected. A broad Cargo invocation also exposed existing
VQ match-arm omissions in tokenizer integration tests, outside this focused
gate; those remain required before final PR qualification.

The first 8K attempt stopped before inference because the old prompt had
only 3,772 tokens. Its failed receipt is retained. The replacement fixture's
11,318-token length was checked before the fresh pair; neither runtime code
nor any passed primitive gate was rerun for that harness correction.

Before enabling VQ snapshots or prefix reuse, add distinct payload family
and index-pool regime identity. Rust currently rejects the shared GLM
payload for VQ; the native GLM restore accepts smaller-context payloads.
A dense-only 2K payload cannot supply the missing 8K index pools, and VQ
BF16 pools cannot be treated as MXQ FP16 pools. This round does not enable
those serving options.

[Evidence](evidence/glm53-vq-sparse-2026-10-16.json) records immutable
artifact/build/library identities, raw receipt hashes, errors and limits.
