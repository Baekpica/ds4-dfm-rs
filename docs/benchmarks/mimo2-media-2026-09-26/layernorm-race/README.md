# MiMo LayerNorm shared-memory race

**Confirmed correctness defect; corrected model baseline verified.** Audio
optimization qualification remains pending fresh profiling and A/B.
This report makes no performance or audio-candidate adoption claim.

## Failure and cause

Unchanged R2 proof binaries produced different logits across fresh audio
processes with identical inputs. Two new audio-0 baseline runs even matched
the candidate ON logits and tokens exactly. This demonstrated baseline
variability, without establishing its cause. [Repeat evidence](baseline-variance.json).

`mimo2_layernorm` calls `m2_block_sum` for the mean and again for variance,
reusing one shared array. The old helper returned `shared[0]` after its final
barrier. A faster warp could overwrite that location during the next call
before another warp consumed the first result.

The regression uses the actual production kernel at the measured audio
shape, 557 rows × 1024 dimensions, 256 threads and three repetitions.
Inputs use full FP32 mantissas. Before any fix, compute-sanitizer reported
the read at helper line 15 racing with the next write at line 9. The test
also failed its numerical and repeated-output checks.

| Check | Before | After |
|---|---:|---:|
| Racecheck exit | 99 | 0 |
| Reported error groups | 3 | 0 |
| Maximum absolute reference error | 0.267560244 | 4.76837158e-7 |
| Repeated outputs byte-exact | no | yes |
| Output canaries/input preserved | yes | yes |

The three baseline launches reported 3648, 3620 and 3628 hazards.
The failing exit was the expected sanitizer/test failure; no memory guard
intervened. [Before log](race-before.txt), [after log](race-after.txt),
[ordinary numeric run](numeric-after.txt), [before/after receipt](before-after.json),
[original source/binary hashes](before-receipt.json).

## Minimal repair and coverage

The helper now copies `shared[0]` into a local value, synchronizes all
threads, then returns it. Every warp finishes consuming the result before
the shared array can be reused. The reduction tree and arithmetic order
remain unchanged. The cost is one extra barrier per helper call, two per
LayerNorm; no tensor allocation is added.

Audio uses dimension 1024 with bias and epsilon 1e-5. Vision calls this
same kernel once after the tower at dimension 1280, without bias, epsilon
1e-6; its per-layer RMS norms use other kernels. An additional ordinary
test covers 31 vision rows and rechecks audio. Both pass at maximum error
4.76837158e-7 with byte-exact repeats and intact inputs/canaries.
[Expanded test output](numeric-vision.txt), [receipt](vision-result.json).
The original test source and its failing/passing receipts are preserved.

Independent review found no blocker in the uniform barrier placement or
the focused test. [Review and limits](review.json). The kernel proof does
not establish that this defect caused every prior full-model difference;
model repeatability and the audio candidate must be checked again on the
corrected baseline.

Source: [helper](../../../../cuda/mimo2_media.cuh),
[regression test and run commands](../../../../tests/mimo2_layernorm.cu).
Raw source snapshots, binaries and isolated fix patch remain under
`scratch/mimo-media-20260926/layernorm-race/`.

## Corrected full-model verification

Two fresh corrected-baseline processes per audio fixture produced byte-exact
152576-logit vectors (610304 bytes), tokens, stop IDs, request/rendered/token
bytes, and media geometry. Both pairs have max absolute/RMS logit difference
zero. All four runs stopped at token 151645: audio-3 generated 30 tokens,
audio-0 generated 42. Audio attention optimization was disabled throughout.

Audio-3 matches the pinned transcript after lowercase, punctuation and whitespace
removal. Audio-0 is repeatable but says “I too agreed” against the reference
“I to agree”; normalized reference equality is false. This character comparison
does not score word-error rate or establish general transcription quality.

One corrected screen-image proof produced 152576 finite logits with 794 prompt
tokens and 768 image tokens. It identifies Project Atlas and Queued 12 correctly,
but the fixed 64-token cap truncates before Running 7 and Failed 3. This verifies
the vision execution/input path; complete count-answer quality and image
repeatability remain unestablished by this single run.

All five payloads and memory guards exited zero. Requested guard caps were
24/21 GiB with a 12 GiB reserve; receipts retain the effective admitted limits.
The wrapper initially failed after the successful screen payload because the
frozen image harness lacks AV's `finish_reason` field. A CPU regression reproduced
that schema error; deriving completion from `stop_id` and the token cap passes.
Raw failure evidence is retained, and validation required no GPU rerun.

[Compact model results](model-result.json), [binary/input/guard receipts](model-receipt.json).
These are correctness checks, without a timing or audio-candidate adoption claim.
