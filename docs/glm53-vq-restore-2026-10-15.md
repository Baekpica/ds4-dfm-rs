# GLM VQ packed restore: bounded 2K A/B

Packed E4M3 loads and BF16 vector stores improve the initial GLM VQ eager
path on GB10. This qualifies one kernel change, not complete model serving.
[Machine-readable evidence](evidence/glm53-vq-restore-2026-10-15.json)
records the build, artifact, commands, clocks and individual runs.

| Median of two fresh processes per variant | Scalar | Packed | Change |
|---|---:|---:|---:|
| Prefill, tok/s | 65.205 | 120.175 | +84.30% |
| Decode, tok/s | 2.475 | 3.625 | +46.46% |
| Prefill, seconds | 30.44 | 16.52 | -45.74% |
| Decode, seconds | 12.93 | 8.83 | -31.72% |

Conditions: 1,984 raw code-prompt tokens, 32 forced non-EOS greedy decode
transitions, context 2,048, prefill chunk 512, one in-process VMM owner,
MTP/capture off. Run order was off/on/on/off. Both modes used the same
binary, linked cuBLAS and 32 MiB cuBLAS workspace. The user clock policy
remained 300–2,200 MHz; observed busy clocks were 2,184–2,190 MHz.

The artifact is `Baekpica/GLM-5.3-Flash-VQ-GGUF` revision
`fe2a446e774624ef8d397136d81524c61db77f65`, derived from original revision
`91b69215804f2d2036a8867f6ce44848e278dcda`. This compares identical VQ
bytes. It does not compare VQ against MXQ, Uncensored or DeepSeek throughput.

## Why this change

The original Nsight Systems trace attributed 78.79% of prefill wall time and
53.22% of decode wall time to BF16 expert restoration. CPU gaps were only
1.71% and 0.96%; synchronous CUDA API waits overlapped GPU work.

NCU on one actual selected down matrix found poor memory-sector use:
1.2 useful bytes per load sector, 4 per store sector, 72.77% L2 throughput,
5.78% compute throughput and no spills. The isolated replay used saved
22-token routing, omitted intervening GEMMs and initially duplicated the
layer codebook per matrix. These cache differences limit bandwidth inference.

The candidate loads an eight-byte codeword, converts packed E4M3 pairs,
and writes sixteen BF16 bytes. It preserves FP32 gain multiplication and
BF16 rounding. Dispatch requires E4M3, supported gate/up/down shapes and
alignment; other cases retain scalar restoration. `DS4_GLMVQ_RESTORE_VEC=0`
is the diagnostic fallback. The default is now enabled.

Alternating isolated timings over 24 actual matrices with a shared layer
codebook were 211.93–213.66 us scalar versus 87.44–89.29 us packed, about
2.4x. Registers rose from 18 to 22, with no shared/local memory or additional
GPU allocations. Both configurations fit the same SM thread limit.

## Correctness and limits

Twelve synthetic BF16 geometry/book cases, ten handoff goldens, 24 actual
matrices and misaligned-output fallback passed. All four full-model runs
had identical finite frontier logits and 32 generated IDs. Unchanged
FP32/FP8, KDA, attention and loader gates were reused.

Exactness is inexpensive for this layout change. Future arithmetic changes
use operation/workload-specific tolerance calibrated from reference
variation, logit distributions and output quality. Packing, positions,
indices and rollback plumbing remain structural correctness requirements.

Minimum system available memory was 35.37–35.48 GiB. Resident device weights
were 80.06 GiB; the graph reported approximately 0.02 GiB KV, 148.82 MiB
state and 1,183.31 MiB workspace. These are component observations, not a
measured allocation peak. OS and CUDA memory must not be counted twice.

Process wall times were 214.88–315.58 seconds, dominated by validation and
variable weight-loading/file-cache costs. The bench's first-token field
measures the first decode transition; it is not HTTP TTFT. Rounded CSV rates
yield the approximate phase times above.

Long context, capture, embedded MTP, vision, common serving combinations,
held-out quality and same-source MXQ comparison remain outside this result.
