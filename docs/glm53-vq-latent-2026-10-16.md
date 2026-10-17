# GLM VQ latent decode attention

Adopt latent attention for eligible single-row queries on the selected
nonexpert KDA Q4_K artifact. Repeated whole-model decode improves **13.01 to
17.17 tok/s (+31.98%)**. One additional 256-output comparison reduces
prefill-plus-decode time **22.54 to 18.50 s (-17.95%)**. The shorter 32-output
workload's median phase time worsens 5.02% amid prefill variation; that result
is retained below. No weight conversion or extra persistent cache is added.

The retained Q4_K profile spends 0.378 of 2.432 decode seconds (15.56%)
re-expanding the full latent KV prefix. The actual BF16 projection already
uses tensor cores: the isolated NCU capture takes 1.044 ms at 61.61% tensor
activity, 222 registers and no local spills. The candidate removes repeated
prefix expansion through a different attention factorization.

For latent cache C and per-head BF16 weights Wk/Wv, compute Qlow = R(Q Wk),
P = R(softmax(FP32(R(R(Qlow C^T) / sqrt(dim))))), then
Y = R(R(P C) Wv^T), where R rounds to BF16. The source instead rounds
expanded C Wk^T and C Wv^T. This deliberately changes rounding positions;
source-output equality is not an adoption criterion.

Only the long P C reduction disables cuBLAS output-precision split sums;
the caller's math mode is restored immediately afterward. Default cuBLAS
showed additional BF16 partial-sum rounding at 1,985 keys, which explains
an initial FP32-only intermediate error-budget failure. The change keeps
tensor cores and avoids another conversion buffer. NVIDIA documents this
[cuBLAS numerical behavior](https://docs.nvidia.com/cuda/cublas/index.html#gemm-algorithms-numerical-behavior).

The new helper is private to the native VQ path. Selection requires one
query, 64 heads, head dimension 256, latent dimension 512, 512–2,048 keys,
aligned BF16 weights/cache and existing capacity checks. Other widths,
formats and alignments retain expanded attention. DS4_GLMVQ_LATENT_ATTN=0
is the diagnostic fallback; DS4_GLMVQ_KV_BF16=0 also retains the legacy
conversion path. No host ABI, artifact bytes, persistent KV allocation,
expert precision or multi-row prefill/verification path changes. A one-row
prefill tail meeting the same prefix predicate also uses the N=1 path.
The measured 1,024/960-row prefill chunks retain expanded attention.

At 1,985 keys, logical projection-plus-attention work falls from 66.736 to
0.294 GFLOP per DSA layer. This is not a hardware-counter FLOP count.
The isolated complete target measures 1.696 to 0.317 ms (5.34x); at 512 keys,
0.540 to 0.278 ms (1.94x). Shorter prefixes lose in the prototype and retain
the old path. Timing rotates 11 copies of actual layer-3 KV_B weights;
query/cache inputs are synthetic and hot, with no intervening model kernels.
It is a target screen, not the whole-model speed result.

The new arithmetic passes independent FP64 per-operation error budgets,
including masked future keys. These budgets check the declared computation,
not closeness to the source model. Source-output relative RMS is 0.00365 at
1,985 keys. The native adapter matches the checked prototype and proves
unchanged cache bytes/positions, short-prefix, unaligned and width-2 fallback,
and restored math policy. Compute Sanitizer reports zero errors and leaks.
An initial native fixture failed because its test-only scratch omitted the
legacy fallback's larger linear workspace; the corrected fixture uses the
existing max(attention, linear) quote. Production code was unchanged.

## Whole-workload measurements

Same final executable and Q4 artifact; fresh processes, inherited OS file cache,
context 2,048, chunk 1,024, forced non-EOS greedy, MTP/capture off.
Busy clocks 2,184–2,190 MHz; configured range 300–2,200 MHz.

| 1,984 + 32 | Prefill tok/s | Decode tok/s | Phase sum, s |
|---|---:|---:|---:|
| latent-off-1 | 386.97 | 12.81 | 7.625 |
| latent-on-1 | 426.14 | 17.13 | 6.524 |
| latent-on-2 | 281.54 | 17.21 | 8.906 |
| latent-off-2 | 427.07 | 13.21 | 7.068 |

Median decode improves 13.01 to 17.17 tok/s (+31.98%). The short-workload
median phase sum worsens 7.347 to 7.715 s (+5.02%) because one candidate
prefill is slow. Its unprofiled cause is not established. Prefill uses the
same 22 expanded DSA calls and 33,356 MMA work items in every arm.
No intrinsic prefill improvement or short-total speedup is claimed.

One additional matched pair was declared after this variation, using 1,536
input tokens and 256 outputs to assess whole latency for longer generation.
The earlier result remains part of the evidence.

| 1,536 + 256 | Prefill tok/s | Decode tok/s | Phase sum, s |
|---|---:|---:|---:|
| latent-long-off | 399.67 | 13.69 | 22.543 |
| latent-long-on | 403.21 | 17.43 | 18.497 |

Long-generation phase time falls 17.95%; decode improves 27.32%. This is
one longer-output pair, alongside the two repeated short-run decode gains.
All 32/256 generated IDs agree in the corresponding workloads and prefill
frontier logits are exact. This is not sufficient quality evidence by itself.

Phase sums are inferred from rounded rates. They exclude model loading and
proof-file output; they are not HTTP wall time or TTFT. Startup variation is
recorded separately in every receipt. No startup improvement is claimed.

## Changed-arithmetic output assessment

Two fresh processes use the same current quality executable and actual
system CUDA/cuBLAS mappings, with four frozen 1,730–1,749-token prompts.
The English and Korean facts, corrected Python coupon function and JSON
arguments remain correct, and every answer reaches EOS. Three answers are
identical; the Korean candidate omits one comma. The code AST and JSON
keys/values match their expected results.

Full-vocabulary logits remain finite at 76 teacher-forced target positions;
72 follow the changed single-row path. Mean target NLL changes -0.001611 nat.
Per-case mean KL is 0.000257–0.000998. Maximum individual logit changes reach
4.133, illustrating why raw maximum error or exact tokens alone are not the
acceptance criterion. These results show no material regression in this
small set; they do not establish a broad quality improvement.

The cases use the existing forced-thinking-off template and are synthetic.
They are not live tool, multi-turn, image or source reasoning-mode tests.
The earlier Q4 quantization tradeoff remains accepted by the user; it was
not reevaluated as a blocker. Existing packing, Q4, KDA and VQ primitive
proofs were reused. The new arithmetic received only its affected checks.

## Retained profile and remaining costs

The final Nsight run records 420.58 prefill / 16.60 decode tok/s. Its decode
NVTX span is 1.928 s. Full-prefix KV expansion disappears from decode:
352 calls become zero. Combined DSA projection/attention GPU time falls
from 0.643239 to 0.097557 s (-84.83%). The old projection is counted under
dense work; the new absorbed projections are inside attention, so these
regions are combined before comparison. The new profile's frontier and
32 generated IDs match the same-binary plain candidate.

| Retained decode region | Seconds | Decode share |
|---|---:|---:|
| Dense/shared/control projections | 0.900 | 46.67% |
| Routed VQ | 0.319 | 16.53% |
| Output head | 0.233 | 12.10% |
| mHC/norm | 0.175 | 9.08% |
| Attention/KDA core, including absorbed projections | 0.132 | 6.85% |
| Router selection | 0.075 | 3.90% |

The largest remaining projection kernels are cooperative Q4 at 0.434276 s
(22.52% of decode) and direct FP8 at 0.354676 s (18.39%). These are current
measured costs, not a claim that either needs a particular replacement.
GPU idle is 0.082802 s (4.29%), an upper bound on host gaps rather than an
isolated CPU dispatch measurement. CUDA API durations overlap GPU work.

Prefill GPU time is 4.125116 s versus 4.133220 s in the prior retained trace;
KDA is 1.100890 s and routed VQ is 1.512711 s. The 71 initial cudaMalloc calls
take 0.058952 s here. This trace does not explain the earlier unprofiled slow
candidate run. No intrinsic prefill gain or resolved allocation-stall cause
is claimed.

Peak DEVICE allocation is **76.508 GiB**, exactly unchanged from the retained
baseline, and the allocation ledger closes with no live blocks. Minimum
system available memory is 39.205 GiB. This is unified memory, not separate
RAM plus VRAM. Guards retain 104/100 GiB max/high, 12 GiB reserve/trip, no
scope swap and a 1,800 s deadline; PSI is recorded without a separate trip.

## Provenance and limits

Artifact: `GLM-5.3-Flash-VQ-Q4K-KDA-GGUF`, manifest
`6ea4054b04cfb1bd2e26db318e022a5918b49c8002d7d5e1cf17e502a074a044`.
Only the previously selected 136 nonexpert KDA Q/K/V/O matrices are Q4_K.
Routed/shared experts, codebooks and KV_B weights retain their bytes.
The same original revision is used throughout; DeepSeek and Uncensored
throughput are not controls.

Benchmark SHA256:
`1cbdffcfe2e82655d754a86641c6e776ee509221d057751679dd0f5c6e5db862`.
Quality SHA256:
`a82c4e76b64fdb19261e85c03e3f7b9a3186bc3a0ce9724d3acc8121a9a41e80`.
Build provenance is `e6a35352` plus the recorded candidate patch. A test-only
scratch correction was relinked separately; the production executables did
not change. After measurements, only the source-comment citation range was
corrected. No rebuild or GPU gate was repeated for that comment or this report.

The [evidence](evidence/glm53-vq-latent-2026-10-16.json) contains commands,
source hashes, actual library maps, clocks, process wall times, intermediate
failures, checked probe sources, all plain comparisons, output assessment
and profile attribution. The actual layer-3 BF16 matrix has SHA256
`f0540a2685b23735442058523c7fca84e6e58096b06db07fa7d2725616d8c7de`;
its extraction is described in the [prior report](glm53-vq-kv-bf16-2026-10-16.md).
The native adapter fixture runs as
`tests/test_glm_vq_kv KV_B.bf16 latent-native-fixture.bin`; the checked probe
sources in the evidence generate the second file. Omitting it selects the
previous handoff fixture with the new arithmetic explicitly disabled.

Qualification remains **2K eager text**. The single-row predicate also
covers a one-row prefill tail; all measured prefill chunks are multi-row.
8K/indexer, GLM MTP, capture, vision, live tools/multi-turn, banks, snapshots
and disk KV remain open. The short-workload phase regression and narrow
quality scope must accompany any performance claim.
