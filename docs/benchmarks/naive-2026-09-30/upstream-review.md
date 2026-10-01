# GB10 recipe implementation review

Reviewed [blazux/qwen3.8-Flash-DGX](https://github.com/blazux/qwen3.8-Flash-DGX/tree/bb661c4302c0a5e8b3fb72d3e5d6740462b19534)
at `bb661c43` on September 30. Its vLLM/NVFP4 results are upstream claims,
not ds4 measurements. This review covers Naive, Qwen and shared native paths.
No recipe patches or weight conversions were installed.

| Implementation point | Current ds4 path | Decision |
| --- | --- | --- |
| [Reduced draft output head](https://github.com/blazux/qwen3.8-Flash-DGX/blob/bb661c4302c0a5e8b3fb72d3e5d6740462b19534/src/patch_mtp_draft_vocab.py) | Qwen already uses low BPE IDs, special tokens and committed input IDs with a Q8 candidate-row GEMV. Target verification remains full vocabulary. | Extend the idea to other drafters only after profiling their head. Naive also scores a Markov bias; both computations and confidence must retain the proposal contract. Measure acceptance and total decode time across languages. |
| [Exact pivot and position-ordered top-k emission](https://github.com/jschmied/qwen38-flash-next-gb10/blob/e0ef69d4f5575dad00d34e05479eaf4c6547bace/patches/kernel-det/persistent_topk.cuh) | Naive has exact leaf/merge selection, signed-zero canonicalization, position tie-breaking and ascending attention IDs. Qwen uses native CUB/stream selection and ascending expansion. | A radix pivot plus ordered scan is a candidate for large histories. It can avoid repeated candidate sorting/materialization. Benchmark separately at actual history lengths; preserve masks, ties and every selected ID. |
| [Deduplicated PLE rows and persistent staging](https://github.com/blazux/qwen3.8-Flash-DGX/blob/bb661c4302c0a5e8b3fb72d3e5d6740462b19534/src/vllm_ple_mmap.py) | ds4 has bounded page caching, worker `pread`, page reuse, mapped pinned storage and batched ready-row leases. It does not use the recipe's mmap-to-pageable-copy path. | Candidate: deduplicate repeated row descriptors and expand on GPU when duplicate rates justify it. Measure CPU map/lease cost, cold page faults and warm GPU gathers; page dedup alone is already present. |
| [Anonymous staging before GPU upload](https://github.com/blazux/qwen3.8-Flash-DGX/blob/bb661c4302c0a5e8b3fb72d3e5d6740462b19534/src/patch_moe_load_clone.py) and bounded reads | Owner uploads and in-process promotion already use four pinned staging slots and bounded chunks; the owner can read through file descriptors. | Retain the existing bounded path. Audit remaining direct mmap uploads only when startup profiling identifies them. Preserve admission's page-cache accounting. |
| [FP8 dense side layers and KV](https://github.com/blazux/qwen3.8-Flash-DGX/blob/bb661c4302c0a5e8b3fb72d3e5d6740462b19534/docs/HOW-IT-WORKS.md) | The Naive artifact already quantizes dense projections/head. Its main KV contract is BF16. Qwen artifacts have their own quantization and cache contracts. | This changes weights or arithmetic. Treat it as a separate artifact/quality campaign with component memory quotes; do not infer the recipe's gain for current GGUFs. |
| [GB10 shared-memory fast-path coverage](https://github.com/blazux/qwen3.8-Flash-DGX/blob/bb661c4302c0a5e8b3fb72d3e5d6740462b19534/Dockerfile) | ds4 uses native CUDA dispatch, not the patched FLA Python gate. | Audit device limits, launch attributes and fallback counters globally. Change a gate only when its actual tensor shape fits and whole-workload evidence shows a useful gain. |

Code anchors: `qwen_mtp_vocab_feed` / `qwen4exp_graph_mtp_step` in `ds4.c`;
`naive_score_key` / `naive_topk_launch` in `cuda/naive_primitives.cuh`;
`indexer_topk_stream512_kernel` / `qwen4exp_qsa_expand_selection_kernel` /
`cuda_stage_copy_to_dev` in `ds4_cuda.cu`; `cache_worker` in `ds4_ple.c`;
batched leases in `cuda/qwen38_ple.cu`; `upload_range_chunked` in the owner.

The shared opportunities are reducing proposal-only work, avoiding redundant
intermediates, and keeping exact selection without repeated sorts. Adoption
requires a fresh whole profile, bounded target profile, scoped capability
predicate, correctness proof and matched end-to-end A/B. Existing equivalents
and unmeasured ideas do not count as new optimization rounds.

## Naive DSpark diagnostic capture

On the retained P4 baseline, one fresh 8K/64-token unprofiled run per mode
measures main-only 17.60, loaded draft off 17.41, and draft on 4.38 tok/s.
Each has a separate fresh Nsys capture. This is a diagnostic scope, not
matched repeated evidence or a new speed round. At depth six and margin
zero, 49 trials verify 342 target rows and keep 59; only 10 of 293 proposals
are accepted. Target Q6 linear operations consume 36.835% of on-mode
Decode GPU time. [Receipts](dspark-profile-evidence.json).

The recipe's reduced draft head does not address this repeated target work.
Qwen already has the compact proposal head; Naive needs acceptance and
verification work profiled before extending it. Automatic speculation
remains off. These results do not qualify draft acceleration.

## Shared transformation follow-up

The retained P4 32K capture also bounds the repeated-transformation
opportunity. Its NVTX prefill GPU kernel sum is 75.918713 seconds;
3872 `quantize_mmq_q8_1` calls consume 0.851701 seconds (1.122%).
Decode's 1.753366-second sum includes 9280 `quantize_q8_1` calls taking
0.010158 seconds (0.579%); its router takes 0.074756 seconds (4.264%).
These percentages describe GPU kernel sums, not wall-time gains.

Producer/consumer fusion or reuse of one quantized input across projections
is a shared-path candidate. Preserve each consumer's layout and arithmetic,
and key any reuse to actual input writes. The existing fused Gate/Up and
Naive SwiGLU producer already remove some repeated work. These measurements
do not establish a new fusion's gain or applicability to Qwen; another
family needs its own profile and numerical contract. Raw capture:
`scratch/naive/round-p4-32k-pair-candidate/nsys-nvtx-kernels.csv`.
