# Evaluation: DeepSeek-V4.1-Flash-1017 hVQ

Mean per-layer median routed-output error 0.2876 → 0.1885 (-34.5%); generated-position KL 0.2899 → 0.2408 (-17.0%). Generated-position NLL 1.3466 → 1.3589 (+0.9%).
On DGX Spark with the released GGUF: generated-position KL 0.2963 → 0.2467 (-16.7%), same top-1 as the original 81.87% → 84.02%, NLL 1.3484 → 1.3603; the serving checklist passes ([below](#on-dgx-spark-gb10)).

## Build identity and verification

This record covers the November 11, 2026 build on one NVIDIA H200 (143,771 MiB),
CUDA 13.2 and PyTorch 2.13, with `OMP_NUM_THREADS=4`. The host had a 250,999,996,416-byte
cgroup memory limit and a 10.2-core CPU quota. Conversion timings describe this
single run; GPU clocks were not recorded and no serving-speed claim is made.

Tools commit: `d2747412ac0ec04451c6f6a2dc57f902bdc624d7`. Calibration run:
`402c1be749e2c36a2de4305d20cfeea177cfb8a5f7edb42dafbe0a7689491fc0`. Assembled GGUF SHA-256:
`d000e79199acd3e443fa6485e4927a0d5f207fa860d0b48f05bbc3dd514f52df`. The release has 40 byte-range parts with a total size
of 113,556,639,424 bytes. See the [model and checksums](https://huggingface.co/Baekpica/DeepSeek-V4.1-Flash-1017-hVQ-GGUF) and the
[machine-readable record](evidence/deepseek41-hvq-2026-11-11.json).

| Check | Result |
|---|---|
| `tests/test_ds41_hvq.py` | All eight groups passed: t=0 identity, FP64 GPTQ reference, correlated-input feedback, gain optimum, rank handling, panel sizes, moment collection and 12/13-bit round-trip |
| Real layer 14 at t=0 | Final hVQ path, iVQ converter rerun and published iVQ blob are byte-identical; SHA-256 `ef03949a92600a6716e03a27fcf725b20130d62416b14c4e7fd62d4aeb68a503` |
| Correlated synthetic inputs | At rho=0.9, output error is 0.25× the t=0 baseline on calibration inputs and 0.26× on fresh inputs |
| `tests/test_run_guards.py` | Calibration identity, resume and assembly variant guards passed |
| `tests/test_score_regions_ds41.py` | Domain totals, generated/prompt masks, NLL and input-alignment checks passed |
| `convert_layer_ds41_hvq.py --verify --layers 0-39` | All 40 blobs match receipt SHA-256 and the device/strict-reader decoders |
| iVQ codebooks and fixed blob fields | All 40 layers byte-identical, including slot tables and fixed payload bytes |
| Assembled GGUF | All 8,359,513,152 bytes outside the expert blobs match the VQ base |
| Unmodified `ds41_inspect` | `validate: ok`, `layout: ok`, `bind: slots=1000 bound=1000 required-missing=0` |
| Pinned inputs | All 83 original-checkpoint files and 52 iVQ files match their pinned content identities |
| `ci-lint` | Passed (exit 0, 75.4 s) |
| `ci-rust` | Failed: unchanged timestamp-sensitive ds4-perf test; see below |
| `ci-rust-other-tests` | Failed: unchanged timestamp-sensitive ds4-perf test; see below |

The test commands run from `tools/vq_calib/`; Make targets run from the repository
root. In this root container, the initial `ci-rust` run failed the unchanged KV
test that expects unlinking in a mode-0555 directory to fail. Dropping
`CAP_DAC_OVERRIDE` for the test subprocess made that exact test pass. The subsequent
workspace run reached a separate failure in
`ds4-perf::artifact::tests::verification_rejects_drift_and_conflicting_digest`:
rapid same-size writes can retain the same mtime and ctime on this filesystem
(88 of 100 writes in a local `/tmp` probe). Both failing modules are byte-identical
to `main`. The remaining workspace and family checks were still running when this record was prepared.
The complete `ci-rust` gate is therefore **not recorded as passed**.
No runtime or test source was changed for these environment findings.
The full conversion and evaluation commands are in the [release recipe](https://huggingface.co/Baekpica/DeepSeek-V4.1-Flash-1017-hVQ-GGUF/blob/main/RECIPE.md).
The loader check establishes metadata/layout/binding compatibility; it does not
establish a DGX Spark GPU-forward or serving qualification; that is recorded
[below](#on-dgx-spark-gb10). The repository's
older official-continuation scorer for DeepSeek V4/Metal was not run; this record
uses the requested V4.1-1017 reference-forward comparisons below.

## Measurements

These tables use the same 26 held-out cases (58,803 tokens) for VQ, iVQ and hVQ
on one H200. Next-token scores cover 58,777 positions, excluding the final token
of each case. Generated positions are raw-text tokens and assistant turns in
chat cases. The numerical floor runs the same original model in two 13-case
batches instead of one 26-case batch. There are 44,157 generated-position
targets; the long-finance case is an input-only chat context and contributes
no generated-position targets.

Group rows average the individual layers' statistics. The per-token median
and the energy-weighted ratio are different summaries: the former gives each
token one observation, while the latter weights large routed outputs more.

The reference forward replaces only the routed experts. Its other weights are
the original FP4/FP8 checkpoint, so these scores do not measure the GGUF's q4_K
body, MTP or sampled serving behavior. Layer-local errors use the original's
inputs and routes; end-to-end scores allow each converted model's hidden states
and routing to diverge. Lower routed error and KL mean closer agreement with the
original; NLL measures the actual held-out text and need not move with KL.

The same held-out set selected t, damping and shrink on four pilot layers.
There is no independent second test set. See [release recipe](https://huggingface.co/Baekpica/DeepSeek-V4.1-Flash-1017-hVQ-GGUF/blob/main/RECIPE.md) for the
settings and provenance, and [model card](https://huggingface.co/Baekpica/DeepSeek-V4.1-Flash-1017-hVQ-GGUF) for the interpretation.

### Per-layer routed-output error (teacher inputs and routes)

| MoE layers | Per token (median): VQ → iVQ → hVQ | hVQ vs iVQ | Layers hVQ < iVQ | Energy-weighted: VQ → iVQ → hVQ | hVQ vs iVQ |
|---|---:|---:|---:|---:|---:|
| 0–13 (13-bit) | 0.239 → 0.215 → 0.115 | -46.4% | 14 of 14 | 0.220 → 0.164 → 0.074 | -55.0% |
| 14–39 (12-bit) | 0.347 → 0.327 → 0.228 | -30.2% | 26 of 26 | 0.345 → 0.075 → 0.014 | -81.7% |
| All 40 | 0.309 → 0.288 → 0.189 | -34.5% | 40 of 40 | 0.301 → 0.106 → 0.035 | -67.3% |

| Layer | VQ median | iVQ median | hVQ median | hVQ / iVQ | VQ energy-weighted | iVQ energy-weighted | hVQ energy-weighted |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.2007 | 0.1423 | 0.0298 | 0.209 | 0.1916 | 0.1150 | 0.0189 |
| 1 | 0.2185 | 0.1786 | 0.0686 | 0.384 | 0.1729 | 0.0701 | 0.0064 |
| 2 | 0.2225 | 0.1895 | 0.0747 | 0.394 | 0.2002 | 0.1243 | 0.0354 |
| 3 | 0.2231 | 0.1936 | 0.0793 | 0.410 | 0.2150 | 0.1540 | 0.0551 |
| 4 | 0.2286 | 0.2047 | 0.0945 | 0.462 | 0.2224 | 0.1924 | 0.0829 |
| 5 | 0.2327 | 0.2106 | 0.1049 | 0.498 | 0.2264 | 0.1994 | 0.0946 |
| 6 | 0.2367 | 0.2190 | 0.1202 | 0.549 | 0.2288 | 0.2018 | 0.1048 |
| 7 | 0.2452 | 0.2305 | 0.1354 | 0.587 | 0.2240 | 0.1782 | 0.0922 |
| 8 | 0.2495 | 0.2335 | 0.1391 | 0.596 | 0.2094 | 0.0999 | 0.0159 |
| 9 | 0.2529 | 0.2386 | 0.1518 | 0.636 | 0.2502 | 0.2248 | 0.1280 |
| 10 | 0.2567 | 0.2432 | 0.1561 | 0.642 | 0.2021 | 0.0686 | 0.0106 |
| 11 | 0.2612 | 0.2466 | 0.1647 | 0.668 | 0.2573 | 0.2356 | 0.1465 |
| 12 | 0.2577 | 0.2435 | 0.1553 | 0.638 | 0.2505 | 0.2319 | 0.1409 |
| 13 | 0.2532 | 0.2361 | 0.1382 | 0.586 | 0.2323 | 0.1946 | 0.0993 |
| 14 | 0.3088 | 0.2894 | 0.1710 | 0.591 | 0.2901 | 0.1608 | 0.0338 |
| 15 | 0.3208 | 0.2929 | 0.1713 | 0.585 | 0.2400 | 0.1099 | 0.0152 |
| 16 | 0.3180 | 0.3016 | 0.1968 | 0.653 | 0.4293 | 0.1544 | 0.0414 |
| 17 | 0.3213 | 0.3052 | 0.2043 | 0.669 | 0.2835 | 0.1865 | 0.0941 |
| 18 | 0.3143 | 0.2970 | 0.1902 | 0.640 | 0.2683 | 0.1582 | 0.0696 |
| 19 | 0.2897 | 0.2589 | 0.1345 | 0.520 | 0.2563 | 0.1467 | 0.0445 |
| 20 | 0.3256 | 0.3070 | 0.1943 | 0.633 | 0.4037 | 0.0517 | 0.0011 |
| 21 | 0.3354 | 0.3165 | 0.2131 | 0.673 | 0.6069 | 0.0261 | 0.0003 |
| 22 | 0.3377 | 0.3184 | 0.2163 | 0.679 | 0.4450 | 0.0529 | 0.0019 |
| 23 | 0.3230 | 0.2994 | 0.1895 | 0.633 | 0.4279 | 0.0374 | 0.0003 |
| 24 | 0.3374 | 0.3223 | 0.2260 | 0.701 | 0.6772 | 0.0326 | 0.0012 |
| 25 | 0.3347 | 0.3208 | 0.2392 | 0.746 | 0.4621 | 0.0434 | 0.0006 |
| 26 | 0.3642 | 0.3515 | 0.2858 | 0.813 | 0.2515 | 0.0525 | 0.0024 |
| 27 | 0.3683 | 0.3561 | 0.2863 | 0.804 | 0.3532 | 0.0414 | 0.0024 |
| 28 | 0.3676 | 0.3548 | 0.2824 | 0.796 | 0.3512 | 0.0637 | 0.0009 |
| 29 | 0.3806 | 0.3679 | 0.2933 | 0.797 | 0.2960 | 0.0597 | 0.0007 |
| 30 | 0.3867 | 0.3756 | 0.3045 | 0.811 | 0.2917 | 0.0615 | 0.0049 |
| 31 | 0.3759 | 0.3614 | 0.2859 | 0.791 | 0.3688 | 0.0254 | 0.0009 |
| 32 | 0.3869 | 0.3729 | 0.2937 | 0.788 | 0.4099 | 0.0898 | 0.0072 |
| 33 | 0.3882 | 0.3731 | 0.2808 | 0.752 | 0.3320 | 0.0451 | 0.0019 |
| 34 | 0.3775 | 0.3577 | 0.2602 | 0.727 | 0.4003 | 0.0246 | 0.0012 |
| 35 | 0.3602 | 0.3336 | 0.2362 | 0.708 | 0.2861 | 0.0540 | 0.0043 |
| 36 | 0.3719 | 0.3446 | 0.2316 | 0.672 | 0.2602 | 0.0920 | 0.0090 |
| 37 | 0.3637 | 0.3370 | 0.2209 | 0.655 | 0.2719 | 0.0817 | 0.0081 |
| 38 | 0.3463 | 0.3096 | 0.1876 | 0.606 | 0.2201 | 0.0534 | 0.0045 |
| 39 | 0.3300 | 0.2687 | 0.1324 | 0.493 | 0.0915 | 0.0477 | 0.0044 |

### Tail of the per-token error (mean over the layers of each group)

| MoE layers | Mean: VQ → iVQ → hVQ | 90th percentile | 99th percentile | Tokens with hVQ worse than iVQ |
|---|---:|---:|---:|---:|
| 0–13 (13-bit) | 0.239 → 0.214 → 0.116 | 0.265 → 0.249 → 0.167 | 0.290 → 0.277 → 0.209 | 0.0% |
| 14–39 (12-bit) | 0.348 → 0.323 → 0.226 | 0.394 → 0.382 → 0.325 | 0.448 → 0.426 → 0.386 | 3.3% |
| All 40 | 0.310 → 0.285 → 0.188 | 0.349 → 0.335 → 0.270 | 0.393 → 0.374 → 0.324 | 2.2% |

### End to end (teacher-forced, 58,803 held-out tokens)

| vs the original | Floor | VQ | iVQ | hVQ | hVQ vs iVQ |
|---|---:|---:|---:|---:|---:|
| Routed-expert error per layer, per token (median, mean of 40 layers) | — | 0.309 | 0.288 | 0.189 | -34.5% |
| Routed-expert set overlap with the original (mean of layers 1–39) | — | 75.2% | 76.5% | 82.2% | +5.7 pt |
| Final hidden state cosine | — | 0.778 | 0.798 | 0.869 | +0.071 |
| KL, all positions | 0.037 | 0.370 | 0.352 | 0.273 | -22.3% |
| KL, generated positions | 0.012 | 0.293 | 0.290 | 0.241 | -17.0% |
| KL, prompt positions | 0.111 | 0.601 | 0.538 | 0.371 | -31.1% |
| Same top-1, all positions | 95.6% | 82.42% | 82.41% | 84.67% | +2.26 pt |
| Same top-1, generated positions | 96.3% | 82.39% | 82.22% | 84.39% | +2.17 pt |
| NLL, all positions (original 1.260) | 1.259 | 1.236 | 1.252 | 1.387 | +0.14 |
| NLL, generated positions (original 1.235) | 1.235 | 1.334 | 1.347 | 1.359 | +0.01 |

### By kind of text

| Held-out text | Tokens | KL: VQ → iVQ → hVQ | hVQ vs iVQ | Same top-1: VQ → iVQ → hVQ | NLL: original → VQ → iVQ → hVQ |
|---|---:|---:|---:|---:|---:|
| WikiText-2 | 4,096 | 0.497 → 0.482 → 0.419 | -13% | 78.6% → 78.5% → 80.4% | 1.12 → 1.43 → 1.41 → 1.42 |
| Web en / ko / zh / ja / es | 10,346 | 0.377 → 0.403 → 0.350 | -13% | 75.3% → 74.4% → 76.8% | 1.90 → 2.05 → 2.11 → 2.11 |
| Math web | 1,476 | 0.217 → 0.204 → 0.138 | -32% | 92.5% → 93.8% → 95.4% | 0.09 → 0.29 → 0.28 → 0.22 |
| Code files (Python, C) | 4,094 | 0.109 → 0.103 → 0.087 | -15% | 96.7% → 96.4% → 97.0% | 0.10 → 0.17 → 0.16 → 0.15 |
| Chat ko / zh / ja | 6,047 | 0.264 → 0.237 → 0.134 | -44% | 84.4% → 85.1% → 89.2% | 1.11 → 1.10 → 1.09 → 1.15 |
| Reasoning (think) | 6,142 | 0.157 → 0.139 → 0.084 | -40% | 88.2% → 88.4% → 91.6% | 0.87 → 0.81 → 0.83 → 0.88 |
| Instruction following | 1,517 | 0.323 → 0.333 → 0.326 | -2% | 79.8% → 80.5% → 81.9% | 0.98 → 0.99 → 1.01 → 1.08 |
| Tool calls / agents (DSML) | 2,659 | 1.330 → 1.015 → 0.563 | -45% | 69.8% → 71.3% → 76.5% | 2.37 → 1.26 → 1.42 → 2.61 |
| SWE agentless | 1,923 | 1.059 → 1.024 → 0.800 | -22% | 71.5% → 72.2% → 75.0% | 2.25 → 1.37 → 1.36 → 2.47 |
| Competitive programming (think) | 4,095 | 0.397 → 0.391 → 0.195 | -50% | 79.7% → 80.5% → 85.7% | 1.21 → 1.03 → 1.01 → 1.22 |
| Long 8K (web, finance) | 16,382 | 0.292 → 0.290 → 0.275 | -5% | 84.7% → 84.3% → 84.8% | 1.23 → 1.29 → 1.29 → 1.35 |

### By domain: all positions

| Domain | Tokens | Floor KL / top-1 | KL: VQ → iVQ → hVQ | Same top-1 | NLL: original → VQ → iVQ → hVQ |
|---|---:|---:|---:|---:|---:|
| agent_swe | 1,923 | 0.3144 / 86.48% | 1.0592 → 1.0244 → 0.8000 | 71.50% → 72.23% → 74.99% | 2.2545 → 1.3691 → 1.3637 → 2.4694 |
| agent_tool | 1,519 | 0.2720 / 84.86% | 1.5327 → 1.1437 → 0.6023 | 68.40% → 70.77% → 73.40% | 2.3428 → 1.1397 → 1.3068 → 2.5918 |
| chat_if | 1,517 | 0.0173 / 97.03% | 0.3234 → 0.3334 → 0.3260 | 79.76% → 80.49% → 81.94% | 0.9839 → 0.9856 → 1.0051 → 1.0836 |
| chat_ja | 1,030 | 0.0055 / 97.96% | 0.2409 → 0.2058 → 0.1015 | 87.67% → 89.22% → 91.65% | 0.8210 → 0.8674 → 0.8244 → 0.8531 |
| chat_ko | 2,980 | 0.0091 / 96.98% | 0.2722 → 0.2406 → 0.1312 | 83.93% → 84.43% → 89.97% | 1.1230 → 1.1027 → 1.0920 → 1.1650 |
| chat_zh | 2,037 | 0.0107 / 96.61% | 0.2634 → 0.2482 → 0.1536 | 83.55% → 83.85% → 86.94% | 1.2250 → 1.2154 → 1.2194 → 1.2868 |
| code_chat | 4,095 | 0.0511 / 94.90% | 0.3969 → 0.3907 → 0.1953 | 79.73% → 80.49% → 85.67% | 1.2060 → 1.0304 → 1.0146 → 1.2233 |
| code_raw | 4,094 | 0.0037 / 99.51% | 0.1091 → 0.1025 → 0.0868 | 96.73% → 96.36% → 97.04% | 0.0971 → 0.1691 → 0.1585 → 0.1517 |
| ds4_prompts | 1,140 | 0.1730 / 88.68% | 1.0610 → 0.8436 → 0.5106 | 71.67% → 72.11% → 80.61% | 2.4086 → 1.4210 → 1.5705 → 2.6300 |
| long_finance | 8,191 | 0.0195 / 98.01% | 0.1660 → 0.1606 → 0.1659 | 93.42% → 93.24% → 93.57% | 0.4669 → 0.4629 → 0.4536 → 0.5653 |
| math_raw | 1,476 | 0.0054 / 99.53% | 0.2174 → 0.2036 → 0.1376 | 92.55% → 93.83% → 95.39% | 0.0867 → 0.2923 → 0.2780 → 0.2162 |
| reason_general | 3,071 | 0.0129 / 96.84% | 0.2018 → 0.1805 → 0.1179 | 85.61% → 86.06% → 89.55% | 1.0865 → 0.9816 → 1.0053 → 1.1091 |
| reason_math | 3,071 | 0.0036 / 98.40% | 0.1127 → 0.0977 → 0.0503 | 90.75% → 90.78% → 93.65% | 0.6518 → 0.6296 → 0.6506 → 0.6597 |
| web_en | 10,238 | 0.0192 / 94.33% | 0.3721 → 0.3736 → 0.3381 | 77.07% → 76.32% → 77.38% | 1.9212 → 2.0493 → 2.0613 → 2.0653 |
| web_es | 1,070 | 0.0086 / 97.01% | 0.2571 → 0.2590 → 0.2396 | 84.02% → 82.24% → 83.46% | 0.9817 → 1.0879 → 1.0910 → 1.0871 |
| web_ja | 1,535 | 0.0213 / 92.70% | 0.5706 → 0.5916 → 0.5123 | 68.14% → 67.23% → 69.71% | 2.2217 → 2.4384 → 2.4678 → 2.4484 |
| web_ko | 3,298 | 0.0132 / 96.03% | 0.3635 → 0.3546 → 0.2989 | 79.02% → 78.71% → 82.02% | 1.4098 → 1.5418 → 1.5773 → 1.5916 |
| web_zh | 2,396 | 0.0185 / 92.78% | 0.4892 → 0.5947 → 0.5348 | 65.69% → 64.86% → 66.15% | 3.0263 → 3.1873 → 3.3469 → 3.3737 |
| wikitext2 | 4,096 | 0.0235 / 95.26% | 0.4971 → 0.4816 → 0.4185 | 78.64% → 78.54% → 80.44% | 1.1183 → 1.4282 → 1.4075 → 1.4185 |

### By domain: generated positions

| Domain | Tokens | Floor KL / top-1 | KL: VQ → iVQ → hVQ | Same top-1 | NLL: original → VQ → iVQ → hVQ |
|---|---:|---:|---:|---:|---:|
| agent_swe | 260 | 0.0096 / 96.15% | 0.1645 → 0.1381 → 0.0889 | 84.23% → 86.54% → 89.62% | 0.9671 → 0.8711 → 0.7948 → 1.0222 |
| agent_tool | 365 | 0.0078 / 96.99% | 0.0925 → 0.0861 → 0.0389 | 90.68% → 91.23% → 92.33% | 0.5526 → 0.4921 → 0.5210 → 0.5873 |
| chat_if | 1,477 | 0.0082 / 97.22% | 0.3152 → 0.3263 → 0.3280 | 80.23% → 80.97% → 82.06% | 0.9258 → 0.9317 → 0.9494 → 1.0294 |
| chat_ja | 918 | 0.0038 / 98.37% | 0.1789 → 0.1290 → 0.0740 | 88.89% → 90.31% → 92.92% | 0.7240 → 0.7720 → 0.7353 → 0.7873 |
| chat_ko | 2,146 | 0.0038 / 98.00% | 0.1824 → 0.1489 → 0.0691 | 86.81% → 87.42% → 92.36% | 0.8763 → 0.8959 → 0.8780 → 0.9023 |
| chat_zh | 1,763 | 0.0055 / 96.99% | 0.1805 → 0.1710 → 0.1055 | 85.31% → 85.14% → 88.15% | 1.0555 → 1.1071 → 1.1019 → 1.1407 |
| code_chat | 3,038 | 0.0058 / 97.37% | 0.2066 → 0.1746 → 0.0940 | 83.97% → 84.73% → 89.07% | 0.8788 → 0.8805 → 0.8772 → 0.8670 |
| code_raw | 4,094 | 0.0037 / 99.51% | 0.1091 → 0.1025 → 0.0868 | 96.73% → 96.36% → 97.04% | 0.0971 → 0.1691 → 0.1585 → 0.1517 |
| ds4_prompts | 274 | 0.0205 / 96.35% | 0.2737 → 0.2324 → 0.0965 | 86.50% → 85.77% → 92.34% | 1.5403 → 1.2556 → 1.3217 → 1.4611 |
| long_finance | 0 | — | — | — | — |
| math_raw | 1,476 | 0.0054 / 99.53% | 0.2174 → 0.2036 → 0.1376 | 92.55% → 93.83% → 95.39% | 0.0867 → 0.2923 → 0.2780 → 0.2162 |
| reason_general | 2,765 | 0.0060 / 97.40% | 0.1527 → 0.1387 → 0.0815 | 87.45% → 87.81% → 91.14% | 0.8471 → 0.7982 → 0.8133 → 0.8600 |
| reason_math | 2,948 | 0.0025 / 98.47% | 0.0947 → 0.0813 → 0.0431 | 91.42% → 91.15% → 93.79% | 0.5892 → 0.5832 → 0.6035 → 0.6016 |
| web_en | 10,238 | 0.0192 / 94.33% | 0.3721 → 0.3736 → 0.3381 | 77.07% → 76.32% → 77.38% | 1.9212 → 2.0493 → 2.0613 → 2.0653 |
| web_es | 1,070 | 0.0086 / 97.01% | 0.2571 → 0.2590 → 0.2396 | 84.02% → 82.24% → 83.46% | 0.9817 → 1.0879 → 1.0910 → 1.0871 |
| web_ja | 1,535 | 0.0213 / 92.70% | 0.5706 → 0.5916 → 0.5123 | 68.14% → 67.23% → 69.71% | 2.2217 → 2.4384 → 2.4678 → 2.4484 |
| web_ko | 3,298 | 0.0132 / 96.03% | 0.3635 → 0.3546 → 0.2989 | 79.02% → 78.71% → 82.02% | 1.4098 → 1.5418 → 1.5773 → 1.5916 |
| web_zh | 2,396 | 0.0185 / 92.78% | 0.4892 → 0.5947 → 0.5348 | 65.69% → 64.86% → 66.15% | 3.0263 → 3.1873 → 3.3469 → 3.3737 |
| wikitext2 | 4,096 | 0.0235 / 95.26% | 0.4971 → 0.4816 → 0.4185 | 78.64% → 78.54% → 80.44% | 1.1183 → 1.4282 → 1.4075 → 1.4185 |

A zero-token row has no generated-position targets; its metrics are undefined, not zero. The long-finance case is an input-only chat context.

### Per-case routed error (median over tokens and layers)

| Case | Tokens | VQ | iVQ | hVQ |
|---|---:|---:|---:|---:|
| wikitext2_a | 2049 | 0.316 | 0.300 | 0.221 |
| wikitext2_b | 2049 | 0.320 | 0.307 | 0.233 |
| en_web | 2048 | 0.313 | 0.293 | 0.195 |
| ko_web_a | 1252 | 0.321 | 0.300 | 0.203 |
| ko_web_b | 2048 | 0.312 | 0.293 | 0.196 |
| zh_web_a | 1166 | 0.325 | 0.310 | 0.223 |
| zh_web_b | 1232 | 0.322 | 0.312 | 0.237 |
| ja_web | 1536 | 0.317 | 0.301 | 0.222 |
| es_web | 1071 | 0.310 | 0.291 | 0.203 |
| math_web | 1477 | 0.312 | 0.295 | 0.210 |
| code_python | 2048 | 0.305 | 0.282 | 0.190 |
| code_c | 2048 | 0.305 | 0.273 | 0.174 |
| ko_chat_nothink | 963 | 0.315 | 0.286 | 0.174 |
| ko_math_think | 2019 | 0.303 | 0.263 | 0.124 |
| en_reason_think | 3072 | 0.304 | 0.273 | 0.157 |
| math_think | 3072 | 0.300 | 0.258 | 0.123 |
| if_chat_nothink | 1518 | 0.303 | 0.282 | 0.170 |
| tool_call_dsml | 1520 | 0.327 | 0.291 | 0.144 |
| swe_agentless | 1924 | 0.315 | 0.278 | 0.149 |
| code_chat_think | 4096 | 0.306 | 0.276 | 0.152 |
| zh_chat_think | 1005 | 0.311 | 0.284 | 0.165 |
| zh_chat_nothink | 1034 | 0.304 | 0.267 | 0.134 |
| ja_chat_nothink | 1031 | 0.301 | 0.262 | 0.132 |
| ds4_agent_dsml | 1141 | 0.332 | 0.294 | 0.115 |
| long_en_doc | 8192 | 0.322 | 0.310 | 0.231 |
| long_finance | 8192 | 0.324 | 0.287 | 0.160 |

### Weight space and calibration-set output error (receipts)

| MoE layers | Plain weight error, iVQ → hVQ | Activation-weighted (D) error, iVQ → hVQ | Output error under H_t (gate/up), iVQ → hVQ | Output error under D½RD½ (gate/up), iVQ → hVQ | (down) H_t | (down) D½RD½ | Indices changed (w1 / w3 / w2) |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0–13 (13-bit, K=8192) | 0.148 → 0.193 (+30.0%) | 0.131 → 0.190 (+44.8%) | 0.095 → 0.039 (-59.2%) | 0.095 → 0.039 (-59.2%) | 0.116 → 0.091 (-20.9%) | 0.116 → 0.091 (-20.9%) | 57% / 57% / 37% |
| 14–39 (12-bit, K=4096) | 0.173 → 0.204 (+17.7%) | 0.121 → 0.151 (+24.6%) | 0.125 → 0.066 (-47.6%) | 0.125 → 0.066 (-47.6%) | 0.141 → 0.119 (-15.8%) | 0.141 → 0.119 (-15.8%) | 48% / 48% / 32% |
| All 40 | 0.165 → 0.200 (+21.5%) | 0.125 → 0.165 (+32.0%) | 0.115 → 0.056 (-51.0%) | 0.115 → 0.056 (-51.0%) | 0.132 → 0.109 (-17.4%) | 0.132 → 0.109 (-17.4%) | 51% / 51% / 34% |

### Conversion time and peak memory per layer

| Layer | K | Collect s | Fit s (assign) | Pack/verify/write s | Total s | Peak GPU GiB collect / convert | Peak host RSS GiB | Tokens routed (min / median / max per expert) | Experts lifted |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 8192 | 78 | 105 (77) | 29 | 212 | 95.5 / 96.8 | 170 | 3,920 / 52,172 / 446,265 | 0 |
| 1 | 8192 | 92 | 104 (77) | 28 | 224 | 94.6 / 96.8 | 170 | 7,221 / 49,217 / 378,173 | 0 |
| 2 | 8192 | 92 | 104 (77) | 29 | 224 | 95.3 / 97.9 | 174 | 4,593 / 48,456 / 448,125 | 0 |
| 3 | 8192 | 86 | 104 (77) | 29 | 219 | 95.5 / 97.9 | 175 | 2,776 / 46,806 / 497,852 | 0 |
| 4 | 8192 | 86 | 104 (77) | 28 | 218 | 95.5 / 97.9 | 175 | 3,523 / 46,029 / 782,777 | 0 |
| 5 | 8192 | 86 | 104 (77) | 29 | 218 | 95.5 / 97.8 | 175 | 3,514 / 48,811 / 434,551 | 0 |
| 6 | 8192 | 86 | 104 (77) | 28 | 219 | 95.5 / 97.9 | 175 | 5,096 / 46,777 / 395,701 | 0 |
| 7 | 8192 | 86 | 104 (77) | 29 | 219 | 95.5 / 97.9 | 175 | 2,352 / 46,287 / 287,603 | 0 |
| 8 | 8192 | 91 | 104 (77) | 28 | 224 | 95.5 / 97.9 | 175 | 2,740 / 48,047 / 408,593 | 0 |
| 9 | 8192 | 86 | 104 (77) | 28 | 219 | 95.5 / 97.8 | 175 | 5,776 / 49,212 / 445,423 | 0 |
| 10 | 8192 | 86 | 104 (78) | 29 | 218 | 95.5 / 97.8 | 175 | 4,148 / 48,531 / 503,776 | 0 |
| 11 | 8192 | 86 | 104 (77) | 29 | 219 | 95.5 / 97.8 | 175 | 5,839 / 49,813 / 330,021 | 0 |
| 12 | 8192 | 86 | 104 (77) | 29 | 219 | 95.5 / 97.9 | 175 | 3,877 / 49,473 / 347,269 | 0 |
| 13 | 8192 | 86 | 104 (77) | 29 | 219 | 95.5 / 97.9 | 177 | 4,832 / 50,370 / 414,989 | 0 |
| 14 | 4096 | 110 | 72 (46) | 23 | 205 | 95.8 / 97.9 | 177 | 1,779 / 45,410 / 624,094 | 0 |
| 15 | 4096 | 86 | 72 (46) | 24 | 182 | 95.5 / 97.9 | 177 | 4,499 / 47,363 / 449,593 | 0 |
| 16 | 4096 | 86 | 72 (46) | 23 | 181 | 95.5 / 97.9 | 177 | 3,734 / 47,015 / 403,774 | 0 |
| 17 | 4096 | 86 | 73 (46) | 23 | 182 | 95.5 / 97.9 | 177 | 5,735 / 46,842 / 519,508 | 0 |
| 18 | 4096 | 86 | 72 (46) | 23 | 182 | 95.5 / 97.9 | 177 | 290 / 44,842 / 554,126 | 0 |
| 19 | 4096 | 86 | 72 (46) | 23 | 181 | 95.5 / 97.8 | 177 | 118 / 53,040 / 363,460 | 0 |
| 20 | 4096 | 94 | 72 (46) | 23 | 189 | 95.8 / 98.2 | 177 | 945 / 46,828 / 513,334 | 0 |
| 21 | 4096 | 88 | 72 (46) | 24 | 184 | 96.0 / 98.2 | 177 | 11 / 46,364 / 503,221 | 0 |
| 22 | 4096 | 89 | 73 (46) | 25 | 187 | 96.0 / 98.2 | 177 | 298 / 45,982 / 787,376 | 0 |
| 23 | 4096 | 89 | 72 (46) | 24 | 185 | 96.0 / 98.2 | 177 | 4,340 / 46,244 / 410,069 | 0 |
| 24 | 4096 | 93 | 72 (46) | 23 | 189 | 96.0 / 98.3 | 177 | 4 / 45,094 / 356,367 | 0 |
| 25 | 4096 | 88 | 72 (46) | 23 | 184 | 96.0 / 98.2 | 177 | 95 / 45,867 / 302,271 | 0 |
| 26 | 4096 | 89 | 72 (46) | 23 | 185 | 96.0 / 98.2 | 177 | 1,706 / 47,279 / 343,205 | 0 |
| 27 | 4096 | 88 | 72 (46) | 23 | 184 | 96.0 / 98.2 | 177 | 29 / 47,178 / 320,572 | 0 |
| 28 | 4096 | 93 | 72 (46) | 23 | 189 | 96.0 / 98.3 | 177 | 20 / 44,947 / 407,532 | 0 |
| 29 | 4096 | 88 | 72 (46) | 25 | 186 | 96.0 / 98.2 | 177 | 8,123 / 44,677 / 436,729 | 0 |
| 30 | 4096 | 93 | 73 (46) | 23 | 189 | 96.0 / 98.2 | 177 | 4,513 / 46,938 / 351,376 | 0 |
| 31 | 4096 | 90 | 72 (46) | 23 | 186 | 96.0 / 98.2 | 177 | 6,984 / 45,160 / 394,828 | 0 |
| 32 | 4096 | 94 | 72 (46) | 23 | 189 | 96.0 / 98.2 | 177 | 2,877 / 44,107 / 373,031 | 0 |
| 33 | 4096 | 89 | 72 (46) | 23 | 184 | 96.0 / 98.2 | 177 | 5,442 / 46,811 / 394,442 | 0 |
| 34 | 4096 | 89 | 73 (46) | 23 | 184 | 96.0 / 98.2 | 177 | 5,854 / 44,691 / 468,509 | 0 |
| 35 | 4096 | 89 | 72 (46) | 23 | 184 | 96.0 / 98.2 | 178 | 5,692 / 44,648 / 478,911 | 0 |
| 36 | 4096 | 94 | 72 (46) | 23 | 189 | 96.0 / 98.3 | 178 | 2,491 / 42,834 / 593,138 | 0 |
| 37 | 4096 | 89 | 72 (46) | 23 | 184 | 96.0 / 98.2 | 178 | 7,556 / 43,683 / 508,185 | 0 |
| 38 | 4096 | 89 | 72 (46) | 22 | 184 | 96.0 / 98.2 | 178 | 5,164 / 44,889 / 673,129 | 0 |
| 39 | 4096 | 89 | 72 (46) | 23 | 184 | 96.0 / 98.2 | 178 | 3,209 / 44,706 / 485,248 | 0 |

Totals: collect 59.3 min, fit 55.7 min (assign 37.9), pack/verify/write 16.7 min (beside the next layer). Peak GPU 96.0 GiB collecting, 98.3 GiB converting; peak host RSS 178 GiB.

Teacher stream vs the iVQ statistics (min / mean / max over layers): count_l1 9.88e-05 / 2.57e-03 / 3.50e-03; x2_rel 1.56e-04 / 2.12e-02 / 1.29e-01; a2_rel 9.21e-06 / 8.42e-03 / 3.57e-02; x2_profile_rel 2.29e-04 / 3.36e-02 / 1.47e-01; a2_profile_rel 5.57e-04 / 3.30e-02 / 1.48e-01

## On DGX Spark (GB10)

November 11, 2026, one DGX Spark: GB10, CUDA 13.4, driver 615.71.10,
`nvidia-clocks.service` active (300–2,200 MHz). Runtime: this branch at `13e9a49f` (engine sources of `main` at `502019da`),
`make cuda-spark`, `ds4-server` SHA-256 `7ba9c33f8493a23853573925435b64fec64f2bc09b2bcf63e84a30d359a26299`.
Artifact: the published parts, assembled to SHA-256 `d000e791…52df`; the vision
sidecar and both Engram shards are the iVQ release's files by SHA-256. Every
model run went through `tools/host_memory_guard.py` (2 GiB reserve and trip),
one model at a time.

### The full GGUF against the original (engine, teacher-forced)

The reference measurements above replace the routed experts inside the
original body. This one runs the released files: q4_K body, engine arithmetic.
The engine's score entry (`ds4_v41_score_ids`, 2,048-row blocks, the Rust
host's engram feed) wrote every position's logits for the same 26 cases; a
host tool outside the tree drove it. The original's logits are its saved
reference final states times `head.weight` (BF16, original checkpoint at
`1bae0e10`). That scoring reproduces the reference table above to every
printed digit.

| vs the original | Floor | VQ | iVQ | hVQ | hVQ vs iVQ |
|---|---:|---:|---:|---:|---:|
| KL, all positions | 0.037 | 0.379 | 0.362 | 0.277 | -23.4% |
| KL, generated positions | 0.012 | 0.302 | 0.296 | 0.247 | -16.7% |
| KL, prompt positions | 0.111 | 0.613 | 0.561 | 0.370 | -34.0% |
| Same top-1, all positions | 95.6% | 82.06% | 82.09% | 84.38% | +2.29 pt |
| Same top-1, generated positions | 96.3% | 81.96% | 81.87% | 84.02% | +2.15 pt |
| Top-1 is the text's token, generated (original 71.72%) | 71.71% | 68.95% | 68.80% | 69.47% | +0.67 pt |
| NLL, all positions (original 1.260) | 1.259 | 1.240 | 1.252 | 1.383 | +0.131 |
| NLL, generated positions (original 1.235) | 1.235 | 1.340 | 1.348 | 1.360 | +0.012 |
| Entropy, generated positions (original 0.816) | 0.818 | 1.155 | 1.077 | 0.889 | -0.188 |

The floor column is the reference forward's own (the original batched two
ways); the engine cannot run the original checkpoint. 58,777 scored
positions, 44,157 generated.

- **The released file keeps the gain.** Against the reference substitution
  the full GGUF adds 0.004–0.010 KL to each of the three; NLL moves by at
  most 0.004. The engine's top-1 is the reference forward's at 94.1%, 93.7%
  and 93.3% of positions (VQ, iVQ, hVQ).
- **Paired, hVQ minus iVQ, generated positions** (95% interval, 2,000
  bootstrap resamples of the 26 cases): KL -0.050 (-0.062 to -0.038); same
  top-1 +2.1 pt (+1.5 to +3.0); top-1 is the text's token +0.7 pt (+0.2 to
  +1.2); NLL +0.012 (-0.003 to +0.031).
- **Breadth.** Generated-position KL is lower under hVQ in all 18 domains
  with generated targets; all-position KL in 25 of the 26 cases.
- **Why NLL does not follow KL.** hVQ's distributions are nearly as sharp as
  the original's; VQ's and iVQ's are flatter (entropy row). Where the
  original's top-1 is the text's token (31,669 generated positions) the NLL
  is 0.154 for the original, 0.350 for iVQ and 0.290 for hVQ. Where it is not
  (12,488), hVQ follows the original, 4.07 against 3.98, and the flatter iVQ
  hedges, 3.88. The median generated-position NLL is 0.086, 0.222 and 0.131.
  Prompt positions repeat this at scale: the original scores 1.335 there and
  hVQ, the closer copy, 1.452 against iVQ's 0.961.

| Held-out text, generated positions | Positions | KL: VQ → iVQ → hVQ | hVQ vs iVQ | Same top-1: VQ → iVQ → hVQ | Top-1 is the text's token: original → VQ → iVQ → hVQ |
|---|---:|---:|---:|---:|---:|
| WikiText-2 | 4,096 | 0.488 → 0.480 → 0.420 | -13% | 78.8% → 78.2% → 80.4% | 74.5% → 67.8% → 68.0% → 69.0% |
| Web en / ko / zh / ja / es | 18,537 | 0.407 → 0.415 → 0.372 | -10% | 75.1% → 74.5% → 75.9% | 58.5% → 55.5% → 55.3% → 55.4% |
| Math web | 1,476 | 0.238 → 0.213 → 0.146 | -31% | 92.2% → 92.8% → 95.1% | 97.8% → 91.4% → 91.7% → 94.0% |
| Code files (Python, C) | 4,094 | 0.111 → 0.111 → 0.090 | -18% | 96.4% → 96.0% → 96.9% | 97.7% → 95.7% → 95.1% → 95.8% |
| Chat ko / zh / ja | 4,827 | 0.191 → 0.164 → 0.090 | -45% | 86.1% → 87.1% → 91.1% | 77.2% → 75.3% → 75.4% → 76.6% |
| Reasoning (think) | 5,713 | 0.131 → 0.117 → 0.069 | -41% | 89.1% → 89.4% → 92.0% | 79.4% → 79.2% → 78.4% → 79.0% |
| Instruction following | 1,477 | 0.322 → 0.326 → 0.325 | -1% | 80.2% → 80.4% → 80.8% | 74.1% → 71.0% → 70.9% → 73.1% |
| Tool calls / agents (DSML) | 639 | 0.177 → 0.156 → 0.072 | -54% | 88.9% → 88.3% → 92.3% | 80.4% → 83.7% → 82.5% → 82.0% |
| SWE agentless | 260 | 0.185 → 0.147 → 0.101 | -31% | 83.1% → 86.5% → 91.5% | 76.5% → 76.5% → 76.2% → 73.1% |
| Competitive programming (think) | 3,038 | 0.222 → 0.188 → 0.102 | -46% | 83.2% → 84.1% → 88.8% | 74.4% → 71.7% → 72.6% → 74.5% |

### Serving

`ds4 --cuda -m MODEL --engram-dir DIR --serving-check --prompt-file
tests/long_context_essay_prompt.txt` (prompt 5,615 tokens, split 2,807):

| Category | Item | Declared | Result | Detail |
|---|---|---|---|---|
| reuse | prefix | qualified | works | kept 2,807, tokens 16/16 |
| reuse | rewind | — | no | prefilled from zero |
| reuse | continue | qualified | works | kept 5,630, tokens 16/16 |
| disk | session | qualified | works | kept 5,615, tokens 16/16 |
| snapshot, banks | — | — | no | V4.1 serves one sequence without snapshots |

Result: pass (works 3, FAIL 0, WRONG 0, undeclared 0), the iVQ artifact's
result in the [November 2 record](benchmarks/serving-checklist-2026-11-02/README.md).

`ds4-server` with `--engram-dir`, `--vision-gguf`, `--mem-budget-mb 110000
--mem-floor-gb 2`: the model was resident 153 s after launch, MemAvailable
10.5 GiB after boot and 6.4 GiB at its lowest over the session (it includes a
66,608-token prefill), memory PSI full avg10 0.00 throughout. Requests sent
`temperature` 1.0 and `top_p` 1.0. One request per surface, all answered as
expected:

| Surface | Request | Result |
|---|---|---|
| `/v1/chat/completions` | Korean question, no thinking field | one-sentence answer, `stop`, no reasoning |
| thinking | `thinking: {type: enabled}`, 17 × 23 | reasoning closed, content `391`, `stop` |
| stream | SSE, usage included | 20 deltas, strict UTF-8, `[DONE]` |
| tools | one function, then its result | `tool_calls` with valid JSON arguments; second turn answers from the result |
| vision | 224 × 224 PNG, base64 | names the left half's colour |
| prefix reuse | second turn of a conversation | 19 of 27 prompt tokens cached |
| `/v1/messages`, `/v1/responses`, `/v1/completions` | one short request each | 200 with the expected text |
| long context | 24,027-token prompt, one fact planted an eighth of the way in | exact recall; prefill 1,153.9 tok/s |

A seeded set at temperature 1.0, the same seeds for both artifacts on the
same binary:

| 44 requests per artifact | iVQ | hVQ |
|---|---:|---:|
| Thinking arithmetic (en, ko): exact answer | 24 / 24 | 24 / 24 |
| Thinking closed and finished with `stop` | 24 / 24 | 24 / 24 |
| Mean completion tokens of those | 128 | 63 |
| Tool call valid (name, city, unit) | 12 / 12 | 12 / 12 |
| Chat ko / en / zh / ja, code, translation: `stop` | 8 / 8 | 8 / 8 |

One full-prefill pass on hVQ (`speed-bench/ds41_spark_sweep.sh`, attach
mode, temperature 0, 128 tokens, MTP auto, clocks 2,184–2,190 MHz): 957.6 /
35.0 tok/s prefill / decode at 2K, 1,134.6 / 32.0 at 8K, 1,156.6 / 36.4 at
16K, 1,043.4 / 32.6 at 32K and 980.0 / 34.6 at 64K, 2.10–2.25 tokens per
step. iVQ was not paired: the two files share every tensor shape and kernel.

### Limits

- One host, one pass of each measurement.
- The held-out set is the one the pilot setting was chosen on.
- The seeded set is small and its tasks are easy: it shows no gross failure
  (unclosed thinking, malformed calls, loops), not a ranking.
- Not measured: long sampled generations, task benchmarks, MTP acceptance by
  workload, more than one image.
