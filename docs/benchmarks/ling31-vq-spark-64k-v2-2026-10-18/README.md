# Ling-3.1-flash VQ v2 on one DGX Spark: 2K–64K with FP8 latents, October 18

The [README sweep](../../../speed-bench/README.md) on the
[Ling-3.1-flash VQ release](https://huggingface.co/Baekpica/Ling-3.1-flash-VQ-GGUF)
with the v2 trunk part, at `709cf8df` (branch `perf/ling31-long-context`):
three fresh `ds4-bench` processes per variant. Every frontier prefills 2,048
new tokens on top of the cached prefix and generates 128 forced greedy
tokens; the session then restores the prompt snapshot before the next
frontier. Two embedded MTP drafts run to 32K only, as before. The
[prefill-round sweep](../ling31-vq-spark-64k-prefill-2026-10-18/README.md)
(`a64e323b`, v1 trunk, merged in #93) is the baseline.

![Ling-3.1-flash VQ v2 prefill and decode on one DGX Spark](throughput.png)

| Configuration (median of three run means) | Prefill tok/s | Decode tok/s | `a64e323b` |
|---|---:|---:|---:|
| MTP off, 2K–64K | 774.1 | 17.70 | 778.8 / 14.86 |
| MTP off, 2K–32K | 861.7 | 17.83 | 867.1 / 15.02 |
| Two embedded MTP drafts, 2K–32K | 861.0 | 23.58 | 859.8 / 21.15 |

Curves show the median at each frontier; shading is the observed minimum–maximum
of the three runs, not a confidence interval. [All samples](tables.md) and the
[summary](summary.json) are retained.

| Frontier | Prefill, MTP off | Decode, MTP off | Prefill, MTP 2 | Decode, MTP 2 | Decode at `a64e323b` (off / MTP 2) |
|---:|---:|---:|---:|---:|---:|
| 2K | 910.5 | 17.63 | 906.4 | 22.12 | 15.16 / 19.43 |
| 4K | 964.9 | 17.94 | 968.5 | 22.25 | 15.15 / 19.81 |
| 8K | 924.6 | 17.85 | 931.0 | 27.32 | 15.12 / 23.91 |
| 16K | 870.2 | 17.78 | 871.3 | 21.03 | 15.03 / 18.46 |
| 32K | 768.8 | 17.73 | 772.0 | 22.00 | 14.86 / 23.45 |
| 48K | 685.9 | 17.57 | — | — | 14.70 / — |
| 64K | 625.8 | 17.44 | — | — | 14.55 / — |

## What changed

- The v2 trunk part stores the KDA decay and gate projections (f, g) as Q4_K
  instead of block-scaled FP8 and the embeddings as Q8_0 instead of BF16
  (trunk 116.4 → 114.0 GB). A decoded token reads 10.75 GB of weights
  instead of 12.6 GB.
- MLA latent rows are E4M3 with one power-of-two scale per row (641 bytes per
  token and MLA layer instead of 1,152); decode attention reads them directly.

## Conditions

- One NVIDIA DGX Spark / GB10, driver 615.71.09; CUDA 13.4, `sm_121a`;
  Rust 1.98.0; Linux 7.0.0-1019-nvidia. Clock range 300–2200 MHz; idle
  299–305 MHz before each run, 2,184 MHz in the one sample taken right after
  a run (not sampled during the sweep).
- Source `709cf8df` (clean tree), `ds4-bench` with NVTX instrumentation
  compiled in and no profiler attached, SHA256
  `ef82e649fd01f5b070ed394612e4b80ff96b7750f5e045d5c629fa71a514451b`.
- Model: the release directory with `body-nonexpert.gguf` of recipe
  `ling31-vq8-mix8r13-60r12-mtp13-q4h-q8e-q4fg-v2` and the other 70 parts of
  v1, `release.json` SHA256
  `168682418795dd28fef68c7ca3d0d7de22dcd5b308c9e35d2186f5804204afb5`. One
  model process at a time under `tools/host_memory_guard.py` (2 GiB floor,
  PSI 40%), one foreground command per run.
- Prompt `speed-bench/promessi_sposi.txt`, SHA256
  `f53e0d80cb2d4492d24ebd63c7000c397b16ae70f9bf09b3763e5d8323ec209f`.
- `--ctx-start 2048 --ctx-max 65536 --step-incr 2048 --gen-tokens 128` (MTP
  off) and `--ctx-max 32768 --mtp-draft 2`; temperature 0 with EOS excluded
  so every frontier generates 128 tokens. Default 2,048-row prefill chunk;
  no warm-up besides the first frontier.

## Reading the curves

- MTP-off decode gains 16% at 2K and 20% at 64K: Q4_K f/g remove 1.85 GB of
  every token's reads, and FP8 latents shrink the part that grows with
  context. Run means were 17.73, 16.89 and 17.70 tok/s; the second process
  ran about 5% slower at every frontier.
- Prefill is unchanged within the run spread (run means 750.4, 781.2 and
  774.1): the first run's 2K frontier took 156 tok/s, a first-use stall.
- MTP decode follows draft acceptance on each frontier's 128 generated tokens
  (20.4–30.1 tok/s). v2 changes some generated tokens, so single frontiers
  move either way (32K: 23.45 → 22.00); the mean gains 11.5%.
