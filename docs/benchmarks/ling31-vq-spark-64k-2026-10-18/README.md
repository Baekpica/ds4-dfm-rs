# Ling-3.1-flash VQ on one DGX Spark: 2K–64K, October 18

The [README sweep](../../../speed-bench/README.md) on the
[Ling-3.1-flash VQ release](https://huggingface.co/Baekpica/Ling-3.1-flash-VQ-GGUF)
at `4d6f096b` (branch `feat/add-ling31-flash`): three fresh `ds4-bench`
processes per variant. Every frontier prefills 2,048 new tokens on top of the
cached prefix and generates 128 forced greedy tokens; the session then
restores the prompt snapshot before the next frontier. Two embedded MTP
drafts run to 32K only: the 1.68 GiB MTP part, a 64K session and the bench's
host-memory snapshot (about 1 GiB at 64K) would leave under 2 GiB of the
128 GB Spark.

![Ling-3.1-flash VQ prefill and decode on one DGX Spark](throughput.png)

| Configuration (median of three run means) | Prefill tok/s | Decode tok/s |
|---|---:|---:|
| MTP off, 2K–64K | 669.6 | 15.24 |
| MTP off, 2K–32K | 772.1 | 15.42 |
| Two embedded MTP drafts, 2K–32K | 751.6 | 20.99 |

Curves show the median at each frontier; shading is the observed minimum–maximum
of the three runs, not a confidence interval. [All samples](tables.md) and the
[summary](summary.json) are retained.

| Frontier | Prefill, MTP off | Decode, MTP off | Prefill, MTP 2 | Decode, MTP 2 |
|---:|---:|---:|---:|---:|
| 2K | 778.0 | 15.57 | 428.2 | 16.65 |
| 4K | 908.1 | 15.56 | 909.5 | 19.73 |
| 8K | 863.5 | 15.53 | 857.7 | 22.14 |
| 16K | 782.5 | 15.44 | 780.3 | 25.31 |
| 32K | 657.2 | 15.27 | 656.2 | 23.09 |
| 48K | 567.6 | 15.09 | — | — |
| 64K | 501.2 | 14.93 | — | — |

## Conditions

- One NVIDIA DGX Spark / GB10, driver 615.71.09; CUDA 13.4, `sm_121a`;
  Rust 1.98.0; Linux 7.0.0-1019-nvidia. Clock range 300–2200 MHz; busy SM
  clock 2,184–2,190 MHz in this session's A/B runs (not sampled during the
  sweep).
- Source `4d6f096b` (clean tree), `ds4-bench` with NVTX instrumentation
  compiled in and no profiler attached, SHA256
  `ef038d3fbd264ef8d346f1bf07a4d647d7b374239ea13056ffe21f4aa3929208`.
- Model `Ling-3.1-flash-VQ-GGUF` (release directory, recipe
  `ling31-vq8-mix8r13-60r12-mtp13-q4h-v1`), unchanged. One model process at a
  time under `tools/host_memory_guard.py` (2 GiB floor, PSI 40%).
- Prompt `speed-bench/promessi_sposi.txt`, SHA256
  `f53e0d80cb2d4492d24ebd63c7000c397b16ae70f9bf09b3763e5d8323ec209f`.
- `--ctx-start 2048 --ctx-max 65536 --step-incr 2048 --gen-tokens 128` (MTP
  off) and `--ctx-max 32768 --mtp-draft 2`; temperature 0 with EOS excluded
  so every frontier generates 128 tokens. Default 2,048-row prefill chunk;
  no warm-up besides the first frontier.
- The first MTP-off run ran in a background batch that the session's
  memory-pressure monitor stopped during the next run's load (about 3 GiB
  available); the other runs ran one per foreground command. One MTP-off
  attempt was stopped by the guard during the weight copy (PSI 42% after a
  compile had shrunk the driver's freed-page pool) and repeated.

## Reading the curves

- Incremental prefill falls 45% from 4K (908) to 64K (501). All nine MLA
  layers attend every cached key; prefill expands each 2,048-key segment of
  the latent cache into per-head K/V and merges segments, so a 2,048-token
  step takes 4.1 s at 64K against 2.3 s at 4K. The 2K point is
  each process's first prefill (first-use allocation; MTP also allocates its
  journal and draft rows).
- MTP-off decode stays within 15.6–14.9 tok/s from 2K to 64K: decode and
  verify attention read every latent key once on tensor cores. The previous
  build (`1c1c6981`, FP32 decode attention) measured 13.90 at 32K and 12.46 at
  64K on the same protocol.
- MTP decode follows draft acceptance on each frontier's 128 generated tokens
  (16.6–26.6 tok/s, +36% over MTP off on 2K–32K). Drafting is greedy only;
  thinking requests, which the server samples, decode without drafts.
