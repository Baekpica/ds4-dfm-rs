# Beam-1.0 VQ on one DGX Spark: 2K–64K after the October 21 optimization

The [README sweep](../../../speed-bench/README.md) on the
[Beam-1.0-501B-A23B VQ release](https://huggingface.co/Baekpica/Beam-1.0-501B-A23B-VQ-GGUF)
(`faf8f1b0`) at `391b0bf1`, the end of the
[optimization round](../../beam-vq-optimization-2026-10-21.md): three fresh
`ds4-bench` processes per variant, interleaved. Every frontier prefills 2,048
new tokens on top of the cached prefix and generates 128 forced greedy tokens;
the session then restores the prompt snapshot before the next frontier. The
DFlash2 variant drafts with the converted
[reflection/Beam-1.0-501B-A23B-DFlash2](https://huggingface.co/reflection/Beam-1.0-501B-A23B-DFlash2)
in the release's `dflash2/` (`--mtp`, draft 7, margin 2). The dashed line is
the [same sweep at `dab8bd75`](../beam-vq-spark-64k-2026-10-21/README.md),
before this round.

![Beam-1.0 VQ prefill and decode on one DGX Spark](throughput.png)

| Configuration (median of three run means) | Prefill tok/s | Decode tok/s | At `dab8bd75` |
|---|---:|---:|---:|
| MTP off, 2K–64K | 1,241.2 | 25.23 | 830.4 / 15.86 |
| MTP off, 2K–32K | 1,326.0 | 26.03 | 881.5 / 18.70 |
| DFlash2 (draft 7, margin 2), 2K–64K | 1,225.7 | 24.73 | 821.1 / 15.41 |

Curves show the median at each frontier; shading is the observed minimum–maximum
of the three runs, not a confidence interval. [All samples](tables.md) and the
[summary](summary.json) are retained.

| Frontier | Prefill, MTP off | Decode, MTP off | Prefill, DFlash2 | Decode, DFlash2 |
|---:|---:|---:|---:|---:|
| 2K | 1,340.8 | 26.61 | 1,251.9 | 25.75 |
| 4K | 1,413.7 | 26.58 | 1,392.8 | 25.98 |
| 8K | 1,385.5 | 26.49 | 1,373.6 | 25.93 |
| 16K | 1,339.8 | 26.09 | 1,323.0 | 25.57 |
| 32K | 1,239.8 | 25.29 | 1,226.8 | 24.83 |
| 48K | 1,162.9 | 24.40 | 1,151.0 | 23.99 |
| 64K | 1,087.2 | 23.67 | 1,078.8 | 23.26 |

## Reading the numbers

- **Decode.** 2K–64K decode rose 59% (15.86 → 25.23 tok/s) and now falls 11%
  from 2K to 64K instead of 50%: the full layers' FP8 rows (1.23 GB per token
  at 64K) are read at about 229 GB/s by the split-key TMA kernel instead of
  ~27 GB/s, and the attention projections run as one launch. At 2K a token
  takes 37.6 ms against about 30 ms for its 7.24 GB of weights at 246 GB/s.
- **Prefill.** 2K–64K prefill rose 49% (830.4 → 1,241.2 tok/s): the RoPE rows
  are computed once per forward, the 12-bit experts run warp-specialized, and
  the full layers' attention takes the warp-specialized FP8 walk. Prefill
  peaks at 1,413.7 tok/s at 4K and is 23% lower at 64K, where the nine full
  layers attend over the whole prefix at 60% of the F16 peak.
- **DFlash2 on raw prose.** The drafter follows chat and code but not this
  Italian novel: decode is 2.0% lower than without it and prefill 1.3% (the
  tap projections).

## Conditions

- One NVIDIA DGX Spark / GB10, driver 615.71.09; CUDA 13.4, `sm_121a`;
  Rust 1.98.0; Linux 7.0.0-1019-nvidia. SM clock 2,184–2,190 MHz before and
  after every run (the host's 300–2,200 MHz cap).
- `ds4-bench` built from `391b0bf1` (SHA256
  `d494935adf4f01155cafe7abac158f12802b009a5cd0b42e6dccf61389d147af`).
- Model: the 51-part release directory, `release.json` SHA256
  `67ed31bfc7f3fd2b093ae6545517876b6e133e81a5d347ebea7ceada983b2e53`.
  Drafter `dflash2/Beam-1.0-501B-A23B-DFlash2-Q8_0.gguf` SHA256
  `7c7fa13e94dbf1d6c243706b636b03a9431623c5f32f69842c0c796f49092521`.
  One model process at a time under `tools/host_memory_guard.py` (3 GiB
  reserve, PSI 40%), page cache dropped before each run.
- Prompt `speed-bench/promessi_sposi.txt`, SHA256
  `f53e0d80cb2d4492d24ebd63c7000c397b16ae70f9bf09b3763e5d8323ec209f`.

```sh
./ds4-bench --cuda -m Beam-1.0-501B-A23B-VQ-GGUF --prompt-file speed-bench/promessi_sposi.txt \
  --ctx-start 2048 --ctx-max 65536 --step-incr 2048 --gen-tokens 128 --csv plain-N.csv
./ds4-bench ... --mtp Beam-1.0-501B-A23B-VQ-GGUF/dflash2/Beam-1.0-501B-A23B-DFlash2-Q8_0.gguf \
  --mtp-draft 7 --mtp-margin 2 --csv dflash2-N.csv
```
