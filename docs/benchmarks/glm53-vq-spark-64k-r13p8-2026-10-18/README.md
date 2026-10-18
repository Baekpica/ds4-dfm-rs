# GLM-5.3 Flash VQ R13+8 on one DGX Spark: 2K–64K, October 18

The [README sweep](../../../speed-bench/README.md) on the R13+8 release of
[GLM-5.3-Flash-VQ-GGUF](https://huggingface.co/Baekpica/GLM-5.3-Flash-VQ-GGUF)
(`35d3833d`: routed layers 3–16 add an 8-bit residual stage, the other 28 stay
R13, same Q4_K nonexperts) against the previous R13 release (tag `q4k-v3`),
both with the same `ds4-bench` binary at `90517b6b`. Three fresh processes per
release, interleaved, two embedded MTP drafts. Every frontier prefills 2,048 new
tokens on top of the cached prefix and generates 128 forced greedy tokens; the
session then restores the prompt snapshot before the next frontier.

![GLM-5.3 Flash VQ R13+8 vs R13 on one DGX Spark](throughput.png)

| Release | Prefill tok/s | Decode tok/s | Tokens per verify | Verify cycle |
|---|---:|---:|---:|---:|
| R13+8, layers 3–16 (median of three run means) | 1,113.7 | 31.81 | 2.226 | 70.0 ms |
| R13, previous release | 1,175.4 | 35.00 | 2.321 | 66.3 ms |
| Change | −5.2% | −9.1% | −4.1% | +5.5% |

Curves show the median at each frontier; shading is the observed minimum–maximum
of the three runs, not a confidence interval. Summary cells are the median of
the run means over 32 frontiers. [All samples](tables.md) and the
[summary](summary.json) are retained. Tokens per verify are 32 × 128 × 42
divided by the routed decode calls; a verify cycle is tokens per verify over
decode tok/s.

## Where the difference comes from

- **Decode bytes.** The residual stage adds one bit per weight to 14 of 42
  routed layers: 7.85 instead of 7.5 GB per token. The R13+8 decode and verify
  expert kernels stream at the R13 kernels' rate (isolated: 205–213 vs
  198–217 GB/s; NCU: ~78% long-scoreboard stalls, one 32-warp CTA per SM
  beside the 64 KiB book), so a verify cycle takes 5.5% longer and one-token
  decode without MTP 2.9–4.5% (8K prompt, two fresh runs each: 24.40 vs 25.14
  tok/s at this commit). L2 prefetch of the next row and loading the next
  task's first block under the current one did not raise them.
- **Draft acceptance.** The two releases generate different greedy
  continuations; on these the R13+8 model accepts fewer drafts (2.226 vs 2.321
  tokens per verify, identical in every run). On an 8K prompt with 128 tokens
  the R13+8 release accepted more (2,058 vs 2,184 verify calls), so this part
  depends on the text.
- **Prefill MMA.** R13+8 layers issue a second `m16n8k32.e4m3` MMA per k32
  step for the residual codewords. With 64-token tiles (residual codes in
  registers, `80b2b181`) such a layer's experts take 1.35× the R13 time for
  twice the MMAs; the earlier 32-token staged tiles took 1.9×. Incremental
  prefill falls 5.2%; a whole 8K prompt 6.2% (1,225.7 vs 1,307.4 tok/s), which
  also carries the first chunk's workspace `cudaMalloc` calls (114 vs 29 ms
  with 11.8 GiB less free memory).

## Conditions

- One NVIDIA DGX Spark / GB10, driver 615.71.09; CUDA 13.4.92, `sm_121a`;
  Linux 7.0.0-1019-nvidia. Clock range 300–2200 MHz; observed busy SM clock
  2,177–2,190 MHz. Memory PSI full avg10 at most 3.6 (R13+8) and 2.0 (R13).
- Source `90517b6b` (clean tree), `ds4-bench` with NVTX instrumentation
  compiled in and no profiler attached, SHA256
  `4e931a8fbc6db422e789d24d61a1ad2b0737237b4768e0af776385ac237ee317`.
- Models: R13+8 `GLM-5.3-Flash-VQ-GGUF` at HF `35d3833d` (manifest
  `fba93aed…`); R13 at tag `q4k-v3` (manifest `18b997f2…`, the artifact of
  the [October 17 sweep](../glm53-vq-spark-64k-prefill-2026-10-17/README.md)).
  One resident model owner; no other GPU work.
- Prompt `speed-bench/promessi_sposi.txt`, SHA256
  `f53e0d80cb2d4492d24ebd63c7000c397b16ae70f9bf09b3763e5d8323ec209f`.
- `--ctx-start 2048 --ctx-max 65536 --step-incr 2048 --gen-tokens 128
  --mtp-draft 2 --mtp-margin 0`, `DS4_GLMVQ_MTP=1`, temperature 0 with EOS
  excluded so every frontier generates 128 tokens. Default prefill chunk.
- The R13 runs reproduce the October 17 sweep (1,163.5 / 35.08, 2.32 tokens
  per verify) at 1,175.4 / 35.00.
