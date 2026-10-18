# GLM-5.3 Flash VQ R13+8 on 19 layers, one DGX Spark: October 19

The [README sweep](../../../speed-bench/README.md) on the 19-layer R13+8
revision of [GLM-5.3-Flash-VQ-GGUF](https://huggingface.co/Baekpica/GLM-5.3-Flash-VQ-GGUF)
(`71c73e7e`: the 8-bit residual stage covers routed layers 3–17, 22, 27, 41
and 44; the other 23 stay R13, same Q4_K nonexperts) against the 14-layer
revision (`35d3833d`, tag `r13p8-l3-16`), both with the same `ds4-bench`
binary at `94d776ac`. Three fresh processes per revision, interleaved, two
embedded MTP drafts. Every frontier prefills 2,048 new tokens on top of the
cached prefix and generates 128 forced greedy tokens; the session then
restores the prompt snapshot before the next frontier.

![GLM-5.3 Flash VQ R13+8 on 19 vs 14 layers on one DGX Spark](throughput.png)

| Revision | Prefill tok/s | Decode tok/s | Tokens per verify | Verify cycle |
|---|---:|---:|---:|---:|
| 19 layers (median of three run means) | 1,101.3 | 30.93 | 2.171 | 70.2 ms |
| 14 layers (3–16), previous | 1,119.6 | 32.02 | 2.226 | 69.5 ms |
| Change | −1.6% | −3.4% | −2.5% | +1.0% |

Curves show the median at each frontier; shading is the observed minimum–maximum
of the three runs, not a confidence interval. Summary cells are the median of
the run means over 32 frontiers. [All samples](tables.md) and the
[summary](summary.json) are retained. Tokens per verify are 32 × 128 × 42
divided by the routed decode calls; a verify cycle is tokens per verify over
decode tok/s.

## Where the difference comes from

- **Decode bytes.** Five more layers read residual indices: 7.98 instead of
  7.85 GB per token (+1.6%). A verify cycle takes 1.0% longer. Decode without
  MTP does not resolve the difference (below).
- **Draft acceptance.** The two revisions generate different greedy
  continuations; on these the 19-layer model accepts fewer drafts (2.171 vs
  2.226 tokens per verify, identical in every run). Most of the MTP decode
  difference comes from this text-dependent part.
- **Prefill MMA.** Five more layers issue the second `m16n8k32.e4m3` MMA per
  k32 step: incremental prefill −1.6%, a whole 8K prompt −2.1%.

## Whole-prompt prefill and decode without MTP

One fresh process per run ([samples](whole-prompt.csv)), the whole prompt
prefilled in the default 4,096-row chunks, then 16 or 256 greedy tokens
without MTP. Prefill averages all runs; decode the 256-token runs.

| Prompt | Runs per revision | Prefill 19 / 14 layers | Decode 19 / 14 layers |
|---|---:|---:|---:|
| 8K | 4 (decode 2) | 1,196.8 / 1,222.1 (−2.1%) | 24.20 / 24.38 |
| 32K | 3 (decode 2) | 1,230.7 / 1,236.8 (−0.5%) | 24.26 / 23.92 |

Decode samples of one revision spread up to 2.7% (8K, 14 layers: 24.70 and
24.06), more than the expected 1–1.6% from the extra bytes.

## Quality

**Held-out text.** Teacher-forced through `ds4-core` on CUDA, one token per
step, on the six held-out cases of the model card's evaluation (5,089
targets; the card adds the 106 fixture tokens): every token after the first
is a target. [Per-case results](heldout.json).

| Case | Targets | PPL 19 layers | PPL 14 layers |
|---|---:|---:|---:|
| Korean prose | 343 | 3.560 | 3.694 |
| English prose | 1,023 | 6.878 | 7.467 |
| Code | 1,021 | 4.467 | 4.567 |
| Korean chat | 1,023 | 3.206 | 3.303 |
| Reasoning chat | 656 | 1.860 | 1.908 |
| Tool calls | 1,023 | 3.867 | 3.954 |
| **All** | **5,089** | **3.895** | **4.042** |

Every case improves; top-1 accuracy 67.9% vs 67.1%. The card's reference
forward (original-precision nonexperts) measured 3.936 → 3.789 on these cases
plus the fixture: the same −3.7%, so the Q4_K nonexperts keep the gain.

**Served answers.** `ds4-server` at `94d776ac`, 32K context, two MTP drafts,
server default sampling (thinking, effort low). One request per seed
(11, 22, …, 110): an English system prompt (the first 62,000 characters of
this repo's `AGENTS.md`, `CONTRIBUTING.md`, `docs/glm53-vq.md` and
`docs/performance.md` at `94d776ac`; 16,985 prompt tokens) and a Korean
question whose answer is five short bullets, at most 1,500 tokens.

| | 19 layers | 14 layers |
|---|---:|---:|
| Finished (`stop`) | 8/10 | 6/10 |
| Repetition loop | 0/10 | 4/10 |

A loop is a 24-character string repeated four or more times in the last
1,200 characters; the 14-layer loops repeat clauses or a phrase ("디코딩
퍼프/" 145 times). The 19-layer model's two length stops are verbose without
repetition. Raw answers: [19 layers](served-ab-l19.json),
[14 layers](served-ab-l14.json).

**Runtime fixture.** `tests/glm_vq_quality` (6 cases, 106 targets): mean
target NLL 0.285 (14 layers) → 0.301; the prose and Korean cases improve,
code, JSON and math get worse (math, 23 targets: 0.564 → 0.683). Too few
tokens to rank revisions. Fast path vs `DS4_GLMVQ_DECODE=0
DS4_GLMVQ_MMA=0`: KL 0.0035, top-1 106/106 (14 layers: 0.0038). The greedy
"대한민국의 수도는…" prompt ends normally in both modes on both revisions.

## Memory

`ds4-server --check-config` at the native 1,048,576-token context with two
MTP drafts quotes 116.2 GiB (weights 88.0, bank 12.8, scratch 10.7 for
4,096 prefill rows, MTP 0.7, floor 4) of 118.3 GiB available: one 1M bank
at the 4,096-row native chunk, as with 14 layers (112.0 GiB).

## Conditions

- One NVIDIA DGX Spark / GB10, driver 615.71.09; CUDA 13.4.92, `sm_121a`;
  Linux 7.0.0-1019-nvidia. Clock range 300–2200 MHz; observed busy SM clock
  2,177–2,190 MHz. Memory PSI full avg10 at most 11.8 (first 19-layer run,
  at the end of model load; its first frontier is not slower than the other
  runs') and 2.0 (14 layers).
- Source `94d776ac` (clean tree), `ds4-bench` with NVTX instrumentation
  compiled in and no profiler attached, SHA256
  `41c04dab766680db1e8e5bd1ea22df3c6b7bd8d9fc202779732b1764f0b5eb54`.
- Models: 19 layers at HF `71c73e7e` (manifest `77835380…`); 14 layers at
  tag `r13p8-l3-16` (manifest `fba93aed…`, the artifact of the
  [October 18 sweep](../glm53-vq-spark-64k-r13p8-2026-10-18/README.md)).
  One resident model owner; no other GPU work.
- Prompt `speed-bench/promessi_sposi.txt`, SHA256
  `f53e0d80cb2d4492d24ebd63c7000c397b16ae70f9bf09b3763e5d8323ec209f`.
- `--ctx-start 2048 --ctx-max 65536 --step-incr 2048 --gen-tokens 128
  --mtp-draft 2 --mtp-margin 0`, `DS4_GLMVQ_MTP=1`, temperature 0 with EOS
  excluded so every frontier generates 128 tokens. Default prefill chunk.
- The 14-layer runs reproduce the October 18 sweep (1,113.7 / 31.81, 2.226
  tokens per verify at `90517b6b`) within 0.7%: 1,119.6 / 32.02, 2.226.
