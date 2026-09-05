# K2 MQ87: continuation campaign (6+6 new accepts)

Base: `2c56fc9` on `feature/k2-horizon-375b-serving`.
The earlier six-round write-up is historical and is not counted here.

## Protocol

GB10 / sm_121a, CUDA 13.3, Rust `ds4-bench`, MQ87 four-shard GGUF.
Raw `speed-bench/promessi_sposi.txt`, first 8192 tokens, 64 greedy decode
tokens, allocated context 8257. Fixture SHA256:
`f53e0d80cb2d4492d24ebd63c7000c397b16ae70f9bf09b3763e5d8323ec209f`.
Default memgov, no MTP or prefix reuse. Fresh process per cell.
`nsys` → bottleneck kernel → owner/worker down → `ncu` → change → A/B.
Kill every model PID and run `/usr/local/bin/clear_cache` between cells.
After each retained round, update this file and the HF model card
`Baekpica/K2-Horizon-375B-A23B-Mixed-Quant-GGUF` (Qwen card style:
date, engine SHA, corpus, token counts, cold/warm, command/env).

## Scoreboard

Same command and corpus every cell. Campaign baseline is HEAD `2c56fc9`
(nsys 8K+64, no owner): **286.55 prefill / 5.52 decode tok/s**.
Each finished round reports both rates and the delta against this
baseline (and against the last accepted round of that workload).

| Round | Prefill tok/s | vs baseline | Decode tok/s | vs baseline | Verdict |
|---|---:|---:|---:|---:|---|
| baseline `2c56fc9` | 286.55 | — | 5.52 | — | locked |
| Prefill 1 kill switch | 286.91 | +0.13% | 5.51 | -0.2% | same path |
| Prefill 1 assign-major `096fc9c` | **295.04** | **+3.0%** | 5.47 | -0.9% | retained |
| Prefill 2 IQ2_XS worklist | 272.03 / 272.14 | **-5.1%** | 5.51 | -0.2% | rejected |
| Prefill 3 IQ1_M slot-loop | **302.94** | **+5.7%** | 5.50 | -0.4% | retained |

Prefill 1 vs last accepted (the locked baseline): **+8.49 tok/s**.
Prefill 3 vs last accepted (`096fc9c` 295.04): **+7.90 tok/s**.
Decode IDs and all 250624 frontier logits are bit-identical across
the kill-switch, Prefill 1, and Prefill 3 cells. Decode tok/s moves
are run noise, not a claimed gain.

## Rounds

| Round | Hypothesis | Measured result | Verdict |
|---|---|---|---|
| Prefill 1 | Assign-major IQ1_M MMVQ, 3-D grid `(M, tokens, used)`, ncols=1 4-warp walk. `DS4_MMQ_IQ1M_PREFILL=0` is the per-token loop. | Synth 17/257/8192 bit-identical (8192×8 = 65536). 8K A/B 286.91 → 295.04 prefill, logits/IDs exact. | Retained |
| Prefill 2 | Reuse compact worklists for raw IQ2_XS down (4.3% of HEAD GPU time). | Real-weight worklist bit-identical; NT4096 17.9 → 8.3 ms. Same-binary on/off 262.86 → 272.03, logits/IDs exact, but both cells sit below the locked 286.55 baseline. Repeat on 272.14. Patch reverted. | Rejected |
| Prefill 3 | Walk top-k slots in one `(M, tokens)` IQ1_M block so the Q8_1 row is reused. `DS4_MMQ_IQ1M_SLOT_LOOP=0` restores the 3-D grid. | Synth 17/257/8192 bit-identical (3-D vs slot-loop). 8K A/B 294.38 → 302.94 prefill, logits/IDs exact vs P1. | Retained |
