# Solar Open2 serving and capped prefill, 2026-09-14

One DGX Spark GB10 with a user-managed 300–2200 MHz SM cap. Solar Open2
250B MXQ-v1 (11 shards). This campaign qualifies disk-KV restart reuse and
HTTP partial fork, then promotes the round-5 warp-specialized prefill
attention kernel from opt-in to default after a completed 64K A/B under
the cap. A second speed round was not retained. The campaign closed at
the operator's request after that record.

Older uncapped HTTP tables and the September 7/12 cold benches use
different clocks and protocols. Do not mix them onto one graph.

## Protocol

- CUDA 13.3, `sm_121a`, one resident VMM/base owner (`--reserve-gb 16`).
- `speed-bench/promessi_sposi.txt`, SHA256
  `f53e0d80cb2d4492d24ebd63c7000c397b16ae70f9bf09b3763e5d8323ec209f`.
- Cold `ds4-bench --cuda`: `--ctx-start N --ctx-max N --gen-tokens 64`,
  N = 8,192 and 65,536. K-FP8/V-FP4, 4,096-token chunks, MTP off.
- Interleaved A1 B1 B2 A2 A3 B3, warmup then a fresh measured process.
  Median of three. SM clock 2190 MHz on every sample.
- All 196,608 frontier logits and 64 greedy IDs saved per measured sample.

## Serving

Native `tests/test_solar_session --disk-kv-only` and `--partial-only`
(including in-place `src==dst`). HTTP Chat with `--kv-disk-dir`,
`DS4_SERVER_FORK_PARTIAL` on, one bank, `reasoning_effort=none`.

| Check | Result |
|---|---|
| Disk-KV serial restart | `cached_tokens=4`, next token matches oracle |
| Disk-KV bank restart | `cached_tokens=6` |
| HTTP restart continuation | `cached_tokens=538` |
| HTTP partial fork (4,177-token mill) | source cached 0; branch cached 4,096, 81 computed, TTFT 416 ms vs 4.7 s |
| Two HTTP launches | `/v1/models` + Chat with non-empty `content`; `/v1/stats` zero sheds |

Fixes:

- Prefill clips each forward at the next checkpoint stride so the copied
  KDA state matches the labeled position. A due-at-chunk-end capture never
  fired when the whole prompt was one final 8K-cap chunk; labeling the
  end-of-chunk state as 4096 was wrong.
- Skip `trim_idle_banks` when partial reuse is on so a 1-bank worker does not
  drop the only hist-valid bank.
- Snapshot-token LCP fallback when rust text records miss.
- In-place partial no longer `evict_bank` (native hist must stay).

The 0.25 serial snapshot logit bound is unchanged. Completions still do not
warm-plan. Prompts shorter than the 4,096-token stride have no interior
checkpoint, so a 537-token mill pair stays engine-cold on HTTP.

## Round 1: default-on warp-specialized prefill attention

`DS4_SOLAR_FATTN_WS` now defaults on. `=0` restores the GQA-pair kernel.
`make test-solar-fattn` remains byte-identical; the 64K tail component is
331 ms (pair) vs 126 ms (WS).

64K nsys under the cap: pair attention 33% / 31.5 s of GPU time (216
launches). WS drops that to 16.6% / 12.5 s. Peak draw ~68 W vs the uncapped
~105 W freeze. Clock stayed 2190 MHz.

| Prompt | Prefill off → on | Change | Decode off → on |
|---|---:|---:|---:|
| 8,192 | 1,050.86 → 1,075.76 tok/s | +2.37% | 17.40 → 17.44 |
| 65,536 | 731.24 → 927.50 tok/s | +26.8% | 13.06 → 13.02 |

All twelve samples match logits and IDs byte for byte. The 64K decode
median −0.31% sits inside the off-arm sample range (13.02–13.06). First
decode at 64K is 0.1700 → 0.1731 s (+1.8%).

Raw rows: [solar-open2-2026-09-14-rounds.csv](solar-open2-2026-09-14-rounds.csv).
Graph: [solar-open2-2026-09-14-throughput.png](solar-open2-2026-09-14-throughput.png)
(`python3 tools/plot_solar_open2_20260914.py`).

## Rejected and stopped

`DS4_CUDA_SOLAR_GQA_CHUNK=128` vs 64, with WS on. 8K prefill unchanged
(1,073.92 vs 1,074.19). Prefill logits identical; greedy IDs diverge from
generated token 30 (34/64). Default stays 64.

A Q3 handoff down-sanitize skip was written but not A/B'd; it is not in
this tree. The campaign stopped there.

## Limits

- Published tok/s are clock-capped cold `ds4-bench`, not the older 2.4–2.5 GHz
  HTTP table.
- 1,048,576-token serving is not claimed.
- Keep the 300–2200 MHz cap on GB10 when WS is default.
