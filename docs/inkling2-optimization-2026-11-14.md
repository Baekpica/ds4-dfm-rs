# Inkling-2 Small VQ optimization on GB10, November 14

This campaign starts at `53043039` with the
[published artifact](https://huggingface.co/Baekpica/Inkling-2-Small-VQ-GGUF)
at revision `b79f2fbd27e981691a7d0c930e0c42b9ece08ec4`.
The local `release.json` SHA256 is
`ff495b121f21e92870db8909f54880da61e02b56a37a9f9af8e60d17691e13ef`;
the loader verifies the model's tensor spans. No weights change.

## Protocol

One GB10, 48 SMs, CUDA 13.4.92 (`sm_121a`), driver user/kernel
610.57.04/615.71.10. The enabled, successful `nvidia-clocks.service` keeps
300–2,200 MHz; busy samples were 2,184 MHz. Boot identity:
`94cfabbf-7a48-4692-b38a-78e43e9be86d`. No profiler changes clocks.
Floors use the [GB10 measurements](gb10-hardware.md): 240 GB/s streaming,
107.8 TFLOPS F16/BF16 and 215.6 TFLOPS FP8 at the clock cap.

Round comparisons use fresh Rust `ds4-bench` processes, one resident model,
2,048-row chunks, default NVFP4 full-layer KV, plain greedy decode, cold
prefill, and `speed-bench/promessi_sposi.txt`. CPU affinity is
`5-9,15-19`. After the previous process exits, caches are cleared. The host
memory guard reserves/trips at 2 GiB, records PSI, and enforces a 1,800 s
deadline. Requested max/high are 128 GiB; admission sets effective limits
from available memory (116.4/115.4 GiB in these runs).

```sh
make -j10 ds4-bench ds4-bench-perf CUDA_ARCH=sm_121
python3 tools/host_memory_guard.py --max-gib 128 --high-gib 128 \
  --reserve-gib 2 --trip-gib 2 --psi-full-limit 0 --timeout 1800 \
  --log "$RUN/memory.jsonl" -- taskset -c 5-9,15-19 ./ds4-bench \
  --cuda -m "$MODEL_DIR" --prompt-file speed-bench/promessi_sposi.txt \
  --ctx-start 8192 --ctx-max 8192 --gen-tokens 64 \
  --dump-frontier-logits-dir "$RUN/proof" --csv "$RUN/bench.csv"
```

Nsight Systems attributes kernels within `ds4.prefill` / `ds4.decode`;
profiled throughput is excluded from A/B rates. NCU profiles isolated
production shapes with `--clock-control none`, never the full model.
Raw traces, counters, full logits, generated IDs, build logs, source diffs,
memory/clocks and process receipts stay in ignored `scratch/inkling2-perf/`.
Cold single-frontier rates are a different protocol from the published
2K–64K suffix sweep. These rounds do not qualify 1M, SSD restore, serving,
DSpark speculation or vision performance.

## P1: keep attention-residual topology in registers (`b3fd2fa6`)

The original 8K trace spends 1,374.457 ms (18.85% of prefill) and
544.178 ms (20.87% of decode) in `ink2_attn_res_kernel`. Runtime stack/query
indices put 592 bytes per thread on the local stack. At 2,048 rows, eight
stacked blocks and one query, NCU reports 2.995 GB of local read/write
requests (79.54% of L1 sectors) and 88.8% long-scoreboard stalls.

Specializing the model's block and query counts removes that stack. The
arithmetic expressions, reduction order and BF16 boundaries stay the same
(playbook class 5.1). The uncommon two-query case retains the runtime path;
`DS4_INKLING2_ATTN_RES_SPEC=0` selects it everywhere. No tensor allocation
or tensor arithmetic is added. Late-layer variants use 80/128 registers
for one/three queries, with no stack or spills.

| Isolated shape (depth 8) | Original median | Specialized median |
|---|---:|---:|
| 2,048 rows, one query | 4.388 ms | 1.077 ms |
| 2,048 rows, three queries | 9.982 ms | 1.450 ms |
| One row, one query | 94.016 µs | 10.923 µs |
| One row, three queries | 156.088 µs | 21.306 µs |

Three fresh isolated pairs, 20 event-timed iterations each, synthetic BF16
states in the production layout. The one-query case has 226.5 MB compulsory
tensor traffic: an optimistic 0.944 ms floor, or 88% of the streaming bound.
Both state passes reaching DRAM would instead cost 1.73 ms; rereads can hit
L2. NCU local read/write requests fall to zero for both query counts.

Full-model comparisons use the archived **original** executable, since
unrolling also changes compilation of the runtime fallback. ABBAAB order:

| Fresh pair | Prefill original → P1 (tok/s) | Decode original → P1 (tok/s) |
|---|---:|---:|
| 8K, 1 | 1120.64 → 1310.37 | 24.70 → 29.54 |
| 8K, 2 | 1121.10 → 1322.47 | 24.69 → 29.76 |
| 8K, 3 | 1126.32 → 1327.52 | 24.46 → 29.66 |
| Mean | 1122.69 → 1320.12 (+17.59%) | 24.617 → 29.653 (+20.46%) |
| 32K, one confirmation pair | 1059.80 → 1241.49 | 23.01 → 27.80 |

All eight runs exit 0; all 200,058 logits are finite and equal between
variants, as are all 64 generated IDs. Minimum available memory is
7.51 GiB, maximum PSI full avg10 is zero. The residual's measured time
falls to 306.083 ms / 108.477 ms in the retained 8K trace (4.95% / 4.98%).
The 162-case `test-inkling2-res` fixture covers row widths 1, 2, 4, 5, 17,
2,048; stack depths 0–8; null, one and three queries; and a nontrivial
block stride. It agrees with the runtime-indexed path. This is one adopted
**Prefill** round; the accompanying decode gain is not a separate round.

Production executable SHA256, original → P1:

```text
bcc68a0162ad24aa9fc517000d2857da3ac041a9f627badf36f455f5ac3efe0d
520a80da8aab9840c2fa0b775a4b9cb906bc2744b8bebf3b20a3389f083fbe9e
```

## P2: read sliding-attention biases ahead of their use

On the retained P1 trace, sliding prefill attention takes 843.569 ms
(13.64%, 160 calls). A late 2,048-row chunk has about 51.5 GFLOP of QK/PV
work, a 0.478 ms BF16 tensor-core floor, but its kernel takes 5.27 ms.
An isolated capture with the same 2,048 rows, position 6,144, 1,024-key
window and 3,072-row ring takes 5.306 ms. Values are synthetic BF16;
shapes, addresses and the execution path match the production case.

NCU reports 78.8% long-scoreboard stalls and only 8.55% issue-slot use.
SASS line mapping attributes 86.35% of long-scoreboard samples to the
relative-bias conversion immediately consuming a scalar global load.
The new path issues the raw BF16 loads before QK. Volatile PTX loads keep
the compiler from sinking them back to score use. It applies only to
BF16 sliding prefill; `DS4_INKLING2_BIAS_PREFETCH=0` restores the old path.
The score arithmetic, masks and reduction order stay unchanged (class 5.1).

Three isolated pairs: 5.368 / 5.441 / 5.323 ms → 3.861 / 3.861 / 3.972 ms.
NCU: 5.306 → 3.804 ms, 221 → 219 registers, no stack/spills, the same
69.63 KB shared memory and no extra tensor allocation. Global-load sectors
are unchanged; executed instructions increase 0.11%. Ordinary source-level
hoisting reaches only about 4.52 ms. The retained path is still latency
limited, at about 13% of the optimistic tensor-core bound.

| Fresh pair | Prefill P1 → P2 (tok/s) | Decode P1 → P2 (tok/s) |
|---|---:|---:|
| 8K, 1 | 1327.36 → 1382.64 | 29.53 → 29.64 |
| 8K, 2 | 1321.65 → 1383.94 | 29.23 → 29.57 |
| 8K, 3 | 1317.04 → 1384.28 | 29.82 → 29.21 |
| Mean | 1322.02 → 1383.62 (+4.66%) | 29.527 → 29.473 (−0.18%) |
| 32K, one confirmation pair | 1240.30 → 1285.26 (+3.62%) | 28.14 → 28.03 |

The decode ranges overlap (29.23–29.82 / 29.21–29.64); no decode path
changes and no material regression is established by these samples.
All eight runs exit 0 with equal, finite 200,058-logit vectors and equal
64-token continuations. Busy clocks are 2,177–2,184 MHz, mostly 2,184;
minimum available memory is 7.34 GiB and PSI remains zero.
In the new trace, sliding attention takes 618.184 ms (10.45% of prefill).

`test-inkling2-attn` passes 12 cases against the original kernel: widths
9, 63, 64, 65, 129 and 2,048 at position zero and after repeated ring wrap,
with a non-power-of-two capacity. These cover causal masking and query tails.
Production/NVTX builds and independent read-only review pass. This is
**Prefill round 2**. P2 production executable SHA256:

```text
134eca6dc8be145e3e92d2f2e8934dfe1cb1ef6ca1b8bc6cd53aa597c56e802d
```
