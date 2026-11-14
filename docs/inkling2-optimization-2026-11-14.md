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

## P1: keep attention-residual topology in registers

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
