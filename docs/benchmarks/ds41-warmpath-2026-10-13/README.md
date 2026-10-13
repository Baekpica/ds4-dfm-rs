# DeepSeek V4.1 Flash: the asset-open cache on the warm path

The port used to redo, on every request, the work the reference engine does once
when it opens its state: the ②/③ sidecar mount (a 40 MB `gr/rb` directory read,
its device upload and its digest fold) and the engram open (GGUF, inventory,
wire, hash tables, shards, io_uring ring). This campaign measures the same work
switched off and on, on one DGX Spark / GB10, with the author's five-case HTTP
suite on the archived case bytes: 45 samples in three passes with the caches in
place, against the archived pre-change pass of 15 samples.

![DeepSeek V4.1 warm path: cache off against cache on](bench-compare.png)

[JLFN](https://github.com/JLFN) ported DeepSeek V4.1 Flash serving from the
[YoungAi engine](https://github.com/fodelf/YoungAi/tree/7fb26cc332d740416792204a29c7d13d2986caf5)
to `ds4-dfm-rs` in [PR #80](https://github.com/Baekpica/ds4-dfm-rs/pull/80); this
campaign is evidence for [PR #83](https://github.com/Baekpica/ds4-dfm-rs/pull/83).
The VQ GGUF and code GR/RB sidecars come from
[YoungAi's model repository](https://huggingface.co/wenzhouwu/YoungAi-DeepSeek-V4.1-Flash/tree/84b86701456f4e4b357e41340b3b88914f51834d),
the original model and Engram weights from
[DeepSeek](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/tree/5254448103b6254fc646dacbca1cbbf20e9216c1).

| Case | Prompt tokens | Warm TTFT before | Warm TTFT after | Before/after |
| --- | ---: | ---: | ---: | ---: |
| prose_short | 54 | 190.9 ms | 15.1 ms | 12.6x |
| code_short | 54 | 196.7 ms | 17.3 ms | 11.4x |
| table_short | 43 | 194.1 ms | 14.1 ms | 13.8x |
| prose_long | 10,356 | 265.7 ms | 99.1 ms | 2.7x |
| table_long | 10,377 | 273.4 ms | 99.5 ms | 2.7x |

The [tables](tables.md) carry every sample and the min-max band. Warm TTFT is
time to first token on a request that reuses the prompt's KV checkpoint, which
is what the cache changes: the per-request 207-255 ms asset prelude leaves the
request path. The first request that sends each case's bytes still carries its
prefill and is measured separately - 0.657 -> 0.427 s, 0.591 -> 0.409 s,
0.544 -> 0.372 s, 8.053 -> 7.957 s, 8.086 -> 8.001 s - so the cache is not
bought with a slower cold path. Decode stays at parity (36.26/35.87,
54.33/53.42, 65.41/65.28, 31.08/32.50, 59.14/58.61 tok/s before/after).

## Conditions

- One NVIDIA DGX Spark / GB10; driver 580.142, kernel 6.17.0-1014-nvidia, CUDA
  compiler 13.0.88. The box is not clock-locked today: observed SM clock
  2,281-2,418 MHz (median 2,372) against a 3,003 MHz maximum. The published
  campaigns record a 300-2,200 MHz range; this campaign's clocks sit above it,
  and that difference is a limit below.
- Merged source `96780cb`. Server `ds4-server` SHA256
  `dd46b778fbb1d4d5815ccf9c27d06036487636be2fa35ca00a53e0f22f822593`; `ds4`,
  `ds4.o` and `ds4_cuda.o` hashes are in the [receipt](evidence.json).
- One server process (pid 217789, started 14:55) served all three passes; it was
  never restarted, which is the one protocol difference from a fresh-process
  campaign. Single serial sequence, automatic embedded MTP, default CUDA
  graphs/VMM, temperature 0, reasoning off, one stream.
- The suite is the author's `suite.py` run unmodified, on the author's
  `cases.json`. The case bytes are the archived ones, so each case's
  `prompt_sha256` matches the earlier published run: the two arms measured
  identical prompts.
- Reuse stays enabled; the engine, its sidecar (`grrb-code_fit_n15360-engine`)
  and the model id `DeepSeek-V4.1-Flash-vq8sh14-q4k-mtpnative` are as served.
  Rep 0 of each case is the first request that sends those bytes; reps 1-2
  repeat them.
- Artifact identity follows the [review campaign](../ds41-spark-review-2026-10-13/README.md#model-and-source):
  the 113,556,639,424-byte GGUF and the Engram shards were verified there and
  neither changed for this campaign.

## Validation and limits

All 45 samples returned `finish_reason=length` with finite positive rates, the
prompt length and completion count are identical in every pass, and the three
passes hash their prompts to the same `prompt_sha256` per case. The kernel
journal for the campaign window holds no line at all - the audit command, its
window and its SHA256 are in the [receipt](evidence.json), and the receipt says
so rather than implying a silent pass. Memory headroom stayed at 6.85 GiB
available or above, memory pressure `full avg10` stayed at 0.00, and the sampler
recorded 421 samples across the run.

- No server restart: the three passes share one process, so a "cold" sample is
  the first request that sends those bytes in that pass, not in a fresh process.
  The process holds one saved checkpoint and it moves as the cases advance, which
  is why each pass has a cold cell at all. A restart-per-pass campaign would
  separate boot state from steady state; it was not taken on a box the owner
  serves.
- The before arm is an archived pass recorded earlier the same day, before the
  cache landed, with no clock receipt. Its clocks are unrecorded, so the
  before/after read is a read across two windows, not a matched A/B. The warm
  cell moves by 2.7x-13.8x, far beyond what a clock difference of a few percent
  can explain, but the absolute numbers belong to their windows.
- Client-side timing: ttft runs from request send to the first content delta. The
  campaign client sat on the box. A second pass set with the client on another
  machine over the LAN (`run-lan-1..3.csv`) reads 14.0, 15.2, 15.6, 99.7 and
  100.7 ms warm - within a millisecond or two of the on-box numbers, so the
  client's location does not carry this metric.
- Decode is reported for parity, not as a gain. The two short cells read 1-2%
  lower after than before, which is inside the spread this box shows between
  windows (an earlier pass in the same hour read 54.3-54.7 tok/s on code_short).
- One serial text workload, five cases, no concurrency, no bank or session
  claim. Nothing here qualifies answer quality.

## Reproduce

With the server serving on `127.0.0.1:8000` and the model id it reports:

```sh
for i in 1 2 3; do
  python3 suite.py --url http://127.0.0.1:8000 --engine ds4-dfm-rs \
      --cases cases.json --csv run-$i.csv --reps 3
done
python3 ../../plot-ds41-warmpath.py   # tables.md, summary.json, results-raw.csv
python3 summarize.py                  # bench-compare.csv, bench-compare.png
```

`plot-ds41-warmpath.py` checks the invariants above and folds the passes; the
comparison and the chart stay the author's `summarize.py` output, run here with
three named edits (the two arm labels, their colors, and the chart title).

Published data: [run 1](run-1.csv), [run 2](run-2.csv), [run 3](run-3.csv),
[the archived before arm](archived-before.csv), [all samples](results-raw.csv),
[the author's comparison](bench-compare.csv), [summary](summary.json),
[receipt](evidence.json), [the harness](suite.py), [the case bytes](cases.json).
The LAN pass set is `run-lan-1..3.csv`; the sampler script, its raw files, the
server log and the kernel audit stay on the box.
