# DeepSeek V4.1 vision weight staging on DGX Spark

Follow-up to merged [PR #83](https://github.com/Baekpica/ds4-dfm-rs/pull/83),
based on `a5e9a62664510cbc8198f30e54886a81e31f7476`.
This candidate changes weight transfer into the vision tower. Image
preprocessing, tensor bytes, arithmetic, device layouts and KV policy stay
unchanged. The [build receipt](build.json) identifies the tested source and
frozen binaries; documentation changes do not alter that runtime.

## Source and resource contract

The reference is **fodelf/YoungAi**, base engine pin `7fb26cc` and vision pin
[`846c929c`](https://github.com/fodelf/YoungAi/tree/846c929c5d11ef4f89d6406e8504e6e66db3ce69).
Its [stage helper and request lifetime](https://github.com/fodelf/YoungAi/blob/846c929c5d11ef4f89d6406e8504e6e66db3ce69/src/core/core_v41_vision.c#L21)
read each block once for all images, free temporary weights after the request
and advise away the sidecar page cache. Its
[run record](https://github.com/fodelf/YoungAi/blob/846c929c5d11ef4f89d6406e8504e6e66db3ce69/fable5.md#L11269)
explicitly replaced whole-sidecar pinning with that policy.

Named port deviation: one **8 MiB request-local pinned window** stages the
same bytes from the mmap before synchronous H2D copies. The final chunk may
be shorter. Allocation failure falls back to direct copies; success and
failure paths release the window. `DS41_VISION_PINNED=0` selects the original
path for diagnosis. Actual `staging=pinned` logs are required for the on arm;
fallback is not counted as an optimization pass.

The [resource account](resources.json) adds 8 MiB to image admission. For this
artifact, one tower invocation explicitly copies **970,924,032 additional CPU
bytes**, and makes **312 weight writes instead of 263**. Device weight shapes
are unchanged. This is per image group, not per image; cached follow-ups skip
the tower. No full-sidecar pinning or cross-request tower residency is added.

## Bottleneck and isolated measurement

A fresh merged-main run completed nine HTTP requests. Cold small/four-image
tower times were 65.73/71.48 s; appending a new image spent 73.17 s in its
tower. Cached text follow-ups already avoided the tower. The earlier
[Nsight trace](../ds41-vision-review-2026-10-14/profile-tower.json) found
49.833 s in 525 H2D copies versus 0.258 s in 1,816 kernels across two towers.
That trace includes profiler overhead and is not an uninstrumented speed
measurement.

The isolated reproducer copies the actual four block matrices in order for
32 blocks: 822,083,584 bytes, unchanged 24.5 MiB device slots. Both arms charge
DMA completion inside each tensor's timer. Pinned allocation/free and CPU
copies are included; byte-exact D2H verification is outside the timer.
Two fresh processes per mode/cache state, three repetitions each, gave
24 exact-byte samples:

| Source pages | Direct median | Pinned median | Result |
| --- | ---: | ---: | --- |
| Advised away; `mincore` confirms zero resident | 5.2235 s | 0.9190 s | 5.68× faster |
| CPU-warm | 0.0349 s | 0.0512 s | 16.3 ms slower |

[Ranges and resources](copy-summary.json). This reproducer excludes main-model
memory pressure and vision kernels. It supports the transfer hypothesis;
whole-request adoption uses the measurements below. No driver, PCIe or
storage root cause is inferred from these timings.

## Whole-request A/B

**Retained for this artifact and workload**, with the cached-follow cost below. All
three pairs improve image TTFT. The off arm varies substantially, so ranges
and individual samples are retained.

| Request | TTFT off, median [range], s | TTFT on, median [range], s | Tower median, off → on, s |
| --- | ---: | ---: | ---: |
| One image | 39.663 [9.202–40.237] | 0.821 [0.816–0.829] | 39.06 → 0.23 |
| Four images | 44.928 [23.357–60.303] | 6.882 [6.843–7.400] | 41.71 → 3.67 |
| Independent one-image turn | 38.778 [20.467–79.687] | 0.822 [0.815–1.026] | 38.18 → 0.23 |
| Append one new image | 44.496 [24.461–84.255] | 4.567 [4.554–4.593] | 42.63 → 2.70 |

Native prefill includes the tower: median one-image rate **5.7 → 279.5 t/s**,
four-image **41.6 → 275.7 t/s**, appended suffix **23.0 → 227.9 t/s**.
The text-only long prompt remains **896.1 → 895.5 t/s**; this is not a
text-prefill optimization.

| Control | Median TTFT off → on, s | Median native decode off → on, t/s |
| --- | ---: | ---: |
| Short text | 0.3824 → 0.3744 | 30.30 → 30.21 |
| Long text | 2.9428 → 2.9495 | 33.56 → 33.90 |
| Exact image replay | 0.0166 → 0.0168 | 37.74 → 37.74 |
| One-image text follow-up | 0.3873 → 0.3830 | 30.96 → 30.21 |
| Four-image text follow-up | 0.5070 → 0.5014 | 36.23 → 35.97 |

**Measured tradeoff:** one-image text follow-up wall time rises from
**1.58785 → 1.61561 s** at the median (+27.8 ms, +1.75%). The three paired
increases are **17.4 / 37.9 / 91.2 ms** (+1.1 / 2.4 / 6.1%). Native decode
cost increases **0.8 / 0.8 / 2.6 ms per token**. This request executes no tower.
The four image-miss cases have median native decode rates spanning
37.17–39.06 t/s in both arms. No general decode speedup or zero-regression
claim is made.

[Auto-MTP records](mtp-shapes.csv) show different round/skip counts on this
follow-up: off **18/7, 18/7, 16/4** versus on **19/8, 19/8, 18/7**.
The unchanged [YoungAi scheduler](https://github.com/fodelf/YoungAi/blob/7fb26cc332d740416792204a29c7d13d2986caf5/src/core/core_draft_sched.c#L31)
chooses verify widths from measured request costs. This explains why equal
output need not imply an identical schedule; it does **not** prove the cause
of the slowdown. Adoption weighs this observed cost and the warm-copy cost
against the large, consistent image-request gains and bounded memory cost.

[All 54 paired samples](paired-results.csv), [ranges](paired-summary.json),
[common responses](responses.json) and [serving audit](serving-audit.json).
All six processes produce identical core requests, responses, token counts
and cache usage, not merely HTTP success. Request-time clocks are
**2184–2197 MHz**, median **2190 MHz**. Minimum sampled host memory available
is **6.55 GiB**. Every run has continuous telemetry and verified GPU/port cleanup.

The [ten extra requests](extra-requests.json) are single on-arm observations:

| Input | Prompt tokens | TTFT, s | Text follow-up TTFT, s |
| --- | ---: | ---: | ---: |
| Five images | 2813 | 9.714 | — |
| Eight images | 3717 | 12.219 | — |
| Screenshot, 1080×2414 | 962 | 4.435 | 0.452 |
| Market panel, 1200×665 | 497 | 1.821 | 0.400 |
| Earth photo, 512×507 | 199 | 0.787 | 0.384 |
| Large fixture, 1920×1080 | 983 | 4.740 | 0.446 |

The five/eight-image native logs confirm all five/eight images execute.
All four extra text follow-ups reuse the complete previous prompt and run
no tower. The campaign completes **64 HTTP requests / 3,821 output tokens**.
Most responses reach the 64-token cap; these prove bounded execution and
continuation, not complete answers or a visual-quality score. [Fixture hashes](fixtures.json).

Same frozen server, six fresh sequential processes, order off/on, on/off,
off/on. Each runs nine requests: short/long text, one image, exact replay,
text follow-up, four images and follow-up, then an independent image prompt
and appended image. Each pair must have identical requests, response content,
token counts and reported cache usage. The last on run also covers five/eight
images and screenshot/document/photo/large-image requests with follow-ups.

DGX Spark / GB10, CUDA 13.4.92, driver 615.71.09. The configured **300–2200 MHz**
range is unchanged. Model: `DeepSeek-V4.1-Flash-vq8sh14-q4k-mtpnative.gguf`,
original Engram tables, code GR, vision sidecar SHA256
`b50974c405aafbd941f8e8767cf90231b522aa4f2fcb506eaea68fbf8b409af5`.
Artifacts remain under `~/workspace/ds4-exaone/models/YoungAi-DeepSeek-V4.1-Flash`;
the [verified artifact receipt](../ds41-vision-review-2026-10-14/vision-artifact.json)
and unchanged file identities pin the inputs.
Byte-verified fixture copies are also collected under that model directory's
`vision-validation/`; its manifest retains the original test paths.

Server settings: `--cuda --mem-budget-mb 110000 --mem-floor-gb 2`, one sequence,
default context/capture, automatic MTP, Engram and code GR, loopback port 8099.
Requests use temperature 0, `max_tokens=64`, `reasoning_effort=none`, SSE with
usage. No logit, trace, rollback or reencode diagnostics run during timing.
The host-memory guard records every run; requested reserve/trip are 2 GiB.
GPU ownership and the listener must be empty before the next model load.

TTFT uses the client's monotonic clock. Native tower/prefill timers are rounded
by the runtime; prefill includes the tower. Decode is reported separately and
depends on MTP acceptance/content. Model loading is outside request timings.
Filesystem caches are not globally flushed. Both arms use the same ordinary
sidecar eviction policy; an image miss means no KV reuse, not proven cold SSD
pages. Compare the paired campaign's ranges, not a separate historical run.

## Numerical and state evidence

[Audit](gpu-audit.json), [cases](gpu-stats.csv),
[comparisons](gpu-comparisons.csv), [graph/rollback counts](gpu-graph.csv):
**19 generations, 360 emitted tokens, three official golden fixtures and six
comparisons passed**. Pure decode records 328 finite full-vocabulary rows.

- Text, single/four images, text follow-up and appended image match tokens and
  every full-vocabulary logit between direct and pinned staging.
- Follow-up and append prefixes are independently seeded in each arm. Both
  restore 220 tokens; text follow-up runs no tower, append encodes one image.
- Fixed-5 MTP matches 32 pure-decode tokens, executes ten captured verify
  batches and eleven rollback checks, ten with rejected drafts. MTP dumps
  contain partial single-decision rows, not full speculative-logit evidence.
- [ASan/UBSan/leak checks](stage-test.log) pass 15 assertions for exact chunk
  and tail bytes, bounded writes, one-window lifetime, allocation fallback,
  failure cleanup, repeated requests, the diagnostic switch and admission.
  The original source failed nine behavioral assertions before the change.

The [provenance record](correctness-provenance.json) preserves a failed fixture
attempt: a shortened raw prompt emitted EOS immediately in the direct control.
Restoring the previously qualified wording passed without changing production
code or weakening output/logit assertions. The resumed run adds 17 generations;
two text controls and three goldens are explicitly inherited from the first
load after checking native objects, settings, artifacts and receipts. There
were **two model opens**, not one uninterrupted passing run.

The merged PR's host checks, long captured/eager proof, tool protocols and exact
disk-restart proof are [inherited for unchanged paths](../ds41-vision-review-2026-10-14/README.md).
They were not rerun. This change does not qualify new context, concurrency,
bank or snapshot limits. A 1M context advertised by metadata is not a completed
1M workload. Bounded image responses do not establish broad visual accuracy.

Raw requests, SSE, native logs, guard telemetry, frozen executables and numeric
dumps remain in ignored `scratch/ds41-vision-staging/`. `measure.py` runs each
server, `paired.py` fixes the six-process order, and `finalize.py` audits the
completed receipts. Compact data above records their final results.
Published CSVs use LF line endings; receipt hashes identify the original raw files.
