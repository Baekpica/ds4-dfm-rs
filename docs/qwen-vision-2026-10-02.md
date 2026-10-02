# Darwin multi-image attention (2026-10-02)

Four-image Darwin TTFT falls **6.286 → 5.583 s (-11.19%)** on GB10.
The shared Qwen4Exp vision attention now reuses K/V across four queries per
warp. The default applies to runtime `sm_121`, 72-value heads and at least
512 total patches. Other devices keep two queries. This report qualifies
Darwin `MQ-Q5-SSD-PLE-BF16` with FP8 PLE; other Qwen artifacts need fresh gates.

## Bottleneck and change

The original whole-worker Nsight trace attributes 3,983.1 ms across 27
attention calls to the four-image request, about 63% of its 6,315.7-ms TTFT.
Its attention-to-attention span is 4,366.4 ms. The screenshot contributes
786.0 ms across 27 calls. Existing concatenated image processing already
shares one vision tower invocation and segmented attention metadata.

NCU identifies load/store instruction pressure. Four queries share each
K/V register load, with 32 queries per block instead of 16. The 32-key
shared tile, per-key dot reduction, online softmax and explicit FMA association
remain unchanged. Segment-crossing blocks use the original per-row helper.
`DS4_QWEN_VISION_QUAD=0` restores two queries; `=1` forces four for diagnostics.

Matched isolated NCU captures use one 3,072-row, 16-head, 72-dimension FP32
attention launch, the fixture's LCG42 input and one image segment. Full counter
coverage uses application replay and `--clock-control none`. This reproduces
geometry/layout/arithmetic, with synthetic values and no model initialization.

| Metric | Pair | Quad |
|---|---:|---:|
| Kernel duration | 29.286 ms | 24.408 ms (-16.66%) |
| LSU utilization | 93.69% | 91.63% |
| Executed warp instructions | 8.831 billion | 7.020 billion |
| Shared-load wavefronts | 452,995,143 | 226,500,651 |
| Global-load requests | 43,401,216 | 22,167,552 |
| Registers/thread | 48 | 56 |
| Static instructions / encoded bytes | 1280 / 20,480 | 2304 / 36,864 |
| Theoretical / achieved occupancy | 83.33 / 82.31% | 66.67 / 65.62% |
| Dynamic shared memory/block | 18,432 bytes | 18,432 bytes |
| Local/shared spills | 0 / 0 | 0 / 0 |

Both captures observe 2.19 GHz. L2 hit rate changes 99.08 → 92.20%; no DRAM
bandwidth claim follows. Output-store requests remain 147,456. The change adds
register pressure and a larger instruction body without a workspace allocation.
Both kernels remain in the binary. Aligned production fixtures
preserve query arithmetic work. Ragged/tail blocks can do extra inactive work
or use more per-row fallbacks; the 31-row-segment diagnostic slows 0.091 →
0.108 ms. That isolated geometry is outside the latency suite below.

## Fresh worker A/B

Hardware: NVIDIA GB10, driver 615.71.09, CUDA 13.3.73. Preserve the user-managed
300–2200-MHz range; all 426 clock samples are 2190–2197 MHz. The same resident
VMM owner supplies one worker at a time. Production settings remain context
262144, max-seqs 1, partial reuse, native chunk 8192, MTP on/draft 2, 2-GiB
PLE cache/16 workers, continuous lane and 32-GiB disk KV. Graph fit/headroom
remain 1/1024 MiB; the worker guard remains max/high 38/36 GiB, reserve/trip 2/2.

Each arm/suite has three fresh workers, interleaved 0/1/1/0/0/1. Each starts
with empty isolated disk KV; every timed prompt has zero cached tokens.
Cases run small/screen/document/photo/large/multi, with text last in the
32-token suite. PLE and allocator state may warm within that fixed order.
Temperature is zero and thinking is disabled. No concurrent build, profiler
or other GPU workload runs during these timings. Cells below are medians;
[individual samples and ranges](benchmarks/qwen-vision-2026-10-02/latency.csv)
and [raw sample rows](benchmarks/qwen-vision-2026-10-02/samples.csv) are retained.

| One-token input | Patches / prompt | Pair TTFT | Quad TTFT | Reduction |
|---|---:|---:|---:|---:|
| Small, 256×256 | 256 / 93 | 287.1 ms | 288.7 ms | -0.56% |
| Screen, 1024×768 | 3072 / 797 | 1695.1 ms | 1561.3 ms | 7.89% |
| Invoice, 1536×1024 | 6144 / 1565 | 4791.1 ms | 4237.4 ms | 11.56% |
| Photo, 512×507 | 1024 / 285 | 567.5 ms | 546.4 ms | 3.72% |
| Large, 1920×1080 | 8160 / 2069 | 7355.5 ms | 6382.4 ms | 13.23% |
| Four fixed images | 10496 / 2659 | 6285.8 ms | 5582.6 ms | 11.19% |

Four-image ranges are 6284.5–6304.9 / 5572.6–5592.2 ms. Small-input dispatch
is unchanged; its 1.6-ms median increase has overlapping ranges. For actual
32-token responses, four-image TTFT is 6298.8 → 5576.9 ms and client wall time
7614.1 → 6889.9 ms (-9.51%). Decode stays 23.8 tok/s; the other image decode
medians also stay stable (photo 23.0 → 23.1). LM prefill throughput remains
1482.5 → 1486.7 tok/s on four images. This gain is in image processing.

The short text control is 31 prompt tokens plus 32 outputs: TTFT 163.0 →
163.4 ms, wall 1072.0 → 1075.8 ms and decode 34.5 → 34.1 tok/s (-1.16%),
with overlapping decode ranges. A longer final-build control is recorded below.

All 78 choices, finish reasons, usage objects, request bytes and fixture hashes
match across paths and repeats, excluding transport ID/time. The audit reads
90 actual stats files and 48 GPU ownership snapshots. Census/governor faults,
observation errors and substrate outstanding counts stay zero. All guards exit
normally; minimum MemAvailable is 20.73 GiB and full memory PSI is zero.

## Numerical and serving gates

The attention fixture passes 13 shapes: dispatch boundaries 511/512/513,
ragged/empty segments, tails, other widths, four-image geometry and cross-image
isolation. Legacy/pair/quad/default/repeat outputs are exact and sampled F64
reference error is bounded. Compute Sanitizer memcheck reports zero errors.

Five actual Darwin cases each run pair/pair/quad/quad. All features, 32 raw
248,320-value logit frontiers, 32 greedy IDs, live cache payload and M-RoPE
data match byte-for-byte. The saved frontier is prompt plus 31 consumed outputs;
output 32 is pending. EOS/EOT are excluded to expose all 32 frontiers. This is
a numerical continuation gate, with plain decode; production MTP2 and natural
stopping are checked separately. [Numeric rows](benchmarks/qwen-vision-2026-10-02/numeric.csv)
record sizes and state boundaries. The retained [manual fixture](../tests/test_qwen_vision_host.c)
uses [whole-prompt Rust token preparation](../tests/fixtures/qwen-images/README.md#rust-tokenized-full-model-gate)
without expanding the inference ABI or invoking the retained C tokenizer.

Final-build functional gates preserve production settings and pass screenshot
Failed **3**, invoice **385**, Earth, changed-pixel Failed **9** with cold KV,
and four-image `record_observation` arguments **3/385**. Exact tool history
returns **3, 385**, reusing 3006/3046 tokens and cached image features.
An 18,135-token image seed answers **3**; its follow-up answers **12** and
reuses 18,136/18,166 tokens. A new worker with zero prior generation routes
restores the same frozen request, reads 1,151,496,300 bytes from `.kv` files,
skips the encoder and returns identical output/token count. An empty-cache
worker computes the encoder, uses zero cached tokens and returns the same **12**.
Fault counters and serving issues remain zero throughout.
[Functional rows](benchmarks/qwen-vision-2026-10-02/functional.csv) retain each gate.

These gates cover seq1, one to four still images and an 18K reuse fixture.
Configured 256K capacity does not establish fresh 256K image quality, long
Agent throughput, other artifacts, other GPUs, video or concurrent clients.
Early HTTP harness attempts assumed short history was cold; observations
showed live reuse, so the harness was corrected without a runtime change.
An initial compute75 build rejected BF16 WMMA; explicit `CUDA_ARCH=sm_121`
builds succeeded. Those attempts are excluded from qualification.

## Provenance and reproduction

Measured source: `3aca84c190c514a26d51ba700a8c6b3d5c7cee1c` plus diff SHA256
`9b1da8c6c81e87ae47b2e2a2e5bbe72e64364dbdce37d7228729644d2cc1b092`.
Measured binary SHA256: `543b789d37c78dc63ab82011a0aaa5f0c239b7b8ec8fd9239ed2af647f64be48`.
The later runtime architecture predicate restricts default adoption to GB10;
the final build and its confirmation gates are recorded below.

Artifact revision: `0caa1f4961fc9d1ef9de400df7f11dd7ac6a6cd1`.
[Fresh full hashes](benchmarks/qwen-vision-2026-10-02/artifacts.json) cover all
three GGUF shards (vision/MTP/tokenizer/template embedded), four FP8 PLE blobs,
scale and PLE manifest. Full manifest SHA256, including the retained IPC manifest,
is `22a1f667f731c776f78fe1eca86ca1dafadde48aa9e35ecf8bf985cae1f24c9b`.
Raw profiler, request/response, process, clock and guard evidence remains in
ignored `scratch/qwen-multi-image-20261002/`. [Counter rows](benchmarks/qwen-vision-2026-10-02/counters.csv)
record the isolated resource/work measurements.

```sh
make -j1 CUDA_ARCH=sm_121 ds4-server tests/test_qwen_vision_attention tests/test_qwen_vision_host
./tests/test_qwen_vision_attention
compute-sanitizer --tool memcheck --error-exitcode 3 ./tests/test_qwen_vision_attention
```

Start each production-shaped worker against the same owner and a fresh empty
KV directory with `DS4_QWEN_VISION_QUAD=0` or `1`, then run the fixed image
client with `--model Darwin-180B-RSI --max-tokens 1` and again with `32` in a
separate worker. The manual full-model gate requires prepared token IDs and
the owner's IPC/PLE environment. Preserve fixture bytes and order.

Upstream review used pinned primary sources: [Transformers segmented vision
attention](https://github.com/huggingface/transformers/blob/35924ec379eec682bbdca219886e16eff4df8b09/src/transformers/models/qwen3_vl/modeling_qwen3_vl.py#L244),
[SGLang vision metadata](https://github.com/sgl-project/sglang/blob/d61a9c28ba81b8cea9eb50b12b9bc7b469fcaa98/python/sglang/srt/layers/attention/vision.py#L139)
and [llama.cpp F16 Flash input](https://github.com/ggml-org/llama.cpp/blob/fb4b2737a808a3fb7c2117a498f43815dc9be53e/tools/mtmd/clip.cpp#L718).
The selected change follows the measured LSU bottleneck while preserving the
existing FP32 arithmetic contract.

## Final-build confirmation

Final worker binary SHA256: `801f862b4fd3038fe3cf44820f49a1d53194566d0db862bafaf50b80c7a0f0f4`.
Its measured and rebuilt pair/quad kernel assembly, encoded instructions and
resource records match exactly; [the comparison](benchmarks/qwen-vision-2026-10-02/sass.json)
records nonzero instruction counts and hashes. The runtime predicate changes host dispatch
only. The retained four-image numeric gate passes all four admissions, and
the final linked attention fixture passes all 13 cases including default dispatch.

Six additional fresh workers use the same 0/1/1/0/0/1 order, each with uncached
four-image cap 1 followed by the unchanged text prompt at cap 128. Four-image
TTFT is 6309.3 → 5598.8 ms (-11.26%). Every text response stops naturally at
106 tokens and matches exactly. Decode is 35.4 [35.3,35.5] → 35.6 [35.2,35.6]
tok/s; wall time is 3163.5 → 3154.4 ms. The short text slowdown does not
reproduce in this longer bounded control. All 18 stats snapshots are fault-free,
all 24 ownership checks pass and 107 clocks remain 2190–2197 MHz.
[Final comparison](benchmarks/qwen-vision-2026-10-02/final-latency.csv) and
[samples](benchmarks/qwen-vision-2026-10-02/final-samples.csv) retain the ranges.
Its source diff SHA256 is `37fa38b976c927f065d2a5c24fd0be426f5184e2d677bc88ff99a795fddaac5d`.

The retained whole-worker reprofile records 27 four-image attention calls
at 3269.8 ms, down from 3983.1 ms (-17.91%); screen attention is 651.3 ms.
Profiled four-image TTFT is 5606.8 ms. Attention remains the largest current
bottleneck, at 52.2% of all captured GPU time and 58.3% of four-image TTFT.
These profiler timings explain the path; the unprofiled A/B establishes speed.

The original 8002 service is restored with this binary, original KV directory
and unchanged owner/settings. A post-restoration screenshot, four-image tool
call and exact tool-result history pass; live KV reuse and encoder skip work,
and fault counters remain zero. Operational PID/launcher/backup records remain
in the ignored restoration receipt, since process identities are transient.
