GLM Q2_K down target preparation; no GPU result is recorded here.

The production SSD path calls `ds4_mmq_glm_moe(type=10)` with M4096/K2048,
1024 flattened assignments (`128 tokens * top8`), used1, 3389 cache slots and
2753688-byte padded expert stride. The observed fallback kernel has grid
867584, block32x8, and 70144B dynamic shared memory on GB10.

Primary routing is synthetic: assignment%288 gives 160 experts with four
assignments and 128 experts with three. Each original token has eight distinct
experts. Cache slot `(53*expert+3388)%3389` disperses these across the real
capacity, including slot3388. This histogram and deterministic dyadic F32 mid
are not a capture of actual router IDs or hidden states. Optional `skew8` uses
eight experts with128 assignments each; it is a distribution control.

The fixture is canonical `blk.3.ffn_down_exps.weight` from the actual artifact,
type10, dimensions[2048,4096,288]. The cache reserves9332248632B of virtual
address space. Primary population is792723456B of source weights plus338688B
of slot padding. The driver reports mapped VMM page charge separately and
rejects an eager reserve fallback. A second canonical288-expert allocation is
an addressing control; its physical charge is additional and printed separately.
Input, all1024 IDs, histogram, all4194304 output floats and the original ordered
top8 sum are saved. Valid outputs must be finite and differ from the 12345
sentinel; padding/output guards and memory fault counters are checked.

Every output and ordered sum must match the canonical-address public API
byte-exactly.36 independent sampled CPU dots decode Q2 codes and implement
D2S6 activation rounding: half scale/64, six half original sums/128, then
quantized sums for the remaining32 values. The CPU bound uses the128 grouped
signed dot terms' absolute magnitude and F32 forward-error allowance; it is
not a whole-model logit tolerance. The frozen baseline numeric/time/NCU runs
passed. Their 12 stored output arrays were CPU checked: all 28311552 floats are
finite and none equals the sentinel (`q2-target-v5/sentinel-coverage.json`).
The added assertion changes test source only; a new binary needs its own hash.

Only test driver compilation/linking is permitted while another GPU job runs:

```sh
python3 tests/test_glm53_q2_fixture.py /home/sunghoon/workspace/ds4-exaone/models/GLM-5.3-Flash-Uncensored-Mixed-Quant-GGUF/GLM-5.3-Flash-Uncensored-Mixed-IQ2XXS-IQ2XS-Q2K.gguf scratch/glm-uncensored/q2-target-v5
/usr/local/cuda/bin/nvcc -O3 -g -lineinfo --use_fast_math -std=c++17 -arch=sm_121 -Xcompiler -pthread -o /tmp/ds4-glm53-q2-v5 tests/test_glm53_q2.cu ds4.o ds4_ple.o ds4_distributed.o ds4_cuda.o cuda/mmq/ds4_ggml_stubs.o cuda/mmq/ds4_mmq.o cuda/mmq/ds4_mmq_d2r.o cuda/mmq/quantize.o cuda/mmq/mmid.o cuda/mmq/mmvq.o cuda/mmq/ds4_repack.o cuda/mmq/ds4_fattn.o cuda/qwen38_ple.o -lm -L/usr/local/cuda/targets/sbsa-linux/lib -L/usr/local/cuda/lib64 -lcudart -lcublas -lcuda
sha256sum tests/test_glm53_q2.cu tests/test_glm53_q2_fixture.py /tmp/ds4-glm53-q2-v5 ds4.o ds4_cuda.o cuda/mmq/ds4_mmq.o
```

After the root releases GPU ownership, use a fresh process per scoped run.
Keep user clocks300–2200MHz; record observed clocks, memory, temperature and
compute ownership before/after with `nvidia-smi`. Do not alter clock settings.
These commands clear diagnostic memset overrides and worklist overrides so
the retained baseline is measured. Allocation/upload and CPU-oracle costs
are outside the five event/wall timed public API calls. The first primary
call warms the shared pool; raw logs retain each timing independently.

```sh
mkdir -p scratch/glm-uncensored/q2-target-v5/numeric
env -u DS4_MMQ_YBUF_MEMSET -u DS4_MMQ_OUT_MEMSET -u DS4_MMQ_WORKLIST /usr/bin/time -v timeout 120s /tmp/ds4-glm53-q2-v5 scratch/glm-uncensored/q2-target-v5/q2.bin numeric scratch/glm-uncensored/q2-target-v5/numeric > scratch/glm-uncensored/q2-target-v5/numeric.log 2> scratch/glm-uncensored/q2-target-v5/numeric.time
mkdir -p scratch/glm-uncensored/q2-target-v5/time
env -u DS4_MMQ_YBUF_MEMSET -u DS4_MMQ_OUT_MEMSET -u DS4_MMQ_WORKLIST /usr/bin/time -v timeout 120s /tmp/ds4-glm53-q2-v5 scratch/glm-uncensored/q2-target-v5/q2.bin time scratch/glm-uncensored/q2-target-v5/time > scratch/glm-uncensored/q2-target-v5/time.log 2> scratch/glm-uncensored/q2-target-v5/time.time
mkdir -p scratch/glm-uncensored/q2-target-v5/ncu
env -u DS4_MMQ_YBUF_MEMSET -u DS4_MMQ_OUT_MEMSET -u DS4_MMQ_WORKLIST timeout 300s ncu --set full --profile-from-start off --kernel-name-base demangled --kernel-name 'regex:.*mul_mat_q<.*10,.*128,.*>' --launch-count 1 --replay-mode kernel --cache-control none --clock-control none --target-processes application-only --force-overwrite --export scratch/glm-uncensored/q2-target-v5/q2-down /tmp/ds4-glm53-q2-v5 scratch/glm-uncensored/q2-target-v5/q2.bin ncu scratch/glm-uncensored/q2-target-v5/ncu > scratch/glm-uncensored/q2-target-v5/ncu.log 2>&1
```

The profiler range contains exactly one baseline public API call after the
canonical control. The kernel filter selects only Q2_K x128 MMQ. Full sections
cover scheduler/warp stalls, launch/occupancy, SM instruction and tensor
throughput, cache/memory transactions and shared-memory behavior. Preserve
all section results and replay-pass count; profiler timing is not the fresh
event/wall timing. If counters are unavailable, retain that failure rather
than changing permissions. No adoption or end-to-end gain is implied.

The private GLM worklist candidate remains unadopted. `GLM_Q2_ROWS` defaults
128; standalone32/76 variants use the same fixture and candidate MMQ object.
Their balanced histograms are256 experts with one assignment and288 experts
with2–3 assignments, respectively. Derived rectangular grids are216896 and
542240;36 CPU dots include each width's last assignment. They preserve the
input, original top8 sum order, output/padding guards and dispersed cache IDs.

After root releases GPU, fresh off/on numerical cases use these already
compiled binaries;128 also gets the optional `skew8` control. Save each arm
separately and compare all down/sum bytes plus fixed mid/ID hashes. Both arms
must independently pass their CPU and canonical-address reference checks.

```sh
mkdir -p scratch/glm-uncensored/q2-target-v5/r32-off
env -u DS4_MMQ_YBUF_MEMSET -u DS4_MMQ_OUT_MEMSET -u DS4_MMQ_WORKLIST DS4_GLM53_Q2_WORKLIST=0 timeout 120s /tmp/ds4-glm53-q2-32-candidate scratch/glm-uncensored/q2-target-v5/q2.bin numeric scratch/glm-uncensored/q2-target-v5/r32-off
mkdir -p scratch/glm-uncensored/q2-target-v5/r32-on
env -u DS4_MMQ_YBUF_MEMSET -u DS4_MMQ_OUT_MEMSET -u DS4_MMQ_WORKLIST DS4_GLM53_Q2_WORKLIST=1 timeout 120s /tmp/ds4-glm53-q2-32-candidate scratch/glm-uncensored/q2-target-v5/q2.bin numeric scratch/glm-uncensored/q2-target-v5/r32-on
```

Repeat with `76` and the preserved128 candidate binary
`/tmp/ds4-glm53-q2-candidate`. Then measure matched fresh target timing and
one broad candidate NCU capture; whole natural2K/8K prefill/decode A/B is
still required. Build identities and the observed model-free admission
RED/GREEN are in `scratch/glm-uncensored/q2-target-v5/candidate-receipt.md`.
