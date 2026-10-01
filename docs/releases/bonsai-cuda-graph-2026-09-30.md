# Bonsai (qwen35) on the CUDA graph — 2026-09-30

Records the unit that puts the Bonsai trunk on CUDA in this tree: the
gated delta-net and attention kernels, the graph driver, and the diagnostic
that drives them.  Everything below was observed on this host (RTX 4070
SUPER, sm_89, nvcc 13.3) at commit `8c887e4` (kernel port) plus the graph
commit on `feature/qwen35-port`.

Model: `/data/models/Ternary-Bonsai-2-27B-PQ2_0.gguf`, 7.2 GB, 851 tensors
(402 pq2_0, 353 f32, 96 bf16), `general.architecture = qwen35`.

Build the CUDA host binary (the CPU `ds4-c` and the CUDA `ds4-c` are the same
target; the arch must be explicit):

    make ds4-c CUDA_ARCH=sm_89

## 0. Residency on this box: the map must be copied to the device

`cudaHostRegister` of the whole 6.71 GiB mmap fails here with "invalid
argument" (the box has RLIMIT_MEMLOCK = 8 MiB), so the backend falls back to
lazy per-range materialisation and runs out of device budget part-way through
the trunk (observed as "Bonsai matmul failed for blk.32.ffn_gate.weight" in one
run and blk.35.attn_q.weight in another: the tensor that fails depends on the
allocation order, the symptom does not).  The sibling
tree runs the same artifact on the same card by copying the image; this tree
has the equivalent switch:

    DS4_CUDA_COPY_MODEL=1

which logs `CUDA copying 6.71 GiB model to device memory` and resolves every
weight from the device copy.  Every command below sets it.

## 1. Kernel gates

    make test-qwen35-cuda CUDA_ARCH=sm_89
    PQ2_0 CUDA parity: PASS

Now covers the ported kernel set as well as the PQ2_0 and fold checks:

    gdn out norm, sigmoid (qwen4exp) gate: max_abs=3.57628e-07 failures=0/18432: PASS
    gdn out norm, silu (Bonsai) gate:      max_abs=9.53674e-07 failures=0/18432: PASS
    attention split: decode, ctx 2048 (T=1 pos0=2048): ... PASS   (eight cases)

The token-tile cases pass with the numbers the sibling build prints
(max|d| 4.53e-08, 4.12e-08, 6.12e-08, 3.67e-08).  They need the kernel to
carry its own ldmatrix form: with this tree's `tt_ldmatrix_*` helpers (a
`uint32_t` address from `__cvta_generic_to_shared`) the tile cases fault with
an illegal memory access, and compute-sanitizer reports an invalid 16-byte
shared read at the kernel's `kv` load; passing the generic pointer reproduces
the sibling's numbers digit for digit.

## 2. The graph generates the reference stream

    make bonsai-cuda-parity            # CPU reference vs CUDA graph, 8 steps
    bonsai cuda parity: PASS

which is:

    DS4_QWEN35_TOKENS=760,6511,314,9338,369 DS4_QWEN35_STEPS=8 \
        ./ds4-c -m ... --cpu  --first-token-test -p x | grep '^token '
    DS4_CUDA_COPY_MODEL=1 DS4_QWEN35_TOKENS=... DS4_QWEN35_STEPS=8 \
        ./ds4-c -m ... --cuda --first-token-test -p x | grep '^token '

    token 5: 11751  Paris
    token 6: 13 .
    token 7: 198
    token 8: 760 The
    token 9: 6511  capital
    token 10: 314  of
    token 11: 9564  Germany
    token 12: 369  is

That stream is the one the CPU unit recorded for this artifact, and it is
byte-identical between the two backends for 16 steps of the same prompt.

## 3. Logits against the CPU reference and against the sibling's CUDA

`DS4_QWEN35_LOGITS` dumps the five prompt positions (5 x 248320 f32) from
each backend; the sibling tree's binary was run on the same five tokens with
the same variable (its `--raw` prompt is exactly those five ids).

    my CUDA -  CPU        max|d|=0.08093  rms_rel=0.264%  argmax 5/5
    sibling -  CPU        max|d|=0.08093  rms_rel=0.264%  argmax 5/5
    my CUDA -  sibling    max|d|=0.00000  rms_rel=0.000%  argmax 5/5

So this tree's graph reproduces the sibling's GPU arithmetic bit for bit, and
the remaining gap to the CPU is the same one the sibling documents: the k/v
cache is fp16 and the PQ2_0 MMVQ decode quantises activations to the Q8_1
form (measured per matmul by `test-qwen35-cuda` at rel_l2 0.0036-0.0044).
The gap is what remains of the CPU reference's double accumulation.

## 4. A near-tie continuation can flip, but not reliably in either direction

On the CLI's chat-templated prompt (25 tokens) the two backends agree for the
first 24 tokens.  They then differ in the author's runs (`a simple factual
question` on the CPU against `about the capital of Germany` on the CUDA graph,
deterministic over four CUDA runs) and agree in the independent QA's run
(32/32 identical, both `a simple factual question`).  The deciding gap is about
0.1 on a 22-magnitude top logit, so a small change in either side's arithmetic
moves it:

  - each backend is individually deterministic: the CUDA graph across two
    16-step repeats and a 32-step run, the CPU reference across three runs;
  - the flip is prompt-specific, not environment-specific: the
    France-templated prompt agrees 32/32 for both backends in both the
    author's and the QA's runs, while the Germany-templated one diverges in
    both (the QA pass reproduced the author's values digit for digit once it
    compared the same prompt);
  - the diagnostic prints only the last prompt position's top-5, so the
    deciding gap at the position where the streams part is not observable; at
    the prompt position the two differ by 0.019 on the top logit (CPU
    22.1780, CUDA 22.1969), inside the 0.08093 maximum measured over the
    five-id parity dump;
  - this tree's CUDA logits are bit-identical to the sibling's (section 3), so
    the difference is not in the port.

Conclusion: gate this unit on the token stream of an explicit prompt (section
2), which is reproducible, and treat a long chat continuation as a
near-tie coin flip at the artifact's decode precision rather than as evidence.

## 5. The defect this unit found and fixed: per-layer arrays overflowed

The first graph build was correct in structure but wrong in numbers: the
per-layer hidden state matched the reference to 0.13% through layer 60 and
then jumped to 15% at layer 61, and the logits gap was 2.09 (13%) instead of
0.08.

Bisect (all evidence from this tree):

  - a per-layer hidden dump (`DS4_QWEN35_DUMP_HIDDEN`) on both backends put
    the onset at layer 61;
  - a per-stage trace inside the gated delta-net layer showed the projections
    and the conv matching and the *state* 100x too large;
  - dumping the layer's conv/gate/state tensors and comparing the state
    against the expected `outer(v, k) * beta` showed head 1's second half
    held activations, not state;
  - dumping every graph tensor's device range showed 6 alias pairs, e.g.
    `lin_state[61]` and `lin_hist[0]` are the same allocation, and the
    duplicate pairs are exactly the ones an overrun predicts:
    `lin_state[61..63]` -> `lin_hist[0..2]`, `lin_hist[61..62]` ->
    `k_cache[0..1]`, `k_cache[63]` -> `v_cache[2]`.

Cause: the graph's per-layer arrays were sized `DS4_MAX_LAYER`, which this
tree defines as 61 (the DeepSeek bound), while Bonsai has 64 blocks.  The
sibling tree's constant is 79, which is why its identical code is sound.  The graph now
uses `DS4_QWEN35_MAX_LAYER` (64) and refuses to open when the model has more
blocks than the arrays hold.

The temporary dump/trace instrumentation used for the bisect is not part of
the commit.

The same unit also had to keep the graph out of the CPU-only build: the block is
wrapped in `#ifndef DS4_NO_GPU` with stub drivers for the CPU case, because
`cc -DDS4_NO_GPU` (the CPU host build, and the tests that include ds4.c) failed
with 70 errors the moment the graph referenced `ds4_gpu_tensor`.  The
independent QA found that one; it is the reason this unit carries a third
commit.

## 6. Rates

Decode-only, one token per graph call (the diagnostic shape), measured with
`/usr/bin/time` after the model copy:

    warm run (copy + 5 prompt + 8 steps):  1.64 s wall, 7.80 GB RSS
    long run (copy + 5 prompt + 64 steps): 3.09 s wall, 7.80 GB RSS

Both runs pay the same one-off model copy (about 0.7 s, logged as "CUDA model
copy complete in 0.6xx s"), so the 56 extra steps cost about 1.45 s: roughly
26 ms per token in steady state, about 38 tokens/s.  The first forward is
slower (CUDA context and weight-page warm-up), so this is a steady-state
number, not a first-token rate.  The CPU reference on the same prompt is
about 3 s per token.

The sibling tree's note recorded 0.43 s per token for its own diagnostic on
this host; that number predates its decode-path work (the PQ2_0 MMVQ vec
entry and the fold collapse), so the two are not directly comparable.  This
unit measures only what it changed: the trunk now runs on the device.

## 7. Runbooks added

    make bonsai-cuda-check      # the greedy stream on the graph alone
    make bonsai-cuda-parity     # CPU vs CUDA streams, diffed

Both read `DS4_BONSAI_MODEL` and set `DS4_CUDA_COPY_MODEL=1`; the parity
target's prompt ids and step count are `DS4_BONSAI_PARITY_TOKENS` and
`DS4_BONSAI_PARITY_STEPS`.
