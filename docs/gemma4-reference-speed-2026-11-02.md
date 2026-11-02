# What makes the ds4.1 VQ engine fast, and what Gemma-4 lacks

Read first-hand from /data/ds4-dfm-rs (read only; nothing there was edited) on
2026-11-02, against this port's state at 2ed17989. The three axes the owner
named — decode, refill, TTFT — are three different mechanisms in the reference,
and this port has one of the three.

## Decode: one graph per step, and kernels that never restart

The engine captures a whole decode step (about 1500 kernels) as one CUDA graph
and replays it with a single cudaGraphLaunch
(ds4_ds41_graph.inc:1-6). Two properties make one capture serve hundreds of
steps:

- Position buckets of 1024 (ds4_ds41_graph.inc:48, DGRAPH_BUCKET). Every
  position-dependent value is read from the device slot st->pos, so the host
  passes the bucket's cap for grid and shared sizes and the kernels compute
  the true sizes from posd[0] (:7-13). Crossing a bucket re-captures.
- Slot uploads are kernels over pinned memory, not memcpy nodes, and the
  argmax tail reads back through a pinned int per row (:11-13, :58-62).

Programmatic dependent launch (PDL) edges are rewritten during capture on
compute capability 9 and above, with a fail-closed per-edge fallback
(cuda/ds41_graph.cuh:31-33). Capture failure is per state, never a process-wide
switch: n=1 failure drops only the n=1 graph, a batch failure drops the batch
graphs, and the engine's note says a global flag once cost a whole serving
night (ds4_ds41_graph.inc:15-21).

The step boundary is accounted in four parts — prep, launch, wait, gap
(ds4_ds41_graph.inc:67), so a host-side stall is a measured number and not
an impression. This port has the graph, but it captures one bucket (the gate
prints `1 bucket(s) captured`), defaults off, and the direct walk submits about
5 ms of host time per token against 13 ms of device wait
(docs/gemma4-a4b-gpu-2026-11-02.md, and today's gate log).

Inside the step, the VQ kernels are shaped to never pay a row head twice:

- Persistent grid = SM count, one block per SM, the layer's codebook copied to
  shared once per layer instead of per block (cuda/ds41_vq_persist.cuh:6-13).
- The index bitstream is read across consecutive rows as one stream, its
  8-round blocks cut by the span-global round, so a row that runs out of rounds
  reduces and finishes inside a block (:46-56, :86-116).
- Gate and up for the same row are one warp's work and the SwiGLU is applied in
  registers, so the gate value never round-trips through memory (:11-13).
- The grouped kernel for a verify batch owns one expert times the m tokens
  that chose it: the bitstream is read once and each codeword decoded once,
  which takes a round from 15.7m L1TEX wavefronts to 11.7 + 4m
  (cuda/ds41_vq_group.cuh:1-24).

The same file names its negatives, and they matter for us more than the wins:
f32 activations in shared went 6.46 -> 6.67 ms, a bitstream ring through
cp.async was noise, and a warp-count knob for the activation multiplier lost
more than it saved (cuda/ds41_vq_persist.cuh:63-67).

## TTFT: prefill is expert-major, chunked, and the last block is special

The prefill kernel's whole design argument is a measurement: a 512-token block
routes about 8 tokens per expert, while an expert holds 35.4M weights, so
dequantizing a whole expert for 8 tokens is arithmetic intensity 8 — 225 ms per
layer of traffic for 2.7 ms of math. Fused, one layer reads the blob once, about
11 ms (cuda/ds41_vq_prefill_fused.cuh:1-25). A work item is therefore
(expert, token segment, count), and each weight row is decoded once and dotted
against the at most 8 activations that expert owns. The v3 13-bit artifact
routes to a tensor-core run (cuda/ds41_vq_prefill_mma.cuh).

Long prompts are walked in chunks with two rules worth copying
(ds4_ds41_forward.inc:1208-1230):

- The slice rule takes the cap, then keeps at least one window of positions for
  the next block (v41_chunk_nc), so the sliding-window boundary never lands
  inside a chunk with no history behind it.
- While a chunk computes, the next chunk's engram prefetch is declared
  (cross-block prefetch), so the read is hidden behind the current block's
  math rather than serialized in front of it.
- A non-final chunk sets ced_skip and returns at the KV-only edge before the
  decoder, so all but the last block execute 21 of 40 layers
  (ds4_ds41_forward.inc:1218-1219; docs/ds41-perf-2026-10-11.md:113-118).

Measured on the reference host (DGX Spark GB10, docs/ds41-perf-2026-10-11.md:53-63):
392 tokens at 501 t/s, 2149 at 980 t/s, 8449 at 917 t/s, TTFT 1.03 s at 392
tokens and 9.79 s at 8449. The five-case record adds 1283-1286 t/s on 10.4k
prompts and warm TTFT of 191-273 ms (docs/benchmarks/ds41-spark-2026-10-13/README.md).

## Refill: reuse the KV at the common prefix, prefill only the suffix

The engine's own account of the disease: every request prefilled from token 0
while a chat or agent step sends the previous prompt plus what the model just
said plus a new piece. Measured 2026-10-11: 31 tool calls adding 5-9k tokens a
step, 40k-190k tokens prefilled a step, 40 s to 3 min a step at a normal
1000 t/s prefill and 40 t/s decode (ds4_ds41_kvreuse.inc:1-9).

The mechanism: after prefill and before decode, save a prompt-end checkpoint of
every buffer decode mutates or later reads — compressed KV and index keys up to
the used group count, the window ring's first SWA rows (position p always lives
in slot p % SWA), the compressor remainder rows, the draft hidden ring and its
counters, per-layer counters, the prompt token sequence, and the last logits
row. The next request memcmp's its prompt against that token key, copies the
checkpoint into a fresh state, and prefills only the suffix from c0 = reused;
an identical prompt restores the logits row and prefills nothing
(ds4_ds41_kvreuse.inc:11-21).

Three details that make it honest rather than clever:

- The named difference from a cold start is that suffix chunks start at the
  checkpoint, so block boundaries differ — FP-level, the same class as changing
  the chunk size (:16-19).
- A copy failure means this request prefills from token 0, never a half
  checkpoint (:24-25).
- The port adds a turn-end checkpoint T so the next turn reuses what the model
  itself said, and a restore requires T.n < np (:27-35).

The host records what was actually done — Cold, Exact, Partial, Fork — and why a
candidate was refused (no checkpoint at or below the LCP, rendered prefix
changed, below threshold, payload mismatch, state replay), so an operator reads
"the template dropped a block" instead of assuming the store is broken
(crates/ds4-core/src/serving.rs:398-433).

## Where this port stands, measured today

- Decode: best 13.98 ms per token (71.5 tok/s) with the graph walk, 14.17-15.19
  direct, layer bands 1.510e-4 / 2.409e-4 / 5.655e-5 / 9.057e-5, top-1 and
  greedy "Stockholm" both green. The engine's own decode measured 31-65 t/s on
  the reference host, so this port's decode is not the problem.
- TTFT: the server's sync() walks the prompt through the same per-token step
  (crates/ds4-server/src/bin/gemma4-server-rs/gemma4_engine.rs:607-621), so
  TTFT is prompt length times the step cost. Measured: 25 prompt tokens,
  ttft 459.7 ms; a 10k-token prompt would cost about two minutes.
- Refill: absent. Every request resets the state and re-walks the whole prompt
  (reset() at the same line).

## What transfers, in the order it pays

1. Prompt prefill in blocks, expert-major for the routed stacks and the tiled
   VQ kernel for the dense matvecs. This is the whole TTFT gap, and the tiled
   kernel already exists for variant 3 (PREFILL / PREFILL_SLICE lines in the
   gate). Needs the batch path through the layer chain: one forward with n>1
   tokens, positions pos0..pos0+n-1, and the attention reading the whole chunk.
2. KV prefix reuse at the common prefix, which compounds with prefill: on an
   agent turn the suffix is the only work. This port's state is one ring per
   layer (25 sliding of 1024 and 5 full), so the checkpoint is the ring rows,
   the position counters and the token key — the same span set the reference
   saves.
3. The graph as the default with position buckets, since the direct walk spends
   about 5 ms of host time per token and the replay is bit-identical today
   (relL2 0.000e0). Buckets are what let one capture serve a whole generation.
4. Chunking rules (the window-preserving slice and the cross-block prefetch)
   once a chunk loop exists.

Named negatives to keep out: activations in shared, a shared-ring bitstream
with cp.async, and — measured in this port today — the cross-row stream on this
container. The stream is bit-exact (0.000e0 through 30 layers) and slower at
every span tried, against the block walk in the same interleaved run: 14.52
against 14.17 at span 1, 15.43 against 15.15 at span 2, 15.54 against 15.27 at
span 3, 16.15 against 15.28 at span 4, 16.42 against 14.15 at span 8. The
reason is occupancy, not the idea: one warp per row
gives 4096 warps, a span of 8 gives 512, and this geometry is latency-bound
enough that the lost parallelism costs more than the amortized row head.

## Limits

- The reference numbers are a different host (GB10, 3003 MHz cap, 110 GB
  budget) and a different model geometry (V4.1 Flash, 40 layers, 2304/5120/...),
  so only the mechanisms transfer, not the rates.
- The 301-1500 t/s prefill figures are from the reference; nothing in this
  document is a projection of this port's prefill, which does not exist yet.
- The graph's 5 ms host submit is one measured token at pos 26 with
  --timing off; it is not a sustained figure.
