# Prism Bonsai (qwen35) decode: split-K attention — 2026-09-30

The graph handed the attention kernel no split-K partial buffer, so the
dispatcher fell to `splits = 1` and one row walked its whole key range serially
in 6 blocks (`(H + 3) / 4`) on a 56-SM card.  At a 15k context that cost 231 ms
per token; /data/ds4, which hands over the buffer, spent 28 ms.  This unit ports
the graph side of that (the kernel side was already here), and decode at 15k now
matches the sibling: 33.81 against 33.83 tokens/s.

Host: RTX 4070 SUPER (sm_89), `/data/models/Ternary-Bonsai-2-27B-PQ2_0.gguf`,
ctx 45056, branch `feature/qwen35-port`.

## What the reference does, and what was missing here

`/data/ds4` (branch `bonsai`) keeps two constants, `DS4_QWEN35_ATTN_ROWS 16` and
`DS4_QWEN35_ATTN_SPLITS 64` (ds4.c:68996, :69001), allocates a partial buffer in
the graph open sized `min(T, 16) * 24 heads * 64 splits * 258 * 4` (:69152-69161),
and in the attention step picks `tokentile = T >= 32 &&
ds4_gpu_qwen4_attn_tokentile_available(...)`, sets `attn_batch = tokentile ? T :
16`, walks the chunk in row batches with tensor views and `pos0 + r0`, and passes
`tokentile ? NULL : g->attn_partial` (:69024-69060).  The dispatcher then cuts
each row's key range into `min(64, (keys + 31) / 32)` ranges and reduces them in
`attn_merge`.

This tree already carried the dispatcher verbatim
(cuda/qwen35_attn_gdn.cuh:609-628) and even a kernel-level comparison of the two
orders against a double-precision oracle (tests/test_qwen35_cuda.cu, the
attention group: split vs single at a 1e-4 bound below the token-tile gate,
token-tile vs split at 1e-3 above it).  What was missing was the graph: it called
`ds4_gpu_qwen4_attn_decode_tensor(..., NULL, ...)` once for the whole chunk, so
the split could never be selected.  A stale comment in the same function claimed
the token-tile kernel was never selected, and two documents repeated it.

## What landed

- ds4.c: the two constants, `attn_partial` in the graph struct, its allocation in
  `qwen35_graph_open` (24.2 MiB at the 16-row bound), the batched attention call
  replacing the single one, and the field in the free list.
- `DS4_QWEN35_ATTN_SPLITK=0` forces `splits = 1`, the old path: the A/B base arm
  and the escape hatch.  It is the one thing here the reference does not have.
- tests/test_qwen35_session.c: the long-prompt pass now decodes the same prompt
  twice, with the split forced off and with it on, and requires identical ids.
- docs/BONSAI.md and docs/releases/bonsai-session-2026-09-30.md: the token-tile
  claim is corrected with the measurement that contradicts it.

## Evidence

A/B, same model, same ctx, same prompt, 64 decode tokens, three interleaved
rounds, one binary serving both of our arms (the kill switch selects the path),
a fresh server process per sample:

    15k prompt        decode tok/s                    ttft
    split-K off       4.14   (4.25 / 4.09 / 4.08)     20.10 s  (20.14 / 20.39 / 19.78)
    split-K on        33.81  (34.94 / 34.10 / 32.38)  19.54 s  (19.12 / 19.46 / 20.05)
    /data/ds4         33.83  (34.81 / 32.33 / 34.34)  19.28 s  (19.05 / 19.38 / 19.41)

    30k prompt
    split-K on        30.29                            51.87 s
    /data/ds4         30.95                            51.33 s

So decode is 8.2x the old path and 0.06% off the reference at 15k, 2.1% off at
30k; prefill is within 1.4% (it was already at parity, and the token-tile kernel
is what carries it: forcing the row-exact path instead costs 9.3 s on a
15k-token prompt).

Correctness.  The split changes the attention reduction order, so the contract is
the token stream, not bit equality: all 21 session checks pass against the CPU
reference (short and long prompt), the new long-prompt check confirms the ids do
not move between the two orders, multichunk passes, and `run-bonsai.sh ids` and
`session` are IDENTICAL on both backends.

An independent audit (agy, Gemini 3.1 Pro, reading the files from disk and
required to cite lines) found one real defect that review had missed: the new
buffer was allocated but absent from `qwen35_graph_free`'s scratch list, so every
graph open/close leaked 24.2 MiB.  Fixed, and it is the reason the field is in
that list above.  The same audit also invented one citation (it claimed
misc/scratch/wt-qwen35/ds4.c:69156 passed the partial buffer; that line is the old
single call), so its findings were checked one by one rather than taken.

## Limits

- The clock readings in the A/B logs are idle samples taken before each request
  (ours 2790 MHz, the sibling's 2505), so this is not a per-clock comparison.
- The 30k row is one round each; the 15k rows are three.
- Decode rates are counted from SSE token deltas, i.e. client-observed.
- The 24.2 MiB is device memory the serving quote does not itemize (scratch reads
  0); the measured margin covers it, and the fresh-sample device use is 7.4 GiB
  against the sibling's 10.65 GiB, because this tree brings the bank in on the
  first request instead of reserving it up front.
- Short-context decode was already in this class (36.8 tokens/s at a 50-token
  prompt) and is unchanged; nothing here helps the first token of a long agent
  prompt, which is prefill of a 30k-token system prompt and is at parity.

## Next, measured

Decode is now level with the reference, so the next targets are elsewhere:
prefill at long context (780 tokens/s at 15k, 389 at 30k, the key range growing
per chunk), the PQ2_0 tile GEMM that is 57.2% of the retained prefill window,
and the serving side of the agent experience — the 30k-token prompt's first token
is either trimmed tools or a qualified prefix reuse for this family (the engine
reports `partial_unqualified`, and on the observed first turn
`cached_prompt_tokens` was 0).
