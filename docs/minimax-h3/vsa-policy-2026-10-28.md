# VSA, the reference's own sparsity rule for the H3 stage (2026-10-28)

The sparse forward this branch just ported
consumes a mask; it does not decide one. For the stage this engine actually
serves - the 50-block H3 DiT under `DENOISE` - the reference's decision rule is
**VSA**: `configs/default.json:12` selects
`video_sparse_attn_h3` for the DiT at `sparsity 0.9`, `tile_size 64`,
`prefix_mode exempt`.

## 1. Provenance

The policy lives in FastVideo, pinned by the reference at
`FASTVIDEO_COMMIT = 3d8ac9d14bd697a89ede8f170cbfbca012a9edcc`
(`runtime/stage1_ops/vsa.py:12`), which the archive verifies by hashing the
installed module against `VSA_SOURCE_SHA = b7c6d4e4...dc0fab9`
(`vsa.py:13`, `:164-166`).

Fetched today and hashed:

    fastvideo/attention/backends/video_sparse_attn_h3.py
      sha256 b7c6d4e470ee232b50d16d71c2b21806f414f9baa4447434b2da66c94dc0fab9
      = the pinned VSA_SOURCE_SHA, byte for byte

So the rule below is read from the same bytes the reference refuses to run
without, not from a summary. Supporting files: `video_sparse_attn.py` (the
shared tile helpers and `compute_topk`), `vsa_utils.py` (tile geometry),
`triton_kernels/fused_compress_topk.py` (the fused block-mean and mask
builders).

## 2. The rule

1. **Pooling** (`video_sparse_attn_h3.py:380-383`): a tile's summary is its
   **mean** - `sum over the tile's tokens / variable_block_size`, in F32, for Q
   and for K. This is the same quantity `h3d_sol_pool_kernel` already computes
   as `kc` and `h3d_sol_threshold_kernel` as the query centroid (`q_bar`).
2. **Scores** (`:629`): `scores = (q_pooled @ k_pooled^T) / sqrt(head_dim)`,
   a `[batch, head, n_tiles, n_tiles]` matrix. Only the ordering of a row
   matters to the mask, so any positive scale gives the same mask; this port
   already applies the same form with its `log2_scale`.
3. **Keep count** (`video_sparse_attn.py:161-163`):
   `k_vid = max(1, min(ceil((1 - sparsity) * num_video_tiles), num_video_tiles))`.
   At `sparsity 0.9` and 288 video tiles that is 29 tiles per query row.
4. **Mask** (`video_sparse_attn_h3.py:387-410`), per row:
   - `exempt` (the reference's setting): the video columns compete for the
     `k_vid` largest scores; **every prefix column is selected**; and
   - `mask[:, :, :num_prefix_tiles, :] = True` - every **prefix query row is
     dense**, whatever its scores.
   - A sparsity of 0 (`k_vid == num_video_tiles`) returns an all-true mask.
   - Non-exempt mode has the prefix compete in one `k_vid + num_prefix_tiles`
     topk over all columns; the reference does not use it.
5. **Consumer**: a block-sparse attention at the tile granularity
   (`block_sparse_attn_triton.triton_block_sparse_attn_forward`, or cuDNN BSA
   on sm_121 at tile 64), fed the bool mask compacted to `(indices, counts)`.
6. **Compression branch** (`:625-641`, `:758-763`): with the trained
   `to_gate_compress` present, a second branch attends the pooled tiles densely
   - `out_c = softmax(scores) @ v_pooled` - and the two are merged under the
   gate. That gate is part of the FastH3 VSA DataFree LoRA artifact, which this
   repository does not have; the reference computes `scores` whenever
   `sparsity > 0`, so the **selection path does not depend on the gate**.

## 3. What the port needs, and the three deviations to name

Everything the rule needs is already in the plan: `kc` is the tile mean of K,
`query_bar` is the tile mean of Q, and the mask is the format the forward
already consumes. The new piece is a selector that keeps a per-row top-k
instead of comparing against a threshold.

Deviations to state, not hide:

1. **Tile geometry.** The reference's tiles are 3D - `VSA_TILE_SIZE = (4, 4, 4)`
   tokens, i.e. 64 tokens of the video latent - and it reorders the sequence
   into tile order before the attention (`get_tile_partition_indices`) and back
   after. This port's mask is over linear 64-token blocks of its own head-major
   sequence. Whether our DiT's token order puts the reference's (4, 4, 4) tile
   members in one 64-token block is a property of the engine's layout and must
   be checked before the mask is called equivalent; if it does not, the port's
   selection is a *different* (still valid) grouping and has to be measured as
   such.
2. **Prefix.** Our sequence's prefix is our engine's own text and audio span.
   The reference requires `num_prefix_tiles >= 1` and exempts it; the port needs
   the same span as a parameter.
3. **Fold against drop.** The reference's VSA consumer *drops* the unselected
   tiles (the compression branch is what carries their mass back). This port's
   forward *folds* them from the pooled summaries, which is a strictly
   better approximation of the dropped mass and costs the nine folded passes.
   Combining a VSA mask with the folding consumer is therefore a named,
   measured deviation - and `tests/h3/sol_forward_bm` already prices both halves
   of it separately.

## 4. The design the port should take

Materialising the `[heads][blocks][blocks]` score matrix is what the reference
does (`torch.matmul` at `:629`), but this port does not need to: one CTA per
(query block, head) with 32 lanes, the `kc` words staged in shared as
`h3d_sol_route_kernel` already does, a per-row running k-best in shared (the
reference's k is 29 of 288 at the shape that matters), the score consumed as
each column is computed so no score array is indexed at runtime, and the mask
bits written once at the end with the prefix rules applied. Estimated cost:
about 0.5 M comparisons per CTA, ~10-20 us per call against the 4.9 ms the
search step already costs - to be measured, not assumed.

The prize is the ratio: at 47% exact the forward measures 73.4 ms against the
dense 173.4 ms at 18432 tokens, and its cost is `exact_share x 150.8 ms + a
small constant` (measured). A 90% fold would put a layer's attention near
`0.1 x 150.8 + ~5 = 20 ms`, which is the reference's own sparsity setting, and
would make the DiT's attention share of `DENOISE` minor rather than dominant.
