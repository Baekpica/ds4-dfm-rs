# Spark-X2.6-4B VQ: artifact and load contract

Spark-X2.6-4B VQ (family `Spark2_5Vq`, architecture `spark2_5`, variant
`spark-x2.6-4b-vq8`, recipe `spark-x2.6-4b-base-r13r8`) is a GVQ1/R13+8
conversion of a Spark-X2.6-4B checkpoint. This page records the load half:
what the release directory contains, what the loader verifies, and what is
deliberately not there yet. It is the Rust host's own contract; the reference
implementation of the graph is the checkpoint's
`modeling_spark.Spark2_5ForCausalLM`, named in `config.json`'s `auto_map`.

## Status

| Piece | State |
|---|---|
| Release identification (`release.json`, `files.json`, part metadata) | implemented, `crates/ds4-core/src/spark2_5/` |
| Shape constant and architecture route (`spark2_5`) | implemented, `SHAPE_SPARK2_5_VQ` |
| Layout table (830 entries, generated from the runtime pack) | implemented, `spark2_5/layout.rs` |
| Inventory check (type and shape per tensor, GVQ1/GVR8 headers, book and residual links) | implemented |
| Tokenizer and chat adapter | not ported |
| Forward pass, KV layouts, serving lane | not implemented |

Nothing in the serving capability table is qualified for this family: banks,
disk restore, snapshots and a drafter are declared absent, and prefix reuse is
declared exact without a measurement. A capability is marked when it is
measured, not when it is plausible.

## Graph

```
embed × 2560 ─▶ [ rms ─▶ GQA sliding 512 / full ─▶ sigmoid head gate ─▶ +h
                   rms ─▶ dense FFN 10,240 (gelu) ─▶ +h ] × 36 ─▶ rms ─▶ tied head
```

- 36 layers, hidden 2,560, vocabulary 131,072 (tied embedding, no `lm_head`).
- 16 query heads and 4 KV heads at head_dim 256 in both geometries, so the
  fused `q_k_v_proj` is 6,144 wide in every layer.
- `layer_types` is three sliding layers then one full, nine times: sliding
  layers 0–2, 4–6, ... and full layers 3, 7, 11, ... 35 (nine of them).
- Dual rotary: sliding layers rotate all 256 dims at base 1e4; full layers
  rotate 0.25 of the head (64 dims) at base 5e6. Both live in
  `rope_parameters` in the checkpoint config, and the runtime must read them
  per layer; `SHAPE_SPARK2_5_VQ` holds the sliding majority and says so.
- Per-head sigmoid output gate (`gate_attn_act_mode`), 16 BF16 columns per
  layer; no q/k norm tensors.
- Dense FFN, 10,240 wide, gelu; no experts, no drafter.

## Artifact

`release.json` names two parts, and the loader refuses any other role:

| Part | Residency | Tensors | Bytes | Contents |
|---|---|---|---|---|
| `spark-x2.6-4b-vq8.gguf` | resident | 829 | 1,262,301,824 | 180 GVQ8 matrices, 360 GVBC books, 180 GVR8 residual planes, 73 F32 norms, 36 BF16 gates |
| `spark-x2.6-4b-vq8.embedding.gguf` | ssd-rows | 1 | 671,089,280 | the BF16 token table, read by row |

Total 1,933,391,104 bytes. Every matrix is the same R13+8 container Gemma-4's
release uses: a 13-bit index per 8 weights into an 8,192-word E4M3 book plus
an 8-bit GVR8 index into a 256-word residual book. The attention projections
are VQ too, so a layer's five matrices (`q_k_v_proj`, `out_proj`, `gate_proj`,
`up_proj`, `down_proj`) carry two books and one residual plane each.

`config.json` sits beside the parts and is hashed in `files.json`; the module
embeds a byte-identical copy and compares the checkpoint's file against it
field by field, so a changed layer count, head split, rope factor, window or
vocabulary is refused before a weight is bound.

## What the loader checks

1. `release.json` and `config.json` match their `files.json` sizes and SHA-256.
2. The release names architecture `spark2_5`, variant and format
   `spark-x2.6-4b-vq8`, source model type `spark2_5`, recipe
   `spark-x2.6-4b-base-r13r8`, format version 2 and context 1,048,576.
   Format version 1 is refused: this family's converter has no plain
   (13-bit, no-residual) recipe to tie such a part to.
3. Each part carries `spark2_5.part_role` (`trunk` or `embedding`),
   `spark2_5.source_revision` matching `release.json`, and only `general.*`,
   `tokenizer.*` or `spark2_5.*` metadata keys.
4. Every tensor's GVQ1/GVR8 header is canonical: version, bits (13 and 8),
   vector width 8, book width (8,192 and 256 words), E4M3 codec, row gains
   finite, payload offsets aligned and padding zero, and the residual payload
   digest at header bytes 96..128.
5. `spark2_5.codebooks` and `spark2_5.residuals` link every matrix to exactly
   one book and one residual plane, and the GVQ1 book digest matches the book
   bytes; there is no unlinked or unused book.
6. The loaded inventory equals the layout table entry for entry: 830 names,
   stored type words and GGUF-order dimensions.

## Not in this unit

The tokenizer adapter (`tokenizer.ggml.pre` chain, added-token matching,
special ids) and the chat renderer; the forward pass with its fused qkv
slicing, per-head output gate and two rotary geometries; KV layouts; and any
serving lane. `--serving-check` therefore reports no qualified item for this
family.

The session payload has no layout word either: `PayloadLayout` in
`crates/ds4-core/src/payload.rs` maps tags that the native engine writes, and
nothing in this tree can write a `spark2_5` payload yet, so the family stays
off that table rather than claiming an unreachable constant. The word is
defined natively, together with the engine, and mapped here at that point.
