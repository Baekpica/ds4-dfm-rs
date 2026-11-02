# Gemma-4-26B-A4B: the RQVQ decode gate (2026-11-02)

The third unit on this family. The first landed the folder and the artifact
contract, the second made the family selectable. This one proves the thing
both left open: that ds4 reads the artifact's residual payloads and gets the
same numbers the builder wrote them from.

The gate is exact, not statistical. Two tensors are decoded through the new
path and measured against the reference checkpoint:

    layers.0.self_attn.q_proj.weight   4096x2816  dim 8  relL2 19.6607 %
    layers.0.router.proj.weight         128x2816  dim 4  relL2 11.1430 %

Both agree with the builder's own `manifest.txt` column to the four decimals
it prints, which is a stronger statement than "within the recorded band": a
wrong stride, a wrong index width or a wrong gain would move relL2 by whole
percent, and a wrong codebook width would not decode at all.

## Why the oracle needed a second parser

`crates/ds4-core/src/vq.rs` was written as the V4.1 oracle and says so: a
`DQVL` container with a slot table at offset 16, a hardcoded `VQ_DIM` of 8
(`cuda_vq_row.inc.cu:31`), and the engine's two payload layouts. A
`vq-release/1` artifact carries the same two payload families, but not the
same bytes:

- the payload starts at offset 32 of a bare `RQVQ` file, not behind a slot
  table;
- the same artifact holds two codebook widths, dim 8 and dim 4 (the router),
  so the width is a parameter, not a constant;
- and, the finding this unit turned on, the magic `DQV3` names **two
  different layouts**.

The last point is worth stating plainly because nothing in either repository
said it before:

| | engine (`/data/YoungAi`) | tools (`/data/VQ_tools`) |
| --- | --- | --- |
| header | 32 bytes | 24 bytes |
| codebook | outside the payload, at `cb_off`, always E4M3 | inline, width from the container's format word |
| main stream | always 12 bits per index | `min(nbit, 12)` bits per index |
| plane | flags bit 1, plus bit 0 must say E4M3 | flags bit 0, no E4M3 bit |
| discriminator | word at 20 is `mnb` = 12 | word at 20 is a zero pad |

Sources: `cuda_vq_row.inc.cu:19-49` for the engine, and for the tools
`c/gpu/build-vq-gpu.c:82-126` (`v3_pack`) with
`crates/vq-format/src/lib.rs:314-361` (the reader). The router is the case
that makes the difference visible: `nc` 256 needs 8 index bits, and the tools
write 8-bit indices where the engine's kernel would have read 12.

Rather than guess a layout from a magic, the two constructors each demand
their own producer's header and refuse the other's. `VqMatrix::open` keeps
reading `DQVL` and still rejects any dim but 8, so the V4.1 path is unchanged;
`VqMatrix::from_payload` reads an `RQVQ` payload slice, v2 or tools-v3.

## What landed

- `crates/ds4-core/src/vq.rs`
  - `dim` and the codebook element width are fields of `VqMatrix`; every
    `VQ_DIM` use inside the decoder now reads them. `codebook_word` stays as
    the `vq8` accessor and asserts its width; `codebook_word_into` is the
    general one.
  - `index`, `gain` and the dequant loop take `dim`, and the v3 main-stream
    geometry is stored per matrix (`mb`, `mrow`, `prow`) instead of being
    recomputed from a constant.
  - `from_payload(pay, cb_width, dim, rows, cols)` parses the tools' layouts.
  - `add_into(out, row0)` accumulates one residual stage into a tensor, and
    `dequant_f32` is now the single-stage case of it.
  - the 3-byte index window reads past its slice as zero rather than
    panicking; the tools' payloads carry no safety byte after the last row,
    and the bits a clamped read drops are above the index width
    (`vq-format/src/lib.rs:406-411`).
- `crates/ds4-core/src/gemma4_a4b/decode.rs` (new)
  - `decode(dir, entry) -> Tensor`: re-checks the `RQVQ` header against the
    release (shape, geometry, codebook, payload kind, exact byte count), then
    sums `stages * groups` payloads at their group's first row.
  - copied entries are read as `RQVF` f32.
- `crates/ds4-core/tests/gemma4_a4b.rs`
  - a byte-built tools-v3 fixture at dim 4 proving the value rule and the
    stage sum;
  - a refusal test for each disagreement (geometry, payload dim, the engine's
    header in a tools payload, the recorded payload kind);
  - an `RQVF` sidecar test;
  - the ignored gate above, which needs `DS4_GEMMA4_MODEL` and
    `DS4_GEMMA4_SOURCE` and prints both relL2 values.

## What the plan got wrong

The handoff's plan said to add
`VqMatrix::from_payload(pay, dim, rows, cols)` "holding what `open` does
after `blob_slot`" and to keep the engine's v3 parse. Reading the artifact's
own bytes first — as the handoff also instructed — showed the payloads are
`DQV3` of the tools' shape, not the engine's, so a single parser could not
serve both. The signature gained the codebook width, and the engine's v3
stayed where it was.

The second correction is the stream length rule: the engine's v2 check
(`((total-1)*nbit >> 3) + 3`) is about a device 3-byte load, while the tools
write `(nidx*nbit + 7)/8 + 1`. Each constructor now uses its producer's rule,
and the clamped window makes the difference a matter of bounds, not values.

## Gates run

    cargo test -p ds4-core --test vq                 8 passed
    cargo test -p ds4-core --test gemma4_a4b        9 passed, 2 ignored
    the ignore-gated decode gate                    1 passed, both relL2 exact
    cargo test -p ds4-core --lib                    782 passed, 3 pre-existing
    cargo check --workspace --all-targets           clean

The three failures are the same ones the registration unit recorded and were
not touched by this unit: two `minimax_h3::lora` tests and
`minimax_h3::render::mux_pipe_lifetimes`.

## What this does not do

- It does not bind the decoded tensors to a layer table, and it has no
  forward pass. The next unit consumes `decode`.
- It decodes a whole tensor into RAM. A 4096x2816 matrix is 46 MB of f32; the
  fused expert stack (`gate_up_proj`, 360448x2816) would be 4 GB. The forward
  needs a streaming or fused gather, not this entry point per tensor.
- It says nothing about quality. relL2 measures how far the artifact is from
  the checkpoint it was fitted on; whether the model answers well is a logits
  gate in the model repository.

## What the template should carry next

The release states each entry's `part`, `role`, `kind`, shape, container
geometry and digest, but not the **payload layout**. `kind: v3` does not say
which of the two v3 layouts, and a consumer has to read bytes 20..24 to find
out. That is exactly the kind of inference the artifact contract exists to
remove. The next template revision should either give the layouts distinct
names in the release or reserve the magic properly, and the format page
should warn that `DQV3` in the two repositories is not one format.

The fused dot landed in `docs/gemma4-a4b-fused-dot-2026-11-02.md`; the
"what this does not do" note above about materialising a whole tensor is
what that unit solved.
