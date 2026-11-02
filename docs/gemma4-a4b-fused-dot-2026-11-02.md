# Gemma-4-26B-A4B: the fused decode-dot (2026-11-02)

The fourth unit on this family. Decode landed the values; this one lands the
shape a forward can actually afford, and gates it against the values.

## Why a dot and not a decoded matrix

Decode materialises a whole tensor. That is fine for `q_proj`
(4096 x 2816, 46 MB of f32) and impossible for `gate_up_proj`
(360448 x 2816, 4 GB), which every one of the 30 layers carries. The runtime
never needs the matrix: it needs the activation dotted with one row, gathered
codebook word by codeword. Both producers already ship that shape -- the
engine as `v41_vq_row_dot`, the tools as `Stage::dot_row_local`
(`/data/VQ_tools/crates/vq-format/src/lib.rs:468-478`) -- and this unit
mirrors the tools' function rather than inventing one.

## What landed

- `VqMatrix::dot_row(row, x)`: one f32 accumulator, indices in row order, each
  codeword summed in dim order, and **the row gain applied after the sum**.
  That last point is not cosmetic: folding the gain into the terms would
  change the rounding and break the equality gate below.
- `VqMatrix::add_dot_into(x, acc, row0)`: every local row added at
  `acc[row0 + r]`, the tools' accumulator contract, so residual stages sum
  into one output row.
- `gemma4_a4b::matvec(dir, entry, x)`: the matrix-vector product of a whole
  entry, fused. `decode.rs` now parses the container once
  (`Container::open`) and both `decode` and `matvec` consume that parse, so
  the header is checked in one place and the two paths cannot drift.

## Gates

- Fixture (`the_fused_dot_matches_the_decoded_row`): `dot_row` and `matvec`
  equal the dot of the decoded row **bit for bit** where the fixture's values
  are exact, which pins the order, not just the tolerance.
- Artifact (the ignored gate): one deterministic activation
  (`x[c] = ((c % 17) - 8) * 0.125`), compared against the dot of the decoded
  row on every row of both tensors:

      q_proj  4096 rows  max |diff| 3.099e-6   largest row dot 4.241
      router   128 rows  max |diff| 5.364e-7   largest row dot 1.377

  The difference is expected and is not a defect: the decoded row stores
  `codebook * gain` per element and the dot applies the gain once at the end,
  so the two orders reassociate the same sum. The test asserts
  `max |diff| <= 1e-4 * max(row scale, 1)`, about 30x the measured value, and
  prints the measurement so a regression is visible before it trips.
- `cargo test -p ds4-core --test vq` 8 passed; `--test gemma4_a4b` 10 passed,
  2 ignored; `cargo test -p ds4-core --lib` 782 passed, 3 pre-existing
  failures (unchanged); `cargo check --workspace` clean.

## What this does not do

- No forward pass: no norms, no attention, no RoPE, no routing, no logits.
  `matvec` is the multiply it will call.
- No performance claim. The artifact gate runs in a debug build and is a
  correctness check; nothing here was timed, and the tools' own GEMV numbers
  (dim-8 0.681 ms) are the device path, not this.
- `dot_row` allocates its codeword buffer per call; `add_dot_into` allocates
  once per payload. A row-at-a-time caller should use `add_dot_into`.

## What the template should carry next

Nothing new: this unit exercised the release's geometry and the two payload
layouts exactly as the previous unit described them. The lesson is for the
port, not the schema: a release contract should state which fused operation
the runtime is expected to provide, because "decode" alone invites a
materialised matrix that the largest tensors cannot afford.
