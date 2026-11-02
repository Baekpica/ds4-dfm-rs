# Gemma-4-26B-A4B: family registration and the per-layer geometry (2026-11-02)

The second unit on this family. The first (`docs/gemma4-a4b-2026-11-02.md`)
landed the folder and the artifact contract; this one makes the family
selectable and gives the two attention geometries a home. It adds no
executor and claims none.

## Why registration came first

Nothing downstream can start until a model can be named. A shape, a variant,
an architecture string, a serving quote and an identification route are what
`Model::open` consults before it touches a device, and the artifact contract
was already complete enough to answer them. Building a forward behind an
unselectable family would have produced code no path could reach.

## The flat `Shape` and the two geometries

`Shape` (`crates/ds4-core/src/shape.rs:122-179`) carries one `n_head_dim`,
one `n_head_kv` and one `n_rot`. This family needs two of each: 25 sliding
layers at head 256, 8 KV heads, full-width rotary and theta 1e4, and 5 full
layers at head 512, 2 KV heads, a quarter-width rotary (128 of 512) and
theta 1e6.

The convention that resolves it is already in the file: the un-prefixed
fields describe the full-attention geometry, which is what `layout.rs:702`
assumes when it picks `n_head` or `n_swa_head` for a layer, and the `n_swa_*`
fields describe the sliding one. `SHAPE_AXK2_VQ` follows the same split
(`rope_freq_base` full, `rope_freq_base_swa` sliding). `SHAPE_GEMMA4_A4B` is
written out in full rather than based on another family's constant, so no
unrelated value (an MLA rank, a YaRN factor) can leak in by default.

Two facts therefore have no field at all: the sliding head width (256) and
the sliding KV head count (8). There is no `n_swa_head_dim` and no
`n_swa_kv_heads`. `gemma4_a4b/config.rs` stays the only authority for those,
and the constant says so in its own comment. `n_swa_period` is 0 because the
five full layers are at 5, 11, 17, 23 and 29 rather than every Nth layer;
the schedule lives in the layer table, which is MiMo's documented rule
(`shape.rs:1478`). `n_full_attn_count` is 5.

## What landed

- `ModelFamily::Gemma4A4b = 19`, `Variant::Gemma4A4bVq = 23`, oracle name
  `gemma4_a4b`, `ArchRoute` arm, `shape_for_variant` arm, and
  `SHAPE_GEMMA4_A4B` (`shape.rs`).
- `gemma4_a4b::is_release` now claims a directory only when the release
  states `vq-release/1` **and** its first entry carries this artifact's own
  `model.language_model.` prefix. The schema string alone is shared by every
  VQ family, so it cannot be the claim on its own.
- `gemma4_a4b::identify` builds an `Identified` from the release with no GGUF:
  `split_count = 1` (one artifact, not a shard set), `n_tensors` = the
  release's 657 named entries, and the GGUF-only fields (`n_kv`, `alignment`,
  `version`) are zero rather than borrowed. `identify_gguf` asks it first for
  any directory (`identify.rs`).
- `serving_caps` refuses the family: every lane `Support::None`, `ReuseKind::
  None`, `MtpKind::None`, `SpecLane::None`, `HostNeed::Cuda`, `ctx_max` the
  checkpoint's 262,144 and nothing qualified (`serving.rs`). A request
  therefore errors instead of being quoted.
- Refusing or descriptive arms, each with its reason in place: `bind.rs`
  (empty name list), `layout.rs` (empty spec list), `validate.rs` (refuses a
  GGUF by name), `tok.rs` (refuses a tokenizer for this family, so no `Vocab`
  can exist), `serving_host.rs` (native limit and chunk), `ds4-server`
  `worker_run.rs` and `ds4-cli` `session_exec.rs` (a plain residual stream of
  `n_embd`).

## What the compiler found that the plan did not

The plan listed the sites to touch; the enum listed them better. Adding the
two variants produced ten non-exhaustive matches in `ds4-core`
(`bind.rs`, `layout.rs`, `serving.rs` twice, `serving_host.rs` twice,
`tok.rs` twice, `validate.rs` twice), then three more in the tokenizer test
tables, one in the serving-check variant table, one in `ds4-server` and one
in `ds4-cli`. Two of the ten — the tokenizer and the validator — were not in
the plan. A family is registered when the compiler stops naming it, not when
the checklist runs out.

The lesson for the next port: add the enum variants first and let
`cargo check --workspace` plus `cargo test --workspace --no-run` enumerate
the surface. Test-only tables are invisible to `cargo check`.

## Gates run

- `cargo test -p ds4-core --test gemma4_a4b`: 6 passed, 1 ignored.
- The ignored artifact test on `/data/vq-artifacts/gemma-4-26b-a4b-8x4096`:
  identifies 657 entries, `split_count` 1, `ctx_max` 262,144, and every
  digest verifies (121 s, debug build).
- `cargo test -p ds4-core --lib`: 782 passed, 3 failed — the same three
  pre-existing `minimax_h3` failures as at the merge commit, unchanged.
- `cargo check --workspace`: clean apart from pre-existing warnings.
- `cargo test -p ds4-cli`: 107 passed, 2 failed; both also fail in the full
  suite at the untouched baseline commit `c44df4cc` and pass in isolation, so
  they are cross-test environment pollution, not this unit.
- The integration suites that compare against the C oracles (`bind`,
  `catalog`, `layout`, `shape`, `tokenizer`) fail only on
  `build the C oracle first` assertions: the oracle binaries are not built in
  this checkout.
- `cargo test --workspace --no-run` stops on a test-only struct initializer
  in `ds4-server/src/h3_video.rs` (`Plan` missing `vae_attention` and
  `vae_gemm`). Neither that file nor `local_api.rs` is touched by this unit;
  it is a pre-existing break in the H3 video tests.

## What this does not do

No forward pass, no attention execution, no MoE routing, no logits, no
tokenizer ids, no chat protocol, no OpenAI route and no RQVQ device kernels.
The next unit is a CPU reference forward over the release: embedding, the two
attention shapes with their two rotary widths, the 128-expert top-8 block and
the softcapped logits, gated on a logits comparison. `hidden_activation` is
`gelu_pytorch_tanh`, not SwiGLU, and the routed experts are stored fused as
`gate_up_proj` 180224 x 2816 — both facts the forward must honour and neither
one expressed by `Shape`.

## What the template should carry next

Porting this family exposed three gaps in `vq-release/1` that a runtime has
to fill by hand. Each is a reason the next port costs more than it should:

1. A per-layer table. The release names shapes but not the attention kind per
   layer, so this runtime re-derives 25 sliding / 5 full from a copied
   `config.json` and must keep the two in step by hand. A `layers` block
   (index, kind, head width, KV heads, rotary width, theta) would make the
   release the single source and let a runtime with a flat shape constant
   validate against it.
2. A family marker. `vq-release/1` is shared by six families, so
   identification had to be inferred from entry-name prefixes. A `family`
   field in the release (the same string the runtime's oracle name uses)
   would make the claim explicit instead of structural.
3. A tokenizer statement. The artifact carries weights only; whether a
   tokenizer is present, absent, or shipped as a sidecar is currently
   knowledge held outside the artifact, so every runtime writes its own
   refusal.
