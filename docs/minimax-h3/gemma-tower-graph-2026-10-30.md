# The Gemma4 text tower: the graph and its loader (2026-10-30)

Unit 2b. The tower's primitives are unit 2a's
([`gemma-tower-primitives-2026-10-29.md`](gemma-tower-primitives-2026-10-29.md))
and the INT8 convrot linear is unit 1's
([`gemma-int8-convrot-2026-10-29.md`](gemma-int8-convrot-2026-10-29.md)); this
is the graph that calls them, the reader that gets the checkpoint onto the
device, and the gate that holds both. The reference read it works from is
[`gemma4-tower-port-map-2026-10-29.md`](gemma4-tower-port-map-2026-10-29.md).

## 1. What landed

- `crates/ds4-core/src/minimax_h3/gemma.rs` (new): the tower. The constants the
  checkpoint's own metadata states, a reader that checks every tensor against
  the graph before a byte moves, one convrot workspace per projection shape,
  the 48-layer loop, and the 49 hidden states.
- `crates/ds4-core/examples/h3_gemma_gate.rs` (new): the gate. A second
  implementation of the reference forward in Rust (the oracle), an op probe,
  and the per-state hash line a golden is compared against.
- `crates/ds4-core/Cargo.toml`, `Makefile`: the example and the
  `tests/h3_gemma_gate` target, built like the other `h3_*_gate`s.

## 2. The 49 hidden states, and a correction to the port map

The states are **not** a list the forward builds. They come from the
output-capture hook (`transformers/utils/output_capturing.py`):

- `:112-118` prepends the first decoder layer's own input - the scaled
  embedding - and appends every layer's output, so the tuple is the embedding
  plus the 48 layers;
- `:270-278` then **replaces the last of those** with `last_hidden_state`,
  which is `self.norm(hidden_states)` (`modeling_gemma4.py:1736`).

So state 0 is the embedding, states 1..=48 are layers 0..47, and **state 48 is
`model.norm`'s output, not layer 47's raw output**. The port map's line
"the embedding first, then layers 0..47" reads as though the last state were
the layer's own output; it is not, and a port that emitted the raw one would
produce a wrong connector input while looking internally consistent.

The tower applies that substitution only for a run that covers all 48 layers
(`layers == LAYERS`). A layer-limited run is a diagnostic, and there the last
state stays the layer's own output so a comparison localises to that layer.

## 3. The graph, step by step

One layer, `Gemma4TextDecoderLayer.forward` (`:1405-1463`), as the port runs it:

| Step | Reference | Call |
|---|---|---|
| `input_layernorm` | `:1417` | `rms_norm` with the layer's weight |
| `q = rope(q_norm(q_proj(x)))` | `:1249-1252` | `linear_int8_convrot`, `rms_norm` at rows x 16, `gemma_rope` |
| `k = rope(k_norm(k_proj(x)))` | `:1265-1268` | the same, at rows x kv heads |
| `v = v_norm(v_proj(x))` | `:1270` | `linear_int8_convrot`, then the weightless `rms_norm` |
| on a `k_eq_v` layer, `v = v_norm(k_proj(x))` | `:1265-1270` | the raw `k_proj` output, not the normalised or rotated key |
| attention | `:824-858` at `scaling = 1.0` (`:1200`) | `gemma_attention` |
| `o_proj` | `:1294` | `linear_int8_convrot` |
| `post_attention_layernorm`, residual | `:1427-1429` | `rms_norm`, `add` |
| `pre_feedforward_layernorm` | `:1432` | `rms_norm` |
| `down(gelu(gate(x)) * up(x))` | `:1088-1090` | `gelu_tanh`, `mul`, `linear_int8_convrot` |
| `post_feedforward_layernorm`, residual | `:1449-1450` | `rms_norm`, `add` |
| `*= layer_scalar` | `:1462` | `scale_bf16` |

Four norms per layer, not three. The full layers carry no `v_proj` - eight of
48 - and their `o_proj` reads 8,192 wide while a sliding layer's reads 4,096.

The embedding is `embed_tokens(ids) * embed_scale` with the scale downcast to
the weight's dtype (`:1474-1477`), so the multiplier is `bf16(sqrt(3840))`,
which is **exactly 62.0** - not 61.9677. BF16's binade [32, 64) steps by 0.125
and 61.9677 is nearer 62.0; the reference's own comment records the same
rounding for Gemma 3 (`sqrt(3072)` = 55.4256 -> 55.5).

## 4. The loader's contract

`Tower::open` reads only the file's header, then every tensor it needs is
checked - dtype, every dimension, and `elements x dtype size ==` the byte span -
before it is read and uploaded. A checkpoint that disagrees is refused by name
rather than reshaped. It also reads one `comfy_quant` blob and checks the group
size it states against `GROUP`, so the route's 256 comes from the artifact.

The reader keeps one host buffer and reuses it, so a 48-layer load holds one
tensor at a time in host memory: 13.7 GB of weights reach the device without a
second copy.

## 5. The gate, part one: the op probe

`h3_gemma_gate --probe --rows 8` runs one device kernel against the oracle's
transcription of it, on synthetic BF16 operands, and compares value by value.
Every one of the eleven checks is **bit-for-bit**, on both hosts:

| Probe | Shapes |
|---|---|
| convrot linear | layer 0 `q_proj` 3840->4096; layer 5 `q_proj` 3840->8192, `k_proj` 3840->512, `o_proj` 8192->3840 |
| RMS norm | 3840 wide, with the layer's weight |
| attention | sliding (16 heads x 256, 8 kv) and full (16 x 512, 1 kv) |
| RoPE | sliding and full, query and key, with the device's own pair tables |

The probe exists because a layer's output difference says nothing about which
step produced it; when the layer comparison below disagreed, this is what
localised it.

## 6. The gate, part two: the layer oracle, and the one difference

`--oracle` runs the reference forward again in Rust - the four norms, the two
attention flavours, the three MLP projections, both residuals and the scalar -
over the same checkpoint, and compares every hidden state with the device's.

To make the comparison about the graph rather than about `cosf`, the oracle is
handed the **device's** pair tables and masks, read back; both are already gated
on their own (`gemma_tower_bm`), and building them twice would inject a known
last-bit difference into the middle of a comparison about wiring.

Three things then have to be reproduced exactly, and are:

- the RMS norm's reduction tree (`h3d_rms_norm_kernel`: 256 partials striding
  by 256 with `fmaf`, then `sums[t] += sums[t + stride]`);
- the attention's, which is the same tree for the max and the sum plus a
  32-lane shuffle-down dot with `fmaf` and a final `fmaf` accumulation in V;
- the rounding structure of every elementwise step, one rounding per torch op.

### The one irreducible difference

`rsqrtf` on the device has no host counterpart. `1.0/sqrt()` is the closest
host evaluation - close enough that **five of six layers are bit-for-bit** -
but where the two disagree in the last bit, the INT8 route turns that into a
whole row. Each activation row's scale is `bf16(absmax / 127)`, so a last-bit
move of one element can step the scale by a BF16 ulp, which re-quantises every
value in that row and reads out across that row's outputs.

Measured on vizzio, `--layers 6 --rows 8`:

| State | differ | rel RMS | rows |
|---|---|---|---|
| 0 embedding | 0 | 0 | none |
| 1..5, sliding layers 0-4 | 0 | 0 | none |
| 6, full layer 5 | 2651 of 30720 | 2.66e-3 | 1 (row 4) |

And the mechanism, isolated by changing only that one evaluation: writing the
oracle's inverse as `powf(-0.5)` instead of `1.0/sqrt()` makes **every row of
every state** differ, at 2.3e-2.

The amplification compounds with depth: a row-local difference is a different
input for the next layer, whose own quantisation steps in turn. At
`--layers 6 --rows 64` every row has moved and the state deviation is 2.9e-2.
So the oracle is a **wiring proof** and the gate refuses to run it past 16 rows;
a deep run is gated on its state hashes instead, against the operator's golden.

## 7. Measurements

Both hosts, the same checkpoint
(the released Gemma-4-12B INT8 convrot shard, 15,372,969,374 B).
The tower reads 994 of its tensors, 12,923,715,168 B:

| Run | vizzio sm_89 | Spark sm_121 |
|---|---|---|
| `--probe --rows 8` | 11 of 11 bit-exact | 11 of 11 bit-exact |
| `--layers 6 --rows 8 --oracle` | 5 bit-exact, 1 row-local | 5 bit-exact, 1 row-local |
| the row-local state | 2651 differ, 2.664e-3, row 4 | 2651 differ, 2.664e-3, row 4 |
| `--layers 6 --rows 1024` | 126 tensors, 3.38 GB, load 29.5 s, run 0.199 s | load 0.907 s, run 0.302 s |
| `--layers 48 --rows 1024` | does not fit (12 GB card) | load 7.893 s cold / 2.564 s warm, run 2.207 / 2.181 s |

The two hosts agree digit for digit, including the one amplified row.

Two more statements about the 48-layer run, both repeated on the Spark:

- **Determinism**: two consecutive runs printed identical `STATE` lines.
- **Prefix consistency**: `--layers 6 --rows 1024` printed the first seven states
  of the 48-layer run byte for byte, so the deep run is anchored to a short one.

### The 49 states, Spark, 1024 rows

These are the lines the operator's golden is compared against; each is FNV-1a
over the state's 3,932,160 BF16 values.

    STATE 0  49f7f31903c38851    STATE 25 7b892949c51b9706
    STATE 1  95c2aef79dfc72c1    STATE 26 e4a9f19fa750839a
    STATE 2  88cd75926bc4749d    STATE 27 b57457494d7ae5b6
    STATE 3  163267816271f2aa    STATE 28 9e06840edff78a89
    STATE 4  dc4c2790326d5cae    STATE 29 7dc56dd994f5b1b9
    STATE 5  7874c86603902859    STATE 30 e99b1896d72de87a
    STATE 6  00f9956ad6b980dc    STATE 31 a368a6fbf9695f54
    STATE 7  45fd5ae351403139    STATE 32 c7f496cce5596b51
    STATE 8  ed870263a178b791    STATE 33 045be58b8721dec8
    STATE 9  7db7e55d9959e19c    STATE 34 07517eb40ae3b1d1
    STATE 10 1b3fd09ee68cb3c3    STATE 35 ee6ae4ae29787bf3
    STATE 11 4b65bc2f4d98a4fd    STATE 36 0e7670895733a6a2
    STATE 12 1a1cd42528784e41    STATE 37 9a27cff4d2c8c47b
    STATE 13 eeabb745b78ffc00    STATE 38 68f6e4d3cca47695
    STATE 14 a666c8398969529c    STATE 39 ede7aabadbf50d45
    STATE 15 ca51116de32c219a    STATE 40 4d493ad9ec1163c9
    STATE 16 8623b21b328c0e58    STATE 41 cb7f98a398a0fc8f
    STATE 17 da6d9d4d556332cd    STATE 42 2810d1a171863b56
    STATE 18 8c1c2fd9eecc34de    STATE 43 0510b1ef6d4aecc3
    STATE 19 b306e0b176fe6487    STATE 44 feb27e177b3d8c86
    STATE 20 d7fe901590a759a2    STATE 45 f28af8052f1a14e9
    STATE 21 5c0bdf99f2c35aa5    STATE 46 39797e77dbafd539
    STATE 22 91fd963eb3c1bbbb    STATE 47 814d5eff59dd67b5
    STATE 23 f6cc6e94b848c521    STATE 48 b71758c08c514a0b
    STATE 24 eba9c784d446a03f

The run's own log is `~/ds4-test/gemma-full-a.log` on the Spark; the second
run's is `gemma-full-b.log` and the prefix run's `gemma-prefix.log`.

## 8. Limits

- `rows <= H3D_GEMMA_ROWS_MAX` (4096), from the attention kernel's shared score
  vector; the release's tokenizer truncates to 1024, so the encoder never asks
  for more.
- A full 48-layer run is Spark-only: 13.7 GB of weights plus the states and the
  eight convrot workspaces do not fit a 12 GB card. A layer-limited run fits
  the workstation.
- The oracle is a small-row wiring proof, for the reason in section 6.
- The tower's own gate is the operator's golden, still owed; the gate prints one
  FNV-1a line per state so the comparison needs no re-derivation.

## 9. What units 3, 4 and 5 still need

- Unit 3, the tokenizer: `tokenizer.py:30-63`, whose assets are U8 tensors
  inside the same checkpoint. The gate's token ids are synthetic until it lands
  - the golden carries the release's own.
- Unit 4, the connector: the eight 1D blocks and the aggregate projection, BF16
  weights until the missing int8 file arrives.
- Unit 5, the cache payload.
