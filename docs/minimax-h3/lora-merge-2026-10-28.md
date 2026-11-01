# Merging the FastH3 distilled adapter into a derived checkpoint (2026-10-28)

The port serves the un-merged base MiniMax-H3 DiT, so every render needs twenty
denoise evaluations. The reference does it in four, and it can because it
merges `FastH3-VSA/vsa-datafree/adapter_model.safetensors` before its FP8
conversion (the release's `configs/default.json:6-9`: `lora:
FastH3_VSA_DataFree`, `dit: W8A8_FP8_after_BF16_LoRA_merge`, `updates: 4`).
Twenty evaluations are also the blur seen at two steps: at two neither arm has
converged, at twenty the picture is finished. This unit performs that merge
offline, in `crates/ds4-core/src/minimax_h3/lora.rs` and the `h3-lora-merge`
binary; the released checkpoint is only ever read.

## 1. Step 0: the naming and packing question, answered

The adapter states its deltas for the *diffusers* packaging, which is not the
packaging this engine loads:

| adapter says | the served checkpoint holds |
|---|---|
| `transformer_blocks.N.attn.to_q` / `to_k` / `to_v` | one `blocks.N.attn.qkv_proj.weight` |
| `transformer_blocks.N.ff.net.0.proj` | `blocks.N.mlp.fc1.weight` |
| `token_refiner.refiner_blocks.N.*` | `token_refiner.blocks.N.*` |
| `norm_out.linear` / `proj_in` / `proj_out` / `context_embedder` / `audio_proj_in` / `audio_proj_out` / `time_embedder.linear_1` / `linear_2` | `final_layer.adaln_proj.linear` / `video_patch_proj` / `final_layer.video_out` / `condition_proj` / `audio_patch_proj` / `final_layer.audio_out` / `time_embedder.proj_in` / `proj_out` |

The mapping was established by comparing the two released packagings byte for
byte, one row range at a time, over HTTP range requests against
`MiniMaxAI/MiniMax-H3` - `transformer/` (diffusers, 638 tensors in 14 shards) and
`FL2VA/transformer/` (the engine's, 535 tensors in 13 shards). Two results came
out of it.

**The revisions are the same checkpoint.** The adapter's `base_revision` is
`9bfb6693f2cf6de171db46d1aa586f67d773a1da`; the copy on disk is
`42ed227ee7df40d41602854ae760620d6eb651fe` (the HF trees cache and the download
metadata dirs both record it). Every one of the thirteen shards has the same
LFS oid and the same size at the two revisions, and the index hashes to
`fb457a26ffa6294660e249b0ddd03a337f2e5393f770b5c34c8b8f90a29a7efb` at both.
There is therefore no revision deviation to name: the adapter's base is exactly
what this engine serves.

**Two tensors are reordered, and nothing else is.**

- `blocks.N.attn.qkv_proj.weight` is `3 * heads * head_dim` rows ordered per
  head, `[q | k | v]`: rows `384h .. 384h + 128` are `to_q` rows
  `128h .. 128h + 128`, `+128` is `to_k`, `+256` is `to_v`. Six consecutive
  128-row slots were compared and matched `to_q @ 0`, `to_k @ 0`, `to_v @ 0`,
  `to_q @ 128`, `to_k @ 128`, `to_v @ 128` in that order. This is also the
  layout the engine's own kernel reads: `h3_qkv_rope_kernel`'s `grouped`
  branch (`h3_gpu_cuda.cu:3093-3097`) is reached for the DiT at `h3_dit.c:1912`
  through `h3_gpu_grouped_qkv_linear_rope_bf16` (`h3_gpu_cuda.cu:3241-3250`),
  and the port mirrors it (`native/h3/h3_dit_gpu.cu:1004-1006`).
- `blocks.N.mlp.fc1.weight` holds the FFN's two halves **swapped** relative to
  `ff.net.0.proj.weight`: engine rows `0..14336` are the packaging's rows
  `14336..28672` and the other way round. Both engines independently agree on
  which half is the gate - the reference kernel takes the first
  (`h3_gpu_cuda.cu:2842-2851`: `gate = fused[base + column]`,
  `up = fused[base + width + column]`) and FastVideo the second
  (`models/dits/minimax_h3.py:126-129`: `hidden_states, gate =
  hidden_states.chunk(2, dim=-1)`).
- Every other path is the same tensor value for value: 21 of 21 identity pairs
  compared byte for byte, covering the main block's `out_proj`, `fc2`, AdaLN,
  both norms and both qk norms, the token refiner's `out_proj`, `fc2`, `norm1`,
  `norm_q` and `final_norm`, and all nine global projections on both weight and
  bias.

**The adapter supplies one parameter the checkpoint does not have.** All fifty
`.set_weight` payloads address `transformer_blocks.N.attn.to_gate_compress`, and
neither packaging carries a `to_gate_compress` tensor - nor any tensor whose
name contains `gate` or `compress`. FastVideo's own loader documents this exact
case as the reason the payload class exists (`models/loader/lora_patch.py:19-27`):
"MiniMax H3's VSA `to_gate_compress` ... exists only under the sparse-attention
backend, and `load_model_from_full_model_state_dict` otherwise zero-initializes
it, which is exactly the 'gate contributes nothing' state a VSA-distilled
student was trained away from". The gate feeds VSA's compression branch, which
attends the pooled tiles densely and is merged under the gate
(`video_sparse_attn_h3.py:625-641`); the selection path does not read it
(`docs/minimax-h3/vsa-policy-2026-10-28.md` section 2.6), and this port
implements the folding consumer instead. The tool therefore writes those fifty
weights beside the derived tree, under `set_weight/attn_gate_compress.safetensors`,
where the engine's per-component shard scan can never see them: `layout.rs:254-257`
refuses an unlisted tensor in the transformer directory.

## 2. What the adapter states, read from the reference

Source of truth is FastVideo at `3d8ac9d14bd697a89ede8f170cbfbca012a9edcc`, the
reference's own pin (`configs/dependencies.json` `sources[fastvideo_root]`),
fetched pristine and checked against the reference's manifest
(`patches/fastvideo-spark-loader.json` records both hashes):

- `models/loader/lora_patch.py:53-56`: `.diff` targets `.weight` and `.diff_b`
  targets `.bias`, both additive; `.set_weight` targets `.weight` and replaces.
  Longest first, so `.diff_b` never matches as `.diff`.
- `:170-186` `apply_to`: `tensor.to(float32) + delta.to(float32) * strength`.
- `:201-210` `replacement_for`: the supplied value times strength is the whole
  weight, because the loader zero-initializes the absent parameter.
- `layers/lora/linear.py:183-195`: `lora_delta = B @ A` with both operands cast
  to the base weight's dtype, `scale = alpha/rank if alpha and rank else 1.0`,
  then `data += lora_delta`. The adapter states no `alpha` key of any kind
  (856 keys: 362 `lora_A`, 362 `lora_B`, 24 `.diff`, 58 `.diff_b`, 50
  `.set_weight`), and the reference sets `lora_strength` to 1.0
  (`runtime/stage1.py:301`), so the scale is exactly one.

Both orderings are mirrored: an additive delta is summed in F32 and rounded to
the destination dtype once, and a rank-64 product is rounded to that dtype
before the add - which is what a bf16 `torch.matmul` followed by an in-place
bf16 `+=` does.

## 3. The tool

    h3-lora-merge --model <MiniMax-H3> --adapter <adapter_model.safetensors> \
                  --out <directory> [--threads N] [--check]

`--check` resolves and prints the plan and writes nothing. On the released
checkpoint it reports: 340 served tensors touched - 258 rank-64 modules (50
`qkv` interleaves, 50 `fc1` half swaps, 154 straight) - 82 additive deltas, 50
whole-parameter modules, and zero unresolved keys.

The merge refuses to skip anything: an adapter key no rule places is an error,
as is an adapter whose header counts disagree with its own metadata, a
destination whose rows are not covered exactly once by the deltas, or a `--out`
that already exists. It writes one shard per released shard, in the released
tensor order, so a shard with no touched tensor is byte for byte its source -
which the verification below uses.

## 4. The derived artifact, and what was checked about it

The merge ran on the Spark in 53.3 s, writing
`/mnt/h3-artifacts/MiniMax-H3-FastH3-4step`: 13 shards, 535 tensors, 340
touched, 66,280,430,144 bytes of tensor data, the released tree hard-linked
around it and the fifty gate weights beside it. The released `MiniMax-H3`
directory is untouched.

Three checks, none of them sharing code with the merge tool:

- **The untouched tensors are the released bytes.** All 535 tensors were
  compared byte for byte between the two trees: 195 identical, exactly the 340
  the merge reported as touched. This is what the writer's released-order layout
  is for.
- **The merged tensors agree with an independent implementation.** Three
  touched tensors were recomputed in float64 with numpy from the adapter and the
  released weights - the `qkv` interleave, the swapped `fc1`, and the 96,768-row
  AdaLN - and compared against the derived bytes:

  | tensor | values | differ | worst |
  |---|---|---|---|
  | `blocks.0.attn.qkv_proj.weight` | 115,605,504 | 3 | 1 bf16 ulp |
  | `blocks.0.mlp.fc1.weight` | 154,140,672 | 3 | 1 bf16 ulp |
  | `blocks.0.adaln_proj.linear.weight` | 260,112,384 | 10 | 1 bf16 ulp |

  Sixteen values of 530 million differ, each by one ulp of bf16, which is the
  expected effect of an F32 accumulation against an F64 one flipping a rounding
  boundary. Both stay well inside the mixed-quant tolerance the repository's
  goals state.
- **The plan is total.** `--check` resolved all 806 placeable keys onto 340
  served tensors with no unresolved key, and the row coverage is asserted at
  plan time: every destination row is written by exactly one delta.

## 5. The render gate

Job A throughout: 768x768, 107 frames, seed 42, the prompt "a red fox walking
through deep snow at sunrise, cinematic", one binary, `sm_121a`, 2.4-2.5 GHz
(the host has no clock policy).

| run | model | steps | decoder | attention | wall | TEXT | LOAD | DENOISE | FRAMES |
|---|---|---|---|---|---|---|---|---|---|
| 1 | released | 20 | default | dense | 628 s | 10.838 | 7.503 | 406.454 | 198.856 |
| 2 | released | 20 | fast | sparse 0.9 | 336 s | 27.779 | 8.218 | 258.386 | 36.861 |
| 3 | derived | 4 | default | dense | 353 s | 27.993 | 39.976 | 79.391 | 200.985 |
| 4 | derived | 4 | fast | sparse 0.9 | 165 s | 28.144 | 40.169 | 50.834 | 40.864 |
| 5 | derived | 4 | default | sparse 0.9 | 328 s | 28.212 | 42.125 | 50.320 | 203.071 |
| 6 | derived | 4 | default | dense | 362 s | 28.209 | 45.847 | 81.147 | 202.391 |
| 7 | released | 4 | fast | sparse 0.9 | ~166 s | 28.437 | 43.402 | 50.271 | 40.335 |

### The step count the merge buys

DENOISE falls from 406.454 s to 79.391 s with the default arms - 5.12x - and
from 258.386 s to 50.834 s with the sparse attention and the fast decoder -
5.08x. Both are the step ratio itself, five against one, which is why the
measurement is robust to everything else on this host.

The wall falls less, because the merge changes nothing about the other phases:
with the default decoder the 201 s of FRAMES caps the win at 1.78x, with the
fast decoder (40.9 s) at 2.04x. The whole stack against the reference
configuration is run 1 to run 4: 628 s to 165 s, 3.8x, or 4.0x if run 1's LOAD
is put on the same footing as the later runs (see below).

### The merge's own contribution, isolated at four steps

Runs 4 and 7 have the same steps, the same arms and the same seed, and differ
only in the checkpoint. They cost the same: DENOISE 50.834 s against 50.271 s,
FRAMES 40.864 s against 40.335 s. What the merge changes is the picture:

- The un-merged model at four steps is under-denoised. The mid frame shows
  smeared foliage and soft fur (`frame-released4-sol-mid.png`), the same failure
  the operator saw at two steps and sharper only in degree.
- The merged model at four steps is a finished picture
  (`frame-gate-merged4-sol-mid.png`).
- Both are finished at twenty steps, which is run 2's picture.

A coarse focus measure on the mid frames puts numbers on that. It is
composition-dependent, so it ranks finished pictures against each other only
loosely; it is decisive for the under-denoised cases, where it separates by an
order of magnitude.

| render | laplacian variance |
|---|---|
| 2 steps, released, sparse + fast | 7.3 |
| 4 steps, released, sparse + fast | 30.6 |
| 4 steps, derived, sparse + fast | 62.2 |
| 4 steps, derived, default arms | 185.8 |
| 20 steps, released, sparse + fast | 48.2 |
| 20 steps, released, default arms | 38.9 |

**The same measure over every frame.** The figures above are one frame; the
whole render says the same thing. Mean of `|frame - luma-4 blur|` over all 107
frames of each render, one binary and one scale for every row:

| render | detail energy |
|---|---|
| 2 steps, released, sparse + fast | 0.824 |
| 4 steps, released, sparse + fast | 1.734 |
| 4 steps, derived, sparse + fast | 3.189 |
| 4 steps, derived, sparse + default | 3.191 |
| 4 steps, derived, dense + default | 4.563 |
| 20 steps, released, sparse + fast | 3.282 |
| 20 steps, released, dense + default | 3.212 |

This settles two things the mid frames could not. The derived four-step render
sits on the twenty-step references - 3.189 against 3.282 and 3.212 - while the
un-merged four-step render sits at half of them, 1.734, and the two-step render
at a quarter, 0.824. And the derived dense arm at four steps is the highest of
the set, 4.563, above both twenty-step renders; the four-step sparse arm in the
default decoder, 3.191, repeats the fast one to 0.002, so the decoder swap
barely moves detail while the checkpoint swap doubles it. The metric remains a
no-reference content-dependent high-pass energy: what makes it decisive here is
not the absolute value but that the twenty-step references land on the same
value as the derived four-step sparse arm.

### PSNR is the wrong instrument here

Against the twenty-step reference, the merged four-step renders score 15.76 dB
and 15.54 dB (SSIM 0.64 and 0.62). That is not a quality verdict. At four steps
against twenty the two renders are different trajectories from the same seed -
different framing, different light, both finished - and the repository's earlier
withdrawn comparison failed for the same reason. The measurement that decides
this unit is the fixed-step one above, where only the checkpoint differs, and it
is decided by looking.

### Reproducibility, and the LOAD artefact

Runs 3 and 6 are the same arm about twenty minutes apart and produced the same
file, sha256
`5fe31c57c06bc3848bd210e5709088abd18a84e605b9c1e13675a6dc3edab736`, so a render
on this host is byte-reproducible and the repeated default arm is a noise floor
of zero for the render itself. DENOISE repeated to 1.76 s between them (79.391
against 81.147), against a claimed 5.12x effect.

The LOAD column needs care and was initially misread. Runs 1 and 2 load in
7.5-8.2 s, every later run in 40-46 s, *including run 7, which loads the
released checkpoint*. The derived checkpoint is therefore not slower to load:
the early runs were taken while that tree had been resident in page cache for
hours, and afterwards both 66 GB trees compete for a 121 GB machine, so every
model now comes from disk. The derived shards themselves read normally - 8 GiB
in 1.65-2.15 s cold and 0.22 s cached, 13 extents against the released file's
16 - so the difference is host cache state, not the artifact. Wall comparisons
between runs 1-2 and runs 3-7 carry that ~35 s and should be read with it in
mind; the DENOISE and FRAMES columns do not.

## 6. What this record does not establish

- Parity with the reference engine's four-step output. Its schedule is
  `video_sigmas [1.0, 0.972972989, 0.923076928, 0.8, 0.0]`
  (`configs/default.json:8`); this engine's own sigma grid is what these renders
  used, so the step *count* matches the reference and the trajectory does not.
- That four steps should become the default. That is an adoption question for
  the operator, and the answer depends on whether the engine also ports
  `video_sigmas`.
- The reference also converts the merged model to FP8 (`W8A8`), which this
  engine does not load; the derived checkpoint stays in the released BF16, and
  the engine's own quantization at load is a separate step.
- It does not port VSA's compression branch, so the fifty gate weights are
  written for fidelity and read by nothing.
- Run 2's phase lines are second-hand: `~/demo20-summary.txt` was truncated to
  zero bytes by another session before this record was written, and those
  numbers come from the handoff, which was written from that file live earlier
  in the same session. Runs 3-7 were measured for this record, and run 1's
  lines are on disk in `~/demo20d-summary.txt`.
- The twenty-step sparse pair - runs 1 and 2 - is from two sessions and was
  never repeated. The sparse arms are compared properly at four steps instead,
  in runs 5 and 6 against run 3: DENOISE 81.147 and 79.391 against 50.320,
  1.60x, with the repeated default arm bounding the spread at 2.2%.
- The byte comparison of the two *packagings* sampled rows, not whole tensors:
  six qkv slots per family, thirty-two to sixty-four row samples per identity
  pair, and all thirteen LFS oids. The derived-against-released comparison in
  section 4, by contrast, covered all 535 tensors in full.
- Nothing here times the merged checkpoint at twenty steps, and nothing checks
  it against the reference engine's own frames: the reference's output is not
  reproducible on this box.
