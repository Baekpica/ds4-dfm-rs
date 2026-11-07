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
| Tokenizer and chat adapter | implemented, `crates/ds4-core/src/tok/spark2_5.rs` |
| Forward pass (36 layers, fused qkv, dual rope, serial KV) | implemented, `ds4_spark2_5_graph.inc`, `cuda/spark2_5.cuh` |
| Full-layer KV as E4M3 (`DS4_SPARK2_5_FULL_KV=fp8`) | implemented, smoke-gated (see KV below) |
| Serving lane (banks, disk KV, snapshots, drafter) | not implemented, declared absent |

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

### Weights

The graph does not decode VQ per projection. `spark2_5_gpu_bind` restores the
180 matrices once per engine into one BF16 image (`ds4_weights.spark_image`,
7.03 GiB) with the family's own kernel (`cuda/spark2_5.cuh`
`spark_restore_kernel`, `ds4_gpu_spark_restore`), which reads both container
versions and applies the stage-2 term in FP32 before the single BF16
rounding. Each projection is then one BF16 GEMM over its slice
(`ds4_gpu_v41_matmul_bf16_dev_tensor`, cuda/ds41_dense.cuh:228-253), the same
entry the DeepSeek vision tower uses for device-resident weights.
`DS4_SPARK2_5_RESTORE=0` skips the image and restores one matrix into a
52.4 MB graph scratch before its GEMM.

The restore is verified element-wise: layer 0 `down_proj`'s first vector
matches a hand decode of the container bit for bit
(`DS4_SPARK2_5_DUMP_WEIGHT=<file>` dumps it for the comparison).

### KV

A BF16 row of one layer is 2 x 1,024 x 2 B = 4,096 bytes. Sliding layers keep
`window + chunk` rows (512 + 512 by default), so their 27 rings stay small;
the nine full layers keep the whole context, which is the cost that decides
the serving context:

| Context | Full layers BF16 | Full layers E4M3 | Sliding rings (27) |
|---|---|---|---|
| 8,192 | 288 MiB | 145 MiB | 108 MiB |
| 32,768 | 1,152 MiB | 580 MiB | 108 MiB |
| 131,072 | 4,608 MiB | 2,320 MiB | 108 MiB |

`DS4_SPARK2_5_FULL_KV=fp8` stores the nine full layers as the row Beam, MiMo
and Gemma-4 already ship: E4M3 K | E4M3 V | one f16 scale per KV head for
each, 2,064 bytes against 4,096 (`cuda/gemma4_attn_fp8.cuh:14-20`,
`ds4_gpu_g4_kv_store_fp8` and `ds4_gpu_g4_attn_fp8`). Sliding layers stay
BF16; their window bounds them. The switch is read where the graph is created
(`ds4_spark2_5_vq.h`), so the allocator, the session fit and the host quote
(`serving_host.rs` `spark2_5_kv_bytes`) price the same rows.

Measured, not yet a quality gate: at 131,072 with `DS4_SPARK2_5_RESTORE=0`
the graph allocates 2.37 GiB of KV and answers "The capital of France is
Paris" and "1+2 = 3" correctly; the FP8-vs-BF16 teacher-forced NLL and top-1
gate at 32K-131K is still required before this lane is called qualified.

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

Two container versions are read, told apart by `spark2_5.format_version`:

- 2, one 13-bit stream per matrix: 829 trunk tensors, 1,262,301,824 bytes.
- 3, two stages per matrix: 1,189 trunk tensors, 2,043,237,120 bytes. Each
  matrix gains a `.stage2` GVQ1 stream and its `.vq_stage2_book`, linked by
  `spark2_5.stages`, and the reconstruction is
  `W = gain1*(C1[i1] + R8[j]) + gain2*C2[i2]`. Measured 10.10% weighted mean
  relL2 against 23.19% for the one-stage build of the same checkpoint
  (`quality.json`, `family.json`). The stage entries are optional in the
  layout table, so a version 2 release binds without them; the artifact
  validator (`validate_stages`) makes them all-or-none inside version 3.

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

## Tokenizer and chat

The trunk carries the source vocabulary under GGUF's own namespace, and the
adapter refuses any other pre-tokenizer:

- `tokenizer.ggml.pre` is `spark2_5`; 131,072 tokens and 130,716 merges. The
  chain is `Split(\p{N}{1,3})`, `Split(CJK)`, `Split(word/punctuation)`,
  `Digits(individual_digits)`, `ByteLevel(add_prefix_space=false,
  use_regex=false)`, then BPE with no byte fallback and no `fuse_unk`.
- `bos` 0, `eos` 1 and `pad` 2, `unk` 5. `<｜end▁of▁sentence｜>` (1) is both the
  turn close and the only stop token; `<｜end▁of▁text｜>` (7) exists in the
  vocabulary and the template never emits it, so a stop set is built from the
  id, not from a rendered string.
- 113 added tokens. The 91 the source marks special are GGUF control tokens
  (type 3) and are matched atomically by the shared control split. The 22
  marked special=false (ids 3–5, 8–13 and 130972–130984) are type 1 in GGUF,
  so the adapter registers them by id; otherwise `<|Bot|>` would pre-tokenize
  as punctuation and letters instead of one piece. Ids 130986–131071 are the
  reserved `<||unused2..87||>` slots and are not stop candidates.
- The chat template is the checkpoint's `chat_template.jinja`, carried
  verbatim under `tokenizer.chat_template` and verified against its pinned
  SHA-256 at load. It is Jinja-only: the legacy chat builder refuses the
  family. The request effort is restricted to `high` and `low` (the template
  raises otherwise), and the generation prompt opens `<think>` unless
  thinking is explicitly off.
- Decoding maps bytes back through the GPT-2 byte table; the role markers
  contain U+FF5C and therefore print literally.

The gates are in `crates/ds4-core/tests/spark2_5_tokenizer.rs`. The chat
fixture is model-free (`tests/fixtures/spark2_5/`); the tokenizer gate reads
only the release's trunk metadata behind `SPARK2_5_RELEASE`, like the other
families. `make_eval_cases.py` regenerates both from the reference pipeline
(`tokenizers` for ids, Transformers' own Jinja environment for bytes) so the
fixture never depends on this tree's Rust code; it also takes an optional
fuzz-case count for a wider differential run.

On the server the family maps to syntax 24 and its input render refuses
without the official template, as Gemma-4 does. Output parsing reuses the
`<tool_call>name<arg_key>k</arg_key><arg_value>v</arg_value></tool_call>`
envelope and `<think>`/`</think>` channel that the GLM path already implements,
which is the grammar pinned in `token-check.json`; it stays unqualified until
a live batch measures it.

## Forward pass

`ds4_spark2_5_graph.inc` runs the reference's graph layer by layer; the host
arms it only on one full CUDA release with no sidecar (the artifact has no
MTP part). The facts that are easy to get wrong, with their source:

- The fused `q_k_v_proj` is one 6,144-wide output sliced q `[0, 4096)`,
  k `[4096, 5120)`, v `[5120, 6144)` (modeling_spark.py:170-176). A split
  kernel distributes it per row; the attention wants the three contiguous.
- `g_proj` produces one score per query head, and `sigmoid(score.float())`
  multiplies the head's 256 values before `out_proj`
  (modeling_spark.py:194-205). The multiply is the shared dots3 headwise gate
  (`ds4_gpu_dots3_gate_mul_tensor`, ds4_cuda.cu:42100), which takes the head
  width as an argument; step37's gate is not it (it hard-codes 128).
- The attention scale `1/sqrt(256)` (modeling_spark.py:146, applied at :84)
  is folded into q once after the rope. `g4_attn` takes no scale because
  Gemma's attention has none; 2^-4 is exact, so `(q * 2^-4) · k` equals
  `(q · k) * 2^-4` bit for bit.
- Both rotary geometries come from `compute_rope_cos_sin`
  (modeling_spark.py:31-40): the first `2 * pairs` dims of a head rotate
  against the offset `pairs`, with `pairs = 32` on the nine full layers
  (64 rotary dims, theta 5e6) and `pairs = 128` on the twenty-seven sliding
  layers (all 256 dims, theta 1e4). `cuda/spark2_5.cuh` implements that
  prefix pairing; the gemma kernel's `i + head_dim/2` offset is a different
  geometry and is not reused.
- 4:1 GQA, sliding window 512 with the query included on 27 layers, full
  causal on 9; exact-erf GELU in the 10,240-wide dense FFN; tied head over the
  BF16 token table; F32 residual stream.

KV rows are BF16 like the reference's cache: a sliding layer keeps
`window + chunk` rows indexed by position, a full layer keeps the session
context. `DS4_SPARK2_5_PREFILL_CHUNK` sets the chunk (default 512, max 1024).

## Gates: what was measured, and against what

The release's matrices are VQ-encoded. `artifact/manifest.txt` records a mean
per-matrix relL2 around 23% (layer 0: qkv 23.71%, down 22.44%), so the
artifact and the unquantized checkpoint are **not** the same weights and no
tight logit comparison between them can pass. Two different gates exist and
they answer different questions:

1. Release vs the checkpoint (the committed fixture
   `tests/fixtures/spark2_5/logits-reference.json`, a 5-token prompt, F32 CPU,
   Transformers 4.57.1): the argmax matches (`19500`, " Paris"). The logit
   vector does not: the checkpoint's top-1 is -1.65, the release's 2.59. This
   gate proves the graph reads the prompt and ranks the right token; it does
   not prove numerical agreement.
2. Release vs a reference run on the **dequantized** weights (VQ_tools'
   `vq-gguf decode` reconstructs each GVQ1+GVR8 pair; the dequantized
   checkpoint feeds modeling_spark.py; fixture
   `logits-reference-dequantized.json`): same argmax, top-64 relRMS 7.6e-3,
   63 of 64 ids shared, top-1 logit 2.59 against 2.56.

A per-stage walk (`make_hidden_reference.py`, `compare_hidden.py`, driven by
`DS4_SPARK2_5_DUMP_HIDDEN=<prefix>`) pins the shape of the error:

| Stage | vs checkpoint | vs dequantized weights |
|---|---|---|
| `embed` | 0 | 0 |
| `norm_attn0` (copied BF16 norm) | 0 | 0 |
| `layer1` | 2.70e-1 | 3.23e-3 |
| `layer18` | 6.33e-1 | 6.12e-3 |
| `layer36` | 7.14e-1 | 7.94e-3 |
| `final` (norm output) | 7.68e-1 | 8.01e-3 |

The controls are the two exact zeros: the embedding table and the first RMS
norm are copied BF16, not encoded, so a nonzero there would be a graph defect
upstream of attention. The checkpoint column starts at the first encoded
projection and compounds smoothly — that is the codebook, not the graph. The
dequantized column stays under 1e-2 through 36 layers; the residual is BF16
rounding and accumulation order (this graph keeps F32 scores, gate and
residual operands where the reference's tensor ops round, as the deviation
list in `ds4_spark2_5_graph.inc` names).

That second column is the numerical gate, and it is the one to quote. The
first is an argmax gate. Reporting the first as "matches the reference" would
be false at any tight tolerance; reporting the second as bit identity would be
false too (1e-2, not 0).

## End to end: the same eight tokens as the reference

Greedy generation is the strongest gate this pair can give, and it passes
when the flag is named: with the same prompt ("The capital of France is")
and `enable_thinking=False`, the engine and the dequantized-weights reference
produce the same eight tokens character for character,
"您好，我是一位名叫“小鹿". With `enable_thinking=True` both produce
"不起劲|\n\n用户提出了一个新的" — also identical. The flag matters: the
earlier apparent mismatch compared a no-think engine run with a reference
that defaulted to thinking on, and the two prompts are different (21 versus
72 tokens). `test` runs are recorded here with the flag because without it
the comparison looks like a failure rather than a pass.

## OpenAI surface

`ds4-server` serves the release; measured on the RTX 4070 SUPER (12 GiB),
ctx 4096, graph shared restored weights (7.03 GiB device BF16):

| Request | Result |
|---|---|
| `GET /v1/models` | 200, context_length 4096, tools and reasoning_effort advertised |
| `POST /v1/completions`, "The capital of France is", 8 tokens | 200, " Paris, which is the capital of the", TTFT 130 ms, decode 17.1 tok/s |
| `POST /v1/chat/completions`, 72-token template, 64 tokens | 200, content empty, 64 tokens in `reasoning_content` (thinking on) |
| same with `"think": false`, 16 tokens | 200, content "您好，我是一位名叫“小鹿”的AI助手，我以“", TTFT 127 ms |
| same, `stream: true` | 200, role chunk then one delta per token |
| tool request, `think: false` | 200, prompt rendered with the tool schema, no `tool_calls` (base checkpoint, not instruction-tuned) |

Three honest notes. Thinking on spends the completion budget inside the
reasoning block; `reasoning_content` carries it and `finish_reason` is
`length`, so a caller that wants visible text sets `"think": false`. Tools are
plumbed (render, parse, replay) but this checkpoint does not emit calls. The
7.03 GiB restored image is allocated before the serving plan is quoted, so the
default 4 GiB floor no longer fits on a 12 GiB card; start that card with
`--mem-floor-gb 1`. Both belong in the scheduler accounting, not only in the
log (handoff Unit 4).

## Not in this unit

Capability parity with Gemma-4 VQ is five lanes plus a qualified context, and
each is absent rather than promised. Gemma-4 declares banks Persistent with
Qualified support, reuse Partial/Qualified, disk Qualified, snapshot
Qualified, an MTP sidecar Present, and ctx_max 262,144
(crates/ds4-core/src/serving.rs:986-1013). Spark declares Serial/None,
Exact/None, disk None, snapshot None, MTP None, and ctx_max None (:1025-1043).
The order below is the one that unblocks the most; none may be marked
Qualified without the gate named here.

1. ctx_max: the model's native maximum is 1,048,576 and 128K now fits with
   E4M3 full-layer rows, but the measured FP8 gate above and a 1M fit are
   still missing, so the declared context stays unqualified.
2. reuse: sync_keep over the ring (window + chunk), gated by an identical
   resend and a re-rendered continuation. serving-check already reports
   prefix 16/17 and continue 17/17 as works-but-undeclared.
3. disk and snapshot: a session payload that serializes the sliding ring and
   the full-context rows, plus the layout word `payload.rs` has no writer for.
4. banks: the bank driver must accept this graph (follow
   ds4_gemma4_batch.inc; the restored weights are already engine-shared).
5. drafter: none exists for this artifact. The DSpark sidecar in the
   checkpoint tree is a DFlash drafter for Spark X2.5 (architecture dflash,
   block_size 7, target layers 3/11/19/26/34), not X2.6.

The FP8 lane is per family, `DS4_SPARK2_5_FULL_KV=fp8`, and does not come
from `DS4_CUDA_FP8_KV`: that switch sizes the DeepSeek compressed-KV lane
(serving_host.rs:2201, `comp_row` from `n_rot`/`n_indexer_head_dim`) and
never reaches this graph.
