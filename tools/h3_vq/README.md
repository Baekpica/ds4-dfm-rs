# tools/h3_vq — the MiniMax-H3 VQ converter

The converter toolchain for the **MiniMax-H3 FL2VA DiT**, transferred here so the
H3 conversion starts from a tool that already works rather than from scratch.
The sources are the ones the `embeddinggemma2` dense bake used; the plan's
[VQ recipe](../../docs/minimax-h3-vq-recipe.md) section 9 is the adaptation
checklist.

Status: **transferred, adapted for H3, and exercised end to end on a fixture;
the rung A encoder (GVQ1 v1 + GVR8) is implemented and gated.** `bake/h3_pack.c`
is the H3 driver — the bake's container, slot layout, plan/layout/merge/check
machinery and encoder calls, with the class table, the kept-name list,
`classify()` and the recipe string replaced. The unmodified bake driver is
still built as `bin/eg2_pack_cpu` for comparison. `bake/h3_gvq.c` +
`bake/h3_gvq.h` are the rung A / A+8 encoder and codec (section "Rung A").

### What has been tested

| Check | Command | Result |
|---|---|---|
| Builds from this directory, CPU only | `tools/h3_vq/build.sh` | 5 binaries, no errors |
| H3 classification | `bin/h3_pack_cpu plan /tmp/h3fix` | 9 VQ matrices into adaln / mlp-fc1 / attn-qkv / mlp-fc2 / attn-out / head; 11 kept tensors verbatim, including `audio_patch_proj` (cols 32) and the five other F32-by-design matrices, which the name list — not the shape — keeps |
| Encode, structure, read-back | `encode` then `check` | `structure OK: 20 slots, names, hashes, alignment, payload ranges` |
| Determinism | two encodes, `cmp` | byte-identical when the matrix has at least as many vectors as codewords (`sha256 7af650e2…`) |
| Fixture generator | `tests/make_fixture.c` | writes H3's names and dtypes at reduced shapes, so no checkpoint is needed for the tool's own gates |
| Rung A codec and encoder | `bin/h3_gvq selftest` | passes: byte account, 13-bit round trip, value rule, thread-count determinism, residual invariance, 20 GVQ1 + 5 GVR8 rejections, all three padding gaps |

## Rung A — the GVQ1 v1 payload (`h3_gvq`)

`bake/h3_gvq.h` is the format and `bake/h3_gvq.c` is the encoder plus the tool's
own gates. The contract is the M3.1 release's `FORMAT.md` field by field, and
where it says "canonical" the operational definition is the strict reader
already in this repository, `crates/ds4-core/src/glm_vq.rs:117-225` (offsets,
checksum placement, gain rules, zero padding). Codewords are E4M3FN through
`youngai/src/common/ds4_fp8.h`; checksums are `ds4_sha256`.

| Item | Rung A | Rung B |
|---|---|---|
| Vector dimension | 8 | 8 |
| Codebook | K 8192, 64 KiB E4M3 | K 4096, 32 KiB E4M3 |
| Index bits | 13 = 12-bit stream + 1-bit plane | 12 |
| Payload per matrix | `align128(128 + 2R + R·C·3/16 + R·C/64)` | `align128(128 + 2R + R·C·3/16)` |
| Rate | 1.625 bits/weight + one F16 gain per row | 1.5 |
| Residual (A+8) | GVR8, K 256, one index byte per 8 weights, shared row gain | — |

The encoder is the published recipe, not a variant of it: equidistant init (no
random source), a stratified pool of at least 1000 vectors per codeword, 32
Lloyd iterations, codewords rounded to E4M3 *before* the final assignment,
least-squares gain refit rounded to F16, and for the residual the target
`w/g - C1[i1]` against the stored gain and index with stage 1 untouched. With
no activation statistics yet it uses the reference's own zero-corpus mode
(uniform column weights); the column profile is the input the statistics step
will supply.

Run:

```sh
bin/h3_gvq selftest
bin/h3_gvq encode <safetensors-dir> <tensor> <prefix> \
    [--k 8192|4096] [--iters N] [--res-iters N] [--pool-per-code N] \
    [--threads N] [--residual] [--book-id N]
bin/h3_gvq check <prefix> [--src <safetensors-dir> --name <tensor>]
```

`encode` writes `<prefix>.gvq`, `<prefix>.book`, optionally `<prefix>.gvr` and
`<prefix>.rbook`, and `<prefix>.report` with the stage timings, the assignment
throughput, the byte account and the reconstruction quality. `check` re-reads
all of it through the strict reader, verifies every value against the value
rule, and compares against the source when given one.

Not implemented yet, and named so it cannot be mistaken for done: the
`imatrix` statistics producer (the column profile is an argument, not a
measurement), the per-block residual sweep, and the GGUF wrap. The residual
stage is fitted per matrix and on by request; which blocks get it is a
measurement the sweep will make.


Two findings from that run, both worth carrying into the real conversion:

1. **A degenerate matrix is not deterministic.** With 8 rows × 64 columns the two
   encodes differed in 132 bytes, all inside codebook regions: a matrix with
   fewer vectors than codewords leaves the unused codeword slots holding
   whatever the allocator returned. At 256×64 — exactly 4,096 vectors for a
   4,096-entry book — the two runs are byte-identical. Every H3 matrix has
   millions of rows, so this cannot bite in the real encode, but it is the
   reason gate 2 compares *bytes* and not cosines.
2. **The CPU encode rate.** 20 tensors and 305 KB of source in 4.4 s on this
   host, for 7 VQ matrices × 4,096 training vectors. H3 has ~8.3 × 10⁹ dim-4
   vectors, so the CPU path is a days-scale job even sharded across the Spark's
   20 cores; the GPU path is why the M3.1 release converted with CUDA. The two
   cost terms are the Lloyd solve on the encoder's capped training subsample
   and the full-matrix assignment pass.

## Provenance

| Component | Source | Revision |
|---|---|---|
| `bake/eg2_pack.c` — the bake's driver, verbatim; built as `bin/eg2_pack_cpu` for comparison | `/data/embeddinggemma2/tools/eg2_pack.c` | `e5d9e9ce30d5b1223999a0a35bc67d8e722a5e36`, branch `feat/eg2vq-attn-perf` |
| `bake/h3_pack.c` — that driver adapted to H3 (the four changes plus the removal of the unreachable block-scalar token branch) | derived from the file above | the same |
| `bake/eg2_vq.c` — single-tensor ladder tool (`tensor <hf-dir> <name> [dim nc]`) | the same | the same |
| `bake/eg2_gguf.c` — GGUF re-wrap of the container, with a manifest mode | the same | the same |
| `bake/ds4_gguf_write.h` — a complete GGUF v3 writer in one header | the same | the same |
| `bake/eg2_artifact_check.sh` — the four-sided artifact check | the same | the same |
| `youngai/gguf-tools/quantize/` — the VQ encoder (`vq_qc.h`, `quants.c`, `onebit_quant.c`, `v41_vq.cu`, `vq_gpu.cu`, …) | `/data/YoungAi` | `28944791d64c55b14e83ce0f5326ca5ebaac226e`, branch `feat/quality-inspection` |
| `youngai/src/common/` — the multi-shard safetensors index (`ds4_st41.h`), dtype helpers, the gold decoders, the GGUF writer | the same | the same |
| `LICENSE-YoungAi` | `/data/YoungAi/LICENSE` | MIT, Copyright (c) 2026 The ds4.c authors, (c) 2023-2026 The ggml authors |

Both directories were copied whole, not file by file, so the include closure is
complete by construction: `youngai/gguf-tools/quantize/` is 67 files and
`youngai/src/common/` is 16.

### Local edits to the copies

Two, both in `bake/eg2_vq.c`, both marked in place with a comment:

- `#include "../../YoungAi/src/common/ds4_st41.h"` → `#include "ds4_st41.h"`,
  likewise `ds4_float.h`. The originals were relative to the bake's directory
  layout (`tools/../../YoungAi`); here the include roots come from `-I`.

Everything else is byte-identical to the sources above. `cmp` against the
originals confirms it.

## Build

```sh
tools/h3_vq/build.sh          # CC=… OUT=… to override
```

CPU only: no CUDA, no checkpoint, no GPU. It produces `bin/h3_pack_cpu`,
`bin/h3_gvq`, `bin/h3_gguf` and `bin/h3_vq`, and the build lines are the bake's
own (`/data/embeddinggemma2/RECIPE.md` section 4) with the include roots pointed
here. `h3_gvq` is new H3 code and links only `ds4_sha256.c`. One correction to
that documented line: `eg2_vq.c` links **without** `onebit_quant.c`, because it
already carries the block-quantizer sources it uses; including that object
twice is a duplicate-symbol error.

## Run

```sh
bin/h3_pack_cpu plan   <safetensors-dir> [plan.tsv]
bin/h3_pack_cpu encode <safetensors-dir> <out.bin> [log] [--cold cls,…] [--limit N] [--only SUBSTR] [--shard i/N] [--profile]
bin/h3_pack_cpu verify <safetensors-dir> <in.bin> [log] [--shard i/N]
bin/h3_pack_cpu check  <in.bin>
bin/h3_pack_cpu merge  <safetensors-dir> <out.bin> <log> <shard.bin>…
bin/h3_vq tensor <safetensors-dir> <tensor-name> [dim nc]
```

The first argument is a **directory of safetensors shards**: `ds4_st41.h`
scans every `*.safetensors` in it and mmaps one shard at a time, on demand.
MiniMax-H3's `FL2VA/transformer/` — 13 shards plus an index — is exactly that
shape, so no staging copy is needed.

## What is CPU and what is GPU

The bake's encode was **CPU-only, as a measured decision**, for three reasons
its RECIPE.md section 8 records: that card has no `pageableMemoryAccess`, so the
reference kernels return `cudaErrorIllegalAddress` on `malloc` buffers; even
with pinned memory the GPU path was 1.25–3× slower per tensor over PCIe (89.6 s
against 58.8 s on a 92-tensor subset, identical cosines); and at that tier the
kernels cannot run at all, because `vq_qc.h` admits them only when the codebook
fits the shared-memory gate, and dim 4 with 4096 entries needs 81,920 bytes
against a 48 KiB default (`youngai/gguf-tools/quantize/vq_gpu.cu:47`).

On a GB10 none of the first two hold — it is a unified-memory part with no PCIe
hop — and the third is a limit that can be raised: the Spark allows 99 KiB
opt-in per block (101,376 B, see `AGENTS.md` "Shared memory caps tiling"), which
is above 81,920 B. So the GPU route is worth having here, and the two candidate
implementations are already in this directory:
`youngai/gguf-tools/quantize/vq_gpu.cu` (the kernels the bake drove through
`EG2_CUDA`) and `youngai/gguf-tools/quantize/v41_vq.cu` (the reference's
device-resident encoder for exactly this shape, which writes its own payload
layout and would need the reader to follow).

## What must change for H3

Per the recipe's section 9.2: `classify()`, `name_is_kept_matrix()`, the
`dim_of`/`nc_of` rung table, and `recipe_string()`; plus the imatrix statistics
step and the render gate, which are the genuinely new work. The sharding the
4 GiB `DQVD` ceiling requires already exists (`--shard i/N` and `merge`), and
H3 has exactly 13 natural shards.

The rung A format work is done in `h3_gvq` (the E4M3 codebook and the 13-bit
layout, plus the GVR8 residual stage); what remains on that path is the imatrix
producer, the per-block residual sweep and the GGUF wrap.

## License

`youngai/` and `bake/` are MIT, Copyright (c) 2026 The ds4.c authors and
(c) 2023-2026 The ggml authors; the notice is in `LICENSE-YoungAi` and must
travel with the code.
