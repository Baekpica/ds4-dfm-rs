# MiniMax-H3 on ds4-dfm-rs — port plan

[Documentation index](../README.md) | [Repository README](../README.md) |
[VQ recipe](vq-recipe.md) | [Performance references](../performance.md) |
[GB10 know-how](../../AGENTS.md) | [VQ calibration](../vq-calibration.md)

Status: **plan, nothing built.** Branch `feature/minimax-h3` off `dev` at
`600e465e`. Date 2026-10-24. Owner: the operator; execution on the DGX Spark
through the operator's two routes (an ssh tmux pane for commands, the gvfs sftp
mount for files), never through a third route.

## 1. Goal

A Rust CLI, in this repository, that generates video and audio from a text
prompt — with optional first/last-frame conditioning — on the DGX Spark (GB10),
matching the C reference engine's output, plus a vector-quantized weight tier
that is measurably faster while holding a declared quality floor.

In scope: the prompt-to-video/audio CLI, the first/last-frame conditioning path,
the reference checkpoint as the only weight source, and the VQ conversion of the
DiT.

Out of scope, explicitly:

| Excluded | Why |
|---|---|
| `webui/` (h3c studio: FastAPI backend, React front end, docker compose) | the operator does not want a web GUI; the compose route also needs the NVIDIA Container Toolkit and there is no `docker` on the Spark |
| Ref2VA | optional in the reference itself (`h3.c:438-451` keeps prompt-only FL2VA usable without the second 144 GB partition) |
| Metal / macOS backend, the Objective-C tokenizer | the port targets one host, sm_121 Linux |
| Face-swap post-processing | an external program the reference repository does not ship |
| Python in the deliverable | project rule 14; the reference's own Python is read as a specification, never shipped as a runtime dependency |

The reference is `/data/h3.c` (fork `JLFN/h3.c` of `matrixfede/h3.c`) at
`eae5d54`. The port reimplements it; the reference source is not modified, and
the reference *binary* is the oracle every gate compares against.

## 2. Verified state, 2026-10-24

Everything in this section was measured in this session, through the two
allowed routes. It is the ground truth the plan stands on.

### 2.1 The Spark (GB10)

| Fact | Value | How checked |
|---|---|---|
| Host | `gx10-e1d2`, aarch64, Ubuntu 24.04.2, kernel `7.0.0-1019-nvidia`, 20 cores | `uname -a`, `nproc` |
| Memory | 121 GiB total, 111 GiB free, 119 GiB available, 15 GiB swap unused | `free -g` |
| GPU | NVIDIA GB10, `compute_cap` 12.1, driver 580.178.04, max SM 3003 MHz, `memory.total` N/A (unified) | `nvidia-smi --query-gpu`, `nvidia-smi -L` |
| CUDA | 13.0, `V13.0.88`, `/usr/local/cuda` | `nvcc --version` |
| Buildable arch | `compute_121` is among `nvcc --list-gpu-arch` | the same |
| Toolchain | gcc/g++ 13.3.0, rustc/cargo 1.98.1 | `--version` |
| Media | ffmpeg and ffprobe 6.1.1 | `-version` |
| HF tooling | `hf` 1.28.0 at `~/.local/bin/hf`, `huggingface-cli` present, python 3.12.3, page cache for HF only 16 M | `hf version`, `du` |
| Disk | `/` 916 G with **547 G** available; `/mnt/models` 953 G (USB, ext4, label STORAGE) with **393 G** available and writable | `df -h`, a write test |
| Artifact disk | `/mnt/h3-artifacts` ext4 label `H3ART` (fstab by UUID, `nofail`), 938 G with **795 G** available, 1.3 GB/s cold direct read | `df -h`, `rsync -an -c`, a 512 MiB write test |
| Model area | `/mnt/models` holds 106 G ds41ﬂash-1017, 106 G ds41-1017, 106 G ds41-1017-unc, 126 G Qwen3.8 GGUF, 6.8 G Bonsai | `du -sh` |
| Repository on the Spark | `~/ds4-dfm-rs`, branch `main` at `32a41683` — the serving deployment, a different lineage from this task's `dev` | `git branch --show-current`, `git log -1` |
| Present | h3.c: **source at `~/h3-oracle`**, built at `eae5d54`. MiniMax-H3 checkpoint: **present** at `/mnt/h3-artifacts/MiniMax-H3`, copied and verified byte for byte | `ls`, `rsync -an -c` |
| ICU | `libicu-dev:arm64 74.2-1ubuntu3.1`, `/usr/include/unicode/unorm2.h` present, `pkg-config icu-i18n`/`icu-uc` 74.2 (the reference asks for 72 or newer, `README.md:73`) | `dpkg -l`, `pkg-config --modversion` |

One gap that is a prerequisite, not a risk to discover later:

- **`libicu-dev` was missing and is now installed** by the operator on
  2026-10-24; the reference links and runs. ICU is used by exactly one source
  file, `h3_tokenizer.c` (NFC, UTF-8/16 conversion, general-category split);
  nothing in the CUDA, DiT, VAE, weights or text-encoder path includes it. The
  Rust port needs none of it: NFC is the `unicode-normalization` crate already
  in `ds4-core`, and the split predicates are covered by the `regex` crate's
  `\p{L}`/`\p{N}`/`\s` classes.
- **No `docker`.** The compose-based path in the reference README is not
  available on this host, which is one more reason the port is native.

One delta worth stating plainly: the reference documents its validated
configuration as driver **595.84** with CUDA 13.0; this host has driver
**580.178.04** with the matching CUDA 13.0. The toolkit supports the arch and
the driver supports the toolkit, so the build is expected to work — but the
reference's own GB10 numbers were taken on a different driver, so every number
this plan quotes as a baseline is a *reference* number until P2 reproduces it
on this host.

### 2.2 The two hosts, and which one does what

The **Spark is the only execution host** for this project. Every GPU step runs
there: the reference build, the reference renders, the attribution, the
converter (its encoder is CPU-only but it reads a 144 GB checkpoint that lives
on the Spark, so running it anywhere else would mean copying the artifact), the
quantized-rung renders and the measurements. The local development machine
(x86_64, RTX 4070 SUPER, 12 GiB) is explicitly **not** part of this plan: it
cannot hold any part of the model (27.06 GB measured DiT peak, 465 GiB
checkpoint) and no number from it would describe the target. It is used for
editing the Rust port and for the model-free test tiers, nothing else.

## 3. The reference and the artifact

### 3.1 What the reference engine already measures

From `h3.c`'s own `docs/GB10_PROFILE.md`, on GB10, CUDA 13.0, driver 595.84,
median of three, with the released checkpoint (256×256, 22 frames, 2 steps, 35
layers):

| Mode | DiT load | Denoise | Load + denoise | DiT peak |
|---|---:|---:|---:|---:|
| Resident BF16 | 96.563 s | 2.715 s | 99.278 s | 27.063 GB |
| `--ssd-streaming` | 51.829 s | 84.798 s | 136.627 s | 1.630 GB |

All nine files share one SHA-256; baseline versus SSD measures SSIM 1.0. The
same document records the max-quality decode (1024×576, 107 frames): video VAE
decode 1112.960 s scalar → 247.737 s with the tiled F32 kernel, DiT denoise
815 s, wall 33:36.8 → 18:56.4.

Read together: **the load dominates a short render (97%), and the denoise
dominates a quality render (73%).** Those are two different problems and the
plan treats them separately.

### 3.2 The artifact, pinned

`MiniMaxAI/MiniMax-H3` at revision `42ed227ee7df40d41602854ae760620d6eb651fe`
(not gated, not private, `lastModified` 2026-08-13). 280 files, 498.5 GB
decimal = 464.2 GiB.

The repository carries two complete mode partitions and a shared set:

| Path | Files | Size | Content |
|---|---:|---:|---|
| `FL2VA/` | 81 | 144.05 GB | one full prompt/first-last-frame bundle: transformer (13 shards), text_encoder (14 shards), video_vae/source, audio_vae, processor, tokenizer, and the model's own Python sources |
| `Ref2VA/` | 81 | 144.05 GB | the ordered-reference bundle, same structure |
| root | 118 | 210.3 GB | `transformer/`, `transformer_ref/`, `text_encoder/`, `vae/`, `audio_vae/` in the diffusers layout — the same weights under diffusers tensor names, verified from its own index: 638 tensors against FL2VA's 535, the same total size to within 64 bytes (`context_embedder` against `condition_proj`, `audio_proj_in`/`audio_proj_out` against `audio_patch_proj`, and separate rather than fused q/k/v and gate/up) |

### 3.3 Why the port downloads `FL2VA/` only

The reference loader requires exactly the files inside the mode partition
(`h3.c:428-433`): `FL2VA/transformer/config.json`,
`FL2VA/tokenizer/tokenizer.json`, and the inventories of
`FL2VA/text_encoder`, `FL2VA/transformer`, `FL2VA/video_vae/source`,
`FL2VA/audio_vae`. Ref2VA is optional and guarded by the presence of
`Ref2VA/transformer/model.safetensors.index.json` (`h3.c:438-451`), with the
comment that prompt-only FL2VA stays usable while that checkpoint is not
installed. The root-level copies are the same weights under diffusers tensor
names, which is what makes the repository 464.2 GiB describe about 134 GiB of
actual model: three packings of the same tensors, plus a fourth for Ref2VA's
DiT. Nothing in the reference's required list reaches for them.

So the download is **144.05 GB, not 464.2 GiB**. It now lives on the artifact
disk as `/mnt/h3-artifacts/MiniMax-H3` (134 GiB, verified byte for byte with
`rsync -an -c`); the older `/mnt/h3` copy is retained untouched. With 795 GB
free there, the BF16 golden, every VQ rung (4.7–16.6 GB each), the renders and
scratch all fit.

`FL2VA/` also ships the model's own reference implementation — `video_vae/`
(`klvae.py`, `vae_cnn.py`, `vae_vit.py`, `minimax_h3_video_vae.py`),
`audio_vae/` (`dac_*.py`, `minimax_h3_audio_vae.py`), the configs, and
`model_index.json` with `_minimax_h3` (`schema_version` 1, `partition` fl2va,
tasks `t2va`/`fl2va`, `sigma_shift_scales` 12.0 video / 3.0 audio). That is the
port's specification for the VAEs: the source-of-truth rule puts the model's own
code above any summary of it.

### 3.4 What the model card adds, and what it changes

Read at revision `42ed227e` (`README.md` in the repository). Four statements
there carry more weight than the numbers above.

**The 39% AdaLN class does not have to be resident during inference.** The card:
"H3-Omni-Transformer is a 33B-parameter dense, single-stream Transformer, with
approximately 13B parameters residing in AdaLN-related branches. Because the
AdaLN modulation outputs can be precomputed and cached, these parameters do not
need to be loaded for inference-only deployment." The card's ~13B matches the
measured class table exactly — `adaln_proj.linear` is 13.006B across the 50
blocks, 39.3% of the DiT — and it splits the model into a **hot set** (the
attention and FFN matrices, ~20.1B, read at every denoise step) and a **cold
set** (the AdaLN branches, ~13.0B, read once per prompt to produce the
modulation cache). That changes both budgets: the per-step resident weights are
~40 GiB in BF16 rather than 61.73 GiB, and the quantization decision that
matters for speed applies to 20.1B of 33.14B, not to all of it. Section 4.5's
tiers and the recipe's rung sizes are quoted for the whole DiT and are
therefore conservative; a hot-only artifact would be ~4.1 GB at rung A rather
than 6.76 GB.

**The released checkpoints are CFG-distilled**, so the sampler does not carry a
classifier-free-guidance pair: one forward per step, not two. The plan's
per-forward cost table is already single-pass, and this is why.

**The biggest documented quality lever is not in the weights.** The card calls
H3-Context-IR "critical to the quality of the final output" and recommends
incorporating it or building an equivalent prompt pipeline; it is *not*
open-sourced. The port implements H3-Base, as the reference does, so the
quality gate in section 5 measures what quantizing the weights costs against
the same prompt on the same binary — it does not measure the port against the
official hosted output. Anyone comparing our renders to hailuoai.video is
comparing two different systems.

**Sparse attention is training-only in this release.** The card: "H3 natively
supports sparse-attention training and inference. The initial open-source
release provides inference with full attention only. Our sparse-attention
implementation will be released in a future update." So the O(n²) attention
cost in section 4.2 stands for now, and a future release could change the
reach calculus rather than the weight story.

Three more card facts worth recording: the checkpoints are BF16 (which is what
the shard headers show), the repository intentionally hosts "the original
checkpoint (`FL2VA/`, `Ref2VA/`) and the diffusers format side by side, so
scope the download to what your framework needs" — which is the same scoping
this plan chose, by the loader's own required-file list — and the license is
`minimax-h3-community-license-agreement`, not a permissive one. A published VQ
artifact is a derivative of those weights, so the artifact's notice and
naming obligations are a compliance item to settle before any release, the way
the MiniMax-M3.1 VQ release carries its own license file.

Two alternatives the card names that this plan does not use: SGLang and vLLM
both publish H3 serving recipes. They serve the BF16 checkpoint, so they give a
working baseline but not the quantized tier this project exists for; they are
worth one afternoon as a sanity reference for render parity, and worth knowing
about before anyone assumes the port is the only way to run H3 at all.


### 3.5 DiT geometry and where the bits are

Shapes and dtypes below are read from the shard-1 safetensors header over range
requests, not inferred. Detail and the full class table are in the
[VQ recipe](vq-recipe.md) sections 1 and 4; the load-bearing facts:

- the DiT is **33.14B params in BF16**, 535 tensors, 66.28 GB on disk;
- `blocks.N.adaln_proj.linear.weight` is `[96768, 2688]`, 260.1M params per
  block, **13.0B across the 50 blocks — 39% of the model**, and the single
  largest quantization decision;
- every quantizable matrix has a column count that is a multiple of 64
  (`5376`, `7168`, `14336`, `21504`, `2688`), which is what the container needs;
- `audio_patch_proj` is `[5376, 32]` and stays native (32 does not divide by
  64), as do all norms, biases, `rope.inv_freq` and `time_embedder`.

## 4. The speed case, before any code

The plan separates what is certain from what must be measured, because the two
have different failure modes.

### 4.1 Certain: the weight read

66.28 GB of BF16 read per invocation at the measured 686 MB/s effective
(96.563 s) becomes:

| Tier | DiT artifact | Weight bytes vs BF16 | Projected load at 686 MB/s |
|---|---:|---:|---:|
| BF16 (today) | 66.28 GB | 1.0× | 96.6 s (measured) |
| **13-bit GVQ1 v1 (the chosen rung)** | **~6.6 GB** | **10.0× fewer** | **~10 s** |
| + GVR8 residual on the measured blocks | 7.6–9.2 GB | 7.2–8.7× fewer | ~11–13 s |
| 9/10-bit GVQ v2 (not needed; a size experiment only) | 4.7–5.2 GB | 12.8–14.1× fewer | ~7–8 s |

The rung is the one the operator's own MiniMax release ships at: GVQ1 v1, 8-dim
vectors, K = 8192, 13-bit (12-bit main stream + 1-bit plane), one E4M3 codebook
per matrix, one F16 gain per row, `imatrix`-weighted Lloyd with least-squares
gain refit — see the [recipe](vq-recipe.md) sections 2 and 3.

Projections assume the read stays I/O-bound at the same rate, which is the
optimistic case and is marked as an estimate: the measurement is P2, and the
real gain is the *four to fourteen times fewer bytes*, which no cache policy
can take away. Every rung also fits the page cache many times over, so a warm
re-run is not I/O-bound at all.

### 4.2 Not certain: the denoise

One 1024×576 / 107-frame DiT forward is 18,432 tokens (`latent 32×36×64`,
patch [1,2,2]). Linear work 2 × 33.14e9 × 18,432 = 1.22 PFLOP, attention
4 × 18432² × 7168 × 50 = 0.49 PFLOP, total ≈ 1.7 PFLOP. The recorded 815 s for
50 steps is 16.3 s per forward ≈ **104 TFLOP/s against the 108 TFLOP/s
BF16/F16 `mma.sync` peak of this part** (`AGENTS.md`, "Peaks") — 96% of peak.

The consequence is uncomfortable and belongs in the plan rather than in a
post-mortem: at that shape a BF16-weight VQ artifact **cannot** speed the
denoise up; the FLOPs are already saturated. Only a rung whose operands are
E4M3 (the FP8 rate is twice BF16) can, and that changes activation precision as
well as weights. At small shapes there is separate headroom that has nothing to
do with VQ: the 256×256 two-step render measures 2.715 s for 2 forwards
(1408 tokens each) ≈ 44% of peak, i.e. launch and occupancy bound, which is
where CUDA graphs, fused kernels and residency work.

### 4.3 Certain: memory, and what it removes

The BF16 DiT peaks at 27.06 GB resident; the compact rungs are 4.7–16.6 GB. The
reference's exact low-memory mode (`--ssd-streaming`, 1.63 GB peak, identical
SHA-256) costs 37.6% more load-plus-denoise on GB10 and is explicitly "not the
GB10 speed default". A VQ artifact removes the reason that mode exists, and its
own exactness argument is moot once weights are quantized by choice.

### 4.4 Memory: what fits, and what memory actually buys

The requirement is that this runs on the Spark and holds as long a video as
possible. The arithmetic below decides it, and it corrects one assumption worth
stating before it costs a project.

Component bytes, from the artifact listing (certain: these are file sizes):

| Component | GB | GiB |
|---|---:|---:|
| DiT, BF16, 33.14B params | 66.28 | 61.73 |
| text encoder, Qwen3-VL, 1058 tensors | 66.71 | 62.13 |
| video VAE | 10.42 | 9.70 |
| audio VAE | 0.605 | 0.56 |
| **sum if all resident at once** | **134.0** | **124.1** |

The Spark measured 121 GiB total with 119 GiB available (`free -g`), and this
tree's memory guard reserves 12 GiB by default (2 GiB in its dedicated-resident
profile, `host-memory-guard.md`), so the usable envelope is **107–117 GiB**.
124.1 GiB of weights does not fit in it — which is exactly why the reference
runs its phases apart: "prompt encoding and the two VAEs run in separate phases
rather than adding their full peaks" (`h3.c` README, section 2). Peaks per
phase, BF16, with activation estimates that the P2 trace replaces with
measurements:

| Phase | Resident | Activations | Peak |
|---|---|---:|---:|
| text encode | text encoder 62.13 GiB | ~1.5 GiB | **63.6 GiB** |
| denoise | DiT 61.73 GiB | ~2–4 GiB at 18,432 tokens (residual 198 MB per buffer, qkv 792 MB, MLP intermediate 1.06 GB) | **64–66 GiB** |
| VAE decode | video + audio VAE 10.26 GiB | measured 10.26 GB peak total | **~10 GiB** |

So **BF16 already fits**: the phase peak is 64–66 GiB against a 107 GiB
envelope, 41–51 GiB spare. VQ is therefore not what makes this fit, and the
plan should not pretend otherwise. What VQ buys is a different operating point:

| Tier | DiT | Whole stack resident at once | Consequence |
|---|---:|---:|---|
| BF16 | 61.73 GiB | 124.1 GiB — does not fit | phases must be torn down and reloaded; each render pays the 96.6 s load |
| **13-bit DiT, text encoder and VAEs exact** | **6.75 GiB** | **79.14 GiB — fits in 107 with 28 GiB spare** | a warm daemon: DiT, text encoder and both VAEs stay loaded; per-render cost is the denoise alone |
| 13-bit + GVR8 on the measured blocks | 7.7–9.4 GiB | 80.1–81.8 GiB | same, with the residual where a sweep says it pays |
| 13-bit DiT and 13-bit text encoder | 6.75 + 6.8 GiB | ~24 GiB | everything resident several times over; only if a measurement says the encoder tolerates it |

The middle row is the design point: **one lossy element (the DiT weights) and
everything else bit-exact**, in a process that never reloads.

And the second half of the requirement — as much video as possible — is **not**
a memory question at this size. Activations scale with tokens, not with the
square of them: the largest legal canvas (product ≤ 768×1344, so 1344×768) has
32,256 DiT tokens and still needs only ~4–7 GiB of activations. What stops a
longer or larger render is time and the O(n²) attention: 1024×576 / 107 frames
is 1.7 PFLOP per forward and ~16.3 s at 96% of peak, while 1344×768 / 107
frames is 3.63 PFLOP (linear 2.14 + attention 1.49) and ~34 s per forward at
peak, i.e. ~28 min for a 50-step render before the VAE. Memory with the tiling
knobs left free is what makes those *decodes* possible without re-tiling the
VAE; `vae_tile_size` 256 with `vae_tile_overlap_min` 64 is the artifact's own
setting, and a tile that does not fit forces more, smaller tiles.

### 4.5 The allocation, once SSD streaming is on the table

A requirement of the form "the highest quality that still fits, with SSD
streaming like the other releases" makes the choice sharper, because it makes
the text encoder and the VAEs free: they run once per render and can be read
from SSD, leaving the resident budget to whatever runs every step.

| Tier | Resident | Streamed from SSD | Lossy element | Spare |
|---|---|---|---:|---|
| **Q0 — all exact** | DiT BF16 61.73 GiB | text encoder 62.13, VAEs 10.26 | **none** | ~41 GiB |
| Q1 | DiT BF16 61.73 + TE 13-bit 6.8 + VAEs 10.26 = 78.8 GiB | — | the prompt encoder, once per render | ~28 GiB |
| Q2 | DiT 13-bit 6.29 + TE BF16 62.13 + VAEs 10.26 = 78.7 GiB | — | the denoiser, fifty times per render | ~28 GiB |
| Q3 | DiT 13-bit 6.29 + TE 13-bit 6.8 + VAEs 10.26 = 23.4 GiB | — | both | ~84 GiB |

Q0 is the answer to "the highest quality possible": **no quantization at all**,
which is only reachable because the DiT alone (61.73 GiB) fits inside the
envelope once the other 72.4 GiB is treated as streamed instead of resident.
Its cost is a text-encoder read per render — ~90 s on the internal NVMe or
~150 s on the USB disk at the rates this plan measures — on top of a DiT that
stays warm. It needs no converter, no VQ artifact and no quality gate; it is
the reference's own phase behaviour, implemented in the port.

Q1 is the tier to buy speed with, if that read hurts: the *denoiser* stays
bit-exact and only the prompt encoder is quantized, once per render, at the
13-bit rung whose error is bounded and measured. Q2 spends the same bytes on
the other side — it makes every render read nothing from SSD, and puts the
quantization inside the 50-step iteration instead. Both fit; the render gate
(section 5) and the P2 attribution decide between them, and the mechanism
argument favours Q1: a changed conditioning vector perturbs the sampler once,
while changed denoiser weights perturb it at every step.

The practical consequence for the port: implement Q0 first, because it is the
reference's own phase structure plus residency and it carries no quality risk
at all; keep the phases as separate *jobs* rather than separate *process
lifetimes*; spend Q1/Q2's surplus on the VAE's tiling and on a
persistent Daemon so that repeated prompts never pay the load again. That, not
a bigger latent budget, is what "as much as possible" buys here.

## 5. The quality question

"Full quality" is the operator's requirement and it needs a definition before
the first encode, because the answer decides whether this project is worth
starting at all.

| Tier | What it is | Exactness | Speed it can buy |
|---|---|---|---|
| T0 | BF16 weights, no quantization | bit-identical output to the reference (the reference's own repeat renders and the SSD comparison are the evidence: identical SHA-256, SSIM 1.0) | load-side only: residency, prefetch, page cache warm, fewer copies — no quality risk |
| T1 | BF16 weights, E4M3 codewords refused, FP8 activations refused | as T0 | ~0 on the denoise at 1024×576; the measured 44%-of-peak headroom at preview shapes |
| T2 | VQ weights, per-matrix books, calibrated | **bounded, measured, and gated** — not identical | the load (4–14× fewer bytes) certainly; the denoise only on an E4M3 rung |

What the tree's own record says about T2, quoted rather than paraphrased:
VQ releases report deltas — per-layer routed-output relative error 0.11–0.52
depending on weighting mode (`vq-calibration.md`, "Measurements (A.X-K2)"), KL
in nats per token around 0.0035–0.0038 for the GLM-5.3 VQ decode paths
(`glm53-vq.md`), paired NLL moves of −0.0007 to −0.012 nat/token. The
`bit-identical` rows in those documents are kernel-scheduling changes holding
*identical weights*, which is a different claim from quantizing weights, and no
document in this tree shows a quantized artifact reproducing the unquantized
model bit for bit.

The one dense bake in this tree gives the curve to start from
(`/data/embeddinggemma2/RECIPE.md` section 5, the same encoder, per class):
dim 4 at K 4096 (3.0 bits per weight) reconstructs at cosine 0.9841–0.9993 with
a worst tensor of 0.9740, and on a labelled retrieval task keeps 98.32% of
MRR@10, 97.00% of recall@1 and 100.68% of recall@10 with mean vector cosine
0.956; dim 8 at the same 12-bit index width (1.5 bits per weight) sits at
0.9144–0.9495, six to twelve points lower on every class. That recipe's own
verdict was that dim 8 "buys size, not quality", and the tier actually shipped
is the 3.0-bit one. So the expectation for H3 is stated in advance: hold the
floor at R1, expect to lose it at R2 and below, and let the render gate decide
rather than the cosine.

The operator's own MiniMax-family release is the closer precedent, and it is
frank about its own result: `Baekpica/MiniMax-M3.1-Flash-VQ-GGUF` (`74e5a9fd`)
ships 428B parameters at 1.98 bits per routed-expert weight and measures, on
held-out teacher-forced positions, KL 0.150 and the same next token 88.2% of
the time on assistant turns — against a *noise floor* of KL 0.003 established
by running the original twice at different GEMM batch sizes. Its card calls
itself "experimental", warns of loops, language switching and broken
characters, and states the requirement in the same breath as the cost. What it
also shows is the shape of a defensible gate: floor first, then a rate
(a percentage), then per-category breakdowns rather than a single verdict.

### 5.1 The recovery lever: multiplicative block-output gains

This tree's VQ line has one mechanism that repairs quantization error after the
fact, and it is the only correction family the reference method found to work
at the output. In `YoungAi` it is 反修 ("repairing backwards"), shipped as the
per-layer multiplicative chain that the engine loads with `--zchain`
(`/data/YoungAi/ds4_zchain.h`): for every expert, one gain per output channel of
its down projection plus one router bias, re-solved so that each quantized MoE
block reproduces the *original* block's output on the fitting corpus. The whole
chain collapses to one per-token scalar λ(x) on the routed contribution,
`F = shared_out + λ(x) · Σ ge·y_e`, with `GE` folded into the router gate
weight; five op types (GL, GLdyn2, GLdyn8, TREF, GE) and a small DQZ2 file.
Absent file, today's path exactly.

Measured, in the method's own tables (`docs/TECHNICAL.md` §4.2 and §5.1, on
DeepSeek V4.1 Flash at 12–13-bit experts): Same top-1 rises 71.80 → 74.38%
(finance), 78.29 → 81.96% (code), 73.20 → 76.45% (law), 69.37 → 73.12%
(medicine), 71.48 → 74.85% (science); mean KL falls 17–30%; general English is
if anything improved (76.76 → 77.9–79.7%). The sidecar is ~40 MB and costs
about an hour of solve per domain on the Spark. The router bias earns +0.24 pp
on top of the gains.

Why this matters for H3, and why it cannot be used as-is:

- **It is a recovery, not a compression.** It buys back quality that the
  quantization spent, at 40 MB and an hour. For a project whose requirement is
  "hold the quality", that is the cheapest quality per byte available anywhere
  in this toolchain.
- **The shipped implementation is MoE-only.** Its two hooks are a per-expert
  gain folded into the router gate and a per-token scalar on the routed sum;
  H3's DiT has no router and no experts. The dense analogue is a per-block
  multiplicative gain on the block's output channels (`mlp.fc2` and
  `attn.out_proj`, 5376 channels each) — the same family, and the family the
  method found empirically effective. It is an experiment, and the plan treats
  it as one (P7), because a dense block has no routing to exploit.
- **The rest of the family is already known to fail**, which saves the
  experiment budget: additive low-rank `y += B·A·x` was rejected (≈0
  first-order effect at the output), mean/bias correction was rejected
  (inside the layer +4 pp, at the output −0.33 pp), and re-solving codebooks on
  target data gave no gain. Only multiplicative gains survived.
- **It needs a teacher fixture.** The solve compares against the *original*
  model's block outputs at the fitting tokens (for DeepSeek: 40 layers × 8192
  tokens = 14 GB, dumped once by the engine's own prefill kernels, with a
  bit-for-bit check that the fixture reproduces the engine's block output on
  ≥ 99.996% of values). For H3 the reference on the Spark can dump exactly
  that: 52 blocks × 2048 tokens × 5376 channels × 2 B ≈ 1.15 GB, and the
  recipe's own rule applies — the gains are solved layer by layer against
  already-mounted upstream layers, on the deployment path, and a block is
  mounted only when its held-out gain clears a significance threshold.

If the 13-bit rung fails the render gate, this is the first thing to try, not a
lower bit rate. It is ordered before any further precision loss for the reason
the method's own tables give: it recovers more quality per byte than bits do.

So the plan proposes the gate in the [VQ recipe](vq-recipe.md)
section 8: a frozen 8-prompt × 2-shape set rendered by the reference binary,
full-video SSIM/PSNR/LPIPS per prompt, and a floor the operator sets. Proposed
default: SSIM ≥ 0.99 full-video, stricter than the two kernel-change precedents
this tree already accepted (0.999303 for the tiled VAE kernel, 0.984985 for the
VAE reduction order). Rung accepted only if every prompt passes; per-class
metrics so a failing class can be split out; the exact T0 path always remains
available for renders that must be bit-identical.

## 6. Architecture of the port

### 6.1 Where it lives

The same shape the repository used for the (unmerged, abandoned) Qwen-Image
image port, which remains the only precedent for a diffusion family here:

| Layer | Path | Content |
|---|---|---|
| Rust core | `crates/ds4-core/src/minimax_h3.rs` + `minimax_h3/` | `checkpoint.rs` (safetensors reader, manifest, layout), `dit.rs`, `text_encoder.rs`, `video_vae.rs`, `audio_vae.rs`, `schedule.rs`, `vq.rs` (container, byte accounts, dequant oracle), `oracle.rs` |
| Rust CLI | `crates/ds4-server/src/h3_cli.rs` (or `ds4-cli`) | the generation CLI and the interactive session |
| Native | `cuda/minimax_h3_*.cuh` | primitives, attention, VQ decode, VAE, audio |
| Converters | `tools/` (C, linking the reference encoder) | plan / encode / merge / check / verify / wrap |

`ModelFamily` (`crates/ds4-core/src/shape.rs`) is the LLM serving contract and
is deliberately **not** extended: it enumerates text-generation families with
KV/bank/MTP state, and a diffusion transformer with no KV cache has none of
that. Whether a sibling enum or a separate registry is the right shape is a P0
decision made against `AGENTS.md`'s "New model-family serving contract", not by
analogy.

The family is additive by project rule (`AGENTS.md`, "MiniMax-H3 (h3) is
additive — the other families are untouched"): an h3 unit's diff names only h3
files plus additive registration lines, and no other family's behavior, types,
defaults or code path changes on this branch.

### 6.2 The host boundary

The port keeps the reference's phase structure — prompt → text/vision encoding →
latent initialisation → denoise loop → video VAE decode → audio VAEs → mux —
because that is what its measurements and quality gates are organised around.
The VQ decode lives inside the linear-algebra primitives the DiT and VAE call,
so a rung changes weight bytes and one kernel, not the model.

### 6.3 The CLI surface

Target parity with the reference's documented controls, taken from the README
and to be taken exhaustively from `h3_cli.c` at P8: `--info`, `-d` model dir,
`-p` prompt, `--width/--height/--frames/--steps/--layers/--reuse/--core-reuse`,
`--token-reduction`, `--render-width/--render-height`, `--ssd-streaming`,
`--use-int8-row-fc2` (CUDA no-op in the reference; drop it and say so),
`--show`, `--zoom`, `--profile`, `--seed`, `-o`, first/last-frame anchors, and
the interactive `!` commands. Parity means the port's numbers can be compared
with the reference's on the same command line.

## 7. Build order and gates

One unit per branch, no unit without its gate written before it starts. The
Qwen-Image precedent is the cautionary tale: that port was closed once for lack
of a performance motive, reopened for ownership, and is now abandoned by the
operator — so every unit here names the evidence it must produce and the metric
it must beat, and section 4.2 states in advance which of them can fail.

**P0 — Environment and the oracle.** On the Spark: install `libicu-dev` (needs
`sudo`), clone `JLFN/h3.c` at `eae5d54`, `make -j20` with `NVCC_ARCH=native`,
`./h3 --info -d <checkpoint>`, run the 2-step GB10 profile command and the
tiled-VAE quality bench. *Gate:* the binary links and runs, `--info` resolves
every required component, and the P2 profile is reproducible. *Evidence:* the
commands, versions, wall times, `nvidia-smi` under load, PSI samples.

P0 is closed except the tiled-VAE quality bench. The source was transferred as
`git archive eae5d54` into `~/h3-oracle` (the clone route was not used) and
built with `make -j20 NVCC_ARCH=native`: zero errors, `h3` and `libh3.a`
linked against `-licui18n -licuuc` 74.2. `./h3 --info -d
/mnt/h3-artifacts/MiniMax-H3` resolves all four components: Qwen3-VL encoder 14
files / 1,058 tensors / 62.133 GiB, FL2VA DiT 13 / 535 / 61.728 GiB, video VAE
1 / 560 / 9.700 GiB, audio VAE 1 / 1,087 / 0.564 GiB, Ref2VA absent. The
reference's own 2-step profile command (`docs/GB10_PROFILE.md:7-10`) ran twice
with a warm page cache: DiT load 17.282 s, Euler denoise 0.730 s, DiT peak
27.063 GB, audio VAE 0.565 s, video VAE 2.587 s, and both outputs hash
`7ba6d49bd58ed67f194901c05ce59c55d0c3ebeda8094efb4b6575d59c37a39f`. That hash
is stable locally but differs from the reference document's
`dbae1b441cface55bfe86aaabe78d44c0c05746909f7874908dd2cb298d8c5c8`, which was
recorded on driver 595.84 against this host's 580.178.04; the delta is
unexplained and is P2's first question. Load time is 5.6x the documented
96.563 s because 116 GiB of the checkpoint were already in page cache.

**P1 — The artifact on the Spark.** Download the `FL2VA/` subset plus
`scheduler/`, `audio_scheduler/`, `model_index.json` at the pinned revision into
`/mnt/models/MiniMax-H3`, with a manifest generated from the HF API's own file
list (name, size, LFS sha256) and verified file by file. *Gate:* every file's
size and sha256 match the manifest; the loader's required-file list
(`h3.c:428-433`) is complete; `--info` passes on the new copy. *Rollback:* delete
the directory; nothing else depends on it.

**P2 — Attribution.** `nsys`/`ncu` over the reference on the Spark, with NVTX
ranges if available or launch-name attribution otherwise: per-phase time, bytes
read, page-cache state, PSI, and the per-forward FLOP efficiency at both the
preview and the quality shape. *Gate:* a table that reproduces or corrects
section 4's arithmetic — in particular the load rate and the 96%-of-peak
finding. *This unit decides how much of the VQ campaign is worth funding.*

**P3 — The port's own reader.** A Rust safetensors reader with a manifest and
inventory (`ds4-core` precedent: an `inventory` example that dumps
name/type/rank/dims/bytes/offset), reconciled against the shard headers and
against the reference's expectations. *Gate:* a two-sided diff with zero
unexplained differences; runs on this host, no GPU.

P3 shipped in two parts: the reader (`minimax_h3::safetensors`) and the DiT
layout (`minimax_h3::layout`), which derives all 535 expected names from
`config.json` and checks each dtype and shape against the reference's loaders
(`h3_dit.c:501-525`, `h3_dit_schedule.c:267-307`). Live gate over the gvfs
route: `minimax_h3_inspect --layout <FL2VA/transformer>` exits 0 with
`tensors=535 blocks=50 refiner=2 hidden=5376 inner=7168 video_patch=96` in
3.1 s, header-only. Negative cases (absent, unlisted, duplicate, wrong dtype,
wrong shape, wrong timestep width) are unit tests, as is the config itself.

**P3.1 — The mapped role table.** The operator deferred the VQ campaign
(2026-10-24) and asked for the exact/BF16 family first, so the next unit is the
loader the graph will call: `minimax_h3::weights` resolves every `Role` to one
shard and one byte span at open (`h3_weights.c:129-142`: sorted path order, the
first shard holding the name wins), maps a shard on first use instead of
copying its payload (`h3_gpu.m:608-660`), and gives the device route a `Span`
(path, absolute offset, bytes) or host code a borrowed `&[u8]`. *Gate:*
`minimax_h3_inspect --weights <FL2VA/transformer>` walks all 535 roles, its
sampled bytes equal the same offsets read with `pread`, and the process maximum
RSS stays under 1 GiB with all 13 shards mapped — that last number is what
proves no payload is copied. *Rollback:* additive module, nothing imports it
yet.

P3.1 shipped as `minimax_h3::weights`. Live gate over the gvfs route, binary
run directly (not through `cargo`, whose own footprint is 519 MB): 535 roles,
13 shards mapped, 66,280,430,144 bytes = 61.728 GiB, sample
`eb95a483889d3913`, 0.52 s wall, 22.6 MB maximum RSS, 329 major faults. Each
tensor's first and last 8 bytes from the mapping equal the same offsets read
with `pread`; a mismatch names the role and exits non-zero. The device route is
`Weights::span` (path, absolute offset, bytes), the host route `Weights::bytes`.

**P3.2 — The prompt tokenizer.** The byte-level BPE the prompt path needs:
`minimax_h3::tokenizer` loads `tokenizer.json` under the reference's own spec
checks (`h3_tokenizer.c:367-477`), reproduces the byte codec (`:560-573`), the
Qwen2/GPT-2 split (`:705-765`), the merge loop (`:766-818`), added tokens and
the empty-prompt pad (`:829-877`), and decode with its single-`U+FFFD` fallback
(`:892-938`). *Gate:* the five vectors of `tests/test_tokenizer.c:38-57` and the
portable fixture's cases, then a line-by-line diff against `tok_probe.c` over
the released vocabulary. *Rollback:* additive module, nothing imports it yet.

P3.2 shipped. On the Spark the reference's own test reports `ok: 60 tokenizer
checks against released Qwen vocabulary`; the port replays the same five vectors
plus both empty-prompt cases (`TOKENIZER cases=7`) in 0.39 s over the gvfs
route. A 31-line corpus (CJK, Greek, Cyrillic, Arabic, Hebrew, Korean, emoji,
astral letters, combining marks, contractions, NBSP, NEL, a roman numeral, a
superscript, ideographic zero, tabs, added tokens, empty and blank lines)
encodes identically on both sides: 31/31 lines, `diff` clean. Those code points
are where the `regex` classes could diverge from ICU; the corpus samples them,
it does not exhaust the Unicode database.

With VQ deferred, the exact track continues at the text encoder, then the DiT
graph, then P8's VAEs and CLI; P4-P7 stay parked.

**P3.3 — The text-encoder path.** The 551 tensors the prompt encoder reads,
named and shaped before its forward is written: `minimax_h3::text` takes the
geometry from `text_config` and refuses a config that disagrees with the
reference's constants (`h3_text_encoder.c:12-27`), then checks the embedding
(`:622-625`) and fifty layers of eleven tensors (`:147-180`) for BF16 dtype and
exact shape (`:219-247`). Tensors the path never touches are counted, not
rejected: the 14 declared-but-unexecuted layers (`:14`), the final norm, the
LM head, and the vision tower. *Gate:* the live check names every missing or
mismatched tensor; the outside counts reconcile with the shard inventory.
*Rollback:* additive module, nothing imports it yet.

P3.3 shipped. Live over the gvfs route, header-only in 0.09 s: `used=551
outside=507` with `deferred_layers=154 final_norm=1 lm_head=1 vision=351
other=0`. Together with the inventory's 1,058 tensors the tally is exact
(551 + 154 + 1 + 1 + 351). The eleven per-layer shapes and the embedding's
`[151936, 5120]` all match, so every tensor the reference's loader would read
is present, BF16, and correctly shaped.

**P4 — Converter, rungs R1 and R2.** The C driver, per-shard containers, nocal
Lloyd, f16 books, gains, and the five converter gates of the
[VQ recipe](vq-recipe.md) section 6. *Gate:* determinism by re-encode
and `cmp`, class totals reconciled, round-trip against the CPU dequant oracle
within a stated bound per class.

**P5 — The VQ decode kernel (f16 books).** Per-matrix book in shared memory,
BF16 fragment path, at 1408 and 18,432 tokens. *Gate:* bit-exact against the CPU
oracle on real rows, then measured GEMV/GEMM throughput against the BF16
baseline, and — per `AGENTS.md` — SASS read before any A/B, and no change to
the reference's numerical order.

**P6 — The render gate, rungs R1 and R2.** The frozen set, the metrics, the
floor. *Gate:* the accept/reject decision per rung, recorded with per-prompt
numbers and per-class weight metrics, plus the operator's verdict.

**P7 — Recovery, then the aggressive rungs.** In this order, because it is the
order the method's own measurements justify. First the dense multiplicative-gain
recovery (section 5.1): dump the BF16 teacher fixture from the reference for 52
blocks on the frozen prompt set, solve per-block output-channel gains against
already-mounted upstream blocks, mount a block only when its held-out render
metric clears the significance threshold, and re-run the P6 gate. Only if the
13-bit rung plus recovery still misses the floor: the GVR8 residual on the
worst blocks, then — as a last resort, and only if a render says the shape is
FLOP-saturated — the E4M3/9-bit rungs that could actually beat cuBLAS BF16 on
the denoise. *Gate:* the render metric before and after each step, per block,
with the recovery's cost in bytes and hours recorded.

**P8 — The rest of the pipeline.** Text encoder and VAEs (native or VQ, decided
by P2's numbers), the schedule, the audio path, the mux, and the CLI, with a
per-step latent-diff harness against dumps from the reference. *Gate:* the port
reproduces the reference's render at the same seed within the P6 floor, with
diffs attributable per phase.

**P9 — Performance pass.** Only after P8: CUDA graphs, residency and prefetch
policy, the 44%-of-peak preview shape, the VAE's F32 attention. *Gate:* each
change measured against P2's attribution with a stated noise reference, and bit
identity kept whenever only scheduling changes.

**P10 — Close.** Evidence file, the rule 19 QA gate, docs, an install script in
the shape of the operator's other Spark deployments (`dgx-recipe/scripts/`,
`install-*-vq.sh`), and the push offer. The QA gate runs once, here, when the
port is complete and runnable — not per unit (operator, 2026-10-24).

## 8. Deployment and the two routes

| Thing | Where |
|---|---|
| Reference engine | `~/h3-oracle` on the Spark, unpacked from `git archive eae5d54`, never modified; the source itself stays `/data/h3.c` on the dev box |
| Source checkpoint | `/mnt/h3-artifacts/MiniMax-H3/` — the model root for this work: ext4 label `H3ART`, 938 GB with 795 GB free, copied from `/mnt/h3` and verified byte for byte (`rsync -an -c`); the older copy stays untouched |
| Converted artifacts | `/mnt/models/h3-vq/<rung>/` with `release.json`, per-shard containers and the GGUF |
| Golden renders | `~/h3-golden/<prompt-id>/` with the render, its sha256 and its `--profile` output |
| Commands | the operator's ssh tmux pane, `tmux send-keys` / `capture-pane` |
| Files | the gvfs sftp mount, `cp` in and out |
| This repository on the Spark | a *separate* working directory from the serving deployment at `~/ds4-dfm-rs` (which is on `main` at `32a41683` and serves live models); the port never edits it in place |

No third route: no direct `ssh` from this shell, no `sshpass`, no mount of our
own. If a route is missing, ask.

## 9. Risk analysis

Decision: commit the port to `feature/minimax-h3` and run the campaign above on
the Spark, with a VQ tier gated on a render floor.

External risks, each checked:

| Risk | Evidence (checked this session) | Effect |
|---|---|---|
| Driver mismatch | reference validated on 595.84, host has 580.178.04 | reference numbers may not reproduce; P0/P2 measure instead of assuming |
| `libicu-dev` absent | `/usr/include/unicode/utypes.h` missing, only `libicu74` runtime | the reference cannot link until the operator installs it |
| Artifact download | 144.05 GB subset, 393 G free on `/mnt/models`; host has pulled 106 G models before | hours of transfer; a partial failure is resumable by `hf download` |
| USB model disk throughput | `/mnt/models` is a USB SSD (ext4) | load-rate measurements on it do not represent the internal NVMe; P2 pins where the checkpoint lives |
| No PyTorch on the Spark | `import torch` fails | no Python oracle; the C engine is the oracle, which is the correct source of truth anyway |
| Upstream drift | `upstream` = Baekpica; this fork's rule forbids upstream pushes/PRs (`feat/image` commit `1b15ba24`) | all work stays on the fork, `dev` → `main` by PR |

Decision risks, each with the check behind it:

| Risk | Evidence | Mitigation |
|---|---|---|
| VQ cannot speed the denoise at the quality shape | 18,432 tokens → 1.7 PFLOP vs 815 s/50 steps ≈ 104 of 108 TFLOP/s; and E4M3 dequant measured at 82–84 of 108 in this tree | the plan's headline benefit is the load; the denoise claim is quarantined in P7 and may be reported as "no" |
| The quality floor is missed | VQ releases here report bounded deltas, never identity; the floor is unset | P6 measures per class before rungs are stacked; T0 remains for bit-identical needs |
| The 39% adaln class is the sensitive one | `[96768, 2688]` × 50 = 13.0B params; it drives every modulation | per-class metrics; adaln can be split out and kept native if needed |
| Scope creep like the Qwen-Image port | that port was closed for motive, reopened, now abandoned by the operator; the branch is unmerged (`feat/image`) | every unit has a pre-registered gate; section 9's stop-loss; no unit starts without its metric |
| Two agents, one repository | a live session (pid `1331103`) is on `feature/minimax-h3` in this working tree; the branch already existed at this plan's start | one unit per branch, explicit hand-off of ownership, and the plan is the shared contract |
| Disk pressure | after 144 GB the model disk has ~249 G; BF16 golden 66 G, rungs 4.7–16.6 G, DQVD intermediates ~12–17 G | delete intermediates after wrap; keep the source until P10 |

Measurement and its limit: the arithmetic in section 4 uses the reference's own
timings and the part's documented peak; it is arithmetic over a third party's
measurements, not a Spark measurement. It bounds the *possibilities* and decides
which claims are worth funding; it does not prove any speedup on this host.

Rollback, per step: the reference tree and the checkpoint are never mutated
(re-clone / re-download is the recovery); converter outputs are files under
`/mnt/models/h3-vq/` (delete the rung directory); the port's commits are on
`feature/minimax-h3` (no push without an explicit offer and the rule 10 scan,
so rollback is `git branch -D` before any push, or a revert commit after).
Nothing in this plan touches the serving deployment on the Spark.

Residual risk, and the one signal that reverses the decision: if P2 shows the
load is *not* I/O-bound on this host (for example if the weights come from an
NVMe RAID at several GB/s and the 96.5 s figure was conversion-and-upload
bound), the VQ load win shrinks to the FLOP/cache story and the campaign should
be re-scoped before P4. That single measurement is the plan's own kill switch.

## 10. Decisions needed from the operator

1. **The quality floor.** Proposed SSIM ≥ 0.99 full-video against the BF16
   reference, per prompt, on the frozen set. This number defines the whole VQ
   campaign; without it the ladder has no accept criterion.
2. **Scope.** `FL2VA/` only (144 GB, prompt + first/last frame), or the full
   464 GiB including Ref2VA? Ref2VA is excluded in section 1 and can be a
   follow-up.
3. **`sudo` for `libicu-dev`** on the Spark, and permission to clone and build
   the reference there.
4. **Which quality tier**: Q0 (all exact, text encoder streamed per render),
   Q1 (prompt encoder at 13-bit, denoiser exact), Q2 (denoiser at 13-bit,
   prompt encoder exact). All four fit; Q0 carries no quality risk and no
   converter work.
5. **Ownership.** This session wrote the plan; a second live session is already
   on `feature/minimax-h3` in the same working tree. One of them should own
   execution, and the other should hand off rather than work in parallel.
6. **First rung to build.** Proposed: GVQ1 v1, 13-bit (12-bit stream + 1-bit
   plane), K = 8192, dim 8, E4M3 book, F16 gains, `imatrix` weighting with
   least-squares gain refit — the configuration the operator's own
   MiniMax-M3.1 release ships and measured. Not the 9/10-bit rungs, which H3
   does not need for fit.
7. **The recovery lever.** Fund the dense multiplicative-gain experiment
   (section 5.1) before any lower bit rate? Proposed yes: it is the only
   correction family the reference method found effective, it costs ~40 MB and
   hours rather than bytes, and it is ordered before further precision loss.
8. **Operating mode.** Phase-separated (BF16, reload each render) or fully
   resident (13-bit DiT + exact text encoder and VAEs, ~79 GiB, a warm daemon
   that never reloads)? Section 4.4 says the second is what the memory is for;
   it costs the VQ quality gate, and it is the only mode in which repeat
   renders are fast.
