# MiMo-V2.6-Flash VQ: format, execution and serving

MiMo-V2.6-Flash VQ (family `Mimo2`, variant `Mimo26FlashVq`, architecture
`mimo2_vq`) is a GVQ1 conversion of
[XiaomiMiMo/MiMo-V2.6-Flash-RL](https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Flash-RL/tree/f657ba2f7d24c9adc5730675baaa2d34de878a49)
and its UNCENSORED tune
([dealignai/MiMo-V2.6-Flash-RL-UNCENSORED](https://huggingface.co/dealignai/MiMo-V2.6-Flash-RL-UNCENSORED/tree/0bdb11d7f37d3f6939e53f822c2bff518e4852fa)),
recipe `mimo26f-vq8-r13x47-r8x32-q8-fp8kv-embssd-v1`:

| Release | Commit |
|---|---|
| [Baekpica/MiMo-V2.6-Flash-RL-VQ-GGUF](https://huggingface.co/Baekpica/MiMo-V2.6-Flash-RL-VQ-GGUF) | `8a15b057` |
| [Baekpica/MiMo-V2.6-Flash-RL-UNCENSORED-VQ-GGUF](https://huggingface.co/Baekpica/MiMo-V2.6-Flash-RL-UNCENSORED-VQ-GGUF) | `f1cd8523` |

Both share one architecture; UNCENSORED differs only in layers 34–47's
`o_proj`, and 38 of its 53 parts are RL's bytes. The mixed-quant MiMo
artifacts ([MiMo release scope](ds4-dfm-model-families.md#mimo-release-scope))
keep their own gates; nothing transfers between the two.

## Model and format

| Part | Shape | Storage |
|---|---|---|
| Attention | 48 layers: 9 full (0, 5, 11, 17, 23, 29, 35, 41, 47; 64 query / 4 KV heads) and 39 SWA-128 (8 KV heads, one sink per head); QK 192 (RoPE on 64), V 128 | Q8_0 `q\|k\|v` (de-interleaved) and `o_proj`, BF16 norms and sinks |
| Dense FFN | layer 0, 16,384 wide | Q8_0 |
| Routed experts | layers 1–47, 256 × top-8, FFN 2,048 | GVQ1 K8192 13-bit on all 47; GVR8 8-bit residual on 32 (5, 8, 11, 13, 17, 19–25, 27–30, 32–47) |
| Embedding | 152,576 × 4,096 | BF16 part, one 8,192-byte row per token from the page cache |
| Head | 152,576 × 4,096 | Q8_0 |
| Embedded MTP | three dense blocks | Q8_0 `mtp-q8_0.gguf`, loaded with `--mtp-draft` ≥ 2 |
| DFlash drafter | five SWA-1024 layers, block 8 | Q8_0 `dflash-q8_0.gguf` |
| Vision / audio | MiMo ViT (28 layers) and merger; audio tokenizer encoder, RVQ books, patch encoder | Q8_0 `vision-q8_0.gguf`, `audio-q8_0.gguf` |

`release.json` names 53 independent GGUF parts (98,520,657,024 bytes). The
host checks the recipe, every part's SHA-256 (`O_DIRECT`), the GVQ1/GVR8
headers and book links, the residual plan against `r13p8_layers` and the Q8_0
scales, then binds the Hugging Face tensor names. Engine open copies 86.82 GiB
(trunk) to the device.

## Execution on GB10

**Full-layer K/V.** The nine full layers store FP8 rows: per token
`[K 4×192 E4M3 | V 4×128 E4M3 | 4 K scales | 4 V scales]`, 1,296 bytes,
scale = f16(amax / 448) per (token, head); 11.39 GiB at 1M. SWA layers keep
f16 rings of 128 + one chunk. `DS4_MIMO2VQ_FULL_KV=f16` stores f16 rows
(diagnostic; does not fit 1M).

**Prefill (4,096-row chunks).** Q8_0 projections run MMQ. Routed experts:
host grouping, tokens quantized once to E4M3 (1×128 scales), MiniMax's
warp-specialized tokens × weights^T kernels (`cuda/minimax_mma8w.cuh`) for
gate, up and down, R13+8 layers adding the residual MMA; rows equal
`glmvq_mma8_kernel`'s bit for bit (`DS4_MIMO2VQ_MMAW=0` restores it). Full
layers attend with `mimo2_ga8::prefill8`: eight warps hold two query heads of
one KV head over 64-key tiles; the stored K codes are E4M3 MMA fragments as
they are (cp.async into shared memory), Q is E4M3 with one scale per (row,
head), and each V tile is widened to f16 once for 128 (row, head) pairs while
the next tile loads. SWA layers keep the f16 HMMA kernel.

**Decode and verify (1–8 rows).** Experts: one row on the GVQ decode kernels
(device routing, books in shared memory), 2–4 rows one pass over the rows'
expert union. Full-layer attention runs `mimo2_ga8::decode`: a block per
(key slice, row) stages whole 32-row steps of FP8 codes with contiguous
cp.async, and each of its four warps (one per KV head) feeds them into MMA
fragments with the 16 query heads of the KV head as the m16 rows; the k order
is chosen so each lane reads contiguous codes. Verify rows share the launch.
`DS4_MIMO2VQ_GA8=0` restores both earlier attention kernels.

**Drafting.** The three embedded MTP blocks draft one token each, in
turn, as SGLang runs them: block d's row i reads the target's row i and
token i+d+1 at position i. The trunk keeps its last 256 rows in a ring, and
each draft step first writes the block's rows still missing within its
128-position window, so the blocks see the prompt and every verified row. The
DFlash drafter proposes seven from its 1,024-position tap window. Both verify
in one trunk forward, whose rows pass through the head in one matmul.

**Numerics.** Full-layer prefill scores E4M3 Q against the E4M3 keys, P
single f16; decode and verify keep single f16 Q and P (the keys carry four
significant bits); exp2 softmax over 64-key prefill tiles. Routed
outputs sum in F32 and round once; SiLU and its product round to BF16.

## Serving

The serving surface is the mixed MiMo runtime's: opt-in continuous banks,
partial prefix reuse, snapshots, disk KV, a serial lane for media and
speculation. The release's vision and audio parts load with the model (no
`--vision`); its MTP part maps with `--mtp-draft` ≥ 2, its DFlash part with
`--mtp <release>/dflash-q8_0.gguf`. Memory plans (`--check-config`, 117.6 GiB
available, 4 GiB floor; trunk 86.83 GiB, vision + audio 1.33 GiB):

| Command (plus `-m` and the release directory) | Lane | Plan |
|---|---|---|
| `--mtp <release>/dflash-q8_0.gguf --mtp-mode on --mtp-draft 7` (1M) | serial | 110.27 GiB |
| `--mtp-mode on --mtp-draft 2` (1M) | serial | 109.79 GiB |
| `DS4_SERVER_CONTINUOUS=0`, MTP off (1M) | serial | 108.88 GiB |
| `-c 524288`, MTP off | one text bank + serial media | 111.77 GiB |
| `-c 262144 --max-seqs 2 --prefix-reuse partial`, MTP off | two text banks + serial media | 109.89 GiB |

A text bank and the serial media session each hold their own full-layer
rows, so the bank lane at 1M (123.2 GiB) is refused; one-sequence 1M serving
runs on the serial lane. A 1M-token sequence holds 11.39 GiB of FP8 rows and
0.79 GiB of SWA rings; the 4,096-row scratch is 2.17 GiB and the media
workspace (largest image, 32,768 patches) 2.38 GiB.

With no options the server plans the native context with two embedded MTP
drafts on the serial lane (109.79 GiB), whose prefix reuse is exact. Partial reuse across edits and forks, and the gated disk KV,
need banks: `--mtp-mode off --max-seqs 2 -c 262144 --prefix-reuse partial
--kv-disk-dir DIR`.

**Gates** ([receipts](benchmarks/mimo2-vq-serving-2026-10-30/README.md)):
the two-bank reuse gate (`tests/serving_reuse_live.py --family mimo2`, MTP
off) passes append (590 of 614 tokens cached, fork), edit (589, partial),
branch (615, exact) and a restart that restores 640 of 664 tokens from disk,
with byte-exact cold replays. The default 1M MTP plan answered text, two
images, a video and three audio clips correctly and copied a digit probe
with speculation active. The serial MTP lane (two drafts, reasoning off)
continued a chat over its live state, 182 of 199 prompt tokens reused, and
repeated the turn byte for byte after an unrelated prompt
([October 31 report](mimo2-vq-optimization-2026-10-31.md#gates-and-publication-sweep)).

## Performance

Three fresh `ds4-bench` processes per variant over the 32 frontiers of the
[2K–64K sweep](benchmarks/mimo2-vq-spark-64k-2026-10-31/README.md)
(`2fa8c08e`, UNCENSORED release), medians of run means: **1,294.9 prefill /
23.96 decode tok/s** without drafting (2K–32K: 1,400.4 / 24.46) and
**1,292.2 / 26.35** with two embedded MTP drafts. Prefill falls from
1,492.2 at 2K to 1,106.1 at 64K, decode from 24.86 to 23.04. Two MTP
drafts on chat prompts: 35.96 (code), 36.29 (a file rewrite) and 25.40
(Korean prose) against 24.9–25.1 plain. The first working build ran
597.9 / 20.72; the October 30 sweep 1,226.5 / 23.97
([October 30](mimo2-vq-optimization-2026-10-30.md) and
[October 31](mimo2-vq-optimization-2026-10-31.md) reports).

## Quality

`tests/mimo2_vq_quality` scores the engine against the handoff's teacher:
the BF16 source's final hidden states on its held-out set, through the
release's own Q8_0 head. Over the 24,785 assistant-turn targets of the
14 chat rows (UNCENSORED, FP8 K/V): KL 0.0884, top-1 91.77%, NLL 0.7486
(teacher 0.7421), within the release evaluation's 0.089 / 91.9% plus one
batching noise floor (KL ≤ 0.109, top-1 ≥ 89.7%); f16-query prefill
measured 0.0891 / 91.75%. Its `--long` mode pairs runs at long frontiers:
after 32K and 131K prompts, 1,024 teacher-forced positions move against
f16-query prefill by ΔNLL +0.0003 ± 0.0041 and +0.0030 ± 0.0044 (top-1
agreement 95.3% and 95.5%), inside the band of accepted arithmetic and
K/V-format switches (95.0–97.1%). Details:
[October 30](mimo2-vq-optimization-2026-10-30.md#numerical-gates) and
[October 31](mimo2-vq-optimization-2026-10-31.md) reports.

## Diagnostic switches

| Variable | Effect |
|---|---|
| `DS4_MIMO2VQ_GA8=0` | full-layer attention on `mimo2_hmma::prefill<Fp8>` and `mimo2_attn_split<true>` |
| `DS4_MIMO2VQ_GA8_QK=f16` | full-layer prefill on f16 Q and widened K (`mimo2_ga8::prefill`); `=fp8x2` adds a residual Q component |
| `DS4_MIMO2VQ_MMAW=0` | prefill experts on `glmvq_mma8_kernel` (bit-identical) |
| `DS4_MIMO2VQ_GAIN_DOT=0` | decode experts apply the gain per weight |
| `DS4_MIMO2VQ_FULL_KV=f16` | f16 full-layer K/V rows |
