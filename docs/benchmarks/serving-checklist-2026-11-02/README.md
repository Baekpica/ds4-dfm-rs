# Serving checklist on 19 local artifacts, one DGX Spark: November 2

`ds4 --serving-check` on one artifact per local `serving_caps` entry:
commit `2820c27b` (`ds4` MD5 `c97d57cb8c70…`, built before four test functions
were renamed: same `.text` and `.rodata`), GB10, CUDA 13.4, driver 615.71.10,
`nvidia-clocks.service` active (300–2200 MHz). One model at a time,
2.3 hours in all. Raw output: [outputs.txt](outputs.txt).

```sh
clear_cache
python3 tools/host_memory_guard.py --max-gib 128 --high-gib 128 \
  --reserve-gib 2 --trip-gib 2 --psi-full-limit 0 -- \
  ./ds4 --cuda -m MODEL --serving-check \
  --prompt-file tests/long_context_essay_prompt.txt
```

IQuest-Q1 and ThinkingCap import their weights from a weight owner, their
qualified shape:

```sh
ds4_weight_server --base MODEL --backend vmm --scope base --reserve-gb 28 \
  --no-repack-iq2-aligned --no-repack-q8-aligned --no-repack-q2k-aligned \
  --manifest /tmp/ds4-chk/weights.ipc
DS4_CUDA_WEIGHT_IPC_MANIFEST=/tmp/ds4-chk/weights.ipc \
DS4_CUDA_WEIGHT_IPC_SCOPE=base ./ds4 --cuda -m MODEL --serving-check ...
```

ThinkingCap's owner takes `--reserve-gb 32 --repack-q8-aligned`, its worker
`DS4_QWEN_PLE_DIR=<artifact>/PLE-FP8`. In one process ThinkingCap fails the
mapped expert range copy before its first prefill, a path this change does
not touch. V4.1 adds `--engram-dir`. The context fits the prompt (8,192;
V4.1 takes its metadata context); the essay renders to 4,935–5,801 tokens.

## Items

T is the rendered prompt, m its split on the prefill grid, G 16 greedy
tokens, X 16 filler tokens, U = T + G + X.

| Item | Runs | Compared with |
|---|---|---|
| reuse prefix | sync T[..m], then T | fresh T |
| reuse rewind | sync T[..m] + X, then T | fresh T; off the grid also T split at the kept row |
| reuse continue | sync T, decode G, sync U | fresh U and T + decode-fed G + X |
| disk session | continue's state saved, dropped, loaded | continue |
| snapshot session | continue's state saved, X synced, loaded | continue |
| banks warm, fork, disk | bank 0 holds T[..m]; T admitted warm, forked, restored | cold bank |
| banks partial | bank 0 holds T[..m] + X; T forked at m | cold bank |
| banks width | T and T[..m] + X admitted together | each alone |

An item works when the engine kept state (the sync's kept rows, the admit's
cached rows) and each row's top-1 is a reference's. Where an item has two
references, the most a row's top-1 lead over its runner-up moves between
them is the noise floor; a flip a reference ranks within it is a near tie.
Verdicts: `works` (declared, working), `no` (undeclared, not working),
`works, undeclared` (a declaration gap), `FAIL` (declared, not working),
`WRONG` (reuse flipped a row beyond the floor).

## Result

| Model | reuse prefix | reuse rewind | reuse continue | disk session | snapshot session | banks warm | banks fork | banks partial | banks disk | banks width | Result |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Kexaone236B (kexaone) | works | works | works | works | works | works | works | works | works | works | pass |
| K2Horizon375B (k2horizon) | works | works | works | works | works | works | works | works | works | works | pass |
| BeamVq (beam-vq) | works | works | works | works | works | works | works | works | works | works | pass |
| Step37Flash (step37) | works | works | works | works | works | works | works | works | works | works | pass |
| SolarOpen2_250B (solar) | works | works | works | works | works | works | works | works | works | works | pass |
| Glm53FlashVq (glm-vq) | works | works | works | works | works | works | works | works | works | works | pass |
| Ling30FlashVl (ling3vl) | works | works | works | works | works | works | works | works | works | works | pass |
| Ling31FlashVq (ling31-vq) | works | works | works | works | works | works | works | works | works | works | pass |
| Qwen38FlashNext (thinkingcap) | works | works | works | works | works | works | works | works | works | works | pass |
| NaiveN05Flash (naive) | works | works | works | works | works | works | works | works | works | works | pass |
| Motif3 (motif3) | works | works | works | works | works | works | works | works | works | works | pass |
| Dots3NotePrev (dots3) | works | works | works | works | works | works | works | works | works | works | pass |
| InklingSmall (inkling) | works | no | works | works | works | no | no | no | no | no | pass |
| Mimo26Flash (mimo-mopd) | works | works | works | works | works | works | works | works | works | works | pass |
| MinimaxVq (minimax-vq) | works | works | works | works | works | works | works | works | works | works | pass |
| Mimo26FlashVq (mimo-vq) | works | works | works | works | works | works | works | works | works | works | pass |
| Axk2Vq (axk2-vq) | works | works | works | works | works | works | works | works | works | works | pass |
| DeepSeek41Flash (v41-ivq) | works | no | works | works | no | no | no | no | no | no | pass |
| IQuestQ1 (iquest) | works | works | works | works | works | works | works | works | works | works | pass |

Inkling and V4.1 declare none of their `no` items: Inkling reuses exact
prefixes in one serial lane; V4.1 serves one sequence without snapshots.
K-EXAONE's and MiMo-V2.6-Flash's cold answers to T end after 5 and 12
tokens, so their bank items compare that many.

Near ties, each flip with its margins:

- k2horizon reuse continue: works, kept 5405, top-1 16/17 (prefill 16, decode-fed 15), near-tie 1 <= 1.832, max_abs 4.453e0; row 0: 373 +0.250 over prefill 1518 +0.012, decode-fed 479 +0.341
- beam-vq reuse continue: works, kept 5595, top-1 16/17 (prefill 15, decode-fed 16), near-tie 1 <= 0.389, max_abs 2.485e0; row 0: 130957 +0.173 over prefill 220 +0.059, decode-fed 220 +0.388
- step37 reuse continue: works, kept 5614, top-1 16/17 (prefill 16, decode-fed 16), near-tie 1 <= 1.477, max_abs 2.821e0; row 10: 769 +0.110 over prefill 10059 +0.119, decode-fed 10059 +0.123
- mimo-mopd reuse continue: works, kept 5595, top-1 16/17 (prefill 14, decode-fed 16), near-tie 1 <= 2.579, max_abs 6.979e0; row 6: 40 +0.440 over prefill 151645 +0.146, decode-fed 151645 +0.785
- mimo-vq reuse continue: works, kept 5595, top-1 16/17 (prefill 16, decode-fed 16), near-tie 1 <= 2.478, max_abs 4.835e0; row 0: 10417 +0.011 over prefill 525 +0.180, decode-fed 130957 +0.349

## What the procedure found

The first run (the engine before this change, a check with only the
prefill reference) failed:

- **rewind** in Solar, Motif-3, dots3, Step 3.7, Ling-3.0-VL, GLM VQ,
  Ling-3.1 VQ, Beam VQ, MiniMax VQ, A.X-K2 VQ and IQuest: each declares
  partial reuse, yet a serial session prefilled every non-extending prompt
  from zero. Serial lanes now resume at the common prefix.
- **continue, disk, snapshot** (one row in 17) in ten families: decode writes
  rows in another accepted arithmetic than prefill. The check now adds the
  decode-fed reference and the floor between the two.
- **K2-Horizon rewind and bank partial** (`no`): it has no sliding layer, so
  both now resume at any common prefix.

K-EXAONE passed, but reading its rewind showed the ring measured from the
live length: after a longer abandoned branch it could resume over
overwritten rows. It now counts from the newest row written.

## Declarations

`serving_caps` marks every measured item qualified. These were `present`:

| Entry | Now qualified |
|---|---|
| K-EXAONE 236B | reuse |
| K2-Horizon 375B | disk, snapshot; reuse is now partial |
| dots3-note-prev | banks, reuse, disk |
| Inkling Small | disk, snapshot |
| MiMo-V2.6-Flash (MOPD weights) | banks, reuse, disk, snapshot |
| MiMo-V2.6-Flash VQ | banks, snapshot |
| GLM-5.3-Flash VQ | banks, snapshot |
| Ling-3.1-flash VQ | banks, reuse, disk, snapshot |
| Beam-1.0 VQ | banks, reuse, disk, snapshot |
| MiniMax-M3.1-Flash VQ | banks, reuse, disk, snapshot |
| A.X-K2 VQ | banks, reuse, disk, snapshot |
| IQuest-Q1 | banks, reuse, disk, snapshot |
| DeepSeek V4.1 Flash | reuse, disk |

Not run: GLM 5.3 Flash, DeepSeek V4 Flash/Pro and Prism Bonsai have no local
artifact and keep their declarations. Darwin-180B-RSI and Qwen3.8 share
ThinkingCap's entry, Ling-3.1-flash VQ v1 the 1019 weights', V4.1 VQ the
iVQ weights'. MiniMax-H3 is not a `serving_caps` family. MTP, context,
prompt and media limits print as `no check yet`.
