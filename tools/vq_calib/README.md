# VQ calibration tools

Calibration-weighted GVQ conversion: activation statistics from a
calibration corpus drive the Lloyd codebook training, the vector assignment
and the row gains of a YoungAi-style shared-codebook VQ (8-dimensional
vectors, E4M3 codebooks per MoE layer, F16 row gains), and the importance
of every non-expert Q4_K tensor. The method, the statistics and the
measurements are in [docs/vq-calibration.md](../../docs/vq-calibration.md).

These are the tools that produced
[Baekpica/A.X-K2-VQ-GGUF](https://huggingface.co/Baekpica/A.X-K2-VQ-GGUF)
from [skt/A.X-K2](https://huggingface.co/skt/A.X-K2/tree/d0805cfd1fc647b4078c0d9f232c80d34ee0c893)
(FP8 E4M3 with 128x128 block scales). The family-specific parts are the
reference forward (`axk2_ref.py`), the calibration mix (`build_calib.py`) and
the non-expert policy (`convert_nonexpert.py`); the weighted Lloyd, the GVQ v2
container and the release tools are family independent.

## Files

| File | Role |
|---|---|
| `axk2_source.py` | FP8 + `weight_scale_inv` reader; `dequant()` = fp8 x scale per 128x128 block in FP32, cropped at partial blocks |
| `axk2_ref.py`, `selftest_ref.py` | layer-streamed A.X-K2 forward (MLA + DSA indexer top-2048, gated norms, grouped router with −inf masking) with statistics hooks; selftest against transformers' `AXK2DecoderLayer` |
| `build_calib.py` | seeded calibration and held-out token sets (raw web/code/math text, A.X-K2 chat template think/non-think, tools, long context) |
| `collect_stats.py`, `run_collect.sh`, `reduce_stats.py` | data-parallel statistics collection (one rank per GPU), FP64 sums, rank reduction |
| `axk2_weighting.py` | weighting modes `none`, `row`, `imatrix`, `imatrix_freq`, `output`; rare-expert shrinkage; column floor and tempering |
| `cuda/axk2_vq_convert.cu`, `axk2_vq_gpu.py` | weighted nearest assignment, weighted FP64 Lloyd, FP32 normalization, least-squares gain refit, 9/10-bit pack and restore |
| `convert_layer.py`, `run_recipe.py` | one MoE layer -> one GVQ v2 part with receipt; resumable multi-GPU driver |
| `measure_layer.py`, `run_measure.sh`, `summarize_measure.py` | variants of one layer judged on held-out activations (routed-output error) |
| `axk2_vq_format.py` | GVQ v2 matrix, GGUF part writer and strict reader, release verification |
| `convert_nonexpert.py` | Q4_K (importance-weighted, `gguf-tools/quants.c`), Q8_0 indexer, source tensors, BF16 embedding part |
| `memory_budget.py` | exact bytes of the recipe and the 256K DGX Spark budget |
| `assemble_release.py`, `publish_release.py` | release directory, manifests and hashes; Hub upload with readback |
| `eval_e2e.py`, `router_mask_ab.py` | teacher-forced end-to-end comparison and goldens (full-vocabulary scoring in 2,048-position chunks); router-mask A/B on the FP8 teacher |
| `run_variants.sh`, `eval_variants.sh` | several full 60-layer variants one after another on all GPUs, each evaluated end to end as soon as its parts exist |
| `select_variant.py` | writes the chosen variant (read from its own receipts) into `recipe.json`, the release docs and the card |
| `generate_compare.py`, `cuda/gen_vq.cu`, `gen_report.py` | greedy/sampled generation from release parts (GPU-resident VQ indices, KV cache, layers pipelined over GPUs) and a side-by-side report |
| `verify_source.py` | every source file against the Hub (LFS SHA-256 or git blob id) |
| `tests/` | `test_axk2_vq_format.py` (CPU), `test_axk2_vq_gpu.py` (one GPU), `fake_stats.py` (stand-in statistics for pipeline tests) |

## Pipeline

Run from this directory. `SRC` is the checkpoint, `W` a work directory.

```sh
SRC=/models/A.X-K2  W=/workspace/axk2  RECIPE=$W/recipe/recipe.json

# 1. calibration and held-out sets (downloads under $W/calib/raw)
python3 build_calib.py all --out $W/calib --model $SRC
# 2. reference forward against transformers (layers 0 and 1)
python3 selftest_ref.py 0 1 --source $SRC
# 3. statistics: one rank per GPU, then the reduction into $W/stats
./run_collect.sh $W/calib/calib.safetensors $W/calib/heldout.safetensors $W/stats "0 1 2 3 4 5 6" --source $SRC
# 4. choose the weighting on held-out activations, one layer per GPU
./run_measure.sh "1 4 8 15 24 30 41 60" "0 1 2 3 4 5 6 7" $W/work/measure $W/stats
python3 summarize_measure.py $W/work/measure
# 4b. full variants end to end (each ~3 min to convert on 8 GPUs, ~2.5 min to evaluate on one)
./run_variants.sh A-none:none:none:shared C-imatrix:imatrix:ls:split3:0.75
./eval_variants.sh A-none C-imatrix
# 4c. an independent held-out set for the final choice, its teacher pass, and generation
python3 build_calib.py build-heldout2 --out $W/calib --model $SRC
python3 collect_stats.py --rank 0 --world 1 --calib <tiny calib> --heldout $W/calib/heldout2.safetensors --out $W/stats-heldout2
python3 generate_compare.py --help
# 5. routed experts (resumable; layers round-robin over the GPUs)
python3 run_recipe.py --source $SRC --recipe $RECIPE --stats $W/stats --work $W/work/convert \
    --output $W/parts --gpus 0,1,2,3,4,5,6,7 --mode imatrix --gain-refit ls --books split3 --col-power 0.75
# 6. non-experts (CPU)
python3 convert_nonexpert.py --source $SRC --recipe $RECIPE --output $W/parts --imatrix-dir $W/stats
# 7. release directory (one expert variant + the non-expert parts) and its strict verification
python3 select_variant.py --experts-dir $W/parts --measured $W/docs/measured.md
python3 assemble_release.py --parts $W/parts --experts-dir $W/parts --source $SRC --recipe $RECIPE \
    --card $W/docs/README.public.md --docs $W/docs/release --tools . --out $W/release
python3 axk2_vq_format.py $W/release --release --workers 16
# 8. end-to-end check against the FP8 teacher, goldens for the runtime
python3 eval_e2e.py --source $SRC --parts $W/parts --stats $W/stats --out $W/eval --goldens
# 9. upload (card first, then weights, remote hash readback)
python3 publish_release.py --release $W/release --message "..."
```

`--mode`, `--gain-refit`, `--books`, `--col-power`, `--col-floor` and
`--shrink-tokens` default to the recipe's `calibration` selection; the step-5
flags above are one of the measured candidates, and the release records the
one it used in `recipe.json` and in every part's `axk2_vq.weighting`
metadata and receipt. `memory_budget.py --ctx 262144 --r10-layers 16,20,24,28,32,36,40`
prints the byte budget of the released placement (bytes depend only on how many layers are 10-bit).

## Hardware and timings (A.X-K2 run)

One Runpod host: 8x RTX PRO 6000 Blackwell Server Edition (96 GB, sm_120),
CUDA 13.2, torch 2.13, 256 CPU threads, 1.5 TB RAM (the 647 GiB FP8 source
stays in the page cache).

| Step | Time |
|---|---|
| Statistics, 3.92M calibration tokens through 61 layers on 7 GPUs | 7.5–9 min per rank |
| Expert conversion, one MoE layer on one GPU | ~12 s (read 2 s from the page cache, Lloyd 2–3.5 s, assign + pack + verify + write ~9 s) |
| Non-expert conversion (CPU, 96 threads) | 6.7 min |
| `measure_layer.py`, one variant of one layer (53K held-out tokens) | 5–10 s |

| Kernel | K = 512 | K = 1024 |
|---|---:|---:|
| Nearest assignment, unweighted | 4.7e9 vectors/s | 2.4e9 vectors/s |
| Nearest assignment, per-column weights | 2.95e9 vectors/s | 1.5e9 vectors/s |
| Weighted Lloyd iteration, 16.5M-vector pool | 28 ms | — |

## Tests

```sh
python3 tests/test_axk2_vq_format.py                 # CPU: GVQ v2 encode/decode/reader rejections
python3 tests/test_axk2_vq_gpu.py --work /tmp/vq-test  # one GPU: weights == 1 bit-identity, packing parity,
                                                       # weighted Lloyd, gain refit, throughput
```

`test_axk2_vq_gpu.py` also checks the unweighted assignment against Beam-1.0's
`beam_vq_convert.cu` at K = 4096 when `--beam-cu` points to it (skipped
otherwise). Build outputs (`cuda/libaxk2_vq_convert_sm*.so`, `native/`) are
not tracked.
