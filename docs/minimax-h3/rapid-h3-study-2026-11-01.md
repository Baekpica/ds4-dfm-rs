# What Rapid-H3 does that we do not (study, 2026-11-01)

Rapid-H3 (`github.com/larryvrh`'s sibling project, read from a clone at `/data/Rapid-H3`) is an
independent ComfyUI kernel layer for MiniMax-H3 on the same GB10 we run. It is the only public work
that profiles our exact model on our exact machine, so its numbers are the benchmark this port should
be measured against. Nothing here is copied code: the point is which ideas are worth porting into the
Rust path and which are already known dead ends.

**Their result, warm, models loaded, one DGX Spark:** stock ComfyUI bf16 at 10 steps 590–630 s for a
768×1344, 141-frame clip; their `ts_s6` preset ≈194 s and `ts_h2` ≈114 s. Per full-size step
56.7 s bf16 → 29.6 s (1.9×) at 141 frames.

## 1. Their bf16 step profile, which we can compare against ours

1344×768, 124 frames (≈40k tokens), per denoising step:

| Class | Their ms/step | Share |
|---|---:|---:|
| Attention (PyTorch flash, 52 calls × ~458 ms) | 23,839 | 55% |
| Linear GEMMs (cuBLAS bf16, ~79 ms per call) | 15,800 | 37% |
| Modulation / elementwise (unfused, 2,400 kernels) | 2,735 | 6% |
| RoPE + norm | 850 | 2% |
| **Total** | **43.2 s** | |

**Ours, measured on this branch** (`outputs/turbo-delivery/run.log`): 63.75 s per step at the same
canvas and frame count, dense attention through `h3d_sol_forward` with a dense plan. If our GEMM and
elementwise sides are anywhere near theirs, then our attention is on the order of 40 s per step
against their 23.8 s: **we are roughly 1.7× slower than a stock flash-attention path**, before any
precision change. That is the first thing to fix, and it needs no quality decision.

## 2. GB10 dense rates on real H3 shapes (their microbenchmarks)

| Op | bf16 | FP8 e4m3 (per-tensor) | INT8 |
|---|---:|---:|---:|
| qkv 5376→16128 | 67.7 ms (102 TF) | **35.4 ms (196 TF)** | 41.5 ms (167 TF) |
| out_proj 5376→5376 | 23.2 ms (100 TF) | **11.5 ms (201 TF)** | 14.0 ms (165 TF) |
| fc1 5376→28672 | 118.3 ms (104 TF) | **62.2 ms (198 TF)** | 73.8 ms (167 TF) |
| fc2 14336→5376 | 65.7 ms (94 TF) | **31.9 ms (193 TF)** | 38.4 ms (161 TF) |
| Attention S=40k, 56×128 | flash 502 ms (91 TF) | no kernel | ck INT8 311 ms (148 TF) |

Three facts that decide the GEMM strategy:

1. GB10 does ≈100 TFLOPS bf16, ≈200 FP8, ≈165 INT8. **FP8 is the fastest format here, not INT8**,
   and it is already exposed in this port as `H3_DIT_GEMM=fp8`, measured in our own campaign at
   212 TFLOPS against 99 for bf16. We have never run it at the delivery canvas.
2. Row-wise FP8 `_scaled_mm` is broken on sm_121 (their finding): only per-tensor scaling works, and
   per-tensor FP8 diverged 5.7% against INT8's 2.8%, so they rejected FP8 for quality. Our `fp8`
   route is per-tensor, so its quality is the open question, not its speed.
3. A slow INT8 GEMM is a kernel problem, not hardware: ComfyUI's int8_convrot file ran *slower* than
   bf16 (108 vs 79 ms per GEMM) purely because of its kernel.

## 3. Techniques worth porting, in order of value here

**a. INT8 attention with the softmax mutex (their config 190).** Q/K/V all INT8 with a signed
Hadamard-128 rotation, `mma.sync.m16n8k32.s8`, per-tile u8 P quantization, integer-domain row max,
fp32 fold. The ordering insight is theirs alone: the softmax mutex section is entered *after* the row
max shuffles, so one warp group runs its latency-bound shuffle→`MUFU.EX2` chain while the other holds
the section. Bit-identical to their previous config, 1.8–2.4% faster per call. Their attention runs at
170–175 TF against comfy-kitchen's 162–164. Their measured optimum is not exotic: 1 CTA/SM, 256
threads, 8 warps × 16 rows, 255 registers with no spills, 128-key tiles, ≈5.3 non-IMMA instructions
per IMMA, which their own occupancy microbenchmark caps at 215–220 TF.

**b. Fuse the query pre-pass.** q/k RMSNorm + split-half RoPE folded into the Q/K quantizer so the
roped q/k never reach memory: 4.8 ms + 3.1 ms → 3.9 ms per layer call, and −1.0 s per video decode.
In our path the equivalent is folding the DiT's qkv RoPE into the attention input quantizer.

**c. Quantizers on a persistent grid over L2 (their `_rot_quant_wide_kernel`).** Same arithmetic, but
1–2 programs per SM re-read rows from the 24 MB L2 instead of DRAM, with `[BM, NG·256]` tiles: 1.6–2.0×
per call, `torch.equal` to before. Real numbers at M=44,455: fc2 input 14.55 → 8.42 ms, out_proj input
7.27 → 4.42 ms, norm/mod 9.49 → 4.81 ms.

**d. The video decode, which is our second-biggest cost.** Theirs is 18–19.5 s warm for 141 frames
(int8_convrot, all 28 tiles of a temporal chunk per call, SwiGLU in the w1 GEMM epilogue, q/k rms+rope
fused into the attention quantizer) against fp16's ≈37 s. Ours is **53.5 s** at 124 frames with
`bf16i`/`tensor`. There is a ≈3× decode on the table, and our own campaign already measured that
`bf16i` differs from `fp32`/`scalar` by bytes, not by look.

**e. Their step-count recipe.** 6 steps instead of 10, no Turbo LoRA in the shipping presets; and
`ts_h2` = 0.7× draft → learned ×2 latent upscale → one full-size refine step. That is the same shape
as the two-stage route this branch deleted, which is a useful independent confirmation that the idea
was sound and the implementation was the problem.

## 4. Their dead ends, so we do not spend a step on them

| Idea | Their result |
|---|---|
| Hand-written persistent INT8 GEMM | 185–191 TF, below CUTLASS's 198–205. Main loop alone 205–213, so the epilogue is the loss. Not wired in. |
| Triton INT8 GEMM | Not adopted. |
| Locking GPU clocks | **Slower**: 233 vs 248 TF sustained; no settable power limit on GB10. (Also barred by our own owner rule.) |
| Paired half-precision softmax exps | sm_121 has no paired `MUFU.EX2`: `ex2.approx.f16x2` assembles to two F16 exp2 plus a PRMT. +2% slower. |
| 3-input `max.f32`, packed `f32x2` FMA | Both assemble to two instructions on sm_121. No gain. |
| Lazy softmax rescale | ≤0.7%: the fold already rides in an FFMA off the critical path. |
| Sparse attention (Sol-Attn, τ=1.3) | Fast and **lossy**: 24.0 → 17.5–18.0 s/step, but per-step divergence 4.8–14.0% against the recipe's 1.3–2.5%, and 11.6% video error at 480p. Rejected, including on the refine step only. |
| NVFP4 GEMM | 356 TF dense on GB10 (~1.8× INT8) but 4-bit activations are lossy. Rejected. |
| Distilled 4-step LoRAs (DMAD, PDMD) | Tested, not adopted in their presets. |
| Smaller drafts (0.6×, 0.5×), 4–5-step `ts_s6`, high-low-high sampling | Tested, not adopted. |
| External fused VAE decoder | Tested, not adopted. |

The sparse-attention result matters directly to us: this port can already run `H3_DIT_SOL=0.9`, and
their measurement says that is a 1.3× step-time trade for a large divergence, **not a free lever**. It
belongs behind an explicit fast mode, not in the delivery path.

## 5. What this port should do next, in order

1. **Profile our step at 1344×768 and attribute attention vs GEMM vs elementwise.** Every decision
   above depends on whether our attention really is ~40 s of the 63.75; the numbers land in a second
   receipt rather than being assumed here.
2. **Make the dense attention match a stock flash kernel** before changing its precision: our dense
   case goes through the block-sparse `h3d_sol_forward` plan, and a dense plan should not pay sparse
   scheduling costs.
3. **Run `H3_DIT_GEMM=fp8` at the delivery canvas** and compare wall and detail against bf16 and
   against the Turbo v4 merge. It is one environment variable and it was measured at 2.1× bf16 here.
4. **Port the decode route that pays 3×**: int8/convrot VAE with all tiles of a chunk in one call,
   SwiGLU in the epilogue, and the q/k pre-pass fused.
5. Only then consider INT8 attention, which is a 1.26× on the attention block against bf16 flash, and
   the fused RNG/quantizer work, which is worth a few percent per call.

## 6. Provenance

Read from `/data/Rapid-H3` at commit `1c8dd2f` (clone 2026-11-01). Sources: `README.md`,
`docs/kernels.md` (their full experiment log, 614 lines), `custom_nodes/rapid_h3/csrc/attn_cake/DESIGN.md`
(the INT8 attention design), `custom_nodes/rapid_h3/csrc/int8_gemm_p.cu` and `rr_row.cuh` (kernels),
`bench/results/` (their measurements). Their project is licensed separately; nothing is copied here.
