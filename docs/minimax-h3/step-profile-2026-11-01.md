# Where our denoise step goes at the delivery canvas (2026-11-01)

`nsys` on a two-step 1344×768, 124-frame render of the Turbo v4 merge
(`~/prof/step2.nsys-rep` on the Spark, `nsys stats --report cuda_gpu_kern_sum`). Instrumented, the
two steps took 160.5 s (80.3 s/step) against 63.75 s/step uninstrumented, so the profiler costs about
26%; the shares below are what matters, and the per-step seconds are scaled by 0.794 to compare with
the uninstrumented runs.

## Our step, per class

| Class | Lapunches / step | Instrumented s/step | Share of denoise | Scaled s/step |
|---|---:|---:|---:|---:|
| Attention (`h3d_attention_mma_kernel` + `h3d_attention_mma_f32_kernel`) | 1,941 | 55.5 | **69%** | **44.0** |
| GEMM (`cutlass 80_tensorop_bf16 128x256`, `nvjet_sm121`) | 8,127 | 28.7 | 36% | 22.8 |
| Elementwise (`cast`, `uncast`, `swiglu`, `scale_add`, `gate`, `adaln`, ...) | 21,585 | 10.7 | 13% | 8.5 |
| RoPE + norm | 5,801 | 8.3 | 10% | 6.6 |

The classes overlap the VAE decode and the text encode as well, so they sum past 100% of one step.
The attention figure is exact: the trace holds 3,932 attention launches in total, which is
2 × 51 (the denoise steps) + 3,780 (`h3d_attention_mma_f32_kernel`) + 50 (the text tower), and the
denoise ones account for 111.0 s of the 160.5 s.

## The one number that matters

`h3d_attention_mma_kernel` averages **906 ms per call** (median 906, max 2,165) at S ≈ 45k, H = 56,
D = 128, non-causal. Rapid-H3 measures the same call on the same GPU:

| | ms/call at S ≈ 45k | vs ours |
|---|---:|---:|
| Ours, `h3d_attention_mma_kernel` | **906** | 1.00× |
| PyTorch flash (their stock bf16 path) | 458–565 | **1.6–2.0× faster** |
| comfy-kitchen INT8 | 385–389 | 2.3× faster |
| Rapid-H3 INT8 config 190 | 363–380 | 2.4–2.5× faster |

So we are paying roughly **20 s of every 63.75 s step** for an attention kernel that is 1.8× off a
stock one, in the same arithmetic (bf16) and at the same numerics. Nothing about that gap needs a
precision change or a quality decision: it is the block-sparse `h3d_sol_forward` plan running what is
in practice a dense attention, and it should be closable to their number first.

For scale, matching their flash number alone would take the step from 63.75 s to ≈43 s (1.5×), and the
INT8 kernel would take it to ≈38 s (1.7×), before any of the GEMM or decode work.

## Secondary observations

- **GEMM is already competitive**: 22.8 s/step against their 15.8 s of *cuBLAS bf16* at the same
  canvas is within the 1.4× that our larger token count and extra small projections explain. The
  `H3_DIT_GEMM=fp8` route (measured at 212 TFLOPS against 99 for bf16 in this port) is still the
  cheapest 2× on this class and has never been run at the delivery canvas.
- **RoPE + norm is 6.6 s/step** where theirs is 0.85 s. That is `h3d_qkv_rope_kernel` at 112.7 ms per
  call, 51 calls per step. It is worth a look after attention: our RoPE is 8× their entire rope+norm
  class.
- **`h3d_attention_mma_f32_kernel` runs 1,890 times per step** at 1.17 ms each (2.6 s/step). An f32
  attention launched nearly 1,900 times inside one denoise step is suspicious on its own; it should be
  a small number of larger launches.
- **Elementwise is 21,585 launches per step** (8.5 s), so launch overhead is a real entry on the bill
  here too, as it was for them (2,400 kernels/step).

## Consequence

The plan in `rapid-h3-study-2026-11-01.md` §5 holds, with attention now confirmed as first. Measure
the dense attention against a straight flash-style kernel on the same q/k/v before touching sparse or
precision, because a dense plan should not pay block-sparse scheduling costs.
