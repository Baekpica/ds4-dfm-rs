# GB10 (DGX Spark) hardware limits

This engine is tuned for one NVIDIA DGX Spark. This page is the single place
for the hardware limits that bound its kernels and serving plans, and for how
each number was measured; [AGENTS.md](../AGENTS.md#gb10-dgx-spark-bottleneck-know-how)
and the family pages cite it instead of repeating figures. Rooflines for an
optimization round (the 85-90% targets of the
[phase order](prefill-decode-optimization-playbook.md#phase-order)) start here.

## Identity

| Item | Value |
|---|---|
| GPU | NVIDIA GB10, compute capability 12.1, 48 SMs |
| Build target | `-gencode arch=compute_121a,code=sm_121a` (`make cuda-spark`); `-arch=sm_121a` alone emits plain `sm_121` and ptxas rejects the FP8/FP4 `mma` kinds |
| Tensor-core instructions | `mma.sync` only: ptxas rejects `wgmma` and `tcgen05` for `sm_121a`; `cp.async.bulk` (TMA-style copies) and mbarriers are available |
| SM clock | device attribute 2,418 MHz (nvidia-smi lists 3,003 MHz). Observed under load: 2,184-2,190 MHz while the host capped clocks at 300-2,200 MHz (2026-09-14 to 2026-10-21); 2,410-2,590 MHz on 2026-10-24 with no cap in effect |
| Memory | 128 GB LPDDR5X shared by CPU and GPU; Linux sees 121.6 GiB |
| Driver, toolkit | 615.71.10, CUDA 13.4 (2026-10-24) |

Every compute peak below scales with the SM clock: record the observed clock
next to any A/B or roofline figure.

## Compute (dense `mma.sync`)

| Instruction | FLOP per clock per SM | At 2,190 MHz | At ~2.5 GHz |
|---|---:|---:|---:|
| BF16 or F16 `m16n8k16`, F32 or F16 accumulate | 1,024 | 107.8 TFLOPS | 128.4 TFLOPS |
| TF32 `m16n8k8` | 512 | | 64.2 TFLOPS |
| E4M3 `m16n8k32` (FP8), S8 (INT8 TOPS) | 2,048 | 215.6 | 255.0 |
| NVFP4 `kind::mxf4nvf4` `m16n8k64` (E2M1, UE4M3 scale per 16) | 4,096 | | 509.5 |

The 2,190 MHz column is [Beam's floors](beam-vq-optimization-2026-10-21.md#floors)
(2026-10-21); the ~2.5 GHz column is `tools/gb10_peaks` on 2026-10-24. F16
accumulation is not faster than F32 accumulation. cuBLAS BF16 GEMMs reached
65-90 TFLOPS at the 2.2 GHz cap; the engine's own kernels are the fast path.

One warp per scheduler reaches the E4M3 peak with two to four independent
accumulator chains: an `m16n8k32` E4M3 MMA issues every ~6.3 ns (~15
clocks) per scheduler, and a dependent chain waits ~11.3 ns (~27 clocks)
per MMA: 139 TFLOPS with one chain, 232 with two, 248 with four
(`tools/gb10_peaks`, 2026-10-24, SM clock 2,489 MHz after the run). A
warp-specialized kernel with one MMA warp per scheduler is therefore
bounded by its operand loads (shared-memory wavefronts, their latency), not
by MMA latency.

## Memory system

| Level | Limit |
|---|---|
| DRAM | LPDDR5X spec 273 GB/s. Streaming reads 241-246 GB/s (Beam floors), 231 GB/s from `tools/gb10_peaks` (four 16-byte loads in flight per thread); a device copy moves 238-244 GB/s (read + write). Decode streaming kernels reach ~250 (BF16 dot), 233-237 (FP8 dot), 216 GB/s (Q4_K) |
| L2 | 24 MiB (persisting carve-out up to 18 MiB); 2.0-2.1 TB/s reading an 8 MiB working set |
| Shared memory | 100 KiB per SM; 99 KiB (101,376 B) per block with the opt-in attribute, 48 KiB static; 32 banks x 4 B, 128 B per clock per SM |
| Registers, threads | 64K 32-bit registers per SM (at most 255 per thread); 1,536 threads per SM |

## Storage

| Device | Limit |
|---|---|
| NVMe `ESL04TBTLCZ` (3.6 TB) | `O_DIRECT`: 6.3 GB/s sequential; 6.2 GB/s of 72 KB random reads at eight in flight (84K IOPS, ~95 µs per wave); ~90K random 4 KiB IOPS; `STATX_DIOALIGN` 512-byte offsets |

## What the limits imply

- **Decode is bytes per token.** A decode step's floor is its bytes at
  ~240 GB/s; a kernel far below that rate is a target, one near it only gets
  faster with fewer bytes.
- **One CTA per SM beside a large table.** A 64 KiB VQ codebook leaves
  ~35 KiB of shared memory for staging; stage sizes and pipeline depth are
  budgeted against 101,376 B, not against occupancy.
- **Prefill MoE streams every expert once per chunk.** Routed-expert weight
  bytes per token fall with the chunk size, so the chunk sets the expert
  floor: MiniMax's 13-bit experts read ~1.8 GB per layer per chunk, 0.21 ms
  per token at 2,048 rows and 0.10 at 4,096, against 0.14 ms of FP8 MMA work.
- **L2 is small for MoE streams.** 48 CTAs x (current + prefetched) blocks
  must fit beside the activations in 24 MiB; pace next-unit prefetches.
- **Unified memory is shared with the host.** GPU allocations, the page cache
  of mapped weights and every process draw from the same 121.6 GiB: run
  model loads under the memory guard, one resident model at a time.
- **Clock policy changes the peaks.** Compare variants at the same clock and
  state it.

## Re-measuring

`make tools/gb10_peaks && ./tools/gb10_peaks` prints the `mma.sync` rates, a
streaming read, a device copy and an L2 read. It loads no model; run it on an
idle GPU and sample `nvidia-smi --query-gpu=clocks.sm --format=csv` beside it.
