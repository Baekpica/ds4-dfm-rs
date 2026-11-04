# CPU MoE, measured fairly — 2026-11-04

The agy review (`cpu_moe_agy.md`, verdict STOP) asked for one thing before the
CPU-expert question closed: replace the warm-L3 scalar microbenchmark with a
fair AVX2 + cold-DRAM instrument (review section 5.A). This is that
measurement, run on the i7-14700F host.

## The instrument

`tests/gemma4_cpu_expert_gemv.c`, four changes over the first version:

1. A persistent barrier threadpool. Threads are created once; each sweep is
   one barrier pair, and 32-row chunks are stolen from an atomic cursor, so
   no `pthread_create`/`join` sits inside the timed region.
2. Codebooks pre-expanded to FP16 at load. Every finite E4M3 value is exact
   in FP16, which the program proves by converting all 8,448 words back and
   comparing. A codeword becomes one 128-bit load plus `vcvtph2ps`.
3. An AVX2 + F16C inner loop, four codewords per iteration, four `__m256`
   accumulators, the row gain hoisted out of the dot. The hoist is Class-B
   (the scalar path rounds `gain*word` to bf16 per weight); an exact-rounding
   variant is kept in the binary to price that difference.
4. A cold sweep over 720 distinct record copies — 450.7 MiB, 13.6x the 33 MiB
   L3 — so the reported token time is the real expert read, not an
   extrapolation from one cached record.

The scalar reference and a double-precision dot are in the same binary, so
every speed number carries a correctness number beside it.

## Numbers

Host: Intel i7-14700F, 8 P-cores (5.3-5.4 GHz, HT) + 12 E-cores (4.2 GHz),
28 hardware threads, 31 GiB. Governor `powersave`, 3.5-5.4 GHz observed.
The machine is shared with the owner's `grateful` fleet (about 1.7 cores) and
a `vq-quality` run (0.6 cores), so minima are quoted and the streaming floor
was measured in the same windows.

| Arm | Time | Note |
|---|---|---|
| scalar, 1 thread, warm record | 1.95 ms/matrix | the old arithmetic, no spawn tax |
| scalar, 20 threads, 720 cold records | 117.3 ms/token | the old loop's real cold cost |
| AVX2, warm single record | 0.050 ms at 28t | not comparable to a token |
| AVX2, 720 cold records | 15.31 ms at 20t | 28t 15.83, 8t 24.64 |
| pure AVX2 stream read, same 472.6 MB | 9.37 ms at 20t | 50.45 GB/s, the bus floor |

The kernel is 1.63x the measured streaming floor, so the remaining gap is
compute, not bandwidth. Per codeword that is 7.4 cycles at 20 threads and
~4.3 GHz, against a bus floor of 4.5.

The old instrument's 0.387 ms/matrix extrapolation (278.4 ms/token) was
inflated about 2.4x by the per-matrix thread spawn; the same loop in the
persistent pool costs 117.3 ms/token cold. The AVX2 kernel is 7.7x faster
than that.

## Correctness

- FP16 book expansion: exact on all finite entries (checked at startup).
- AVX2 vs scalar reference: max |difference| 4.77e-7 on max |y| 2.93; the
  worst per-row relative figure is 6.0e-5 and belongs to a row whose dot
  nearly cancels (|y| ~ 5e-4), which is the expected signature of reduction
  order, not of wrong weights.
- Hoisted and exact-rounding variants agree to nine digits on this record, so
  the gain hoist costs nothing measurable here.

## agy's advice, and what it measured

agy read the disassembly and counted 89 instructions per four codewords. Its
ranking, with my measurements beside each lever:

1. Step by 8 codewords with wide SWAR loads (predicted -1.5 to -2.5 ms).
   Implemented and measured: 16.35 ms against 15.61 ms for the 4-wide
   version, a 4.8% regression. The byte loads it targeted are single-uop
   loads on two spare AGUs; replacing them with shifts and masks moved work
   onto the ports that are actually saturated. Not adopted.
2. Two-row blocking, sharing the x loads (predicted -0.8 to -1.5 ms).
   Implemented and measured twice: one alternating best-of-six favored it by
   1.1 ms, a best-of-ten with medians favored the one-row kernel by 0.7 ms.
   The effect is inside this machine's noise. Not adopted.
3. P/E pinning (predicted <3%). Measured: 8 P-cores 23.9 ms, 16 P threads
   25.7 ms (SMT hurts), 12 E-cores 38.6 ms, all 28 threads 18.5 ms in the
   same window. The E-cores add real throughput; no pinning adopted.
4. Software prefetch (predicted neutral to worse). Not implemented; the
   kernel is not latency-bound on the 128 KiB book, which stays in L2.
5. Eight accumulator chains (predicted neutral or worse). Not implemented.
6. Pre-expanded FP32 book (predicted +2 to +5 ms regression). Not
   implemented; the reasoning (double the gather and the footprint) is
   sound.

## Verdict

The review's conclusion stands, but its numbers were too pessimistic and the
reason has changed. It estimated 38-55 ms/token for a perfect AVX2 CPU MoE;
the measured kernel does 15.3 ms. Against the GPU's 9.29 ms/token for the
same 240 experts, CPU experts are 1.65x slower, not 4-6x. End to end at
batch 1 that would put CPU MoE at roughly 35-38 ms/token (26-29 tok/s)
against the GPU's 29-32 ms (31-35 tok/s): about 85% of the GPU rate, not a
third.

What still rules it out is not raw expert speed:

- It consumes all 28 hardware threads to reach that number, on a machine
  that also serves the fleet.
- MTP verify collapses it: at draft 6 the seven rows route to roughly
  1,500 experts per step, 2.8 GiB of expert reads, and the review's
  DDR5 analysis puts that at 180-250 ms/step.
- The GPU active-expert path already exists and is measured.

So: no host decoder port. The instrument stays in the tree as the number any
future CPU-expert proposal has to beat.
