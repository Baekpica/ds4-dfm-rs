# Step 3.7 Flash: post-landing performance campaign

This campaign starts from landed main (`59ee609` / merge `1d83368`)
after the BASE-artifact round in
[step37-optimization-2026-09-13.md](step37-optimization-2026-09-13.md).
That earlier work is not counted here.

## Protocol

- One GB10, CUDA 13.3, `sm_121`. MQ83 nine-shard GGUF plus official Q8 MTP.
- Resident raw MTP-only VMM owner. Local BASE artifacts stay on.
- `speed-bench/promessi_sposi.txt`, SHA-256
  `f53e0d80cb2d4492d24ebd63c7000c397b16ae70f9bf09b3763e5d8323ec209f`.
- 2048 prompt tokens, 64 greedy generated tokens, allocated context 2120,
  `--mtp-draft 3`. The campaign started at chunk 512 and left the default
  at 2048.
- Fresh processes, warmup-then-fresh, three unprofiled samples plus nsys.
- With speculation, report MTP tok/s and a `DS4_MTP_SPEC_DISABLE=1`
  ordinary-decode control. Gate decode on the ordinary control, not MTP
  acceptance.

Pinned binaries, hashed workload and scouts live under
`scratch/step37/perf-r2/`. `ds4-perf compare` stays **Incomparable**:
the optional CUDA identity helper is absent. Numbers below are manual
reviews of the scout CSVs.

## Scoreboard

Campaign baseline is the landed binary (`bench-baseline`).

| Round | Prefill tok/s | vs base | Decode MTP | Decode ordinary | Verdict |
|---|---:|---:|---:|---:|---|
| baseline | 698.88 | — | 21.32 | 19.49 | locked |
| Prefill 1 Step SWA HMMA | **917.55** | **+31.3%** | 17.60 | **19.70** | retained |
| Prefill 2 chunk 1024 | **1063.70** | **+52.2%** | 17.01 | **19.62** | retained |
| Prefill 3 chunk 2048 | **1233.87** | **+76.5%** | 21.40 | **19.72** | retained |
| Decode 1 verify GQA | 1243.02 | +77.9% | **22.43** | **19.83** | retained |
| Decode IQ2 pair-vec n≤8 | 1234.43 | +76.6% | 21.27 | 19.67 | rejected |
| Decode Q4 vec @32 assign | 1235.26 | +76.8% | 20.34 | 19.64 | rejected |

MTP decode tok/s on Prefill 1 fell because acceptance/launch counts
changed. Ordinary decode did not regress. The two later Decode probes
lost MTP tok/s and were not kept. The campaign closed after Decode 1.

## Prefill 1: Step SWA HMMA tiles

Bottleneck: `exaone_attn_prefill_kernel` 0.744s / 144 launches (25.8% of
profiled Prefill). Full-attention layers already used
`ds4_fattn_hmma_gqa2_kernel`. Sliding layers stayed on the warp walk
even though that kernel already applies the window mask and `src % kv_cap`
ring.

Step SWA is 96 query / 8 KV heads, window 512, head dim 128. The GQA2
pair kernel requires an even group; group size 12 is even. EXAONE/K2
SWA remains on the warp path. `DS4_EXAONE_PREFILL_HMMA=0` restores every
warp path. `DS4_STEP37_NO_SWA_HMMA=1` restores only Step SWA.

`tests/test_exaone_kernels` compares production HMMA against the warp
path on 96×8, window 512, 64 tokens, `kv_cap=576`: rel RMS **1.463e-04**
(fp16 MMA vs f32 warp).

| Metric | Baseline samples | Candidate samples | Median change |
|---|---|---|---:|
| Prefill tok/s | 700.42 / 698.52 / 698.88 | 918.11 / 913.43 / 917.55 | +31.3% |
| Decode tok/s, MTP draft 3 | 20.98 / 21.40 / 21.32 | 17.72 / 17.60 / 17.59 | −17.4% |
| First decode call, seconds | 0.252 / 0.237 / 0.243 | 0.241 / 0.240 / 0.242 | flat |

Profiled Prefill wall 2.94s → 2.20s. Attention moved to
`ds4_fattn_hmma_gqa2_kernel` 0.074s / 192 launches. New Prefill #1 is
Q4 `ds4_moe_worklist_mmq_kernel` 0.678s / 232.

MTP-off (`DS4_MTP_SPEC_DISABLE=1`, one fresh process each): Prefill
700.95 → 920.01 tok/s; Decode 19.49 → 19.70 tok/s; first-token 0.1581 →
0.1596 s.

## Prefill 2: default chunk 512 → 1024

After Prefill 1, Q4 `ds4_moe_worklist_mmq_kernel` is 0.678s / 232 launches.
At 512 tokens the worklist sees ~21 rows per expert (8-of-288). 1024 doubles
those assignments and halves the 2K chunk count (4 → 2). Scratch at the
campaign context grows 291 → 582 MiB; sliding KV adds one window of extra
rows. `DS4_STEP37_PREFILL_CHUNK=512` restores the previous cap.

Same-binary MTP-off, three fresh processes at 1024 versus the Prefill 1
ordinary cell:

| Metric | Chunk 512 | Chunk 1024 samples | Median change |
|---|---|---|---:|
| Prefill tok/s | 920.01 | 1079.15 / 1063.70 / 1066.37 | +15.9% |
| Decode tok/s | 19.70 | 19.50 / 19.83 / 19.62 | flat |
| First decode call, seconds | 0.1596 | 0.1574 / 0.1537 / 0.1560 | flat |

MTP-on (draft 3) Prefill 1062.15 / 1064.24 / 1063.70 tok/s. Decode stays
acceptance-limited (~17.0 tok/s). 1024 repeats are byte-identical. Versus
chunk 512: same frontier argmax, top-10 9/10, top-50 47/50, rel RMS 5.02%,
KL 6.48e-4, max \|Δlogit\| 0.90; 40/64 generated tokens match; first
difference at index 37. Mixed-quant chunk-width MoE order is the expected
source. A 2048-token probe reached 1240 Prefill tok/s and is not the default.

`tests/test_step37_forward` asserts the default cap is 1024 and that
`DS4_STEP37_PREFILL_CHUNK=512` restores 512.

## Prefill 3: default chunk 1024 → 2048

A 2K prompt is then one chunk instead of two. Worklist rows per expert
double again. Scratch at the campaign context grows 582 → 1163 MiB.
`DS4_STEP37_PREFILL_CHUNK=1024` restores the previous cap.

Same-binary MTP-off, three fresh processes:

| Metric | Chunk 1024 median | Chunk 2048 samples | Median change |
|---|---|---|---:|
| Prefill tok/s | 1066.37 | 1240.11 / 1232.43 / 1233.87 | +15.7% |
| Decode tok/s | 19.62 | 19.85 / 19.72 / 19.64 | flat |
| First decode call, seconds | 0.1560 | 0.1554 / 0.1479 / 0.1504 | flat |

2048 repeats are byte-identical. Versus 1024 at 2K: same frontier argmax,
top-10 9/10, top-50 47/50, rel RMS 6.01%, KL 2.84e-3. One MTP-on cell is
1244.71 / 21.60 tok/s.

16K + 64 ordinary: chunk 1024 is 1054.69 / 18.61 tok/s; chunk 2048 is
1269.26 / 18.28 tok/s. Prefill does not fall off versus 2K. The 16K
frontier swaps a 0.06-logit top-2 tie (ids 27353 / 201); top-10 9/10,
KL 1.83e-3.

`tests/test_step37_forward` asserts the default cap is 2048 and that
`DS4_STEP37_PREFILL_CHUNK=1024` restores 1024.

Prefill 3's locked MTP median at default 2048 is 1238.78 / 21.40 tok/s
(three fresh processes). That is the Decode 1 comparator.

## Decode 1: MTP verify through batched GQA

MTP verify is n=2..4, so it used `exaone_attn_prefill_kernel` (one block
per row×head). A single GQA-pair launch now covers every query row on
`grid.z`. The kernel already applies the window and KV-cap ring. n=1
decode and n≥64 HMMA prefill are unchanged.
`DS4_EXAONE_PREFILL_GQA=0` restores the warp walk.

`tests/test_exaone_kernels` compares n=4 Step SWA against the warp path:
rel RMS **0** (byte-identical). Full-model 2K frontiers and 64-token
streams match the Prefill 3 binary exactly.

| Metric | Prefill 3 samples | Decode 1 samples | Median change |
|---|---|---|---:|
| Prefill tok/s, MTP | 1239.90 / 1235.61 / 1238.78 | 1243.02 / 1252.36 / 1241.78 | flat |
| Decode tok/s, MTP draft 3 | 21.40 / 21.37 / 21.48 | 22.48 / 22.36 / 22.43 | **+4.8%** |
| Ordinary decode tok/s | 19.76 | 19.83 | flat |

A serial per-row decode loop was slower (20.47 tok/s) and was not kept.

## Rejected Decode probes

IQ2 aligned pair-vec already accepts n≤16, but production used it only
at n=1. Forcing n=2..8 onto that kernel was numerically close to SOA
(rel RMS 1.81e-4) and slower: MTP 21.33 / 21.27 / 21.25 tok/s versus
Decode 1's 22.43.

Q4 MTP verify is 4×8=32 assignments, just above
`DS4_ROUTED_VEC_MAX_ROWS` (20), so it uses the compact worklist. Moving
those rows onto pair-vec / mmvq (rel RMS 1.04e-4 versus worklist) was
slower: MTP 20.34 / 20.46 / 20.22 tok/s. The worklist stays.

## Remaining bottlenecks (after Decode 1)

Prefill: Q4 worklist MMQ, IQ2 D2R gate/up, generic `mul_mat_q`.
Decode, profiled after Decode 1: Q8 NC
`q8_0_aligned_dense_vec_nc_kernel` 0.682s / 7644, `mul_mat_vec_q`
0.453s / 4044, Q4 worklist 0.408s / 1566, n=1 Q8 vec 0.362s / 166.
Largest host gap in `ds4.decode` is 0.123s (85.6% GPU coverage).

## Campaign close

The owner ended the campaign after the retained rounds above. Final
locked cells on the 2048+64 protocol, default chunk 2048:

| Cell | Prefill tok/s | Decode tok/s |
|---|---:|---:|
| MTP draft 3 median | 1243.02 | 22.43 |
| Ordinary (`DS4_MTP_SPEC_DISABLE=1`) | 1242.95 | 19.83 |
| 16K+64 ordinary, chunk 2048 | 1269.26 | 18.28 |

16K Prefill does not fall off versus 2K. Those 16K numbers are
post-Prefill-3, MTP-off. Decode 1 does not change n=1 ordinary
attention. Scratch at the campaign context is 1163 MiB at chunk 2048.
`DS4_STEP37_PREFILL_CHUNK=512` is the pre-campaign compatibility
rollback; `1024` is only the intermediate stage.
