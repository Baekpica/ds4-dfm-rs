QA report: feature/qwen35-port — Bonsai (qwen35) CUDA kernels and CUDA graph

Unit under test: 8c887e4 "feat(cuda): port the Bonsai attention and gated
delta-net kernels", 349e93b "feat(cuda): run the Bonsai trunk on the CUDA
graph", and 7818a7a "fix(cuda): keep the Bonsai graph out of the CPU-only
build", on top of a02ecd2. 7818a7a answers the single defect this QA pass
found in the first round; the re-verification is below.
Branch feature/qwen35-port. Base origin/main. Head 7818a7a (tree clean, no
tracked modification during this pass).
Tester: independent QA session (rule 19). Date 2026-09-30.
Host: RTX 4070 SUPER (sm_89, CUDA 13.3), 28 cores, RLIMIT_MEMLOCK 8192 KiB.
Artifact: /data/models/Ternary-Bonsai-2-27B-PQ2_0.gguf (6.71 GiB, 851
tensors, blk.0..blk.63 = 64 blocks).
Oracle: /data/ds4 branch bonsai tip bbaf298, prebuilt ./ds4 (read only, not
modified).
Every CUDA model run below sets DS4_CUDA_COPY_MODEL=1 and waits for the
single-instance slot. Scratch work lives in misc/scratch/qa/ (gitignored); no
tracked file other than this report was touched.

VERDICT SUMMARY

Pass. The CUDA unit verifies end to end (kernel gate, both runbooks, the
recorded stream, bit-identical logits against the sibling, the nine entries'
guards, the engine gate, the per-layer bound), the CPU-only build regression
that failed the first round is fixed and re-verified three ways, the release
report's three falsified statements are corrected, and the chat-prompt near
tie is now explained by measurement rather than by assumption.

RE-VERIFICATION OF THE FIX (7818a7a) — the defect of the first round is closed

The fix wraps the graph section in #ifndef DS4_NO_GPU and adds refusing stubs
for the CPU build (ds4.c: qwen35_graph_open/forward/free, plus the small
struct). Re-run of all three reproductions:

1. The ds4_cpu.o rule's exact command:
     cc -O3 -ffast-math -g -march=native -Wall -Wextra -std=c99 \
        -D_GNU_SOURCE -fno-finite-math-only -DDS4_NO_GPU -c -o /dev/null ds4.c
   -> exit 0, 0 errors, 17 warnings (all the pre-existing unused-function
   set, same as the a02ecd2 baseline). Before the fix: exit 1, 70 errors.
   The same command with -o misc/scratch/qa/ds4_cpu_postfix.o -> exit 0.

2. make -B tests/test_qwen35_rows -> exit 0, 6.63 s; then
     make test-qwen35-rows
   -> exit 0, 0.86 s:
     qwen35 rows: 851 tensors matched the reference dequantizer (first 64 rows each)
   Before the fix this target died with
   "tests/../ds4.c:69351:10: error: implicit declaration of function
   'ds4_gpu_flush_commands'" and make Error 1.

3. tests/test_motif3_loader: make -B tests/test_motif3_loader -> exit 0, and
   the raw compile with the target's flags (this time including -lm -pthread,
   which my first-round command omitted, so its earlier exit 1 was my link
   line and not the tree) -> exit 0, 0 errors.

Full CPU link, end to end, without touching the tree's ds4-c: the ds4_cpu.o
rule compiled into scratch and linked with the existing CPU objects
(ds4_cli_cpu.o linenoise.o ds4_cpu_postfix.o ds4_ple.o ds4_distributed.o
-lm -pthread) -> exit 0, and the resulting binary runs:

    DS4_QWEN35_STEPS=1 misc/scratch/qa/ds4-c-cpu -m <artifact> --cpu \
      --first-token-test -p x
    token 21: 760 The
    ds4: prompt 21 token(s); next-token top-5: 760(20.5268)The ...
    ds4: diagnostic run completed on the native cpu path.
    exit 0, WALL 73.32 s
    misc/scratch/qa/ds4-c-cpu -m <artifact> --cuda -p x
    ds4: Bonsai (qwen35) runs only through --first-token-test, ...
    exit 1

So the CPU stubs resolve at link time and the CPU host works. make cpu itself
was deliberately not run: its recipe links ds4-c and would overwrite the
CUDA-linked binary in this shared tree. The compile, the link and the run
above cover the same ground per object.

CUDA SIDE UNTOUCHED BY THE FIX (re-measured on the current build)

    make ds4-c CUDA_ARCH=sm_89            -> exit 0 ("ds4-c is up to date";
    ds4.c 09:29:18, ds4.o 09:30:16, ds4-c 09:30:17; the guard is a no-op for
    the CUDA build, which compiles the same block either way)
    make test-qwen35-cuda CUDA_ARCH=sm_89 -> exit 0, 20.74 s,
      PQ2_0 CUDA parity: PASS
      gdn out norm, sigmoid (qwen4exp) gate: max_abs=3.57628e-07 failures=0/18432: PASS
      gdn out norm, silu (Bonsai) gate: max_abs=9.53674e-07 failures=0/18432: PASS
      eight attention cases, token-tile at 4.53e-08 / 4.12e-08 / 6.12e-08 / 3.67e-08
    make bonsai-cuda-parity              -> "bonsai cuda parity: PASS", exit 0,
      43.90 s, the two token files identical (the eight recorded ids)

Logits re-measured on this build (three fresh dumps, 5 x 248320 f32):

    cuda - cpu    : max|d|=0.08093 rms_rel=0.264% argmax 5/5 bit_identical=False
    sibling - cpu : max|d|=0.08093 rms_rel=0.264% argmax 5/5 bit_identical=False
    cuda - sibling: max|d|=0.00000 rms_rel=0.000% argmax 5/5 bit_identical=True
    new cuda dump - first-round cuda dump: bit_identical=True
    new cpu  dump - first-round cpu  dump: bit_identical=True

The entry-guard probe was rebuilt against the post-fix hooks object and
re-run: 101 checks, 90 refusals (got=0), 11 valid runs (got=1), 0 FAIL.

CAST OF THE FIRST ROUND'S DEFECT, KEPT FOR THE RECORD

349e93b added the graph block unguarded, so the -DDS4_NO_GPU compilation of
ds4.c failed with 70 errors on 67 distinct lines (first ds4.c:68987 unknown
type name ds4_gpu_tensor, last ds4.c:69351 implicit declaration of
ds4_gpu_flush_commands), which broke make cpu and every target that compiles
ds4.c with that switch (test-qwen35-rows, test-motif3-loader,
test-motif3-reference, test-exaone-tokenizer, tokenizer_c_oracle). The
regression boundary was verified then: a02ecd2 compiled clean with the same
command. Fixed in 7818a7a and re-verified above.

KERNEL GATE AND RUNBOOKS (first round, re-confirmed above)

    make test-qwen35-cuda CUDA_ARCH=sm_89 -> PQ2_0 CUDA parity: PASS, with the
    restored gdn-output-gate group (sigmoid 3.57628e-07, silu 9.53674e-07) and
    the eight attention split cases including the four token-tile ones.
    make bonsai-cuda-check  -> exit 0: the greedy stream on "The capital of
    France is", tokens 25..36.
    make bonsai-cuda-parity -> PASS (above).
    make bonsai-fold-selftest, make bonsai-ref-check, make pq2-0-test,
    make test-qwen35-rows, make test-qwen35-cuda: all exit 0.

REFERENCE STREAM AND LOGITS

With DS4_QWEN35_TOKENS=760,6511,314,9338,369 DS4_QWEN35_STEPS=8 the CUDA
graph prints exactly token 5..12 = 11751, 13, 198, 760, 6511, 314, 9564, 369,
and the 64-step run reproduces the recorded stream for its first 16 positions
(adds 19241 Berlin). The logits comparison is in the section above.

NINE ENTRIES AND THEIR GUARDS

Scratch probe misc/scratch/qa/entry_guards.cu (built by
misc/scratch/qa/build-guards.sh with the flags and objects of
tests/test_qwen35_cuda), log misc/scratch/qa/52-entry-guards-postfix.log:

    101 checks, 90 refusals, 11 valid runs, 0 FAIL
      ds4_gpu_qwen4_conv_stream_tensor          10 checks / 1 valid
      ds4_gpu_qwen4_gdn_prep_tensor             12 / 1
      ds4_gpu_qwen4_gdn_scan_tensor             11 / 1
      ds4_gpu_qwen4_gdn_out_tensor (sigmoid)     9 / 1
      ds4_gpu_qwen35_gdn_out_tensor (silu)       9 / 1
      ds4_gpu_qwen4_attn_decode_tensor          17 / 3
      ds4_gpu_qwen4_attn_tokentile_available     8 / 1
      ds4_gpu_qwen35_attn_prep_tensor           16 / 1
      ds4_gpu_qwen35_matvec_bf16_tensor          9 / 1

Each malformed case isolates one guard while the rest of the request is
valid, so a dead guard would show as a 1 where a 0 is required: T=0, K out of
2..4, D<32 / D>128 / D%32, Hv%Hk, H%Hkv, D not in 32/128/256, nrot>64 and odd
nrot, pos0+T>cap, every undersized tensor, a weight offset at or past the map
end, a weight range that overflows it, a NULL map, and the missing or
undersized sparse / split-K tensors.

ds4_gpu_qwen35_matvec_bf16_tensor numerics: kernel against the float-activation
reference max|d| = 1.297e-06, against the bf16-rounded-activation reference
3.023e-02, so the activation is not rounded to bf16 (the property the entry
exists for).

Reachability: six of the nine are called from the ds4.c graph (69046 matvec,
69108 conv, 69113 gdn_prep, 69118 gdn_scan, 69122 gdn_out, 69145 attn_prep,
69155 attn_decode). ds4_gpu_qwen4_attn_tokentile_available (ds4_gpu.h:1058,
defined in cuda/qwen35_attn_gdn.cuh) has no caller in this tree: the same
decision lives in qwen35_attn_tokentile_ok inside the header, which the decode
dispatcher uses, and the sibling's graph calls the entry (its ds4.c:69037).
Reachable through the ABI and exercised by the probe; a note, not a defect.

Probe artefact worth recording: with the first probe version, which freed its
mmap'd scratch maps and let the next mmap reuse the address, the matvec valid
case returned 0 with "CUDA model range copy failed for Bonsai weights ...
invalid argument (serving mapped)". The resolver caches host ranges by address
(comment at ds4_cuda.cu:1745). Keeping the maps alive for the process lifetime
makes every case pass; the model run resolves from the real artifact mapping.

ENGINE GATE

    ./ds4-c -m <artifact> --cuda -p x    exit 1
    ./ds4-c -m <artifact> --cpu  -p x    exit 1
    ./ds4-c -m <artifact>        -p x    exit 1
    all three: "ds4: Bonsai (qwen35) runs only through --first-token-test, on
    the CPU reference (--cpu) or the CUDA graph (--cuda); the session and
    server paths are not implemented yet"
    ./ds4-c -m <artifact> --cpu --first-token-test -p x    exit 0
    and the CPU-only scratch binary refuses --cuda the same way.

PER-LAYER ARRAY BOUND

DS4_MAX_LAYER = 61 (ds4.c:317, the DeepSeek bound); DS4_QWEN35_MAX_LAYER 64u
with the aliasing comment (68982); the four arrays sized with it (68990-68993);
qwen35_graph_open refuses a model with more blocks than the arrays hold
(69233-69236, "Bonsai CUDA graph: %u blocks exceed the %u-slot state
arrays"), and the string is in the built ds4-c. The artifact really has 64
blocks (independent GGUF parse, blk.0..blk.63). The live adequacy proof is the
bit-identical CUDA logits against the sibling, which aliased state would
break. The refusal path itself cannot be provoked here (no >64-block
artifact).

CHAT-PROMPT NEAR TIE — SETTLED BY MEASUREMENT

All runs on the current binary, one model process at a time, artifact
--first-token-test with the CLI's 25-token chat template. Exact command shape:

    env DS4_CUDA_COPY_MODEL=1 DS4_QWEN35_STEPS=<n> ./ds4-c -m <artifact> \
        --cuda --first-token-test -p "The capital of Germany is"
    env DS4_QWEN35_STEPS=<n> ./ds4-c -m <artifact> \
        --cpu  --first-token-test -p "The capital of Germany is"

    1) Germany, cuda, 16 steps:
       760(22.1969)The 90700(16.3264)Thinking 77264(13.6300)Hmm
       1421(13.5868)User 15893(12.9160)Simple     token 29: 883  about
       (WALL 2.34 s)
    2) Germany, cuda, 16 steps, repeat:
       byte-identical top-5 and stream              token 29: 883  about
       (WALL 2.31 s)
    3) Germany, cuda, 32 steps:
       byte-identical top-5 and stream              token 29: 883  about
       (WALL 2.78 s)
    4) Germany, cpu, 16 steps:
       760(22.1780)The 90700(16.3378)Thinking 77264(13.6195)Hmm
       1421(13.5884)User 15893(12.9233)Simple       token 29: 264  a
       (WALL 141.37 s)
    5) Germany, cpu, 32 steps:
       byte-identical top-5 and stream              token 29: 264  a
       (WALL 195.15 s)
    control) France, cuda, 16 steps:
       760(21.9915)The 90700(16.0678)Thinking 1421(13.8667)User
       15893(13.5789)Simple 77264(13.4640)Hmm        token 29: 264  a
       (WALL 2.37 s)

What this settles:

- The CUDA stream is deterministic in this environment: two 16-step repeats
  are byte-identical, and the 16- and 32-step runs agree on the whole stream.
- On the Germany prompt the CUDA values are the author's to the digit
  (22.1969 / 16.3264 / 13.6300 / 13.5868 / 12.9160 and 883 "about"), and the
  CPU values are the author's to the digit (22.1780 / 16.3378 / 13.6195 /
  13.5884 / 12.9233 and 264 "a"). So the author's CPU-versus-CUDA divergence
  on that prompt reproduces here; it is not an environment difference.
- My earlier "the backends agree" observation was made on the France prompt
  (The capital of France is), where this environment gives CUDA 760(21.9915)
  with 264 "a" and CPU 760(21.9951) with 264 "a" (32/32 identical). The
  control run above repeats the France prompt on the same binary and gets the
  same 264 "a", so the difference between the two observations is the prompt
  (last token 9338 France against 9564 Germany), not nondeterminism on either
  backend.
- The step count does not change the first-token top-5 on either backend
  (16 and 32 identical), so there is no step-count-dependent top-5 defect to
  name.
- The doc's "the deciding gap is about 0.1 on a 22-magnitude top logit" could
  not be checked: the diagnostic prints the top-5 of the prompt position only,
  not per-step candidates, so the 883-against-264 gap at token 29 is not
  observable from these runs. What is measurable on the same prompt is the
  top-1 difference between the backends: 22.1969 - 22.1780 = 0.0189.

RELEASE REPORT CHECK (docs/releases/bonsai-cuda-graph-2026-09-30.md)

The first round falsified three statements; all three are corrected in
7818a7a:

- Section 4 no longer claims the flip as expected behaviour. It now says the
  backends agree for the first 24 tokens and then differ in the author's runs
  while the QA's run agreed 32/32, that the CPU is stable across the author's
  three runs, and concludes that the long chat continuation is a near-tie coin
  flip at the artifact's decode precision, with the explicit-prompt stream as
  the gate. Addressed. Two nits remain in the same section: its parenthetical
  compares 21.9951 (the QA's France-prompt CPU top-1) with 22.1780 (the
  author's Germany-prompt CPU top-1), which mixes prompts — the like-for-like
  numbers are in the section above, and same-prompt the two environments agree
  on each backend; and the "about 0.1" candidate gap is not measurable from
  the diagnostic (see above).
- Section 5 now says the sibling tree's constant is 79 (which is what
  /data/ds4/ds4.c:496 defines) instead of 64, keeping the same conclusion.
  Addressed.
- Section 0 now marks the failing tensor as one run's observation and notes
  that it varies with allocation order. Addressed (my run died at
  blk.35.attn_q.weight, the author's example was blk.32.ffn_gate.weight; the
  symptom, "CUDA host registration skipped: invalid argument" then a matmul
  failure part-way through the trunk, is the same).
- The report also records the CPU-build guard as the third commit. The rest of
  the document re-checks against this pass: section 1 kernel gates, section 2
  parity, section 3 logits, section 6 rates (1.66/3.10 s and 0.703 s copy
  measured here against 1.64/3.09 s and 26 ms per token claimed) all hold.

SURFACES COVERED (the gate's list for this branch)

Live in this pass: ds4_gpu_qwen4_conv_stream_tensor,
ds4_gpu_qwen4_gdn_prep_tensor, ds4_gpu_qwen4_gdn_scan_tensor,
ds4_gpu_qwen4_gdn_out_tensor, ds4_gpu_qwen4_attn_decode_tensor,
ds4_gpu_qwen4_attn_tokentile_available, ds4_gpu_qwen35_gdn_out_tensor,
ds4_gpu_qwen35_attn_prep_tensor, ds4_gpu_qwen35_matvec_bf16_tensor (probe, 101
checks); ds4_gpu_matmul_pq2_0_tensor, ds4_gpu_qwen35_fold_forward_tensor,
ds4_gpu_qwen35_fold_inverse_tensor (make test-qwen35-cuda); DS4_QWEN35_TOKENS,
DS4_QWEN35_STEPS, DS4_QWEN35_LOGITS (stream, logits and chat-prompt runs);
DS4_QWEN35_FOLD_SELFTEST (make bonsai-fold-selftest); bonsai-cuda-check,
bonsai-cuda-parity, bonsai-fold-selftest, bonsai-ref-check, pq2-0-test,
test-qwen35-cuda, test-qwen35-rows (all exit 0; test-qwen35-rows was the
target the first round found broken and now builds and runs).

Not re-run in this pass, present at HEAD, outside this unit's diff (they come
from the earlier catalogue commit and none of their sources is touched by the
three commits): in crates/ds4-core/src/qwen35.rs and
crates/ds4-core/src/shape.rs:
  pub const CTX_MAX, pub const FOLDABLE_SUFFIXES, pub const FULL_ATTN_INTERVAL,
  pub const LIN_CONV, pub const LIN_HEAD_DIM, pub const LIN_K_DIM,
  pub const LIN_K_HEAD, pub const LIN_V_DIM, pub const LIN_V_HEAD,
  pub const SHAPE_QWEN35, pub fn is_foldable_weight_name,
  pub fn layer_is_full_attention.
Their C-against-Rust parity harness (make test-catalog-parity) was not
executed here.

NOT RUN, AND WHY

- make cpu itself: its recipe links ds4-c and would overwrite the CUDA-linked
  binary in this shared tree. The ds4_cpu.o compile, the full CPU link into
  scratch and a CPU-only run stand in for it.
- No forced full rebuild of ds4-c: the binary already postdates ds4.c and the
  guard does not change the CUDA build's code; freshness was established from
  timestamps plus make -q.
- The >64-block refusal path cannot be provoked (no artifact with more than
  64 blocks here).
- Prefill throughput is not measured by this diagnostic (the release report
  says so too); only the decode shape was timed.
- The token-tile attention kernel is not reached by the trunk (T stays under
  its 32-token gate), so it is verified through the kernel gate only.
- The 883-against-264 candidate gap at token 29 is not observable from the
  diagnostic's output (prompt-position top-5 only).
- The Rust catalogue items listed above were not re-executed.

verdict: overall PASS
