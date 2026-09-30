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

================================================================================
QA pass 2 — Bonsai session unit: 4a2a412 "feat(core): run Bonsai sessions on
both backends" (docs/releases/bonsai-session-2026-09-30.md)

Unit: 4a2a412 on feature/qwen35-port-cpu, HEAD at pass time 4a2a412 (the
working tree carried only this report and the gitignored misc/scratch/qa logs;
no tracked source was modified). Tester: independent rule 19 QA session,
2026-09-30. Host RTX 4070 SUPER (sm_89, CUDA 13.3).
Artifact /data/models/Ternary-Bonsai-2-27B-PQ2_0.gguf. Every CUDA run sets
DS4_CUDA_COPY_MODEL=1. Runs were serialised on /tmp/ds4.lock (pgrep -a -x
ds4-c was empty before each heavy run; no process not started here was killed).

Build — the release report's trap is real: the tree held a CPU-only ds4-c newer
than ds4.o, so the documented CUDA rebuild is a no-op unless the binary is
removed first.

    rm -f ds4-c && make ds4-c CUDA_ARCH=sm_89        -> exit 0
    ldd ds4-c | grep libcudart
      libcudart.so.13 => /usr/local/cuda-13.3/lib64/libcudart.so.13

The four surfaces the gate requires for this unit, each exercised:

1. DS4_QWEN35_SESSION

   Baseline (no session), CUDA:
     DS4_CUDA_COPY_MODEL=1 DS4_QWEN35_TOKENS=760,6511,314,9338,369 \
     DS4_QWEN35_STEPS=8 ./ds4-c -m <model> --cuda --first-token-test -p x
     prompt: 760 6511 314 9338 369
     token 5: 11751 Paris / 6: 13 . / 7: 198 / 8: 760 The /
     9: 6511 capital / 10: 314 of / 11: 9564 Germany / 12: 369 is   exit 0

   With DS4_QWEN35_SESSION=1, same env and args:
     ds4: Bonsai session path (ds4_session create, sync, eval)
     ds4: Bonsai prefill chunk: 14 tokens (ctx 14)
     the identical eight ids                                        exit 0

   With DS4_QWEN35_SESSION=1 on the CPU backend, same env and args:
     ds4: Bonsai session path (ds4_session create, sync, eval)
     the identical eight ids                                        exit 0

   PASS — the diagnostic drives a real ds4_session (create, sync, eval) on both
   backends and prints the same stream. The CPU session is also exercised by
   the make target below and by run-bonsai.sh session.

2. DS4_QWEN35_PREFILL_CHUNK

   Same session command with the variable set, ids diffed against the baseline:

     DS4_QWEN35_PREFILL_CHUNK=2    -> "Bonsai prefill chunk: 2 tokens (ctx 14)"
                                      ids identical to the baseline
     DS4_QWEN35_PREFILL_CHUNK=0    -> "Bonsai prefill chunk: 14 tokens (ctx 14)"
                                      ids identical (0 falls back to the default)
     DS4_QWEN35_PREFILL_CHUNK=2048 -> "Bonsai prefill chunk: 14 tokens (ctx 14)"
                                      ids identical (>1024 falls back)
   Default (unset) on a 32768 context, plain generation -n 4:
     "Bonsai prefill chunk: 512 tokens (ctx 32768)" — the documented 512 default.

   PASS — the log line follows the variable and an out-of-range value falls back
   to the default instead of being accepted silently. The multichunk make target
   below crosses many chunk boundaries through the same path.

3. test-qwen35-session  (make target, both backends)

     make test-qwen35-session CUDA_ARCH=sm_89        -> exit 0
       CUDA: 20 PASS, 0 FAIL, "qwen35 session path: PASS"
         scenarios plain / prefix reuse / rewind replay / invalidate /
         context bound on the 5-token prompt, and the same four (no context
         bound) on the 68-token long prompt; every id compared against the
         in-process CPU reference (the oracle hook ds4_test_qwen35_ref_greedy,
         ds4.c:68867).
       CPU (DS4_TEST_BACKEND=cpu, 4 steps, long pass skipped): 10 PASS, 0 FAIL,
         "qwen35 session path: PASS"

   PASS on both backends.

4. test-qwen35-session-multichunk  (make target, DS4_QWEN35_PREFILL_CHUNK=2)

     make test-qwen35-session-multichunk CUDA_ARCH=sm_89  -> exit 0
       every scenario logs "Bonsai prefill chunk: 2 tokens", 20 PASS, 0 FAIL,
       "qwen35 session path: PASS"; the prefill crosses many chunk boundaries
       and the ids still match the reference.

   PASS.

    ./run-bonsai.sh session                            -> exit 0
      "IDENTICAL: the CPU session reproduces the CPU reference" and
      "IDENTICAL: all 16 generated token ids agree".

Engine gate — accepts a session, keeps the refusals:

    ./ds4-c -m <model> --cuda --first-token-test -p x --mtp /tmp/qa-mtp-sidecar.gguf
      -> exit 1: "ds4: Bonsai (qwen35) runs on the CPU reference (--cpu) or on
         the CUDA graph (--cuda); tensor parallelism, distributed ranks, SSD
         streaming, MTP/DSpark and steering are not supported"
    the same with --cpu                                -> exit 1, same message
    DS4_QWEN35_SESSION=1 DS4_QWEN35_LOGITS=/tmp/qa-logits.bin ... -> exit 1:
      "ds4: DS4_QWEN35_LOGITS needs the per-position logits and is not
       available with DS4_QWEN35_SESSION"; /tmp/qa-logits.bin was NOT created
    ./run-bonsai.sh server                             -> exit 2:
      "ERROR: \"server\" is not implemented in this tree."
    plain generation ./ds4-c -m <model> --cuda -p "The capital of France is" -n 4
      -> exit 0, opened a session ("Bonsai prefill chunk: 512 tokens
         (ctx 32768)") and printed "The user is asking", so the gate admits the
         family outside --first-token-test too.

Snapshot refusal — source-verified, not live: ds4_session_payload_bytes returns
0 for the family (ds4.c:54790-54793); ds4_session_save_payload refuses "Bonsai
session snapshots are not implemented yet" (ds4.c:55024-55028) and
ds4_session_load_payload the same (ds4.c:55485-55493). No CLI flag exposes
payload save/load in this tree, so this one claim rests on the source read, not
on a command; the context bound is covered live by scenario 5 of the test.

Regression gates (all exit 0):

    make test-qwen35-cuda CUDA_ARCH=sm_89           "PQ2_0 CUDA parity: PASS"
    make bonsai-fold-selftest                       fold selftest line printed
    make bonsai-ref-check (DS4_BONSAI_STEPS=12)      12 token lines, exit 0
    make bonsai-cuda-parity                         "bonsai cuda parity: PASS"
    ./run-bonsai.sh ids                             IDENTICAL (the 8 ids)
    DS4_BONSAI_MODEL=<model> make test-qwen35-rows  "851 tensors matched"
    bash tests/run.sh                               green (see below)

    Note: `DS4_QWEN35_STEPS=4 make bonsai-ref-check` does not pass 4 through —
    the recipe assigns DS4_QWEN35_STEPS=$(DS4_BONSAI_STEPS)=12, so the
    environment value is superseded and the target ran 12 steps. The target is
    green either way; the anomaly is in the invocation, not the gate.

Release report check: docs/releases/bonsai-session-2026-09-30.md matches this
pass — the 20/10 PASS counts, the "Bonsai prefill chunk" log, the session diff
IDENTICAL, the CPU-build trap, and the stated limits (decode is one row per
eval; no snapshots or disk KV; the CPU reference is ~3 s per forward). No
falsified statement was found.

Surfaces this pass adds to the gate's coverage: DS4_QWEN35_SESSION,
DS4_QWEN35_PREFILL_CHUNK, test-qwen35-session, test-qwen35-session-multichunk
(all named and exercised above); the rest of the gate's list is covered by the
earlier section.

verdict: overall PASS

================================================================================
QA pass 3 — Rust-host load path: 78c086f "fix(core): apply the fold and the
rope on the host path"

Unit: 78c086f on feature/qwen35-port (recorded as feature/qwen35-port-cpu in
the handoff), HEAD at pass time 78c086f. Working tree carried only this report
and gitignored misc/scratch; no tracked source was modified. Tester:
independent rule 19 QA session, 2026-09-30. Host RTX 4070 SUPER (sm_89, CUDA
13.3). Artifact /data/models/Ternary-Bonsai-2-27B-PQ2_0.gguf. Every CUDA run
sets DS4_CUDA_COPY_MODEL=1. Runs were serialised on the single model slot;
pgrep -a -x ds4 / ds4-c was checked before each heavy run and nothing not
started here was killed.

BUILD

    rm -f ds4-c && make ds4-c CUDA_ARCH=sm_89   -> exit 0
    ldd ds4-c | grep libcudart
      libcudart.so.13 => /usr/local/cuda-13.3/lib64/libcudart.so.13 (CUDA-linked)
    make ds4 CUDA_ARCH=sm_89                    -> "ds4 is up to date"
      (ds4 14:28:40 and the rebuilt ds4-c both newer than ds4.c 14:27:49 and
      Makefile 14:28:14, so the Rust host already carried the fix)

FIRST ATTEMPT AND THE MODEL SLOT

The first make test-qwen35-rust-host run reported both legs FAIL with an empty
"got" because a foreign ./ds4 -m <model> --backend cuda -p "The capital of
France is" -n 12 --temp 0 (PID 1123973, not started here) held the model slot.
I waited 30 s for it to exit and re-ran; nothing was killed.

THE NEW SURFACE: test-qwen35-rust-host (make target, both backends)

    DS4_BONSAI_MODEL=/data/models/Ternary-Bonsai-2-27B-PQ2_0.gguf \
      make test-qwen35-rust-host
      qwen35 rust host parity (cuda): PASS
      qwen35 rust host parity (cpu): PASS
      exit 0

The target runs ./ds4 (the Rust host, the default binary) with
--token-ids 760,6511,314,9338,369 --predict 8 --temp 0 on --backend cuda (with
DS4_CUDA_COPY_MODEL=1) and --backend cpu and pins the output to the eight ids
the C host prints. PASS on both backends is the surface verified live.

HALF 1 — THE RUST HOST NOW AGREES WITH THE C HOST

Rust host (cuda):

    DS4_CUDA_COPY_MODEL=1 ./ds4 -m <model> --backend cuda \
      --token-ids 760,6511,314,9338,369 --predict 8 --temp 0
    stdout: 11751 13 198 760 6511 314 9564 369 19241        exit 0

C host (cuda, the oracle):

    DS4_QWEN35_TOKENS=760,6511,314,9338,369 DS4_QWEN35_STEPS=8 \
      DS4_CUDA_COPY_MODEL=1 ./ds4-c -m <model> --cuda --first-token-test -p x
    prompt: 760 6511 314 9338 369
    token 5: 11751  Paris
    token 6: 13 .
    token 7: 198
    token 8: 760 The
    token 9: 6511  capital
    token 10: 314  of
    token 11: 9564  Germany
    token 12: 369  is                                       exit 0

The Rust host's leading ids 11751 13 198 760 6511 314 9564 369 are byte-for-
byte the C host's tokens 5..12. They agree.

HALF 2 — THE FIX'S PREMISE, THE HOST PATH, BOTH DIRECTIONS

Positive (post-fix, this binary). The Rust host's stderr carries the fold line
on the ids run:

    DS4_CUDA_COPY_MODEL=1 ./ds4 -m <model> --backend cuda \
      --token-ids 760,6511,314,9338,369 --predict 8 --temp 0 2>err
    err: ds4: prism.hadamard folding: block 1024, 3 sign vector(s), gdn_v_grouped 1
         ds4: Bonsai prefill chunk: 512 tokens (ctx 32768)

and the chat prompt answers coherent English:

    DS4_CUDA_COPY_MODEL=1 ./ds4 -m <model> --backend cuda \
      -p "The capital of France is" -n 12 --temp 0
    stdout: The user is asking a simple factual question: "The capital   exit 0
    (the same run's stderr carries the hadamard folding line)

Negative (live, cheap). I built the parent commit 8d172e7 in a scratch
worktree (misc/scratch/qa/prefix-wt, gitignored: git worktree add --detach,
then make ds4 CUDA_ARCH=sm_89, exit 0) and ran the same two commands on it:

    DS4_CUDA_COPY_MODEL=1 prefix-wt/ds4 -m <model> --backend cuda \
      -p "The capital of France is" -n 12 --temp 0
    stdout: etalorry途bewendanấnavourума!/学前estinahaf          exit 0
    stderr hadamard-folding lines: 0 (absent)

    DS4_CUDA_COPY_MODEL=1 prefix-wt/ds4 -m <model> --backend cuda \
      --token-ids 760,6511,314,9338,369 --predict 8 --temp 0
    stdout: 55783 211805 72082 96641 80597 97146 101425 132488 174713

Both reproduce the commit body's recorded pre-fix output exactly (the nonsense
string and the "55783 211805 72082 ..." stream), and the pre-fix binary never
prints the fold line. So the divergence was the load path rather than the
tokenizer, the kernels or the session path, and the fix closes it. The
worktree was removed with git worktree remove --force after the run; the main
tree's source was not touched.

REGRESSION GATES

    cargo test -p ds4-core        -> exit 0; 5 passed (lib) + 4 passed
      (validate.rs), 0 failed
    make bonsai-cuda-parity       -> "bonsai cuda parity: PASS", exit 0
    ./run-bonsai.sh ids           -> "IDENTICAL: both backends print the same
      8 ids", exit 0 (cuda 1.65 s, cpu 44.01 s)
    make test-qwen35-session CUDA_ARCH=sm_89 -> 30 PASS, 0 FAIL; both legs
      print "qwen35 session path: PASS" (CUDA then CPU)
    bash tests/run.sh             -> green on the re-run below; its only earlier
      red was this gate's own "report covers surface: test-qwen35-rust-host"
      check, which this pass fixes by naming the surface here

SURFACE THIS PASS ADDS

test-qwen35-rust-host (named and exercised live on both backends above). Every
other surface in the gate's list stays covered by the earlier sections.

verdict: overall PASS

QA report: feature/qwen35-port — Bonsai (qwen35) served on ds4-server

Unit under test: a7fbee4 "feat(server): serve Bonsai (qwen35) on ds4-server".
Branch feature/qwen35-port. HEAD at pass time:
a7fbee442e089f6b835ab523cfbdc91daa5391a9 (tree clean apart from this report
and pre-existing untracked test binaries; no tracked source was modified by
this pass).
Tester: independent QA session (rule 19), model deepseek-v4.1-CC-flash.
Date 2026-09-30. Host: RTX 4070 SUPER (sm_89, CUDA 13.3.73), 28 cores,
Debian 6.12.107, RLIMIT_MEMLOCK 8192 KiB.
Artifact: /data/models/Ternary-Bonsai-2-27B-PQ2_0.gguf (6.71 GiB).
Every CUDA run set DS4_CUDA_COPY_MODEL=1 (run-bonsai.sh sets it; the
hand-started servers were started with it explicitly), and every model run
was serialized on the single-instance flock /tmp/ds4.lock (pgrep -x
ds4-server/ds4/ds4-c checked before each heavy step; the slot was free each
time). Captures: misc/scratch/qa39/ (gitignored). The unit's own servers were
stopped with pgrep -x ds4-server | xargs -r kill; no foreign process was
touched.

CLAIM 1 — the CUDA server answers "Paris." (verified)

    ./run-bonsai.sh server
    server:  up; listening on 127.0.0.1:8899
             model_id=Ternary-Bonsai-2-27B-PQ2_0 engine=open
    wall:    1.30s for the request
    finish:  stop
    answer:  Paris.
    reasoning: The user is asking a simple factual question: "The capital of
    France is". I just need to complete the sentence. The capital of France
    is Paris.
    usage:   prompt_tokens 45, completion_tokens 36, total_tokens 81
Exit 0. Answer "Paris.", non-empty reasoning block, finish_reason "stop" —
all three as claimed. Log: misc/scratch/qa39/claim1-cuda-server.log.

CLAIM 2 — the CPU reference prints the same answer (verified)

    DS4_BONSAI_BACKEND=cpu DS4_BONSAI_CTX=8192 DS4_BONSAI_PORT=8898 \
      ./run-bonsai.sh server
    wall:    282.04s for the request
    finish:  stop
    answer:  Paris.
    reasoning: the same sentence as the CUDA leg
    usage:   prompt_tokens 45, completion_tokens 36, total_tokens 81
Exit 0, ~283 s as stated. One discrepancy in the given command, not in the
unit: run-bonsai.sh:81 reads DS4_BONSAI_SERVER_PORT (docs/BONSAI.md:166
documents the same name), so the given DS4_BONSAI_PORT=8898 was ignored and
the server listened on the default 8899. The claim it tests (same answer on
the CPU reference) holds. Log: misc/scratch/qa39/claim2-cpu-server.log.

CLAIM 3 — model ids (verified)

    GET /v1/models -> 200, data ids ['Ternary-Bonsai-2-27B-PQ2_0']
    GET /v1/models/prism-bonsai-2-27b -> 200,
      {"id":"prism-bonsai-2-27b","object":"model",
       "name":"Ternary-Bonsai-2-27B-PQ2_0","context_length":32768,...}
    GET /v1/models/prism/bonsai-2-27b -> 200, id "prism/bonsai-2-27b"
    GET /v1/models/no-such-alias -> 404 {"error":{"message":"unknown endpoint"}}
The advertised id is the GGUF stem; the sibling names resolve as model
lookups (alias list crates/ds4-server/src/models.rs:32-37).

CLAIM 4 — the memory quote is honest (verified, exact arithmetic)

    ./ds4-server -m <artifact> --backend cuda -c 32768 --mem-floor-gb 1 \
      --check-config
    exit 0; quote: per_bank 2570354688, shared_weights 7206168928,
    floor 1073741824, available 12158238720, banks 1, total 10850265440;
    only warn partial_unqualified.
    same with --mem-floor-gb 4 -> exit 2, error quote_overflow
      ("memory quote cannot host the mix at one bank"), per_bank unchanged.
    same with -c 65536 --mem-floor-gb 1 -> exit 2, error quote_overflow,
      per_bank 4718362624.

Own derivation from ds4.c (qwen35_graph_open, ds4.c:69341-69430; shape at
ds4.c:962; layer predicate ds4_qwen35_layer_is_linear, (il+1)%4, = 16
gated-attention + 48 delta-net layers), ctx 32768:
    16 x 32768 x 2 x 4 x 256 x 2 = 2147483648   (fp16 k+v caches)
    48 x 6144 x 128 x 4          =  150994944   (delta-net state)
    48 x 3 x 10240 x 4           =    5898240   (conv window)
    chunk transient (cap 512)    =  264984576   (17 f32 scratch rows
        512x129120x4, tokens 512x4, h_row 5120x4, pos3 32768x16)
    f32 logits row 248320 x 4    =     993280
    sum                          = 2570354688   (2.3938 GiB) — exactly the
    reported per_bank; ctx 65536 gives 4718362624, also exact. The fence in
    crates/ds4-core/src/serving_host.rs:2066 (qwen35_bank_bytes, transient at
    :2115) mirrors the native allocations line for line. Live honesty check:
    with that accepted plan serving, nvidia-smi memory.used peaked at
    9988 MiB (samples 9936, 9988) against a quoted total of 10850265440
    bytes = 10.10 GiB on a 12282 MiB card — the plan fits and runs, and the
    quote is not optimistic.

CLAIM 5 — --kv-disk-dir is refused by name (verified)

    ./ds4-server -m <artifact> --backend cuda -c 32768 --mem-floor-gb 1 \
      --kv-disk-dir misc/scratch/qa39/kvdisk --check-config
    exit 2; issues include {"level":"error","code":"disk_unsupported",
    "message":"qwen35 session snapshots are unsupported"}; stderr prints
    "error: qwen35 session snapshots are unsupported (disk_unsupported)";
    effective.disk=false, so the directory is dropped rather than silently
    used (resolve_disk, crates/ds4-core/src/serving.rs:2049-2056).

CLAIM 6 — the two crash fixes (verified live; historical segfault not rebuilt)

Static: diffing the QWEN35 lines of 78c086f:ds4.c with HEAD ds4.c adds
exactly two arms: ds4.c:71751 (ds4_engine_supports_batching -> false) and
ds4.c:72589-72595 (ds4_engine_session_graph_fit_quote -> budget-less
fail_open inside the family bound). At 78c086f the batching function falls
through to the DeepSeek slab body and the quote into the generic estimate;
render.rs maps engine id 13 to ModelSyntax::DeepSeek at 78c086f and to
ModelSyntax::Qwen35 at HEAD (crates/ds4-server/src/render.rs:106-107).

Live: a hand-started server (DS4_CUDA_COPY_MODEL=1, ctx 32768, floor 1,
setsid) served four requests in a row, pid alive after each:
    req1 non-stream "The capital of France is" -> 200, stop, "Paris." (1.24s)
    req2 non-stream "Reply with the single word: hello", model id
         prism-bonsai-2-27b -> 200, stop, "hello"
    req3 stream:true -> 200, 37 SSE chunks, [DONE] seen, finish stop,
         reassembled content "Paris.", reasoning 144 chars
    req4 non-stream "Say OK", max_tokens 16 -> 200, finish "length", empty
         content (the cap was consumed inside the reasoning block)
The log had no segfault/panic/abort marker; the process died only to the
tester's kill. Not re-run: the 78c086f segfault itself — no prebuilt old
server exists (the old worktree has ds4-c only) and a release rebuild of that
commit was outside this pass. The accepted evidence (code delta + current
survival across four requests) is what this check rests on.

CLAIM 7 — regression gates (verified)

    ./run-bonsai.sh ids -> "IDENTICAL: both backends print the same 8 ids",
      exit 0 (CUDA 1.65 s, CPU 41.03 s)
    make bonsai-cuda-parity CUDA_ARCH=sm_89 -> "bonsai cuda parity: PASS",
      exit 0
    make test-qwen35-rust-host CUDA_ARCH=sm_89 -> "qwen35 rust host parity
      (cuda): PASS", "(cpu): PASS", exit 0
    cargo test -p ds4-core -> exit 0; lib 295 passed/0 failed/4 ignored,
      bind 13, catalog 5, chat_* (7+1+9+25+1+7), inkling_catalog 7 (+1 ign),
      layout 13, ling3vl 3, mimo2 1, native_api 2, payload 5, serving_docs 1,
      session 4, shape 1, tensors 7, tokenizer 5, validate 4 — all ok
    bash tests/run.sh -> "tests/run.sh: all checks passed", exit 0
      (pq2-0-test, test-catalog-parity and tests/qa-gate.sh all PASS)

cargo test -p ds4-server --lib (the named pre-existing failure): at HEAD
314 passed, 1 failed (cache_identity::tests::bounded_sidecar_and_ple),
1 ignored, exit 101. Reproduced at 78c086f in a scratch worktree
(misc/scratch/qa-wt detached at 78c086f, CARGO_TARGET_DIR
misc/scratch/qa39/target-78): first full run 312 passed, 3 failed — the two
extra ones (tool_memory::tests::ktm_codec_matches_c_oracle and
ktm_decode_matches_c_string_nul_truncation) were fresh-worktree artifacts:
the compiled tests/parity/kv_c_oracle was absent (Command::new hit ENOENT,
tool_memory.rs:584-588); after make tests/parity/kv_c_oracle in the worktree
the suite ran once at 315 passed/0 failed. The cache_identity failure is
timing-dependent, not deterministic: isolated at HEAD it failed 3/3; isolated
at 78c086f it failed 2/3 with one pass; the complete worktree suite passed
315/0 once. The test (crates/ds4-server/src/cache_identity.rs:400-417)
rewrites a 4-byte file and asserts the stat snapshot changed; this host's
tmpfs /tmp gives both writes the same coarse timestamp tick (observed
mtime/ctime identical, [1790776944, 180865037]), so the assert_ne fails
whenever the writes share a tick. cache_identity.rs is not in the unit's diff
(a7fbee4 touches ds4.c, run-bonsai.sh, crates/ds4-core/src/serving_host.rs,
crates/ds4-server/src/{bin/ds4-server-rs.rs,generate.rs,models.rs,render.rs,
tools.rs,worker_run.rs} and docs), so it is pre-existing and unrelated. The
worktree and the copied target dir were removed afterwards; the main tree's
sources were not touched.

NOTES
- The given claim-2 command's DS4_BONSAI_PORT is not a variable the script
  reads; DS4_BONSAI_SERVER_PORT (run-bonsai.sh:81, docs/BONSAI.md:166) is.
  Script and docs agree with each other; only the test instruction's name
  differs.
- This pass adds no new public surface; every surface listed by
  tests/qa-gate.sh stays covered by the earlier sections of this report.

verdict: overall PASS
