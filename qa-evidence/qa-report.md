# CUDA ldmatrix review (PR 67)

QA evidence: ldmatrix address operand fix, ds4_cuda.cu plus the mmq extension (extended unit)

Unit identification
Repository /data/ds4-dfm-rs, branch fix/ldmatrix-address-operand. HEAD 7a78fcd,
identical to origin/main. All changes are uncommitted working-tree changes.
Files in the extended unit:
FILE ds4_cuda.cu, modified,
  sha256 ca69e1999f7028e9d51690f8ceb1824b218d5db08ab1ad831cca939562a72738
  (byte-identical to the revision the Part A QA pass verified; re-checked with
  sha256sum against that pass's backup)
FILE cuda/mmq/ds4_fattn.cu, modified,
  sha256 a227e30ea586f3d76799d02c4f9c44523b04ace11fe668b051223c23e5e9f823
FILE cuda/mmq/inkling_attention.cuh, modified,
  sha256 2e37571cb2ffc3df928b17fff4a176e3e6a92770653489e833ca1acf0c7096b7
FILE Makefile, modified (adds the tests/cuda_tokentile_ldmatrix rule and the
  test-cuda-tokentile-ldmatrix target)
FILE .gitignore, modified (adds /tests/cuda_tokentile_ldmatrix)
FILE tests/cuda_tokentile_ldmatrix.c, new file, untracked (token-tile guard)
FILE tests/qa-gate.sh, new file, untracked (rule 19 guardrail mount)
FILE tests/run.sh, new file, untracked (local unit entry)
The extension since the previous QA pass: the same "r"(cvta) to "l"(ptr)
operand fix in the five mmq ldmatrix helpers (solar_fattn_ldsm_x4,
solar_fattn_ldsm_x4_trans, dots3_ldsm_x2 in cuda/mmq/ds4_fattn.cu and
ia_ldsm_x4, ia_ldsm_x4_trans in cuda/mmq/inkling_attention.cuh), plus the
tests/qa-gate.sh + tests/run.sh guardrail mount. Both changed mmq surfaces are
in one translation unit: cuda/mmq/ds4_fattn.cu includes inkling_attention.cuh
(line 42), so one object rebuild covers both.

Environment
GPU RTX 4070 SUPER, sm_89, CUDA 13.3 at /usr/local/cuda. Shared GPU; runs
serialized one at a time. Builds: make -j4 <target> CUDA_ARCH=native. The three
changed sources carry no changes to any other toolchain (Metal/macOS untouched
by this unit).

Covered surfaces
FILE ds4_cuda.cu
FILE cuda/mmq/ds4_fattn.cu
FILE cuda/mmq/inkling_attention.cuh
FILE Makefile
FILE .gitignore
FILE tests/cuda_tokentile_ldmatrix.c
FILE tests/qa-gate.sh
FILE tests/run.sh
FUNCTION tt_ldmatrix_x4_addr, tt_ldmatrix_x2_addr, tt_ldmatrix_x2_trans_addr
  (and the pass-through wrappers tt_ldmatrix_x4/x2/x2_trans)
FUNCTION tt_hmma_score_stage, tt_pv_mma_stage
FUNCTION attention_tokentile_hmma_kernel
FUNCTION solar_fattn_ldsm_x4, solar_fattn_ldsm_x4_trans, dots3_ldsm_x2
FUNCTION ia_ldsm_x4, ia_ldsm_x4_trans
TARGET test-cuda-tokentile-ldmatrix
TARGET test-solar-fattn, test-inkling-attention

PART A, the original token-tile unit (verified in the previous QA pass; the
changed file is byte-identical, and the claims that must still hold were
re-checked live, see Part C)

Claim A1: the defect is in the committed revision and gone after the fix.
Ran: grep -n '"r"(a)' ds4_cuda.cu and on origin/main:ds4_cuda.cu.
Observed: origin/main has exactly 3, at lines 14931/14942/14953; the worktree
has 0; repo-wide 0 outside misc/scratch. CLAIM HOLDS.

Claim A2: reachability. The three helpers are called only from
tt_hmma_score_stage (line 15578) and tt_pv_mma_stage (lines 15723, 15728), both
instantiated only inside attention_tokentile_hmma_kernel; the three wrappers are
dead code. Three dispatch gates launch that kernel: 30630 (decode-mixed, allows
n_comp == 0), 31092 (indexed, requires n_comp != 0), 31500 (zero-prefix
prefill). The shape gates are head_dim 512, n_head 64, window 128; reading the
shape catalog, only the DeepSeek V4 Flash shape meets all three (DS4 Pro fails
n_head 128, GLM 5.3 Flash fails n_swa 0, MiMo2 fails head_dim 192). The unit's
cited gate lines 30624/31086/31490 are off by 6-10 lines; same gates. CLAIM
HOLDS.

Claim A3: mechanism. Pre-fix object SASS: all 104 LDSM sites in
attention_tokentile_hmma_kernel take their address register from
IADD3 Rx, Ry, -c[0x0][0x18], RZ. The operand probe (ldm_variants.cu) faults for
every "r" form and loads for every "l" form; compute-sanitizer names the LDSM PC
("Invalid __shared__ read of size 16 bytes ... out of bounds"). Post-fix the
same IADD3 remains but its input is the generic shared-window base, so it is the
correct conversion. The vendored cuda/mmq/mma.cuh uses "l" at all four of its
ldmatrix sites. CLAIM HOLDS.

Claim A4: the guard catches the defect. Stashed ds4_cuda.cu, rebuilt
(make -j4 tests/cuda_tokentile_ldmatrix CUDA_ARCH=native, rc 0), ran
./tests/cuda_tokentile_ldmatrix 0: cudaErrorIllegalAddress from the attention
kernel, exit 1. Restored (sha256 verified, grep 0, stash empty), rebuilt, same
run: PASS, exit 0. The unit's guard-ab.log prints "case0 rc=0" for the failing
pre-fix run, which is a capture artifact; the true exit code is 1. CLAIM HOLDS.

Claim A5: make test-cuda-tokentile-ldmatrix CUDA_ARCH=native exits 0 with three
PASS lines. CLAIM HOLDS (re-run live in Part C).

Claim A6: compute-sanitizer memcheck 0 errors on the tile path with the fix.
CLAIM HOLDS (re-run live in Part C).

Claim A7: the oracle is independent of the kernels and models the documented
semantics. Every modeled element maps to the dense builder, raw mirror,
softmax/sink code; the kill-switch fallback (a different kernel) matches the
same reference at 1.08e-6, corroborating the model. Residual: ref_row leaves
out[] unwritten for nk == 0 or nk >= MAX_KEYS; no committed case reaches it
(nk 97 to 224). CLAIM HOLDS.

Claim A8: warnings. ds4_cuda.cu emits the same two pre-existing warnings (lines
1258 and 9084) in both states; warning delta zero; the literal phrase
"warning-free" is inaccurate since those two remain. CLAIM HOLDS AS DEFINED.

Claim A9: honesty checks. cuda_long_context_smoke fails identically (rc 1,
byte-identical stderr) pre-fix and post-fix; the failing step is a host-side
tensor upload in a second attention case (cannot reach the token-tile gates).
Running one case per process hides nothing about this fix. No throughput claim
was made. CLAIM HOLDS.

PART B, the extension (claims of this pass)

Claim B1: the mmq diffs are the operand change with the address arithmetic
unchanged.
Ran: git diff -- cuda/mmq/ds4_fattn.cu cuda/mmq/inkling_attention.cuh, plus:
  git stash push -- <the two files>; grep -c '"r"(addr)' (3 and 2 matches);
  git stash pop; grep -c '"r"(addr)' (0 and 0); sha256sum -c against backups.
Observed: each of the five helpers loses the
  const uint32_t addr = (uint32_t)__cvta_generic_to_shared(...); line and passes
  the pointer itself: "l"(row_ptr) in the three ds4_fattn.cu helpers,
  "l"(p) in the two inkling helpers. No arithmetic line changes: the callers
  still pass the same row_ptr/p expressions, and the row/column index code is
  not in the diff. The diff hunks are only those five functions plus added
  comments. The pre-fix grep count (3 and 2) equals the number of patched
  helpers, and the post-fix count is 0. CLAIM HOLDS.

Claim B2: the extension is what revives the kernels (A/B re-derived).
Ran: sha256 backups; git stash push -- the two files; make -j4
tests/test_solar_fattn tests/test_inkling_attention CUDA_ARCH=native (rc 0,
0 warnings); ./tests/test_solar_fattn -> rc 1 with
  "ds4: CUDA synchronize failed: an illegal memory access was encountered"
  "FAIL line 96: ds4_gpu_synchronize()"
./tests/test_inkling_attention -> rc 1 with
  "ds4: CUDA tensor read failed: an illegal memory access was encountered"
  "FAIL line 184: ds4_gpu_tensor_read(f->out, 0, got, bytes)"
Then git stash pop; sha256sum -c: both OK; grep count 0/0; stash list empty.
Rebuilt the same way (rc 0); ./tests/test_solar_fattn -> rc 0, last line
  "Solar attention rollback parity passed", different=0 on every printed shape
  (largest: rows=4096 pos=61440 cap=65601 heads=64 pair=309.918 ms ws=188.576 ms
  different=0; the unit's log has pair=319.030 ws=194.922 for that shape, timing
  noise on a shared GPU).
./tests/test_inkling_attention -> rc 0, last line
  "Inkling attention/KV checks passed" (tensor-core=2706.979 us grouped=12638.393
  per-head=120582.137 at extent=1024/rows=512 and 792.395/726.331/5088.208 at
  rows=16; the unit's log has 2922.678/13146.133/134980.209 and
  803.624/787.637/5048.379).
Log consistency: the unit's misc/scratch/ldm/mmq-ab.log is consistent with my
run line for line in every failure and pass signature. One apparent oddity in
that log, a PASS line ("attention extent=512 chunk=8201 outputs/cache passed
(exact)") printed after the pre-fix inkling FAIL lines, is a stdout/stderr
buffering artifact: that line is the pre-HMMA exact pass, whose block-buffered
stdout is flushed at exit after the unbuffered stderr FAIL lines; my own capture
reproduces the same ordering. CLAIM HOLDS.

Claim B3: the mmq-shape probe.
Ran: nvcc -arch=native -o ldm_mmq_variants misc/scratch/ldm/ldm_mmq_variants.cu;
./ldm_mmq_variants 1..6.
Observed:
  v1 (x4 "r"):      sync=an illegal memory access was encountered r0=0
  v2 (x4 "l"):      sync=no error r0=0x3c003c00
  v3 (x4t "r"):     sync=an illegal memory access was encountered r0=0
  v4 (x4t "l"):     sync=no error r0=0x3c003c00
  v5 (x2 "r"):      sync=an illegal memory access was encountered r0=0
  v6 (x2 "l"):      sync=no error r0=0x3c003c00
CLAIM HOLDS.

Claim B4: the gate enforces rather than rubber-stamps.
Ran before the report update: QA_MODEL=deepseek-v4.1-CC-flash bash
tests/qa-gate.sh -> exit 1, exactly two red checks:
  "FAIL report covers surface: tests/qa-gate.sh"
  "FAIL report covers surface: tests/run.sh"
with the verdict check PASS (the old report's last line was the verdict) and
"SKIP freshness (the branch has no pushed base yet;
origin/fix/ldmatrix-address-operand is unborn)". Note: the two mmq surfaces
passed coverage in that run only because the old report mentioned them in a
finding, which shows the coverage check is a substring presence test.
Negative controls on scratch copies of a report (QA_SURFACES=ds4_cuda.cu):
  last non-empty line "verdict: overall FAIL" -> red, rc 1
  "verdict: overall PASS" followed by a stray non-empty line -> red, rc 1
  missing report file -> 3 red (exists, verdict, coverage), rc 1
  clean "verdict: overall PASS" -> green, rc 0
CI skip: GITHUB_ACTIONS=true bash tests/qa-gate.sh -> "SKIP tests/qa-gate.sh ...",
rc 0.
Exclusion filter: the raw git-status entry list is
  .callgraph-index.bin, cuda/mmq/ds4_fattn.cu, cuda/mmq/inkling_attention.cuh,
  ds4_cuda.cu, ds4-dfm-rs-handoff.md, .gitignore, graphify-out/, Makefile,
  qa-evidence/, tests/cuda_long_context_smoke, tests/cuda_tokentile_ldmatrix.c,
  tests/qa-gate.sh, tests/run.sh
and after the default exclusion the surfaces are the unit's own five modified
files, the regression test and the two guardrail scripts; dropped are
.callgraph-index.bin, ds4-dfm-rs-handoff.md, graphify-out/, qa-evidence/ and the
untracked built binary tests/cuda_long_context_smoke. The built token-tile test
binary does not appear because .gitignore covers it. The filter keeps every
reviewable surface and drops only machine-local or regenerable ones. Run again
after the report update (the state this file is in): exit 0, "QA GATE: overall
PASS", report exists PASS, operative verdict PASS, all eight surfaces covered,
freshness still SKIP. CLAIM HOLDS
with two recorded limitations: the freshness check is skipped on this branch
because origin/<branch> does not exist (it would engage once the branch is
pushed), and surface coverage is a presence test, not a content check.

Claim B5: tests/run.sh ordering, exit handling, --full.
Ran: read tests/run.sh; ran QA_MODEL=deepseek-v4.1-CC-flash bash tests/run.sh
before the report update and (after the update) again.
Observed (pre-update run, log qa2-run-preupdate.log): the gate output comes
first and is red (same two surface failures), then all three suites still run
and pass (three token-tile PASS lines, "Solar attention rollback parity passed",
"Inkling attention/KV checks passed", 0 "FAIL line" occurrences), and the script
exits 1. So ordering is gate first as documented, every failure is captured by
the per-step "or status=1" guard, later suites still run after an earlier
failure (no short-circuit),
and the accumulated status is propagated. After the update: the first post-update
run of the same command failed in the inkling suite with "FAIL line 45: t" (the
ds4_gpu_tensor_alloc inside tests/test_inkling_attention.c upload() returned
NULL) while a co-tenant held the GPU at 10147 MiB used of 12282 MiB, 100 percent
utilization; the same binary passed alone immediately after ("Inkling attention/KV
checks passed") and the next run of bash tests/run.sh exited 0 with 0 "FAIL line"
occurrences and all three suites passing (log qa2-run-postupdate2.log). That is
an environmental allocation failure of the shared box, not a code regression.
--full is recognized only as the first
argument and adds make test, whose recipe runs ds4-eval --self-test-extractors,
ds4_test and tests/test_split_gguf; that matches the documented model-free suite
and the documented usage "bash tests/run.sh --full". I did not execute the
--full path end to end (it builds and runs the full C suite); the branch was
judged by reading, and everything else on that path was exercised live. CLAIM
HOLDS.

PART C, earlier claims re-checked live after the extension

C1 token-tile parity: make test-cuda-tokentile-ldmatrix CUDA_ARCH=native exits 0
with three PASS lines: max_rel 0.000983 (limit 0.004), 0.00101, 1.37e-07.
C2 compute-sanitizer: DS4_ATTN_TOKENTILE=1 compute-sanitizer --tool memcheck
--launch-timeout 300 ./tests/cuda_tokentile_ldmatrix 0 -> PASS line, "ERROR
SUMMARY: 0 errors", rc 0.
C3 warnings: cuda/mmq/ds4_fattn.cu compiles with 0 warnings pre-fix and 0
post-fix (my own A/B builds, logs qa2-prefix-build.log / qa2-postfix-build.log).
ds4_cuda.cu is byte-identical to the version whose A/B showed the same two
pre-existing warnings (lines 1258, 9084) in both states, so its warning delta
remains zero.

Findings
1. The gate's surface coverage is a substring presence test and its freshness
   check is skipped while origin/<branch> does not exist; both are recorded
   limitations, not claim failures. (Evidence in Claim B4.)
2. run.sh continues through the suites when the gate is red and still exits
   nonzero; that is a reasonable local-entry design but means a red gate does
   not short-circuit the run.
3. Claim A8 wording: the ds4_cuda.cu translation unit is not literally
   warning-free; two pre-existing warnings remain in both states. The change's
   warning delta is zero.
4. The unit's guard-ab.log records "case0 rc=0" for the pre-fix failing run;
   the true exit code is 1. Background artifact of that log's capture, not of
   the guard.
5. The regression test's header comment frames the token-tile path as mattering
   for "a DeepSeek V4 or GLM 5.3 artifact"; per the shape catalog only the
   DeepSeek V4 Flash shape meets all three gates (GLM 5.3 Flash has n_swa 0).
   Context wording only, no claim depends on it.
6. Oracle robustness nit: tests/cuda_tokentile_ldmatrix.c ref_row leaves out[]
   unwritten if nk == 0 or nk >= MAX_KEYS (no committed case reaches it).
7. The five out-of-scope ldmatrix sites flagged by the previous QA pass are now
   fixed and covered by Claims B1-B3; that finding is closed.
8. The unit's mmq-ab.log ordering oddity (PASS line after FAIL lines in the
   pre-fix inkling section) is a stdout buffering artifact, reproduced and
   explained in Claim B2.
9. Observed once during this pass: a transient device-allocation failure in the
   inkling suite under co-tenant GPU pressure (FAIL line 45, tensor alloc NULL;
   10147 MiB of 12282 MiB used by other processes). The binary passed alone and
   the next full run.sh run was green. Environmental, recorded for the box's
   shared-GPU context.

What I could not verify
- The make test (--full) path end to end: it builds and runs the full model-free
  C suite; the branch was verified by reading, the rest of run.sh live.
- Root cause of the pre-existing second-round tensor-upload failure
  (cudaErrorInvalidValue) that forces one case per process; only its
  pre-existence and pre/post identity were checked.
- Runtime of the indexed (31092) and zero-prefix (31500) token-tile tiers; only
  the decode-mixed tier is driven by the test, the other two are source-read
  reachability.
- End-to-end serving with real model artifacts: everything is verified at the
  kernel geometry with synthetic tensors.
- Performance: no throughput claim exists for this unit, so nothing
  performance-related was measured.

verdict: overall PASS

# Prism Bonsai review (PR 69)

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

========================================================================
UNIT 2026-09-30 — run-bonsai.sh `serve` (keep a ds4-server up for a client)
========================================================================

Diff: run-bonsai.sh (serve_pid/serve_start/serve_stop/serve_status/serve_logs
and dispatch), docs/BONSAI.md, docs/releases/bonsai-serving-2026-09-30.md.
Uncommitted working tree; the rule 19 gate diffs commits only, so none of
these paths is in its surface list. All checks below run LIVE against the
ds4-server already serving /data/models/Ternary-Bonsai-2-27B-PQ2_0.gguf on
127.0.0.1:8899.

FUNCTION serve_status — `./run-bonsai.sh serve status` (exit 0):
    server:  running (pid 1356568)
             listening on 127.0.0.1:8899 model_id=Ternary-Bonsai-2-27B-PQ2_0 ...
    base_url: http://127.0.0.1:8899/v1
    serving: Ternary-Bonsai-2-27B-PQ2_0 ctx 45056
    vram:    10815 MiB, 12282 MiB
  ENDPOINT /v1/models returned id Ternary-Bonsai-2-27B-PQ2_0 and
  context_length 45056, the same id and context status printed. The vram line
  is the host's whole-card use: 10815 MiB once requests had run, 7449-7474 MiB
  on a freshly started server before any request.

FUNCTION serve_logs — `serve logs 5` tailed the last 5 capture lines (exit 0);
  `serve logs` with no arg tails 20. The capture is
  misc/scratch/bonsai-serve.log; serve_start clears it before each start.

ENDPOINT /v1/chat/completions — advertised id and alias both answer (temp 0):
  model=Ternary-Bonsai-2-27B-PQ2_0, max_tokens 16: finish_reason length,
  content "", reasoning_content "The user is asking a simple factual
  question: ...", usage 45/16/61.
  model=prism-bonsai-2-27b, max_tokens 128: finish_reason stop, content
  "\n\nParis.", reasoning_content "... The capital of France is Paris.\n",
  usage 45/36/81.
  The family emits the reasoning block before any content, so a short
  max_tokens returns empty content with finish_reason length. Matches the
  release doc's "finish stop, usage 45/36/81"; the exact content carries two
  leading newlines (the doc writes it as "Paris.").

ENDPOINT /v1/chat/completions stream:true — SSE: 27 `data:` lines, deltas
  carrying reasoning_content, a final delta:{} with finish_reason length, then
  `data: [DONE]`. exit 0.

CLIENT METHOD open-grok chat completion — the operator's path:
    mkdir -p /tmp/bonsai-og-qa && open-grok --cwd /tmp/bonsai-og-qa
      -m bonsai-local --max-turns 1 -p "Reply with exactly this text and ..."
  stdout "BONSAI-OK", exit 0. The [model.bonsai-local] block (config.toml:308-
  318) uses model "prism-bonsai-2-27b", base_url .../v1, context_window 45056,
  matching the served context. Read and verified.

FUNCTION serve_pid / serve_stop — pid-file safety. Planted a foreign pid (a
  `sleep 300`) in the pid file: `serve stop` printed "server:  not running",
  exit 0, and the sleep was STILL ALIVE. serve_pid rejects the number because
  /proc/<pid>/cmdline carries no "ds4-server"; kill "$pid" runs only on the
  recorded pid.

FUNCTION serve_start — refusals:
  - second start with the server up: "server:  already running (pid 1356568);
    ./run-bonsai.sh serve stop first", exit 0 (started nothing).
  - slot held by another ds4 (pid file moved aside so the pid guard did not
    short-circuit): "another ds4 process holds the single model slot; stop it
    first:" plus the pgrep line naming pid 1356568, exit 1, no server started.
  - bad subcommand: "ERROR: serve wants start, stop, status or logs (got:
    bogus)", exit 1. `serve` with no arg == `serve start`; `help` lists the
    serve line (usage() sed range moved 5,14p -> 5,15p).

FUNCTION serve_stop / serve_start cycle (left running, as required):
    serve stop  -> "server:  stopped (pid 1356568)"; real 1.008s; port 8899
                   refused (curl exit 7); /tmp/ds4.lock released; no ds4-server.
    serve start -> up, new pid 1383741; real 2.306s; id read back from
                   /v1/models; listening line printed.
  Timing matches the release doc's "start 2.3 s, stop 1 s". Final state:
  `serve status` reports running pid 1383741, serving ctx 45056, vram 7471 MiB;
  curl /v1/models -> Ternary-Bonsai-2-27B-PQ2_0 45056; ss shows
  LISTEN 127.0.0.1:8899 users:(("ds4-server",pid=1383741)).

bash tests/run.sh -> first run: "tests/run.sh: all checks passed", exit 0
  (pq2-0-test, catalog parity, tests/qa-gate.sh all PASS). Second run: exit 1
  — make test-catalog-parity failed on
  serving_host::tests::cpu_quote_uses_ram_not_discrete_fb, assert_eq!(cpu,
  metal) at serving_host.rs:3950 (left 19929518080, right 19929145344, diff
  372736 bytes). Both calls read /proc/meminfo MemAvailable
  (host_available_bytes -> meminfo_available, serving_host.rs:699,2281-2282),
  so the assert compares two consecutive samples of a live kernel counter and
  races under parallel suite load. Isolated: 5/5 pass; full ds4-core lib suite
  6/6 pass (295 passed each). serving_host.rs is not in this unit's diff
  (run-bonsai.sh + docs only): the flake is pre-existing and unrelated.

NOTES
- docs/BONSAI.md's status example shows "vram: 7474 MiB" (a pre-request
  reading: reproduced 7449-7474 MiB right after start) and the sizing table
  calls it "7.47 GiB device use measured". 7474 MiB is 7.30 GiB, so the GiB
  figure is ~0.17 GiB high — a MiB/GiB slip in the prose, not a live failure.
  After requests run, `serve status` vram reads ~10.8 GiB (the KV bank and
  graph buffers allocate on first use); the 45056 context still opens and
  serves. Not exercised: a client window above 45056 (the doc says the session
  refuses past its bound).
- stop only ever kills the pid the file records; a reused pid number is
  rejected unless the reused process is itself a ds4-server (the
  /proc/<pid>/cmdline text test is the guard's only check).

========================================================================
UNIT 2026-09-30 — Bonsai bf16 gated delta-net matvec tile (prefill)
========================================================================

Diff: cuda/qwen35_attn_gdn.cuh (matvec_bf16_tiled + the dispatch in
ds4_gpu_qwen35_matvec_bf16_tensor), tests/test_qwen35_cuda.cu (test 6
test_bf16_matvec_tile), docs/releases/bonsai-perf-2026-09-30.md.
Uncommitted working tree; tests/qa-gate.sh diffs commits only, so none of
these paths is in its surface list. The one gate surface this unit moves,
ds4_gpu_qwen35_matvec_bf16_tensor, stays covered by the earlier sections.
Read-only pass: no tracked file other than this report was touched.
Binaries current with the edit: cuda/qwen35_attn_gdn.cuh 17:04:03,
ds4-server/ds4-c 17:04:54, and both embed DS4_QWEN35_BF16_MATVEC_TILED plus
the two kernels (nm _Z11matvec_bf16... and _Z17matvec_bf16_tiled...);
tests/test_qwen35_cuda is newer than the test edit.

1. PARITY — `make test-qwen35-cuda CUDA_ARCH=sm_89` (exit 0). Full group 6,
   every width bit-identical, rel L2 ~2-4e-07 (the float-rounding band):

    bf16 tile T=1:   bit-identical to the untiled kernel, max abs 1.45e-05, rel L2 3.52e-07
    bf16 tile T=8:   bit-identical to the untiled kernel, max abs 1.72e-05, rel L2 2.42e-07
    bf16 tile T=9:   bit-identical to the untiled kernel, max abs 1.91e-05, rel L2 2.42e-07
    bf16 tile T=11:  bit-identical to the untiled kernel, max abs 1.81e-05, rel L2 2.32e-07
    bf16 tile T=486: bit-identical to the untiled kernel, max abs 2.67e-05, rel L2 2.36e-07
    bf16 matvec tile parity: PASS
    PQ2_0 CUDA parity: PASS

   The T=9/11/486 lines are byte-identical to the three the release doc
   quotes. The rest of the suite (PQ2_0 row lookup, shape guards, host embed,
   fold, gdn gates, attention split and token-tile, MMQ/MMVQ) all PASS;
   PQ2_0 CUDA parity: PASS.

2. A/B on the served path — /tmp/ab-tiled.sh, three interleaved pairs, one
   fresh ds4-server per sample, model prism-bonsai-2-27b, a 2140-token prompt
   with 64 decode tokens, base = DS4_QWEN35_BF16_MATVEC_TILED=0:

    arm base (TILED=0)                     arm tiled (default)
    ttft 3408.8/3295.8/3275.6 ms           2242.2/2130.7/2141.7 ms
    prefill 640.1/662.7/666.2 tok/s        983.7/1033.3/1030.0 tok/s
    decode 17.30/17.60/17.90 tok/s         17.10/17.80/18.00 tok/s
    mean 3326.7 ms / 656.3 tok/s           2171.5 ms / 1015.7 tok/s

   -34.7% ttft / +54.8% prefill, against the report's means 3291.9 / 663.2
   and 2154.0 / 1023.3 (-34.6% / +54.3%): reproduced within ~1%. Decode means
   17.60 vs 17.63 tok/s — unchanged, as the T<=8 dispatch intends. The
   64-token completions are identical between the arms (same reasoning_content,
   finish_reason length). Clocks 2775-2790 MHz of a 3105 MHz maximum,
   119-123 W, 51-55 C.

   Kernel level, nsys, the doc's own method (ds4-c --cuda --first-token-test
   with DS4_QWEN35_SESSION=1, one 483-token chunk, one step):

    base  matvec_bf16        96 calls x 3,013,202 ns = 289.27 ms (41.2% of GPU)
    tiled matvec_bf16_tiled  96 calls x   168,583 ns =  16.18 ms ( 3.8%)

   Total GPU kernel time 702.9 -> 427.1 ms; the matvec delta 273.1 ms is 99%
   of the 275.8 ms drop. Both arms print the same step token (id 760). The
   doc's 259.0 -> 13.7 ms at 463 tokens is the same kernel and ratio (17.9x
   here vs 18.9x there); the absolute figures differ because my prompt is 483
   tokens and the clocks are not the doc's.

3. DISPATCH BOUNDARY AND TAIL — cuda/qwen35_attn_gdn.cuh:703-706:
   `if (T > MATVEC_BF16_TT && !(tiled && tiled[0] == '0'))` with
   MATVEC_BF16_TT = 8 (line 465): T <= 8 takes the untiled matvec_bf16, T > 8
   the tiled kernel, and a first char '0' in DS4_QWEN35_BF16_MATVEC_TILED
   forces the untiled path. The claim holds.
   Tail safety from the kernel (467-491): gridDim.y = ceil(T/8),
   t0 = blockIdx.y * 8, nt = min(8, T - t0); every x read (t0+j) and out
   write (t0+j) is guarded by j < nt, and t0 < T for every launched block
   (by <= ceil(T/8)-1), so t0 + nt <= T for any T — no read or write outside
   [0, T). M = 48 is a multiple of 4 here and `if (row >= M) return` covers
   the rest. Observed tails: T=11 (nt=3) and T=486 (nt=6) are bit-identical
   in the suite; the served 2140-token run's last chunk is 92 (nt=4, since
   ds4.c:74458 chunks by prefill_cap 512) and the nsys chunk T=483 has nt=3
   (483 = 60*8+3), both with correct output. T=15 was not added: the width
   list is hardcoded in the test (1069) and this pass is read-only.
   compute-sanitizer --tool memcheck --kernel-name regex=matvec_bf16 reported
   no invalid __global__ access in the matvec kernels, but its run is polluted
   by cudaErrorInvalidValue on cudaMemcpy (the test's mmap/register path) that
   fails four earlier groups, so the tool is inconclusive as a gate — not a
   device OOB finding. The OOB conclusion rests on the code argument above.

4. KILL SWITCH — the A/B base arm (TILED=0) is the SLOW arm (3326.7 ms ttft,
   nsys shows only matvec_bf16); the default arm is the fast one (2171.5 ms,
   only matvec_bf16_tiled). The switch changes the executed kernel.

5. `bash tests/run.sh` -> "tests/run.sh: all checks passed", exit 0
   (make pq2-0-test PASS; make test-catalog-parity, ds4-core lib 295 passed /
   0 failed / 4 ignored plus every integration suite PASS; tests/qa-gate.sh
   overall PASS). No flake this run.

HOST STATE — after the pass pgrep -a ds4-server / ds4-c / ds4 / nsys are all
empty; the A/B's own `serve stop` and the self-terminating session runs left
nothing behind. GPU idle: 242 MiB of 12282 MiB (Xorg + cinnamon), 39 C,
12.43 W, clocks 465 / 3105 MHz.

NOT RUN / UNVERIFIED
- The doc's "Unchanged gates" line (./run-bonsai.sh ids, session,
  make test-qwen35-session and -multichunk) was not re-run here; this pass
  verified the CUDA unit gate and the served path only.
- The doc's parenthetical "486 the chunk this host really prefills" is not a
  width I could tie to the server: the chunk knob default is 512
  (DS4_QWEN35_CHUNK_DEFAULT, ds4.c:69059) and my session run used 483. A
  comment imprecision, not a functional issue.
- The doc's "58 blocks for a 463-token chunk" is gridDim.y; the grid is
  12 x 58 = 696 blocks of 128 threads. Prose, not a defect.
- ncu remains unusable on this host per the doc (ERR_NVGPUCTRPERM); not
  re-checked.

==========================================================================
RULE 19 RE-VERIFICATION — fb8ee47 "perf(cuda): split the Bonsai decode
attention over its key range" (branch feature/qwen35-port, HEAD fb8ee47)
==========================================================================

Claim (docs/releases/bonsai-splitk-2026-09-30.md): the qwen35 graph now hands
the decode attention a split-K partial buffer; at a 15k context decode goes
4.14 -> 33.81 tokens/s, level with /data/ds4 (33.83), prefill unchanged and
the token ids unchanged.

1. KILL-SWITCH A/B — one interleaved pair, a fresh server process per arm.
   Server: /data/ds4-dfm-rs/ds4-server -m
   /data/models/Ternary-Bonsai-2-27B-PQ2_0.gguf --cuda -c 45056
   --mem-floor-gb 1 --host 127.0.0.1 --port 8899, with DS4_CUDA_COPY_MODEL=1.
   Body /tmp/cmp-body-15k.json (85090 bytes, ~15k-token prompt, 64 decode
   tokens, streamed); timing by /tmp/cmp-measure.py, which counts SSE
   content/reasoning deltas (ttft = first such token, decode rate = (n-1)/
   window).

     arm                          decode tok/s   ttft      first_byte  window
     DS4_QWEN35_ATTN_SPLITK=0     4.11           20.70 s   5.03 s      15.34 s
     default (split-K on)         34.11          19.53 s   5.03 s      1.85 s

   Reproduces the report within noise: base 4.11 vs 4.14, split-K 34.11 vs
   33.81, ttft 20.70/19.53 vs 20.10/19.54 — an 8.3x gap between the arms. The
   SLOW arm is the DS4_QWEN35_ATTN_SPLITK=0 arm, so the knob changes the
   executed kernel, not merely a code path.

2. Both arms run the row-exact path; only the split selection changed. Decode
   is T=1 (ds4.c:69664, 69773 call qwen35_graph_forward(...,1u,...)), so
   tokentile = T >= 32 && ... is false in both arms (ds4.c:69306,
   cuda/qwen35_attn_gdn.cuh:579-585) and the dispatcher takes attention<DIM>
   (row-exact) in both. The only difference is partial = NULL (splits = 1)
   versus g->attn_partial (splits = min(64, (keys+31)/32) then attn_merge)
   (cuda/qwen35_attn_gdn.cuh:609-628). At T >= 32 prefill both arms hand no
   partial buffer and take the token-tile kernel, so prefill is unchanged and
   the ~19.5 s ttft is prefill in both arms.

3. qwen35_graph_free: g->attn_partial is in the scratch[] list (ds4.c:69374,
   the GPU definition at :69370), so the 24.2 MiB buffer is freed on graph
   close; :69586 is the DS4_NO_GPU stub. The leak the audit found is closed.

4. Gates. make test-qwen35-cuda CUDA_ARCH=sm_89: exit 0, 60 PASS, 0 FAIL; the
   attention group (split vs single and token-tile vs split at ctx 2048 and
   32768, T = 1 / 16 / 32 / 33 / 64) all PASS.
   DS4_CUDA_COPY_MODEL=1 DS4_TEST_MODEL=... DS4_TEST_BACKEND=cuda
   ./tests/test_qwen35_session: exit 0, 21 PASS, 0 FAIL (368.8 s), including
   "PASS  long prompt: the split-K order does not move the ids".
   Multichunk (same plus DS4_QWEN35_PREFILL_CHUNK=2): exit 0, 21 PASS, 0 FAIL
   (375.1 s). ./run-bonsai.sh ids: IDENTICAL on both backends, exit 0.

5. bash tests/run.sh: exit 1 — the only red is tests/qa-gate.sh "report covers
   surface: DS4_QWEN35_ATTN_SPLITK", the new env knob absent from this report
   until this section. make pq2-0-test and make test-catalog-parity (ds4-core
   299 passed / 0 failed / 4 ignored, every integration suite PASS) are green.
   Re-run after this section: tests/run.sh: all checks passed, exit 0.

HOST STATE — the operator's server is back on 8899 (pid 2522625); /v1/models
answers and a chat completion returned (ttft 292.8 ms, decode 39.6 tok/s, 8
tokens). No stray ds4 processes. GPU 10779 MiB of 12282 MiB used (this
server's compute 10446 MiB — the bank materialises on the first request, per
the release doc's limit note), 37 C, clocks 2505 MHz.

NOT VERIFIED / LIMITS
- One interleaved pair (base then fix), one sample per arm; not the report's
  three rounds.
- The /data/ds4 sibling arm and the 30k rows were not re-measured: the "level
  with /data/ds4" clause rests on the release report's numbers.
- The row batch (16) and split ceiling (64) are the reference's; no sweep.
- Clock readings are idle samples before each request (base 2805 MHz, fix
  2505 MHz), not a per-clock comparison.
- The first session-test run was killed at a 600 s wrapper cap while inside
  the long-prompt CPU-reference; the re-run finished in 368.8 s, so that was
  the wrapper budget, not a test hang.
- GPU memory at rest exceeds the pre-pass idle sample (7525 MiB) because the
  restored server has served one verification request; the state left is a
  live, answering server as required.

verdict: overall PASS

## Maintainer CPU regression review - 2026-10-01

Scope: `ds4.c` stateless FFN projection; no arithmetic change.
`tests/test_qwen35_ref.c` exercises unfolded and Hadamard-folded projections
without recurrent state. `make test-qwen35-ref` failed before the fix with
UBSan null member access and passes after it. Catalogue/tokenizer parity and
`make pq2-0-test` pass. The new target runs in `tests/run.sh` and
`.github/workflows/host-parity.yml`; `Makefile` cleanup and `.gitignore` cover
the binary. These model-free checks do not extend artifact qualification.

## Maintainer integration review - 2026-10-01

Merged main at 22a3f05c into the Bonsai branch. The three add/add conflicts
were confined to this report, tests/qa-gate.sh, and tests/run.sh. Both dated
reports remain above. The QA gate retains file and ABI coverage, fork/base
selection, and quiet ref validation. The test entry runs model-free checks
by default; --cuda runs all four CUDA suites; --full also runs make test.

Reviewed the changed Rust family binding, layout, tokenizer, and serial
serving paths:
- crates/ds4-cli/src/session_exec.rs
- crates/ds4-core/src/bind.rs
- crates/ds4-core/src/layout.rs
- crates/ds4-core/src/lib.rs
- crates/ds4-core/src/tensors.rs
- crates/ds4-core/src/tok.rs
- crates/ds4-core/src/validate.rs
- crates/ds4-core/tests/layout.rs
- crates/ds4-core/tests/tokenizer.rs
- crates/ds4-server/src/bin/ds4-server-rs.rs
- crates/ds4-server/src/generate.rs
- crates/ds4-server/src/tools.rs
- crates/ds4-server/src/worker_run.rs
- native/bridge/ds4_host_load.h
- ds4.h
- ds4_cli.c

Reviewed the PQ2_0 layout, MMQ/MMVQ dispatch, fold, and native GPU integration:
- cuda/mmq/common.cuh
- cuda/mmq/ds4_ggml_stubs.h
- cuda/mmq/ds4_mmq.cu
- cuda/mmq/ds4_mmq.h
- cuda/mmq/ggml-common.h
- cuda/mmq/mmq.cuh
- cuda/mmq/mmvq.cu
- cuda/mmq/vecdotq.cuh
- cuda/qwen35_primitives.cuh
- ds4_qwen35_gpu.cuh

Reviewed the fixtures and qualification bounds:
- tests/parity/tokenizer_c_oracle.c
- tests/pq2_0/pq2_ref_generator.c
- tests/pq2_0/reference_checksums.txt
- tests/test_pq2_0.c
- tests/test_qwen35_rows.c
- tests/test_qwen35_session.c
- docs/README.md
- docs/releases/bonsai-cpu-reference-2026-09-30.md

Local catalog/tokenizer parity, PQ2_0, and UBSan CPU regression passed. The
CPU regression and PQ2_0 were repeated after integration and passed again.
PR 67's token-tile, Solar, and Inkling GPU checks passed on GB10. Actual
Bonsai artifact and serving qualification remains bounded by the original
RTX 4070 SUPER records above; it was not rerun on GB10 in this review.
Combined CUDA validation is required separately before merge.

verdict: overall PASS
