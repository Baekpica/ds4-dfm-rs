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
