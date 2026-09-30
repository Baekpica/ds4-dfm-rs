QA report: feature/qwen35-port (Prism Bonsai 2 27B / qwen35 CPU reference unit)

Scope: the eleven commits origin/main..HEAD. Head 99ae2cf, base origin/main.
The unit under test is b59784c, e7c3bbe, 835efd2, c265685, 789eada, 1f5ff5c,
97c64a3, 1d3fefd, b8b4412; d248d22 fixes the one defect the previous QA pass
found, and 99ae2cf mounts this gate.
Tester: independent QA session (rule 19). Date 2026-09-30.
Host: RTX 4070 SUPER (sm_89, CUDA 13.3), 28 cores, 31 GiB RAM.
Artifact: /data/models/Ternary-Bonsai-2-27B-PQ2_0.gguf (6.71 GiB, 851 tensors).
Oracle: /data/ds4 branch bonsai tip bbaf298, prebuilt ./ds4.
Nothing in the repository was modified; scratch work lives in misc/scratch/qa/.
All gates below were re-run on the post-fix tree; the earlier finding is closed.

BUILD
- make cpu -j4: exit 0, relinked ds4-c, ds4-server-c, ds4-bench-c, ds4-eval,
  ds4-agent-c from objects that postdate the fix (ds4.c 06:00:54, ds4_cpu.o
  06:01:12, ds4_cli_cpu.o 06:01:01, ds4-c 06:02:58).
- From scratch on the post-fix sources: cc -O3 -ffast-math -march=native
  -Wall -Wextra -std=c99 -D_GNU_SOURCE -DDS4_NO_GPU -c ds4.c -> exit 0, only
  the 17 pre-existing -Wunused-function warnings (exaone_forward_token_cpu,
  model_get_u64, solar_kv_*, ling3vl_swiglu_clamp_*, qwen4exp_*, ...); ds4_cli.c
  -> exit 0, zero warnings.
- make -B tests/test_qwen35_cuda CUDA_ARCH=sm_89 rebuilt ds4_cuda_test_hooks.o,
  ds4_cuda.o and cuda/mmq/*.o from source (355 s, exit 0; only the pre-existing
  BN and g_rr_scratch_bytes warnings).

FIX RE-VERIFICATION (d248d22) - the defect from the previous pass is closed
- The gate now also requires opt->first_token_test (ds4.c:70096); ds4.h adds
  ds4_engine_options.first_token_test and ds4_cli.c sets it from the parsed
  --first-token-test (ds4_cli.c:1813).
- The exact probe that segfaulted: ./ds4-c -m
  /data/models/Ternary-Bonsai-2-27B-PQ2_0.gguf --cpu -p "hi" -> exit status 1
  (real exit, stdout+stderr captured to a file, no pipe), output:
  "ds4: Bonsai (qwen35) runs only through the CPU reference: use --cpu
  --first-token-test (the graph backend is not implemented yet)". No crash, no
  "prefill layer" line. The same refusal, exit 1, for the bare default-backend
  form without --cpu, for --cuda, and for --cpu --power 90.
- Not weakened: --cpu --power 90 --first-token-test still refuses (exit 1), so
  the new requirement adds a condition rather than replacing the option set.
- Reference driver intact: DS4_QWEN35_STEPS=0 ./ds4-c -m <artifact> --cpu
  --first-token-test -p "x" -> exit 0, prompt rendered, "diagnostic run
  completed on the native cpu path".
- Struct safety: ds4_engine_options gains a public bool mid-struct. Every
  in-tree constructor is deterministic (ds4_cli.c:1608 and ds4_server.c:24108
  use designated initializers, so C99 zero-fills it; ds4_bench.c and ds4_eval.c
  likewise; native/bridge/ds4_bridge.c:222 memsets the struct), and there is no
  Rust or other mirror of the struct (grep over *.rs: none). No field reorder,
  only one insertion.

CLAIM 1 - loader reads PQ2_0 and the artifact loads
- ./ds4-c -m /data/models/Ternary-Bonsai-2-27B-PQ2_0.gguf --inspect printed
  "ds4: prism.hadamard folding: block 1024, 3 sign vector(s), gdn_v_grouped 1"
  and then the full model report, exit 0: arch qwen35, gguf v3, 851 tensors,
  types f32 353 / bf16 96 / pq2_0 402. It does not stop at weight bind.
- pq2_0 is gguf type 142, 128 weights in 34 bytes (ds4.c gguf_types[142]).
  An independent GGUF parse (misc/scratch/qa/gguf_plan.py, python, not ds4's
  loader) read the same directory: 851 tensors, 49 kv, alignment 32.
PASS.

CLAIM 2 - PQ2_0 block format
- make pq2-0-test, re-run after the fix: "pq2_0: all checks passed (6 reference
  blocks, 34 bytes/block, 2.125 bpw)", exit 0.
PASS.

CLAIM 3 - every weight row against the exporter dequantizer
- DS4_BONSAI_MODEL=/data/models/Ternary-Bonsai-2-27B-PQ2_0.gguf make
  test-qwen35-rows, re-run after the fix: "qwen35 rows: 851 tensors matched the
  reference dequantizer (first 64 rows each)", exit 0.
- Timing correction: this takes 7.4 s here, not the 0.9 s the commit message
  records. Correctness is unaffected.
- Fixture provenance, checked hard: tests/pq2_0/pq2_ref_generator.c is a real
  ggml generator (ggml_get_type_traits(type)->to_float over the mmapped tensor
  rows, FNV-1a over the f32, printing exactly the fixture's format). What the
  tree does NOT contain is the library it links (-I<fork>/ggml/include,
  -lggml-base): the PrismML fork is not vendored here and no pq2_ref binary and
  no build rule exist, so the fixture cannot be regenerated from this tree alone.
  I therefore reproduced it independently: misc/scratch/qa/gguf_plan.py parses
  the GGUF itself and misc/scratch/qa/pq2_fixture_check.c dequantizes with the
  fork's own PQ2_0 reference code, read from the fork at
  /data/llama.cpp-prism branch origin/prism-v7 (ggml-common.h block_pq2_0,
  QK_PQ2_0 128, GGML_TYPE_PQ2_0 = 142; ggml-quants.c dequantize_row_pq2_0,
  byte j/4, bits 2*(j%4), level (q-1)*d). Result: "fixture check: 851 tensors
  reproduced, 0 mismatch(es)" - every checksum bit-exact, every row sum within
  1e-3. So the fixture is exactly what the PrismML dequantizer produces on this
  file. Caveat: my checker shares the format reading with the fork by
  construction, but the checksums are exact, so a wrong layout, scale or level
  map would have shown.
PASS.

CLAIM 4 - CPU reference forward bit-identical to the sibling
- Re-run after the fix on the same five tokens
  (DS4_QWEN35_TOKENS="760,6511,314,9338,369", DS4_QWEN35_STEPS=8,
  DS4_QWEN35_LOGITS=<file>): cmp of this tree's dump against the sibling tree's
  dump (misc/scratch/qa/ref_logits.bin, generated from /data/ds4 at bbaf298 with
  ./ds4 -m <artifact> --cpu --first-token-test --raw -p "x") is byte-identical,
  4966400 bytes each, md5 00d3a420c88898fd86f3896164be9bc8 for both - the same
  hash as before the fix. Greedy stream: 11751 Paris, 13 ., 198, 760 The,
  6511 capital, 314 of, 9564 Germany, 369 is, with top-5 11751 14.2724,
  25 10.8312, 198 10.6854, 31586 10.4535, 248046 10.1684.
PASS.

CLAIM 5 - fold transform vs the explicit Sylvester matrix
- make cpu && make bonsai-fold-selftest, re-run after the fix: "fold selftest:
  blocks 2, 4 and 1024 match the explicit Hadamard matrix, blocks stay
  independent, forward/inverse round-trips, and the gdn permutation follows the
  tiled-to-grouped index map", exit 0. DS4_QWEN35_FOLD_SELFTEST=1 by hand gave
  the same line.
- The expectation is not circular: qwen35_hadamard_selftest (ds4.c 68830-68945)
  builds (-1)^popcount(row & col)/sqrt(n) per block and compares it against
  ds4_hadamard_rotate, then checks block isolation, forward/inverse round-trip
  and the explicit gdn index map h + hd*(r + rep*k).
PASS.

CLAIM 6 - CUDA PQ2_0 and fold kernels
- nvidia-smi before the run: RTX 4070 SUPER, no compute apps, so the card was
  free and only one GPU test ran at a time.
- After the fix, the test binary was force-rebuilt from source and re-run:
  exit 0, 49 PASS, 0 FAIL, ending "PQ2_0 CUDA parity: PASS".
  Highlights: PQ2_0 row lookup 8x5120 mismatches=0; host embed_token and
  embed_tokens bit-exact; matmuls rel_l2 <= 0.0042 against the 0.05 guard at
  17408x5120, 5120x17408, 5120x6144 and 6144x5120 including N=8/64/256, plus
  the exactly-representable-activation cases (MMQ max_abs 2.5e-05,
  MMVQ 1.0e-03); fold rotate/forward/inverse max_abs 0 at 5120, 6144, 17408 and
  1024 wide rows, round trip 5.4e-07, and the block-1024 forward against the
  explicit matrix at 5.4e-07.
- The test drives the real entries: ds4_gpu_matmul_pq2_0_tensor in
  test_host_wiring ("host matmul_quant" lines), ds4_gpu_embed_token_quant_tensor
  and ds4_gpu_embed_tokens_quant_tensor, and ds4_gpu_qwen35_fold_forward_tensor /
  ds4_gpu_qwen35_fold_inverse_tensor for the fold rows.
- Omissions are stated in the test file: the gdn output-norm gate pair and the
  Bonsai attention core are not ported (no equivalent kernel here yet).
PASS on what it claims to cover.

CLAIM 7 - the engine refuses anything but the CPU reference by name
- The gate is ds4.c:70091-70110, message "ds4: Bonsai (qwen35) runs only
  through the CPU reference: use --cpu --first-token-test (the graph backend is
  not implemented yet)". Verified live, exit 1 each, for the default CPU
  generation request, for --cuda, and for --cpu --power 90.
- This is the item that FAILED the previous QA pass: the same invocation used to
  SIGSEGV (exit 139) in layer_attention_raw_swa_batch at ds4.c:14527
  (attn_q_a_norm NULL on a gated delta-net layer) instead of being refused.
  d248d22 closes it; the probe now exits 1 with the named message and no crash.
PASS.

CLAIM 8 - C/Rust catalogue parity
- make test-catalog-parity, re-run after the fix: exit 0, 24 test-result lines,
  every one "ok", 0 failed and 0 panicked (full log at
  misc/scratch/qa/catalog-parity-after-fix.log).
- DS4_QWEN35_MODEL=/data/models/Ternary-Bonsai-2-27B-PQ2_0.gguf cargo test
  -p ds4-core --test layout: 13 passed, including
  validate_qwen35_artifact_when_configured. Negative control: the same test with
  DS4_QWEN35_MODEL pointing at the mmproj GGUF panics with UnsupportedArch, so
  the pass is real work, not a skip.
PASS.

SURFACE TOKENS (what was checked, what was observed)
- ds4_gpu_matmul_pq2_0_tensor: declared ds4_gpu.h:907, defined ds4_cuda.cu:25049; called live by tests/test_qwen35_cuda.cu:720 and passing at M=5120 K=6144 N=1 and N=4 (rel_l2 0.0037, guard 0.05).
- ds4_gpu_qwen35_fold_forward_tensor: declared ds4_gpu.h:922, implemented in ds4_qwen35_gpu.cuh:66; exercised live by the CUDA test (fold rotate/forward/inverse, max_abs 0 at 5120/6144/17408/1024).
- ds4_gpu_qwen35_fold_inverse_tensor: declared ds4_gpu.h:933, implemented in ds4_qwen35_gpu.cuh:75; exercised live in the same rows, round trip max_abs 4.77e-07.
- DS4_QWEN35_FOLD_SELFTEST: ds4.c:68967; ran live after the fix (make bonsai-fold-selftest and by hand), printed the selftest line, exit 0.
- DS4_QWEN35_LOGITS: ds4.c:68997; used live after the fix, wrote 5 x 248320 f32, byte-identical to the sibling tree's dump.
- DS4_QWEN35_STEPS: ds4.c:68985; used live with 0 and 8 after the fix (0 for the driver probe, 8 for the parity run); 8 greedy steps printed.
- DS4_QWEN35_TOKENS: ds4.c:68973; used live with "760,6511,314,9338,369" after the fix, same stream as the sibling.
- pq2-0-test: make target, ran green after the fix (6 reference blocks, 34 bytes/block, 2.125 bpw).
- test-qwen35-rows: make target, ran green after the fix, 851 tensors matched.
- test-qwen35-cuda: make target, rebuilt from source and ran green after the fix, 49 PASS / 0 FAIL.
- bonsai-fold-selftest: make target, ran green after the fix.
- bonsai-ref-check: make target, ran green after the fix (DS4_BONSAI_STEPS=8): prompt 25 tokens, top-1 760 21.9951, tokens 760 The, 1156 user, 369 is, 9859 asking, 264 a, 4145 simple, 57879 factual, 3296 question - the same stream as before the fix.
- pub const CTX_MAX: crates/ds4-core/src/qwen35.rs:20 = 262144. Checked against real metadata: qwen35.context_length 262144 read independently from the artifact, and config_validate_qwen35_model pins it to DS4_ROPE_ORIG_CTX (ds4.c:7204), which must match for the artifact to load.
- pub const FOLDABLE_SUFFIXES: qwen35.rs:36, 10 suffixes. Cross-checked against the fixture: of the artifact's 402 pq2_0 tensors, exactly 401 match the whitelist and the one that does not is token_embd.weight, the declared inverse-only name; the C twin qwen35_is_foldable_weight_name (ds4.c:6862) has the same 10 kinds plus the output.weight special case, and prism.hadamard coverage validation passes at load.
- pub const FULL_ATTN_INTERVAL: qwen35.rs:17 = 4; the artifact's qwen35.full_attention_interval is 4 (read independently and pinned by the validator), and 64 layers / 4 = 16 full-attention layers, matching SHAPE_QWEN35 n_full_attn_count 16.
- pub const LIN_CONV: qwen35.rs:15 = 4; artifact qwen35.ssm.conv_kernel = 4, pinned to DS4_N_SSM_CONV.
- pub const LIN_HEAD_DIM: qwen35.rs:13 = 128; artifact qwen35.ssm.state_size = 128, pinned to DS4_N_KDA_HEAD_DIM.
- pub const LIN_K_DIM: qwen35.rs:23 = LIN_K_HEAD * LIN_HEAD_DIM = 2048; consistent with the fixture's attn_qkv row width (2*2048 + 6144 = 10240 = the fixture's rows/10240).
- pub const LIN_K_HEAD: qwen35.rs:9 = 16; artifact qwen35.ssm.group_count = 16, pinned to DS4_N_LIN_K_HEAD.
- pub const LIN_V_DIM: qwen35.rs:25 = LIN_V_HEAD * LIN_HEAD_DIM = 6144; artifact qwen35.ssm.inner_size = 6144, pinned to DS4_N_LIN_V_HEAD * DS4_N_KDA_HEAD_DIM.
- pub const LIN_V_HEAD: qwen35.rs:11 = 48; artifact qwen35.ssm.time_step_rank = 48, pinned to DS4_N_LIN_V_HEAD.
- pub const SHAPE_QWEN35: crates/ds4-core/src/shape.rs:906 (64 layers, 5120 embd, 248320 vocab, 24/4 heads of 256, dense FFN 17408, rms 1e-6, rope base 1e7, rope_orig_ctx 262144, n_swa_period 4, n_full_attn_count 16, n_kda_head_dim 128, n_ssm_conv 4, no experts/MTP/hyper-connections). The artifact's own metadata matches every field this unit uses, the C arm DS4_SHAPE_QWEN35 was live-verified by the loading, row and parity tests, and the layout test resolves it against the artifact's 851 tensors.
- pub fn is_foldable_weight_name: qwen35.rs:50; same rule as the C function (output.weight whole, blk.<n>.<suffix> with the 10 suffixes), and the counts above agree with the artifact. No standalone tape asserts the two are equal, so the agreement is by source inspection of both, not by a test. Accepted, unverified-by-test.
- pub fn layer_is_full_attention: qwen35.rs:29, (il+1) % 4 == 0; the complement of ds4_qwen35_layer_is_linear ((il+1) % DS4_N_SWA_PERIOD != 0) with n_swa_period 4, giving layers 3,7,...,63. No standalone parity tape either; accepted, unverified-by-test.

MOUNTS ADDED BY 99ae2cf
- tests/qa-gate.sh and tests/run.sh are now tracked in the tree (the previous
  pass saw them as untracked files). tests/qa-gate.sh does not list them among
  the surfaces it greps for: its extractors cover ds4_gpu.h entries, DS4_*
  getenv knobs in ds4.c, Makefile targets and Rust pub items, and it prints the
  same 24 surfaces it printed before d248d22 (d248d22 touched ds4.c by one
  condition, ds4.h and ds4_cli.c, none of which adds a surface of those four
  classes). bash tests/qa-gate.sh on this report: every "report covers surface"
  check PASS. tests/run.sh chains pq2-0-test, test-catalog-parity and this gate.

NOT VERIFIED
- The fixture cannot be regenerated from anything inside this tree (no fork, no
  ggml, no build rule); it was verified by reimplementing the fork's published
  dequantizer instead, which reproduced all 851 tensors.
- The refusal is family-gated by ds4_model_is_qwen35(), so it cannot affect other
  families; I could not exercise that on this host because the only other GGUF
  here (Qwen3.8-27B-GSQ-RCO-IQ3_XXS) also declares general.architecture qwen35
  and is rejected at bind for an iq2_s token_embd.
- test-qwen35-cuda's binary is rebuilt by make -B, but the Makefile does not list
  ds4.h as a prerequisite of ds4_cuda.o. Harmless here (ds4_cuda.cu neither
  includes ds4.h nor uses ds4_engine_options), noted as build hygiene only.

verdict: overall PASS
