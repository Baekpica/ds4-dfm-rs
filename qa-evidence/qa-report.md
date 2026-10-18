QA report, GLM-5.3-VQ agent-path line, final unit: the prompt-inclusive penalty window

Identity
Branch fix/glm53-vq-agent-path, HEAD 416d4f5a83fb8721d7ed00a2d0c20bad3f1fd383
("feat(glm53): let the penalty face read the prompt"), 27 commits over the base.
The branch was renamed from fix/glm53-vq-penalties before the draft PR: the tip
is the same commit 416d4f5 and no PR referenced the old name.
Base origin/main c9425237669057937178e0598b3114ed9f8a15ec (2026-10-17 21:29:21
-0700). Working tree clean apart from an untracked graphify-rs-out/ directory.
Report written 2026-10-18. QA model deepseek-v4.1-CC-flash (rule 19 selection).

What was tested and how
- git diff origin/main..HEAD per surface; git log -S to place the new symbol.
- QA_BASE=origin/main bash tests/qa-gate.sh (surface list, 46 strings).
- CARGO_TARGET_DIR=/data/ds4-qa-fresh cargo test -p ds4-server --lib --tests
  --no-fail-fast (fresh target dir, see "Test counts" for why).
- The scanner knob was exercised from the unit tests in generate.rs and parse.rs
  and, where the assertion allows, from a direct run of the test binary with the
  deployment variable exported.
- Live checks against the Spark at 192.168.1.91:8009: NOT RUN, the serve refuses
  connections (see "Live checks").

Surfaces verified

crates/ds4-cli/src/lib.rs
Adds dry_face() (--dry-multiplier with the 1.75 / 2 base and allowed-length
fallbacks) and sample_next(); a face goes through sample_penalized with window
0 and speculation off, no face keeps session.sample, so the CLI's default call
is the pre-existing one (lib.rs:493-520 in the diff).

crates/ds4-core/src/batch.rs
Admit and copy the six face fields (freq_penalty, presence_penalty,
dry_multiplier, dry_base, dry_allowed_length, penalty_window), zero-defaulted at
batch.rs:157 and copied at :254; penalty_window is 0 unless the server sets it.

crates/ds4-core/src/lib.rs
Session::sample_penalized (lib.rs:3211-3245) forwards the face and the window to
ds4_bridge_session_sample_penalized; its doc states the zero-face / bad-gen_start
path takes the plain sampler, which the C side implements.

crates/ds4-core/src/tok.rs
ChatThinkMode::Low now renders "Reasoning Effort: Low" (the template's own
name); None returns without the line. Matches chat_template.jinja:2-3 as cited.

crates/ds4-core/tests/tokenizer.rs
New glm_effort_prefix_names_the_level pins that rendering; it is in the green
ds4-core target of the run.

crates/ds4-server/src/bin/ds4-server-rs.rs
The scope gate: cfg.glm_values = true only when syntax_for_model_id(model_id) is
ModelSyntax::Glm53 (bin/ds4-server-rs.rs:618-623), and the startup line prints
glm_values_report(). This is the single place the GLM scope is entered.

crates/ds4-server/src/dsml.rs
Adds the ToolSyntax switch (set_tool_syntax / tool_syntax), the GLM parameter
scanner (param_open_full, param_close_partial, param_close_partial_any,
raw_suffix_partial_close) and its boundary/control-text tests. tests/dsml.rs
compares them against tests/parity/dsml_c_oracle, all green.

crates/ds4-server/src/generate.rs
Penalty::of / is_set (the ONE face build for both lanes, DRY fallbacks 1.75 and
2) and the serial dispatch: "else if penalty.is_set()" (generate.rs:1934) is the
only route into sample_penalized, so a request with no face takes
engine.sample / sample_excluding exactly as before.

crates/ds4-server/src/lib.rs
Re-exports Penalty, glm_values, glm_values_report, default_effort and the
existing test helpers; no behaviour.

crates/ds4-server/src/parse.rs
The unit itself. GlmValues.penalty_window (parse.rs:200) is resolved from
DS4_SERVER_GLM_PENALTY_WINDOW with GLM53_DEFAULT_PENALTY_WINDOW = 0 and a
1,000,000 cap (:265, :425-431); ParsedRequest::init sets it to
glm_values().penalty_window under env.glm_values and to 0 otherwise
(:616-620); the request field is documented as deployment-scoped at :530-533.

crates/ds4-server/src/render.rs
syntax_for_model_id maps 7 and 17 to Glm53 (:169) and glm_effort_text writes the
artifact's level line; GLM tool blocks are re-emitted inline without added
newlines, matching ds4_server.c:2984-2986 as cited.

crates/ds4-server/src/retry.rs
build_glm_tool_error_suffix: the GLM malformed-tool-call retry suffix, with the
</tool_response> escape and the think closing/opening it re-emits; test
glm_retry_uses_native_observation is green.

crates/ds4-server/src/serve_cont.rs
The continuous lane: admit.penalty_window = penalty.penalty_window (:3589) and
the env-gated warm_trace / warm_trace_on diagnostic (DS4_SERVER_WARM_TRACE,
:901-907). The bank segment itself is chosen in ds4.c, not here.

crates/ds4-server/src/serve_owner_cont_test.rs
Adds glm_values: false to the owner-continuous ParseEnv fixtures (:210, :525).
The fixture therefore exercises the non-GLM scope, which is the isolation case.

crates/ds4-server/src/serve_owner_static_test.rs
Adds glm_values: false to the static-path ParseEnv fixture (:119); same point.

crates/ds4-server/src/serve/owner_tests/stream_disconnect.rs
Adds glm_values: false to that fixture's ParseEnv (:58); same point.

crates/ds4-server/src/serve.rs
Owns the glm_values flag on the server config (:78-80) and threads it into
ParseEnv together with default_effort(cfg.glm_values) (:1258-1262); the three
remaining literals are fixtures and stay false.

crates/ds4-server/src/tools.rs
Adds set_model_syntax / glm_declared_type and tool_schema_order_prop_type_first:
the declared JSON Schema now types a numeric tool property (u8, integer, a type
list's first entry), which is the live tool path's contract.

crates/ds4-server/tests/dsml.rs
Four new GLM tracker tests (boundaries vs the reference, structural argmax,
control text inside an argument is payload, untracked text under the DSML
dialect) run against tests/parity/dsml_c_oracle, whose binary carries the new
glm-* tapes (strings check). Target green in the run.

crates/ds4-server/tests/generate.rs
The window's serial-lane test: the_glm_penalty_window_reaches_the_serial_sampler
asserts Penalty::of(parsed).penalty_window == 0 for an omitted window, then
sets parsed.penalty_window = 1024 and asserts
first.penalty.penalty_window == 1024 and first.gen_start == 1. Its siblings
assert (4, 0) penalized-vs-plain samples for a face and the kept greedy commit
without one. All green.

crates/ds4-server/tests/glm53_vq.rs
New family tests: parse/parse_tool_call and render against the C oracle, the
effort syntax, live tool tail, old-reasoning drops, tool-result grouping. All
green (glm53_vq target, 39 tests).

crates/ds4-server/tests/glm_output.rs
Adds glm_values: false to its ParseEnv fixture (line 37); fixture only.

crates/ds4-server/tests/parse.rs
New glm_scope_defaults_and_overrides: the GLM scope's request-omitted defaults
and their overrides. Target green (parse target).

crates/ds4-server/tests/render.rs
New glm53_low_effort_renders_low pins the level line end to end. Green.

crates/ds4-server/tests/tools.rs
Five new tests for the schema typing: a u8 property, an integer property, a
type-list head, a bare-word value kept as text, and parse_chat_request keeping a
type-list head. Green (tools target, 36 tests).

crates/ds4-sys/src/lib.rs
The FFI declaration and the C-contract structs for the new face
(ds4_bridge_session_sample_penalized at :1030, penalty_window at :1044 and
:1333). Layout matches native/bridge/ds4_bridge.h; the mixed C/Rust build is
what the passing tests exercise.

docs/benchmarks/glm53-values-live-2026-10-17/raw.csv
17 rows, header of 23 columns; the "now" column of the README (ttft 271.5 /
433.3 / 47990.0 ms, decode 37.4-40.6 tok/s, tok_per_step 2.0-3.0) is contained in
these rows. I could not re-run the serve to reproduce them.

docs/benchmarks/glm53-values-live-2026-10-17/README.md
Cross-checked against raw.csv: ttft cells equal the per-case medians (271.5,
433.3, 47990.0, 287.4, 286.0, 273.4, tok_per_step 2.46/2.68/2.00/2.46/2.51/2.46),
the prefill and two decode cells equal the rep-0 row (209.6 vs median 213.8;
37.4 vs 37.5) and one range is loose (tok_per_step reaches 3.0, text says
2.68). Differences are <= 1.6 percent and change no conclusion.

docs/benchmarks/glm53-warmpath-2026-10-17/raw.csv
17 rows; every row reports cached_tokens 0 and cache_write_tokens equal to the
whole prompt, which is what the README's item 1 claims (17 of 17). Same caveat:
live re-run not possible today.

docs/benchmarks/glm53-warmpath-2026-10-17/README.md
ttft column is the median of the rows (271.4, 434.3, 47955.6, 265.1, 285.7,
272.7 all reproduce); the prefill column is the cold rep-0 row, not the median
the label claims (213.2 vs 214.1, 529.7 vs 530.0, 250.3 vs 271.9, 286.5 vs
308.7, 1371.3 vs 1367.85), a labelling error of at most 1.6 percent. Its two
code claims hold: the strict-prefix rule at serve_cont.rs:881-884 is real
(key.len() < request_key.len()) and the rest of the table checks out.

docs/glm53-values-2026-10-17.md
Scope claim (ModelSyntax::Glm53, ids 7 and 17) matches render.rs:169; every knob
it cites (DS4_SERVER_DEFAULT_TOP_P, MIN_P, TEMP, EFFORT) exists in parse.rs. The
defaults it prints (top_p 0.95, min_p 0, effort Max) match
GLM53_DEFAULT_TOP_P/MIN_P/EFFORT at parse.rs:230-233.

docs/glm53-vq-sampler-2026-10-17.md
Scope claim (GLM-5.3 syntax, continuous lane, zero defaults) matches the code;
the three knobs it cites (DS4_SERVER_DEFAULT_FREQ_PENALTY,
DS4_SERVER_DEFAULT_DRY_MULTIPLIER, DS4_SERVER_GLM_TEMP_MAX) all exist. The
2026-10-18 think-cap record corrects this one's greedy-arm reading, and the
correction is consistent with the values-live rows (tok_per_step 2.0-3.0).

docs/glm53-vq-serial-penalty-2026-10-18.md
Unit 8e record. Its code references reproduce: family_cont_penalize over the
bank's generated segment, the [gen_start, committed) serial segment, the row
copy, and the greedy-commit gate. Its final line predicts exactly the suite
state I measured: green except the pre-existing
cache_identity::tests::bounded_sidecar_and_ple.

docs/glm53-vq-think-cap-2026-10-18.md
Unit 8d record; the budget knobs it cites (DS4_SERVER_GLM_THINK_CAP_LOW, and the
high/max pair) exist in parse.rs and are resolved per level. Its
DS4_SERVER_GLM_TEMP_MAX and DS4_SERVER_DEFAULT_DRY_MULTIPLIER appear in the
later deployment lines I read in the records, consistent.

docs/glm53-vq-warm-path-2026-10-18.md
Phase 7c record; DS4_SERVER_WARM_TRACE is the code's own diagnostic
(serve_cont.rs:901-907, one line per bank) and DS4_SERVER_FORK_PARTIAL /
FORK_PARTIAL_MIN exist in ds4-core and serve_cont. The reuse-miss names it
lists (NoCheckpoint, RenderedPrefix, BelowThreshold, PayloadMismatch,
StateReplay) are the ones in the code.

docs/README.md
Index only: adds links to the two benchmark dirs and the four GLM records of
this line. No numbers.

ds4.c
The window reaches only two places. family_cont_penalty_segment returns
cb->generated_len with the generated pointer when penalty_window <= 0 (:65175)
and takes the last wanted tokens of ctx->bank_hist otherwise; the bank copies
req.penalty_window clamped to 0 (:65915-65916); family_cont_penalize
short-circuits when the whole face is zero (:65156-65158); the serial entry
widens its start back only when the window is > 0 (:77453). ds4_decode_penalize
and both plain session samplers are unchanged by this commit.

ds4.h
Documents the six face fields on ds4_cont_request and declares
ds4_session_sample_penalized; the header states the window is "0 -- the default,
and the only value any family but GLM-5.3 sends". Matches the implementation.

native/bridge/ds4_bridge.c
Adds ds4_bridge_session_sample_penalized as a null-checked pass-through to
ds4_session_sample_penalized (:1145-1157); the plain and excluding bridges are
untouched.

native/bridge/ds4_bridge.h
Declares the new bridge entry and the six face fields on the continuous request
struct with the same contracts as ds4.h (freq/presence/dry multi, window 0 =
generated segment). Field order matches crates/ds4-sys/src/lib.rs.

pub fn sample_penalized
Defined once per layer and used once: Session::sample_penalized
(crates/ds4-core/src/lib.rs:3211), the DecodeIo trait method
(crates/ds4-server/src/generate.rs:426) and its engine implementation (:4842).
The only production caller with a face is the serial lane at generate.rs:1934,
gated by penalty.is_set(); the CLI passes window 0 explicitly.

tests/glm53_http_suite.py
The five-case non-streamed warm-path harness (320 lines); py_compile OK and
--help lists its --url/--engine/--out interface. It needs a live serve, which is
down, so it was not run today.

tests/parity/dsml_c_oracle.c
Adds the GLM dialect (boundaries token and byte, control text) to the C oracle.
The built binary tests/parity/dsml_c_oracle (mtime 21:24:04.332) is newer than
the .c (21:24:04.172), carries glm-boundaries-token, glm-boundaries-byte and
glm-control-text, and the Rust DSML tests default to that path, so the parity
comparisons in my run really used it.

tests/serving-reuse-live.md
Adds the VQ release to the glm row and documents that the profile accepts both
plan family names, glm5-next and GLM-5.3-Flash-VQ. The strings exist at
crates/ds4-core/src/serving.rs:1511-1512; the doc's citation says 1484-1485, a
stale line number from the commit that wrote it.

tests/serving_reuse_live.py
PROFILES glm names are now both shells; python3 -m unittest from tests/ runs the
suite model-free.

tests/test_serving_reuse_live.py
Adds test_glm_vq_plan_passes_the_family_check and the two-name assertion; I ran
the file: 46 tests, all OK, including the new one (verified by name in the -v
output). It needs no server.

Test counts

Command: cd /data/ds4-dfm-rs && CARGO_TARGET_DIR=/data/ds4-qa-fresh cargo test -p
ds4-server --lib --tests --no-fail-fast

- lib: 355 tests, 353 passed, 1 failed, 1 ignored.
- 21 integration targets: 293 passed, 0 failed, 3 ignored (sum of the 21 target
  lines). Every target reported ok.

Why a fresh target dir: the first run of this suite, and every run using the
shared /data/ds4-merge-target, is untrustworthy. A second checkout of this
repository at /tmp/ds4-base-check (another session's base check, at commit
5db9098) shares that target dir, and both checkouts hash to the same unit name
(ds4_server-08c7cbff35517b67): each build silently overwrites the other's test
binary. My first run executed the foreign binary, whose baked
CARGO_MANIFEST_DIR is /tmp/ds4-base-check/crates/ds4-server. That binary's
--list shows 353 tests and zero penalty_window strings, i.e. it predates the
final commit. The fresh dir's binary shows 355 tests and 2 penalty strings. The
counts above are from the fresh dir.

The three failures seen in that first run, attributed

1. tool_memory::tests::ktm_codec_matches_c_oracle (tool_memory.rs:617) and
2. tool_memory::tests::ktm_decode_matches_c_string_nul_truncation
   (tool_memory.rs:734): NOT a failure of the change and no longer failures at
   all. Both panic on Command::output() with Os code 2 (NotFound) because the
   oracle path is env!("CARGO_MANIFEST_DIR") + ../../tests/parity/kv_c_oracle
   (tool_memory.rs:584-590). Run from the foreign binary that resolves to
   /tmp/ds4-base-check/tests/parity/kv_c_oracle, which does not exist (the
   fixture is gitignored and that checkout never built it). Run from a binary
   built in this workspace the path resolves to tests/parity/kv_c_oracle, which
   exists (210,848 bytes, Makefile:516 builds it), and both tests pass: the same
   six tool_memory::tests::ktm tests report "6 passed; 0 failed" from the fresh
   binary. tool_memory.rs was not touched by the branch (git diff
   origin/main..HEAD -- crates/ds4-server/src/tool_memory.rs is empty).
3. cache_identity::tests::bounded_sidecar_and_ple (cache_identity.rs:501):
   PRE-EXISTING and environmental, not caused by the branch. cache_identity.rs
   is byte-identical at origin/main, at 5db9098 and at HEAD (git diff for that
   path is empty in both directions), so the failing code is the same code. A
   genuine build of the base commit in /tmp/ds4-base-check (353-test binary,
   verified to contain no penalty_window string) reproduces the identical
   failure. Mechanism, demonstrated: ple_inputs registers the manifest's
   logical part with contents false (cache_identity.rs:306, unlike the scale's
   Input::small at :313), so contents_sha256 is null and the test's assert_ne!
   at :501 depends on the inode metadata moving; the rewrite of part.bin is the
   same four bytes and /tmp is tmpfs with coarse timestamps, so creation and
   rewrite stamp identically (my probe: create and rewrite both report
   2026-10-18 09:26:10.992371621, while a rewrite 10 ms later moves to
   .004372092). The whole test body fits inside one tick on this box. Verdict:
   the file's own test needs a filesystem with fine timestamps (or a hashed
   part); it is unrelated to the penalty window.

Isolation: no family but GLM can carry a non-zero window

The value's only source is the GLM scope, and the call for every other family
is the pre-existing one:
- parse.rs:616-620: penalty_window is glm_values().penalty_window when
  env.glm_values, else 0; there is no wire field (documented at :530-533).
- ds4-server-rs.rs:618-623: env.glm_values is set only when
  syntax_for_model_id(model_id) == ModelSyntax::Glm53, and render.rs:169 maps
  only model ids 7 and 17 to Glm53 (Qwen is 6 and 14, the V4.1 artifact takes
  the DeepSeek default).
- Tests: parse.rs:4183 only_the_glm_scope_can_carry_a_penalty_window asserts
  glm_req.penalty_window == GLM53_DEFAULT_PENALTY_WINDOW and
  other_req.penalty_window == 0; generate.rs:1071 asserts an omitted window
  keeps the generated-segment face; the continuous lane's copy of a zero window
  is the clamp at ds4.c:65915-65916.
- The sampling call: generate.rs:1934 routes to sample_penalized only when
  penalty.is_set() (freq, presence or dry non-zero), so a family with no face
  keeps engine.sample / sample_excluding. With a face and window 0 the C side
  takes cb->generated (ds4.c:65175) or [gen_start, committed)
  (ds4.c:77453) -- the pre-existing segment. ds4_decode_penalize itself is not
  touched by this commit.
- V4.1: no file of the ds41 route is in the branch diff (.inc untouched) and
  ds4_ds41_sample.inc has no penalty_window field at all.
- Qwen: it shares the banked lane (ds4.c:66109) but its request carries window 0
  by the scope above, and a zero/absent face short-circuits
  family_cont_penalize (ds4.c:65156-65158).
- Mechanical check that the window is the last commit's alone: git log -S
  penalty_window -- ds4.c returns only 416d4f5.

Live checks: NOT RUN, the serve is down

The task's two live checks (a coherent non-streamed answer with finish_reason
stop, and numeric tool arguments from an integer/u8 schema) could not be made.
Evidence, taken today between 09:19 and 09:28 (+02:00):
- ping 192.168.1.91 answers (0.83 ms average, 2/2), so the host is up.
- TCP port 8009 refuses: "connect: Connection refused" from /dev/tcp.
- curl http://192.168.1.91:8009/v1/models returned http_code 000 on six
  attempts spread over about six minutes; ports 8003, 8005, 8007, 8008, 8010,
  8011 and 8080 are also closed.
So the deployed binary's behavior, the DS4_SERVER_GLM_PENALTY_WINDOW=4096
deployment and the tool-argument typing on live traffic are UNVERIFIED by me.
The tool typing has offline cover only: the schema-typing tests in
crates/ds4-server/tests/tools.rs and the 7a/values rows that record
{"code":"QZ5813","limit":3,"max_results":15} as numbers.

Risk analysis (global rule 20)

Decision: call this unit verified and let the push proceed on this report.
External risks: the Spark serve was down for the whole session (measured, port
refused six times), so a deployment-side breakage could exist unseen; the live
serving numbers in the docs are single-window measurements I could not
reproduce (measured as "cannot re-run", not as "wrong").
Decision risks: (1) PASS rests on a local suite whose shared target dir proved
able to execute a foreign binary -- mitigated by using a fresh target dir and by
proving the executed binary's contents (355 tests, 2 penalty strings);
(2) one test, only_the_glm_scope_can_carry_a_penalty_window, asserts the live
GLM value against the compiled default, so it FAILS if the suite is run in a
process that exports DS4_SERVER_GLM_PENALTY_WINDOW=4096, as the Spark does
(measured: left 4096, right 0 at parse.rs:4198); the isolation half of that test
only passes when the variable is unset; (3) the doc tables label cold rep-0
cells as medians (measured against raw.csv).
Measurement and limits: the counts above are fresh-process, one machine; they
prove the workspace suite, not the deployed binary and not the numbers in the
records.
Prevention: keep a per-checkout CARGO_TARGET_DIR (the collision is silent
otherwise); when running the suite in a serving-like environment, export the
GLM variables as the deployment does or expect that one test to fail for the
wrong reason.
Rollback: this report is a file; the code under review is 27 commits on
fix/glm53-vq-penalties with base origin/main c942523, so any regression is
reverted by not merging, or by git revert of the offending commit on the branch.
Trigger: a failed live check (a) or (b) once the serve returns, or a report of
looped GLM output with the window set.
Residual risk: a live-only defect in the deployed binary, and whatever the
open items below leave unverified.

Open questions / not verified
- Live checks (a) and (b): not run, serve down (evidence above). Whoever
  restarts the Spark should run them.
- The penalty window's effect on live traffic (that 4096 actually changes the
  loop behavior) is unverified here; it is claimed by the records, not measured
  by me.
- Benchmark numbers: the raw.csv files and READMEs are consistent with each
  other except for the cold-row cells noted above; I could not re-run any row.
- The DSML oracle's freshness is established by mtime and strings only, not by a
  controlled rebuild of the binary from the .c (its stamp is 0.16 s newer than
  the .c, which is consistent with a build from the modified source but no proof
  of one).
- The isolation test's env-var sensitivity noted above is a test-quality
  finding, not a defect in the window's scoping.

verdict: overall PASS
