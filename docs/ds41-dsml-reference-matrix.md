# DS41-DSML reference coverage matrix

Status: draft for CTR revision R3 (the report folds this in whole as its
reference section). Prepared by ds4-dfm-rs work session, 2026-10-18; updated the
same evening at a2937a4c (O-01), 0bf8c79d (G8 and the repair half of G9),
54afd0bc (G6's builder) and 2debe05c (R-12's host half), which also adds the
`enc-think-budget` case.

This document answers one question: for every DSML requirement this project
has written down, which case exercises it, and what did that case actually
measure? Its rows come from the two maintained documents and nowhere else:

- `docs/ds41-dsml-conformance.md` (CTR R2) — requirements R-01..R-19 (clause
  6.3), nonconformities NC-01..NC-09 (clause 8), phases P0..P8 (clause 9);
- `docs/ds41-dsml-port-inventory.md` — gaps G1..G9 (section 4).

A row whose requirement has no case stays OPEN and blocks any "all covered"
claim. The leg's own findings — differences the oracle leg measured that
neither source lists yet — are rows O-01..O-04 in part D; they are findings,
not requirements, and each names whether it is a defect, a delta between the
two references, or a decision the operator owes.

## 1. The oracle

| Field | Value |
| --- | --- |
| File | `tests/ds41_ref/oracle/encoding.py` (vendored in this repo) |
| sha256 | `13231c592e958fb39e2af6001dae69d90e7a36b265702b9b390236fb387599e2` |
| Provenance | operator-supplied, read 2026-10-18 from `/data/ds4/test/encoding.py`; the vendored copy is byte-identical, and that external tree is no longer consulted |
| What it is | the official DeepSeek-V4.1 text/vision encoding implementation: the implementation that writes the prompt the model was trained on and asserts the completion shape it produces |
| Driver | `tests/ds41_ref/ref_drive.py` — `encode CASE MODE` prints the prompt bytes, `parse COMPLETION MODE` prints the parsed message; a hash other than the pinned one aborts the run |
| Inputs | `tests/ds41_ref/cases/*.json` (encode cases, the oracle's own case format), `tests/ds41_ref/decode/*.txt` (completion bytes, built from the oracle's templates by `tests/ds41_ref/mk_decode_input.py`) |
| Fixtures | `crates/ds4-server/tests/fixtures/ds41-ref/` — `<case>.prompt.txt` (oracle prompt bytes), `<case>.calls.json` (oracle parse), `<case>.txt` (the completion the oracle parsed). Regenerate with `bash tests/ds41_ref/regen_fixtures.sh` |
| Rust tests | `crates/ds4-server/tests/ref_encoding.rs` (encode side), `crates/ds4-server/tests/ref_decode.rs` (decode side) |

The oracle is the carriage's single authority. The CTR still quotes clause
text and line numbers taken from the C implementation that once served as a
second reference; those citations are historical quotations of the texts the
requirements were written from, not a live reference, and where the two
disagreed the disagreement is a delta row with both citations (O-01, O-03,
O-04). A row is decided by the oracle.

Traps measured on this host (python 3.13.5) while building the leg, recorded
so the next reader does not re-learn them:

- a case-level `tools` key is dropped unless the case goes through
  `load_cases`, which moves it into `messages[0]` (encoding.py:847-869);
- `thinking_mode` accepts only `chat` or `thinking` (encoding.py:596);
- the thinking-mode completion must not carry the opening `<think>` token:
  the oracle asserts that no special token appears in content or reasoning
  (encoding.py:982), so it takes the generation, not the prompt tail.

## 2. Part A — the oracle leg's cases

Encode side: our route renders the case, the fixture is the oracle's bytes
(`crates/ds4-server/tests/ref_encoding.rs`).

| Case | Pins | Verdict |
| --- | --- | --- |
| `enc-tools-chat` | the tools preamble and the per-schema line in chat mode (TOOLS_TEMPLATE, encoding.py:481-506) | pass |
| `enc-plain-chat` | the no-tools head: BOS, system, user, assistant content, EOS, and the assistant header before the last turn | pass |
| `enc-argtypes-chat` | the type attribute — `string="true"` for a string child, `string="false"` with the value in json.dumps shape otherwise (encode_arguments_to_dsml, encoding.py:524-556) — plus a merged tool result | pass |
| `enc-think-tools` | thinking mode with tools: the effort header at 75 and the open `<think>` the assistant turn must carry | pass |
| `enc-think-max` | the numeric effort header at 100 (encoding.py:443-472) | pass |
| `enc-think-low` | the same header at the low effort, which the oracle maps to 50 (encoding.py:448-453) | pass (closed 2026-10-18, `render.rs` `v41_effort_prefix` with `RequestedEffort`, `route/mod.rs`) |
| `enc-think-budget` | the numeric effort the client asked for, rendered as itself | pass (added with the low fix, same commit) |
| `enc-midsystem-chat` | a mid-conversation system message | pass (closed 2026-10-18, render.rs `pending_merge` + the in-place system branch) |
| `enc-toolresults-order` | tool results ordered by the assistant's call order (encoding.py:233-266) | pass (closed 2026-10-18, render.rs `tool_run_in_call_order`) |

Decode side: the completion bytes are the oracle's own, and our parser must
agree with the oracle's parse (`crates/ds4-server/tests/ref_decode.rs`).

| Case | Pins | Verdict |
| --- | --- | --- |
| `dec-one-call` | one invoke, one string and one integer child, prose before the block | pass |
| `dec-two-calls` | two invokes in one wrapper | pass |
| `dec-complex-args` | every JSON child kind, including a nested array and object | pass (values); the argument-string shape is O-04 |
| `dec-unicode` | a non-ASCII string child round-trips raw (the oracle's json.dumps is `ensure_ascii=False`) | pass |
| `dec-marker-in-value` | a block marker inside a parameter value is data, never structure (inventory G9, see part C) | pass |
| `dec-thinking` | the reasoning/prose split in thinking mode | pass |

## 3. Part B — CTR requirements R-01..R-18

"Case" names the artifact that exercises the requirement; "measured" means it
was run in the session that wrote the row, with the output quoted in the CTR's
evidence index or in the test itself.

| Row | Requirement (clause 6.3) | Case | Verdict at this revision |
| --- | --- | --- | --- |
| R-01 | OpenAI chat serves one DSML call as `tool_calls` | TP-01 / A-01 (`tests/spark/dsml_conformance.sh`) | pass at the served baseline (`iso-baseline-20261018-113636`); the claim is per-artifact, so the gate re-runs it |
| R-02 | typed parameters survive as JSON types | TP-02 / A-02 | pass at the baseline |
| R-03 | two invokes in one block | TP-03 / A-03 | pass at the baseline |
| R-04 | streaming carries the call | TP-04 / A-04 | pass at the baseline |
| R-05 | Anthropic serves `tool_use` | TP-05 / A-05 | pass at the baseline |
| R-06 | Responses serves `function_call` | TP-06 / A-06 | pass at the baseline |
| R-07 | chat turn 2 reuses the sampled block, G > 0 | B-01 | reserved: no case (CTR clause 3 proves the ceiling without the excluded KV work) |
| R-08 | replay survives a differently spelled client echo | B-02 | reserved as R-07 |
| R-09 | Anthropic turn 2 reuses the sampled block | B-03 | reserved as R-07 |
| R-10 | Responses continuation reuses the sampled call | B-04 | reserved as R-07 |
| R-11 | a cut block is not served as assistant content | C-01 (suite) + `tests/dsml.rs::live_tool_root_block_recovers_its_calls` (parser) + `tests/dsml.rs::tracker_enters_tool_mode_on_the_quoted_tool_root` and `tests/generate.rs::v41_live_tool_root_turn_streams_calls_without_markup` (tracker) | served FAIL before R6 at the gate (`conformance-eb67d493-20261019-065549`, C-01: `finish=stop calls=0 raw-DSML lines in content=2`), unchanged from the baseline (at R4 the same observed shape with `length`). Parsed-path pass stands (the parser half landed at 23eb6b2, the tracker half at d5ad35a), so the nonconformity is the missing invalid-DSML continuation, not the parse. R5 re-scopes it: the served route's own reference has no continuation by design — `server_generate_v41.c:10` lists it among the route's absent features and `:258-263` reports `error` for a cut block instead — so R-11's criterion describes the V4 pull route and closing it was an operator decision between recording the boundary and building the continuation as a named deviation; CTR 8.2 carries the reference position, and the operator decided at R6 to record the boundary: R-11 is struck as inapplicable to this surface, C-01 was re-scoped to what the route owes its client (`finish_reason=length`, no call fabricated from a cut block) and re-measured green at the tip (CTR 7.7) |
| R-12 | replay and trace diagnostics are available | `--trace` module test (`trace.rs::tests::a_section_carries_the_prompt_tokens_and_parsed_message`) and the usage text | measured pass at the gate: all three flags exist and every flag is advertised by the served binary (`--disable-exact-dsml-tool-replay`, `--tool-memory-max-ids`, `--trace FILE`, `bin/ds4-server-rs.rs:225-240`); `--trace` then captured 29 sections across one suite run, each with the rendered prompt, the prompt tokens and the parsed message (`conformance-eb67d493…`, traces in `trace-ab-20261019-071811/`). CTR NC-03 is closed |
| R-13 | plain-text KV reuse intact (protection) | TP-13 / D-01 | measured pass at the gate (second read cached 19) and at the baseline; re-run every phase |
| R-14 | determinism witness reproducible (protection) | TP-14 / D-02, witness sha16 `adf7157c8a5bbb4b` | measured pass at the gate with the same witness value as the baseline, on the newer 1017 artifact; re-run every phase |
| R-15 | exact replay reaches every chat-shaped surface | host: `generate.rs::dsml_replay::sampled_bytes_reach_the_prompt_on_every_chat_surface` (:7184); served: B-01..B-04 with `--trace` | served pass at the gate: the replay-ON and replay-OFF arms' turn-2 prompts differ on all four probes (chat 1762 vs 1800 bytes, Anthropic 1766 vs 1804, Responses 1762 vs 1800), and the OFF arm reproduces the R1 pre-replay token counts (396→403 chat/Responses, 400→407 Anthropic). The strict "verbatim against the raw completion" form is not directly assertable: the trace's generated-text block is empty on a tool-call turn (CTR 7.5 records the limitation) |
| R-16 | the replay path has a kill switch | host: `generate.rs::dsml_replay::the_disable_switch_leaves_the_canonical_render` (:7224); served: B-01 with the flag | measured pass at the gate: `--disable-exact-dsml-tool-replay` moves every B probe's turn-2 prompt, the canonical arm writing the parameter's `string="true"` attribute and the replay arm not; `tool_memory.rs:230` makes the store's `put` a no-op under `ExactReplay::Off` |
| R-17 | a cut turn keeps its truncation signal and a legal finish | C-01, C-02, C-03 (`tests/spark/dsml_conformance.sh:266-312`) | measured PASS at `9f441876` (the NC-07 unit, CTR 7.6): C-02/C-03 report budgets `8:length 16:length 24:length 32:tool_calls 48:tool_calls`, every value inside the client's enum (7.6 also records the SSE arm), and the compliance gate's truncated-block finish reads `length`. The R3 measurement (8/16/24 → `stop`) is what warranted the change. Mechanism at HEAD: `generate.rs:3240` pre-sets `length`, `:3323-3335` overwrote it with `error`, `:3398-3400` remaps `error` to `stop`; the R3 note's `:2548`/`:2612`/`:2615` are the serial path's same-shaped lines, which the served artifact never enters (corrected in CTR 8.7). C-01 is unchanged and still FAILs on its own criterion (R-11) |
| R-18 | the client of record completes a real tool loop | `tests/spark/open_grok_leg.sh` (G1/G2/G3) | measured pass at the gate (`client-eb67d493-20261019-070016`, 3/3): one-tool and two-tool turns complete under `/data/open-grok` against the served binary and the malformed turn stays inside the client's finish enum |
| R-19 | a user-supplied image placeholder in a textual field is refused (the oracle's `_validate_no_image_sp_tokens`, `encoding.py:355-367`) | served: the five probes of `…/probe-placeholder-04a0ab1f-20261019`; host: `tests/stray_image_token.rs` (9) | measured and CLOSED at R7 (`04a0ab1f`): at R6 the two text-only shapes answered `200` with the literal tokenizing to placeholder id `129264` in the prompt, the mixed case was refused for its count instead, and both controls answered `200`; the port now refuses the literal on every message surface with the oracle's own sentences, the controls stay `200`, and every R6 conformance row is byte-identical (CTR 7.8, 8.9). The engine does not implement the rule — `core_v41_vision.c:189-193` counts placeholders against images only and `server_parse_tools.c:290` writes the placeholder itself — so this row was settled by mirroring the oracle |

## 4. Part C — inventory gaps G1..G9

| Row | Gap (inventory section 4) | Case | Verdict at this revision |
| --- | --- | --- | --- |
| G1 | ds41 has no session, so no replay, tool memory, continuation or canonicalization | R-15/R-16 rows carry the replay half; the rest is reserved | partially covered: the host-side replay carriage landed (8891d3e) and `tool_replay_scope` is now `kind == Chat` (`generate.rs:1367`), so a session is no longer the carrier; canonicalization and the in-turn continuation stay reserved (CTR clause 9) |
| G2 | replay was OpenAI-chat-only, off under thinking and live ids | `generate.rs:1367-1369` + the per-surface host test (R-15) | closed for the carriage: the scope is the reference's rule and the attach is per surface; the served verdict rides R-15 |
| G3 | no tool-checkpoint canonicalization | none | out of scope for this line (CTR clause 1.3 excludes the KV store; clause 9 reserves it) |
| G4 | no invalid-DSML tool-error continuation | none yet | R5: not a port gap — the served v41 route's own reference documents the continuation as absent by design (`server_generate_v41.c:10`, `:258-263`, which needs a session). The repair half is P3 (unit 3.4); the continuation half was decided at R6 — the operator recorded the boundary (CTR 8.2), so this gap is closed as reference-absent, TP-11 is struck and no continuation is built |
| G5 | no `complete_tool_call_inside_thinking` equivalent | none | OPEN, unverified: an Anthropic live-path decision (NC-06), which the oracle leg does not reach |
| G6 | no `build_responses_visible_assistant_suffix` | `render.rs` `responses_visible_assistant_suffix` + `tests/render.rs::responses_visible_suffix_matches_client_replay`, the reference's own test (ds4_server.c:20386) | half closed 2026-10-18: the builder and its test exist; the store that consumes it (`responses_live_remember`, :14559) is the reserved checkpoint unit, so no caller lands here |
| G7 | the two replay flags were absent | R-12's flag check | closed 2026-10-19: both flags and `--trace` exist in the served binary and were exercised (R-12) |
| G8 | `try_repair_dsml` deviates: (a) closes unbalanced invokes/params, (b) omits the spaced spelling, (c) picks the style by presence not by earliest occurrence | `tests/retry.rs::dsml_repair_matches_the_reference_balance_contract` (the reference's self-test, ds4_server.c:19322) + the C-Rust tapes `repair-dsml`, `repair-dsml-param-open`, `repair-dsml-invoke-open`, `repair-dsml-spaced` (`tests/parity/retry_c_oracle.c`) | closed 2026-10-18 at 0bf8c79d: only a missing wrapper is supplied, the style comes from the earliest tool start (:6730-6745), and the spaced spelling is repaired; measured before the fix the truncated-parameter input came back with `</…parameter></…invoke></…tool_calls>` appended |
| G9 | no `find_tool_structural_text` counterpart | parse side: `dec-marker-in-value` (pass); streaming side: `stream.rs::text_stream_stray`; repair side: `retry.rs` `find_last_structural` with the balance test above | partially closed 2026-10-18: the structural scan is ported for the repair path (a marker inside a parameter body is not structure, and the think scope uses it); the parse-side wrapper-aware scan stays UNVERIFIED as covered |

## 5. Part D — the leg's own findings

| Row | Finding | Evidence | Class |
| --- | --- | --- | --- |
| O-01 | CLOSED 2026-10-18: the oracle renders the requested budget verbatim: a name maps to 50/75/100, an integer in [1,100] renders as itself, and either is written only in thinking mode at index 0 (encoding.py:443-473) | `enc-think-low`, `enc-think-budget`, `tests/parse.rs::requested_effort_budget_survives_the_engine_collapse` | pass at a2937a4c: `ParsedRequest::effort` (`RequestedEffort`) carries the requested mode beside the collapsed `think_mode`, the V4.1 head writes the number, and the engine's mode, reasoning cap and sampling switches do not move. Measured before the fix: low drew no line (fixture 1372 bytes against our 1311) and a numeric effort was refused at the parse |
| O-02 | CLOSED 2026-10-18: a mid-conversation system message writes the system token again at its position, introduces its own turn, and counts as user-like for the assistant header that follows it (encoding.py:621-627, :733-739, `find_last_user_index` :562-582) | `enc-midsystem-chat` | pass; measured before the fix, "Answer in one line." landed before the tools preamble instead of after the user turn. The fix added the in-place system branch and split `pending_merge` from `pending_assistant` in `render.rs`, so a user message following a mid system opens its own `<｜User｜>` exactly as the oracle's merge rule requires (encoding.py:184-231) |
| O-03 | CLOSED 2026-10-18: tool results are ordered by the preceding assistant's call order, not by arrival (encoding.py:233-266) | `enc-toolresults-order` | pass; measured before the fix, the oracle's block carried body a then body b and ours body b then body a, with the other 1767 bytes matching. `tool_run_in_call_order` (`render.rs`) sorts a run of tool-result messages by the ids the preceding assistant declared; a result whose id was never declared keeps the oracle's default key 0 and the sort is stable. This also makes the prompt canonical for a client that sends results in completion order, which the OpenAI wire permits (CTR clause 2.3.8) |
| O-04 | our call arguments minify a composite child (`tool_call_json_args_add`, `tools.rs:151-167`, mirroring `ds4_server.c:5971-5982` over `json_minify_raw_value`, `crates/ds4-server/src/json.rs:379`, called at `tools.rs:160`) while the oracle keeps the parameter body verbatim; the values are equal and both parse | `composite_arguments_are_minified_as_the_reference_does` (pass) | recorded delta, pinned from our side; no action |
| O-05 | CLOSED 2026-10-19 (R7): the oracle and the engine disagree about a user-supplied image placeholder in a textual field. The oracle refuses it (`encoding.py:355-367`, measured on this host: `ValueError` at `:359` for content and `:365` for `reasoning_content`); the engine accepts it, because `ds4_v41_images_prepare` compares the placeholder count against the image count only (`/data/YoungAi/src/core/core_v41_vision.c:189-193`) and the server writes the placeholder itself (`src/server/server_parse_tools.c:290`), with no user-input check anywhere in that tree | `…/probe-placeholder-498bd1a2-20261019` (the two `200`s), `…/probe-placeholder-04a0ab1f-20261019` (the two `400`s), `stray_image_token.rs` | decided by the operator at R7 in favour of the oracle, the carriage's declared authority (`tests/ds41_ref/README.md`), on the ground that the accepting path emitted prompt bytes no reference defines (the literal is placeholder token `129264` and nothing counts it, `generate.rs:3107`). The port mirrors the oracle; the divergence is recorded rather than carried, and the legacy `/v1/completions` `prompt` field — a pre-rendered prompt the client owns, which the oracle's rule does not address — keeps the engine's behavior |

## 6. Part E — the client of record's wire contract

The operator's goal for this line is stated as OpenAI compatibility with
`/data/open-grok`, the configured consumer, so the client's own wire types are
the second acceptance oracle beside the CTR.  Every row below was read out of
the client's source at `27b48a6` in this session; "satisfied by construction"
means our writer's branch was read, and the case is what measures it end to
end.

| Row | The client's requirement (source) | Our counterpart | Case | Verdict |
| --- | --- | --- | --- | --- |
| E-01 | `ChatCompletionResponse` requires `id`, `object`, `created`, `model`, `choices` (no serde default) — `xai-grok-sampling-types/src/types.rs:353-359` | final writer emits all five (`stream.rs:2355`), chunks emit id/object/chat.completion.chunk/created/model (`stream.rs:569, 645, 665, 686, 970, 1000, 1010`) | tests/stream.rs, tests/generate.rs, client leg | satisfied by construction |
| E-02 | `ChatChoice.index` and `ChatResponseMessage.role` are required; `Role` is `system|user|assistant|tool` (`types.rs:366-370`, `:273-278`, `:384-385`) | writers emit `index:0` and `role:"assistant"` | tests/stream.rs, client leg | satisfied by construction |
| E-03 | `finish_reason` is a CLOSED enum `stop|length|tool_calls|content_filter|function_call`; any other value fails that chunk's deserialization (`types.rs:373-381`) | the ds41 route remaps `error` to `stop` before every writer (`generate.rs:3398-3400`; the same remap, moved 9 lines by the NC-07 unit) | R-17's C-02/C-03 probes at the gate and again at `9f441876` | satisfied on the ds41 route, and `length` now survives to the wire: the enum carries a value the route can reach |
| E-04 | every tool delta needs `index`: `ToolCallDelta.index` has `#[serde(default)]`, so an omitted index collapses distinct calls onto 0 (`types.rs:495-499`; assembly `stream/chat_completions.rs:74-77`, `:199-231`) | every delta carries `"index":N` (`stream.rs:214`) | tests/stream.rs, tests/tool_stream.rs | satisfied by construction |
| E-05 | a call id is required: `ToolCallResponse.id` is a plain `String` (`types.rs:399-401`) | the writer falls back to a generated id when the parse left none (`stream.rs:179-184`, `:208-213`) | tests/stream.rs | satisfied by construction |
| E-06 | `function.name` and `function.arguments` are required strings, arguments a single string (`types.rs:406-409`) | both always emitted, arguments as one JSON object string | tests/stream.rs | satisfied by construction |
| E-07 | streaming is read as SSE `data:` chunks with a tool-call accumulator and a `[DONE]` terminator (`stream/chat_completions.rs:120-260`) | our SSE framing (`stream.rs:569` onward) | the client leg (G1/G2) | OPEN until the client leg runs (unit 3.3) |
| E-08 | reasoning is read from `reasoning_content` only, and the client synthesizes a reasoning sibling only when it carried text (`types.rs:384-396`; `stream/chat_completions.rs:262-266`) | we emit `reasoning_content` on the reasoning channel (`stream.rs:824`, `:867`, `:2365`) | R-15/R-16 host legs; the client leg | partially measured: the host legs pass, the served leg is open |
| E-09 | in thinking mode the client echoes `reasoning_content` on EVERY assistant message, empty included, and replays `function.arguments` as the raw string it holds (invalid JSON replaced by `{}`) (`conversation.rs:3727-3741`, `:3516-3527`) | our parse reads both fields and does not require reasoning; unknown keys are skipped | R-15's host leg (four surfaces parse the client's own echo) | host pass |
| E-10 | tool results are appended in completion order with `parallel_tool_calls = true`, so the server receives them out of call order | `tool_run_in_call_order` re-sorts a run of results by the assistant's declared call order | enc-toolresults-order | closed 2026-10-18: the oracle's order is applied, so a client that sends results in completion order still renders the canonical prompt (O-03) |

## 7. What blocks an "all covered" claim

After the 2026-10-19 gate the served legs of E-07/E-08 with R-18, R-12, R-15,
R-16 and R-17 are measured, so what remains OPEN is: G5 (the Anthropic live
path, the one row with no unit behind it), G2's served restatement (closed for
the carriage, but its row is scored by R-15, now pass), and the one question the
gate's measurements raise -- whether R-11's cut-block criterion is recorded as
reference-absent on the served route or met by building the continuation as a
named deviation (CTR 8.2, R5). The truncation-signal reclassification R-17 measured
reachable landed at `9f441876` (CTR 7.6, 8.7). The stray image placeholder in a
textual field (R-19) was measured at R7 and closed on the spot by `04a0ab1f`,
mirroring the oracle (CTR 8.9, part D row O-05). R-07..R-10 with G1's remainder, G3 and G6's caller
are reserved to a unit that may alter KV state. Three parts of the oracle's
surface had no case at all; all three now have one, landed 2026-10-19:
`image_refusals_cover_the_oracle_path_and_url_forms` pins the base64-only
refusals (the OpenAI bare-path and string spellings, the Responses `input_image`
URL and the Anthropic `url` source, alongside the two that already existed);
`accepted_response_format_is_not_a_render_input` pins that the accepted
`type:"text"` value reaches no render input, since the oracle's
`response_format_template` (encoding.py:64-66, :628-629) belongs to the encoder
and neither reference server reads the field; and
`chat_tool_names_pass_through_verbatim` pins the two tool-name forms this server
sees (a qualified name taken as given, a tool-level `namespace` key not folded,
which is the oracle's encoder rule at encoding.py:88-97, :116, while both
reference servers handle namespaces on the Responses surface only). The
oracle's image-placeholder validations in a textual field
(`_validate_no_image_sp_tokens`, encoding.py:355-367), which fire at prompt
expansion and therefore needed the served artifact, were probed at R7: the
artifact accepted them, the operator decided to mirror the oracle, and R-19
carries the closure.

Covered today, measured at the gate or in the sessions that landed the units:
R-01..R-06, R-11 at host level and measured FAIL at the served level (re-scoped
by R5, see its row), R-12,
R-13, R-14, R-15 and R-16 on both legs, R-17 measured pass at 9f441876 (both
halves, CTR 7.6), R-18, R-19 measured and closed at R7 (`04a0ab1f`, CTR 7.8/8.9),
E-01..E-06 and E-09 by construction and host test, G8 and the
repair half of G9, G6's builder, O-01, and every part A case marked pass.

For the operator's stated goal -- a plain OpenAI server any compatible caller
can drive, the way a hosted DeepSeek endpoint is driven -- E-07 and E-08 are
now measured, not inferred: the client of record completed a one-tool and a
two-tool loop against the served binary at `eb67d493` and did not fail the
stream on the malformed turn.
