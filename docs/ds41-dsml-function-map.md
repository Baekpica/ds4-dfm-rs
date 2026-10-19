# DS41-DSML function map

One row per function and per named constant of the DeepSeek-V4.1 encoding
oracle, then one row per DSML function of ours that the oracle does not
define.  This is the review list: what the oracle decides, where this repo
implements it, which case proves it, and what is still open.

The oracle is the official V4.1 text/vision encoding implementation,
operator-supplied 2026-10-18.  It is the authority for the carriage this
project serves; the DSML line serves no other markup for the V4.1 family.
It is vendored byte-identical at `tests/ds41_ref/oracle/encoding.py`
(sha256 `13231c592e958fb39e2af6001dae69d90e7a36b265702b9b390236fb387599e2`,
provenance in `tests/ds41_ref/oracle/README.md`) and pinned by that hash in
`tests/ds41_ref/ref_drive.py:39`, so a later edit of it fails the run loudly.
Line numbers below are into that vendored file.

## How to read a row

- Oracle: the function or constant, with its line in the oracle file.
- Decides: the one thing it fixes about the bytes.
- Ours: the counterpart in this repo, `file:line`.  "not located in this
  pass" means this session did not find a counterpart; it is a gap, not a
  claim that none exists.
- Case: the reference case that exercises it, under `tests/ds41_ref/`, with
  its fixture under `crates/ds4-server/tests/fixtures/ds41-ref/`.
- Verdict: `pass` (measured byte-identical to the oracle's fixture today),
  `OPEN` (a case exists or is owed and the bytes differ), `no case` (no case
  yet; nothing measured), `harness` (test-side only, not a served rule),
  `out of surface` (defined by the oracle but outside the OpenAI chat wire).

Verdicts are measured, not assumed: `pass` for the encode side means our
prompt bytes equal the oracle's prompt bytes for that case
(`crates/ds4-server/tests/ref_encoding.rs`), and for the decode side that our
parse yields the oracle's call names and argument values
(`crates/ds4-server/tests/ref_decode.rs`).

## 1. Encode side — request to prompt

| Oracle | Decides | Ours | Case | Verdict |
| --- | --- | --- | --- | --- |
| `to_json` (76) | the JSON literal shape for non-string values: `ensure_ascii=False`, python separators | `crates/ds4-server/src/render.rs:295` `pyjson_write`, `:288` `pyjson_raw_value`, `crates/ds4-server/src/json.rs:379` `json_minify_raw_value` | `enc-argtypes-chat`, `dec-unicode` | pass |
| `tools_from_openai_format` (84) | the schema string handed to the model, in the client's declared order | `crates/ds4-server/src/parse.rs:2222` `parse_tools_value`, `:1828` `openai_function_schema_from_tool`, `:2038` `tool_schema_orders_add_json_wire` | `enc-tools-chat` | pass |
| `_split_tool_name` (101) | the `namespace::name` split, and the two asserts around it | `crates/ds4-server/src/parse.rs:2099`, `:2140` (Responses path); no chat-path rule located | none | OPEN, no case: an OpenAI `namespace` field on a chat tool, and a name containing `::`, are both unmeasured |
| `_tool_name_for_encoding` (116) | the name printed in the invoke tag, namespace-qualified | `crates/ds4-server/src/render.rs:535` `append_dsml_tool_calls_text` | `enc-argtypes-chat` (bare name) | pass for the bare name; the qualified form has no case |
| `tool_calls_from_openai_format` (124) | a replayed assistant's `tool_calls` array into the internal call shape | `crates/ds4-server/src/parse.rs:941` `parse_tool_calls_value`, `:885` `add_tool_call_id` | `enc-argtypes-chat` | pass |
| `decode_dsml_to_arguments` (156) | the arguments JSON string: `string="true"` wraps the body as a JSON string, `string="false"` takes the body as the value | `crates/ds4-server/src/tools.rs:376` `parse_dsml_generated`, `:1271`, `crates/ds4-server/src/json.rs:379` | `dec-one-call`, `dec-complex-args` | pass on names and values; our bytes are minified where the oracle keeps the body verbatim (matrix O-04) |
| `merge_tool_messages` (184) | a `tool` role becomes a `tool_result` block inside a user message, merging into an adjacent merged user | `crates/ds4-server/src/parse.rs:1178` `parse_messages`, `:1252` `append_tool_result_text`; `crates/ds4-server/src/render.rs:1462` | `enc-argtypes-chat` | pass |
| `sort_tool_results_by_call_order` (233) | tool_result blocks inside one user message are ordered by the preceding assistant's call order | `crates/ds4-server/src/render.rs:1390` `tool_run_in_call_order` with `:1369` `preceding_call_order` | `enc-toolresults-order` | pass (closed 2026-10-18) |
| `parse_tagged_text` (273) | the `<image>path</image>` text form expands to image blocks, and a malformed tag raises | not located (`<image>` appears nowhere in `crates/ds4-server/src`) | none | OPEN, no case |
| `_is_image_block` (300) | which content blocks count as images (`image`, `image_url`) | `crates/ds4-server/src/parse.rs:996` `parse_chat_image_url`, `:443` `add_image_data_uri` | none | no case: the reference harness asserts text-only |
| `_extract_image` (305) | the url/data/source normalization, and the refusal when none is present | `crates/ds4-server/src/parse.rs:402` `add_image_base64`, `:443` | none | no case |
| `_process_image_blocks` (323) | images are replaced by `<｜deepseek_image｜>` in place, nested tool-result images included, and the records keep prompt order | `crates/ds4-server/src/render.rs:1336` `append_dsml_user_content`, `crates/ds4-server/src/parse.rs:592` `write_tool_image_placeholders` | none | no case on the reference leg; the vision gate exercises the served path |
| `_validate_no_image_sp_tokens` (355) | a user-supplied image placeholder in `content` or `reasoning_content` is refused | `crates/ds4-server/src/parse.rs:560` `reject_stray_image_in_content`, `:571` `reject_stray_image_in_reasoning`, called from every place the client's own text is read (message content, text blocks, a tool result's text, `reasoning_content`, the Responses text blocks) | `stray_image_token.rs` (9) and the served probes `…/probe-placeholder-04a0ab1f-20261019` | CLOSED at R7 (04a0ab1f): the oracle's two sentences on the wire; the engine does not implement the rule (`core_v41_vision.c:189-193`, `server_parse_tools.c:290`), so the port mirrors the oracle, CTR 8.9 |
| `process_image_messages` (370) | the per-message image pass, and `content` rebuilt from text blocks with `\n\n` | `crates/ds4-server/src/parse.rs:1029` `parse_openai_message_content`, `:546` `append_text_part` | none | no case |
| `_read_until_stop` (399) | the earliest-of-stops scan the whole parser is built on | `crates/ds4-server/src/tools.rs:169` `split_reasoning_content` | every `dec-*` case | pass |
| `find_last_user_index` (562) | the last user-like index; a mid-conversation system message counts as one; a tool-result-only user message does not start a turn | `crates/ds4-server/src/render.rs:1397`-`:1429`, `:222` `role_is_user_like` | `enc-plain-chat`, `enc-midsystem-chat` | pass, mid-system half included (closed 2026-10-18); the tool-result-only half has no case |
| `render_reasoning_effort` (456) | the effort header: thinking mode, index 0 only, `low/high/max` mapped to 50/75/100, an integer in [1,100] taken as itself, `None` to high | `crates/ds4-server/src/render.rs` `head_effort_prefix`/`v41_effort_prefix` with `crates/ds4-server/src/route/mod.rs` `RequestedEffort` | `enc-think-low`, `enc-think-max`, `enc-think-budget` | pass (closed 2026-10-18) |
| `render_tools` (509) | the tools preamble verbatim, with the schema list appended | `crates/ds4-server/src/render.rs:412` `append_dsml_tools_prompt_text` | `enc-tools-chat` | pass |
| `encode_arguments_to_dsml` (524) | one parameter element per argument key, in the call's key order, `string="true"` for a string child and its raw form otherwise; a non-dict `arguments` falls back to a single `arguments` key, a JSON string is decoded twice | `crates/ds4-server/src/render.rs:349` `append_dsml_arg`, `:375` `append_dsml_arguments_from_json` | `enc-argtypes-chat` | pass for a dict; the JSON-string and double-encoded fallbacks have no case |
| `render_message` (585) | one message's bytes, the role branches, the transition tokens, and the assistant header rule | `crates/ds4-server/src/render.rs:1432` `render_dsml_chat_choice_ex` | 8 encode cases | pass on 7 of 8; O-01 is the one rule still open, and O-02/O-03's rules closed 2026-10-18 |
| `_drop_thinking_messages` (749) | before the last user, an assistant's `reasoning_content` is dropped, the message kept | not located in this revision: no drop-thinking rule exists in `crates/ds4-server/src/render.rs` (neither `drop_thinking` nor `reasoning_content` appears there) | none | no case: no case carries a historical reasoning behind the last user |
| `_encode_messages_text` (770) | BOS, the loop over messages, and `effective_drop_thinking` forced false when any message carries tools | `crates/ds4-server/src/render.rs:1432` (the entry) with its message loop at `:1456` | all encode cases | pass |
| `encode_messages` (810) | the public entry: merge, sort, image pass, then the text render | `crates/ds4-server/src/render.rs:1432` behind the chat route | all encode cases | pass on the cases measured |
| `load_cases` (847) | the harness entry: a case-level `tools` key moves into `messages[0]`, a bare message list is wrapped | `tests/ds41_ref/ref_drive.py:61` | every encode case | harness |
| `encode_case` (872) | the harness entry: encode one case and return its images, case `thinking_mode` winning over the command line | `tests/ds41_ref/ref_drive.py:63` | every encode case | harness |

## 2. Decode side — completion to assistant message

| Oracle | Decides | Ours | Case | Verdict |
| --- | --- | --- | --- | --- |
| `parse_tool_calls` (890) | the invoke/parameter grammar: `>\n` after every open tag, one `name` per invoke, the type attribute, duplicate parameters refused | `crates/ds4-server/src/tools.rs:376` `parse_dsml_generated` | `dec-one-call`, `dec-two-calls`, `dec-complex-args`, `dec-unicode`, `dec-marker-in-value` | pass |
| `parse_message_from_completion_text` (946) | the thinking split at `</think>`, content up to EOS or the block start, no content after the block, no special token left in content or reasoning | `crates/ds4-server/src/tools.rs:376`, `:169` `split_reasoning_content` | every `dec-*` case, incl. `dec-thinking` | pass |
| `tool_calls_to_openai_format` (139) | the wire shape: `type: "function"`, `function.name`, `function.arguments`, `namespace` when present | `crates/ds4-server/src/stream.rs:173` `append_tool_calls_json` | every `dec-*` case (the fixture is the oracle's parse) | pass on names and values |

## 3. Constants and templates

Every literal the two sides must agree on.  Where a row says "other
families", the constant sits beside ours in the same table but belongs to a
model family outside this line.

| Oracle | Ours | Verdict |
| --- | --- | --- |
| `bos_token` (29) | `crates/ds4-server/src/render.rs:78` `DSML_BOS` | pass |
| `eos_token` (30) | `:81` `DSML_EOS` | pass |
| `thinking_start_token` (31) | `:86` `DSML_THINK_START` | pass |
| `thinking_end_token` (32) | `:87` `DSML_THINK_END` | pass |
| `dsml_token` (33) `｜DSML｜` | folded into the tags below | pass |
| `USER_SP_TOKEN` (35) | `:79` `DSML_USER` | pass |
| `ASSISTANT_SP_TOKEN` (36) | `:80` `DSML_ASSISTANT` | pass |
| `SYSTEM_SP_TOKEN` (426) | `:53` `DSML_SYSTEM_TOKEN` | pass |
| `LATEST_REMINDER_SP_TOKEN` (37) | not located | out of surface: the role is not on the OpenAI wire |
| `IMAGE_PLACEHOLDER` (39) | `:92` `DSML_IMAGE` | no case |
| `IMAGE_TAG_PATTERN` (40) | not located | OPEN, no case |
| `DS_TASK_SP_TOKENS` / `VALID_TASKS` (43, 51) | not located | out of surface: internal classification tasks |
| `assistant_msg_template` / `wo_eos` (60, 61) | `render.rs:1432` (EOS always written) | no case for `wo_eos` |
| `response_format_template` (64) | `crates/ds4-server/src/format.rs:19` `parse_output_format_value`, called from `crates/ds4-server/src/parse.rs:3075`; no render-side counterpart located | OPEN, no case |
| `tool_output_template` (68) | `crates/ds4-server/src/render.rs:1650` (`<tool_result>`, escaped by `:1316` `append_tool_result_text`) | pass via `enc-argtypes-chat` |
| `tool_calls_block_name` (428) `" calls"` | `crates/ds4-server/src/tools.rs:44` `DSML_TOOL_CALLS_START_SPACED` | pass |
| `tool_call_tag_name` (429) `" invoke"` | `:46` `DSML_INVOKE_START_SPACED`, `:47` end | pass |
| `tool_parameter_tag_name` (430) `" parameter"` | `:48` `DSML_PARAM_START_SPACED`, `:49` end | pass |
| `tool_call_template` / `tool_calls_template` (432, 435) | `crates/ds4-server/src/render.rs:535`-`:560` | pass |
| `REASONING_EFFORT_TEMPLATE` (443) | the line shape | `crates/ds4-server/src/render.rs` `V41_EFFORT_TEMPLATE` with the 75/100 constants | `enc-think-low`, `enc-think-max`, `enc-think-budget` | pass |
| `REASONING_EFFORT_MAPPINGS` / `DEFAULT_REASONING_EFFORT` (448, 453) | low 50, high 75, max 100, `None` to high | `crates/ds4-server/src/route/mod.rs` `mode_budget`/`RequestedEffort::head_budget` | `enc-think-low`, `enc-think-budget` | pass (closed 2026-10-18) |

## 4. Ours with no oracle counterpart

Functions of this repo's DSML line that the oracle does not define.  They are
kept because a served turn needs them, not because the oracle asks for them.

| Ours | Why it exists |
| --- | --- |
| `crates/ds4-server/src/tools.rs:2155` `find_tool_start` (with `crates/ds4-server/src/dsml.rs:389`) | the streaming tracker: the artifact's root spelling (quoted, unquoted, short) is not the oracle's canonical one, so the scan is tolerant where the oracle's grammar refuses |
| `crates/ds4-server/src/dsml.rs:550` `dsml_decode_state_for_text` | the incremental decode state machine that decides when markup is withheld from the stream |
| `crates/ds4-server/src/tools.rs` `DSML_*_SHORT`, `DSML_*` (25-36) | the two older root spellings the artifact still emits; the oracle knows only the spaced one |
| `crates/ds4-server/src/retry.rs:137` `try_repair_dsml` | truncated-call repair; the oracle has no repair path (matrix G8, P3) |
| `crates/ds4-server/src/render.rs` `responses_visible_assistant_suffix` | the visible assistant tail a Responses client can replay (ds4_server.c:12821); the store that consumes it (`responses_live_remember` :14559) is the reserved checkpoint unit |
| `crates/ds4-server/src/trace.rs` | C `--trace FILE` (ds4_help.c:323): one section per request with the prompt, its token ids and the parsed message (unit 3.2, TP-12/TP-15) |
| `crates/ds4-server/src/retry.rs` `find_any_tool_start`, `find_last_structural` | the repair path's style choice and think scope (ds4_server.c:5842, :5890), ported with G8 |
| `crates/ds4-server/src/render.rs:93`-`:138`, other-family markers | Motif, EXAONE, Solar, Qwen, GLM, K2 tag sets: other families, outside this line |
| `crates/ds4-server/src/render.rs:161` `syntax_for_model_id` | family dispatch; the oracle covers V4.1 only |

## 5. The open rows

Ordered by what unblocks them, with the evidence each needs.

1. The image surface (rows `parse_tagged_text`, `_is_image_block`,
   `_extract_image`, `_process_image_blocks`, `_validate_no_image_sp_tokens`,
   `process_image_messages`): six functions, no case.  The reference harness
   asserts text-only, so nothing here is measured against the oracle.  The
   server's base64-only rule is a deliberate deviation from the oracle's
   path and URL forms (AGENTS.md, the carriage section), so the cases owed here
   are the refusals.
3. `response_format` rendering (row `response_format_template`): parsed on our
   side, not rendered, no case.
4. The namespace surface (rows `_split_tool_name`, `_tool_name_for_encoding`):
   the qualified name has no case on either side.
5. O-04 arguments bytes: ours minify, the oracle keeps the parameter body
   verbatim.  Values agree; the strings differ.  Recorded, not a defect
   against the wire, since the client echoes our string.

Closed since the map was written: O-01 (the effort budget, both halves: the
requested effort now reaches the V4.1 head and a numeric `reasoning_effort` is
accepted), O-02 (a mid-conversation system message renders in place), O-03 (tool
results ordered by call order) and the repair rows of G8/G9.  Every one of them
was measured against its fixture before the fix and after it.

## 6. Reproducing

```bash
# the oracle's own bytes for a case, and its parse of a completion
python3 tests/ds41_ref/ref_drive.py encode tests/ds41_ref/cases/enc-plain-chat.json chat
python3 tests/ds41_ref/ref_drive.py parse  tests/ds41_ref/decode/dec-one-call.txt chat

# the two reference legs against the pinned fixtures
cargo test -p ds4-server --test ref_encoding
cargo test -p ds4-server --test ref_decode
```

Regenerate the fixtures only when the oracle or a case changes:
`bash tests/ds41_ref/regen_fixtures.sh`.
