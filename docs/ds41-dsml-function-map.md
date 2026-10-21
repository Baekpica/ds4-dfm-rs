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
`tests/ds41_ref/ref_drive.py:46`, so a later edit of it fails the run loudly.
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

Every `file:line` in this document was re-verified against the working tree on
2026-10-20 (each symbol grepped in its file); the earlier revision of this map
had been written against older files and carried a uniform 40-line offset on
`render.rs` and a varying one on `parse.rs`.

## 1. Encode side — request to prompt

| Oracle | Decides | Ours | Case | Verdict |
| --- | --- | --- | --- | --- |
| `to_json` (76) | the JSON literal shape for non-string values: `ensure_ascii=False`, python separators | `crates/ds4-server/src/render.rs:336` `pyjson_write`, `:329` `pyjson_raw_value`, `crates/ds4-server/src/json.rs:379` `json_minify_raw_value` | `enc-argtypes-chat`, `dec-unicode` | pass |
| `tools_from_openai_format` (84) | the schema string handed to the model, in the client's declared order | `crates/ds4-server/src/parse.rs:2405` `parse_tools_value`, `:2011` `openai_function_schema_from_tool`, `:2221` `tool_schema_orders_add_json_wire` | `enc-tools-chat` | pass |
| `_split_tool_name` (101) | the `namespace::name` split, and the two asserts around it | not implemented on the chat surface: a tool-level `namespace` key is not read there and a `::`-qualified name is taken as given (`crates/ds4-server/src/parse.rs:2405`); the Responses surface keeps a namespace for `type:"namespace"` items (`:2282` `responses_namespace_function_schema`, `:2336` `append_responses_namespace_tool_schemas`) | `tests/parse.rs` `chat_tool_names_pass_through_verbatim`, `responses_namespace_tools_keep_both_spellings` | out of surface on the chat wire the oracle decides: the client's model-facing tool spec is name/description/parameters only (`conversation.rs:861-869`, `From<ToolDefinition> for ToolSpec` `:1039-1045`), and its `namespace` rides the editor-facing ToolCall meta instead (`acp_session.rs:1741` asserts the meta, not the wire), so no chat request can carry the field; the oracle's `namespace::name` qualification belongs to the message dicts it is handed |
| `_tool_name_for_encoding` (116) | the name printed in the invoke tag, namespace-qualified | `crates/ds4-server/src/render.rs:576` `append_dsml_tool_calls_text` | `enc-argtypes-chat` (bare name) | pass for the bare name; the qualified form is out of surface with the row above |
| `tool_calls_from_openai_format` (124) | a replayed assistant's `tool_calls` array into the internal call shape | `crates/ds4-server/src/parse.rs:1079` `parse_tool_calls_value`, `:1023` `add_tool_call_id` | `enc-argtypes-chat`, `enc-args-double` | pass |
| `decode_dsml_to_arguments` (156) | the arguments JSON string: `string="true"` wraps the body as a JSON string, `string="false"` takes the body as the value | `crates/ds4-server/src/tools.rs:376` `parse_dsml_generated`, `:1262` `json_value_is_complete`, `crates/ds4-server/src/json.rs:379` | `dec-one-call`, `dec-complex-args` | pass on names and values; our bytes are minified where the oracle keeps the body verbatim (matrix O-04) |
| `merge_tool_messages` (184) | a `tool` role becomes a `tool_result` block inside a user message, merging into an adjacent merged user | `crates/ds4-server/src/parse.rs:1330` `parse_messages`, `:1411` `append_tool_result_text`; `crates/ds4-server/src/render.rs:1484` | `enc-argtypes-chat` | pass |
| `sort_tool_results_by_call_order` (233) | tool_result blocks inside one user message are ordered by the preceding assistant's call order | `crates/ds4-server/src/render.rs:1442` `tool_run_in_call_order` with `:1421` `preceding_call_order` | `enc-toolresults-order` | pass (closed 2026-10-18) |
| `parse_tagged_text` (273) | the `<image>path</image>` text form expands to image blocks, and a malformed tag raises | not implemented, by contract: pixels arrive base64-only (AGENTS.md), so the tag is never markup | `tests/parse.rs` `the_image_path_tag_form_is_text_and_not_markup` | out of surface, measured 2026-10-20: no image is registered, the content keeps the tag, and the rendered prompt carries it verbatim |
| `_is_image_block` (300) | which content blocks count as images (`image`, `image_url`) | `crates/ds4-server/src/parse.rs:1134` `parse_chat_image_url`, `:456` `add_image_data_uri` | `enc-image-chat` | case added 2026-10-20: the placeholder and its position agree with the oracle, the block separator does not (O-06) |
| `_extract_image` (305) | the url/data/source normalization, and the refusal when none is present | `crates/ds4-server/src/parse.rs:415` `add_image_base64`, `:456` | `enc-image-chat` (the oracle's own record), `tests/parse.rs` `image_refusals_cover_the_oracle_path_and_url_forms` | the oracle's path and URL forms are refused by name; the data-URI form is measured by `enc-image-chat` |
| `_process_image_blocks` (323) | images are replaced by `<｜deepseek_image｜>` in place, nested tool-result images included, and the records keep prompt order | `crates/ds4-server/src/render.rs:1388` `append_dsml_user_content`, `crates/ds4-server/src/parse.rs:638` `write_tool_image_placeholders` | `enc-image-chat`; `tests/parse.rs` `openai_tool_message_carries_the_image_placeholder`; `tests/render.rs` `dsml_v41_places_tool_result_images_inside_the_envelope` | placeholder and order agree; the separator does not (O-06) |
| `_validate_no_image_sp_tokens` (355) | a user-supplied image placeholder in `content` or `reasoning_content` is refused | `crates/ds4-server/src/parse.rs:567` `reject_stray_image_in_content`, `:578` `reject_stray_image_in_reasoning`, called from every place the client's own text is read (message content, text blocks, a tool result's text, `reasoning_content`, the Responses text blocks) | `stray_image_token.rs` (9) and the served probes `…/probe-placeholder-04a0ab1f-20261019` | CLOSED at R7 (04a0ab1f): the oracle's two sentences on the wire; the engine does not implement the rule (`core_v41_vision.c:189-193`, `server_parse_tools.c:290`), so the port mirrors the oracle, CTR 8.9 |
| `process_image_messages` (370) | the per-message image pass, and `content` rebuilt from text blocks with `\n\n` | `crates/ds4-server/src/parse.rs:1167` `parse_openai_message_content`, `:587` `append_text_part` | `enc-blocks-chat`, `enc-image-chat` | O-06 (measured 2026-10-20): the pieces and their order agree, the rebuild's `\n\n` does not |
| `_read_until_stop` (399) | the earliest-of-stops scan the whole parser is built on | `crates/ds4-server/src/tools.rs:169` `split_reasoning_content` | every `dec-*` case | pass |
| `find_last_user_index` (562) | the last user-like index; a mid-conversation system message counts as one; a tool-result-only user message does not start a turn | `crates/ds4-server/src/render.rs:1545` (the loop) with `:263` `role_is_user_like`, `:1050` (the drop rule's copy) | `enc-plain-chat`, `enc-midsystem-chat`, `enc-toolresult-only` | pass, both halves: mid-system closed 2026-10-18, the tool-result-only tail added 2026-10-19 |
| `render_reasoning_effort` (456) | the effort header: thinking mode, index 0 only, `low/high/max` mapped to 50/75/100, an integer in [1,100] taken as itself, `None` to high | `crates/ds4-server/src/render.rs:98` `head_effort_prefix` / `:105` `v41_effort_prefix` with `crates/ds4-server/src/route/mod.rs:383` `RequestedEffort` | `enc-think-low`, `enc-think-max`, `enc-think-budget` | pass (closed 2026-10-18) |
| `render_tools` (509) | the tools preamble verbatim, with the schema list appended | `crates/ds4-server/src/render.rs:453` `append_dsml_tools_prompt_text` | `enc-tools-chat` | pass |
| `encode_arguments_to_dsml` (524) | one parameter element per argument key, in the call's key order, `string="true"` for a string child and its raw form otherwise; a non-dict `arguments` falls back to a single `arguments` key, a JSON string is decoded twice | `crates/ds4-server/src/render.rs:390` `append_dsml_arg`, `:416` `append_dsml_arguments_from_json`, and the retry at `:629` with `crates/ds4-server/src/json.rs:476` `json_string_contents` | `enc-argtypes-chat`, `enc-args-double`, `enc-args-nondict`, `enc-args-plain` | pass on all four, measured 2026-10-20; the stringified-arguments pass was added to the render that day and the C engine has no such pass (`ds4_server.c:3443-3463`), which `enc-args-double` pins |
| `render_message` (585) | one message's bytes, the role branches, the transition tokens, and the assistant header rule | `crates/ds4-server/src/render.rs:1484` `render_dsml_chat_choice_ex` | 19 encode cases | pass on every case measured; O-01 and O-02/O-03 closed 2026-10-18, the consecutive-assistant half of the header rule was found open and closed 2026-10-19 (`enc-think-two-assistants`, below) |
| `_drop_thinking_messages` (749) | before the last user, an assistant's `reasoning_content` is dropped, the message kept | `crates/ds4-server/src/render.rs:1634` `keep_reasoning` (the rule inline in the assistant branch rather than as a message pass) | `enc-think-drop-thinking`, `enc-drop-thinking`, `enc-think-tail-reasoning` | pass (closed 2026-10-19): the drop fires only in thinking mode with no tools anywhere, and the chat-mode case pins that the reasoning cannot leak |
| `_encode_messages_text` (770) | BOS, the loop over messages, and the drop rule forced false when any message carries tools | `crates/ds4-server/src/render.rs:1484` (the entry) with its message loop at `:1543` | all encode cases | pass |
| `encode_messages` (810) | the public entry: merge, sort, image pass, then the text render | `crates/ds4-server/src/render.rs:1484` behind the chat route | all encode cases | pass on the cases measured |
| `load_cases` (847) | the harness entry: a case-level `tools` key moves into `messages[0]`, a bare message list is wrapped | `tests/ds41_ref/ref_drive.py:68` | every encode case | harness |
| `encode_case` (872) | the harness entry: encode one case and return its images, case `thinking_mode` winning over the command line | `tests/ds41_ref/ref_drive.py:70` | every encode case; `:83` for the media driver | harness |

## 2. Decode side — completion to assistant message

| Oracle | Decides | Ours | Case | Verdict |
| --- | --- | --- | --- | --- |
| `parse_tool_calls` (890) | the invoke/parameter grammar: `>\n` after every open tag, one `name` per invoke, the type attribute, duplicate parameters refused | `crates/ds4-server/src/tools.rs:376` `parse_dsml_generated` | `dec-one-call`, `dec-two-calls`, `dec-complex-args`, `dec-unicode`, `dec-marker-in-value` | pass |
| `parse_message_from_completion_text` (946) | the thinking split at `</think>`, content up to EOS or the block start, no content after the block, no special token left in content or reasoning | `crates/ds4-server/src/tools.rs:376`, `:169` `split_reasoning_content` | every `dec-*` case, incl. `dec-thinking` | pass |
| `tool_calls_to_openai_format` (139) | the wire shape: `type: "function"`, `function.name`, `function.arguments`, `namespace` when present | `crates/ds4-server/src/stream.rs:173` `append_tool_calls_json` | every `dec-*` case (the fixture is the oracle's parse) | pass on names and values |

## 3. Constants and templates

Every literal the two sides must agree on, pinned against the oracle's own
source by `crates/ds4-server/tests/ref_constants.rs` (added 2026-10-20): the
test extracts the value the oracle assigns and compares it with ours, so a
row below is measured even when no case happens to carry the literal.  Where
a row says "other families", the constant sits beside ours in the same table
but belongs to a model family outside this line.

| Oracle | Ours | Verdict |
| --- | --- | --- |
| `bos_token` (29) | `crates/ds4-server/src/render.rs:118` `DSML_BOS` | pass |
| `eos_token` (30) | `:121` `DSML_EOS` | pass |
| `thinking_start_token` (31) | `:126` `DSML_THINK_START` | pass |
| `thinking_end_token` (32) | `:127` `DSML_THINK_END` | pass |
| `dsml_token` (33) `｜DSML｜` | folded into the tags below | pass |
| `USER_SP_TOKEN` (35) | `:119` `DSML_USER` | pass |
| `ASSISTANT_SP_TOKEN` (36) | `:120` `DSML_ASSISTANT` | pass |
| `SYSTEM_SP_TOKEN` (426) | `:78` `DSML_SYSTEM_TOKEN` | pass |
| `LATEST_REMINDER_SP_TOKEN` (37) | not located | out of surface: the role is not on the OpenAI wire |
| `IMAGE_PLACEHOLDER` (39) | `:132` `DSML_IMAGE` | pass (pinned by the constants leg and carried by `enc-image-chat`); the refusal beside it is measured by `stray_image_token.rs` |
| `IMAGE_TAG_PATTERN` (40) | not located | out of surface with the `parse_tagged_text` row above |
| `DS_TASK_SP_TOKENS` / `VALID_TASKS` (43, 51) | not located | out of surface: internal classification tasks, not on the OpenAI wire |
| `assistant_msg_template` / `wo_eos` (60, 61) | `crates/ds4-server/src/render.rs:1655` (EOS always written) | the template body is pinned by the constants leg; out of surface for `wo_eos`: no OpenAI request field reaches it (`wo_eos` appears nowhere under `crates/`), while the oracle reads it from the message dict for its own continuation flow (`encoding.py:607`, `:697-703`) |
| `response_format_template` (64) | `crates/ds4-server/src/format.rs:19` `parse_output_format_value`, called from `crates/ds4-server/src/parse.rs:3345`, `:3515`, `:3678`; no render-side counterpart | measured refusal, kept by decision: nothing is served that is not supported (see section 5) |
| `tool_output_template` (68) | `crates/ds4-server/src/render.rs:1368` `append_tool_result_text`, written at `:1619`/`:1621` | pass via `enc-argtypes-chat` |
| `tool_calls_block_name` (428) `" calls"` | `crates/ds4-server/src/tools.rs:44` `DSML_TOOL_CALLS_START_SPACED` | pass |
| `tool_call_tag_name` (429) `" invoke"` | `:46` `DSML_INVOKE_START_SPACED`, `:47` end | pass |
| `tool_parameter_tag_name` (430) `" parameter"` | `:48` `DSML_PARAM_START_SPACED`, `:49` end | pass |
| `tool_call_template` / `tool_calls_template` (432, 435) | `crates/ds4-server/src/render.rs:576`-`:652` | pass |
| `REASONING_EFFORT_TEMPLATE` (443) | the line shape | `crates/ds4-server/src/render.rs:44` `V41_EFFORT_TEMPLATE` with the 75/100 constants at `:36`/`:38` | `enc-think-low`, `enc-think-max`, `enc-think-budget` | pass |
| `REASONING_EFFORT_MAPPINGS` / `DEFAULT_REASONING_EFFORT` (448, 453) | low 50, high 75, max 100, `None` to high | `crates/ds4-server/src/route/mod.rs:368` `mode_budget` / `:395` `RequestedEffort::head_budget` | `enc-think-low`, `enc-think-budget` | pass (closed 2026-10-18) |

## 4. Ours with no oracle counterpart

Functions of this repo's DSML line that the oracle does not define.  They are
kept because a served turn needs them, not because the oracle asks for them.

| Ours | Why it exists | Covered by |
| --- | --- | --- |
| `crates/ds4-server/src/tools.rs:2157` `find_tool_start` (with `crates/ds4-server/src/dsml.rs:389` `dsml_find_tool_start`) | the streaming tracker: the artifact's root spelling (quoted, unquoted, short) is not the oracle's canonical one, so the scan is tolerant where the oracle's grammar refuses | `tests/dsml.rs` `syntax_greedy_spaced_dsml`; `dsml.rs` `override_tests::auto_tool_start_forces_function_prefix`, `::required_prefix_finishes_after_tool_start` |
| `crates/ds4-server/src/dsml.rs:550` `dsml_decode_state_for_text` | the incremental decode state machine that decides when markup is withheld from the stream | `tests/dsml.rs` `syntax_greedy_spaced_dsml`; `generate.rs` `disk_sync_tests::continued_decode_gate_stops_after_tool_syntax_begins` |
| `crates/ds4-server/src/tools.rs` `DSML_*_SHORT`, `DSML_*` (25-36) | the two older root spellings the artifact still emits; the oracle knows only the spaced one | `tests/dsml.rs` `syntax_greedy_spaced_dsml` plus the `dec-*` cases through `ref_decode` |
| `crates/ds4-server/src/parse.rs:2282` `responses_namespace_function_schema`, `:2336` `append_responses_namespace_tool_schemas` | the Responses `type:"namespace"` item: the prompt advertises `namespace + function` and the order keeps both spellings so the answer can be emitted under the client's own name (`crates/ds4-server/src/stream.rs:234` `responses_tool_wire_name`, `:287` the `namespace` field) | `tests/parse.rs` `responses_namespace_tools_keep_both_spellings` |
| `crates/ds4-server/src/retry.rs:144` `try_repair_dsml` | truncated-call repair; the oracle has no repair path (matrix G8, P3) | `tests/retry.rs` `dsml_repair_matches_the_reference_balance_contract`, `repair_and_decide_match_c` |
| `crates/ds4-server/src/render.rs:1681` `responses_visible_assistant_suffix` | the visible assistant tail a Responses client can replay (ds4_server.c:12821); the store that consumes it (`responses_live_remember` :14559) is the reserved checkpoint unit | `tests/render.rs` `responses_visible_suffix_matches_client_replay` |
| `crates/ds4-server/src/trace.rs` | C `--trace FILE` (ds4_help.c:323): one section per request with the prompt, its token ids and the parsed message (unit 3.2, TP-12/TP-15) | `trace.rs` `tests::a_section_carries_the_prompt_tokens_and_parsed_message` |
| `crates/ds4-server/src/retry.rs:260` `find_any_tool_start`, `:277` `find_last_structural` | the repair path's style choice and think scope (ds4_server.c:5842, :5890), ported with G8 | `tests/retry.rs` `repair_and_decide_match_c` |
| `crates/ds4-server/src/render.rs:133`-`:179`, other-family markers | Motif, EXAONE, Solar, Qwen, GLM, K2 tag sets: other families, outside this line | `tests/render.rs` `glm53_chat_and_image_protocol_match_official_template`, `ling31_uses_bailing_syntax`; `tests/parse.rs` `qwen_image_inputs_normalize_across_all_three_surfaces` |
| `crates/ds4-server/src/render.rs:202` `syntax_for_model_id` | family dispatch; the oracle covers V4.1 only | `tests/render.rs` `syntax_for_model_id_matches_c` |

## 5. The open rows

Measured on 2026-10-20, in order of what decides them.  Every row above now
carries a verdict; these are the ones that needed a judgement rather than a
fixture.

1. O-06, the block separator (rows `process_image_messages`,
   `_process_image_blocks`, `_is_image_block`, `process_image_messages`): the
   oracle joins the pieces a client sent as separate content blocks with a
   blank line -- `_process_image_blocks` turns an image into a
   `<｜deepseek_image｜>` text block and the user branch writes
   `"\n\n".join(parts)` (encoding.py:323-353, :631-656, and the `content`
   rebuild at :386-393) -- while this port concatenates them: at parse time for
   text pieces (`append_text_part`, parse.rs:587) and at render time for the
   placeholder (`append_dsml_user_content`, render.rs:1388).  The C lineage
   concatenates too (ds4_server.c:5865), so the port inherited it.  Measured
   the same day on `enc-image-chat` and `enc-blocks-chat`: the two prompts
   carry the same pieces in the same order and differ by that one separator,
   which is exactly what those two cases pin (`check_block_join`).  A third
   place shares the rule and has no case yet: a tool-result message's mixed
   blocks (`write_tool_image_placeholders`, parse.rs:638), asserted today as
   `"shot:<placeholder>done"` by `tests/parse.rs`
   `openai_tool_message_carries_the_image_placeholder`.  The fix is one join,
   gated to the V4.1 head, in the render's parts loop plus the parse's text
   pieces; it is the operator's call because it moves every image-bearing
   prompt and every multi-block message on that head.
2. The image surface's oracle leg (rows `parse_tagged_text`,
   `_is_image_block`, `_extract_image`, `_process_image_blocks`,
   `_validate_no_image_sp_tokens`, `process_image_messages`): `ref_drive.py`
   refused image cases by construction (`assert not images`, kept for the
   text-only legs), so the leg was rebuilt on 2026-10-20 as `encode-media`
   (`ref_drive.py:75`), which prints the prompt and reports the oracle's own
   image records on stderr.  `_validate_no_image_sp_tokens` is CLOSED at R7;
   the other five now have the oracle's bytes, with O-06 the only difference.
3. The namespace surface (rows `_split_tool_name`, `_tool_name_for_encoding`):
   out of surface.  The client's model-facing tool spec is
   name/description/parameters only (`conversation.rs:861-869`, the conversion
   at `:1039-1045`) and its `namespace` rides the editor-facing ToolCall meta
   (`acp_session.rs:1741`), so no chat request can carry the field and a
   `::`-qualified name is never sent.  Both halves are pinned by
   `tests/parse.rs` (`chat_tool_names_pass_through_verbatim`,
   `responses_namespace_tools_keep_both_spellings`).
4. `wo_eos` (`assistant_msg_template` / `wo_eos`): out of surface.  `wo_eos`
   appears nowhere under `crates/`; the oracle reads it from the message dict
   for its own continuation flow (`encoding.py:607`, `:697-703`).
5. `IMAGE_TAG_PATTERN`, the `<image>path</image>` text form, and
   `LATEST_REMINDER_SP_TOKEN` / `DS_TASK_SP_TOKENS` / `VALID_TASKS`: out of
   surface.  The carriage is base64-only by design (AGENTS.md) and the rest are
   internal classification roles, not on the OpenAI wire.  The tag form is
   measured rather than assumed (`tests/parse.rs`
   `the_image_path_tag_form_is_text_and_not_markup`).
6. `response_format` rendering (row `response_format_template`): the one row
   whose verdict is a decision and not a measurement.  Ours refuses
   `json_object` and `json_schema` with 400 "structured output is unsupported"
   (`format.rs:6-15`, the v0.6.3 refusal of ds4-on-spark#10), measured on the
   served binary today; the oracle renders the schema into the prompt instead
   (`encoding.py:628-629` with the template at `:64`).  The client of record can
   put `response_format: json_schema` on the chat wire, from
   `ConversationRequest.json_schema` (`conversation.rs:4107`), at two production
   sites -- the auto-mode permission classifier (`sampler_turn.rs:862`) and the
   goal evaluator (`goal_evaluator.rs:172`) -- while its main turns hardcode
   None (`request_builder.rs:148`, sampler `client.rs:4071`), and no capability
   gate guards those two sites.  The operator decided 2026-10-20 that the
   refusal stands: the field carries `strict: true`, i.e. a guarantee that the
   served text conforms to the schema, and only a decode-time constraint can
   give that guarantee.  A prompt-side instruction would be an invented
   approximation of a guarantee, so nothing is served for this field, and the
   client's own error branch records the side query as failed
   (`session_summary.rs:131-137` for the neighbouring title query).  Serving it
   for real means a JSON-schema-constrained decoder, a unit of its own.
7. O-04 arguments bytes: ours minify, the oracle keeps the parameter body
   verbatim.  Values agree; the strings differ.  Recorded, not a defect against
   the wire, since the client echoes our string.

Closed since the map was written: O-01 (the effort budget, both halves), O-02
(a mid-conversation system message renders in place), O-03 (tool results
ordered by call order), the repair rows of G8/G9, the image-placeholder
refusal (R7), the drop-thinking rule together with its chat-mode counterpart,
the stringified-arguments fallback (2026-10-20, `enc-args-double`), and the
`find_last_user_index` tool-result-only half (`enc-toolresult-only`).

One divergence was found by the 2026-10-19 pass and closed in code.  Two
assistant messages in a row share one `<Assistant>` header, so the second has
no header to open its thinking, and the oracle still writes that message's
reasoning and its closer as part of the body (`encoding.py:678-712`).
Measured before the fix: ours rendered 275 bytes against the oracle's 304,
dropping `SECOND-THOUGHT-MARKER` whole (`enc-think-two-assistants`).  The
assistant branch now writes the thinking body when it has no header, gated to
the V4.1 head so the port's own head keeps its bytes.

One divergence was found by the 2026-10-20 pass and closed in code.  A
replayed call whose `arguments` is itself a JSON string still renders that
object's parameters, because the oracle decodes the field twice before it
gives up (`encoding.py:528-537`).  Measured before the fix: ours wrote one
`arguments` parameter carrying the quoted text (1534 bytes against the
oracle's 1511, `enc-args-double`).  `append_dsml_tool_calls_text` now retries
the parse on the unwrapped string (`json_string_contents`, json.rs:476), and
the C engine's lack of that pass is pinned by the same case.

## 6. Reproducing

```bash
# the oracle's own bytes for a case, and its parse of a completion
python3 tests/ds41_ref/ref_drive.py encode       tests/ds41_ref/cases/enc-plain-chat.json chat
python3 tests/ds41_ref/ref_drive.py encode-media tests/ds41_ref/cases/enc-image-chat.json chat
python3 tests/ds41_ref/ref_drive.py parse        tests/ds41_ref/decode/dec-one-call.txt chat

# the three reference legs against the pinned fixtures
cargo test -p ds4-server --test ref_encoding
cargo test -p ds4-server --test ref_decode
cargo test -p ds4-server --test ref_constants
```

Regenerate the fixtures only when the oracle or a case changes:
`bash tests/ds41_ref/regen_fixtures.sh`.
