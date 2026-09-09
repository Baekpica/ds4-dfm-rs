# Compatibility

"Compatible" here means one thing precisely: for a given `(chat_template, input)`, this crate
produces **the same bytes** as Python `transformers.apply_chat_template(..., tokenize=False)`.
That claim is enforced by `tests/m3_corpus.rs`, which renders real model templates and diffs
against committed reference output from `transformers` (see `tests/corpus/README.md` for the
pinned version and provenance).

## Verified models

Byte-identical in CI:

| Model | Cases exercised |
|---|---|
| Qwen/Qwen2.5-0.5B-Instruct | basic, no-system, single-user, tool calling |
| Qwen/Qwen3-0.6B | basic, no-system, single-user, tool calling |
| HuggingFaceTB/SmolLM2-1.7B-Instruct | basic, no-system, single-user |
| microsoft/Phi-3-mini-4k-instruct | basic, no-system, single-user |
| NousResearch/Hermes-3-Llama-3.1-8B | basic, no-system, single-user, named `tool_use` template |
| LiquidAI/LFM2-1.2B | basic, no-system, single-user, tool list (standalone `chat_template.jinja`) |
| HuggingFaceTB/SmolLM3-3B | standalone `chat_template.jinja`, `{% generation %}` block (reasoning) |
| mistralai/Mistral-7B-Instruct-v0.3 | basic, no-system, single-user, tool calling (`[INST]` / `[AVAILABLE_TOOLS]`) |
| Qwen/QwQ-32B | basic, no-system, single-user, tool calling (reasoning, `</think>` split) |
| deepseek-ai/DeepSeek-R1-Distill-Qwen-7B | basic, no-system, single-user (reasoning, `<think>`) |
| deepseek-ai/deepseek-llm-7b-chat | basic, no-system, single-user (`User:` / `Assistant:`) |
| openchat/openchat-3.5-0106 | basic, no-system, single-user (`GPT4 Correct …`, `.title()`) |
| HuggingFaceH4/zephyr-7b-beta | basic, no-system, single-user (`<\|user\|>` / `<\|assistant\|>`) |
| 01-ai/Yi-1.5-9B-Chat | basic, no-system, single-user (ChatML variant) |
| tiiuae/falcon-7b-instruct | basic, no-system, single-user (`.strip()` / `.replace()`) |
| ibm-granite/granite-3.1-8b-instruct | `strftime_now` date stamp (clock pinned), with/without system |
| google/gemma-2-9b-it | `<start_of_turn>` format, no-system (raises on system role), multi-turn |
| google/gemma-3-4b-it | `<start_of_turn>`, system merged into the first turn, string/parts content |
| CohereForAI/c4ai-command-r-v01 | named `default` / `tool_use` / `rag` templates (macros, recursion) |
| meta-llama/Llama-3.1-8B-Instruct | `<\|start_header_id\|>` format, tools in the user turn, `date_string` |

This is the corpus: 20 models, 68 cases. It spans the major template families people run
(Mistral `[INST]`, ChatML, DeepSeek, Gemma `<start_of_turn>`, Command-R, reasoning templates with
`<think>` / `{% generation %}`, `strftime_now` date stamping, standalone `chat_template.jinja`
layouts, named `tool_use` / `rag` templates, and tool calling). It includes the gated marquee models
(Llama-3.1, Gemma-2, Gemma-3, Command-R).

## Jinja surface supported

Confirmed against real templates and the corpus:

- **Control flow & whitespace**: `trim_blocks` and `lstrip_blocks` on, `keep_trailing_newline` off,
  and `{%- … -%}` whitespace control, matching the `transformers` Jinja environment. Off is what
  the reference does, so a template whose source ends in a newline has that one newline stripped,
  the same as Python.
- **`namespace()` cross-scope mutation** — `{% set ns = namespace(found=false) %}` then mutating
  `ns.found` inside loops.
- **Macros & recursion** (e.g. Hermes's `tool_use` template).
- **Loop variables**: `loop.index0`, `loop.last`, `loop.first`, `loop.previtem`, `loop.nextitem`.
- **`raise_exception(msg)`** — surfaces as a distinct [`Error::TemplateRaised`], not an engine
  error, so you can tell "the template rejected this conversation" from "the library has a bug."
- **`strftime_now(fmt)`** — via an injectable [`Clock`]: the default `SystemClock` (UTC),
  `LocalClock` (local time, the `strftime` feature), or `FixedClock` (pinned). The strftime
  specifiers are proven byte-identical by the corpus (`granite-3.1-8b-instruct`, date pinned).
- **`tojson`** — a custom filter matching Python `json.dumps(x, ensure_ascii=False)`: `", "` /
  `": "` separators and **insertion-ordered** keys (minijinja's built-in sorts keys and omits
  spaces). Honors `tojson(indent=N)`.
- **`pycompat`** (default feature) — Python methods on values: `.strip()`, `.lstrip()`,
  `.rstrip()`, `.split()`, `.startswith()`, `.endswith()`, `.title()`, `.upper()`, `.lower()`,
  `| items`, `| trim`, and more, provided by `minijinja-contrib`.
- **`{% generation %}` block** — the `transformers` AssistantTracker block (marks assistant token
  spans for `return_assistant_tokens_mask`). It is rewritten to an output-neutral `{% if true %}`
  before compilation, so reasoning templates that use it (SmolLM3) render byte-identically. The
  span metadata itself is not exposed; this crate emits a string, not a token mask.

## Known divergences & caveats

- **`strftime_now` defaults to UTC.** `SystemClock` reads `SystemTime` as UTC; `transformers`
  uses Python local time. Templates that call `strftime_now` (Granite 3.x, SmolLM3) will differ
  by timezone with the default clock. Templates that take the date from a caller-supplied variable
  instead, such as Llama-3.1's `date_string`, are unaffected because no clock is involved.
  To match `transformers`, enable the `strftime` feature and
  inject `LocalClock` (local time), or inject a `FixedClock` to pin a specific instant. The corpus
  pins dates (`clock_unix_secs`) for date-stamped models so their references stay reproducible.
- **Undefined variables are lenient by default** (`UndefinedBehavior::Lenient`), matching the
  common `transformers` behavior. Override via the builder if a template needs strict semantics.
- **`pycompat` coverage is `minijinja-contrib`'s set.** An exotic Python method a template relies
  on may be missing; if you hit one, it shows up as a render error, not silent wrong output.
- **No automatic BOS doubling.** If a template emits `bos_token`, set `add_special_tokens = false`
  at encode time. This crate never strips or adds special tokens behind your back.

## Reporting a mismatch

If you find a template that renders differently from `transformers`, that's a bug worth a report.
Ideally include the `chat_template`, the input messages, and both outputs. Adding it to the corpus
(`tests/corpus/`) and regenerating references with `tools/gen_reference.py` turns the fix into a
permanent regression test.

[`Error::TemplateRaised`]: https://docs.rs/hf-chat-template/latest/hf_chat_template/enum.Error.html
[`Clock`]: https://docs.rs/hf-chat-template/latest/hf_chat_template/trait.Clock.html
