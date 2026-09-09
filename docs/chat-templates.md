# Model chat templates

For Jinja-backed artifacts, production Chat, Messages, Responses, one-shot
CLI and REPL input use `ds4-core::chat_template::Template`. The local adapter
wraps pinned `hf-chat-template` 1.0.0 with Python compatibility enabled.
Models compile their template once at load time; inference does not contact HF.

```text
API / CLI messages and tools
        |
Normalize roles, JSON arguments, reasoning and media content
        |
Local adapter -> model Jinja -> rendered prompt
        |
Model tokenizer + image/audio processor -> token IDs / feature spans
        |
Native inference -> family output protocol -> API / CLI output
```

New families should start with architecture/tensor layout, loading, tokenizer
and output protocol. Supply the official template as an artifact and verify it
against Python. Do not translate its role delimiters, tool schemas, thinking
rules or message ordering into another imperative Rust renderer. Output stop
tokens, streamed reasoning/tool parsing and media processing remain explicit
family contracts; a Jinja file does not implement those capabilities.

## Finding the template

Selection uses the directory containing the opened GGUF (the first shard for
a split model), in this order:

1. `chat_template.jinja`.
2. `tokenizer_config.json`'s `chat_template` string or `default` template
   from a named array or object.
3. GGUF `tokenizer.chat_template`.

A selected empty, invalid or unsupported template fails loading; the engine
does not silently switch grammars. Outside the DeepSeek V4 exception below,
a missing template allows raw Completions, but chat requests fail explicitly.
Special token values come from the local
tokenizer configuration and GGUF token metadata. Rendered chat is tokenized
without adding a second BOS/EOS wrapper.

For a missing sidecar, resolve the GGUF's base-model source revision, download
that revision's official template, and put it next to the GGUF. Pin the source
revision and record its SHA-256; a quantization repository's current `main`
may refer to a different source revision. For example:

```sh
hf download thinkingmachines/Inkling-Small chat_template.jinja \
  --revision 8cc5877b44d343f88b92086aa1fb72897950f06a \
  --local-dir ./models/Inkling-Small/MQ85GB
sha256sum ./models/Inkling-Small/MQ85GB/chat_template.jinja
```

The [fixture manifest](../tests/fixtures/chat-template/README.md) records the
official sources, licenses and exact bytes used by the reference tests. The
local qualification also stores `chat_template.provenance.json` beside each
downloaded template. K2's pinned official file differs from its GGUF's embedded
template; the verified sidecar takes precedence.

**DeepSeek V4 is an encoder exception.** Its pinned official tree provides
[`encoding_dsv4.py`](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/blob/60d8d70770c6776ff598c94bb586a859a38244f1/encoding/encoding_dsv4.py),
not an official Jinja. The existing native encoder remains available, and the
historical embedded Jinja is not selected automatically. No replacement Jinja
was invented or downloaded from a different family.

## Normalization and continuation

The adapter receives structured messages and JSON tool arguments. API
normalization preserves tool IDs, reasoning and the order of media parts;
Anthropic tool results become individual `tool` messages. Missing assistant
reasoning becomes an empty string when no thinking field was supplied.

The common render context supplies `tools`, `add_generation_prompt`,
`enable_thinking`, `reasoning_effort` and tokenizer special tokens. Existing
effort aliases remain output-policy choices. GLM and K2 always open a thinking
channel in their official templates; disabled thinking closes that channel
with an output prefill. Their source templates remain unchanged.

Qwen and Inkling templates render their media placeholders. GLM's official
text template does not accept media arrays, so its image processor supplies
placeholder text before rendering. Native processing expands placeholders
into feature spans afterward. Existing modality and lane limits still apply.

Messages/Responses tool-only continuation restores the retained structured
conversation before rendering. Chat Completions clients replay full history.
A continuous bank must still match its saved generation/frontier.
Full caller history remains authoritative. REPL turns also retain structured
messages and reuse the server's output parser; the CLI therefore has a host
dependency on `ds4-server`, without starting an HTTP server.

Templates may change earlier text between turns. KV reuse is validated against
the complete rendered token sequence; a matching text prefix alone is not
sufficient. Changed prefixes are refilled. Invalid tool output is discarded
before a non-streaming correction retry re-renders the valid conversation.

The built-in `ds4-agent` executor implements DeepSeek DSML and rejects other
families before loading weights. Use HTTP tool clients for their supported
output protocols. The retained imperative renderers serve the DeepSeek
exception and C parity fixtures, not the production Jinja path.

## Verification

The pure `Template::render` API executes supplied context without API
normalization. `Template::render_chat` adds canonical defaults and output
prefill policy. Inspect a local artifact without loading weights or a GPU:

```sh
cargo run -p ds4-core --example chat_template -- MODEL.gguf CONTEXT.json
cargo test -p ds4-core --test chat_corpus --test chat_json --test chat_assets --test chat_options
cargo test -p ds4-server --test chat_input
```

The corpus compares 76 renders/errors against independently generated Python
Jinja results, covering all nine local Jinja source variants. JSON regressions
cover Python keyword behavior, Unicode, key order and float representation.
The vendored dependency needs small generic JSON/float compatibility patches;
see [upstream pin and patch rationale](../vendor/hf-chat-template/VENDOR.md).
Templates are preserved byte-for-byte. Do not update goldens to hide drift.

Live artifact results and limits belong to the [v0.1.2 ledger](releases/v0.1.2.md).
