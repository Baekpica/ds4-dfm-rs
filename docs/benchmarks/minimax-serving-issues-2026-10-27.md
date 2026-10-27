# MiniMax output parsing and streaming: issues #116 and #117

This is a host-protocol regression report, not a model-quality or GPU gate.
The baseline is `3f064f67`. Input continues to use the artifact's official
`chat_template.jinja`; the template does not parse or constrain sampled output.

## Framework baseline

Sources were inspected on 2026-10-27 at the following commits, rather than
assuming a common policy from OpenAI-compatible endpoint names.

| Framework | Tool-output handling | Streaming and keepalive |
|---|---|---|
| [vLLM `40fe396b`](https://github.com/vllm-project/vllm/blob/40fe396b60b0fbf312b5ac55dbf3f68a4e12b785/rust/src/parser/src/tool/minimax_m3.rs#L162) | M3 text mode searches for the full namespaced `<tool_call>` wrapper. A bare namespace or wrapperless invoke is text. An incomplete recognized tool block can fail parsing. | Arguments wait for a complete invoke. Its optional [SSE keepalive](https://github.com/vllm-project/vllm/blob/40fe396b60b0fbf312b5ac55dbf3f68a4e12b785/vllm/entrypoints/serve/utils/sse_keep_alive.py#L17) sends comments, not completion chunks. |
| [SGLang `635430d0`](https://github.com/sgl-project/sglang/blob/635430d07378e2b12381353781a23b0704969909/python/sglang/srt/function_call/minimax_m3.py#L108) | Full-wrapper detection; missing wrappers remain text. The buffered parser falls back to original text on failure. | Declared string parameters [stream before their closing tag](https://github.com/sgl-project/sglang/blob/635430d07378e2b12381353781a23b0704969909/python/sglang/srt/function_call/minimax_m3.py#L326); complex/non-string values buffer. The inspected Chat route uses `StreamingResponse`, without injecting periodic completion chunks. |
| [TensorRT-LLM `2d3e60e2`](https://github.com/NVIDIA/TensorRT-LLM/blob/2d3e60e27e7293973e3b79f34392492a65386999/tensorrt_llm/serve/tool_parser/minimax_m3_parser.py#L136) | Full-wrapper detection; without it, the parser returns text and no calls. | M3 buffers through the block close. The inspected [Chat generator](https://github.com/NVIDIA/TensorRT-LLM/blob/2d3e60e27e7293973e3b79f34392492a65386999/tensorrt_llm/serve/openai_server.py#L2063) emits actual postprocessor results, without a periodic completion heartbeat. |
| [llama.cpp `7ecb57f7`](https://github.com/ggml-org/llama.cpp/blob/7ecb57f7adbea2928aa468ad2aecf05b10725ad4/common/parsers/minimax-m3.cpp#L186) | The M3 grammar requires the full wrapper and declared tool names. Text is read until that wrapper; a failing recognized grammar [throws](https://github.com/ggml-org/llama.cpp/blob/7ecb57f7adbea2928aa468ad2aecf05b10725ad4/common/chat.cpp#L1701). | Partial M3 ASTs [produce argument prefixes](https://github.com/ggml-org/llama.cpp/blob/7ecb57f7adbea2928aa468ad2aecf05b10725ad4/common/chat-peg-parser.cpp#L1180), projected as argument deltas. Its [SSE ping](https://github.com/ggml-org/llama.cpp/blob/7ecb57f7adbea2928aa468ad2aecf05b10725ad4/tools/server/server-context.cpp#L5420) is a comment. |
| [TGI `b4adbf2f`](https://github.com/huggingface/text-generation-inference/blob/b4adbf2f6e2e721280bd0ea5f91d70f7d033f5ed/router/src/chat.rs#L33) | The inspected tool path parses grammar-generated JSON and returns `ToolError` for invalid JSON. This is not evidence of a MiniMax-M3 XML parser or M3 model support. | Tool-mode token text becomes argument deltas. The [Chat route](https://github.com/huggingface/text-generation-inference/blob/b4adbf2f6e2e721280bd0ea5f91d70f7d033f5ed/router/src/server.rs#L1247) uses Axum `KeepAlive::default()` (SSE comments). |

The SGLang and TensorRT-LLM M3 parser classes were also executed in isolation,
using their source methods with import/dependency plumbing replaced by simple
data containers. The serving stacks and models were not launched. For each
parser, both buffered and byte-at-a-time streaming returned raw content and
zero calls for these three inputs:

- `]<]minimax[>[ slide` followed by more than 80 bytes of prose;
- `]<]minimax[>[<invoke native pptx table` (missing a function name);
- a complete namespaced invoke without the `<tool_call>` wrapper.

With an open, correctly wrapped call and a declared string parameter containing
40,000 ASCII characters, SGLang emitted 40,001 argument bytes before the closing
tag (including the JSON quote); TensorRT-LLM emitted none. The vLLM and
llama.cpp observations above are source traces, not executed parser tests.

## Decisions

**#116:** the supplied bytes reproduce a model-format failure and the existing
fallback, not a Jinja-input regression unique to ds4. Preserve full-wrapper
recognition, as in the M3 implementations above. Do not infer a missing tool
name/arguments, accept wrapperless calls, or turn arbitrary namespace prose into
an API failure. The reported higher incidence of malformed strings with VQ
makes recovery relevant, but does not identify the missing call or justify a
different default protocol. No live sampling-frequency or quantizer diagnosis
is claimed here.

**#117:** preserve comment heartbeats and emit actual OpenAI Chat tool argument
deltas, following the SGLang/llama.cpp approach. MiniMax XML decoding belongs
to the Rust server's family output adapter; tool IDs, JSON wire framing and SSE
use the existing common `ToolSink`. Neither the input template nor native
MiniMax/CUDA code changes.

Declared string values stream incrementally, with incomplete UTF-8 and closing
tag prefixes withheld. Other values use the existing final parser's schema
conversion after their parameter closes. Finalization does not emit the same
call again. A malformed recognized OpenAI tool stream reports an API error;
token-budget truncation retains `length`. This follows vLLM's recognized-block
failure behavior without changing bare-namespace fallback.

This closes the silent interval for a long `write_file` string. It does not
promise data events while queued, during prefill, or while buffering one large
non-string/nested parameter. Ordinary MiniMax reasoning already uses
`reasoning_content` deltas. Anthropic/Responses tool streaming is outside this
OpenAI Chat change.

## Reproduction and validation

Model-free failing-first regression:

```sh
cargo test -p ds4-server --lib --locked minimax_long_string_deltas -- --test-threads=1
```

On the baseline, this fails because no argument delta arrives before the close.
The repaired test checks incremental receipt and exact JSON reconstruction.
Other regressions cover byte-split UTF-8/tags, nested objects/arrays, numeric
values, multiple calls/wrappers, a nested value despite a string schema,
truncation and no duplicate final emission. Scripted generation tests exercise
both serial and continuous lanes, buffered and streaming, through the real host
projectors. The synthetic Jinja input in those tapes is not an official-template
parity claim.

An actual OpenAI Python 3.39.0 client consumed the continuous projector's SSE
through an HTTP mock transport: a comment-only stream yielded zero SDK chunks;
the repaired 42,000-byte UTF-8 string produced 17 chunks, one tool name/ID,
exact reconstructed arguments and `finish_reason=tool_calls`. An unfinished
recognized call raised `openai.APIError`.

Final host checks all exited 0:

- `cargo fmt --all -- --check`;
- `cargo clippy --workspace --all-targets --locked` (existing warnings remain);
- the eight C/Rust host-parity targets from `CONTRIBUTING.md`, built/run
  together with `make -j1`;
- `cargo test --workspace --locked -- --test-threads=1`: 2,190 passed,
  34 ignored, no failures;
- `cargo check --workspace --all-targets --locked`;
- `cargo check -p ds4-server --lib --bin ds4-server-rs --features native --locked`
  (type-checking, not a linked CUDA build).

The host is Linux aarch64 on GB10, CUDA 13.4.92. No model was loaded, no GPU
job was started, and the existing MiniMax service was left untouched. These
results establish host parsing/framing behavior, not a replay of the reporter's
116K prompt, temperature-0.7 sampling or a 15-minute agent watchdog session.
