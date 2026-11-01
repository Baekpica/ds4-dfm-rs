# C continuation

Rust owns the host and serving policy. Native code keeps what has to be
native: kernels, device state, GPU execution, weight upload/VMM and the
opaque bridge over them. Other C either has a Rust owner, and is a deprecated
frozen copy, or is a gap that moves to Rust when a change touches it.
ds4 is a serving engine: agents and frameworks connect over HTTP. The
built-in coding agent was removed rather than continued.

Recorded on v0.1.5 (`c80f6283`), 2026-11-01. Line counts are approximate;
re-check call sites before acting on a row.

## Native by necessity

| Source | Why it stays native |
|---|---|
| `ds4_cuda.cu`, `cuda/`, `ds4_*_gpu.cuh` | CUDA kernels and vendored MMQ |
| `ds4.c` engine core, `ds4_*.inc`, `ds4_gpu.h` | device state, family graphs, GPU execution, weight upload/VMM, fit arithmetic, payload GPU tails, the weight materializer's governor check, V4.1 device sampling |
| `ds4_session_sync` reuse decision | duplicates `SessionLedger` (session oracle) but decides against GPU state |
| `ds4_ple.c` page cache | `O_DIRECT` reads and row leases that CUDA stream callbacks release |
| Memory fit and admission checks | compare live free memory (`cudaMemGetInfo`, MemAvailable) with native allocation estimates at the moment of allocation; Rust passes the policy values they use |
| `ds4_metal.m`, `metal/` | inherited Metal backend |
| `native/bridge` | the opaque ABI over `ds4.h` that `ds4-sys` binds |
| `tools/ds4_weight_server.cu`, `tools/gb10_peaks.cu`, `tools/vq_calib/cuda` | CUDA weight owner, peak probes, VQ converters |
| `tools/minimax_h3/*.c` | probes compiled against the C reference engine |
| `linenoise.c`, `rax.c`, `vendor/stb_image.h` | vendored third-party C |

## Deprecated C with a Rust owner

No features or fixes; frozen until removed. Rust hosts do not reach them.

| C | Rust owner | Still used by |
|---|---|---|
| `ds4_cli.c`, `ds4_bench.c` | `ds4-cli` | `make proof-*-opp-c`, Bonsai diagnostics |
| `ds4_server.c` (DSML, metrics windows, warm gate) | `ds4-server` | `tests/ds4_test.c`, `tests/test_deepseek_budget.c` |
| `ds4_kvstore.c`, `rax.c` | `ds4-kv` | C server, `kv_c_oracle` |
| `ds4_mem_gov.h` evaluator calls in the C server | `ds4-core::mem_gov` | C server |
| `ds4_ple.c` manifest parser and `DS4_QWEN_PLE_*` reads | `ds4-core::ple` | C hosts, native PLE tests |
| Memory policy env reads (`DS4_MEM_FLOOR_GB`, `DS4_SESSION_GRAPH_*`, `DS4_BATCH_FIT_*`, `DS4_MEMGOV*`) | `ds4-core::mem_policy` | C hosts |
| `ds4.c` `vocab_load`, BPE, chat encoders, token text | `ds4-core::tok` | C hosts, `ds4-eval`, native family tests, tokenizer oracle |
| `ds4.c` `parse_tensors`, `config_validate_*`, `weights_validate_*` for the base model | `ds4-core` catalog | C hosts; skipped when Rust passes host tables |
| `ds4.c` C sampler body | `ds4-core::sample` | C hosts, `ds4-eval`; Rust registers the host sampler at open |
| `ds4.c` family stop table (`vocab_token_is_generation_stop`) | `Vocab::is_stop`, passed at open | C hosts |
| `ds4_distributed.c` worker runtime, one-shot coordinator, CLI parsing (3.1K lines) | `ds4-dist` | C hosts, `ds4-eval` |
| `tests/parity/*_c_oracle.c` | — | frozen copies for `make ci-parity` |

## Gaps

C that does not have to be C but still runs on a Rust path, largest first,
then tools with no Rust host.

| Gap | Today | Next step |
|---|---|---|
| Distributed coordinator | `ds4`, `ds4-bench` and `ds4-server` coordinators run `ds4_dist_session_*` inside the native session: 4.1K of 8.2K lines of `ds4_distributed.c` (listener, routing, work frames, payload save/load). `ds4-dist::Coordinator`, the route planner and snapshot transport run only in tests. `--dist-replay-check` is inert. | Host coordinator session over `ds4-dist`; bridge the output head from hidden state and session logits; live coordinator/worker gates. About 1.3-2.1K Rust lines. |
| Image preprocessing | Qwen4Exp (probe, pixel hash, bicubic resize, patches), GLM-5.3 (EXIF, smart resize, CLIP normalization) and Ling-3.0-VL (probe, decode, resize) decode with stb_image in `ds4.c`. Step, Inkling, MiMo, MiniMax and V4.1 already preprocess in Rust. | Port with pixel parity. C rounds resize weights with `round(q±0.5)` where Pillow truncates, the `image` crate's JPEG decoder may differ from stb_image, and the Qwen pixel hash is a KV-cache identity. |
| Sidecar GGUF parsing | Engine open clears the host tables, then parses the vision, audio, MTP and DSpark GGUFs in C, although Rust validates them first. `DS4_DSPARK_MODEL` opens a sidecar Rust never saw. | Pass sidecar tensor tables like the base model's. |
| Bank speculation match | `cont_spec_verify` is a C copy of `sample::match_rows`; its draws are already Rust. | Register the match next to the host sampler. |
| Process policy at open | `ds4_engine_open` reads MTP, DSpark, residency and IPC settings from the environment and runs `malloc_trim`. The instance lock is the host's (`Model::open`); native takes it only for C hosts. | Own the settings in Rust; native returns errors. |
| Payload header | Native writes the DSV4 header and token ids before the GPU tail; `ds4-core::payload` has the encoder, used only by tests. | Host writes the prefix, native appends the tail. |
| Logit readouts | `argmax`, `top_logprobs` and `token_logprob` run in C on host logits. | Keep unless a host copy is measured free on decode. |
| V4.1 run loop | EOS stop and frequency/presence/DRY penalties run in `ds4_ds41_*.inc`, with breakers built from C token text, mirroring `/data/YoungAi`. | Owner decision: the reference engine decides V4.1 behavior. |
| `ds4-eval` | C-only DeepSeek benchmark evaluator; the main user of the deprecated C tokenizer, encoder and sampler. | Port to `ds4-cli`, checked by its `--self-test-extractors` and `--regrade-trace`, or retire. |
| Bonsai diagnostics | `run-bonsai.sh` and `make bonsai-cuda-*` drive `ds4-c --first-token-test`. | `./ds4 --backend {cpu,cuda} --token-ids`, as `test-qwen35-rust-host` does. |
| Metal imatrix collector, `--perplexity-file` | `ds4-c` on Metal only. | Retire with Metal or port. |
