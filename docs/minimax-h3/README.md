# MiniMax-H3 documentation

Everything about the MiniMax-H3 FL2VA family lives under this directory. No
h3 document sits loose at the top of `docs/`; the same rule holds for
`tools/minimax_h3/` and `crates/ds4-core/src/minimax_h3/`.

Branch `feature/minimax-h3`. Model root on the Spark: `/mnt/h3-artifacts`.

| Document | Covers |
|---|---|
| [port-plan.md](port-plan.md) | The port plan: phases P0-P10, the Spark budget, the DiT rung, the render gate |
| [post-port-phases.md](post-port-phases.md) | After the port: P9 speed, P10 server with a resident engine, P11 YouTube/Instagram presets, P12 conditioning, P13 close |
| [cheatsheet.md](cheatsheet.md) | Sizes, quality, reach, memory and layout in one page |
| [vq-recipe.md](vq-recipe.md) | The 13-bit GVQ1 v1 conversion of the DiT and its fallback |
| [openai-videos.md](openai-videos.md) | Historical: the OpenAI Videos surface as P10 built it. Not served any more; kept as the record of that phase |
| [local-v1-contract.md](local-v1-contract.md) | The served local v1 contract: three routes, the five-field body, the canvases, the refusals, the deviations |
| [api-usage.md](api-usage.md) | Running the local v1 host: start, create, poll, download, errors, limits, what is not served |
| [v1-local-gate.txt](v1-local-gate.txt) | The local v1 wire gate, 2026-10-25: stand-in renderer, a real socket, every route and every refusal |
| [upstream-m31-survey.md](upstream-m31-survey.md) | The M3.1 text-family branch: what transfers to h3, what is rejected, why h3 has no KV cache |
| [ref-media-tests.md](ref-media-tests.md) | The conditioning matrix: FL2VA anchors and Ref2VA image/video/audio cases, fixtures and acceptance |

Related, outside this directory:

| Surface | Path |
|---|---|
| Rust reader | `crates/ds4-core/src/minimax_h3/` |
| DiT tensor layout | `crates/ds4-core/src/minimax_h3/layout.rs` |
| Mapped weight table | `crates/ds4-core/src/minimax_h3/weights.rs` |
| Text-encoder path | `crates/ds4-core/src/minimax_h3/text.rs` |
| Encoder weight table | `crates/ds4-core/src/minimax_h3/text_weights.rs` |
| Encoder RoPE tables | `crates/ds4-core/src/minimax_h3/text_rope.rs` |
| Encoder forward | `crates/ds4-core/src/minimax_h3/text_forward.rs` |
| Encoder gate | `crates/ds4-core/examples/h3_text_gate.rs` (`make tests/h3_text_gate`) |
| Native encoder primitives | `native/h3/h3_text_gpu.cu` |
| Vision tower layout and host prep | `crates/ds4-core/src/minimax_h3/vision.rs` |
| Vision weight table | `crates/ds4-core/src/minimax_h3/vision_weights.rs` |
| Vision forward | `crates/ds4-core/src/minimax_h3/vision_forward.rs` |
| Vision gate | `crates/ds4-core/examples/h3_vision_gate.rs` (`make tests/h3_vision_gate`) |
| Native vision primitives | `native/h3/h3_vision_gpu.cu` |
| Multimodal presentation (`<Picture n>:`, mRoPE, tags) | `crates/ds4-core/src/minimax_h3/multimodal.rs` |
| Multimodal gate | `crates/ds4-core/examples/h3_multimodal_gate.rs` (`make tests/h3_multimodal_gate`) |
| Host arithmetic (canvas, noise, sigmas, reuse window, Euler step) | `crates/ds4-core/src/minimax_h3/host.rs` |
| Packed sequence layout | `crates/ds4-core/src/minimax_h3/sequence.rs` |
| DiT forward, Euler denoise, block stream, token reduction, layer mask, core reuse | `crates/ds4-core/src/minimax_h3/dit.rs` |
| DiT gate | `crates/ds4-core/examples/h3_dit_gate.rs` (`make tests/h3_dit_gate`) |
| Native DiT primitives | `native/h3/h3_dit_gpu.cu` |
| Audio VAE decode | `crates/ds4-core/src/minimax_h3/vae_audio.rs` |
| Video VAE decode | `crates/ds4-core/src/minimax_h3/vae_video.rs` |
| F32 weight store (both VAEs) | `crates/ds4-core/src/minimax_h3/vae_store.rs` |
| Output stage (canvas rules, RGB24, resize, mux) | `crates/ds4-core/src/minimax_h3/render.rs` |
| Denoise preview (resident decoder, PPM, terminal) | `crates/ds4-core/src/minimax_h3/vae_video.rs` (`PreviewDecoder`), `crates/ds4-core/src/minimax_h3/terminal.rs` |
| Device probe (`--info`) | `crates/ds4-core/src/minimax_h3/device.rs` + `native/h3/h3_device_gpu.cu`; gate `crates/ds4-core/examples/h3_device_gate.rs` (`make tests/h3_device_gate`) |
| Renderer CLI | `crates/ds4-cli/src/bin/h3-rs.rs` (`make h3-rs`) |
| Spark install and run script | `tools/minimax_h3/install-minimax-h3-spark.sh` |
| Frame comparator (PSNR, bbox, bands) | `tools/minimax_h3/ppm_diff.c` |
| Balloon composite for a delivered clip | `tools/minimax_h3/letterballoons.sh` |
| Task store and state machine | `crates/ds4-core/src/minimax_h3/video_api.rs` |
| Local v1 server | `crates/ds4-server/src/h3_video.rs`, `h3-server-rs` |
| Prompt tokenizer | `crates/ds4-core/src/minimax_h3/tokenizer.rs` |
| Inventory example | `crates/ds4-core/examples/minimax_h3_inspect/main.rs` |
| Reference-parser probe | `tools/minimax_h3/st_probe.c` |
| Reference-tokenizer probe | `tools/minimax_h3/tok_probe.c` |
| Reference DiT probe | `tools/minimax_h3/dit_probe.c` |
| VQ converter | `tools/minimax_h3/vq/` |
| Additive-family rule | `AGENTS.md`, "MiniMax-H3 (h3) is additive" |
