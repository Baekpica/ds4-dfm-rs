# MiniMax-H3 documentation

Everything about the MiniMax-H3 FL2VA family lives under this directory. No
h3 document sits loose at the top of `docs/`; the same rule holds for
`tools/minimax_h3/` and `crates/ds4-core/src/minimax_h3/`.

Branch `feature/minimax-h3`. Model root on the Spark: `/mnt/h3-artifacts`.

| Document | Covers |
|---|---|
| [port-plan.md](port-plan.md) | The port plan: phases P0-P10, the Spark budget, the DiT rung, the render gate |
| [cheatsheet.md](cheatsheet.md) | Sizes, quality, reach, memory and layout in one page |
| [vq-recipe.md](vq-recipe.md) | The 13-bit GVQ1 v1 conversion of the DiT and its fallback |
| [openai-videos.md](openai-videos.md) | The OpenAI Videos API surface, its renderer bridge and its deviations |

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
| Host arithmetic (canvas, noise, sigmas, reuse window, Euler step) | `crates/ds4-core/src/minimax_h3/host.rs` |
| Packed sequence layout | `crates/ds4-core/src/minimax_h3/sequence.rs` |
| DiT forward, Euler denoise, block stream | `crates/ds4-core/src/minimax_h3/dit.rs` |
| DiT gate | `crates/ds4-core/examples/h3_dit_gate.rs` (`make tests/h3_dit_gate`) |
| Native DiT primitives | `native/h3/h3_dit_gpu.cu` |
| Audio VAE decode | `crates/ds4-core/src/minimax_h3/vae_audio.rs` |
| Video VAE decode | `crates/ds4-core/src/minimax_h3/vae_video.rs` |
| F32 weight store (both VAEs) | `crates/ds4-core/src/minimax_h3/vae_store.rs` |
| OpenAI Videos wire | `crates/ds4-core/src/minimax_h3/video_api.rs` |
| OpenAI Videos server | `crates/ds4-server/src/h3_video.rs`, `h3-server-rs` |
| Prompt tokenizer | `crates/ds4-core/src/minimax_h3/tokenizer.rs` |
| Inventory example | `crates/ds4-core/examples/minimax_h3_inspect/main.rs` |
| Reference-parser probe | `tools/minimax_h3/st_probe.c` |
| Reference-tokenizer probe | `tools/minimax_h3/tok_probe.c` |
| Reference DiT probe | `tools/minimax_h3/dit_probe.c` |
| VQ converter | `tools/minimax_h3/vq/` |
| Additive-family rule | `AGENTS.md`, "MiniMax-H3 (h3) is additive" |
