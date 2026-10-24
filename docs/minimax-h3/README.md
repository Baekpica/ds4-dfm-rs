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

Related, outside this directory:

| Surface | Path |
|---|---|
| Rust reader | `crates/ds4-core/src/minimax_h3/` |
| DiT tensor layout | `crates/ds4-core/src/minimax_h3/layout.rs` |
| Inventory example | `crates/ds4-core/examples/minimax_h3_inspect/main.rs` |
| Reference-parser probe | `tools/minimax_h3/st_probe.c` |
| VQ converter | `tools/minimax_h3/vq/` |
| Additive-family rule | `AGENTS.md`, "MiniMax-H3 (h3) is additive" |
