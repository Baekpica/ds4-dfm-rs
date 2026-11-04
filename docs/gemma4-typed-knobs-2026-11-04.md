# Gemma-4 typed option surface (2026-11-04)

Every arithmetic knob of the Gemma-4 family used to be a hand-parsed
`DS4_GEMMA4_*` getenv string read at the point of use in the C and CUDA
headers. Nothing validated a value, nothing listed it in `--check-config`, and
a typo took the zero branch: `DS4_GEMMA4_ATTN_SPLIT_SPAN=12x8` parsed as 12 and
was silently clamped to 16, `DS4_GEMMA4_MOE_MISS=cpux` meant evict, and a
misspelled boolean meant off. The Rust host already had a typed request
surface with named errors and a plan JSON; this unit moves the Gemma knobs
there.

## The surface

One table drives the environment parse, the `--gemma4-*` flags, the plan JSON
and the report. `crates/ds4-core/src/gemma4_knobs.rs::KNOBS`.

| Env | Flag | Default | Accepted | Effect |
| --- | --- | --- | --- | --- |
| `DS4_GEMMA4_ATTN_SPLIT_SPAN` | `--gemma4-attn-split-span` | 128 | 16..4096 | keys per split-K attention CTA |
| `DS4_GEMMA4_ATTN_SPLIT_MMA` | `--gemma4-attn-split-mma` | off | on/off | tensor-core full-layer split decode |
| `DS4_GEMMA4_ATTN_SPLIT` | `--gemma4-attn-split` | off | on/off | split-K decode attention |
| `DS4_GEMMA4_ATTN_FAST` | `--gemma4-attn-fast` | off | on/off | grouped-KV decode attention |
| `DS4_GEMMA4_ATTN_PREFILL` | `--gemma4-attn-prefill` | off | on/off | tiled prefill attention |
| `DS4_GEMMA4_ATTN_MMA` | `--gemma4-attn-mma` | off | on/off | tensor-core prefill attention |
| `DS4_GEMMA4_FUSE_ROPE` | `--gemma4-fuse-rope` | off | on/off | fused RMSNorm + RoPE |
| `DS4_GEMMA4_HEAD_GEMV` | `--gemma4-head-gemv` | off | on/off | family BF16 vocabulary head |
| `DS4_GEMMA4_ROUTER_DOT` | `--gemma4-router-dot` | off | on/off | family router projection |
| `DS4_GEMMA4_ROUTER_PAR` | `--gemma4-router-par` | off | on/off | parallel top-k selection |
| `DS4_GEMMA4_Q8_BATCH` | `--gemma4-q8-batch` | off | on/off | batched Q8_0 dense down |
| `DS4_GEMMA4_MOE_PERSIST` | `--gemma4-moe-persist` | off | on/off | persistent layer-book MoE decode |
| `DS4_GEMMA4_MOE_FUSED` | `--gemma4-moe-fused` | off | on/off | fused gate/up + GELU per route |
| `DS4_GEMMA4_VQ_PREFILL` | `--gemma4-vq-prefill` | off | on/off | direct VQ prefill GEMM |
| `DS4_GEMMA4_EXACT_PREFILL` | `--gemma4-exact-prefill` | off | on/off | one-row calls for a chunk (exactness gate) |
| `DS4_GEMMA4_Q4_DECODE` | `--gemma4-q4-decode` | off | on/off | family Q4_K decode |
| `DS4_GEMMA4_CPU_MOE` | `--gemma4-cpu-moe` | off | on/off | routed experts computed on the host |
| `DS4_GEMMA4_CPU_MOE_THREADS` | `--gemma4-cpu-moe-threads` | 20 | 1..64 | host expert pool width |
| `DS4_GEMMA4_MOE_HOST` | `--gemma4-moe-host` | off | on/off | slot fills read the model mmap |
| `DS4_GEMMA4_MOE_MISS` | `--gemma4-moe-miss` | evict | evict/cpu | cache-miss policy |
| `DS4_GEMMA4_PREFILL_CHUNK` | `--gemma4-prefill-chunk` | 0 = bind default (min(ctx, 512)) | 1..8192 | prefill rows per forward |
| `DS4_GEMMA4_FULL_KV` | `--gemma4-full-kv` | bf16 | bf16/fp8 | full-layer KV format |
| `DS4_GEMMA4_VQ_GEMV` | `--gemma4-vq-gemv` | on | on/off | direct VQ decode GEMV |
| `DS4_GEMMA4_VQ_GEMV_VARIANT` | `--gemma4-vq-gemv-variant` | auto | auto/warp/block | decode GEMV launch shape |
| `DS4_GEMMA4_VQ_GEMV_THREADS` | `--gemma4-vq-gemv-threads` | 256 | 32..512 step 32 | decode GEMV width |

Booleans are strict: `1`, `0`, `true`, `false`, `on`, `off`. Anything else is
an error naming the variable or the flag that carried it. Ranges are rejected,
never clamped.

## Where the value goes

    environment  ->  Gemma4Options::from_env()   (strict)
    --gemma4-*   ->  Gemma4Options::set_flag()   (same validation)
                         |
                         v
                  ServingRequest.gemma4          (server)
                  BenchArgs.gemma4               (bench)
                         |
                         v
              ModelOpenOption::Gemma4(options)
                         |
                         v
      ds4_bridge_model_open_options.gemma4 (by value, validated)
                         |
                         v
               ds4_engine_options.gemma4
                         |
                         v
        ds4_gemma4_knobs()  process-wide, installed at ds4_engine_open
                         |
                         v
      g4_attn_fast(), g4_split_span(), g4_moe_*(), ... read the fields

`ds4_gemma4_options.present` is the switch. A Rust host always fills the whole
struct, defaults included, so every field is an effective value and no helper
parses a string. A C host (the `tests/` oracles and the deprecated `ds4-*-c`
binaries) leaves `present` clear and keeps the legacy getenv path unchanged.

The one compatibility repair: `DS4_GEMMA4_PREFILL_CHUNK` used to be read only
by the C bind, so the host quote priced the bind default while the engine ran
the environment's larger chunk. The server now folds the family chunk into
`ServingRequest.native_chunk` before the quote, so the quote, the plan and the
engine agree.

## Plan JSON and `--check-config`

`ResolvedPlan::to_json` carries a top-level `gemma4` object with all 25 values
and an `explicit` flag; the human report prints the same values as `=`
assignments. `--check-config` therefore answers "what will run" instead of
listing only the non-Gemma controls. Unknown names under the `DS4_GEMMA4_`
prefix are reported as warnings (`gemma4_env_unknown`) rather than silently
ignored.

Diagnostics stay environment variables, by design: `DS4_GEMMA4_DEBUG`,
`_CPU_MOE_DEBUG`, `_CPU_MOE_DUMP`, `_GPU_MOE_DUMP`, `_HASH_LAYER(S)`,
`_Q4_DECODE_VERIFY`, `_MOE_PERSIST_VERIFY`, `_VQ_VERIFY`, `_ATTN_CMP`. They
switch printing, not arithmetic.

## Gates

Unit tests (`cargo test -p ds4-core --lib gemma4_knobs`): 12 tests covering the
defaults, inclusive ranges, rejection instead of clamping, strict booleans and
enums, the 32-step GEMV width, the flag route naming the flag, the FFI mirror,
the plan JSON, and the knob-table/JSON-key bijection.

Typo gate on the 4070 (exit status checked):

    DS4_GEMMA4_ATTN_SPLIT_SPAN=12x8 ./ds4-server ... --check-config
    -> ds4-server-rs: DS4_GEMMA4_ATTN_SPLIT_SPAN wants an integer in 16..4096
       (got '12x8'), exit 2
    ./ds4-server ... --gemma4-cpu-moe-threads 65 --check-config
    -> ds4-server-rs: --gemma4-cpu-moe-threads wants 1..64 (got '65'), exit 2

Parity gate on the 4070 (ctx 64, greedy, the full fast set + FP8 full-layer
KV): the pre-change env-driven binary and the new typed-surface binary produced
byte-identical `tokens-64.json`, `cycles-64.json` and the 2.28 MB
`frontier_000064.logits.json`:

    tokens [506, 41348, 39844, 236751]
    sha256 tokens   5b8579075fabadd84fc70b96502a693a03bca77ab0444708a1e71425acfff4bc
    sha256 logits   b6155bb6dc6d4dee1f0ca7a5a809a228a34b471f92e0c35c5d089f0e5a12bd4a

A second pair with `DS4_GEMMA4_ATTN_SPLIT_SPAN=64` against
`--gemma4-attn-split-span 64` matched the same three artifacts.

## Files

- `crates/ds4-core/src/gemma4_knobs.rs` - the surface, validation, FFI mirror.
- `crates/ds4-core/src/serving.rs` - request field, plan echo, warnings.
- `crates/ds4-core/src/lib.rs` - `ModelOpenOption::Gemma4`, FFI plumbing.
- `crates/ds4-sys/src/lib.rs` - the ABI struct.
- `native/bridge/ds4_bridge.{h,c}` - carry the struct into the engine options.
- `ds4.h`, `ds4.c` - the engine field, install at open, process-wide accessor.
- `ds4_gemma4_knobs.h` - the C struct and the typed/legacy accessors.
- `ds4_gemma4.h`, `ds4_gemma4_gpu.cuh`, `ds4_gemma4_graph.inc`,
  `ds4_gemma4_stream.inc`, `ds4_gemma4_bind.inc`, `cuda/gemma4_cpu_moe.cuh` -
  the helpers now read the typed fields.
- `crates/ds4-server/src/bin/ds4-server-rs.rs` - flags, request, open option.
- `crates/ds4-cli/src/bench.rs`, `crates/ds4-cli/src/lib.rs` - bench and CLI.
- `crates/ds4-perf/src/knobs.rs` - the knobs are sweepable with the same
  bounds.
