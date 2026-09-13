# Step 3.7 MQ83 contract fixtures

`metadata.json` records official GGUF metadata at
`stepfun-ai/Step-3.7-Flash-GGUF@0b69336d2fd2adfdef9c66e425f7778196c31482`,
plus the mixed artifact's source revision marker. The test constructs a
metadata-only file with placeholder vocabulary strings; it is not a tokenizer
or inference fixture.

`mq83.tsv` records the independently inventoried 754 tensor names and GGUF
dimensions, with the owner's locked mixed recipe: Q8 critical/shared matrices,
Q4_K routed edges/down, IQ2_XXS interior gate/up, F32 controls. Payload size is
83,001,512,448 bytes. No weight data is included.

Original model revision: `5f6244077ac62e04eec3f320501ff8c2b293373a`.

`mtp.tsv`, `vision.tsv` and their metadata JSON files come from the handoff's
`sidecar-inventory.json`. They preserve the official Q8 MTP and F16 projector
layouts without payload data. Tests use them independently of runtime specs.

`primitives.h` exports FP32 values from the handoff's
`ds4-initial/fixtures/step37-primitives.json` (Torch 2.11.0+cu130, H100 NVL,
seed 3707). It covers clamps 0/7/16 and 288-expert top-8 routing, including
logits near -30 that expose the legacy normalization floor.

## Native gates

`text-inputs.json` preserves the eight text token streams from the Spark
handoff's BF16 probes. Only the inputs are reused; BF16 outputs are not MQ83
goldens. The native driver is an integration gate. The independent oracle is
StepFun llama.cpp at `8f34864def3d351316aaea5ce9b4a06e12198d3c`.

```sh
python3 tests/fixtures/step37/make_text_cases.py scratch/step37 --label native
python3 tests/fixtures/step37/make_text_cases.py scratch/step37 --label oracle
make tests/test_step37_forward CUDA_ARCH=sm_121

# Run each model process serially under tools/host_memory_guard.py.
tests/test_step37_forward "$STEP37_MAIN" @scratch/step37/native.cases - 64 8
step-text-oracle "$STEP37_MAIN" @scratch/step37/oracle.cases - 64 8
python3 tests/step37_logits.py scratch/step37/oracle/arithmetic.f32 scratch/step37/native/arithmetic.f32

# Same execution width; whole-history KV is the independent ring control.
STEP37_VERIFY_KV=1 tests/test_step37_forward "$STEP37_MAIN" scratch/step37/fixtures/ring832.tokens scratch/step37/ring832.f32 64 8
```

`STEP37_MAIN` is the first of the nine MQ83 shards. Both drivers accept a
single input/output pair or `@manifest` with whitespace-separated pairs.
Every case gets a fresh graph/context; the model weights stay resident.
Each output has nine little-endian F32 rows of 128896 logits: one after
prefill, then eight after greedy decode. These short choices do not qualify
long generation or answer quality. Use the production Rust benchmark for
performance: initial native forward includes lazy weight materialization.

Build `text_oracle.cpp` as an executable linked to `llama` in the pinned
checkout. For diagnostics, `STEP37_NO_FA=1` disables FA. The optional
`oracle-f32-attn.patch` (apply with `git apply --unidiff-zero`), together with `STEP37_NO_FA=1 STEP37_F32_ATTN=1
NVIDIA_TF32_OVERRIDE=0`, retains F16 KV but evaluates attention in F32.
Keep the unmodified FA run as a separate control. The strict comparator
still fails cross-engine rows; see [the measured status](../../../docs/step37-initial.md).

`STEP37_TRACE=DIR` records prefill intermediates in either driver.
`STEP37_REPLAY=REFERENCE_DIR` in the native trace driver supplies the same
reference input to each layer. Replay is a local operator check, not a full
model trajectory. The KV proof instead requires byte-identical logits and
all live KV rows across full/ring layouts and a matched-width tail rewrite.

The public session gate keeps one model mapping and compares native session
logits/KV with a separate graph using identical chunk widths:

```sh
make tests/test_step37_session tests/test_step37_state tests/test_cuda_span_lease CUDA_ARCH=sm_121
tests/test_step37_state
tests/test_cuda_span_lease
tests/test_step37_session --policy
# Run under host_memory_guard.py; includes the public generation callback.
tests/test_step37_session "$STEP37_MAIN" scratch/step37/fixtures/ring832.tokens
```

The small span-lease regression needs CUDA but no model. Host-only state
and ledger tests reject stale logits and preserve only untouched checkpoints
on errors. Production Step sessions use `DS4_STEP37_PREFILL_CHUNK` (default
512, valid 1–4096, capped by context); 64 is the locked structural gate width.

Rust server output tests use `cargo test -p ds4-server --test step37_output`
and `--test chat_input`; `make -j1 test-server-parity` covers existing families.
The live `tests/chat_template_live.py` protocol passes all 18 requests with
`--reasoning-effort none --max-tokens 128`, then all 18 with
`--reasoning-effort high --max-tokens 256`, against a single Step server at
context 4096. It exercises all three APIs, buffered/SSE tool calls and result
continuations. Keep MTP and image qualification separate.
