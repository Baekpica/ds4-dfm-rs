# FFI contract

Rust talks to the native inference runtime through **one** header:
[`native/bridge/ds4_bridge.h`](../../native/bridge/ds4_bridge.h).

NVTX is host observability: `ds4-cli` uses the optional official NVIDIA Rust
SDK directly. Profiler symbols and handles do not belong in this ABI,
`ds4-sys`, or `ds4-core`. The optional `ds4-perf-gpu` helper isolates
profiling-only CUDA/CUPTI calls and callbacks behind maintained cudarc bindings.
That adapter is independent of this inference ABI; see the
[profiling contract](../ds4-perf.md).

```text
Rust application
    ↓
ds4-core (safe)
    ↓
ds4-sys (unsafe)
    ↓
ds4_bridge.h          ← the source-matched inference ABI
    ↓
ds4_bridge.c
    ↓
existing ds4.h / ds4.c / ds4_cuda.cu internals
```

## Hard rules

1. Do **not** bindgen `ds4.h`, `ds4_gpu.h`, or `cuda/mmq/*.h`.
2. Keep native engine/device structures opaque. Only the documented
   host-owned metadata descriptors and scalar/POD options cross by layout.
3. Do **not** put `CUstream`, device pointers, MMQ descriptors, or
   graph execs in Rust application code.
4. `unsafe` belongs in `ds4-sys` or a tiny reviewed native/OS adapter.
   Application policy/runtime code showing `unsafe {` is an architecture
   defect; localized POSIX/linenoise adapters must carry a `SAFETY` rationale.
5. Errors cross the boundary as `int` + caller-provided `char *err`
   buffer, matching the existing C session helpers. No C++ exceptions.
   No Rust panic across FFI.
6. Call-scoped strings and token buffers are borrowed until return. The
   bridge must copy retained data or document a longer borrow backed by a
   live Rust owner (for example, model vocabulary strings).
7. Every successful `*_create` / `*_open` has exactly one `*_free`.
   Rust `Drop` is the only application-side destructor.

## Opaque handles and safe ownership

`ds4_bridge_model`, `ds4_bridge_session`, snapshots and batch contexts are
opaque handles. The [safe wrappers](../../crates/ds4-core/src/lib.rs) own them
through `NonNull` and matching `Drop` implementations. `Session<'m>` borrows
its `Model`; snapshots and batches preserve the same model lifetime. Native
execution handles are thread-affine, not transferable application state.

Do not typedef these handles to `ds4_engine` / `ds4_session`. The bridge
structs may contain those pointers; Rust callers must not inspect them.

## Selected ABI operations

The header is the authoritative symbol list. This table explains contracts
used by the host and retained C parity helpers; it is not an export-count
freeze or permission to bind native internals.

| Function | Meaning |
|---|---|
| `ds4_bridge_bind_plan_check` | host inventory + required-name table; 0 if native can consume it |
| `ds4_bridge_bind_plan_match` | host vs native slot identity (name/need/found/type/dims/offsets/bytes/shard + remap) |
| `ds4_bridge_model_open` | mmap-backed `ds4_engine_open`; optional `opt->plan` is checked first; optional `opt->tensors` replaces `parse_tensors` (owned name copies); optional `opt->shape` applies the pinned C literal + DeepSeek compress table and skips `config_validate_model`; optional `opt->vocab` applies host token/merge/specials and skips `vocab_load`; optional `opt->bind` is the host name→tensor-dir index so `model_find_tensor` skips the C name walk and the main-model `weights_validate_layout` |
| `ds4_bridge_model_boot_prewarm` | idempotent `ds4_engine_boot_prewarm`; server calls it after batch placement and before listen/accept |
| `ds4_bridge_model_free` | `ds4_engine_close` |
| `ds4_bridge_session_create` | `ds4_session_create` |
| `ds4_bridge_session_free` | `ds4_session_free` |
| `ds4_bridge_session_sync` | prefix tokens → `ds4_session_sync` |
| `ds4_bridge_session_sync_cb` | one `ds4_session_sync` plus call-scoped durable `prefill_chunk` frontiers |
| `ds4_bridge_eval` | `ds4_session_eval` one token |
| `ds4_bridge_session_argmax` | greedy next id |
| `ds4_bridge_session_pos` | native committed timeline (host `SessionLedger` is authoritative) |
| `ds4_bridge_session_ctx` | session context length |
| `ds4_bridge_session_rewind` | `ds4_session_rewind` |
| `ds4_bridge_session_invalidate` | `ds4_session_invalidate` |
| `ds4_bridge_session_generation` | Inc 5a content generation |
| `ds4_bridge_session_prefill_cap` | chunk cap used by the native graph |
| `ds4_bridge_session_exaone_rewind_span` | EXAONE sliding-window reuse span |
| `ds4_bridge_session_sample` | `ds4_session_sample` (caller-owned rng) |
| `ds4_bridge_session_save_payload` | path wrapper over `ds4_session_save_payload` (native writes header+tokens+GPU tail) |
| `ds4_bridge_session_load_payload` | path wrapper; `payload_bytes` = file size |
| `ds4_bridge_session_load_payload_range` | path + checked `offset`/`length`; native consumes exactly that embedded DSV4 range |
| `ds4_bridge_tokenize_text` | caller-owned `int32_t *` + cap + `n_out` |
| `ds4_bridge_tokenize_rendered_chat` | same buffer contract, special-token path |
| `ds4_bridge_token_text` | caller-owned byte buffer; C frees the malloc |
| `ds4_bridge_token_eos` | engine EOS / family EOT |
| `ds4_bridge_token_is_stop` | `ds4_token_is_stop` (1/0) |
| `ds4_bridge_model_id` | `ds4_engine_model_id` (syntax dispatch) |
| `ds4_bridge_mem_census_snap` | process-global CUDA census image (seqlock + last-stable torn cache); `supported=0` when the backend keeps no census |
| `ds4_bridge_mem_observe_snap` | typed observation (`status`/`source` + free/total/cuda_free/meminfo) |
| `ds4_bridge_mem_substrate_outstanding` | `ds4_gpu_substrate_outstanding` (0 on Metal/CPU stubs) |

`ds4_bridge_session_sync_cb` borrows its token buffer, callback, and userdata
until the call returns. The callback runs synchronously on the calling thread,
receives only native `prefill_chunk` events (never `prefill_display`), and is
cleared before both successful and failed returns. It must not re-enter
sync/eval. The safe wrapper exposes a scoped `PrefillCheckpoint` instead of the
session; that value permits exact `prompt[..current]` inspection and synchronous
payload save only. Rust contains callback panics before they reach C, and the
host ledger commits the full prompt only after native sync succeeds.

Family/shape identify (`ds4-core::identify_gguf`) is host-owned: it
mmaps GGUF metadata and does **not** call `ds4_bridge_model_open`.
Tensor inventory + `-0000N-of-` shard remap (`ds4-core::TensorInventory`)
is also host-owned; `Model::open` builds that plan before the bridge
call. The family `weights_bind` name catalog (`ds4-core::BindPlan`) is
host-owned and is passed as `ds4_bridge_bind_plan` so native
`ds4_bridge_bind_plan_check` consumes it before `ds4_engine_open`.
The full host inventory (`ds4_host_tensor_dir` / `opt->tensors`) is
installed for that open. When present, native skips `parse_tensors`
and applies the host table (names are `strdup`'d; the Rust
`CString`s die when `Model::open` returns). Optional `opt->shape`
(`ds4_host_shape`) is the host `config_validate` result: native
applies the pinned `g_ds4_shape` literal plus the verified DeepSeek
compress table and skips C `config_validate_model`. Optional `opt->bind` (`ds4_host_bind_map`) is the host-resolved
name→tensor-dir index: when installed, `model_find_tensor` uses it
instead of scanning `m->tensors`. Names not in the map (MTP/DSpark
siblings) fall back to the C walk. When that map is installed, native
also skips the main-model `weights_validate_layout` because
`Model::open` already ran the host table. Host owns the DeepSeek
MTP/DSpark sibling name and expected-layout catalogs (`mtp-flash` /
`dspark-pro`) and can resolve/validate a sibling `BindPlan` against a
host inventory (`--bind-names mtp-flash --bind-plan`). Optional
`opt->mtp_path` / `opt->dspark_path` attach the DeepSeek-only sibling
support models through the same open; the optional `opt->mtp_bind` /
`opt->dspark_bind` maps are host-resolved name→index tables for THAT
sibling's tensor dir (`Model::open_with_support` runs sibling
resolve + expected-layout validation first). Native keeps them in
separate slots, swaps each into the active map only around its own
sibling open+bind window, and skips that sibling's C layout check —
sibling pointer assignment stays C. After base
`weights_bind`, native clears host tensor-dir / bind-map / vocab /
shape so a later sibling `model_open` cannot apply the base GGUF
tables. The C CLI/server
leave tensors/shape/vocab/bind and the sibling paths/maps NULL, so
the GGUF cursor walk, C validate,
C `vocab_load`, C `model_find_tensor` name walk, and C
`weights_validate_layout` (base and sibling) stay the C oracle
behavior. Weight upload / VMM bind stay native. Tokenizer encode / decode / special / stop
(`ds4-core::Vocab`) is host-owned; `Model` keeps the `Vocab` so
native token-string pointers stay valid for the engine lifetime.
`--tokenize` / `--validate` still must not open the engine. Session timeline / sync plan /
rewrite / rewind / generation (`ds4-core::SessionLedger`) is host-owned;
`Session::pos` / `generation` read the ledger. The DSV4 payload prefix
(13×u32 LE header + token ids; magic `DSV4`, version 3) is host-owned
(`ds4-core::payload`). `Session::load_payload_range` reads only that
prefix, rejects overflow / out-of-file ranges before FFI, and passes the
bounded range to native. A bridge-side preflight failure preserves the
existing host ledger; failures after native load begins follow the native
generation and invalidate the host checkpoint. GPU / logits / family tensor
tails stay native.
The Inc 5 continuation registry (`ds4-server::ContRegistry`) is
host-owned: publish / resolve / hold / pin / TTL / bank claim do not
cross FFI. Weight bind and native prefill/eval still go through the
bridge. `Model` tokenize/stop/eos/`token_text` use the host `Vocab`.
The bridge tokenize helpers remain for the C engine path.

Open options stay a small C struct of scalars, `const char *`
paths, and optional borrowed plan/tensor-dir/shape/vocab/bind pointers
(model path, backend enum, thread count, defer-prewarm, then the
optional `mtp_path` / `dspark_path` / `mtp_bind` / `dspark_bind`
sibling fields appended at the end). Do
not pass `ds4_engine_options` by value into Rust — that struct will
keep growing on the C side and is not the ABI.

Token arrays are `const int32_t *` + length. Do not export
`ds4_tokens`.

## Ownership

| Object | Allocated by | Freed by | Rust view |
|---|---|---|---|
| `ds4_bridge_model` | bridge | `ds4_bridge_model_free` | `NonNull`, `Drop` |
| `ds4_bridge_session` | bridge | `ds4_bridge_session_free` | `NonNull`, `Drop` |
| error buffer | Rust caller | Rust caller | `&mut [u8]` scratch |
| token scratch | Rust caller | Rust caller | `&[i32]` |
| GPU tensors / graphs | native session | native session free | invisible |

The bridge must not return interior pointers into engine arenas.

## Native boundary scope

These remain C-internal or later *narrow* additions with their own
review, not dump-the-header expansions:

- `ds4_gpu_*` (thousands of lines of tensor ops)
- `ds4_batch_ctx` / reclaim / governor ledgers (the `/metrics` `/v1/stats` text is host-owned; live census + observation copy through `ds4_bridge_mem_*_snap`, not a bindgen of `ds4_mem_census.h`)
- family-specific test hooks (`ds4_engine_motif3_*`, dots3 logits)
- Metal / CUDA types
- distributed pthread/socket internals
- KV store `kv_buf` / malloc arena

KV and distributed protocols use explicit codecs and file-format modules
in Rust. They do not ride on a giant bindgen. New native exports require a
concrete inference need, reviewed ownership and relevant parity evidence;
keep host policy and profiling annotations out of this boundary.

## Linkage

Native Rust builds link the selected Make backend objects plus
`native/bridge/ds4_bridge.c`. The CUDA Driver API, CUDA Runtime, and
cuBLAS stay on the native link line (`-lcuda -lcudart -lcublas`).

Rust must not add a second CUDA stack (cudarc-driven kernels, a
second context, a second VMM arena).

## Versioning

The bridge is a source-matched contract. A signature or descriptor change
updates the header, native implementation, `ds4-sys`, safe wrappers and
relevant parity tests together. The production Rust consumer is `ds4-sys`;
C proof programs may also include the bridge header.

There is no promise of binary compatibility with out-of-tree callers. The
[v0.1.0 boundary](../releases/v0.1.0.md) establishes stable ownership and
behavior; it does not freeze an obsolete migration-era symbol count.
