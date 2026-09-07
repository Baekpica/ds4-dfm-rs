# Parity matrix

Frozen migration evidence. Preserve its fixture definitions and recorded
results; use the [v0.1.0 ledger](../releases/v0.1.0.md) for current release
qualification. Passing this historical matrix is not a new candidate pass.

> **Recorded result (2026-08-31 KST):** the complete `v0.6.5-dfm` live matrix is
> green at `eb4ba77`: PASS 57 + PASS* 3 (C-shared E-2, E-3, E-6), FAIL 0,
> BLOCKED 0. The final host-only fix and 13/13 cheap/parity re-audit are green
> at `d126e56`. Qwen Q5+Sidecar owns the campaign's only two-hour soak;
> DeepSeek retains its ordinary functional, KV, API, proof, and performance
> cells.

## Qwen v0.6.5 delta re-stamp (`8006198`)

Full identities, artifact scope, commands, hashes, and raw-evidence paths are
recorded in
[QWEN_V065_RESTAMP_2026-08-31.md](QWEN_V065_RESTAMP_2026-08-31.md).

| Axis | Result |
|---|---|
| Numerical | Q5 loader, SSD-PLE, QSA, MoE, GDN, two-bank batch, embedded MTP, and shared MMQ gates PASS |
| Token | Qwen tokenizer goldens PASS; C→Rust→Rust→C text and image outputs are exact |
| KV | same-image reuse, changed-pixel rejection, cross-worker disk restore, two-bank disk/partial-fork lifecycle PASS |
| Wire/API | image Chat, Responses, and Anthropic PASS; continuous width 2, static width 2, serial fallback, `/v1/models`, and `/v1/stats` PASS |
| Performance | text ABBA: prefill 99.97%, decode 100.00%, TTFT +0.95%, worker VmHWM +4.40%, GPU residency exact; image prefill 100.20%, TTFT +0.32%; exact-262K prefill 99.64% |
| Hard gates | 7,202.3-second Qwen soak: 3,610/3,610 requests, 158 barriers, 79 images, RSS drift 0.0011%/0.0%, swap 0; 262K direct: 248,320 finite logits, argmax 198 exact, f32 mismatches 0; KV 4-way, MTP 1.01→2.00 tok/step with exact output, forced static route 2/2 PASS |

This is the requested Q5 main GGUF plus shared Sidecar scope. Safetensors,
resident BF16, Q6, and their reference gates were intentionally not loaded.

## Current v0.6.5 full re-stamp (`eb4ba77` + `d126e56`)

Evidence root:
`scratch/rust-host-live/v065-full-restamp-20260831-184200/`. The classified
live ledger is `gates-results.txt`; the final host/parity audit is under
`final-audit-d126e56/`.

| Group | Logical result |
|---|---|
| G0 cheap/parity/shared kernels | 13 PASS |
| G1 DeepSeek KV/API/proof | 23 PASS |
| G2 Motif family/shutdown/barrier/ABBA | 9 PASS; E-4/E-5 fixed and re-run green |
| G3 Solar | 8 PASS + 1 PASS*(E-3) |
| G4 K-EXAONE | 2 PASS + 1 PASS*(E-2) |
| G5 dots3-note | 2 PASS + 1 PASS*(E-6) |

All three PASS* cells were re-run against a C control in this campaign. E-2
and E-6 used exact `v0.6.5-dfm` sm_121a test binaries; E-3 reproduced in the
frozen C server. The candidate and exact-tag Dots3 runs produced identical
`first=63594/63594`, `batch_cos=0.98607464`, and
`batch_nrmse=0.201895`. No C-pass/Rust-fail cell remains.

The current Motif ABBA ratios are 99.90% prefill, 99.33% decode, +1.07% TTFT,
and +1.03% host VmHWM, with exact GPU residency and zero swap. The proof set
passes smoke 2/2, captured/eager long 6/6, tracked v0.6.5 OPP-C 5/5, and
C→Rust OPP-C 5/5. After the host-only socket compatibility fix, all 13 G0
cells, clippy, final default/C builds, and `ds4-eval --self-test-extractors`
pass at `d126e56`.

The detailed incremental evidence below is retained for provenance. Statements
that a later phase was “pending” describe the named historical commit only and
are superseded by this current re-stamp.

## Historical post-promote §10 (`cb11c0b`)

Evidence: `.omo/evidence/task-53-rerun-cb11c0b.txt`.
PASS 55 + PASS* 5 (E-2..E-6) + FAIL 0 + BLOCKED 0.

| Axis | Result |
|---|---|
| Numerical | family kernels + mmq PASS; family IMA/decode gaps = E-2..E-6 (C-shared) |
| Token | DeepSeek logprob vectors PASS; Motif batch expected-vector miss = E-4 |
| KV | DeepSeek 4-way ordinary/continued/tool-map/periodic/evict/partial PASS; Motif shutdown 4-way PASS |
| Wire/API | DeepSeek 4 surfaces × serial/cont/static PASS (C=`ds4-server-c`, Rust=`ds4-server`); barrier width=2 PASS |
| Performance | Motif ABBA PASS (prefill 99.95%, decode 100%, TTFT +1.1%, HWM +1.0%); proofs smoke/long/opp-c/rust-opp-c PASS |

The historical two-hour mixed DeepSeek soak is not repeated. This campaign's
only long soak was the now-green Qwen Q5+Sidecar two-bank gate; DeepSeek still
runs the ordinary functional, parity, proof, and performance cells.

“Rust works” is not `cargo build`. A host change is accepted only
when all five axes that it can affect are green. CUDA optimization
commits already require a correctness proof **and** a speed proof
(`AGENT.md`). The same bar applies to migration commits.

## Five axes

| Contract | Requirement | Oracle |
|---|---|---|
| Numerical | logits / fixtures stay inside existing tolerances | family CUDA fixtures, `test-mmq-parity`, `test-model-family-kernels`, CPU-vs-GPU refs where they exist |
| Token | deterministic path emits the same token ids | greedy / temp-0 benches, tokenizer goldens, motif/solar sentinels |
| KV | save / restore / reuse / prefix / eviction identical | KVC 4-way matrix; `ds4_session_save_payload` / load; bank restore |
| Wire/API | schema, event automaton, route engagement, semantic equivalence | `docs/ds4-api-surface-matrix.md` + `ds4_test --server` |
| Performance | no unexplained regression vs C on GB10 | [BASELINE.md](BASELINE.md) thresholds; AB/BA or ABBA |

Live sampled HTTP bodies are **not** a byte oracle. Continuous temp-0
can jitter. Live gates check schema, automata, routing, and semantic
equivalence — the same rule the API matrix already records.

Do not refresh goldens to hide a Rust miss.

## Phase gates

### Phase 1 — workspace + FFI skeleton

- `cargo check -p ds4-sys` (and empty member crates) succeeds
- Header exists; no bindgen of `ds4.h`
- No production binary replaced

### Phase 3 — FFI shadow (`ds4-rs` / `ds4-bench-rs` / `ds4-agent-rs`, historical)

Same model, same arguments, C vs Rust wrapper around the **same**
C core:

| Check | Pass |
|---|---|
| Token output | Fixed-seed sampled C/Rust stdout is byte-identical (`ae12f463...`); post-port greedy stdout is also exact (`a12b7cb4...`). Fixed-seed non-TTY thinking output is exact in both modes (`--think`: 292 bytes, `ad6d107f...`; `--nothink`: 52 bytes, `ae12f463...`). Same-prompt server pair returned `Hi.` / stop / 13+2 |
| KV behavior | save/load round-trip matches C (synthetic). Live Rust benchmark restored the 64-token frontier before incrementally syncing to 128; checkpoint sizes were 18,632,800 and 24,556,624 bytes |
| Prefill / decode tok/s | Local C/Rust benchmark ABBA is green below; full production-path performance remains pending |
| Memory | sequential; C server peak `nvidia-smi` 102833 MiB, Rust server 99077 MiB; teardown + `clear_cache` returned ~115 GiB available |
| Agent shadow | `make test-agent-parity` is green: the 7,579-byte built-in prompt, fixed-input datetime message, selected non-TTY projector tapes, both supported DSML openers, and the 393,216-token high-thinking boundary are covered. The release binary links over the existing native core; live generation, transcript token tape, streaming, malformed/in-think DSML, tools, KV, MTP, and distributed paths remain pending |

If FFI alone regresses performance, **stop**. Do not port subsystems
on a dirty seam.

### Phase 4 — KV store (historical incremental evidence; current gate green)

Preserve magic `KVC`, file version `1`, payload ABI `2`
(`ds4_kvstore.c`), 48-byte fixed header, endianness, key = rendered
byte prefix SHA, eviction and prefix policy, trailer hooks.

Hard gate (all four):

```text
C save    → Rust load
Rust save → C load
Rust save → Rust load
C save    → C load
```

No new checkpoint format.

The isolated seams are green for metadata-only KVC indexing (including a
sparse 4 GiB payload fixture), bounded text-prefix comparison, embedded DSV4
range validation, and a file-backed Rust payload writer. The writer stages the
complete stream before eviction, protects a replaced destination, preserves
the C-compatible O(trailer) fast path, and produces payload/trailer bytes read
exactly by the C oracle. The no-GPU C oracle also proves nonzero seek, exact
length, EOF rejection, and `uint64` overflow rejection.

The ordinary serial Motif-3 no-think/no-tools slice is now live-green under the
policy both servers implement (`cold=0, continued=0`). In
`scratch/rust-host-live/rust-kv-no-think-final-IHtJue/`, C and Rust generated
the same 301,972,392-byte
`5522739e3318261231ec3ddd49a24da9e66a5118.kv`; visible text and payload bytes
were exact, apart from volatile header counters/timestamps after reads. All four
restart cells returned `RESTORED_OK` with 6,896 cached prompt tokens and 15
new prompt tokens:

```text
C save    → Rust load   PASS
Rust save → C load      PASS
Rust save → Rust load   PASS
C save    → C load      PASS
```

Request SHA-256 values were `a765063d149094c6542f42e7156735a8d466cc7f4eab1eac71e0fc40a7cbadbb`
(A), `083ab2beb504500ee28dafe0d8184b53d7795d6958712c78b1c045cdf2c3d827`
(evict), and `4ae9a992140aa7cd98a7ec0467b299c66ffcb4154c2429ca202239caa8cc9d4b`
(restore). The 94,162,541,472-byte GGUF was
`Motif-3-MQ87-88-FIT.gguf`. The C binary was
`04f25a86040940674984c35160d2e4eec7f6c6d4e30313815ce10309cd57e662`
and the Rust four-way binary was
`dae70ae2f909fe6c0713a609a285c272db8d1b6ec0e2856a70bd361753625ff4`.
The common launch contract was serial (`DS4_SERVER_COALESCE_MAX=1` /
`--cont-width 0`), ctx 8192, max output 128, disk budget 2048 MiB, and minimum
checkpoint 512 tokens; C additionally set both unimplemented policy knobs to
zero. Follow-up `e5830c218361ec2288e98992cd388be1a5a24a04` corrected the Rust
timing porcelain so a live Rust cache hit now reports
`prefill_tokens=15` and `prefill_cached_tokens=6896`, matching C (post-fix
binary `52471e49c4c22bde2f1abb063755847ff5d738da705896e925e8b7468b8ec197`).

The initial comparable single cells exposed a real lifecycle regression: C was
879--934 ms TTFT while Rust was 1,590--1,715 ms because the Rust server opened
with deferred boot prewarm and never called it. Commit `f4a632e` added the
opaque prewarm call after batch placement, preserving C's placement contract.
The same model, request C, KVC, serial lane, and cache-clear protocol then gave:

```text
C     880.9 ms
Rust  836.0 ms
Rust  860.4 ms
C     897.5 ms
```

C mean was 889.2 ms and Rust mean 848.2 ms (-4.61%), so the scoped restore TTFT
gate is green. All four cells returned `RESTORED_OK`, 6,896 cached tokens, and
15 computed tokens. Commit `4fc8ef4` then matched C's reported prefill timing
scope (computed suffix sync only): live Rust was 69.2 tok/s versus C
68.2--69.0, with TTFT still 840.4 ms. The ABBA binary SHA-256 was
`00da345a5e09887d82b514c80b4674aa37311e0903ad50ff9cae82b1db3a34b3`;
the timing-scope binary was
`ea714d4d7ae4e2618bf052aa842d40e290cd978ac404ac1f39aa15f5d76779f9`.

The C-default ordinary cold slice (`cold_max=30000`, trim 32, align 2048) is
live-green for this fixture under
`scratch/rust-host-live/rust-cold-default-0IeCd9/`. For the 6,782-token Motif
fixture, both hosts selected the same 6,144-token prefix. The rendered text
SHA-256 was
`ebd540ec7d9b9059649a1066c222dae4b02ca54bed54570da6fda0b81fd66f5e`
and the payload SHA-256 was
`0f0af1d1e189561aee24534cb4918bc292ef68a418522a51e615ac0bfade5862`
for both C and Rust. Rust→Rust, Rust→C, and C→Rust restart cells restored the
prefix and emitted the same 106-token output
(`49b5f8a8b7c4e3aa2f768fd3afd29ad33a1d2cc327fe0361a0cd6d2c21afcdd9`)
at 2,038.4/2,078.9/2,018.0 ms TTFT. Header timestamps and hit counters remain
intentionally volatile; the text and payload bodies are exact. These cells are
a correctness/cross-read gate, not a cold-policy ABBA performance claim.

The ordinary serial continued final-sync call/order is CPU-green, while the
decode frontier is live-green under
`scratch/rust-host-live/continued-fourway-oJTYrf/`. C and Rust used ctx 8,192,
`cold=0`, `continued=6800`, trim 0, align 1, and request A
(`a765063d...`). Both wrote
`411e439f9951c2df3addaa93e73cabee465bf0b2.kv`, 300,423,338 bytes,
reason 2, ext 0, model 3, tokens 6,800, ctx 8,192. The rendered-text SHA-256
was `025231236a8afc77532eeecdb6ea16b3f94b51456641e29224438813605764ae`
and payload SHA-256 was
`be4d59a936d236809bfeae3953408b067df3a5dac86212b85df21ecf98550838`
for both hosts; their 114-token answer was exact.

The first restart fixture was intentionally retained as a negative oracle:
Motif live generation contains `<think></think>`, while a closed no-think
assistant history drops that empty pair in both C and Rust. It therefore
misses the record by construction. The positive fixture prepends the same
empty pair to the assistant replay (SHA-256 `6bbd72ee...`) so its rendered
bytes preserve the live KVC prefix. Four isolated/cross-host loader cells
then passed:

```text
C save    → Rust load   PASS  cached=6800 computed=111 output=4
Rust save → C load      PASS  cached=6800 computed=111 output=4
Rust save → Rust load   PASS  cached=6800 computed=111 output=4
C save    → C load      PASS  cached=6800 computed=111 output=4
```

All returned `RESTORED_OK` with semantic response SHA-256 `f61bf763...`.
The C binary was `04f25a86...`; the live-tested Rust binary was
`beadabe4...`. The subsequent cheap-frontier build `42a1ddfb...` removes the
non-due token-vector copy and passed the full CPU/parity/link gates; it did not
need another 100 GiB live matrix because checkpoint bytes and ordering are
unchanged. Teardown and `clear_cache` ran between resident models, and the
final host had no compute application or listener. These cells prove
decode-frontier continued correctness and cross-read; the final-sync
call/order is covered by the CPU integration test. Neither is a performance
gate.

The scoped ordinary-serial DeepSeek tool-map slice is live-green under
`scratch/rust-host-live/tool-fourway-20260825/` at `cc1bf0f`. The producer
fixture used OpenAI Chat, no-think, `pair_values(a: integer, b: integer)`, and
asserted `finish_reason=tool_calls`; literal `tool_choice=required` was not
used because that path ends in the same tool parse error on the C oracle. C
and Rust both sampled arguments `{"a":1,"b":2}` with 365 prompt + 59 output
tokens, then an unrelated `EVICT_OK` request saved the live state.

Both hosts wrote a 25,851,198-byte record named
`29cf875c11df369237f528cb8155b8d1979f5b32.kv`: reason 3, ext flags 1,
model 0, payload ABI 2, 424 tokens, 1,913 text bytes, 25,848,936 payload bytes,
and a valid one-entry 297-byte `KTM\x01` trailer. The rendered-text SHA-256 is
`20e492de0058ac24efd1bc22b72b7467caad69d7cec8d3d0468b1bde2ddb7c11`
and the payload SHA-256 is
`f427f9263acf8316bf9bd9de03467a52cb77909ae16882e5316148f33cbf9c8e`
for both producers. The 244-byte sampled DSML occurs in the checkpoint text
and in each trailer. The trailers differ only in the process-unique tool ID;
header timestamps are run-specific:
`call_361e9fbdea60bfa15a23be22020fede1` for C and
`call_aefef96b236c98f22ce68ef16d907c57` for Rust.

Each loader ran in a fresh process against an isolated copy of the selected
record. Its assistant-history arguments were deliberately reordered to
`{"b":2,"a":1}`, so a canonical re-render cannot match the saved prefix
without KTM restoration:

```text
C save    → Rust load   PASS  cached=424 computed=28 output=5 TTFT=436.9 ms
Rust save → C load      PASS  cached=424 computed=28 output=5 TTFT=493.1 ms
Rust save → Rust load   PASS  cached=424 computed=28 output=5 TTFT=435.8 ms
C save    → C load      PASS  cached=424 computed=28 output=5 TTFT=485.8 ms
```

Every cell returned `RESTORED_OK`, finish `stop`, prompt tokens 452,
`prefill_cached_tokens=424`, and the serial OpenAI Chat route. C binary
SHA-256 was `04f25a86040940674984c35160d2e4eec7f6c6d4e30313815ce10309cd57e662`;
Rust binary SHA-256 was
`6e920f1d46e79ac5f733ce3243c5b1799091d80f885d57025d0fb143bf6975ef`.
The complete process was sequential; every resident exited before
`clear_cache`, and the final host had no GPU compute application.

#### Intermediate-prefill continued checkpoint

The scoped DeepSeek ordinary-serial gate is live-green under
`scratch/rust-host-live/intermediate-prefill-fourway-40gDCF/` at `8361116`.
The deterministic OpenAI Chat/no-think producer request SHA-256 is
`efecb01d300f7e4f0e67420a7e9ce3be286d9665d716c55662aaa67e0ce79fd7`:
6,778 raw user tokens plus four chat-framing tokens made a 6,782-token prompt.
With ctx 8,192, `cold=0`, `continued=4096`, trim 0, and align 1, the 4,096
frontier is reached inside prefill and the final prompt plus output stays below
the next 8,192 frontier. Both producers returned `PREFILL_OK`, finish `stop`,
and the serial route.

C and Rust wrote the same
`af7f4f7b6cc1a33c0ca93d55f7dc11bb08c0c507.kv`:

```text
size:          41,956,589 bytes
reason/ext:    2 / 0
model/ABI:     0 / 2
tokens/ctx:    4,096 / 8,192
text/payload:  14,617 / 41,941,920 bytes
text SHA-256:  2dfc39c82411c86cadeabdfc517d0450054a0cba7c32bf4afb2b5f197dea9384
payload SHA:   00d8719acee5341f3e92feb45347a43fa41d11f0102148c1775feffcdd318f1c
```

Each loader was a fresh process with an isolated copy of that one record and
`continued=0`. The loader request SHA-256 is
`6f29e35888928f2ac2228046695f3859c2c0d816301e6574c56bcb5e3beb3443`:

```text
C save    → Rust load   PASS  cached=4096 computed=2686 output=5 TTFT=3642.8 ms
Rust save → C load      PASS  cached=4096 computed=2686 output=5 TTFT=3656.3 ms
Rust save → Rust load   PASS  cached=4096 computed=2686 output=5 TTFT=3644.0 ms
C save    → C load      PASS  cached=4096 computed=2686 output=5 TTFT=3729.4 ms
```

All cells returned `RESTORED_OK`, finish `stop`, prompt tokens 6,782, matching
prefill usage/timing counts, and the serial route; normalized semantic
SHA-256 is
`7f0f6d75bd5f83e33fbd25eb224ced322825960f7e94a9467eca8ce71a125dcd`.
The 86,720,111,488-byte DeepSeek V4 Flash IQ2XXS/w2Q2K model was shared.
Its SHA-256 is
`ca22ae2f838e14077c22bc1c1417b71b45b5e5a3687bd96c2ac6e17fdb6261c0`.
C binary SHA-256 was
`04f25a86040940674984c35160d2e4eec7f6c6d4e30313815ce10309cd57e662`;
Rust binary SHA-256 was
`3fd0e40320d06eae6891c5399fe622df16b9dfdfa081d92dd1df1519ff003736`.
Every resident exited before `clear_cache`; final inspection found no GPU
compute application or listener and 117 GiB available. This is a correctness
and cross-read gate, not an ABBA performance claim, and it does not itself
cover Motif/Solar callback semantics or the separately green width-1 bank lane.

#### Width-1 continuous-bank shutdown checkpoint

The scoped Motif-3 OpenAI Chat, no-think/no-tools bank lane is live-green under
`scratch/rust-host-live/bank-fourway-20260825-093322/`. `e9dfd77` exposes
opaque native bank snapshots and bounded payload save/load, `0e4a178` adds the
strict-suffix bank candidate, and `15b016c` owns width-1 host metadata/policy,
warm/disk admit, retirement, checkpoint/evict/shutdown persistence, and
graceful shutdown ordering. CPU/oracle/link gates at that commit were
`make test-kv-parity` (38 unit + 11 cross-language),
`make test-server-parity` (88 unit + all integration matrices), and
`make ds4-server-rs`. The native rolling scheduler and opaque bank payload
remain C.

The live fixture used the 94,162,541,472-byte
`Motif-3-MQ87-88-FIT.gguf`, ctx 8,192, max output 128, temperature 0,
thinking disabled, no tools, disk budget 2,048 MiB, minimum checkpoint 512,
`cold=0`, `continued=0`, and `DS4_SERVER_BANK_CHECKPOINT=0`. The C oracle
allocated two banks while Rust used `--cont-width 1`; every cell admitted one
request into bank 0, so this is not multi-bank evidence. Producer and loader
request SHA-256 values were `a765063d...` and `14042672...`.

C and Rust graceful-shutdown producers wrote the same
`3faf064c1bf3e92ef70f356f5c1c7baeb0dd62bc.kv`:

```text
size:          301,972,407 bytes
reason/ext:    8 / 16 (bank-shutdown / EXT_BANK_REPLAY_V1)
quant/model:   2 / 3
tokens/ctx:    6,896 / 8,192
text/payload:  24,591 / 301,947,764 bytes
trailer:       0 bytes
text SHA-256:  bb9a55e900337c73272f430fc3b66efa01966e6ba70fe31ec01a4e60aea9856c
payload SHA:   df417b44399e62795a644431dafa537c5780a5753ab4d193d73f4c43208044c2
```

Each loader was a fresh process with an isolated copy of the selected record:

```text
C save    → Rust load   PASS  cached=6896 computed=15 output=4
Rust save → C load      PASS  cached=6896 computed=15 output=4
Rust save → Rust load   PASS  cached=6896 computed=15 output=4
C save    → C load      PASS  cached=6896 computed=15 output=4
```

All four returned `RESTORED_OK`, finish `stop`, prompt tokens 6,911, and the
continuous OpenAI Chat route. The C binary SHA-256 was `04f25a86...`; the Rust
four-way/ABBA binary was `287c821a...` at `15b016c`.

The same C-produced bank record then drove a 114-token C→Rust→Rust→C restore
ABBA. The request SHA-256 was `d68b6990...`, and all completion text was exact
(`faf7c021...`):

| Cell | Prefill tok/s | Decode tok/s | TTFT ms | Device census GiB |
|---|---:|---:|---:|---:|
| C #1 | 76.0 | 14.9 | 375.0 | 99.15 |
| Rust #1 | 76.4 | 14.8 | 388.4 | 98.82 |
| Rust #2 | 76.1 | 14.7 | 387.4 | 98.82 |
| C #2 | 76.1 | 14.9 | 398.7 | 99.15 |

C/Rust means were 76.05/76.25 tok/s prefill, 14.90/14.75 tok/s decode
(-1.01%), and 386.85/387.90 ms TTFT (+0.27%), inside the provisional gate.
Peak host RSS was not captured, so this is not full memory-performance
closure.

Commit `98d81b9` then passed native continuous decode duration/token/step
through the opaque callback; CUDA execution and bank behavior did not change.
Its bridge/core/server/KV/link gates are green. A post-fix C→Rust short loader
cell used Rust binary `6c25d952...` and returned `RESTORED_OK` with
cached/computed/output 6,896/15/4, 12.3 decode tok/s, and 1.25 tok/step. That
matches the C loader cells' 12.0--12.1 tok/s and 1.25 tok/step and closes the
short-response timing-porcelain discrepancy. The 114-token ABBA was not
repeated after this timing-only commit; its `15b016c` binary provenance remains
explicit above.

At that historical snapshot this was not the whole Phase 4 gate. Live default configured
10,000-token/effective aligned 10,240-token reason-bank-checkpoint, live
reason-bank-evict, full default-policy ABBA, multi-bank fork/partial and
pin/claim behavior, bank tool/thinking/extension integration, other families
and surfaces, and peak host-RSS/soak evidence remain pending.
Intermediate-prefill and tool-map replay retain their separately scoped
limitations above.

### Phase 5 — web utility

Resource ownership (`OwnedFd`, `TcpStream`, `Child`, `Vec<u8>`)
without changing the subprocess/search contract. No Tokio.

### Phase 6 — distributed

Explicit integer codecs. Wire bytes match C. Do not serialize
`#[repr(C)]` Rust structs.

Runtime is production-integrated through blocking Rust coordinator/worker
orchestration: CLI/route, HELLO/WORK/RESULT, receive prefetch, hop relay,
reconnect, and snapshot gather/scatter ordering are host-owned. Layer-slice
evaluation and opaque GPU snapshot bytes stay native.

### Phase 7 — server shadow

`route_decide` + `request_compute_needs` are the routing oracle.
The HTTP door (reader, native error envelopes, schema-format refusal,
`/v1/models` porcelain), the four JSON parsers (chat / completion /
Anthropic / Responses, tokenize/render cut off), the four tape
stream projectors plus buffered finals and incremental live DSML tool stream (OpenAI + Anthropic; Responses has no incremental tool stream in C),
enqueue/shed + job-id mint, the host-owned `/metrics` `/v1/stats`
porcelain including the memgov census format (live CUDA census +
observation via `ds4_bridge_mem_*_snap` when an engine is open;
absence-not-zero when unsupported), family
render including tool-schema / invoke reconstruct
(DSML / Motif / EXAONE / dots3 / Solar C oracle),
NULL-handle tokenize/sample FFI, the serial decode driver
(scripted tape, no GGUF), generated-message tool parse
(DSML / Hermes / dots3 / Solar C goldens), SemAccum stop /
no-tools cut / DSML `tool_calls` verdict, and finalize
`tool_calls` / `tool_use` wire, the Inc 5a/5b/5c
continuation registry (publish / resolve / hold / pin / TTL /
bank claim + serial 503/409), and corrective recovery retry
(`decode_again` / model-visible tool error + tag-completion
repair) are `make test-server-parity`.
The loaded-engine shadow now accepts and parses in client threads,
then routes jobs through a bounded FIFO to one non-`Send` inference
owner. Stable continuation pins, queue-inclusive TTFT, ordered
terminal publication, nonblocking bounded sends, real TCP disconnect,
slow-reader, and exactly-once settlement regressions are green at
`6545d44` (`make test-server-parity`).
Live Motif generate: content/finish/usage-count/`cache_write_tokens`
match. Live CUDA census: `census_supported=1` epoch=1636
weight_artifact 86.07 GiB (`scratch/rust-host-live/`).
Serial buffered responses emit a C-shaped `timings` object.
Live serial/continuous/static routing, width-2 barriers, multi-bank
fork/partial/evict/shutdown, four surfaces, tool/continuation/KV extensions,
disconnect/slow-reader handling, and the smoke/long/OPP-C proof harness are
green in the current 60-cell matrix. Qwen additionally covers image requests,
MTP, exact/configured 262K, and the two-hour soak.
Do not improve the table.

All four surfaces:

```text
POST /v1/chat/completions
POST /v1/completions
POST /v1/messages
POST /v1/responses
```

plus `/v1/models`, `/v1/stats`, `/metrics`.

Preserve ID formats (including the documented Anthropic
`chatcmpl-N` quirk), `finish_reason` mapping, native error
envelopes, stream event order, continuation / trust-domain
semantics, chunked bodies, typed schema-format refusal (v0.6.3).

Replay the `ds4_test --server` fixture inventory listed in the API
matrix. Live gates follow that document, not byte dumps.

### Phase 8 — `ds4.c` leaves

| Slice | Gate |
|---|---|
| Model metadata | enum/shape values match `g_ds4_shape` / catalog — **green** (`make test-catalog-parity`) |
| GGUF catalog | mmap; no full-file `Vec<u8>`; first-shard `split.count` identity — **green** on synthetic v3; host `weights_bind` name catalog + bind-plan check/match + host tensor-dir apply (skip `parse_tensors` when installed) + host `config_validate` / apply shape+compress (skip C validate when installed) + host bind map (skip C name walk when installed) + host `weights_validate_layout` (skip C main-model layout when bind map installed) + host MTP/DSpark sibling name/layout catalogs + sibling BindPlan resolve/validate + sibling bind map FFI (native swaps the sibling map around its own open+bind window and skips that sibling's C layout check; live DSpark drafter 81/81 + Rust `--dspark` decode green) + host-table clear before sibling `model_open` green; sibling pointer assignment / CUDA upload still native |
| Tokenizer | encode/decode/special/template/stop exact token ids — **green** on synthetic GGUF (`make test-tokenizer-parity`); family chat transcript framing is exact for DeepSeek, Motif-3, Solar Open2, EXAONE, and Dots3 across none/low/high/max think modes, with byte payload and malformed-control errors covered; host vocab apply (skip C `vocab_load` when installed) + `Model` host tokenize green; live Motif specials + encode `hi`→8320 |
| Session | ownership + sync/eval/rewind/payload — **ledger + DSV4 prefix green** (`make test-session-parity`: rewrite/prefix/plan/rewind/generation + header/token tapes); GPU/logits tail / CUDA eval still native |
| Backend dispatch | CUDA path unchanged; no extra dynamic dispatch on the hot path |

### Proof harness (any phase that can touch CUDA execution)

```text
make proof-cuda-smoke
make proof-cuda-long
make proof-cuda-opp-c
make proof-rust-cuda-opp-c
```

`proof-cuda-opp-c` remains the native-drift gate against a tracked
golden. A persisted `CUDA_ARCH=sm_121` build selects the GB10 snapshot
validated against the `v0.6.5-dfm`/Qwen C cut; the older v0.6.3 snapshot is
retained under its own namespace. Other builds keep their original generic
snapshot and require separate architecture approval. `proof-rust-cuda-opp-c`
is the separate host-parity gate: it writes a temporary snapshot with the
current C oracle, then checks the Rust binary through the same stable runner
path. Its artifacts remain under `/tmp/ds4_proof/proof-rust-cuda-opp-c.*`
for audit; it does not rewrite or replace either tracked native golden.

Both OPP-C gates must be green before the Rust execution path becomes
default. Long-context captured-vs-eager stays a release gate.

### Phase 9 / split

See work instruction §26. `SPLIT_READINESS.md` is now green because Runtime,
Correctness, Performance, and Architecture (unsafe confined to reviewed
FFI/native/OS adapters) passed the current audit.

```bash
rg -n 'unsafe \{' crates/
```

Almost every hit must sit in `crates/ds4-sys` or a native adapter.

## Family regression set (must stay green)

Makefile variables, not flags:

| Family | Targets |
|---|---|
| Motif-3 | `test-motif3-{loader,reference,tokenizer,cuda,resident,batch}` |
| Solar | `test-solar-{loader,tokenizer,forward,session,kda,kda-prefill,kda-chunk,gates,kv}` |
| K-EXAONE | `test-exaone-{ref,kernels,batch}` |
| dots3-note | `test-dots3-{loader,tokenizer,resident}` |
| Qwen3.8 Flash Next | `test-qwen4exp-{loader,tokenizer,ple,ple-cuda,primitives,hc-forward,ple-compute,qsa,qsa-forward,moe,moe-forward,gdn,gdn-forward,batch}` on Q5+Sidecar |
| Shared | `test-model-family-kernels`, `test-mmq-parity` |

Resident tests load real weights. Use tmux + workspace-local
`../scripts/guarded-run.sh` (outside this repository).
One resident model at a time.

## Performance protocol

Same host, same artifact, clocks recorded.

Minimum order:

```text
C → Rust → Rust → C
```

or ABBA. Publish SHA + GGUF identity + command/env with the numbers.
Unexplained 2–3% is a fail even if it sits inside the provisional
percent table.

### Recorded ABBA (2026-08-31, Qwen Q5+Sidecar)

Candidate `6ca85c8`, C Qwen cut `4d40d97`, Q5 three-shard main GGUF plus the
shared four-file SSD-PLE Sidecar, ctx 196,608, two banks, prefill chunk 8,192,
embedded MTP draft 2, temperature 0. Every cell used a fresh owner/worker pair;
C and Rust were never resident together, and `clear_cache` ran only after both
PIDs exited.

All text cells returned the same 64-token output (SHA-256 `3ab21f93...`):

| Cell | Prefill tok/s | Decode tok/s | TTFT ms | Worker VmHWM KiB | Worker GPU MiB |
|---|---:|---:|---:|---:|---:|
| C #1 | 519.7 | 14.4 | 7,954.7 | 1,852,404 | 23,471 |
| Rust #1 | 518.6 | 14.5 | 8,044.9 | 1,933,444 | 23,471 |
| Rust #2 | 518.8 | 14.4 | 8,041.3 | 1,929,004 | 23,471 |
| C #2 | 518.0 | 14.5 | 7,979.9 | 1,847,240 | 23,471 |

Rust/C means were 99.97% prefill, 100.00% decode, +0.95% TTFT, +4.40%
worker VmHWM, and identical GPU residency. All four image cells returned
`MEN WALK ON MOON` (SHA-256 `991530ac...`); Rust/C means were 100.20% image
prefill and +0.32% image TTFT. The six-token image decode sample is not used
as a performance gate.

### Recorded ABBA (2026-08-24, cross-lane)

Host SHA `9463b3c` (+ scratch runner `scratch/rust-host-live/abba.sh`),
GGUF `Motif-3-MQ87-88-FIT.gguf`, ctx 8192, cold 6782-token prompt,
114 decode tokens, temp 0, thinking disabled, sequential loads with
teardown + `clear_cache` between cells:

| Cell | Lane | prefill tok/s | decode tok/s | ttft ms | peak GPU MiB |
|---|---|---|---|---|---|
| C #1 | continuous | 633.4 | 14.8 | 10755 | 112145 |
| Rust #1 | serial | 579.8 | 15.4 | 11834 | 102357 |
| Rust #2 | serial | 577.5 | 15.5 | 11877 | 102357 |
| C #2 | continuous | 633.0 | 14.8 | 10780 | 114529 |

All four completions are byte-identical greedy text (SHA1
`4d34f24352e7`) — the Token axis passes across hosts and lanes.
Pair spreads are ≤0.4%. The −8.6% prefill / +4.4% decode delta was
attributed to the lane difference and re-measured after the
continuous port (below).

### Recorded ABBA (2026-08-24, same lane: both continuous)

Same protocol, `ds4-server-rs --cont-width 2` routing
`openai_chat_continuous`, all four completions byte-identical
(same SHA1 `4d34f24352e7`):

| Cell | Lane | prefill tok/s | decode tok/s | ttft ms |
|---|---|---|---|---|
| C #1 | continuous | 579.2 | 14.9 | 11767 |
| Rust #1 | continuous | 590.8 | 14.8 | 11521 |
| Rust #2 | continuous | 592.9 | 14.7 | 11480 |
| C #2 | continuous | 633.9 | 14.9 | 10751 |

Rust pair spread 0.4%; the C pair spread is 9.0% (579/634 — the
cross-lane run's C pair sat at 633.4/633.0), so the −2.4% Rust mean
prefill delta is inside the C cell noise; decode is within 1%. No
unexplained regression. At that recorded snapshot Rust continuous still had one active job:
client ingress is concurrent behind the FIFO, but the sole owner runs
each job to completion. The rolling scheduler, bank admit, per-seq eos,
and engine usage split are the same native path; live multi-client width
and the Anthropic/Responses cont promotion are follow-ups.

### Recorded ABBA (2026-08-24, local benchmark shadow)

Host commit `a9ba7b38cbf683f03c21e2604e055356ec2a2ad2`
(`v0.6.3-dfm`), NVIDIA GB10, driver 610.43.02, CUDA 13.3. C binary
SHA-256 `3c0af3d1d860cf3d2800261d51e729c3ca96aae3d048551ed721ed717aed0f8c`;
Rust binary SHA-256
`a66d97c48755af0e54178c57a5d70765019eade45cb9a038ccfd5615c940dae3`.
Model artifact:
`DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-0731.gguf`
(86,720,111,488 bytes; model label `DeepSeek V4 Flash`, revision label
`0731-chat-v2`, quant label `IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8`).

Order was C → Rust → Rust → C with teardown and `clear_cache` between
cells. Both binaries used the same raw prompt, frontiers 1024→2048,
`ctx_alloc=2081`, 32 decode tokens, width 1, and default packed FP8 KV
+ FP4 index; no `DS4_*` override was set. Logs and CSVs are under
`scratch/rust-host-live/bench-abba-20260824/`.

```bash
$BIN --cuda -m "$MODEL" --prompt-file tests/long_context_essay_prompt.txt \
  --ctx-start 1024 --ctx-max 2048 --ctx-alloc 2081 --step-incr 1024 \
  --gen-tokens 32 --csv "$CELL.csv"
```

| Cell | Frontier | Prefill tok/s | Decode tok/s | Steady decode tok/s | First decode token s | Host max RSS KiB | KV bytes |
|---|---:|---:|---:|---:|---:|---:|---:|
| C #1 | 1024 | 838.58 | 22.78 | 22.94 | 0.0531 | 8,972,888 | 28,482,336 |
| C #1 | 2048 | 945.37 | 23.10 | 23.10 | 0.0433 | 8,972,888 | 32,968,864 |
| Rust #1 | 1024 | 850.48 | 22.83 | 22.96 | 0.0517 | 9,024,668 | 28,482,336 |
| Rust #1 | 2048 | 953.26 | 23.10 | 23.10 | 0.0433 | 9,024,668 | 32,968,864 |
| Rust #2 | 1024 | 845.62 | 22.88 | 23.01 | 0.0518 | 9,025,144 | 28,482,336 |
| Rust #2 | 2048 | 954.29 | 22.93 | 22.92 | 0.0433 | 9,025,144 | 32,968,864 |
| C #2 | 1024 | 843.66 | 22.91 | 23.06 | 0.0523 | 8,972,584 | 28,482,336 |
| C #2 | 2048 | 947.30 | 23.10 | 23.11 | 0.0439 | 8,972,584 | 32,968,864 |

Rust/C mean ratios were 100.82% / 100.79% prefill and 100.04% /
99.63% decode at the two frontiers. Rust host max RSS was +0.58%; KV
bytes matched exactly. This passes the Phase 3 FFI-overhead thresholds.
GPU resident peak was not sampled, so this is not the full pre-split
performance gate.

## Status of this matrix

The superseded subsystem status is available as an
[archived snapshot](https://github.com/Baekpica/ds4-dfm-rs/blob/ac750d61ef0306d30cc595081883f61ec847c3d8/docs/rust-migration/STATUS.md).
Current release gates belong in the [v0.1.0 ledger](../releases/v0.1.0.md).
