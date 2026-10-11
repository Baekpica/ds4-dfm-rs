# DeepSeek V4.1 Flash on ds4-dfm-rs — port plan

[Repository README](../README.md) | [Model families](ds4-dfm-model-families.md) |
[Performance](performance.md) | [Prefill/decode playbook](prefill-decode-optimization-playbook.md)

Status: proposal, not started. Branch: `ds41` (off `dev` 7b6448b).

Resume protocol: this document is the state carrier. There is no handoff file and
no rotation to a fresh session; the session compacts (/compact) and continues, so
every step below must be executable from this text alone.

Authored 2026-10-08. The port has one authority and it is now a real source tree,
not a decompilation: the working C/CUDA engine at `/data/YoungAi` (clone of
JLFN/YoungAi, branch `restructure`, commit 3946dbc). Citations marked (a) were
collected by read-only inventory agents in this session and carry file:line for
re-checking; citations marked (v) were verified first-hand.

## 1. Goal

**ds4-dfm-rs serves the published DeepSeek V4.1 Flash vq8 artifact natively, with
the serving behaviour this tree already gives every other family: prefill, KV
reuse, its own sampling defaults, its own DSML tool path, and the four API
contracts.**

The port is a port of capability, not a rescue: the C engine already runs this
model, measured (artifact `README.md` §1). What is missing is the whole V4.1
substrate in the Rust host.

| Id | Success criterion | Proof |
| --- | --- | --- |
| G1 | The artifact loads: the GGUF's 40-44 tensor types and the VQ blobs are parsed and the tensor inventory matches the engine's accepted inventory | the inventory diff, both engines |
| G2 | Numerically faithful: tokens and logits match the C engine's golden set on fixed prompts at temperature 0 | the golden-set comparison, per position |
| G3 | Served through the existing API surface: chat, completions, responses, Anthropic | one served request per contract |
| G4 | Tool calls work end to end, no translating shim | a tool call issued, executed, fed back |
| G5 | Speculative decode runs: MTP towers load and the verify/rollback keeps the committed cache at the accepted-prefix boundary | the losslessness gate, cache-frontier |
| G6 | Sidecars work: a domain sidecar changes the output as the engine's does, and a mismatched posttrain pair is refused | the sidecar A/B and the `base.fnv` refusal |
| G7 | Performance ledger, measured not assumed | the ledger, same prompts as §2 |

Non-goals for v1: no Metal (V4.1 is CUDA-only in the reference, README §7),
no distributed inference, no quantizer or solver work (the sidecar and
post-training solvers are the C toolchain's job), no re-quantization.

## 2. The artifact, as measured

Read over the gvfs mount of the Spark
(`/run/user/1000/gvfs/sftp:host=192.168.1.91,user=leandro/home/leandro/youngai/model`) (v).

| item | value |
| --- | --- |
| base | `DeepSeek-V4.1-Flash-vq8sh14-q4k-mtpnative.gguf`, 113,556,639,424 bytes, 40 parts |
| base sha256 | `9469cfa9ff47c5b9f1e2bbf31b4226ef11b4de9ddb92a0c7641c4a9b623e4654` (`SHA256SUMS`; every part also hashed) |
| sidecars | five domain dirs `…-grrb-{vqfin41_vqhalf_a_n8192,code_fit_n15360,law_fit_n15360,med_fit_n15360,sci_fit_n15360}-engine`, each `gr_Lnn.bin` (1,044,492 B) + `rb_Lnn.bin` + `manifest.txt` |
| post-train | `posttrain-experimental-20260924/`: `gr_L39.bin` + `base.fnv` (20 B) |
| engram | not in the artifact: two official shards, read in place via `--engram-dir`. Present on the Spark at `/home/leandro/youngai/deepseek-engram`: `model-00047-of-00048.safetensors` 101,535,150,936 B and `model-00048-of-00048.safetensors` 101,537,926,640 B (190 GB, measured over sftp 2026-10-08) |
| bins | `bin/ds4`, `bin/ds4-server`, built on the Spark with `make cuda-spark` |

Engine invocation the port must be able to reproduce (artifact README §3, §7):

    ./bin/ds4-server --cuda -m <base>.gguf --zchain <sidecar_dir> \
        --engram-dir <shards_dir> --mem-budget-mb 110000 [--posttrain <dir>] [--no-dspark]

Endpoints `/v1/chat/completions`, `/v1/completions`, `/v1/responses`,
`/v1/messages`; request-omitted sampling follows the model card (temperature 1.0,
top_p 1.0, no min_p); thinking is off unless requested.

## 3. Method: derive from the engine, cite it, gate on the golden set

The reference implementation is `/data/YoungAi` at 3946dbc. Nothing is
re-derived from a spec or from memory: every format, index width, slot address
and kernel geometry is read from that tree and cited in the code, the evidence
file and the commit message (the tree's own rule 0.2 doctrine).

The acceptance instrument is the C engine's captured behaviour on the Spark:
for a fixed prompt set at temperature 0, the token stream and the logits. The
port consumes that capture and nothing else as its correctness oracle. Until it
exists, G2 has no gate and "it works" is an impression.

Fidelity to the Rust host's standards (its `AGENTS.md`): common serving surface
across families, requested/effective/qualified limits reported, correctness
proof AND speed proof per optimization commit, and the inference boundary
preserved (host -> ds4-core -> ds4-sys -> native).

## 4. What the Rust host already has (v, (a))

- Family identity and shapes: `ModelFamily::DeepSeek4 = 0`, `SHAPE_FLASH` 43
  layers, `SHAPE_PRO` 61 (`crates/ds4-core/src/shape.rs:20-33`, `:435-492`,
  `:494-551`).
- The DeepSeek validator and its required metadata: `validate_deepseek`
  (`crates/ds4-core/src/validate.rs:317-456`), including
  `hyper_connection.count`, `hyper_connection.sinkhorn_iterations`,
  `hyper_connection.epsilon`, `compress_ratios`, `swiglu_clamp_exp`,
  `attention.indexer.*`.
- Weight binding with the unified expert naming that DeepSeek uses
  (`bind.rs:656-700`, `layout.rs:2383`), and the DSpark sibling bind slots in
  the native boundary (`native/bridge/ds4_host_load.h:323-331`).
- DSpark device kernels: `dspark_capture_mean`, `dspark_gather_concat`,
  `dspark_markov_step` (`ds4_cuda.cu:16308`, `:16322`, `:19621`; launchers
  `:21815-21833`), mirrored in `metal/dsv4_misc.metal:1330-1391`. These are
  the V4-era drafter's kernels; the V4.1 artifact needs the three-tower MTP
  drafter instead (inventory §8), so this is a re-use question, not a port.
- Serving caps for the family, including `MtpKind::DeepSeek` and
  `SpecLane::Bank` (`crates/ds4-core/src/serving.rs:886-904`), and the
  DeepSeek DSML tool path (`crates/ds4-server/src/tools.rs:20-31`, `:260+`).

So the base V4 architecture is not the work. The work is the V4.1 container
format and its subsystems.

The function-level inventory of the C engine's V4.1 surface (host and device,
with citations) lives in [the inventory](deepseek41-port-inventory.md) and is
the work list this plan is built from. It is being completed subsystem by
subsystem (VQ, engram and sidecars, MTP draft, core and serving).

## 5. Gap inventory

"Absent" = zero grep hits in the Rust tree; the C reference is cited per row.

| id | gap | C reference (source of truth) |
| --- | --- | --- |
| G1 | GGUF tensor types 40-44 (`go1b` 256/34, `go2b` 256/68, `vqblob` 1/1, `fp4x32` 32/17, `fp8_32x32` 1024/1025) unknown: the host table is 31 entries covering ids 0-30 plus `pq2_0`=142 (`crates/ds4-core/src/tensors.rs:55-96`) (v) | `src/core/core_gguf.c:89-104`; engine enum `ds4_internal.h:177-181` (a) |
| G2 | VQ expert decode absent. `DQVL` blob header, v2 `DQVQ` per-matrix payload (own codebook) and v3 `DQV3` (per-layer codebook + 12-bit main stream + 1-bit plane, E4M3 codewords, per-row f16 gain) | `vq_fmt.h:1-20`, decode `vq_fmt.h:57`/`:91` (v); device geometry `src/cuda/cuda_vq_row.inc.cu:19-47`; dispatch by (version, bit width) `cuda_vq_decode_launch.inc.cu:21-30` (a) |
| G3 | MTP draft towers absent: 3 towers reusing the layer struct, per-expert `fp4x32` or a single `ffn_exps_vq.blob`, five shared heads | bind `src/core/core_bind_v41.c:101-145`; tower types `core_draft_tower_types.h`; forward `core_v41_draft.c:1-11` (a) |
| G3b | DONE 2026-10-09 (`0d3378d`, `356a100`): the artifact carries one `blk.L.ffn_exps_vq.blob` per layer and no `ffn_gate_exps`/`ffn_up_exps`/`ffn_down_exps`; the V4.1 bind and layout require the blob, and the plan binds the artifact's 1000 tensors with `layout: ok` | `core_bind_v41.c:41` (a), inventory §1 |
| G4 | Speculative verify/rollback for V4.1 absent. Note the engine has TWO drafters: the V4-era DSpark (`dspark.*`, `cuda_dspark.inc.cu`) which this host already mirrors, and the V4.1 three-tower MTP (`mtp.*`, `core_v41_draft.c`) which is the one this artifact needs | `v41_spec_snapshot` `src/core/core_v41_forward.c:241`, `v41_spec_rollback` `:267`; round drive `core_v41_api.c:322`, `:365`; tower contract and the two rollback cautions: inventory §8 (a) |
| G5 | Engram absent: two tables, 264 B rows (256 B fp8 e4m3 + 8 B ue8m0 scale), rolling 4-gram hash, O_DIRECT per-row reads | `src/core/core_v41_engram.c:6`, `:21`, `:30`, `:63`; metadata `core_validate_v41.c:118-154`; flag `--engram-dir` `src/cli/cli_opts.c:269`, `--v41-no-engram` `:371` (a) |
| G6 | Domain sidecars absent: `gr_Lnn.bin` per-row gains, `rb_Lnn.bin` router bias, `amp_Lnn.bin`, the `base.fnv` pair fingerprint, `--zchain`, `--posttrain` | `src/core/core_v41_amp.c:41`, `:88`, `:125-147`, `:160-186`, `:227`; fingerprint `src/common/ds4_gr_fnv.h`; flags `cli_opts.c:258-268` (a) |
| G7 | V4.1 metadata is not identified: `deepseek4.attention.kv_source_layers` and `index_source_layers` are read nowhere, and there is no V4.1 variant/shape branch | V4.1 metadata load `core_validate_v41.c:46-154`; shape `core_shape_select.c:81-116`, `:211` (a) |
| G8 | V4.1 serving surface absent: no `--zchain`/`--posttrain`/`--engram-dir`/`--no-dspark` on the Rust server, and request-omitted sampling has no family default (the host is request-driven) | server flags `src/server/server_config.c:314-329`; defaults `server_msgs.c:162-164` from `ds4.h:54-56` (a) |
| G9 | `manifest.txt` is documentation only in the C engine too (no reader); the Rust port must not invent one | grep for "manifest" in `src/`: no reader (a) |

## 6. Build order and gates

Each phase is gated on a comparison, never on inspection. The Spark is the only
machine the reference was tested on (sm_121, CUDA 13, 128 GB unified), so P4
onward run there; all Spark work goes through the owner's ssh session and asks
before any state-changing command.

| phase | work | gate |
| --- | --- | --- |
| P0 | Freeze inputs: verify the assembled artifact hash (the 40 part files are deleted after assembly, so `SHA256SUMS`' part lines cannot resolve); capture the C engine's golden set with `tests/capture_ds41_golden.sh`, with the engram tables as the primary instrument and `NO_ENGRAM=1` as the fixture variant; record the artifact's accepted tensor inventory | hashes re-verify; the golden set exists as files with its own MANIFEST |
| P0 | DONE 2026-10-08 | see 6.5 |
| P1 | DONE 2026-10-08 (`c1fdc04`, `7c21279`): tensor types 40-44 in `tensors.rs`, the VQ decode oracle, the type table | v2 decode bit-exact against `ds4vq_dequant_f32` (fixtures in `tests/fixtures/vq`); v3 at unit level (12-bit, 13-bit plane); types match `core_gguf.c:89-104` |
| P2 | DONE 2026-10-09: the engram half (`79756c0` + `0af1d10`, §6.6.2) and the sidecar half (`dac818e`, §6.6.3: gr/rb/amp readers, the fp4x32 decoder, the `base.fnv` gate, both directions verified on the real directories) | the tensor/key inventory matches the engine's; a mismatched posttrain pair is refused |
| P3 | DONE 2026-10-09 (`a73fe31`, `bd393ad`): the `Variant::V41` shape (`d53079b`), the metadata wire, the bind arm (`0d3378d`), the layout table (`356a100`) and the ②/③ zchain merge (§6.6.4); the engram session state moves with P4's read path | the loader accepts the artifact: `identify` + `validate: ok` + `layout: ok` + `bind: slots=1000 bound=1000 required-missing=0` on the real file (§6.6.1), and the real sidecar pair merges bit-exact (§6.6.4) |
| P4 | Native V4.1: the variant skeleton (P4-0), the VQ decode family (P4-1; the artifact's real blob is DQVL v3 13-bit, measured), the fp8_32x32 path and the engram read path; units and gates in §6.7 | G2 on device: logits match the golden set |
| P5 | MTP towers and DSpark verify/rollback | byte-identical greedy output with drafting on and off at N=1; cache-frontier gate at N>1. The engine's own cautions: the main-hidden ring trim in rollback, and the compressor pending-row snapshot taken before the shift |
| P6 | Serving: the flags, the sampling profile for this family, the four API contracts, the DSML tool path | a served tool call, end to end, no shim |
| P7 | Performance ledger: prefill and decode on the §2 prompts, cache reuse demonstrated | the ledger, published with its method |
| Z | DONE 2026-10-10 (`f377e87`..`91dba0f`, §6.18): the install.sh startup contract: `--zchain` (the ② mount over the P4-7 stores + the P2/P3 Rust reader), `--posttrain` (③), `--mem-budget-mb` (the engine's L1 85 % gate; the Linux footprint watchdog is inert by the engine's own design, core_util.c:141-145), and the install.sh smoke on the v41 route | both sides booted exactly as install.sh boots; served ids byte-identical with the finance sidecar mounted; the smoke passes (all PASS 2026-10-10, §6.18/§6.19) |
| P7b | The qwen38-derived serving candidates (2026-10-10, §6.21; reference `docs/ds41-qwen38-serve-lessons.md`): P2 the engram overlap (engine mirror), Q-1 the MTP draft shortlist, Q-2 the MTP prefix cache | each: the four parity gates green + a >= 2 % measured median gain on its phase; kill switch per item (P2 mirrors the engine, none needed) |
| PT | Post-train (2026-10-11, §6.24): PT-1 the `ds4_posttrain` module + the JSONL upload surface (now, host-only), PT-2 the ③ amp apply (now), PT-3 the trainer + the backward family (after the port closes), PT-4 `posttrain_reward` (optional, legacy) | PT-1/PT-2: model-free + the `ds4-train` upload oracle on the Spark; PT-3: the engine's own checks + the wt2 gate |

P0 and P1 can start without further owner decisions.

## 6.1 Where the work hooks into the Rust host

From the call-graph index (`callgraph-mcp`, project ds4-dfm-rs, queried
2026-10-08):

- `validate_file` is the per-family dispatch: it calls `validate_deepseek` and
  its ten siblings (`crates/ds4-core/src/validate.rs`). A V4.1 arm extends this
  dispatch.
- `validate_deepseek` has 15 call sites at depth 2 from `open_impl`,
  `validate_gguf`, `probe_model_artifact`, `check_metadata`, `dump_validate`
  and `host_catalog_uses_exact_contract`; any signature change to it is felt by
  all of them.
- `bind_names` has 14 call sites from `resolve`, `matches_published_tensors`,
  `bind_plan_requires_every_published_tensor`, `layout_covers_bind_catalog` and
  the Flash-inventory tests; the V4.1 bind path must keep those catalogs in
  agreement (layout vs bind vs published tensors).

## 6.3 What stays native, what moves to Rust

The dividing rule is the repo's own boundary rule (`AGENTS.md`: keep new
host/control-plane code in Rust; existing CUDA/MMQ stays native; the inference
boundary host -> ds4-core -> ds4-sys -> native is preserved). Applied to this
port, per subsystem:

Moves to Rust (host side: parsing, validation, policy, orchestration, I/O):

- The GGUF type table entries 40-44 and their byte geometry
  (`src/core/core_gguf.c:89-104`): done in `crates/ds4-core/src/tensors.rs`.
- The VQ container and payload parse, both blob versions
  (`vq_fmt.h`, `src/cuda/cuda_vq_row.inc.cu:19-49` geometry): done in
  `crates/ds4-core/src/vq.rs`. This is the oracle the loader validates with; it
  is not the execution path.
- Metadata validation and identification (`core_validate_v41.c:46-154`,
  `core_shape_select.c:81-116`) into `validate.rs`, `identify.rs`, `shape.rs`.
- Tensor naming and layout (`core_bind_v41.c:41-145`) into `bind.rs` and
  `layout.rs`.
- Sidecar formats and policy: `gr_Lnn.bin`, `rb_Lnn.bin`, `amp_Lnn.bin`, the
  `base.fnv` fingerprint check (`core_v41_amp.c:41-227`, `ds4_gr_fnv.h`) are
  file formats and admission policy, so they belong to `ds4-core`.
- Engram metadata parse, table open, row addressing and the prefetch policy
  (`core_validate_v41.c:118-154`, `core_v41_engram.c` open/pool parts): host
  I/O in Rust. The wkv matmul and its gate stay native.
- Serving: the family's sampling profile, the `--zchain`/`--posttrain`/
  `--engram-dir`/`--no-dspark` flags and their defaults
  (`server_config.c:314-329`, `server_msgs.c:162-164`), the V4.1 request
  branch, and the DSML/thinking handling (`server_generate_v41.c`).
- The speculative round loop's policy: when to draft, the acceptance rule, the
  commit boundary, and the snapshot/rollback orchestration
  (`core_v41_api.c:322,365`, `core_v41_forward.c:241,267`) - the host drives
  it, the device state itself stays native.

Stays native (never re-implemented in Rust):

- Every CUDA kernel: the VQ decode/GEMV/prefill/MMA family
  (`src/cuda/cuda_vq_*.inc.cu`), the V4.1 attention, indexer, hyper-connection,
  fp4/q4k and sampler kernels (`cuda_v41_*.inc.cu`), the draft and DSpark
  kernels (`cuda_v41_draft.inc.cu`, `cuda_dspark.inc.cu`). They stay in the
  single-TU native backend behind the existing ABI.
- The kernel-internal geometry: block loads, the swizzle, the shfl index
  distribution, register budgets, the 12-bit/13-bit block arithmetic - the
  measured micro-optimizations are device facts, not host policy.
- The device-side sampling kernel and the verify forward's device state.
- The C engine itself stays where it is as the behaviour oracle and the
  reference for every kernel; nothing in it is deleted by this port.

Decision test for anything not listed: if it runs per token on the device, it
stays native; if it runs once at load, per request, or is policy, validation or
file-format work, it moves to Rust.

## 6.4 Test strategy: the Spark, through the owner's tmux session

Owner waiver, recorded 2026-10-08: the rule 19 QA-tester gate does not apply
to this workstream; the agent tests its own work and reports the evidence.

All builds and tests for this port run on the DGX Spark (gx10-e1d2, sm_121,
CUDA 13, 128 GB unified), which is the only machine the reference engine was
tested on and the only one holding the artifact. The Rust host must therefore
be present on the Spark (clone or rsync of the `ds41` branch) — see O5.

Working rules for the Spark, non-negotiable:

- THE PORT RUNS ONLY ON THE SPARK (owner rule, 2026-10-10: "we will only run
  this in the nvidia-spark box that you have tmux ssh access to").  The
  deployment (install.sh) enforces it in its preflight (aarch64 + GB10 +
  CUDA 13 + 128 GB); model-free unit tests may still run on the dev box.
- Drive it only through the owner's ssh tmux pane, or an askpass session; there
  is no standing ssh key (owner rule, 2026-10-03).
- Read-only by default. Any state-changing command (build, write, stop a
  process) is asked for explicitly, per connection.
- Tests are launched from the pane and their output is captured from the pane;
  long runs are left visible rather than backgrounded.
- The memory budget is the guard: the engine refuses to start above
  `--mem-budget-mb 110000`, so a test run must not hold another large model.
- THE ORACLE IS THE SOURCE-BUILT ENGINE (owner rule, 2026-10-10 evening:
  "the test for this version ARE NOT allowed to use these
  [~/youngai/model/bin] cause this is not from this project").  Every gate
  that uses the engine as its oracle must use the engine BUILT FROM THE
  PINNED SOURCE, not the HuggingFace prebuilt binaries in
  ~/youngai/model/bin: clone github.com/JLFN/YoungAi (or /data/YoungAi),
  check out 3946dbcc360461fd6de5b64213bd619496094243, `make cuda-generic`
  (= make -B ds4 ds4-server ... CUDA_ARCH=native, the engine's own Linux
  recipe), and record the binary's sha256 with the evidence.  Runs against
  the prebuilt binary are preliminary readings only.

## 6.5 P0 evidence (2026-10-08, the Spark)

Frozen inputs, all measured:

| item | value |
| --- | --- |
| engine binary | `/home/leandro/youngai/model/bin/ds4`, sha256 `c03be9b4ac5de52b197b3740ba6ec60001b7454b7ab5d0f111cda3f3fee8a7f5` |
| artifact | sha256 `9469cfa9ff47c5b9f1e2bbf31b4226ef11b4de9ddb92a0c7641c4a9b623e4654`, recomputed by the capture script and equal to the `SHA256SUMS` assembled line |
| engram | `--engram-dir /home/leandro/youngai/deepseek-engram`, two shards, 190 GB |
| golden set | `/home/leandro/youngai/model/golden/`, MANIFEST with every hash |

Segment PPL of the five prompts (with engram): p1 112.0845 (S=8), p2 24.9582
(S=11), p3 7.2809 (S=7), p4 14.3524 (S=18), p5 876.7182 (S=7, Chinese). The
no-engram variant of p1 differs (104.3867), which is the expected effect of the
subsystem and the reason the primary capture runs with the tables.

Discovery worth the phase order: the score path writes a per-layer trace next
to the logits — `x_L<nn>.bin` (the layer input, S x 5120 f32), `y_L<nn>.bin`,
`hce_L01/L14.bin` and `erows_L01/L14.txt` for the engram layers. That is a
layer-by-layer gate instrument, not just an end-of-model one, and the port
should use it from P3 onward: a wrong layer shows up at its own layer index
instead of being smeared into the final logits.

## 6.6 P2/P3 first check: the Rust host reads the artifact (2026-10-09)

`cargo run -p ds4-core --example ds41_inspect -- <gguf>` on the Spark, against
the real 113.6 GB file:

| item | value |
| --- | --- |
| header | 73 kv entries, 1000 tensors, data_pos 4,787,744, alignment 32 |
| types | f32 531 / 171.6 MB, q4_k 330 / 4.41 GB, i32 1 / 517 KB, i64 3 / 832 B, bf16 65 / 333 MB, **vqblob 43 / 107.75 GB**, **fp8_32x32 27 / 880 MB** |
| identity | `general.architecture` = `deepseek4`, `general.name` = `DeepSeek V4.1 Flash` |
| V4.1 keys | all 18 present: context_length 1048576, kv sources 4, index sources 8, candidate source L20 (2048 blocks x 8), engram 2 layers (max n-gram 4), MTP 3 towers / 128 experts / block 5 / top 3 / noise 128799 / markov rank 256 / 3 target layers |

So the container is readable by this host today (the type table addition was
the only blocker), and the blob count confirms the tower contract: 40 layers
plus 3 towers, one blob each. Types 40, 41 and 43 do not occur in this
artifact; the towers use blobs too.

The V4.1 arm landed in three steps after this check: the variant/shape
(`d53079b`), the metadata wire with the bind (`0d3378d`) and the layout table
(`356a100`), all recorded in §6.6.1. The loader now accepts the artifact.

### 6.6.1 The V4.1 tensor contract, read from the artifact

Names dumped from the artifact itself (`ds41_inspect --tensors`, saved on the
Spark at `~/youngai/model/golden/ds41-names.txt`), not from the engine's list:

- Globals: `token_embd.weight`, `output_norm.weight`, `output.weight`,
  `engram.token_map|multipliers|primes|offsets`. There is no `output_hc_*`:
  the head reuses the last `ffn_pre`, exactly as the engine's bind comments say.
- Every layer: `hc_attn_fn|scale|base`, `attn_norm`, `attn_q_a`, `attn_q_a_norm`,
  `attn_q_b`, `attn_kv`, `attn_kv_a_norm`, `attn_sinks`, `attn_output_a`,
  `attn_output_b`, `hc_ffn_fn|scale|base`, `ffn_norm`, `ffn_gate_inp`,
  `exp_probs_b.bias`, **`ffn_exps_vq.blob`**, `ffn_gate_shexp`, `ffn_up_shexp`,
  `ffn_down_shexp`. The three per-expert tensors the V4 host bound do not exist
  in the artifact at all; the V4.1 catalog requires the blob instead.
- Source layers carry extra names, and which layers those are comes from
  metadata arrays, not a formula: `blk.20` (the candidate source) has
  `attn_compressor_kv`, `attn_compressor_norm`, `indexer.wk`, `indexer.k_norm`,
  `indexer.attn_q_b`, `indexer.proj`. Note what is absent: no
  `attn_compressor_ape`, no `attn_compressor_gate`. Layers with source tensors:
  1, 2, 8, 14, 20, 24, 28, 32, 36.
- Engram layers (1 and 14) add `engram_wkv` (fp8_32x32), `engram_q`, `engram_k`.
- Towers: `mtp.{0,1,2}.` mirror a layer's set including `ffn_exps_vq.blob`, plus
  the shared heads `mtp.main_proj`, `mtp.main_norm`, and the rest of the five
  named in the engine's bind; 72 `mtp.*` tensors in total.

The bind work item this section opened is DONE (2026-10-09, `0d3378d`): the
V4.1 catalog is resolved from the metadata wire, not the shape. `V41Wire::load`
mirrors `v41_load_metadata` (hard stop on a missing key, nearest-source wiring,
the two wiring checks) and `bind_names_v41` mirrors `weights_bind_v41` (blob
experts, source-layer compressor/indexer names with no ape and no gate at ratio
1, engram triples, three towers with the blob-or-per-expert form chosen by the
inventory). `resolve_bind_plan` dispatches both open paths; the shape-only
`bind_names` keeps returning the V4 catalog.

Measured state, before and after, on the Spark against the real artifact:

| run | result |
| --- | --- |
| before (`d9dadfb`) | `bind: slots=966 bound=843 required-missing=123` |
| after (`0d3378d`) | `bind: slots=1000 bound=1000 required-missing=0`, `v41 wire: kv-sources=[2,8,14,20] index-sources=[2,8,14,20,24,28,32,36] engram=[1,14] towers=3 experts=128` |

The catalog equals the artifact's published 1000 tensors in both directions;
`crates/ds4-core/tests/v41.rs` holds that gate model-free (the artifact's names
are a fixture, `tests/fixtures/v41/artifact-names.txt`, sha256
`a8b4a0e0385230df2e411c51c6586b92419867b5fa4b62fbf684ff249b0a2400`), and covers
the wiring derivation, the refusals, and both tower expert forms.

Next unit: the V4.1 layout arm (`expected_deepseek` branching on the variant,
driven by the same wire), which is what `validate_layouts` needs before
`probe_model_artifact` accepts the artifact end to end.

The layout arm is DONE (2026-10-09, `356a100`): `expected_layouts_v41` and
`validate_layouts_v41` take the same wire and inventory, the new classes are
`v41-skel` (q4_K or fp4x32, `expect_skel`), `v41-dense` (the `v41_tproj`
dispatch set), `v41-blob` (ndim only: the single dim is the per-layer byte
size) and `v41-ndim` (the shared draft heads the engine checks nothing for),
and `open_bind_plan` resolves, refuses and validates in one place for both open
paths. Gate: the table matches the artifact's own tensor directory, type and
dims for all 1000 tensors, as a model-free test over
`tests/fixtures/v41/artifact-tensors.txt`; on the Spark against the real file
the inspector now prints `layout: ok` and `bind: slots=1000 bound=1000
required-missing=0` after `identify` and `validate: ok`.

The loader half of P3 is therefore closed: the host accepts the artifact's
metadata, names, types and dims. Next: P2's remaining loaders (engram metadata
and table open, the sidecar `gr`/`rb` reader, the `base.fnv` check) and the
engram/sidecar state.

### 6.6.2 P2: the engram loader (2026-10-09, `79756c0`, `0af1d10`)

`EngramHash` mirrors `v41_engram_hash` and `EngramShard` mirrors
`v41_engram_open_shard`/`v41_edio_pread`; the wire gained the table metadata
(num_embeddings, both plane offsets, the per-layer paths, the pad id) and
`apply_engram_dir` for `--engram-dir`. Gates:

| gate | result |
| --- | --- |
| the engine's own hash function as a C harness (`fixtures/engram/gen_engram_ref.c`, compiled and run) | every row of both engram indices over every position reproduced |
| the artifact's real constants against the engine's captured golden rows (fixtures from `golden/p1*`) | all 8 positions x 2 layers identical |
| the Spark, real shards: `--engram-dir`, `--ids p1.ids`, `--erows-out` then `diff` against `golden/p1.logits.bin.erows_L01/L14.txt` | `EROWS-IDENTICAL`; both shards open with O_DIRECT (`direct=true`), sizes 101,535,150,936 and 101,537,926,640 exactly as §2 measured |

One real bug came out of the Spark run: aarch64 defines `O_DIRECT` as
0o200000 and `O_DIRECTORY` as 0o40000, the reverse of the asm-generic value
x86_64 uses, so the first Rust build opened the shard as a directory, failed
with ENOTDIR and silently took the FADV_RANDOM fallback (`direct=false` in the
run). `0af1d10` fixes the constant per arch and the comment records the
symptom; the fallback is a reproducibility risk (page-cache churn), not a
performance note.

### 6.6.3 P2: the sidecar readers and the base.fnv gate (2026-10-09, `dac818e`)

`GrSidecar`, `RbSidecar`, `AmpSidecar`, `deq_fp4x32` and `gr_dir_fnv` +
`check_base_fnv` mirror `core_v41_amp.c:27-223` and `ds4_gr_fnv.h:18-33`.
Gates:

| gate | result |
| --- | --- |
| the engine's own `ds4_deq_fp4x32` as a C harness (`fixtures/sidecar/gen_fp4_ref.c` with the vendored `ds4_fp8.h`) | bit for bit, including the e=0 and e=255 scale edges |
| the three readers against synthetic files of every storage type, plus header/type/truncation refusals | model-free tests |
| the Spark, real ② directory `…-grrb-vqfin41_vqhalf_a_n8192-engine`: `--zchain` | `gr=39 rb=27 amp=0 fnv=f4a3fd3988135e75/66`; `posttrain: base.fnv ok f4a3fd3988135e75/66` |
| the Spark, wrong ② directory `…-grrb-code_fit_n15360-engine` | `base.fnv MISMATCH want f4a3fd3988135e75/66 have 4530bf1df4ebeeaf/68`, refused |

The fingerprint check is a real-data end-to-end one: `f4a3fd3988135e75` and 66
files are what the post-train solver wrote into
`posttrain-experimental-20260924/base.fnv`, and the Rust FNV lands on it
exactly. The reader also decoded the real fp4x32 `gr_L00.bin` (`factor[0] =
1.125000` = stored `s-1` restored to `1+raw`). P2 is closed.

### 6.6.4 P3: the ②/③ zchain merge (2026-10-09, `a73fe31`, `bd393ad`)

`V41Zchain::load` mirrors `v41_amp_load` (`core_v41_amp.c:227-270`): the
`base.fnv` gate before any read, router biases from the ② directory only
(`v41_rb_load` is never called with ③), per-layer gains as the element-wise
product starting from 1 (② then ③), and the low-rank pairs concatenated by
rank, ② rows first, with β scaling ②'s A rows only and the rank sum capped at
8192. No directory is the naked base (`Ok(None)`); a directory pair holding
nothing at all is refused, and a broken file refuses the whole load instead of
half-loading. The ③-alone note fires before the gate, from the state attach
(`core_v41_state.c:79-83`), so a refusal still warns.

Gates:

| gate | result |
| --- | --- |
| model-free tests (`tests/zchain.rs`, 8) | the ②×③ product, fp4 `s-1` through the merge, rb ②-only (a ③ rb file is ignored), amp concatenation with β on ②'s A only (B and ③ untouched), the 8192 rank-sum refusal, a broken ② file stopping before ③ is read, the gate refusing before any layer file |
| the Spark, real pair ② `…-grrb-vqfin41_vqhalf_a_n8192-engine` + ③ `posttrain-experimental-20260924` | `zchain-merge: gr=39 rb=27 amp=0 k=none base=ok f4a3fd3988135e75/66 beta=1` |
| merged L39 (the only layer both directories carry) against an independent python f32 computation of `(1+r₂)×(1+r₃)` | bit-identical at f[0]/f[100]/f[1000]/f[4095] (`0x3f800000`/`0x3f810000`/`0x3f7ef400`/`0x3f7b0400`) and `nonzero=3246/4096` on both sides |
| the Spark, wrong ② `…-grrb-code_fit_n15360-engine` | `zchain-merge: sidecar-base-mismatch`, refused before any layer file |
| the Spark, ③ alone | the note, then the gate refusal with the seed/0 expectation (`have 14650fb0739d0383/0`) |

The `first gr L00 from2=true from3=false f[0]=1.125000000` line reproduces the
§6.6.3 sample through the merge. What has no live data: no directory on the
Spark carries `amp_Lnn.bin`, so the rank concatenation and β are covered
model-free only; their live gate arrives with P4's device upload. The engram
session state (the token history and the prefetch pipeline) stays with P4's
read path: the host-side pieces that are data and policy — the hash constants,
the row addressing, the shard open policy and the span checks — are already in
`engram.rs` and proven against the golden rows (§6.6.2), and the pipeline
itself is execution.

## 6.7 P4 reconnaissance: the native side (2026-10-09)

Three read-only passes mapped the ground before any kernel work (explore
agents over both trees; every citation below is re-checkable).

**ds4-dfm-rs native today — no V4.1 anywhere.** The DeepSeek arm is V4-shaped:
`weights_bind` (ds4.c:9333-9398) binds the per-expert trio and `output_hc_*`;
`weights_validate_layout` (6585-6650) validates them; decode runs
`metal_graph_encode_decode_layer_impl` (18491) with the expert matmul chosen at
19390-19402 (`ds4_gpu_routed_moe_one_tensor`); the variant reaches native only
as `ds4_host_shape {variant, n_compress, compress}`
(native/bridge/ds4_host_load.h:29-34), switched in `model_apply_host_shape`
(ds4.c:2507-2561) — variant 16 currently dies at the default arm (2560). Source
layers are ratio-derived there (compressor when ratio != 0, indexer when ratio
== 4; ds4.c:18760/18893); V4.1 needs the explicit source tables threaded in. So
the native port needs a variant skeleton before any kernel can be called.

**Family conventions (ds4-dfm-rs).** Kernels live in `cuda/<family>_*.cuh`
(device only); launchers in a root aggregator `ds4_<family>_gpu.cuh`
(extern "C"), prototypes in `ds4_gpu.h`, Makefile deps beside ds4_cuda.o's rule
(:803), ds4.c hooks via `ds4_<family>_{graph,bind,session,stub}.inc`. Tests are
`tests/test_<family>_<aspect>.cu` with a Makefile rule and a runner; they may
link `ds4_cuda_test_hooks.o` (ds4.c built with -DDS4_TEST_HOOKS) for reference
oracles, and print PASS/FAIL with the exit code as the verdict.

**The VQ decode contract (first kernel unit).** `v41_vq_fused_moe`
(cuda_vq_decode_launch.inc.cu:12) dispatches by (version, nbit) to four
instances; n=1 with v3 goes through the persist kernels
(cuda_vq_persist.inc.cu), v2 through gateup+down (cuda_vq_decode.inc.cu);
`v41_vq_reduce_kernel` folds the routed sum; the activation is packed to bf16
by `v41_vq_xpack_kernel` first. Dependencies to bring: v41_scratch/v41_grow,
v41_bf16r, `V41_GEMV_MAX_TOK` = 8, cuda_ok (already present, ds4_cuda.cu:1333),
a stream (ds4_cuda_moe_stream() is the fit), v41_pdl_wait/register, the DQVL
magics, g_v41_gr. The five decode scratch buffers are grow-only TU statics
(cuda_vq_decode.inc.cu:221). Out of scope for the n=1 unit: the group kernels,
the probe watchdogs, the order kernel, the whole prefill family, the backward
files.

**The artifact's real blob, measured on the Spark** (`blk.0.ffn_exps_vq.blob`,
abs 456249888, 2,767,968,272 B): DQVL v3, 384 experts; payload DQV3 d=8
nc=8192 rows=2304 cols=5120 flags=3 mnb=12 cb_off=9232 — **the v3 13-bit
instance with the 13th-bit plane** (`<13,1,1>`), the hardest of the four. The
layer codebook is 8192 E4M3 words shared by the layer's three matrices, at
blob+9232.

**The fp8_32x32 set** (27 tensors): the two `engram_wkv` (blk.1/14, 6144x25600)
and the 25 tower dense tensors (`mtp.{0,1,2}` attn/shexp plus
`mtp.main_proj`) — the fp8 path serves the engram wkv now and the towers in P5.

**Workflow.** This workstation has nvcc 13.3 (sm_121 cross-compiles without a
device) and an RTX 4070 SUPER (sm_89, 12 GB): kernel units can be developed,
compiled for both archs and tested with synthetic payloads locally; the
real-artifact gates stay on the Spark. One portability deviation is required
and named: `v41_pdl_wait`'s `griddepcontrol.wait` is sm_90+ asm, so the port
guards it with `__CUDA_ARCH__ >= 900` — identical code on sm_121, a no-op below
it, where no PDL graph can exist (the engine's own comment says the wait
returns immediately when the kernel is not launched through PDL).

P4 unit order, each with its own gate:

| unit | work | gate |
| --- | --- | --- |
| P4-0 | native V4.1 variant skeleton: variant 16, `DS4_SHAPE_V41_FLASH`, the host dispatch, the metadata selector/validation, the compress pattern, the source tables and engram wiring over the host-shape ABI, the bind/layout arm (blob experts, engram triples, no `output_hc_*`) | the native opens the real artifact and binds 1000/1000, equal to the Rust plan |
| P4-1 | VQ decode family port (format header, primitives, row, decode, persist, launch; n=1) | device == Rust `vq.rs` on a synthetic v3 13-bit payload (local 4070) and on the real layer-0 payload (Spark) |
| P4-2 | MoE launcher wiring in the native forward for variant 16 | one-hot probes bit-exact; the worker's routed sum against a Rust emulation within the recorded tolerance |
| P4-3 | fp8_32x32 decode and the engram read path | erows and the wkv matmul against the golden `hce_L01/L14` traces |
| P4-4 | forward bring-up on the golden prompts | G2: per-layer `x_Lnn`/`y_Lnn` traces first, then logits |
| P4-5 | the sparse-attention tensor-core family: the two-pass seg decode, the single-pass prefill mma, the scalar split-K, the merge, and the entry's try order (`cuda/ds41_attn_mma.cuh`) | G2: the n<=8 prompts bit-exact vs the bare golden (§6.12); the prefill arm (n>=64) and the 9..63 scalar band gate with unit C's longer prompt |
| P4-6 | the n>8 prefill arms: f32/bf16 dense GEMMs, the q4_K prefill GEMM, the VQ routed-MoE prefill (scheduler + the v2 fused arm + the v3 tensor-core vqm/vqs), and the cuBLAS-state fix (cublasSetStream before each ds41 GEMM, mirroring the engine's documented workspace reset) | G2: p1..p5 + p6k64 (n=64) bit-exact vs the bare golden, 258 zero-diff trace lines, zero nonzero (§6.13) |
| P4-7 | the zchain sidecar: the gr/rb override stores (`ds4_gpu_v41_set_gr_override`, `ds4_gpu_v41_set_rb_override` + the router-entry consult), the decode-arm gr wiring, and the harness `--zchain` loader | G2: the with-sidecar golden (the ORIGINAL capture) bit-exact on p1..p5 (215 zero-diff lines, zero nonzero, PPLs exact), plus the bare no-op control unchanged (258 zero-diff lines) (§6.14) |

## 6.8 P4-1 evidence: the VQ decode family vs the host oracle (2026-10-09)

Unit commits c9b5313 (the gate) and 84301f8 (the stream-ordering fix).

Built: `ds4_gpu_v41_vq_row_probe` (ds4_ds41_gpu.cuh) — one warp, n consecutive
rows against one activation. A one-hot x at column c makes row_dot exactly one
nonzero term, so out[i] IS the decoded value at (row+i, c): the compare is
bit-exact without replicating the kernel's accumulation order. Instance
dispatch mirrors v41_vq_fused_moe; a slot that fails to open writes NaN.
Fixtures (tests/fixtures/ds41/vq/gen_v3_13b.py, written from the spec, never
from a decoder): v3 13-bit with plane at the real column geometry, v3 12-bit,
v2 12-bit, v2 11-bit; v3 refs from the generator's own arrays (locally) or
vq.rs (the ds41_vq_ref example, which extracts one expert into a mini-blob);
v2 refs from the vendored ds4vq_dequant_f32 inside tests/test_ds41_vq.cu.

Local (RTX 4070 SUPER, sm_89), `make test-ds41-vq`:

    DS41 VQ row probe: PASS (blob v3, 35 probes, 1432 values bit-exact)
    DS41 VQ row probe: PASS (blob v3, 13 probes, 265 values bit-exact)
    DS41 VQ row probe: PASS (blob v2, 9 probes, 198 values bit-exact)
    DS41 VQ row probe: PASS (blob v2, 9 probes, 198 values bit-exact)

and `cargo run --example ds41_vq_ref -- tests/fixtures/ds41/vq/v3_13b.blob 0 0
<dir>` reproduces the generator's v3 ref byte for byte (vq.rs == generator).

Spark (GB10, sm_121): the four fixtures PASS with the same refs, then the
real layer-0 payload (blk.0.ffn_exps_vq.blob at abs 456249888):

    which 0: rows=2304 cols=5120 nc=8192 probes=13 values=9225
    which 1: rows=2304 cols=5120 nc=8192 probes=13 values=9225
    which 2: rows=5120 cols=2304 nc=8192 probes=9 values=15366
    blob ver 3 nexp 384 expert 0: 35 probes, 33816 values
    DS41 VQ row probe: PASS (blob v3, 35 probes, 33816 values bit-exact)

The three real payloads measure 2400808/2400808/2406440 B — exactly the
extraction lengths computed from the payload headers (no padding).

Two findings the gate produced:

- The first generator interleaved the 13th-bit plane per row; the device and
  vq.rs agreed against it and exposed the layout error: the plane region
  follows ALL rows' main streams (m.ex = m.ix + rows*mrow,
  cuda_vq_row.inc.cu:41).
- A rare Spark flake (once: 2/35 probes returning the PREVIOUS probe's value
  bit-for-bit; 20 further pre-fix runs clean): the probe launches on the
  non-blocking moe stream, which does not inherit the legacy default stream's
  ordering, and a pageable H2D cudaMemcpy returns once staged, its DMA
  possibly still in flight. Fixed in 84301f8 (event on the legacy stream, the
  moe stream waits on it); post-fix 50/50 runs clean.

## 6.9 P4-0 evidence: the native opens the artifact (2026-10-09)

Unit commit 06ba7b2. Before it the native could not open the artifact at all:
variant 16 died in `model_apply_host_shape`, the V4 bind required per-expert
tensors that do not exist in V4.1, and the type table lacked ids 40-44 so the
C path could not even parse the file. What landed: variant 16, the
`DS4_SHAPE_V41_FLASH` literal (core_shape_select.c:81), the metadata selector
arm, the wiring state `g_ds4_v41` (mirror of ds4_internal.h:57-82), the C-path
wiring loader (core_validate_v41.c:46-154, keys included), the V4.1 bind and
layout arms (core_bind_v41.c:41-202: blob experts, engram triples,
source-layer compressor/indexer names with no ape and no gate at ratio 1, no
`output_hc_*`), the draft towers' native home
(core_draft_tower_types.h:13-32), the five type-table entries, and a
`DS4_HOST_BIND_CENSUS=1` diagnostic that counts the native's own name
resolutions against the host plan.

The ABI-delta table (the P4-0 contract; ds4_host_shape in
native/bridge/ds4_host_load.h, mirror crates/ds4-sys/src/lib.rs, filled in
crates/ds4-core/src/lib.rs from V41Wire, consumed by
`model_apply_host_v41_wiring`): the three existing fields keep their order;
appended, borrowed and n_layer long, NULL/0 for other variants:
`v41_kv_source` (bind: compressor/indexer names per layer), `v41_index_source`,
`v41_kv_source_of` / `v41_index_source_of` (stored; forward P4-2),
`v41_engram_index_of` (bind: engram triples), `v41_n_engram`,
`v41_engram_layers`, `v41_engram_max_ngram`/`heads`/`head_dim` (layout dims),
`v41_engram_pad`, the candidate triple (stored; P4-2), `v41_mtp_towers` /
`v41_mtp_experts` (bind: tower names). Not carried yet, named: engram
rows/weight_off/scale_off/table_path (P4-3), ctx, the DSpark draft parameters
(P5).

Two findings this unit produced:

- `host_compress_ratios` derives the V4 formula, which returns zeros for
  V4.1; the ABI now carries the metadata ratios (`wire.compress_ratios`), or
  the native would have dropped the ratio-2 compressor gates and bound the
  wrong tensor set.
- The artifact's indexer key projection is `blk.L.indexer.wk.weight`; the V4
  field for that role is `indexer_attn_k`, so the layer struct gained
  `indexer_wk` (the engine's V4.1 name, core_bind_v41.c:76).

Gates, both on the Spark against the real artifact (logs /tmp/p4_0_c_gate.log
and /tmp/p4_0_rs_gate.log):

C path, `DS4_HOST_BIND_CENSUS=1 ./ds4-c --inspect --model <artifact>` (the
metadata loader, bind and layout; no host plan):

    ds4: engram table 0 not readable at /home/fodelf/... (warn only; the converter's path)
    ds4: engram table 1 not readable at /home/fodelf/... (warn only)
    ds4: V4.1 wiring: kv sources 4 / index sources 8 / candidate L20 (2048 blocks x 8) / engram 2 layers (384006168+384016682 rows on disk)
    ds4: [v41] layout ok (skel q4_K/fp4x32, 40 layers, 3 towers)
    ds4: bind census: bound=1000 required-missing=0 optional-missing=0 asks=1000 (no host plan)

plus the model summary (DeepSeek V4.1 Flash, 1000 tensors, train context
1048576, vqblob 43 / fp8_32x32 27). The wiring line equals the Rust plan's
(§6.6: kv sources [2,8,14,20], index sources [2,8,14,20,24,28,32,36], engram
[1,14]), and asks=1000 proves the native's ask set is exactly the plan's
1000 slots.

Rust host path, `DS4_HOST_BIND_CENSUS=1 ./ds4 --model <artifact> --cpu
--lifecycle -c 4096` (the extended ABI; the native never re-reads the
metadata on this path):

    ds4: V4.1 wiring: kv sources 4 / index sources 8 / candidate L20 (2048 blocks x 8) / engram 2 layers
    ds4: bind census: slots=1000 resolved=1000 bound=1000 required-missing=0 optional-missing=0 asks=1000 fallback=0
    lifecycle ok backend=Cpu ctx=4096 pos=0

`resolved=1000` is a bitset count of DISTINCT plan entries, so it cannot be
inflated by duplicates; `fallback=0` proves every name the native asked was in
the plan. The census equals the Rust plan's line (`bind: slots=1000 bound=1000
required-missing=0`, §6.6.1) exactly. Local (RTX 4070 SUPER): the sm_89 link
(`make ds4`) and the model-free suites (v41 11/11, catalog 5/5, ds4-core 362
passed) are green; the one ds4-core failure seen once is the pre-existing
`cpu_quote_uses_ram_not_discrete_fb` meminfo race (passes standalone).

Next: P4-2 (the MoE launcher wiring in the native forward for variant 16).

## 6.10 P4-2 evidence: the MoE entry, and what actually failed the gate (2026-10-09)

Unit commits: 5b19e39 (entry + wiring WIP), 263f843 (the harness fix and the
criterion), 2fa9b89 (geometry parameters), plus the closing commit. What
landed: the tensor-level entry `ds4_gpu_v41_routed_moe_tensor` (host blob
header read, range-resolved device pointer, ver/nc dispatch), the decode-path
arms for variant 16 in ds4.c (the routed VQ call, the shared expert through
`plain_graph_matmul_tensor`, fused shared gate/up and down_hc forced off for
v41), the whole ds41 family on `ds4_current_stream()` (the engine's
g_cur_stream convention; the P4-1 event band-aid removed), the MoE fixture
(gen_moe.py: 2 experts, shared gate/up slots, one probe down with a single
nonzero column, 13-bit plane states covered), the Rust emulation
(ds41_moe_ref.rs), the device gate (tests/test_ds41_moe.cu), and the geometry
parameters so the same gate can drive the real payload.

The gate failed for three days of wall time on a HARNESS bug, not a kernel
bug, and the distinction is the unit's main lesson:

- `parse_cases` read with fgets into an 8 KB buffer; an x line is IN_DIM hex
  words (~18 KB at IN_DIM 2048). Every x was silently truncated after ~910
  values: one-hot cases 4-5 (c=1024, 2047) read as all-zero x and returned
  exact zeros, and the random cases lost most of their activation. The
  earlier "mid wrong 1419/2048" bisect was this truncation, reproduced
  against the emulation. getline fixed it (263f843).
- The persist kernels were never at fault: with the activation intact, six
  onehot cases and one dense random case are bit-exact, and the one
  remaining divergence is fully measured (below).

Acceptance criterion (random cases) and why: the emulation is an
order-divergent oracle (sequential f32 sums; the device is lane-split dot8
chains with FMA contraction and --use_fast_math expf). Both round the same
mathematical value; where an intermediate sits within ~1e-5 relative of a
bf16 quantization boundary the two round to different sides. Measured on the
synthetic case 6: ONE gate dot 5e-5 from a boundary flips 2 bf16 ulps, the
mid inherits it, 15/3072 down partials flip by <=5 bf16 ulps, and the folded
output moves by sum_k |w| * that quantum. Per-element relative error blows
up wherever the weighted partials cancel (worst 0.107 on a value ~20 among
outputs ~1e5), so the gate compares against the reference vector's own
scale: max |delta| <= 1e-3 * max |ref| (measured 2.74e-4 synthetic,
7.7e-4/2.6e-4 real). The mid/partial dumps are what made this measurable
(DS41_MOE_DUMP_MID, DS41_MOE_REF_DUMP; both kept, diagnostic-only).

Gates:

1. Local (RTX 4070 SUPER, sm_89): `make test-ds41-moe` 8/8 PASS — cases 0-5
   onehot bit-exact, case 7 random bit-exact, case 6 scale 2.743e-04.
2. Spark (sm_121): `make test-ds41-moe` 8/8 PASS, numbers identical to
   sm_89 (case 6: max abs 4.608e+02, 5 diffs, scale 2.743e-04) — the
   divergence is deterministic across architectures.
3. Spark: `make test-ds41-vq` 70/70 PASS (4 fixtures, 2093 values
   bit-exact) — the P4-1 regression.
4. Spark, the real layer-0 payload (blk.0 at abs 456249888, expert 0
   extracted by ds41_vq_ref; gate/up [2304][5120] R=20, down [5120][2304]
   R=9 — the persist path's real regime, both cases random K=6):
   `./tests/test_ds41_moe <blob> <cases> <ref> 5120 2304 5120` 2/2 PASS
   (scale 7.716e-04 and 2.648e-04).
5. Spark, same payload at the dump level: mid 0/13824 bit-diffs
   (bit-exact), down partials 12/30720 at exactly 1 bf16 ulp.
6. Spark, the engine's own equivalence doctrine ("gate = cmp against the
   v2-shaped kernels"): DS41_VQ_NO_PERSIST=1 on the synthetic fixture gives
   the same 8/8 with identical numbers, and on the real payload the persist
   and plain paths' mid and partial dumps are byte-identical (cmp). The
   switch stays as a diagnostic; it cannot run on sm_89 (the plain gateup
   kernel needs 92 registers at 1024 threads there).

Contracts and notes:

- The n=1 persist kernels pipeline 8-round blocks: every row needs
  R = cols/256 >= 8 rounds and every span (n rows) >= 8 rounds. The real
  artifact is R=20/9; the synthetic fixture sits at the R=8 floor. A smaller
  artifact would decode garbage in rows 2+ of a span — this is a payload
  format precondition, not a runtime check.
- The shared-expert arm (plain_graph_matmul_tensor, q4_K) is wired but not
  exercised by this gate: it runs with the forward (P4-4).
- The scale tolerance's margin on the real payload is 1.3x (7.7e-4 vs
  1e-3) — thin, and honest to record. It absorbs boundary-rounding noise
  only; a structural error moves whole outputs far past it (verified during
  the bisect: the truncated-activation failure sat at scale ~1).

Next: P4-3 (fp8_32x32 decode and the engram read path).

## 6.11 P4-3 evidence: fp8_32x32 and the engram chain on real rows (2026-10-09)

Unit commits: 6b06531 (the engram wiring over the host ABI), de38264 (the
device port + gates), deb785f (real-mode argv fix).

Built:

- Host ABI (step 2): `ds4_host_shape` gains `v41_engram_rows`,
  `v41_engram_weight_off`, `v41_engram_scale_off` and
  `v41_engram_table_path` (native/bridge/ds4_host_load.h + the ds4-sys
  mirror), filled from `V41Wire` in ds4-core and consumed in
  `model_apply_host_v41_wiring` (ds4.c) under the same count-without-arrays
  rule as the wiring arrays; the shard paths travel as borrowed NUL-terminated
  pointers bounded by the native's 1024-byte buffer. The stale "not on this
  ABI yet (P4-3)" comment is gone and `v41_report_wiring` prints the row
  counts on the host path too.
- `cuda/ds41_fp8blk.cuh` (device): the fp8_32x32 decode family — the NT/XB
  GEMV kernel, the grouped `grid.y` row-offset form, the wkv entry's
  bf16+cuBLAS arm for n>8, the round_out pass and the engram row dequant;
  sources cited per section in the file header. Named port deviation: no
  `cublasSetStream` — the port does not compile with `--default-stream
  per-thread`, so stream 0 is the legacy default for both our launches and
  cuBLAS; the engine hands cuBLAS `cudaStreamPerThread` only because of PTDS
  (`cuda_v41_1.inc.cu:56-58`). The split-K workspace prep follows the port's
  S1.1a rule.
- `cuda/ds41_engram.cuh`: the gate kernel (`cuda_v41_3.inc.cu:109-134`) and
  the read path's device primitives — `ds4_gpu_host_alloc/free` (pinned
  mapped), `ds4_gpu_host_flag_wait` (the 1-thread `__ldcv` spin with the ~5 s
  timeout) and `ds4_gpu_tensor_write_zerocopy` (the `__ldcv` hostcopy
  kernel), all from `cuda_decode_graph.inc.cu:118-171` /
  `cuda_graphcap.inc.cu:254-259`. The host half of the read path is the P2
  Rust reader (`EngramShard`, §6.6.2).

The recon finding that corrects the gate line: §6.7's P4-3 row said "erows
and the wkv matmul against the golden `hce_L01/L14` traces", but the golden
`hce` is the engram OUTPUT (hc after the update — the engine dumps it after
the gate, `core_v41_engram.c:471-475`), and reproducing it needs `hc_before`
(the layer-entry hc), which no golden file carries (the set has per-layer
x/y, erows, hce and logits only; the layer-entry hc exists solely in the
ptrain save path, `core_v41_forward.c:190-191`). The hce end-to-end
comparison therefore lands in P4-4's G2, where the port produces `hc_before`
itself. P4-3 gates what its inputs allow, on the engine's own row ids: the
erows against the golden files, then the real rows -> decode -> wkv chain
against an f64 emulation of the same bytes.

Gates:

1. Local (RTX 4070 SUPER, sm_89), `make test-ds41-fp8` 9/9 PASS: onehot
   cases bit-exact (plain n=1, round n=2 with round_out, grouped n=2);
   dense n=1/n=2 at 1.6e-7/1.1e-7 relative (the f32 GEMV vs the f64
   emulation); n=12 (the >8 bf16+cuBLAS arm) at 2.2e-6; the round_out dense
   case 2.0e-3 with 3 diffs, covered by the recorded allowance: max |delta|
   <= max(1e-3 * max|ref|, one bf16 quantum at max|ref|).
2. Local, `make test-ds41-engram`: rows bit-exact (1280 values); the read
   path live (pinned zero-copy upload, host flag wait); gate cases: absorb
   and zero bit-exact, dense within the term-relative bound
   (1e-5 * max(|got|,|want|,|h|,|val|) + one bf16 ulp at the output). The
   measured worst dense element: |d| 9.5e-7 at a 1.5e-4 output where h and
   gate*val cancel (val 0.32) — 3e-6 relative to the update scale; a bound
   on the output's own ulp alone would fail a correct kernel at 2 grid
   steps there.
3. Spark (sm_121): `make test-ds41-fp8` 9/9, numbers identical to sm_89
   (the >8 cuBLAS arm's accumulation differs slightly across archs:
   1.38e-6 vs 2.20e-6, both far inside the criterion); `make
   test-ds41-engram` PASS with identical numbers; the regression
   `make test-ds41-vq` 70/70 and `make test-ds41-moe` 8/8.
4. Spark, erows re-run with the P4-3 ABI in place (`ds41_inspect
   --engram-dir ... --ids golden/p1.ids --erows-out`): diff against
   `golden/p1.logits.bin.erows_L01/L14.txt` gives EROWS-L01-IDENTICAL and
   EROWS-L14-IDENTICAL.
5. Spark, the real chain (`ds41_engram_real.rs` preads the golden row ids
   through EngramShard and reads the layer's wkv tensor from the GGUF; the
   test's `--real` mode runs the device rows + wkv on those exact bytes):
   L01 rows 49152 values bit-exact, wkv max rel 9.25e-8 (max abs 7.6e-6 at
   scale 82.5); L14 rows bit-exact, wkv max rel 8.2e-8 (max abs 1.5e-5 at
   scale 185.8).

Fixture notes (kept): the fp8/engram images are page-aligned with
page-aligned tensor offsets, because the range resolver's register tier
page-rounds ranges and a 16-byte-aligned base makes neighbor registrations
overlap — the source then straddles a boundary and the cold copy fails with
invalid argument (the engine's documented kv_rms_weight shape,
`ds4_cuda.cu:1795-1803`; the first fixture version failed its grouped cases
exactly there). The engram fixture uses the artifact's own geometry (E=5120,
HC=4, head_dim=256, eps=1e-20) at n=2.

Next: P4-4 (the forward bring-up; G2: per-layer x/y traces first, then
logits — where hce_L01/L14 joins the comparison via hc_before).

## 6.12 P4-4 evidence: the forward bit-exact on the golden (2026-10-09)

Unit commits: 1e7eb72 (the eager forward + trace harness; the kernel
families e877471/6b59081), then the blob-residency fixes 3d8779f, ac21f4c,
dd4ac05, 4b2b3e5, bb74bc0; the mma attention family (P4-5) landed as
6ad1ace.

Built: the eager single-state forward (`ds4_ds41_forward.inc`, the engine's
`core_v41_forward.c` driver) and the trace harness
(`tests/test_ds41_forward.cu`: golden ids -> the score entry -> per-layer
`x_Lnn`/`y_Lnn` + `hce_Lnn` + logits compared against the golden dir,
argmax + both PPLs). The golden must be the BARE variant (the capture's
NO_ZCHAIN=1): the port does not apply the zchain sidecar yet (unit D), so
the with-sidecar golden would measure the sidecar, not the port. Bare
values: p1 46.8780 (S=8), p2 22.0679 (S=11), p3 9.5414 (S=7), p4 10.7011
(S=18), p5 6662.9080 (S=7).

Root cause 1 (nondeterminism + 3-13 s forwards): the flat 24 GiB cache
budget (`WEIGHT_CACHE_LIMIT_GB`) funded only 37/85 units (23.97 GiB); the
~96 GiB of VQ blobs stayed on the reclaimable mapping, and GB10 HMM does
not mlock `cudaHostRegisterMapped` (the port's own note, ds4_cuda.cu
~4601-4618), so cold pages read through the mapping are slow and can
return garbage. Fixed by porting the engine's unified budget rule
(unified memory -> total_phys - 8 GiB; `cuda_q8_repack_1.inc.cu:399-417`)
and the eager source-page drop (`cuda_modelmap.inc.cu:330-343`). Evidence:
with the mechanism forced through the existing knobs
(`DS4_CUDA_WEIGHT_CACHE_LIMIT_GB=100 DS4_CUDA_EAGER_SOURCE_DISCARD=1`),
two runs gave bit-identical logits and all 40 layer dumps.

Root cause 2 (deterministic garbage logits, PPL 372234.9112, identical
across four runs): the unit compiler merged the mtp.2 VQ blob (DQVL v3,
nexp 128) with ~132 MB of trailing tensors (markov_embd/head, confidence,
both out_norms) into one 938 MiB unit; the aligned copy relocated the whole
mixed span, and the head's output_norm resolved into the relocated image.
Fixed by the engine's "a blob is its own span" rule: `DS4_TCAT_VQ_BLOB` +
never-merge (`core_model_map.c:113-115, 184`) with a compiler self-test
(ds4_server.c). `DS41_DUMP_HEAD` forensics proved it: unit[83] off
112199472480 bytes 984322592; the norm served from 0x1ea0000000 +
984302112 = its exact source delta (/tmp/p44h2.log).

The vq-align mechanism (the engine's `cuda_vq_align.inc.cu`): each VQ
blob's payloads relocate to 128 B bitstream alignment into a device-
resident image. v1 copied straight from the cold mapping and wedged the
boot (100% user spin, idle GPU, frozen at 32.54 GiB); v2 (ac21f4c) builds
the image on the host and uploads it pageable. `DS4_CUDA_NO_VQ_ALIGN=1`
is the flat-path kill switch (clean cold). Boot cost recorded: 76-117 s
aligned vs ~62 s flat; revisit if it matters.

The P4-5 mma family (6ad1ace): ported the engine's live decode family
(`cuda_sparse_attn_mma.inc.cu`, `cuda_v41_attn_split.inc.cu`,
`cuda_v41_attn_mma_decode.inc.cu`) and the entry's try order
(`cuda_v41_2.inc.cu:229-303`: mma decode for n<=8 at any key count, the
draft-block arm, scalar split-K, prefill mma n>=64, scalar fallback). The
P4-4 named deviation (scalar-only attention) is closed for n<=8; the
scalar kernel remains exactly where the engine keeps it (the 9..63 band
and refused shapes). The family is not bit-equal to the scalar kernel by
construction (tensor-core accumulation order, different online-softmax
grouping) — which is why the delta below is the measurement that matters.

Gates:

1. Determinism (before P4-5, at bb74bc0; default config, no env knobs):
   E8-E12, five runs into unique /tmp dirs — 87 units funded 87/87,
   populated 105.75 GiB, [vq-align] active, every run bit-identical
   (logits + all 80 layer dumps), every PPL 47.6362, every forward 0.1 s.
   That PPL was the scalar-attention value; the delta vs the golden
   46.8780 (1.6%) was the recorded scalar-vs-mma accumulation-order gap.
2. The P4-5 gate (6ad1ace, this session; default config): p1 (n=8) logits
   0/1034240 diffs, argmax 8/8, PPL 46.8780 == golden; p3 (n=7) 0/904960,
   7/7, 9.5414; p5 (n=7) 0/904960, 7/7, 6662.9080; every layer
   `x_Lnn`/`y_Lnn` and `hce_L01/L14` zero-diff (129 zero-diff lines, zero
   nonzero-diff/FAIL/mismatch lines in /tmp/p45a + /tmp/p45c). Boot 76.0 s
   with vq-align; "[attn] decode tensor-core shared 41 KB enabled". The
   1.6% delta collapsed to 0.00%: the attention family was the only
   accumulation-order difference between port and engine on this path.
3. p2 (n=11) and p4 (n=18) are blocked by unit C, not by this unit: the
   forward hits "f32 prefill (n_tok 11 > 8) is not ported yet" at the MoE
   router gate (ds4_ds41_forward.inc:543 ->
   ds4_gpu_v41_matmul_f32_tensor, ds41_dense.cuh:199-200). The n>8 arms
   (unit C) also unlock the 9..63 scalar attention band's gate.

Local gates (this session): ds4_cuda.o builds for sm_89;
test-ds41-vq / -moe / -fp8 / -engram all pass.

Diagnostics kept (env-gated): `DS41_DUMP_HEAD` (head resolve + raw bytes +
covering ranges/units), `ds4_gpu_v41_debug_read_weight` /
`ds4_gpu_v41_debug_dump_raw`, `DS4_CUDA_NO_VQ_ALIGN`.

Unit: P4-4 complete

## 6.13 P4-6 evidence: the n>8 prefill arms + the cuBLAS-state fix (2026-10-09)

Unit commits: cc10fb2 (the four prefill families), d34b3ba (the long-prompt
fixture), 27b285d (the cuBLAS-state fix), 451862c (the layer-0 stage-dump
instrument, kept).

1. What was ported (all engine-cited in the file headers). The f32 and bf16
   dense prefill arms (cuda_v41_1.inc.cu:291-296, :308-330); the q4_K prefill
   GEMM with its to-bf16 kernel and the row tiling by V41_BF16_STAGE_ELEMS
   (cuda_v41_q4k.inc.cu:183-205, :330-390; the engine's v41_wc_* weight cache
   is training-only and is a named omission); the VQ routed-MoE prefill as
   three files — the scheduler (cuda_vq_prefill.inc.cu: header cache, stable
   counting sort, fixed-pick reduce), the v2 fused arm
   (cuda_vq_prefill_fused.inc.cu) and the v3 tensor-core arms
   (cuda_vq_prefill_mma.inc.cu vqm + cuda_vq_reg_mma.inc.cu vqs, selected by
   the engine's own shape and alignment checks; vqs's sm_90+ L2 bulk prefetch
   is compiled out below sm_90, a hint only). The retired NVFP4 path and the
   backward capture are not ported (named). The routed_moe entry now takes
   the engine's n<=8 decode / n>8 prefill split.

2. The divergence that this unit's gate first exposed, and its root cause.
   With the four families in, p1/p2/p3/p5 were bit-exact but p4 (n=18)
   diverged: x_L00 differed in 71 values (K=17: 69), rows 1..15, all on the
   bf16 grid, 1-4 bf16 ulps, PPL 12.35 vs the golden 10.70. Bisection on the
   Spark (fresh engine runs on truncated ids; the engine is causal and
   deterministic): bit-exact at K=12 and K=16, divergent at K=17/18. Both
   sides switch at n=17 (the engine's own K=16 vs K=18[:16] delta is 11199
   values, the port's 11203; K=17 == K=18[:17] exactly on both sides), and
   the two sides' 16->18 deltas disagree in 62 positions. Layer-0 stage
   dumps (DS41_DUMP_L0) put the switch inside v41_attention: the attention
   input is width-invariant (l0axn 0 diffs at 16/17/18), the attention
   output switches (l0ao 18038). No n=17 threshold exists in either tree
   (checked: 8/64/1/2048 are the only gates); the 16->17 change is cuBLAS's
   own N-dependent kernel pick, which both sides inherit because they load
   the same libcublas.so.13. Root cause of the residual 62: the engine calls
   cublasSetStream(g_cublas, v41_cublas_stream()) before every V4.1 GEMM
   (cuda_v41_1.inc.cu:219,259,292,318,360; cuda_v41_q4k.inc.cu:359), and per
   the cuBLAS docs (2.4.7) that call unconditionally resets the bound
   workspace to the default workspace pool — so the engine's GEMMs actually
   run on the default pool, not on the 32 MB user buffer its init installs
   (cuda_lifecycle.inc.cu:25-28). The port never made the call and kept its
   user workspace bound; cuBLAS keys algorithm selection on the bound
   workspace, so the two sides picked different kernels at N>=17. The fix
   (27b285d) mirrors the call in the four ds41 GEMM entries (f32, bf16, q4k,
   fp8blk wkv; the grouped q4K entry reaches it through v41_q4k_gemm). The
   port's stream is the legacy default, so ordering is unchanged;
   cuda_cublas_ws_prep now only touches the unbound user buffer. Measured
   proof, same session: pre-fix at K=17 vs the engine 69 x_L00 diffs;
   post-fix K=16/17/18 vs fresh engine runs bit-identical (logits and
   x_L00); pre-fix vs post-fix differ. A local sm_89 sweep of the same
   shapes found the binding inert there — the heuristic's bucket differs by
   arch; sm_121 is where it bites.

3. The gate (post-fix, one invocation, bare golden + the long-prompt
   fixture): p1 (n=8) 0/1034240 diffs, PPL 46.8780; p2 (n=11) 0/1422080,
   22.0679; p3 (n=7) 0/904960, 9.5414; p4 (n=18) 0/2327040, argmax 18/18,
   10.7011; p5 (n=7) 0/904960, 6662.9080; p6k64 (n=64) 0/8273920, argmax
   64/64, 8.1912. 258 zero-diff lines across the x/y/hce traces, zero
   nonzero-diff/FAIL/mismatch. p6k64 = the first 64 ids of the 835-token
   fixture (tests/ds41-long-prompt.txt, captured with the engine like
   p1..p5): n=64 exercises the sparse-attn prefill mma at its exact
   threshold and the VQ prefill at 384 pairs. p2 and p4 also pass the
   9..63 scalar attention band's gate, deferred by P4-5.

4. External review, on the record: three independent reviewers (two peer
   open-grok sessions — the engine's own YoungAi session and a second
   ds4-dfm-rs session — plus a GPT ASTRA 6 subagent) converged on the same
   candidate from the source and the vendor doc; the sm_89 measurement said
   inert, the sm_121 measurement settled it. The peer sessions' enumerations
   (no 16/17 threshold anywhere; the workspace as the only GEMM-affecting
   state difference; the prefill-mma/9..63 arm matching arm-for-arm) are
   consistent with the measurement.

5. Recorded gap: a >64-token logits gate. The engine's own trace gate
   (core_v41_score.c:36 sets dump_prefix only when n <= 64) withholds both
   the per-layer traces and the engram erows for longer prompts, so the
   835-token fixture cannot be fed to the harness (no rows) nor compared
   (no traces). Gating it needs the engram row-hash ported into the harness
   (validated against the golden erows at n<=64) — a future instrument, not
   a port unit. The multi-slice vqs/vqm work items (an expert with >8 tokens
   of the batch) therefore remain ungated; they need that instrument.

Diagnostics kept (env-gated): `DS41_DUMP_L0` (layer-0 stage dumps: the
attention input, the attention kernel output, the o-projection output, the
post-attention hc), in addition to the P4-4 set.

Unit: C complete (P4-6)

## 6.14 Unit D evidence: the zchain gr/rb sidecar (2026-10-09)

Unit commit: e41b495 (the two stores, the decode-arm wiring, the harness
loader).

1. What was ported (engine-cited in the file headers). The gr store
   `ds4_gpu_v41_set_gr_override` (cuda_vq_prefill.inc.cu:246-258: free the
   old table, cudaMalloc n_expert*out_dim*4, H2D; host NULL unloads) next to
   `g_v41_gr` in cuda/ds41_vq_prefill.cuh; the rb store
   `ds4_gpu_v41_set_rb_override` (cuda_v41_3.inc.cu:48-77: the on-disk
   exp_probs_b read by UVA cudaMemcpyDefault, + delta, a process-level
   <=64-entry device table keyed by bias_offset, host NULL unloads all/one)
   in cuda/ds41_router.cuh, with the router entry now consulting the table
   (engine :85). The decode arm of `ds4_gpu_v41_routed_moe_tensor` passes
   `g_v41_gr` exactly as the engine does (cuda_v41_3.inc.cu:189-192). The
   consumers already existed from P4-1/P4-6 and match arm-for-arm: the
   row-dot gain multiply `acc * f16(m.gr[r]) * (m.gov ? m.gov[r] : 1)` is
   identical on both sides (ds41_vq_row.cuh:265-266 vs cuda_vq_row.inc.cu:248;
   the prefill mma at ds41_vq_prefill_mma.cuh:210 and the vqs form at :447
   vs cuda_vq_prefill_mma.inc.cu:192 / cuda_vq_reg_mma.inc.cu:191).

2. The gate-side loader. The harness gains `--zchain <dir>`: a C mirror of
   the engine's single-directory loader (core_v41_amp.c:44-117 — gr header
   {n_expert, D, 1=f32 | 2=f16 | 43=fp4x32 storing s-1}, rb header
   {n_expert, 1=f32}, fp4x32 = the 17-byte block decode of
   ds4_quantfmt.c:28-40 over the ds4_fp8.h primitives), validating the
   headers against the model shape through two new accessors
   (`ds4_v41_shape`, `ds4_v41_router_bias_ref` — the engine's
   model_find_tensor + m->map/m->size, core_v41_amp.c:104) and mounting in
   the engine's order (rb unload-all first, v41_amp_load :227-229). A
   present-but-broken file is a hard stop, never a skip. The real directory
   mounts gr=39 rb=27 — the same counts the engine's own golden log printed
   (golden/p1.log: "27 层挂上路由偏置侧车" / "39 层挂上逐专家增益覆盖").
   Production keeps the Rust reader (sidecar.rs, P2); the C copy is the gate
   driver only.

3. Gate B — the with-sidecar golden (the ORIGINAL capture, `--zchain`): one
   harness invocation, p1..p5, 215 zero-diff lines, zero
   nonzero-diff/FAIL/mismatch: p1 (n=8) 0/1034240 diffs, argmax 8/8, PPL
   112.0845; p2 (n=11) 0/1422080, 11/11, 24.9582; p3 (n=7) 0/904960, 7/7,
   7.2809; p4 (n=18) 0/2327040, 18/18, 14.3524; p5 (n=7) 0/904960, 7/7,
   876.7182 — every PPL equal to the golden's to the printed digit, every
   x/y/hce trace bit-identical. This closes both halves at once: the gr
   application (39 layers; the decode arm for p1/p3/p5, the prefill arm for
   p2/p4) and the rb application (27 layers, routing moved through the
   selection score).

4. Gate A — the bare no-op control (no `--zchain`, the bare golden):
   p1..p5 + p6k64 unchanged from unit C's numbers (46.8780 / 22.0679 /
   9.5414 / 10.7011 / 6662.9080 / 8.1912), 258 zero-diff lines, zero
   nonzero — the new code is inert when nothing is mounted (g_v41_gr
   all-NULL, g_v41_rb_n = 0).

5. Recorded limits. The amp half of the sidecar is not ported: the real
   directory carries no amp_Lnn.bin (39 gr + 27 rb + manifest.txt, amp=0)
   and no Spark directory does, so there is nothing to gate. Its apply point
   is the engine's v41_layer tail (core_v41_forward.c:148-155, y += x·(B·A)
   rounded to bf16; the towers take their own ampA at :149; the solve hook
   :152 can end the forward) and its loader rides the same amp-directory
   wiring (core_engine_open.c:122-124 → v41_amp_load, core_v41_amp.c:223) —
   recorded for that future unit together with the base.fnv fingerprint gate
   (v41_pt_base_ok, core_v41_amp.c:163: FNV-1a over gr_Lnn.bin in layer
   order; missing warns and allows, mismatch stops). Single-directory
   captures have no base.fnv, consistent with the per-file mount rule.

Unit: D complete

## 6.15 Unit E evidence: the greedy loop, the DSpark spec round and the decode graph (2026-10-10)

Unit commits: 8284be8 + a6b424b (E0), cd59876 (E1), 9771b1d + 1284acd +
a3b3f8c (E2), ed0e408 + 783f9ff + 74d012a (E3).

1. What was ported (engine-cited in the file headers). The greedy generate
   entry (core_v41_api.c:426-470 + the round drive :175-425; head_last_only
   and the xlast GEMV head, core_v41_forward.c:334-342). The DSpark draft
   towers (core_v41_draft.c: main_x, the markov cache, the block forward;
   core_v41_attn.c:239-280: the tower attention and the window push). The
   spec round (core_v41_forward.c:241/267 snapshot/rollback + the win-ring
   snap kernel cuda_kv_ring.inc.cu:34-68 + the cpre snapshot BEFORE the
   destructive shift, core_v41_attn.c:76-85 + the confidence scheduler
   core_draft_sched.c:31-97 + the [dspark] trace core_v41_api.c:246-406).
   The feed's prepare protocol (the engine's full-block hist memcpy,
   core_v41_forward.c:377). The decode-step CUDA graph (core_decode_graph.c:
   one graph per batch size n, position buckets of 1024, the "device
   position" convention ds4_gpu_v41.h:116-124, the rollback snapshot inside
   the graph, closed-form host accounting, the capture-failure policy:
   request-local n1_off, ONE batch_off for every batch shape) with its
   primitives (cuda_decode_graph.inc.cu: ThreadLocal capture, the PDL edge
   rewrite, the zero-copy upload/readback), the three posd host entries
   (cuda_v41_2.inc.cu:248-260 sparse attn, cuda_kv_ring.inc.cu:46-56 win
   commit, :59-68 win ring snap), the compress-step-n kernel
   (cuda_v41_2.inc.cu:74-128, bit-identical to the pool it replaces), the
   draft graph (core_v41_draft.c:326-370) and the host-gap accounting
   (core_v41_api.c:401-411).

2. Named port adaptations, each at its site in the code. (a) The capture
   stream: the engine captures on cudaStreamPerThread (-default-stream
   per-thread); the port captures on ONE dedicated stream created with
   BLOCKING flags and routes ds4_current_stream() onto it via the existing
   ds4_capture_set_stream — the legacy-default-stream implicit sync then
   orders it with the eager stream 0 in BOTH directions, which is what makes
   the rollback (eager, stream 0) safe before the next graph launch; a
   nonblocking stream would not have that ordering.  (b) The engram provider
   is synchronous (the P4-4 design): dg_launch calls the feed's prepare
   before the capture/launch and the graph records only the zero-copy upload
   — no flag-wait/arm/serve; provider success, the declared pos0 and the
   captured pinned pointers are validated on EVERY launch (the captured
   upload bakes the addresses; a moved buffer invalidates and re-captures).
   (c) No hist (the provider owns history) and no dev_sample/spec_q (sampling
   is its own unit; the graph's tail is one argmax per row and the wait picks
   next[i] = slot[4i], core_v41_sample.c:24-37's argmax arm).  (d) The PDL
   edge rewrite is skipped below compute capability 9 (the engine only ever
   runs sm_121); on the Spark it rewrote 1600/1543 edges, 0 unsupported.
   (e) DS41_NO_GRAPH=1 is the harness's graph switch (the engine's
   ds4_engine_v41_set_graph(0)); the CLI flags come with the serving unit.

3. Gate E-1 — the 56-token generate gate (tests/ds41_generate_gate.sh;
   Spark /tmp/e3_gate.log, GATE_RC=0). Non-spec PASS: 56/56 ids, the row
   reference L01/L14 clean (prompt 8/8, emitted 56 checked, 0 mismatches),
   history equal to the golden, and the n=1 graph walked 55 steps on 1
   capture (1703 nodes, 1600 PDL edges; 23.68 t/s steady). Spec (VK=5) PASS:
   the [dspark] trace line-identical to the engine's (19 rounds), the 6-row
   verify-batch graph walked 18 rounds on 1 capture (1790 nodes, 1543
   edges). DS41_NO_GRAPH=1 control PASS with no graph. The required negative
   controls FAIL as designed: DS41_EMIT_OFFSET=3 refused by name ("engram
   feed pos0"), DS41_ROW_SHIFT=1 mismatched with the port quiet.

4. Gate E-2 — the long-run gate (tests/ds41_graph_gate.sh; engine goldens
   /tmp/long_{nospec,spec}.log, 1100 tokens from the 8-token prompt; the
   engine's own spec == non-spec byte-identity verified on them). Non-spec
   PASS: the n=1 graph captured [9,1023] then re-captured [1024,1117], 1099
   steps on 2 captures. Spec PASS: the batch graph captured [1022,1023] then
   [1028,1112] — the re-capture fires at the first row whose LAST row leaves
   the cap, the engine's pos0+n-1 > cap condition mirrored exactly — 232
   rounds, 9 draft-graph captures with 203 graph rounds, the
   scratch-generation invalidation path exercised ("backend scratch moved,
   all re-capturing"), [dspark] diff empty. Scheduler run (k unpinned) PASS:
   258 rounds, 11 verify captures across widths 4/5/6 rows, 10 draft captures
   with 221 graph rounds, 2 k=0 rounds walked on the n=1 graph, and the ids
   still equal the pure-decode golden — k only affects speed, never the
   output.

5. Recorded limits. The port's synchronous engram provider shows up in the
   host-gap line (~9-37 ms/round of preparation the engine overlaps with the
   graph): that is the P4-4 host-IO decision's measured cost, not a
   correctness gap. Sampling and the penalties, the sampling draft arm
   (dev_sample), the amp/distillation sidecars, and the Rust-host ABI for
   the draft parameters (P5: block_size / expert_used_count / noise_token_id
   / markov_rank / target_layers) remain their units. The engine's extra
   re-captures when a batch's last row straddles a bucket cap are mirrored
   (the [1022,1023] capture above is that case).

Unit: E complete

## 6.16 Unit F evidence: the serving surface (2026-10-10)

Unit commits: e1f62a7 (P5.1, the draft parameters on the host-shape ABI),
61e4e12 (P5.2, the run surface: native switches, the host rows provider,
the bridge entry, the CLI flags, the Rust feed), ef6cd09 (P5.3, the
serving contract row), 3b6ad8f (P5.4a, the [emit] instrument + the Rust
gate runner + the ds4.o dependency fix), f730e0a (P5.4b, the ctx/vocab
host-shape wiring), 25c970c (the P5.4 evidence close), 591ae82 (P5.5, the
push-based server route), c1a1f70 (the HTTP gate script + target), d1a375a
+ 2e3b13e + 7a7f34b (the three boot/first-request fixes), 7c2bd9a (the
gate reorder: the CLI reference runs before the server boot).

1. What P5.1-P5.3 ported (engine-cited in the file headers). The
   host-shape ABI carries the draft parameters (block_size, expert_used_
   count, noise_token_id, markov_rank, target_layers) and the native arms
   the drafter with the GGUF path's rules (core_validate_v41.c:91-110;
   block 0 = unarmed + the engine's warning, mtp_target_slot starts at -1).
   The native switches (ds4_engine_v41_set_dspark/set_graph/set_emit_trace/
   set_prof/set_progress) mirror the engine's CLI wiring, and the rows
   provider (ds4_v41_feed_open) owns the PINNED buffers the captured graphs
   read (ds4_gpu_host_alloc is not on the Rust ABI; pinned memory is
   device-adjacent, so the buffers stay native-owned). The bridge entry
   ds4_bridge_v41_generate mirrors the engine's single-request server path
   1:1 (server_generate_v41.c:412-414: set_progress + generate_argmax with
   the emit callback; the req object is only for the --batch lanes). The
   Rust route (crates/ds4-core/src/v41_run.rs) opens the feed (hash +
   shard preads), sets the switches, and drives the callbacks through
   catch_unwind tramps; the CLI runs it on --gen-ids with the engine's own
   flag spellings. The serving contract row declares serial/none banks,
   no reuse, no disk KV, no snapshots, embedded MTP (qualified) and
   metadata-only context; --max-seqs>1 and prefix reuse are refused by
   name (v41_serving_contract_refuses_what_it_lacks).

2. The [emit] instrument (P5.4a). The engine's --emit-trace prints
   `[emit] <absolute position> <token id>` once per emitted token
   (core_v41_api.c:246 at the loop top, :387 for a spec round's earned
   tokens), and the port's round drive now prints both sites verbatim,
   gated by g_ds4_v41_emit_trace. NAMED DEVIATION from the P5.4 plan
   text: the instrument lives in the native round drive, not in the CLI's
   emit closure — the closure sees only the token id, and a Rust-side
   position counter would re-derive what the live state knows and could
   never show a rollback position drift, which is the failure the
   instrument exists to catch (core_v41_api.c:244-245).

3. The gate found a real wiring gap (P5.4b, f730e0a). The Rust route's
   first live run refused at the generate entry: "ds4: [v41] model
   metadata has no deepseek4.context_length". The C loader path fills
   g_ds4_v41.ctx from the GGUF (ds4.c:7226-7228) but the host-shape ABI
   carried no ctx, so model_apply_host_v41_wiring left it zero and both
   the generate entry (ds4_ds41_draft.inc:650) and the state alloc
   (ds4_ds41_forward.inc:280) refused. The fix appends v41_ctx (and the
   engram vocab pair, which the C path fills as required keys,
   core_validate_v41.c:152-153, and the Rust path left zero — inert today,
   same divergence class) to the host shape; V41Wire reads the three keys
   with the engine's hard stops (missing ctx = MissingKey, 0/over-u32::MAX
   = CtxInvalid). Artifact values read first-hand from the GGUF on
   2026-10-10: context_length 1048576, engram vocab_size 16000000,
   compressed_vocab_size 99092. The Makefile's ds4.o rule was also
   missing ds4_ds41_draft.inc among its dependencies (ds4.c includes it
   at :78847), so an edit to that file alone did not trigger a rebuild —
   make ds4 would have linked a stale loop; added.

4. Gate F-1 — the C generate gate re-run after the P5.2/P5.4a native
   changes (tests/ds41_generate_gate.sh, the same pin as E: p1.ids, N=56,
   ROWS_REF p1g56.logits.bin, ENGLOG_SPEC /tmp/gen56_spec_a.log, VK=5;
   Spark /tmp/p54_cgate.log, CGATE_RC=0): positive PASS (non-spec + spec,
   row reference and history clean, n=1 and batch graphs walked with
   reuse, no-graph control golden), negative controls FAIL as required.
   The port's own [emit] lines ride in the harness logs now (56 lines in
   both the non-spec and spec runs, first line `[emit] 8 455` == the
   engine golden's first line).

5. Gate F-2 — the Rust serving gate (tests/ds41_serving_gate.sh; Spark
   /tmp/ds41_serving_gate.{default,nograph,nodspark,spec,noids,temp}.log,
   RGATE2_RC=0): "positive PASS (default + no-graph + no-dspark + spec,
   [emit] ids and positions equal to the engine golden, [dspark] trace
   identical, n=1 and batch graphs walked with reuse), negative controls
   FAIL as required". Per mode: default (drafter armed, graph on) — 56/56
   ids equal to the engine's [emit] golden, positions sequential from the
   prompt length, 4 verify-batch captures (6/2/1/3-row) reused over 20
   walked rounds, DSpark 25 rounds (scheduler skipped 3), average accept
   1.04/5; --no-graph — the same ids with no [graph] line (direct ==
   graph == golden); --no-dspark — the same ids, the n=1 graph walked 55
   pure-decode steps on 1 capture (43.85 ms/step steady = 22.80 t/s),
   no DSpark summary; --dspark-verify 5 — the same ids, the [dspark]
   round lines identical to the engine's spec golden (19 rounds, average
   accept 1.95/5, one 6-row capture reused over 18 walked rounds), and
   the port's own spec ids == its non-spec ids (the engine's byte-identity
   rule). Negatives refused by name: a text prompt without --gen-ids
   ("the V4.1 family runs on --gen-ids <file>") and --temp 0.5 ("V4.1
   sampling is not ported").

6. Recorded limits. The Rust gate runs one process per mode (six model
   loads, ~21 min on the Spark) — the CLI has no batch-mode harness; the
   C gate has the same shape. Sampling, the V4.1 chat rendering (DSML/
   thinking), the lanes scheduler and the sidecars remain their units;
   the caps row refuses them by name.

7. What P5.5 ported (engine-cited in the file headers). The engine's V4.1
   server answer is the one-shot generate with two callbacks
   (server_generate_v41.c:412-414: set_progress + generate_argmax with
   v41_emit; the req object belongs to the --batch lanes scheduler and
   --multi-probe, NOT this path). The port mirrors it: DecodeIo gained
   is_v41/v41_generate/v41_last_spec_stats (defaults refuse by name),
   NativeDecode implements them through V41ServeRoute (model path +
   V41RunOptions), the dispatch is one variant check at the top of
   generate_terminal_prepared (the engine's own one-line check,
   server_generate.c:19), and generate_terminal_v41 reuses the port's
   stream/stop/tool machinery (v41_gen/v41_emit/v41_finish). The finish
   drops exactly the session-addressed pieces the engine drops there too:
   no invalid-DSML retry, no tool replay / live bindings / checkpoints,
   and no tool-memory write (the port's tool memory is wired to a session
   store this route never creates); an unterminated tool call is an
   error, never a repair. The DSML output parse was already live
   (syntax_for_model_id(16) -> DeepSeek). Server flags (cli_opts.c:258-395
   spellings): --engram-dir, --v41-no-engram, --no-dspark, --dspark
   (bare; the server has no V4-era collision), --dspark-verify, --no-graph,
   --emit-trace, --v41-prof; --zchain is refused by name (the sidecar
   mount is not ported) and an explicit --ctx on a V4.1 model is refused
   by name (core_validate_v41.c:51-53). Six route tests cover the surface
   (completion, stream deltas, budget, ctx clamp, stop string, empty
   prompt) against a scripted v41 engine.

8. The gate found three boot/first-request crashes, all fixed and pushed.
   (a) d1a375a — boot quote_overflow: the host quote priced a V4 session
   bank at the 1M metadata ctx for the v41 row (serving.rs); the v41 row
   is the only one with reuse None and now prices no persistent bank.
   (b) 2e3b13e — boot SIGSEGV (exit 139): the port-only boot prewarm
   crashed on v41 (the engine has NO prewarm); the DS4_NO_BOOT_PREWARM=1
   control proved it (139 without the skip, boots with); prewarm is now
   skipped for the v41 family.  (c) 7a7f34b — first-request SIGSEGV:
   serial admission asked the native for a session-graph fit quote on a
   family that never has a session (gdb: run_serial ->
   NativeDecode::native_graph_fit -> Model::session_graph_fit_quote ->
   ds4_bridge_session_graph_fit_quote -> ds4_engine_session_graph_fit_quote
   -> metal_graph_session_fit_check -> metal_graph_alloc_bytes_estimate);
   serial_session_probe and native_graph_fit now answer None for v41 —
   the engine creates a session only for non-v41 (server_main.c:65) and
   dispatches v41 before any session machinery (server_generate.c:19).
   The class is "port machinery a session-less family must not enter";
   each fix carries its own reasoning in the commit.

9. Gate F-3 — the HTTP serving gate (tests/ds41_serving_http_gate.sh +
   `make test-ds41-serving-http`; Spark /tmp/httpgate_c.log,
   HTTPGATE_C_RC=0 at 14:49:38). PASS line: "CLI reference, boot +
   /v1/models, non-stream and stream completions, the served ids equal
   the CLI's on the same prompt, --ctx and --zchain refused by name".
   Evidence: the served [emit] ids == the CLI's 3/3 (`5 11111`, `6 16`,
   `7 1`; prompt 5 tokens, positions sequential from the prompt length);
   non-stream " Paris." finish stop, prompt_tokens 5 / completion_tokens
   2, ttft 162.8 ms, 8.7 tok/s decode; stream 3 JSON chunks + [DONE];
   both boot refusals fire by name. The gate runs the CLI reference
   FIRST — the single-instance guard admits one model-loading process at
   a time, and the first attempt (14:38, /tmp/httpgate_b.log) failed
   exactly there; fixed in 7c2bd9a. The full gate is 2 CLI loads + 1
   server boot, ~6 min on the Spark.

10. Recorded limits (P5.5). /v1/completions is the gateable surface;
    chat + tools stay refused until P6 (the prompt side admits model_id 0
    or completions only, chat_input.rs:20; the artifact carries no chat
    template — all 73 metadata keys scanned — so the engine's C renderer
    is the source). Sampling stays refused by name until unit G. The
    lanes scheduler (server_sched_v41.c + core_v41_req.c), the amp/
    distillation sidecars, --v41-chunk / --dspark-block and the sampling
    draft arm remain their units.

Unit: F complete

## 6.17 Unit G evidence: the V4.1 chat rendering (2026-10-10)

Unit commits: 350d34f (the admission + the V4.1 head + the tail fix + the
[ptok] instrument + the golden tests), 9cd321a (the special-token scan
entry), 9a84a4d (the gate's id-list comparison), b236968 (the gate
completes the tool call), 53a091c (the gate compares the answer as the
engine returns it).

1. What unit G ported (engine-cited in the file headers). Admission:
   chat_input.rs admits Variant::DeepSeek41Flash (16) beside Flash (0) --
   the engine renders every chat request through its C renderer regardless
   of model (server_parse_chat.c:205), and the artifact carries no chat
   template (all 73 metadata keys scanned). The head:
   render_dsml_chat_choice writes BOS, then -- if the effort prefix or the
   system body is non-empty -- the <｜System｜> token
   (server_dsml_render.c:381), then the effort prefix, then the system
   body; the token text comes from the tokenizer that owns it
   (core_bpe.c:332-333; core_engine_api.c:94-96), so a V4 tokenizer falls
   back to plain text. The effort prefixes are the official encoding.py
   texts (core_globals.c:14-17: HIGH "Reasoning Effort: 75 ...", MAX
   "... 100 ..."), not the port's pre-2026-09-21 V4-era texts; the head is
   selected explicitly (V4 vs V41).

2. The live-tool-tail fix. The Rust DSML tail arm delegated to the full
   renderer and stripped only BOS, so High/Max think modes leaked the
   effort prefix into the suffix; both sources write EOS + the messages
   only, no head (the port's own C ds4_server.c:4700-4748; the engine
   server_live_prep.c:21-69). It is now a standalone port.

3. The [ptok] instrument. The engine's --emit-trace prints the rendered
   prompt's ids as `[ptok] <pos> <id>` (core_v41_api.c:171-173); the
   port's generate entry prints the same, under the same switch -- the
   gate's measuring stick, because the engine server --trace dumps the
   rendered prompt text plus both id lists (server_trace.c:222-230).

4. The gate found a tokenizer gap. The port's rendered-chat special scan
   lacked the <｜System｜> entry (the engine's table has it, core_bpe.c:360),
   so the head text split into three plain pieces (28217/27824/28217)
   instead of the single id 128799; 9cd321a added the entry.

5. Gate G-1 (local). The engine's official encoding.py golden
   (tests/server_tests_render_v41.c:36: the four cases plus the
   V4-tokenizer fallback) ported 1:1 into tests/render.rs
   (dsml_v41_head_matches_official_encoding), the tail pin
   (dsml_live_tool_tail_has_no_head), and the admission test with a
   scripted engine; render 33/33 and generate 72/72 green this session.

6. Gate G-2 (Spark; /tmp/chat_gate.log, CHAT_GATE_RC=0 at 16:17:23;
   `make test-ds41-chat`). One chat+tools request (temp 0,
   reasoning_effort none), engine first (--trace), then the port
   (--emit-trace). PASS line: "prompt ids, generated ids and the response
   message equal the engine's on the same chat+tools request". Evidence:
   prompt ids 354/354 byte-identical (head `0 128799 372 ...` -- BOS,
   <｜System｜>, then the system body and the tool-schema block;
   prompt.diff empty); generated ids 41/41 identical (gen.diff empty);
   the response message identical (msg.diff empty; tool-call ids dropped
   -- random per process, server_msgs.c random_tool_id); finish_reason
   equal on both (stop).

7. The artifact's tool-call style, recorded. At greedy on this request the
   model emitted the DSML block in a mixed tag style: the <｜DSML｜tool_calls>
   and <｜DSML｜invoke opens are decorated but the parameter tags and the
   closing tags are plain. The engine's parser picks one style from the
   open tag (server_dsml_parse.c:338; three accepted styles) and the mix
   matches none, so the engine returns the call as assistant text with
   finish=stop -- no tool_calls array, no repair -- and the port
   reproduced that byte for byte. Tool calling on this path is exactly as
   capable as the engine's; the mixed style is the artifact's behavior at
   this quant, and it is the datum to watch for opencode/Claude Code tool
   use.

8. The gate's speed reading, and a named port gap (recorded, not yet
   fixed). Same request: port prefill 354 tokens 4.5 s (78.0 t/s; ttft
   4.65 s) against the engine's 1.4 s (254.3 t/s); port decode 38.4
   ms/token (25.7 t/s; DSpark 16 rounds, accept 1.50/5) against the
   engine's 23.9 ms/token (14 rounds, 1.93/5). A --v41-prof run localized
   the prefill difference: the forward body is at parity (1224.7 ms
   total, layers ~1150, head 3.0 ms) and the ~3.1 s sits in the Rust
   engram feed's prepare (crates/ds4-core/src/v41_run.rs:135: one
   positioned read per row per layer), which the engine does with an
   io_uring batch (its boot line says "取行走 io_uring";
   core_v41_engram.c). The decode round's engram prep (24.668 ms, in the
   port's own host-gaps line) sits inside the verify window (87.4 vs 60.5
   ms) and inflates the scheduler's cost model. Porting the batched
   io_uring row reads is the next performance unit.

9. Recorded limits. Sampling (temperature/top_p/min_p/penalties and the
   engine's request-omitted defaults, server_generate_v41.c:355-366) is
   unit H; the v41 route serves greedy today and ignores the request's
   temperature (the CLI refuses --temp != 0). The Anthropic and Responses
   surfaces have the machinery (http.rs:285-288; generate_terminal_v41)
   but no live gate yet; the thinking/effort aliases and the effort-name
   mapping divergence are recorded for unit H.

Unit: G complete

## 6.18 Unit Z plan: the install.sh startup contract (2026-10-10)

The engine's own deployment recipe is install.sh (fetched from
huggingface.co/wenzhouwu/YoungAi-DeepSeek-V4.1-Flash; on the Spark at
~/youngai/model/install.sh). Its `start` boots the server with exactly
this line (install.sh, the `nohup "$MODEL/bin/ds4-server" ...` call):

    ds4-server --cuda -m <gguf> --zchain <sidecar_dir> --engram-dir <dir> \
               [--posttrain <dir>] --mem-budget-mb 110000 --host H --port P

with `--zchain` = the domain sidecar picked by `sidecar_of`
(install.sh:41-48; finance default -> `…-grrb-vqfin41_vqhalf_a_n8192-engine`)
and `--posttrain` = `posttrain-experimental-20260924` (finance only,
opt-in). Every item of that line must work on the port, flag for flag: a
served contract that only works after dropping the deployment's own flags
is not the engine's contract, and a gate that boots the engine without
them measures a configuration nobody serves. Status, item by item:

| item | engine | port status |
|---|---|---|
| `--cuda` | server_config.c | ported (`Backend::Cuda`) |
| `-m` / `--model` | | ported |
| `--zchain DIR` | server_config.c:314-317 -> core_engine_open.c:120-124 (`ds4_engine_v41_set_amp_dir`, core_v41_api.c:75) -> per-state `v41_state_plugins` (core_v41_state.c:77-87) -> `v41_amp_load` (core_v41_amp.c:227-270): rb with a full unload first (:113-120), the gr ②×③ product (:130-152), the amp concat (:234-256). Mount failure = request failure, never a silent bare run (:84-86) | NOT ported: the server refuses it by name, the CLI has no flag. The P4-7 stores and the P2/P3 Rust reader/merger (§6.6.3-4) exist; the mount path is missing |
| `--engram-dir DIR` | | ported (P5.1-P5.5) |
| `--posttrain DIR` | server_config.c:318 (`set_posttrain_dir` at parse); ③ is the `base.fnv`-gated third file | NOT ported (unknown argument) |
| `--mem-budget-mb N` | server_config.c:339-340 -> `ds4_set_mem_budget_mb` (core_util.c:198-200): in-process red line; watchdog aborts at 90 % (DS4_WATCHDOG_TRIP_FRAC, :9; sampler :156-189; `_exit(137)`), L1 load gate at 85 % (DS4_L1_GATE_FRAC, :10; :236-256), the expert AUTO verdict reads the budget (core_engine_open.c:319) | NOT ported (the port's host-memory-guard is a different, external policy) |
| `--host` / `--port` | | ported |
| smoke request: `model: "deepseek-chat"`, temp 0 | server_msgs.c:300-301 (alias -> thinking off) | the alias logic exists; the smoke path is not yet gated on the v41 route |
| install.sh shell guardrails: START_FLOOR_MB 100000 before load, RUN_FLOOR_MB 10000 after, KILL_MB 2500 watchdog | install.sh itself | the port deploys under the host-memory-guard; map the floors or record the deviation |

Work, in order:

1. The `--zchain` mount: bridge entries over the P4-7 stores (gr set /
   rb set / rb clear), the Rust mount call (`V41Zchain::load` -> push per
   layer, per run, mirroring the engine's per-state mount at every
   generate), the server and CLI flags. A dir carrying `amp_Lnn.bin` is
   refused by name (the amp apply is not ported; no Spark dir carries
   one - the recorded limit of §6.14 stands until one exists). The port
   prints its own mount counts for the gate to compare with the engine's
   `[反修]` boot lines.
2. `--posttrain` through the same path (the Rust module already merges
   and gates it, §6.6.4).
3. `--mem-budget-mb`: arm the engine's two guardrails at the engine's own
   fractions (90 % watchdog abort, 85 % L1 load gate) on the port's
   process.
4. The smoke: the install.sh request served by both sides with the
   finance sidecar mounted, ids equal.

Gate Z-1: the unit H gates (sampling A-D + the wire gate) re-run with BOTH
sides booted exactly as install.sh boots (`--zchain` finance +
`--engram-dir`); the served ids must stay byte-identical with the sidecar
mounted, and the port's gr/rb mount counts must equal the engine's `[反修]`
lines. Gate Z-2: the install.sh smoke request (`deepseek-chat` alias)
served by the port, ids equal to the engine's under the same boot. Unit
Z's readings land in this section when it closes; unit H's evidence
follows as §6.19, measured under the same canonical boot.

### Evidence (2026-10-10 evening, Spark; oracle = the source-built engine)

Oracle (owner rule, §6.4): the engine built from the pinned source
(github.com/JLFN/YoungAi @ 3946dbc, `make cuda-generic` on the Spark) —
`~/youngai-src/ds4-server`, sha256
`ace08941e4132e834b631c101b09e21a41f46eadb1be95d5c8b5b6fbe2de6826`; the
prebuilt HuggingFace binaries are not an oracle. Both sides booted exactly
as install.sh boots, modulo the gates' own trace instruments: `--cuda -m
<gguf> --zchain <finance sidecar> --engram-dir <dir> --mem-budget-mb
110000 --host 127.0.0.1 --port 8099`.

Golden re-check (same evening): `tests/capture_ds41_golden.sh` re-run with
the source-built CLI (`BIN=~/youngai-src/ds4`, `OUT=/tmp/golden_src`)
reproduces the recorded golden set bit-for-bit — both variants, all five
prompts: the `logits.bin` files are byte-identical to
`~/youngai/model/golden`'s (cmp SAME on all ten) and the PPLs match
(with `--zchain`: 112.0845 / 24.9582 / 7.2809 / 14.3524 / 876.7182; bare:
46.8780 / 22.0679 / 9.5414 / 10.7011 / 6662.9080). The prebuilt-captured
golden therefore stands under the source-built oracle; no re-capture
needed.

Gate Z-1 (both gates; the id readings are §6.19): PASS 19:21-19:38 —
`tests/ds41_sampling_gate.sh` (cases A-D byte-identical) and
`tests/ds41_wire_gate.sh` (anthropic + responses + smoke identical) both
green under that boot. Mount parity: the port's `[v41] zchain mount: gr=39
rb=27 amp=0` equals the engine's `[反修·权重侧] 39 层挂上逐专家增益覆盖
(盘上 40.7 MB)` + `[反修·路由] 27 层挂上路由偏置侧车 (盘上 0.041 MB)`,
and the port's L1 line `ds4: L1 budget gate: planned resident 4.44 GiB
within 107.42 GiB budget` carries the engine's own figure.

Gate Z-2, the deploy demo (the port's own install.sh at the repo root,
19:38:45-19:42:41): preflight `machine ok: NVIDIA GB10, CUDA 13 runtime,
124546 MB memory`; `artifact files ok`; `make ds4-server` up to date; boot
19:38:45 -> loaded 19:41:55 (`11293 MB of memory left`); the smoke
answered: "A price-to-earnings ratio is a valuation metric that compares a
company's current stock price to its earnings per share, indicating how
much investors are willing to pay for each dollar of earnings."; `ready:
http://127.0.0.1:8000/v1, domain finance`; `bash install.sh stop` ->
`stopped`; status -> `not running`, 121834 MB available. The server loaded
only `$REPO/ds4-server` — the reference binary is never the port's runtime
(owner rule).

Recorded limits: a dir carrying `amp_Lnn.bin` is refused by name (no Spark
dir carries one; the §6.14 limit stands); the engine's expert-offload AUTO
verdict has no port counterpart (the port's residency is its own memgov
plan; the engine's own VQ-blob rule forces offload for this artifact); the
Linux footprint watchdog is inert by the engine's own design
(core_util.c:141-145), so the port carries the L1 load gate — the active
half on Linux.

Unit: Z complete

## 6.19 Unit H evidence: sampling and the wire surfaces under the canonical boot (2026-10-10)

Unit H's implementation landed earlier (`082a90e` + `da61df2`, §6.17's
era); these are its gates re-run with the source-built oracle and both
sides on the install.sh boot (the runs §6.18 records).

The sampling gate (tests/ds41_sampling_gate.sh, 19:21-19:32): PASS. Cases
A (temp 0.7 + seed 12345), B (the omitted-temperature default) and C (temp
0.7 + frequency_penalty 0.5) on the `--no-dspark` boot, and D (sampling
under speculation) on the default boot: served ids byte-identical to the
engine's on all four (8/18/13/8 generated), finish stop on all four. The
port samples through `cuda/ds41_sample.cuh` with the engine's own coins
and rejection decisions (case D also exercises the draft draws under
speculation); the earlier prebuilt-oracle pass (19:0x) is superseded by
this run and its ids are unchanged.

The wire gate (tests/ds41_wire_gate.sh, 19:33-19:38): PASS. `/v1/messages`,
`/v1/responses` and the install.sh smoke request (`deepseek-chat` alias,
temp 0, max_tokens 48, the finance question) served by both sides under
the same boot: prompt ids, generated ids (8/8/39) and finish reasons
identical.

Unit: H complete

## 6.20 Unit P: the engram io_uring read path (2026-10-10)

Spec (6.17.8, unit G's speed reading): the port's `V41Feed::prepare` read
one row per position per layer with two serial positioned reads each,
~3.1 s on the G-2 gate's 354-token prompt; mirror the engine's io_uring
channel so a round costs about one disk latency.

Landed (`b7f5ad8`): `crates/ds4-core/src/uring.rs` mirrors
`core_v41_ering.c` 1:1 with raw syscalls (no liburing, like the engine):
`io_uring_setup` + SQ/CQ/SQE mmaps, one 2-block bounce slot per entry with
the free-slot stack (`:82-88`), the push loop (`:89-112`), the GETEVENTS
drain that keeps the queue full (`:122-144`), and the O_DIRECT
aligned-superset read (`:96-108`). `v41_run.rs` keeps the engine's round
shape per engram layer (`core_v41_engram.c:238-254`): hash, one request
per plane per row, one submit, one wait, with the row-range refusal of
err=1 (`:247`). The serial loop stays as the non-Linux fallback (named
limit; the engine falls back to its thread pool there).

Evidence (2026-10-10 ~20:10, Spark, source-built oracle
`~/youngai-src/ds4-server` sha256 `ace08941...`, both sides on the
canonical boot incl. the finance `--zchain` and `--mem-budget-mb 110000`):

- `tests/ds41_chat_gate.sh`: PASS - prompt ids 354/354, generated ids and
  the response message equal the engine's on the same chat+tools request.
- Port prefill 354 tokens in 1.3 s (273.0 t/s) against the engine's
  1.247 s - within 4.3% (was 4.5 s, 78.0 t/s; 6.17.8).
- The port's host-gaps line: engram prep per decode round 2.608 ms (was
  24.668 ms); verify->draft 8.88 ms, draft->verify 5.46 ms.
- The engine's decode on the same request: 43.56 t/s.
- Local: the ring unit test (100 unaligned reads, O_DIRECT and plain legs,
  more requests than the queue depth, byte equality against plain reads)
  plus the ds4-core lib suite 369/369.

Recorded limits: the rounds are block-synchronous (every layer read before
the block's GPU work starts) - the engine's own overlap structure is unit
P2, 6.21; non-Linux keeps the serial path.

Unit: P complete

## 6.21 Unit P2 and the qwen38-derived candidates (2026-10-10)

Owner question answered in `docs/ds41-qwen38-serve-lessons.md`: study the
qwen38 route (already served by this repo) and take what makes ds41 fast
the same way. Three candidates, in order; every source citation and every
test lives in that document.

P2 - the engram overlap (engine mirror, no doctrinal change). The engine
prefetches at forward start and waits per layer, submitting the next
round as each completes (`core_v41_forward.c:390`;
`core_v41_engram.c:311-341`, `:344-379`; the fetch thread for rounds
larger than the queue, `:119-150`); the port reads all layers
synchronously (6.20). The qwen38 PLE's ready-row lease is the same
principle: never let the GPU wait on the SSD. Change: an optional
per-layer wait on `ds41_engram_feed` (`ds41_forward.h:41-53`), `prepare`
submits round 0 only, the native waits per layer at the point of use
(`ds4_ds41_forward.inc:440-466`), and big rounds go to a fetch thread.
Gate P2-1: the four parity gates below + prefill not worse than 1.3 s at
354 tokens + the 835-token long prompt (`tests/ds41-long-prompt.txt`)
measured + the host-gaps line showing the engram prep out of the critical
path. Fallback: the synchronous path when no ring (current behavior).

Q-1 - the MTP compact draft vocabulary (named deviation, owner-gated).
The qwen38 route scores a shortlist in the draft only (`ds4.c:26778-26845`
and `:27368-27373`; +3.04%/+3.87% median decode with exact ids,
`docs/qwen38-perf-2026-09-14.md`). For ds41 the draft head is
full-vocabulary (`ds4_ds41_draft.inc:46,155,165`). Gate Q-1: emitted ids
byte-identical on the sampling, wire and chat gates; the generate gate's
spec leg gates on emitted ids with the `[dspark]` draft-id diff marked
informational; a decode A/B >= 2% median (fresh processes, alternating
pairs, clock medians). Kill switch `DS4_V41_MTP_FULL_VOCAB=1`.

Q-2 - the MTP prefix-cache preparation (named deviation, owner-gated).
The qwen38 route skips the draft prefix's discarded computation
(`ds4.c:27536`; +0.90%/+2.55% median prefill with exact ids). The ds41
counterpart is the draft arm/prefix path (`ds4_ds41_draft.inc:125-198`).
Gate Q-2: same emitted-id gates; a prefill A/B (354- and 835-token
prompts) >= 2% median. Kill switch `DS4_V41_MTP_FULL_PREFIX=1`.

Parity gates for P2, Q-1 and Q-2 (Spark, source-built oracle, canonical
boot): `tests/ds41_sampling_gate.sh`, `tests/ds41_wire_gate.sh`,
`tests/ds41_chat_gate.sh`, and `tests/ds41_generate_gate.sh` with
`DS41_FEED_DIGEST`.

## 6.22 Unit P2 close: the engram overlap measured (2026-10-11)

Commits: `c276943` (the split feed protocol), `bbf8bbe` + `e4f1be4` +
`fb60666` (the long gate and its fixes). The window readings below ran
against the source-built oracle (`~/youngai-src/ds4-server`) under the
canonical boot, with the port binary built from `fb60666`.

1. The 354-token A/B (chat gate, alternating legs, four runs):
   decode P2 23.2 / 23.5 ms/token vs sync 23.8 / 23.7 (P2 ahead on
   both pairs; medians 23.35 vs 23.75); prefill 1.3-1.4 s both
   protocols; per-round engram prep 2.465 -> 0.232 ms in the A legs.
2. The 835-token measurement (`tests/ds41-long-gate.sh`, prompt 839
   tokens incl. the head): PASS. Prefill P2 2.1 s (342.8 t/s) vs sync
   2.5 s (296.7 t/s); decode P2 32.4 t/s vs sync 31.5 t/s; engram
   prep/round 0.100 vs 1.468 ms; prompt and generated ids equal to the
   engine's on both protocols; the withhold control returns HTTP 500 and
   the native refuses the engram layer as required. Logs:
   `/tmp/win_long.log` (+ `/tmp/ds41_long_gate/`).
3. The P2 close window (2026-10-11, operator's server stopped): chat
   gate PASS (the mixed-style call parses into tool_calls), sampling gate
   PASS (cases A-D byte-identical), wire gate PASS (anthropic +
   responses + the install.sh smoke ids and finish reasons equal).
4. Adoption: P2 is an engine mirror, no kill switch; the measured gain
   (decode +2.9 % median at 839 tokens; engram prep out of the critical
   path) meets the section 6.21 bar.

## 6.23 The V4.1 tool path: mixed-style DSML served natively (2026-10-11)

Commits: `21956c6` (the tolerant scan), `bc18803` (the SSE finish
reconcile), `ad9a4ab` (the finish mapping), `6f4ceaa` (the wrapper-less
hold-back), `9ade4fa` (the spaced render), the render byte fix + goldens
(2026-10-11, uncommitted at this writing).

0. Root cause, found upstream: the official V4.1 `encoding.py` spells the
   DSML tags with the long prefix plus a space (`<｜DSML｜ calls>`,
   `<｜DSML｜ invoke`, `<｜DSML｜ parameter`; the LONG `DSML` between two U+FF5C
   bytes) and the V4.1 server had been rendering the V4 spelling in the
   tool template, so the model was prompted with a form it never trained on
   and drifted mid-block (the upstream fix `4294136`, issue #1). The
   spaced port supersedes the old template for the V4.1 route.

1. What failed, and why. The artifact emits the DSML tool block in a MIXED
   spelling: the wrapper and invoke opens carry the fullwidth decoration
   (U+FF5C, bytes EF BD 9C) while the parameter tags inside the same block
   are plain (measured 2026-10-10). The engine's parser keys on one style
   from the open tag (`server_dsml_parse.c:338`); the mix matches none, so
   the engine returns the call as assistant text with finish=stop -- the
   exact failure the tool shim exists for (`~/youngai/tool-shim/shim.py`)
   and the reason an open-grok session on this model saw no tools at all
   (owner report 2026-10-10). open-grok itself has no DSML path: the
   sampler parses SSE tool_calls deltas only (the first delta carries
   id+name, later deltas append arguments; `/data/open-grok/crates/codegen/
   xai-grok-sampler`), so a clean tool call must come from the server. A
   second failure surfaced the same night: the route's
   unterminated-tool-call state is finish `error` (engine
   `server_generate_v41.c:230-234`), and open-grok rejects unknown finish
   variants outright ("serialization error: unknown variant `error`",
   session `01a127a0` 23:17:34, updates.jsonl).

2. The fixes (all scoped to the V4.1 route; other families untouched):
   - `21956c6`: the tolerant fallback scan
     (`crates/ds4-server/src/tools.rs:489`, `parse_dsml_tolerant`), run
     only when the strict pass found no calls. It mirrors the shim's rules
     (`shim.py:44-190`): normalize the decorated spellings (`:440`), scan
     the three starts `["<tool_calls", "<calls>", "<invoke"]` (`:492-494`),
     tolerate a missing closer (next invoke, wrapper close, EOF), accept a
     bare `<parameter name=...>`, skip undeclared names, and coerce each
     value to the declared schema type (`:595`). A value running to EOF
     without a boundary is refused, so a truncated stream is never served
     half an argument.
   - `bc18803`: the SSE finish reconcile (`stream.rs` +
     `tool_stream.rs`): the live stream had emitted the invoke name and
     then failed on the plain parameter tag, so the finish's `emitted_any`
     guard dropped the parsed calls and a streaming client held a
     name-only call with finish_reason=tool_calls.
     `partial_invoke_index()` gates an arguments-only delta for the
     started index; later calls emit in full. Well-formed live streams
     keep their exact bytes.
   - `ad9a4ab`: the route's internal `error` finish is carried as `stop`
     on all three client surfaces; the engine keeps its internal
     distinction in its logs. The OpenAI finish_reason set has no `error`.
   - `6f4ceaa`: the live start detector's DeepSeek candidates gained the
     invoke spellings and the short `<...calls>` wrappers the shim
     records (`shim.py:44-48`). Without them the artifact's wrapper-less
     emission streamed as visible content while the finish still served
     the calls (both at once; observed in session `01a127cf`).

3. Tests and verification. Local: `--test tools` 22/22 (the captured mixed
   block, the shim's variants, coercion, undeclared names, nested
   parameters, the truncation refusal), `--test stream` 10/10 (the
   bare-invoke block with the split-marker hold-back and the finish
   reconcile), ds4-server lib green. Spark, on the deployed binary
   (`install.sh start --domain code`): non-stream finish=tool_calls with
   `get_weather {"city":"Stockholm"}`; stream: content deltas + one
   complete tool_calls delta + finish=tool_calls. The chat gate now
   encodes the parse (`bbf8bbe`): it checks the port's parsed call against
   the engine's text block, because the port deliberately diverges from
   the engine here -- this section is the named deviation. The
   Anthropic/Responses surfaces share the machinery but were not re-gated
   with tools.

4. Open, recorded: a tool block emitted INSIDE a thinking section still
   streams as reasoning_content (the reasoning arm has no tool detection,
   mirroring the engine) -- watch it in real sessions. Thinking itself is
   config-side: the open-grok ds4 model block gained
   `supports_reasoning_effort = true` + `reasoning_effort = "high"`;
   verified live (none -> no thinking per the engine default, high ->
   reasoning_content).

5. The render side verified against the official goldens (2026-10-11).
   `crates/ds4-server/tests/render.rs` gained the engine's
   `test_render_matches_official_spaced_dsml_tools` as
   `dsml_v41_spaced_tools_match_official_encoding` (case1 = the official
   repo's `encoding/tests/test_input_1.json`, byte-identical to its
   `test_output_1.txt`; case2 adds parallel calls, a zero-parameter call,
   json.dumps-shaped values, `<` and `&` inside a value, two adjacent tool
   results merged into one <｜User｜>, and a schema with a \u escape).  Writing
   the test exposed two byte defects in `9ade4fa`: the template's tag
   literals had lost the `DSML` text between the U+FF5C pair (a display
   artifact: `｜DSML｜` renders as a bare `｜`), and one think-tag spelling
   had lost its angle brackets; both are corrected, and the spaced render
   now matches the official goldens byte for byte (render 34/34).

6. The sampling side ported (P3, the engine's c6e2706).  The kernel's
   slot [3] now carries the raw argmax (`cuda/ds41_sample.cuh`, replacing
   the reserved-set count), `v41_sample_pick` asks the face's greedy hook
   per position (`ds4_ds41_sample.inc`: the caller's text + the accepted
   prefix `batch[1..i]`; the asking stops once the draft is not accepted),
   the single-row penalty route asks with temperature 0 (v41_next_token),
   and the face (`ds4_decode_sampling` + the bridge mirror) carries
   `greedy_fn`/`greedy_ud` down to the Rust accumulator
   (`SemAccum::syntax_greedy` rebuilds the tracker copy and the thinking
   state over the emitted text + the ahead tokens; `dsml_decode_state_is_syntax`
   is the verdict).  The engine's vector
   `test_syntax_greedy_spaced_dsml` is ported as
   `syntax_greedy_spaced_dsml` in `tests/dsml.rs` (dsml 3/3).  The Spark
   gates (the second oracle at 221b9e2, the chat/sampling re-run and the
   20-40-call tool-call test) are the next step.

## 6.24 Post-train port plan: the ③ producer, the JSONL upload, the apply (2026-10-11)

The engine's post-train subsystem was read end to end at 221b9e2.  Pin
drift first: `/data/YoungAi` at 3946dbc carries the OLDER `core_ptrain*`
bodies (the 10-01 list/`.qa` format, last-layer-only defaults) and no
studio; `/data/YoungAi-restructure` (221b9e2) is the "one JSONL, one
command" tree -- cite the restructure for the trainer.
`ds4_posttrain.c/.h` are byte-identical between the two trees.  The full
research note (call tables, HTTP routes, formats, the 14-case corpus) is
`misc/posttrain-research-2026-10-11.md`.

What it is: one JSONL + one command.  `./ds4 --ptrain <cfg>`
(`core_ptrain.c:178-348`) trains per-layer low-rank amps (`amp_Lnn.bin`,
the ③) on top of the deployed ①② model against a top-K teacher
(`core_ptrain_teacher.c`), Adam on `[layer_lo, layer_hi]` (validation
requires `layer_hi == DS4_N_LAYER-1`), and writes f32 `{A[K][D],
B[K][D]}` plus a `base.fnv` fingerprint of the ② directory (`pt_save_amp`,
`core_ptrain_bwd.c:173-200`).  The studio (`ds4-train` + `web/studio.html`,
`src/train/*`) wraps it: upload JSONL/documents, run the job (stop the
model child, train, wt2 five-metric gate, restart), pick a ckpt, hot-swap
②/③ on the live server.  At inference ③ mounts via `--posttrain`:
`v41_amp_load` concatenates ② then ③ rows (cap 8192) and the forward tail
applies `y += x·(B·A)` + bf16 round (`core_v41_amp.c:322-329`,
`core_v41_forward.c:148-155`).

The owner's three questions, answered:

(a) `ds4_posttrain.c/.h` standalone: PORTABLE NOW -- pure closed-form
    math (output-space 1-bit row scale, `ds4_posttrain.h:12-18`), no GPU,
    no model, and no callers anywhere in the engine (`docs/clear.md:113`
    schedules the module for deletion: zero references).  Porting it
    preserves a validated formula (cos 0.55 -> ~0.70,
    `gguf-tools/docs/ALGORITHM.md:57`; the header's own citation
    `gguf-tools/go-onebit/ALGORITHM.md` is stale), not a live path.
    Preserve exactly: `double` accumulation in `wx`-then-`bx` order,
    `sign(0)=+1` via `< 0.0f` in both the bit fill and the accumulation,
    `den <= 1e-30 -> 0` with the `mean|w|` fallback, the cosine
    scale-cancel convention.

(b) Tests for uploading training data as JSON: PORTABLE NOW, host-only.
    The oracle is the studio's `api_upload` (`train_api.c:138-185`):
    raw-body POST (not multipart), 64 MB cap (`TR_UPLOAD_MAX`,
    `train_internal.h:34`), `kind=jsonl` stores the body byte-exact at
    `gguf-tools/data/posttrain/<name>` after a per-line LOOSE check
    (JSON object + contains "messages" or "text", `:155-158`); `kind=doc`
    writes `chunks/<base>.txt` + regenerates `<dir>.text.jsonl`
    (`rebuild_text_jsonl`, `:113-134`).  The trainer's reader
    (`core_ptrain_data.c:193-260`) is STRICTER (field whitelist; rejects
    `reward/mode/split`, duplicate roles, trailing bytes) and the port
    must reproduce BOTH layers, not unify them.  The 14-case corpus and
    the acceptance rules are in the research note, section 9.

(c) LOADED and TRAINED: LOAD is half-done.  The port reads, merges and
    fingerprints ②③ (`crates/ds4-core/src/sidecar.rs`, `zchain.rs`,
    §6.6.4) and REFUSES the amp apply by name (`v41_run.rs:1013-1017`,
    `amp_Lnn.bin ... not ported`).  The apply itself is small -- one GEMM
    pair + bf16 round at the engine's forward tail -- and rides the P4-7
    device stores (PT-2).  TRAINED is BLOCKED on the inference port
    closing: it needs the whole GPU backward family (24 `ds4_gpu_bwd_*`
    entries, `ds4_gpu_bwd.h` + the five `cuda_bwd_*.inc.cu` files) and
    the port has no backward pass (`cuda/ds41_vq_prefill_mma.cuh:595-597`).
    A hard dependency, not a scheduling preference.

Unit split (each with its own gates):

| unit | scope | gate |
| --- | --- | --- |
| PT-1 (now) | the `ds4_posttrain` module port + the upload surface (validate/store/list/preview, 64 MB cap) | model-free: module vs the C oracle on a captured fixture; upload vs `ds4-train` on the Spark (the 14-case corpus, byte-equal stored files) |
| PT-2 (now) | the ③ apply: `ds4_gpu_v41_amp_apply_tensor` mirror + the forward-tail hook; lift the refusal when the merge carries amp | model-free parity vs a synthetic `amp_Lnn.bin`; the live gate the day a directory carries one (none on the Spark today, §6.14) |
| PT-3 (after P4/P5 close) | the trainer (`core_ptrain*.c`) + the backward kernel family + the studio job/gate; assets: `hold.jsonl`, the wt2 anchor + `anchor_metrics` (release-pack externals) | packcheck/rccheck/hccheck/gradcheck, then the wt2 gate rule (`dpp >= -0.5`, `rel <= 0.03`, `train_gate.c:109`); memory floors re-measured on the Spark before adopting the engine's numbers |
| PT-4 (optional) | `posttrain_reward` -- LEGACY (the 10-10 trainer dropped the reward loop; no studio consumer) | skip unless the owner wants it |

Open items recorded: `hold.jsonl`'s producer is unverified (shipped by
`release_pack.sh:24`, no in-tree generator); the studio's upload check is
intentionally looser than the trainer's (preserve the split); the
trainer's 3000 MB / 2500 MB floors are the engine's numbers -- re-measure
before adopting (rule 20).

## 7. Numerics contract

- Device arithmetic and every format detail follow the C engine's code, and the
  PTX/SASS it compiles to where the two disagree.
- The VQ gain application and the E4M3 codeword interpretation follow
  `cuda_vq_row.inc.cu`, not a plausible reading of the format description.
- Greedy (temperature 0) comparisons are the gate. Sampled comparisons are
  reported as distributions, because the Rust host samples with its own profile
  and the streams diverge after the first non-greedy step. (For the v41 route
  unit H superseded this: the port samples with the engine's own coins, so a
  pinned seed is gated byte-identical, §6.19.)
- Where the Rust host's existing DeepSeek kernels and the engine disagree
  numerically, the engine's captured behaviour wins for this family, and the
  deviation is recorded rather than smoothed over.

## 8. Risks

| risk | evidence | mitigation |
| --- | --- | --- |
| The golden set never lands, so P4+ has no gate | none captured yet; P0 is the first step | P1-P3 proceed regardless and say so |
| Scope is large: four subsystems (VQ, engram, MTP, sidecars) | §5, each with its own kernels | phase gates; the base V4 family already works, so nothing existing regresses |
| The host memory budget: 113.6 GB base + sidecar, 203 GB engram on SSD | artifact README §3 | engram is optional by design (`--v41-no-engram` exists in the reference) |
| Two engines, one model: the Rust host is not the C engine and will not be bit-identical everywhere | P4 gate is the first real test | gate on logits/tokens at temperature 0 and record deviations |
| Spark access: no standing key (owner rule), state-changing commands need per-connection approval | memory 2026-10-03 | drive the Spark only through the owner's ssh tmux pane or askpass, read-only by default |
| The port quietly becomes a reimplementation of the engine | this is the doctrine's whole point | cite or refuse; no invented mechanism |

## 9. Rollback

The port is additive: a V4.1 variant/keys loader plus new device paths; no
existing family's code path changes. Rollback is reverting or deleting the
branch. No migration, no data, no protocol change.

## 10. Open questions for the owner

- O1 Does v1 ship without engram (the 203 GB shards) and without speculative
  decode? Both are disableable in the reference, so both are phasable.
- O2 Sampling: the reference ships the model card's recipe as the default
  (temperature 1.0, top_p 1.0, speculative on). Does this tree adopt that
  per-family profile, or keep request-driven sampling only?
- O3 Where does the port run? The reference is tested on the Spark alone; this
  host has no DS4 GGUF and no 110 GB of free VRAM-class memory.
- O4 Is parity with the C engine the goal (G2 as written, expensive), or
  capability plus speed with a looser numerical tolerance?
- O5 How does the `ds41` branch reach the Spark: clone the fork there (needs a
  credential for a private repo), or rsync the working tree over the existing
  gvfs/ssh path? The Spark has no Cargo workspace for this tree yet.
