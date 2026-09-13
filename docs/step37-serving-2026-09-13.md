# Step 3.7 Flash: banked serving and cache reuse

This work adds several Qwen3.8 serving options to Step 3.7 Flash: disk KV checkpoints, opt-in multiple sequence banks with fork and warm
reuse, and per-bank disk KV. It builds on landed main (`32842f3`, the post-#37
optimization campaign) and does not change the numerics of the serial forward,
MTP, or vision paths.

## What Qwen has and where Step stood

The Qwen worker runs `ds4-server --cuda -c 262144 --mtp-draft 2
--kv-disk-dir ... --kv-disk-space-mb ...` with `DS4_SERVER_COALESCE_MAX=2`,
`DS4_SERVER_FORK=1`, `DS4_SERVER_FORK_PARTIAL=1`, `DS4_SERVER_CONTINUOUS=1`,
`DS4_SERVER_WARM`. All of it hangs off a continuous-lane `BatchCtx`.

Before this change Step refused `BatchCtx` creation
(`ds4_engine_supports_batching` returned false for `STEP37`) and both serial
disk-KV entry points were stubbed
(`"Step session snapshots are not implemented yet"`). So every Step request took
the serial lane at width 1, disk KV silently no-oped, and a follow-up replayed
its whole history.

## Disk KV (serial lane, default on)

`ds4_session_save_payload` / `load_payload` now serialize Step. The payload is
the shared DSV4 file (13×u32 header + token ids + tail), tagged with a new
`DS4_SESSION_STEP37_LAYOUT_MAGIC` (`"STP3"`) and mirrored in the Rust host as
`PayloadLayout::Step37`.

Step keeps only the causally live frontier, written in logical order so the
file does not depend on the prefill-chunk (`cap`) or context it was written
with:

- every full-attention layer: rows `[0, n)`;
- every sliding layer (window 512): the last `min(n, 512)` rows;
- when `--mtp` is loaded, each of the three predictor rings' live window plus
  the held target hidden rows (`spec.tail`), so a restored session passes
  `step37_spec_valid` and can decode or extend immediately.

The ring stores logical position `p` at slot `p % cap`, so save/restore walk
the occupied slot range in at most two contiguous spans per layer. Images are
excluded (their device features would have to travel with the rows), so a
session holding image features reports zero payload bytes and is not
snapshotted. A checkpoint written with MTP predictor state reloads into a
plain (non-MTP) session by skipping it; a plain checkpoint is refused by an
MTP session, which needs the predictor rings.

`tests/test_step37_session` restores a base-only session from a truncated and a
full payload and checks the restored frontier and every layer's live KV rows
against a fresh serial replay, then decodes one more token. `tests/test_step37_spec`
snapshots mid-generation, invalidates, reloads, and confirms the target and
all three predictors resume byte-identically. Both pass on the MQ83 model.

## Multiple sequence banks (continuous lane, opt-in)

`DS4_STEP37_BATCH=1` enables `ds4_engine_supports_batching` for Step, exactly as
`DS4_QWEN_BATCH=1` does for Qwen. Left unset, Step stays on the serial lane with
MTP.

`ds4_step37_batch_runtime` holds one self-contained `ds4_step37_graph` per bank
(its own 45-layer KV rings and prefill scratch), so admission, prefill and
decode reuse the single-graph forward that the earlier campaign already
validated — a bank is byte-identical to a serial session. It plugs into the
shared `family_banked_*` continuous scheduler alongside EXAONE, Solar and
Motif-3, which gives, at `DS4_SERVER_COALESCE_MAX=2`:

- two independent sequences interleaved on the lane;
- full-frontier fork (`DS4_SERVER_FORK=1`): a request sharing a bank's whole
  committed frontier clones its KV instead of re-prefilling;
- warm prefix reuse and per-bank disk KV (`--kv-disk-dir`), reusing the same
  `STP3` payload as the serial lane.

The follow-up adds below-frontier partial fork. Up to 32 shared checkpoints
store the sliding layers' live 512-row windows and frontier logits;
full-attention rows copy from the source bank. Slots map on demand within
the memory reserve. Request boundaries and periodic prefill/decode frontiers
are captured; a cut resumes from the nearest retained checkpoint and replays
its suffix. `DS4_SERVER_FORK_PARTIAL=0` disables this pool.

Forks inherit checkpoint references. Reset and disk restore discard the old
lineage. A compact restore refuses rewinds below its saved window; older ring
slack was not captured. Missing checkpoints fall back to cold prefill.

With `--mtp`, each bank owns three predictor rings and held target hidden
rows. Greedy decode uses the existing native trial/commit with Rust selecting
the accepted prefix. Forks and disk checkpoints carry predictor state too.
Cancellation, EOS and forced protocol tokens can shorten the emitted prefix;
only that prefix commits. Sampled requests use ordinary decode while keeping
predictors current. `DS4_MTP_SPEC_DISABLE=1` keeps ordinary decode for diagnosis.
Image requests fall back to the serial session automatically
(`prepare_qwen_images` refuses a non-Qwen model, so the continuous prompt is
not prepared and the router picks the serial lane).

`tests/test_step37_cont` (base only, no weight owner needed) checks, on MQ83:

- two banks decode 16 greedy tokens byte-identically to two independent serial
  sessions;
- a full-frontier fork of a 175-row committed bank matches a serial session
  prefilling those rows;
- a bank disk-KV snapshot round-trips 190 committed tokens.
- a partial fork restores the 160-row prompt checkpoint after generation has
  advanced, then matches all 16 serial continuation tokens.

`tests/test_step37_checkpoint` is a model-free GPU copy gate. It checks every
live KV row after wrapped-window and in-place restoration, all vocabulary
logits, source-bank preservation, shared lineage and the restored rewind floor.
The follow-up MQ83 gate passed with the GPU locked to 300–2200 MHz (observed
2197 MHz); this is correctness evidence, not an uncapped speed comparison.

## Verified configurations

Serial text (32768 configured context; long-answer probes use about 17.8K):

```sh
./ds4-server --cuda --port 8000 -c 32768 \
  -m "$STEP/MQ83/Step-3.7-Flash-MQ83-00001-of-00009.gguf" \
  --mtp "$STEP/MTP/Step3.7-flash-mtp-Q8_0.gguf" --mtp-draft 3 \
  --kv-disk-dir /path/to/step-kv --kv-disk-space-mb 32768
```

Concurrent text (two 65536-context banks; bounded 6.3K requests):

```sh
DS4_MEM_FLOOR_GB=12 DS4_STEP37_PREFILL_CHUNK=2048 \
DS4_STEP37_BATCH=1 DS4_SERVER_COALESCE_MAX=2 DS4_SERVER_CONTINUOUS=1 \
DS4_SERVER_FORK=1 DS4_SERVER_FORK_PARTIAL=1 \
./ds4-server --cuda --port 8000 --mem-floor-gb 12 -c 65536 \
  -m "$STEP/MQ83/Step-3.7-Flash-MQ83-00001-of-00009.gguf" \
  --mtp "$STEP/MTP/Step3.7-flash-mtp-Q8_0.gguf" --mtp-draft 3 \
  --kv-disk-dir /path/to/step-kv --kv-disk-space-mb 32768
```

The recorded tests import one shared BASE+MTP VMM owner with
`DS4_CUDA_WEIGHT_IPC_MANIFEST` and `DS4_CUDA_WEIGHT_IPC_SCOPE=both`.
They keep a 12GiB host-memory guard. Set both `DS4_MEM_FLOOR_GB=12` and
`--mem-floor-gb 12`: the native bank planner reads the environment while the
Rust serial-admission policy reads the CLI setting. A configured context is
not a full-length capacity proof.

For mixed text/image serving, the tested two-bank configuration is `-c 8192`,
`DS4_STEP37_PREFILL_CHUNK=512`, with the same MTP options and
`--vision "$STEP/vision/mmproj-step3.7-flash-f16.gguf"`. Nine image requests
pass across three APIs: multiple screenshot resolutions, a changed count,
invoice and photo follow-ups, and four-image streaming. Images take the
serial lane beside the text banks. At chunk 4096 the planner reduced the
bank count to one, then image admission returned HTTP 503 because no serial
graph fit beside it. Smaller workspaces are required for this combination.

The two 64K text banks serve concurrent requests through the continuous
route, with no fallback. A repeat and both concurrent requests reuse 4096
of about 6324 prompt tokens and preserve their answers. Native full/partial
fork and disk round-trip correctness are established by the gates below.

Disk files are written successfully, but a normal Chat follow-up after
restart was cold: the official template removes the prior empty thinking
block, so the saved text no longer matches the replayed prefix. Disk restore
drops partial-checkpoint lineage and cannot recover that earlier frontier.
A completion request also applies the template; passing a serialized prompt
there does not bypass it. Do not infer cross-restart Chat cache hits from
payload round-trip tests. Live partial reuse and disk reuse have different
qualification boundaries.

A `-c 262144`, two-bank startup fitted down to one 14.72GiB bank and was
stopped by the host-memory guard before readiness. It is not qualified by
this campaign. No clock-limit violation was recorded. These limits remain
relative to the Qwen worker; the Step options do not imply identical memory
costs or context capacity.

The follow-up MTP gate uses one MQ83 mapping and an owner-imported Q8 MTP.
Two banks, full/partial fork and serial controls match all generated tokens,
all 128,896 logits, every live target/predictor KV row and held hidden row.
The disk gate clears KV before restoration and compares the complete saved
payload byte-for-byte. Synthetic wrapped-ring restoration, cancellation,
EOS and forced-token boundaries also pass under the 300–2200 MHz cap.
See the [capped performance report](step37-optimization-2026-09-13-r3.md) and
its sample data for the matched A/B protocol.
