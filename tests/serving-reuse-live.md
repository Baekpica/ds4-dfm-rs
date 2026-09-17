# HTTP reuse regression

`serving_reuse_live.py` only sends Chat requests to an already running local
server. The operator owns model startup, shutdown and cache directories.
Use a dedicated endpoint: any extra generation invalidates the route trace.

The fixture uses actual returned assistant content, including whitespace.
Arithmetic correctness and byte-exact warm/cold output comparison are separate
checks. A wrong answer on both paths fails arithmetic even when parity passes.
Timings are functional observations, not a performance qualification.

The `literal-arithmetic-v2` answer contract declares four accepted strings per
case in `fixtures/serving-reuse.json`: the plain number, that number with a
period, and the matching literal equation with or without a final period.
Only outer whitespace is ignored for arithmetic acceptance. Operators, operands,
internal spaces and all other characters must match a declared string; there is
no numeric extraction or substring match. The accepted forms are copied into
the frozen case and its summary receipt. Reasoning, tool calls and non-`stop`
completion remain failures.

This gate qualifies these arithmetic answers and cache parity, not number-only
format following. The prompts still request just the number, so an accepted
equation can violate that formatting instruction. Earlier number-only runs
remain failures under their original contract: `2 + 2 = 4.` must not retroactively
turn such a formatting failure into a pass. Start a new evidence directory;
v3 refuses to resume an older frozen fixture. Never add answer forms after
seeing output within a campaign.

| Profile | Artifact scope | Warm reuse | MTP |
|---|---|---|---|
| `qwen` | Qwen3.8 Q5 main plus the selected BF16 or FP8 SSD-PLE sidecars | partial | explicitly off, or separately on with draft 2 |
| `solar` | Solar Open2 MXQ-v1, all 11 shards | partial | off |
| `motif` | Motif-3 MQ87-88-FIT canonical GGUF | partial | off |
| `deepseek` | exact Flash/PRO artifact; include any loaded MTP/DSpark sidecar | exact | explicitly off, or separately on with the declared draft |

Provide the same verified artifact manifest to every phase. The runner records
its SHA256 and copies it into the evidence directory; it does not read or
rehash model weights. Owner imports and derived weight artifacts retain their
existing family launch requirements. Different artifacts or MTP settings need
separate evidence directories.

Use two banks, CUDA, context 2048 or larger, native prefill chunk 64, and no
other requests. The short fixture needs these settings on seed/warm/restored:

```sh
DS4_SERVER_CONTINUOUS=1 DS4_SERVER_FORK=1 \
DS4_SERVER_PIN_MIN_TOKENS=0 DS4_SERVER_PERSIST_MIN_TOKENS=1 \
./ds4-server --cuda -m "$MODEL" --model-id "$MODEL_ID" \
  --host 127.0.0.1 --port "$PORT" --ctx 2048 --max-seqs 2 \
  --native-chunk 64 --prefill-chunk 64 --prefill-chunk-live 64 \
  --prefix-reuse partial --mtp-mode off \
  --kv-disk-dir "$CACHE_DIR" --kv-disk-space 2G --kv-cache-min-tokens 1
```

For DeepSeek, replace `--prefix-reuse partial` with `exact`. Keep the model's
existing owner/PLE/sidecar arguments. The server's requested lane is `auto`;
the runner's explicit `--lane continuous` checks the actual route. There is
no server `--lane continuous` option. Do not set `DS4_SERVER_CONTINUOUS=0`.
For an additional MTP-on run, pass `--mtp-mode on --mtp-draft 2` to the server
and runner, and `--expect-speculation on` to the runner. Solar/Motif reject on.

For Motif, also set `DS4_MOTIF3_BATCH_TRACE=1` and redirect the server's stderr
to a regular file. The runner reads only the new bytes from that PID's stderr
for each request; it records the file identity, byte range and raw trace hash.
Its official template removes the generation-only empty thinking pair, so
append/branch may restore a partial checkpoint at the canonical history
frontier. This is reported as `partial`, including when native code copies
that checkpoint to another bank.

Start with an empty, dedicated disk cache and evidence directory. `$PID` is
the inference server PID, not its owner, shell or watchdog. Record clocks and
memory guard receipts alongside this evidence when running on the GPU host.

```sh
python3 tests/serving_reuse_live.py seed \
  --url "$URL" --pid "$PID" --output "$OUT" \
  --artifact-manifest "$ARTIFACTS" --family qwen --model "$MODEL_ID" \
  --context 2048 --banks 2 --native-chunk 64 --lane continuous \
  --mtp-mode off --expect-speculation off

python3 tests/serving_reuse_live.py warm \
  --url "$URL" --pid "$PID" --output "$OUT" --artifact-manifest "$ARTIFACTS"
```

The operator then gracefully stops the worker and restarts the same executable
with the same artifact/settings/cache. Run `restored` as its first generation:

```sh
python3 tests/serving_reuse_live.py restored \
  --url "$URL" --pid "$RESTART_PID" --output "$OUT" --artifact-manifest "$ARTIFACTS"
```

For cold controls the operator starts a third process with the same shape,
chunks and MTP, `--prefix-reuse off`, and **without disk-cache arguments**:

```sh
python3 tests/serving_reuse_live.py cold \
  --url "$URL" --pid "$COLD_PID" --output "$OUT" --artifact-manifest "$ARTIFACTS"
```

| Request | Accepted literal forms (after outer whitespace trim) | Warm trace/cache |
|---|---|---|
| seed: 2 + 2 | `4`, `4.`, `2 + 2 = 4`, `2 + 2 = 4.` | cold, zero cached |
| append: 4 + 1 after actual seed reply | `5`, `5.`, `4 + 1 = 5`, `4 + 1 = 5.` | exact/fork, positive proper prefix |
| edit: replace second user turn with 4 + 2 | `6`, `6.`, `4 + 2 = 6`, `4 + 2 = 6.` | partial for Qwen/Solar/Motif; exact/fork for DeepSeek |
| fork: extend the retained append branch with 5 + 3 | `8`, `8.`, `5 + 3 = 8`, `5 + 3 = 8.` | exact/fork, positive proper prefix |
| restart: extend actual fork reply with 8 + 1 | `9`, `9.`, `8 + 1 = 9`, `8 + 1 = 9.` | exact/fork as first generation after restart |

The warm phase must observe at least one actual bank fork. For Motif, append
and retained-branch continuation additionally accept `partial`, and the native
trace must confirm a successful copy to a different bank, with the reported
cached count, unchanged source frontier and matching target frontier. A
`partial`/`fork` label alone cannot satisfy this Motif check. This demonstrates
the copy and frontier; tensor/source-content preservation is a separate native
gate. Other families require at least one `fork` request trace. A later branch can
reuse its still-resident parent with `exact`; the scheduler need not copy a
bank again for that request.

Every cold request uses the identical saved body, requires zero cached tokens
and `cold` trace, and compares the full assistant message, finish reason and
completion-token count against its matching seed/warm/restored response.
Two different accepted forms still fail this byte-exact comparison.
The run also checks effective context, bank count, chunk, MTP and disk policy,
request lane/MTP/reuse settings, actual speculation and absence of fallback.
Both scheduler chunks must equal the declared native chunk.
DeepSeek's exact-only contract never counts an edited request as partial reuse.

Each phase writes process/binary identity, request/response/stats/summary files
and a result with hashes. Later phases verify the preceding fixture digest;
cold requests also match the original recorded request bodies. PID-to-endpoint
ownership is an operator assertion; the process receipt does not bind the TCP
listener to that PID. A phase with failed numerical or trace checks still
retains its requests for cold diagnosis and returns failure. Existing phase
records are never overwritten; use a new evidence directory for a new trial.
The final generated `fixture.json` is the exact replay fixture, including the
actual assistant replies. These short checks do not qualify long-context,
multimodal, tool, cancellation or throughput behavior.

Model-free runner checks:

```sh
python3 -m unittest discover -s tests -p test_serving_reuse_live.py
```
