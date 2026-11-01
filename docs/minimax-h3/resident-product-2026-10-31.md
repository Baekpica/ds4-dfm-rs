# The product route in one process: a job list on the resident engine (2026-10-31)

Handoff section 3 of the H3 campaign. A one-shot process renders one clip and
exits, so every clip pays the prompt encode, the 62 GiB DiT read and every
weight load again. This unit puts the whole route in one process behind one
engine, states the request list as a file, and measures what the second clip
costs.

Spark `gx10-e1d2`, sm_121a, CUDA 13.0, `nvidia-clocks.service` absent
(persistence Enabled), so the numbers below carry the host's boost clocks and
nothing else.

## 1. What was added

- **The job list** (`crates/ds4-core/src/minimax_h3/jobs.rs`): one request per
  line, seven TAB-separated fields — prompt, seed, steps, width, height,
  frames, output — with blank and `#` lines skipped. A tab and not a comma
  because a prompt carries commas, quotes and spaces, and because a list has to
  be writable from a shell without an escape language. The list is this port's
  own surface: the reference CLI has no batch mode (`h3_cli.c`), so the only
  rule it inherits is the engine's, that one request runs at a time.
- **`--jobs FILE`** (`h3-rs`): reads the list, validates every canvas and frame
  count before the first weight is read, then renders the requests through one
  `EngineHandle`. Per-request flags on the command line are refused rather than
  dropped: the line owns the request.
- **The decoder's arms are options now** (`RenderRequest.vae_attention`,
  `vae_gemm`): `None` leaves the context on `H3_VAE_ATTENTION`/`H3_VAE_GEMM`'s
  word instead of forcing an arm. Without that, a job list could not run the
  `bf16i` arm this host measured at 48.2-48.7 s (`vae-decode-2026-10-31.md`).

## 2. The acceptance

`tools/h3-jobs-accept.sh` (`H3_ARTIFACTS` required, nothing hardcoded) runs the
same request two ways at one arm:

1. the list twice, in one process (`h3-rs --jobs scratch/jobs.txt`);
2. the one-shot route the list replaces: `h3-rs ... --hash` (the `--hash` flag
   is new; it prints the decoded frames' FNV, the engine's own `frames_hash`).

Command (Spark, `H3_VAE_GEMM=bf16i H3_VAE_ATTENTION=tensor`):

```sh
H3_ARTIFACTS=/mnt/h3-artifacts H3_VAE_GEMM=bf16i H3_VAE_ATTENTION=tensor \
  bash tools/h3-jobs-accept.sh
```

One request: the release's draft prompt at 672x384x124, 4 steps, seed 42,
into a 1344x768x124 clip. Phase table, seconds:

| phase | job 1 | job 2 |
|---|---|---|
| prompt encode (NVFP4) | 12.015 | 0.000 (hit) |
| DiT load | 7.074 | 0.000 (hit) |
| denoise, 4 steps | 21.352 | 21.705 |
| audio VAE | 2.953 | 3.151 |
| video VAE decode | 54.317 | 51.049 |
| mux | 1.039 | 0.953 |
| **wall** | **100.158** | **78.273** |

Two clips in one process: `JOBS wall=178.431s` (stage wall 179.424 s including
the guard wrapper).

So the second clip costs **78.273 s against the first clip's 100.158 s**, and
the two clips in one process cost 179.424 s. The equality the route has to keep:
the two clips in the process and the one-shot render all produce the same
frames hash (`395e1b371570e399`).

The latent and audio hashes agree in the same way (`video_fnv=199fad94f9df0ea7`,
`audio_fnv=ea7070c8d4cddbeb` in both jobs).

Four runs of this acceptance on the same tree (the last four build/rebuild
steps of the unit) printed the same hashes; the table is the last one,
measured with `h3-rs` built from `56242e84`.

Job 1's own loads are page-cache dependent: a run on cold pages measured
`text=15.639s load=12.788s wall=111.346s` for the same request, and the
one-shot route on warm pages still spends 15.1 s on the prompt and 15.5 s on
the DiT read. That is the work the second clip no longer repeats.

## 3. What it does not establish

- **The video decoder is not resident.** Each job loads the 9.8 GB decoder
  inside `video`; the second clip's 51.1 s includes that read. Holding it beside
  the resident DiT is a separate memory decision, not attempted here.
- **The decode's cost is not explained.** The isolated `bf16i` arm measured
  48.2-48.7 s (`vae-decode-2026-10-31.md`); the second clip decoded in 51.1 s
  while job 1's decode took 57.5 s.
  Job 1's extra ~6 s is unexamined — cold page cache for the decoder weights,
  or contention with the resident DiT — and this unit does not attribute it.
- **One request, one shape, one seed, one arm.** A second prompt or a second
  canvas would miss the embedding and the prepared DiT by design; the list's
  hit rules are the engine's `PreparedKey`, unchanged.
- **The list carries text only.** No references, anchors, first/last frames or
  media; the engine's own fields exist, the file format does not state them.
- **No clock policy.** `nvidia-clocks.service` is absent on this host, so
  absolute times move with the boost clocks; the two compare runs
  were interleaved within one minute of each other, which is what makes their
  comparison usable.
- **The MP4 bytes are not compared**, only the decoded frames' hash; ffmpeg
  muxing remains its own surface.
