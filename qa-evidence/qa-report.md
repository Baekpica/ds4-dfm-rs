Rule 19 QA report - MiniMax-H3 (h3): the three commits pushed since 6cdc1679 and
the four worktree surfaces the next commit will carry (sixth pass)

repository : /data/ds4-dfm-rs
branch     : feature/minimax-h3
HEAD       : 875a43fdc477e5019d4fe0a3f51663fad4f0470e, and
             fork/feature/minimax-h3 resolves to the same sha. The three
             Group A commits are therefore already pushed and are the last
             three commits of HEAD.
worktree   : modified, unstaged, exactly four files (the Group B surfaces):
             crates/ds4-cli/src/bin/h3-rs.rs,
             crates/ds4-core/src/minimax_h3/vae_video.rs,
             docs/minimax-h3/v1-local-gate.txt,
             native/h3/h3_dit_gpu.cu.
             git diff fork/feature/minimax-h3 --stat is 302 insertions and 60
             deletions over those four files and nothing else. This pass
             committed nothing, pushed nothing, created no branch and wrote
             this report and nothing else in the repository.
date       : 2026-10-26
qa model   : deepseek-v4.1-CC-flash (the project's recorded QA model)
host       : x86_64 "vizzio", LMDE 7 / Debian 13, kernel 6.12.113, 28 CPUs,
             rustc and cargo 1.98.0 (the pinned toolchain; rustfmt and clippy
             present). GPU: NVIDIA GeForce RTX 4070 SUPER, compute capability
             8.9, driver 610.43.03, CUDA UMD 13.3, toolkit 13.3.73 at
             /usr/local/cuda -> /usr/local/cuda-13.3. Media: ffmpeg and
             ffprobe 7.1.5-0+deb13u1. The DGX Spark (192.168.1.91, GB10,
             sm_121a) is NOT reachable from this host and was not touched; no
             /mnt/h3-artifacts and no ~/ds4-h3-port exist here.
shas read  : 7ba78b8a, 7b9e2f1e, 875a43fd (Group A, git show each), the
             worktree against fork/feature/minimax-h3, and /data/h3.c (the
             read-only reference engine) at h3_video_vae.c, h3.c, h3_weights.c,
             h3_gpu_cuda.cu, h3_text_encoder.c, h3_ffmpeg.c, main.c, Makefile.

WHAT CHANGED SINCE THE LAST PASS

The fifth pass closed at 6cdc1679 with verdict PASS. Three commits have landed
since, all pushed, none of them QA'd: 7ba78b8a gates the stub renderer in the
routes test, 7b9e2f1e records the reverted load experiment in the gate
document, 875a43fd bounds the CI media install. On top of HEAD, four files sit
unstaged and carry the next commit's surface: a pinned staging buffer and an
8 MiB-piece upload loop in native/h3/h3_dit_gpu.cu, a once-per-decode
TileScratch in crates/ds4-core/src/minimax_h3/vae_video.rs, an
ANCHORS ... seconds= mark around read_anchors in
crates/ds4-cli/src/bin/h3-rs.rs, and a rewritten docs/minimax-h3/v1-local-gate.txt
whose 2026-10-26 sections withdraw the previous entry's conclusion about the
copy path and attribute the earlier output change to a missing -arch. This pass
verified the three pushed commits' subjects live where this host allows, then
verified the four worktree surfaces: a measured compile of the modified native
translation unit under the exact release arch flags, the VAE scratch against
the reference's own allocation order, the CLI mark against every parser in the
tree, and a claim-by-claim trace of the two new document groups to files and
recorded artifacts that exist here.

GROUP A, THE THREE PUSHED COMMITS

crates/ds4-server/src/h3_video.rs - 7ba78b8a "fix(h3): gate the renderer in the
routes test"

  Commands run: cargo test -p ds4-server --lib h3_video -- --test-threads=1;
  then ten consecutive isolated runs of
  cargo test -p ds4-server --lib --locked
  h3_video::tests::the_three_routes_serve_create_poll_and_content -- --exact
  --test-threads=1.

  Results: the module ran 17 tests, 17 passed, 0 failed. The single routes test
  ran 10 of 10 PASS, each in 0.01 s, exit 0 every time (a first batch of six
  runs before that was also 6 of 6 PASS, so 16 consecutive passes). No hang, no
  flake, no timing dependency in the observed behaviour.

  Code read (crates/ds4-server/src/h3_video.rs:853-1011). The gate is
  gated_renderer() at :855-863: an mpsc channel whose Receiver sits in a Mutex
  inside the render closure, which blocks on recv() before writing the 8-byte
  output; the test holds the Sender and only calls release.send(()) at :988,
  after the early GET has been asserted 404 with error code video_not_ready
  (:982-984). The happy path is still asserted after the release: poll_until
  "completed" (:990), completed_at non-null, size 1344x768, seconds 10, frames
  243 (:991-994), then content 200 with Content-Type video/mp4 and
  Content-Length 8 (:996-1002) and a HEAD whose wire bytes end in CRLF CRLF,
  "HEAD carries no body" (:1004-1010). The 404 is a decision on job state, not
  on the file: VideoServer::content (:607-625) returns video_not_ready whenever
  video.status != Completed, before it ever looks at the path.

  Can a test that fails before the release hang the suite? No, and the
  mechanism is in the code: if an assert before :988 panics, the test unwinds,
  the Sender is dropped, the renderer's recv() returns Err and the let _ =
  discards it, so the closure falls through to write_output and the render
  thread finishes. The render thread is detached anyway (thread::Builder spawn
  at :545-551, only spawn.is_err() is inspected), and nothing in the file joins
  it - the only join in the module belongs to a test-local stderr reader
  (:360) and to the socket test's own accept thread (:1528-1540).

  Live CI cross-check (GitHub Actions API, Baekpica/ds4-dfm-rs): run
  40226663635 at 8b9ca95a (the sha the commit message names) concluded
  failure; run 40226994758 at 7ba78b8a concluded success. The fix's subject is
  therefore confirmed on the runner as well as here.

docs/minimax-h3/v1-local-gate.txt - 7b9e2f1e "docs(h3): record the reverted
load experiment"

  This commit added a 2026-10-25 section that the worktree edit replaces
  wholesale, so it was read with git show 7b9e2f1e and checked where its
  numbers survive in the new text and in the recorded CSV.

  Traceable: "a blocking cudaMemcpy ... 116.1 -> 106.7 s" and "reading each
  tensor into an anonymous buffer first ... 106.7 -> 23.6 s, wall 467.7 s"
  agree with the worktree table's loadfix (106.7) and stage (23.6, 467.7) rows.
  "the port's committed output is byte-identical to the reference at this shape
  (74d862fef9dc904d69c20f3dffec74f4); with either change it is
  7991a015b1a161ec95d6b3f1aae8e3fc" agrees with the worktree text, which now
  explains the change as the arch rather than the copy path. Its reference
  witness citations (h3_weights.c:144-186, h3_gpu_cuda.cu:375-425) resolve in
  /data/h3.c; the second is the device route that reads a tensor's own byte
  span, the first is load_tensor's dtype/dimension checks (the port's own
  text_weights.rs:14 cites the same region as :375-424).

  Flagged, not a defect of the file: the commit message's conclusion ("A copy
  cannot change numerics, so the difference is unexplained") is withdrawn by
  the worktree text, and the withdrawal is explicit and better evidenced (the
  no-arch compile command reproduced below). Flagged as an observation: "within
  8% of the reference's 433.39 s" is arithmetically right ((467.72 - 433.39) /
  433.39 = 7.9%) but 433.39 is the reference's profile=1 row (perf.csv row 3),
  while the port rows are profile=0; tests/h3/perf/README.md:38-40 says wall
  may only be compared between rows with equal profile. Against the clean
  reference row (441.5) the gap is +5.9%, so the conclusion survives either
  way.

.github/workflows/host-parity.yml - 875a43fd "ci: bound the media install and
drop recommends"

  GitHub Actions cannot be run locally, so the file was checked structurally
  and its behaviour was checked against the live runs of this workflow. There
  is no YAML parser on this host (jq has no YAML reader; yq, yamllint and
  python are absent), so the structural check is an exact-indentation read, not
  a parse; it is stated as such. What the read found, mechanically confirmed:

  - the file has no tab characters (grep -Pc "\t" -> 0), so no YAML
    indentation error of the classic kind;
  - timeout-minutes: 30 sits at 4-space indentation, i.e. on the job
    (line 18), and is unchanged: the job is still bounded at 30 minutes;
  - timeout-minutes: 12 sits at 8-space indentation, i.e. on the Install media
    tools step (line 31), which is what the commit claims;
  - --no-install-recommends is present (1 occurrence) and only on the install
    call, which keeps ffmpeg while dropping the recommended set;
  - both apt-get calls carry the bound: Acquire::Retries=3 twice,
    Acquire::http::Timeout=30 twice, Acquire::https::Timeout=30 twice, over 2
    apt-get invocations (one update, one install);
  - both version lines are printed: "ffmpeg -version | head -1" and
    "ffprobe -version | head -1" each appear once, inside that step's run
    block;
  - the run block uses sudo env DEBIAN_FRONTEND=noninteractive, so the
    timeouts are visible to apt and not to a shell wrapper only.

  Live evidence from the API of the same workflow:
  - run 40229947520 at 7b9e2f1e was cancelled after its Install media tools
    step ran 20:16:15Z to 20:46:13Z, and every later step is "skipped" - the
    step consumed the job's whole 30 minute budget, which is the failure the
    commit describes;
  - run 40232884549 at 875a43fd (the current head) concluded success with the
    job 20:53:31Z to 21:01:13Z, and its Install media tools step ran
    20:53:45Z to 20:54:07Z, 22 seconds. The bound is not merely declared, it
    took effect. Format, Clippy, C and Rust host parity, Serialized workspace
    tests, All host targets and Native serving host all succeeded in that run,
    so the file was parsed and executed by GitHub as written - the strongest
    available check short of running the runner.
  - the job log itself is not downloadable without credentials (GET on
    /actions/jobs/121288176158/logs -> HTTP 403), so the two version lines
    were verified in the file and by the step's success, not read back from
    the runner's output. Recorded as an accepted risk.
  - one figure is stale: the commit message and the file's own comment both say
    run 40229947520 "spent 26 of its 30 minute budget inside this step", while
    the API's step timestamps say 29 minutes 58 seconds. Substance unchanged
    (the step ate the budget and everything after it was skipped), number
    understated by four minutes.

GROUP B, THE FOUR WORKTREE SURFACES

native/h3/h3_dit_gpu.cu

  Diff read in full (h3d_context gains stage, stage_done, stage_pending;
  H3D_STAGE_BYTES = 8 MiB; a new static h3d_upload_range; h3d_upload and
  h3d_upload_at both route through it; h3d_destroy releases the pinned buffer
  and the event). Commands run and results:

  - /usr/local/cuda/bin/nvcc -O3 -std=c++17 -Xcompiler -march=native
    -Xcompiler -pthread -gencode arch=compute_121a,code=sm_121a
    -DDS4_CUDA_HAVE_MXF4=1 -c -o /tmp/h3qa/h3_dit_gpu_sm121a.o
    native/h3/h3_dit_gpu.cu -> NVCC_SM121A_EXIT=0, a 499,360 byte object in
    2.4 s. These are the exact flags the tree emits for CUDA_ARCH=sm_121
    (Makefile:51-57, 408, 434-435), so the modified translation unit compiles
    as the Spark would compile it. This is a compile, not a run: no sm_121a
    device exists here, and the object was written to /tmp, never over
    native/h3/h3_dit_gpu.o.
  - make -n native/h3/h3_dit_gpu.o prints
    /usr/local/cuda/bin/nvcc -O3 -std=c++17  -Xcompiler -march=native
    -Xcompiler -pthread -c -o native/h3/h3_dit_gpu.o native/h3/h3_dit_gpu.cu,
    i.e. with no -arch at all when CUDA_ARCH is empty. That is the doc's
    quoted no-arch command, verbatim, produced by the tree's own rule.

  Correctness from the code:
  - the new fields are initialised: h3d_create calloc()s the context
    (:103-106), so stage is NULL, stage_done is NULL and stage_pending is 0
    before the first upload, and h3d_destroy's NULL guards (:126-128) and
    cudaFreeHost(NULL)/cudaEventDestroy(NULL) are safe;
  - the hazard the commit message names is handled: every piece waits on
    cudaEventSynchronize(stage_done) before refilling the pinned buffer
    (:195-204, the wait at :201), and because the flag survives between calls,
    the tail of the PREVIOUS upload is waited for too - that is precisely the
    loadstage corruption (first 8 MiB of every tensor after the first) the
    document records;
  - ordering is unchanged: the DMA is issued on context->stream, the same
    stream every kernel and h3d_read use, so stream order still gives every
    consumer its bytes; the event is recorded after each piece, so the wait
    bounds only the staging buffer, never the data;
  - narrow ABI: only h3d_upload and h3d_upload_at changed behaviour, the
    header is untouched, and the uploads in this port are issued from the
    load path of one context on one thread (h3d_upload/h3d_upload_at are the
    only entry points, called from crates/ds4-sys/src/h3_dit.rs:697 and :710;
    no upload path in crates/ds4-core/src/minimax_h3 spawns workers), so a
    shared staging buffer is not racing with itself;
  - the mechanism the comment cites is the official rule: CUDA Runtime API
    "2. API Synchronization behavior" states, under Asynchronous 3, "If
    pageable memory must first be staged to pinned memory, the driver may
    synchronize with the stream and stage the copy into pinned memory"
    (docs.nvidia.com/cuda/cuda-runtime-api/api-sync-behavior.html, read during
    this pass). The comment's "asynchronous rule 3" citation is correct;
  - arithmetic of the comment checks: 126,551 MB in 105.4 s is 1.20 GB/s, and
    the same bytes in 4.44 s are 28.5 GB/s.
  Not exercised: any GPU run of this path. The recorded measurement that
  justifies the deviation is the document's sm121-new row (staged upload plus
  hoist, arch-native, 8.2 s load, 417.2 s wall, output 74d862fe, identical to
  the reference). See ACCEPTED RISKS.

crates/ds4-core/src/minimax_h3/vae_video.rs

  Diff read in full, then read against the reference at /data/h3.c.

  - The reference allocates the activation set once per decoder, never inside
    the tile loop: allocate_activations is defined at h3_video_vae.c:397-421
    (note: line 396 is blank - the shipped comment's "396-421" is off by one)
    and called at :944 (the preview decoder's creation) and at :1089 inside
    h3_video_vae_decode (note: ":1088" is prepare_rope(...) &&, the call is
    the next line - also off by one). The new comment in vae_video.rs:688-695
    carries the same two numbers (on :689-690).
  - The reference's per-tile latent does stay per tile: prepare_input builds
    vae->latent at h3_video_vae.c:343 and the resident loop frees and rebuilds
    it for every tile (free_tensor(&vae.latent) then prepare_input at
    :1123-1124). run_tile keeps exactly that shape: it still allocates and
    uploads its own latent per tile (vae_video.rs:577-578) and only takes
    post, patch_hidden, zero and the Buffers from the scratch.
  - The zero suffix row: the reference allocates a tile-local zero tensor at
    h3_video_vae.c:535-541 and uses it once, as the SOURCE of the suffix copy
    at :559-561 ("pack tiled video VAE suffix"); h3_gpu_copy_f32 takes
    (destination, destination_offset, source, source_offset, elements), so the
    zero tensor is read, never written. In the port the same copy at
    vae_video.rs:611-616 passes scratch.zero as the source and nothing else in
    the file writes to it (grep of zero in vae_video.rs: only :705-706 the
    one-time alloc and upload, :614 the copy source). A constant that is only
    ever read is indistinguishable from the reference's fresh per-tile buffer,
    so hoisting it preserves the arithmetic. The deviation is named in the
    comment, as rule 0.2 requires.
  - Sharing is safe for the call order. run_tile ends with an explicit
    device(gpu.sync()) plus the projected readback before returning
    (:641-651), so a tile's computed results are complete on the host before
    the next tile rewrites post/patch_hidden; the post and patch_hidden
    buffers are written only at the head of each tile, after that sync. In
    decode() one scratch is built after the Decoder load and before both
    paths (:1135-1136), and it serves the two-token path (:1149) and every
    chunk of the loop (:1170s) sequentially; PreviewDecoder::decode builds its
    own scratch per call (:1071-1075) - per call, not per tile, which is the
    reference's own lifetime for the preview decoder (:944, once at creation,
    so the port is the more conservative of the two). Nothing shares a scratch
    across two Decoders, and each scratch's tensors belong to the one Gpu that
    made the Decoder.
  - Freeing less often does not change peak memory materially: the scratch
    holds the same set the previous code held for the duration of one tile,
    and the reference holds it for the whole decode at about 9 GiB resident
    (comment at h3_video_vae.c:530-532). The decoder already keeps its weights
    resident, so this is the reference's own budget.

crates/ds4-cli/src/bin/h3-rs.rs

  Diff read: an Instant around the single read_anchors call plus
  seconds={:.3} appended to the ANCHORS line.

  - read_anchors is still called exactly once. The CLI has one call site,
    h3-rs.rs:141, and the gating is unchanged: the timing wraps that one call
    (:140-142) and the println! is inside the same if !anchors.keyframes
    .is_empty() block (:143-149). The method of the same name in
    crates/ds4-core/src/minimax_h3/engine.rs:513 is a different function, the
    served path's, untouched by this diff.
  - No behaviour change: the only edit to the call is the Instant::now()
    before it and the elapsed() after it; read_anchors' arguments, order and
    error propagation (? on :141) are untouched.
  - Nothing parses the line. A repository-wide grep (excluding target and
    .git) finds ANCHORS in exactly four places besides this one: the docs
    (docs/minimax-h3/v1-local-gate.txt:573-579,
    docs/minimax-h3/post-port-phases.md:489-490, :565), the handoff
    (ds4-dfm-rs-handoff.md:57, :97), and two producers that print their OWN
    ANCHORS lines for other binaries - crates/ds4-core/examples/h3_dit_gate.rs
    :206 (keyframes= elements= fnv=) and tools/minimax_h3/dit_probe.c:732
    (same shape). No shell script, gate, test or Makefile target extracts a
    field from the h3-rs line or matches the word with an awk/sed/cut over its
    columns; tests/h3_perf_sweep.sh parses only TEXT/LOAD/DENOISE/AUDIO/FRAMES
    /VIDEO seconds= lines (its port patterns at :102-103). Appending a field
    to a line nothing consumes is inert.
  - The format is consistent with the document's new section, which prints
    ANCHORS 52.772 and 35.893: seconds={:.3} yields exactly that shape.

docs/minimax-h3/v1-local-gate.txt

  The worktree change replaces the 2026-10-25 entry with four 2026-10-26
  sections ("the load experiment re-read: the arch, not the copy path", "the
  arch split, measured shut", "the wall now adds up: the anchor phase has a
  mark", "TEXT is the drive, measured"). Each factual claim was traced to a
  file in this repo, a file in /data/h3.c, the committed recorded artifact, or
  a measurement the document itself states; the residue is listed below and in
  ACCEPTED RISKS. What checks out:

  - /data/h3.c/Makefile:25-26 is exactly "NVCC_ARCH ?= native" (25) and
    "NVCCFLAGS ?= -O3 -std=c++17 -arch=$(NVCC_ARCH) ..." (26). The claim that
    arch-native is the reference's own choice is exact.
  - The repo Makefile's warning block is lines 40-48 and says, verbatim,
    "compute_75 PTX, JIT'd onto the device with older codepaths: slower, and a
    different numeric profile than the arch-native SASS the tree's other
    objects carry" (44-46). "the Makefile's -include is silent" is line 49,
    "-include .ds4-cuda-config.mk".
  - The quoted no-arch build command is what the tree emits when CUDA_ARCH is
    empty - see the make -n output above. The whole causal claim of the new
    sections (later builds recompiled h3_dit_gpu.o without an arch, so the
    output hash followed the arch) is thus anchored in a command reproducible
    here, not only in the Spark's logs.
  - The committed record agrees where it can be checked. tests/h3/perf/perf.csv
    row 2 (port, clean-r1, 512x512, 56 frames, 8 steps, seed 7) is the table's
    "clean-r1 port, pageable, arch-native" row: text 29.509, load 116.067,
    denoise 279.819, audio 2.265, frames 78.447, wall 560.040, 720242 bytes -
    the table's 116.1 / 279.8 / 2.27 / 78.4 / 560.0 / 720242 to the digit.
    Row 1 (reference, clean-r1) is wall 441.522, 720242 bytes, same anchor and
    the same checkpoint bytes. Row 3 (reference, ref-phases, profile=1) is
    21.899897 load, 279.879493 denoise, wall 433.387, 718093 bytes, same
    binary sha256 d404df32... as row 1, same anchor, same checkpoint, same
    prompt hash. The new sections' reference figures (21.9, 279.9, 441.5,
    433.39, the 718,093 B / cea7cf1a profile outlier with the same identity
    columns) therefore all trace to the committed CSV, and the open profile
    question they describe is real in the record.
  - The timestamp arithmetic closes. The row's ts_utc is the run's end: the
    reference's ref-phases row ends 19:33:31Z and the log name the document
    cites (20261025T192618Z) plus the row's wall (433.387 s) lands on
    19:33:31Z. The repeat "at 2026-10-26T04:56:24Z, 9.5 h after clean-r1" is
    9 h 30 m after the port's clean-r1 row end (19:25:52Z) and 9 h 40 m after
    the reference's (19:16:32Z) - exact for the former, a loose rounding for
    the latter, and the sentence does not say which row it means.
  - The 2026-10-25 section that this replaces is consistent with the new text
    on every number it repeats (loads 116.1 / 106.7 / 23.6, walls 560.0 /
    626.0 / 467.7, hashes 74d862fe / 7991a015).
  - h3.c:691 and h3.c:1307 resolve: :691 is the "video VAE encoder" progress
    emit of h3_video_encoder_progress_bridge and :1307 is the
    h3_video_vae_encode call in the image-conditioning path, i.e. the anchor's
    own encode. main.c:501 is "case OPT_PROFILE: profile = 1; break;" and
    main.c:534 is the setenv("H3_PROFILE", "1", 1), so "--profile only sets
    H3_PROFILE" is exact.
  - The H3_PROFILE list is substantially right: h3_gpu_cuda.cu:659-670 is
    h3_gpu_profile_mark (a cudaStreamSynchronize plus a print), and
    h3_dit.c:1232, :2630, :2636, :2891, :3048, h3_video_vae.c:922, :1069 are
    the uses the document lists. The claim "every H3_PROFILE use in the engine
    is a stream sync or a print" holds for the two it omits as well -
    h3_gpu_cuda.cu:310 is the teardown print in h3_gpu_free and h3_gpu.m:176
    is the Metal backend's equivalent; there is also tests/test_cuda_runtime.c
    :101, a test - but the enumeration is incomplete as written.
  - The harness-gap bullet is right about the mechanism and the cells: the
    committed ref-phases row leaves audio_s, vae_s and mux_s empty, and the
    script's reference patterns (tests/h3_perf_sweep.sh:112-117) do require a
    trailing "<number>s$", which the reference's teardown line ("video VAE
    decoder: 43.256265s GPU, ...") does not satisfy. Its trailing clause is
    too generous for one of the three: the reference prints no mux timing at
    all (h3_ffmpeg.c and main.c only print "h3: wrote <path>"), so for mux the
    number is not in the log. "records the anchor's path but never its bytes
    and records no arch at all" is exact (the anchor column exists, only the
    prompt gets a sha256, and the script contains no arch or uname anywhere).
  - "The 320 rows carry dirty_files=0 because the sweep could not resolve the
    revision" is consistent with the script: GIT_REV falls back to "none" and
    GIT_DIRTY is wc -l over an empty porcelain when the binary's directory is
    not a git tree - the committed reference row shows exactly git_rev=none,
    dirty_files=0.
  - text_forward.rs:251-270 covers the eight-lane ring: the doc comment at
    :251-252, PREFETCH_LANES = 8 at :253 and prefetch_lanes() at :263-274,
    with the reference's clamp at h3_text_encoder.c:124-133
    (text_prefetch_threads, verified line by line). text_weights.rs:17 does
    say "the encoder reads 48.8 GiB of weights", and the document's own
    correction is arithmetically coherent (50.316 GB is 46.86 GiB), though the
    checkpoint headers behind that correction are on the Spark.
  - The ANCHORS/TEXT arithmetic adds up. 52.772 + 29.849 + 96.870 + 277.465 +
    2.184 + 74.254 + 0.331 = 533.725, printed as 533.7 of 534.7 s;
    35.893 + 28.939 + 106.310 + 119.231 + 2.274 + 64.322 + 0.354 = 357.323,
    printed as 357.3 of 358.2 s. The TEXT figures (28.9-29.8 s for 50.316 GB =
    1.69-1.74 GB/s, 92-96% of the 1.82 GB/s eight-lane rate, and "at most
    ~2.4 s of a 417 s wall (0.6%)") all recompute correctly, as does
    50.316 GB = 46.86 GiB.

  Findings that must be read with the document, all minor and none of them
  changing a conclusion:
  T1 the two off-by-one line citations above (h3_video_vae.c:396-421 should be
     397-421; :1088 should be :1089). They exist identically in the worktree
     Rust comment, so they are one source error copied twice.
  T2 the harness bullet's "though the numbers are in its log" is true for
     audio and video VAE and false for mux (no mux timing exists to extract).
  T3 the H3_PROFILE location list omits h3_gpu_cuda.cu:310 and h3_gpu.m:176.
  T4 "9.5 h after clean-r1" is exact against the port's clean-r1 row and 9.7 h
     against the reference's; the row is unnamed.
  T5 "The port's binary for that row was built ... from 7c68367a" reads
     against the row's own git_rev=6cdc1679, dirty_files=5. The commit
     timeline reconciles them (7c68367a committed 18:01:16 local, the build at
     18:16:44, the docs-only 6cdc1679 at 18:46:50 while the binary stayed
     unchanged), but the sentence only holds if the reader does that work.
  T6 the section sums 66.715 GB from three parts that add to 66.716 GB - a
     rounding artifact of three rounded figures, worth one word if the number
     is ever quoted alone.
  T7 the tables' rows beyond the three committed ones (loadfix, stage,
     base-vaehoist, vaehoist, loadstage2, loadstage, sm121, sm121-new,
     base-320, new-320, det-320a/b) are not in tests/h3/perf/perf.csv as
     committed here, and the replacement text dropped the previous entry's
     pointer to where they live (it said tests/h3/perf/perf.csv; they are in
     that file's Spark-side copy). The numbers therefore have no repository
     record here, only the document's word - which the previous entry's
     closing line used to name.
  T8 "Five different DiT loaders" is a rhetorical grouping: the enumerated
     five include the staged upload with and without the VAE hoist (one
     loader) and the 320-shape pair (two configurations of the same code).

A DEFECT OBSERVED OUTSIDE THE FOUR SURFACES: THE SERIALIZED WORKSPACE TESTS ON
THIS HOST

  Command: cargo test --workspace --locked -- --test-threads=1, the exact
  command of the CI job's Serialized workspace tests step. Result: exit 101.
  52 test targets reported ok; crates/ds4-perf failed two tests -
  artifact::tests::verification_rejects_drift_and_conflicting_digest
  (crates/ds4-perf/src/artifact.rs:449) and
  artifact::tests::operation_reuses_shared_inputs_across_artifacts (:489).
  Thirty consecutive isolated runs of the first: 28 failed, 2 passed.

  Mechanism, measured here. Both tests rewrite a file with the SAME byte length
  ("first" -> "other" is 5 -> 5 bytes at :441/:448; "same input" -> "new
  input!" is 10 -> 10 at :469/:488) and then require Verification to notice.
  Verification caches the digest against a FileStamp of size, mtime and ctime
  and the unix identity (artifact.rs:108-136) and returns the cached digest
  when that stamp is unchanged (:163-172). On this host the kernel hands out
  coarse file timestamps: emulating the test's own three-step sequence 30
  times, the size+mtime+ctime+inode tuple was identical before and after the
  rewrite 15 times, and a printed pair is byte-identical in all of them
  (2026-10-26 07:07:14.462926433 +0100 before and after, same inode, same
  size). Inside one timestamp granule a same-length rewrite is invisible, so
  the cached digest is reused and the assertion fails. With a 0.2 s gap
  between the writes the stamps differ, which is why this is timing-dependent
  and why the test's own faster in-process sequence fails more often than a
  shell emulation.

  It is not caused by and not fixable from these surfaces: crates/ds4-perf is
  untouched by the worktree (git diff fork/feature/minimax-h3 -- crates/ds4-perf
  is empty), the failing code landed in a48bea84 (2026-09-17, PR 53) and its
  blob is identical at 7ed11690, and the GitHub runner passed this very step at
  this very head (run 40232884549, Serialized workspace tests, success,
  20:56:52Z-21:00:14Z). The same command that fails ~93% of the time here
  succeeded there, so the failure is host-specific (this machine's coarse
  filesystem timestamps), not a defect of the change under QA and not a
  blocker for it. It is recorded because it is real, reproducible here, and a
  fragile test by any reading; the fix belongs to a ds4-perf unit (compare the
  bytes when the stamp is ambiguous, or give the rewritten files different
  lengths/identities). Nothing in this pass may edit it: this pass writes only
  this report.

ACCEPTED RISKS

R1  The Spark is unreachable and no GB10/sm_121a device exists on this host
    (RTX 4070 SUPER, sm_89). Every number in the new document sections - the
    load/denoise/audio/frames/wall tables, the binary shas 325640ec / 283a147c
    / 06b69a90, the forced rebuild, det-320a/b, the drive lane table, the PSNR
    and audio MD5s - is citable but was not re-measured here. What was checked
    is the mechanism underneath (the exact no-arch command, the exact sm_121a
    compile flags, the CSV rows, the arithmetic) and the command surface. No
    x86 number is offered as a target measurement.
R2  No h3 render, gate or fixture can run here: no checkpoint, no
    /mnt/h3-artifacts, no ~/ds4-h3-port, no sm_121a device, and
    tests/h3/perf/runs/ is empty, so no MP4 or AAC stream can be re-hashed.
    The native and VAE changes were verified by compile, by reading, and
    against the reference source, not by a model run.
R3  GitHub Actions cannot be executed locally and its job log needs
    credentials (HTTP 403). The workflow was verified by an exact-indentation
    structural read (no YAML parser exists here - said plainly rather than
    implied) plus the live run and step timings from the API; the printed
    ffmpeg/ffprobe version lines were verified in the file, not in the
    runner's log.
R4  The reference engine is read at /data/h3.c; the Spark's copy could differ.
    Every citation the new sections make resolves exactly here except the two
    off-by-one line numbers named under T1.
R5  crates/ds4-server/src/h3_video.rs was exercised only through its unit
    tests; the served surface was not started (no model, no h3-server-rs
    build) and the CI green at 7ba78b8a is cited from the API, not re-run
    here.
R6  The ds4-perf failure above cannot be fixed from this pass by construction
    (report only), and it is expected to keep failing on hosts with coarse
    file timestamps.
R7  The 2026-10-25 document section that 7b9e2f1e pushed is no longer in the
    file; it was verified through git show and through the numbers the new
    text repeats, not as live text.

verdict: overall PASS
