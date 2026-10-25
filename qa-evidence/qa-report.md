Rule 19 QA report - MiniMax-H3 (h3) P12.5 harness, the two skills, the systemd
unit, the documentation revisions, the first completed served render, and the
performance sweep script (fifth pass, final)

repository : /data/ds4-dfm-rs
branch     : feature/minimax-h3
HEAD       : 6cdc1679de84d549d901723128dcff70ad436705
worktree   : uncommitted, as the unit has been throughout. Changed since the
             fourth pass: tests/h3_perf_sweep.sh (two defect fixes) and
             tests/h3/perf/README.md (the matching documentation). Nothing was
             committed, nothing was pushed, and only this report was written by
             this pass. No stub artifact is left anywhere: tests/h3/perf/runs/ is
             empty (0 files) and tests/h3/perf/perf.csv does not exist.
date       : 2026-10-25
qa model   : deepseek-v4.1-CC-flash
host       : x86_64 "vizzio", RTX 4070 SUPER (driver 610.43.03), ffmpeg/ffprobe
             7.1.5, nvidia-smi present. The Spark was read only through the
             operator's gvfs sftp mount; no command was sent to it.

Revision under test, stable for the whole pass (checked again at the end):
  tests/h3_perf_sweep.sh     md5 ffdcf87fe8a1db45a95909dba6ca9d6a, 246 lines,
                             mtime 20:11:44
  tests/h3/perf/README.md    3031 bytes, mtime 20:11:54


STATUS OF THE FOURTH PASS'S TWO DEFECTS AND THE README

D1 (a comma corrupted the row) - FIXED, and verified with the two cases the brief
   names.
   - Sanitization: tests/h3_perf_sweep.sh:181 builds the tag once,
     `tag=$(printf '%s' "${LABEL:-run}" | tr -c 'A-Za-z0-9_.-' '-')`, and the row
     uses that same tag for the CSV label cell (:224), so the file name and the
     label cannot disagree.
     Verified: `--label 'a,b"c'` produced a row with exactly 31 comma-separated
     fields and label cell "a-b-c", and the kept files were named
     "...-512x512-f56-s8-a-b-c.log/.mp4".
   - Anchor quoting: csv_field (:166-177) quotes a field that contains a comma, a
     double quote or a newline, per RFC 4180, doubling inner quotes; the anchor
     cell goes through it (:232).
     Verified: `--anchor /tmp/qa-sweep/kf,comma.png` wrote the line ending
     `...60be5faff9a6b8ced0948ec77606cc1690e4ddbae0e8d8b332916cb4a2e2,"/tmp/qa-sweep/kf,comma.png"`.
     A naive `awk -F,` sees 32 parts on that line; a real CSV parse sees 31 fields
     with the anchor intact (`python3 -c "csv.reader(...)"` ->
     proper_csv_fields=31, anchor=/tmp/qa-sweep/kf,comma.png, label=anch). The
     README (tests/h3/perf/README.md:45-47) says the label "can never shift a
     column" and that the anchor "is quoted when it contains a comma, which makes
     that one line standard CSV rather than plain comma-separated text" - so it
     documents the quoting and its purpose, but it does not say that a naive comma
     split sees one extra part (see O1).
D2 (a row could be lost silently) - FIXED, with the exit codes read directly after
   each command, never through a pipe.
   - The header write is checked (:146-150): `... >"$OUT" || fail "cannot write the
     header to $OUT"`.
   - The row is built into a variable and appended with an explicit check (:233):
     `printf '%s\n' "$row" >>"$OUT" || fail "cannot append a row to $OUT"`, and
     `fail` exits 1.
   - Read-only CSV: `--out` on a chmod 444 file gave exit 1, the message
     "h3_perf_sweep: cannot append a row to /tmp/qa-sweep/ro2/out.csv" (plus the
     shell's own "Permission denied"), and the file still contained only its
     original line "H".
   - `--out` pointing at a directory: exit 1, "h3_perf_sweep: cannot write the
     header to /tmp/qa-sweep/diro2" (plus the shell's "Is a directory"). The
     fourth pass saw exit 0 and "1 row(s)" in both cases; both are gone.
   - Two smaller improvements in the same revision, both verified: `bytes=$(stat -c
     %s "$out_file" 2>/dev/null)` now suppresses the stat error instead of leaking
     it to stderr (:215), and the peak-RSS comment (:217-219) now says the column
     is empty by construction and documented as such.
README - checked statement by statement against the script:
   - ":42-44 peak_rss_kb is empty by construction: the child is reaped before the
     row is written, so its high-water mark is gone. The column stays so the schema
     does not move." matches :217-219 and the unconditional `rss_peak=""` (:219);
     observed empty in every row produced this pass.
   - ":45-47 label sanitized to [A-Za-z0-9_.-] ... anchor ... quoted when it
     contains a comma" matches :181 and :166-177, :232, and both were reproduced
     live (above).
   - ":25-26 refuses to run without an executable and a checkpoint, writes the
     header once, appends nothing on a failed run, prints the kept path" - all four
     verified this pass (:122-123, :146-150, :197-202, :237-246).
   - The columns list at :30-34 names all 31 header fields in order, including
     `profile`; the header line itself carries 31 fields (:148).
   - ":6 tests/h3/perf/perf.csv one header, one row per run (tracked)" and ":7-9
     runs/* ignored" match .gitignore:290-291 (git check-ignore confirms both
     rules).


WHAT THIS PASS RAN

The new script, revision ffdcf87f, against stub binaries and a stub model root
under /tmp/qa-sweep (never the Spark), with every CSV and keep-dir under /tmp:
  - bash -n: exit 0.
  - --help: exit 0, prints the header block including the current defaults.
  - --dry-run: exit 0, printed the exact command (binary, -d model, -p prompt,
    shape, seed, --first-frame, -o <csv>), created no CSV and no CSV directory; it
    did create the keep dir, as before (O2).
  - CSV shape: header 31 columns, printf format 31 placeholders, and every row
    written had 31 fields (checked with `awk -F,` on each CSV produced).
  - Last matching line per phase: a log with two TEXT lines (11.111 then 24.694)
    gave text_s=24.694.
  - Empty, never zero: stubs without FRAMES/VIDEO lines gave empty vae_s and mux_s;
    the reference-format log (Qwen text encoder / load / denoise / audio VAE /
    video VAE / "h3: wrote") parsed to text 24.694, load 133.083, denoise 291.869,
    audio 2.318, vae 81.038 and left mux empty.
  - per_step_s = denoise_s / steps: 40.000/4 = 10.000; 291.869/8 = 36.484;
    40.000/8 = 5.000; with --steps 0 the cell is empty.
  - profile gating: without --profile-phases, H3_PROFILE=unset, no --profile in
    argv and profile=0 in the row; with it, H3_PROFILE=1, --profile in argv and
    profile=1. The reference's phase parser fills the cells in both modes.
  - Failure path: a stub exiting 3 gave exit 1, "run failed (status 3); log kept
    at ...", the .log present in the keep dir, the .mp4 removed, and a CSV holding
    only its header line.
  - Identity fields are read, not constants: binary_sha256
    186c8ae82af91591849254cca49063a02f6d79ad67741631a324ba3c94d05ac1 equals
    sha256sum of the stub; prompt_sha256 d7c660be... equals sha256sum of the
    default prompt; ckpt_bytes 0 equals du -sb of the stub model root; driver
    610.43.03 and the clocks are live nvidia-smi values (210/2505/375 MHz across
    runs); the cache cells are /proc/meminfo Cached and move between runs; and with
    a repo-local binary git_rev 6cdc1679 = `git rev-parse --short HEAD` and
    dirty_files 12 = `git status --porcelain | wc -l` at that moment, sha
    abdb235d42 = sha256sum of that binary.
  - Header not overwritten: a pre-existing file with its own header and row was
    appended to; its two lines survived untouched and --repeats 2 added two
    31-field rows.
  - Refusals: unknown option, a non-executable --binary, a missing --model-dir and
    an unknown --engine each exit 1 with a named message.
  - D1 and D2 cases: as above.

The rest of the surface, re-run:
  - make tests/h3-suite from a clean invocation (the three copied binaries removed
    first): all three rebuilt and green, exit 0 -
      == tests/h3-test_host_portable   ok: portable RGB resize
      == tests/h3-test_av_mux          ok: concurrent FFmpeg video/PCM pipes created
                                          /tmp/h3-av-mux-test.mp4 (48065 bytes)
      == tests/h3-test_tokenizer_portable  ok: portable ICU byte-level BPE tokenizer
  - cargo fmt --all --check: exit 0.
  - diff -r, all four pairs IDENTICAL: skills/h3-video and skills/h3-prompt-writing
    against their ~/.opengrok/skills and ~/.claude/skills copies.
  - Cleanup: every run of this pass used --out and --keep-dir under /tmp, and
    tests/h3/perf/runs/ ends this pass empty (0 files); no .log, .mp4 or perf.csv
    was created in the checkout.


COVERED SURFACES

FILE tests/h3_perf_sweep.sh - read in full (246 lines) and exercised in eighteen
  invocations; bash -n and --help clean; the whole check list above passes; both
  fourth-pass defects are fixed. PASS.
FILE tests/h3/perf/README.md - read in full; every statement it makes about the
  script was checked against the code and the live runs (sanitized label, quoted
  anchor, peak_rss_kb by construction, header once, no row on a failed run, the
  31-column list, the ignored run media). PASS, with O1.
FILE tests/h3/perf/runs/ - empty at the end of this pass (0 files) as required;
  the directory itself is the script's default keep dir and is git-ignored
  (.gitignore:290). PASS.
FILE .gitignore - /tests/h3/perf/runs/ and /tests/h3/artifacts/ ignored,
  tests/h3/perf/perf.csv deliberately not (it is the record), the earlier
  /tests/h3-*, /h3-rs, /h3-server-rs rules unchanged. PASS.
FILE tests/h3/test_host_portable.rs, tests/h3/test_av_mux.rs,
  tests/h3/test_tokenizer_portable.rs, fixtures/tokenizer_portable_fixture.json,
  Makefile targets tests/h3-<name> and tests/h3-suite - unchanged since the third
  pass; rebuilt and green again today. PASS.
FILE crates/ds4-core/Cargo.toml, crates/ds4-core/examples gating - unchanged; the
  default-build gating was verified in the first pass and nothing has moved since.
  PASS.
FILE skills/h3-video/{SKILL.md,AGENTS.md,scripts/*}, skills/h3-prompt-writing/
  {SKILL.md,AGENTS.md} - unchanged since the fourth pass; all four installed pairs
  are diff -r identical to the repo. PASS.
FILE tools/minimax_h3/systemd/{h3-server-rs.service,env.example} - unchanged; the
  systemd-analyze result and the flag cross-check stand from the first pass. PASS.
FILE tools/minimax_h3/standin-renderer.sh - unchanged since the fourth pass; still
  referenced by nothing (O4).
FILE docs/minimax-h3/v1-local-gate.txt, docs/minimax-h3/local-v1-contract.md -
  unchanged since the fourth pass; the retraction, the solo reading and the
  concurrency lesson were verified then and nothing has moved. PASS.
FILE tests/h3/artifacts/ - unchanged; the served clip and both CLI renders are
  still there with their recorded hashes (O5).
FILE qa-evidence/qa-report.md - this file.


DEFECTS

None. Both fourth-pass defects are fixed, the fixes are verified with the exact
cases the brief names, and the README that documents them matches the script.


OBSERVATIONS

O1  tests/h3/perf/README.md:46-47 documents the anchor quoting and calls the
    result "standard CSV rather than plain comma-separated text", but it does not
    spell out the consequence the brief asked about: a naive comma split of such a
    line sees one extra part (32 instead of 31). That is inherent to RFC 4180
    quoting, not a defect, but a reader who counts commas by hand or with
    `awk -F,` on an anchored row will mis-count; one clause would close it.
O2  --dry-run creates $KEEP_DIR (mkdir -p at :95, before the dry-run exit at
    :131) and creates nothing else: no CSV, no CSV directory, and no binary run.
    Harmless, but not literally "nothing".
O3  A binary that exits 0 without writing its output still yields a normal row,
    with an empty output_bytes and (now silenced) stat error, reported as success.
    The tool cannot tell "rendered nothing" from "rendered"; if that matters, the
    row could carry the file's absence as a value instead of an empty cell.
O4  tools/minimax_h3/standin-renderer.sh is still referenced by no document,
    Makefile target or test; the P15.5 gate record describes the wire run without
    pointing at a repo-local stand-in. Carried from the fourth pass.
O5  tests/h3/artifacts/croc-preview.mp4 (the port's CLI render) and
    croc-preview-ref.mp4 (the reference engine's render) remain byte-identical,
    md5 bda69c4025bc10f7e3cd1d1ffe17175b, as does the Spark's own
    ~/h3-api/out/ref-preview.mp4. A port/reference bit-identity on a full
    DiT+VAE+mux render is stronger than this project's stated acceptance criterion
    and is worth the operator's own confirmation; nothing in the artifacts
    distinguishes it from a copied file. Carried from the fourth pass.
O6  Makefile:552-563 - the "3 -> skip" branch is still unexercised: no test in
    H3_REFERENCE_TESTS can exit 3 yet. Carried from the earlier passes.


ACCEPTED RISKS

R1  The sweep script was verified against stub binaries, not the real engines:
    this host has no checkpoint, no h3-rs and no reference h3 binary. Every branch
    the script has was exercised (both engines, both phase parsers, profile gating,
    failure, repeats, dry-run, unwritable targets), with the identity columns taken
    from this host's real driver, clocks, memory and git state.
R2  No render was measured in this pass. The 22 min 8 s, the CLI phases and the
    byte counts remain the third pass's verified record, re-read here.
R3  The Spark's API is still down (curl to 192.168.1.91:8090 refuses the
    connection), so no live record query is possible; the Spark was touched only
    through the read-only gvfs mount.
R4  The MLX goldens under misc/fixtures are still absent from the reference clone,
    so the 16 artifact-dependent reference rows cannot be replayed here.
R5  The systemd unit's deployment paths and the Spark URL remain unverified for
    the reasons given in the earlier passes.
R6  /data/MiniMax-H3 is still gone from this host (its trash entry is empty), so
    the prompt guides and the reference repository's README cannot be read here;
    the restore sentence in both prompt-writing documents is the mitigation.
R7  The script's default --out (tests/h3/perf/perf.csv, :89) was deliberately not
    exercised, so that no stub row could land in that tracked record; the default
    is documented in --help and in the README, and the file does not exist yet.


verdict: overall PASS
