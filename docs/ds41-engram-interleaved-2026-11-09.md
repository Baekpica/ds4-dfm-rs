# DS4.1 interleaved engram sidecar (2026-11-09)

Question: the engram reads one 256-byte weight span and one 8-byte scale span
per row from two planes ~98.3 GB apart, so every row costs two aligned reads.
Does interleaving them into one 264-byte record, and halving the read count,
buy a measurable prefill gain on a served request?

## Method

Layout. One derived file, `ds41-engram-interleaved.bin`, inside `--engram-dir`:
a 32-byte fixed header (`DS41ENGM`, version, layer count, head dimension,
record size, total length), one 40-byte layer record per engram layer
(row count, byte offset, and the source file's name hash, length and mtime),
then zero padding to the first 4096-byte boundary and one record per row.
A record is the row's 256 weight bytes followed by its 8 scale bytes, both in
the order the two planes already store them. `SIDECAR_MAGIC`,
`SIDECAR_FIXED`, `SIDECAR_LAYER` and the record math are in
`crates/ds4-core/src/engram.rs:406-411, 458-601`; `record_off` is the ring's
per-row address (`:545`), `read_records` the bulk form (`:552`), and
`matches` refuses a sidecar whose head dimension, row counts or source
len+mtime disagree with the wire (`:596`). Reads stay O_DIRECT with the
4096-byte alignment the two-plane path already uses.

Packer. `crates/ds4-core/examples/ds41_engram_pack.rs`, run on the Spark:

```
cargo build --release -p ds4-core --example ds41_engram_pack
./target/release/examples/ds41_engram_pack <model.gguf> <engram_dir> <out.bin> [--plan|--check]
```

It reads 4096 rows per chunk, drops each chunk from the page cache
(`POSIX_FADV_DONTNEED`), writes `<out>.partial` and renames at the end, and
pads the file to a 4096 multiple so a bounce read never runs past EOF. `--check`
re-reads every record and compares it against the two source spans.

Loader. `V41Feed::open_sidecar` (`crates/ds4-core/src/ds41_run.rs:519`) opens the
file when it exists and matches the wire, logs
`engram rows: interleaved sidecar <path> (2 layers, 202.8 GB)`, and the round
then issues one `req` per row (`fill_round`, `:1145`) or one `row` call per row
(`serial_round`, `:1228`). Dropping the file is the opt-in; `DS41_ENGRAM_PLANES=1`
forces the two planes and is the control below.

Measurement. One window on the Spark, binary
`7434debb13340f66bb89ff68fde7c753d09b507fb3a630ef7dd1621ade6e992a`
(`make ds4-server CUDA_ARCH=sm_121` in a scratch tree), four boot arms of
`speed-bench/ds41_spark_sweep.sh` through `tools/host_memory_guard.py`
(`--max-gib 128 --reserve-gib 2 --trip-gib 2`), frontiers 2048 and 8192,
128 generated tokens, `--mtp-mode off --mem-floor-gb 2 --no-update-check`,
boot 208-210 s per arm. Arms ran sidecar, planes, planes, sidecar, so each
layout has one run in each order. Every arm booted its own server and was
killed before the next; the live server (`139aef25...`) was restored after
the window, pid 196200 on 8008. Observed clocks 2450-2470 MHz, the same band
in all four arms.

## Results

| frontier | prefill tokens | sidecar prefill | planes prefill | prefill gain |
|---|---|---|---|---|
| 2048 | 2149 | 1058.2 / 1054.0 tok/s | 967.1 / 970.6 tok/s | +9.0% |
| 8192 | 8449 | 1199.9 / 1197.9 tok/s | 1088.1 / 1082.7 tok/s | +10.5% |

| frontier | sidecar TTFT | planes TTFT | delta |
|---|---|---|---|
| 2048 | 2.0335 / 2.0416 s | 2.2249 / 2.2169 s | −0.183 s (−8.3%) |
| 8192 | 7.0501 / 7.0625 s | 7.7739 / 7.8128 s | −0.737 s (−9.5%) |

Decode and reuse in all four arms: 30.2 tok/s, `tok_per_step` 1.00,
`prefill_cached_tokens` 0.

## Reading

- The sidecar halves the round's read count and it shows end to end: prefill
  +9.0% at 2048 and +10.5% at 8192, TTFT −0.183 s and −0.737 s. Within-arm
  spread is 0.17-0.50%, so the gain is 18-20x the observed noise.
- The one-off reading was wrong. A single block-0 round would have produced a
  roughly constant saving at both frontiers; the saving instead scales with
  the prompt, 85 µs per 1,000 tokens at 2048 and 87 µs at 8192. Every
  2048-token block pays its own round, and the cross-block prefetch hides only
  part of it, so the sidecar cuts what stays exposed on each block.
- Decode is untouched, as expected: the engine's engram prep is 0.101 ms per
  decode round, and both arms measured 30.2 tok/s and `tok_per_step` 1.00.
- The control is the same binary, the same artifact and the same rows; the
  arms differ only in which file the feed reads, so the comparison isolates
  the read count and the layout, not the arithmetic. Completions are not
  compared here because no token differs: the feed hands the same bytes to
  the forward in both arms.

## Limits

- Both arms ran `--mtp-mode off`. Prefill and TTFT are the measured columns and
  MTP does not enter them; the decode columns record the configuration only.
- Two frontiers, one request each, two boots per layout. The repeat is an
  order check, not a sustained run.
- Clocks were observed at 2450-2470 MHz, above the 2200 MHz policy this repo
  targets. Both arms sat in the same band; the paired delta is the claim, not
  the absolute rates.
- The sidecar costs 202.8 GB of disk beside the 190 GB two-plane store, and
  `/` went from 326 GB to 137 GB free. It is opt-in by file presence; nothing
  removes the planes.

## Verdict

Adopt as an opt-in derived file: +9.0-10.5% prefill and −0.18 to −0.74 s TTFT
on every request that pays a full prefill, with decode unchanged, at the cost
of a one-time 202.8 GB copy and a loader that accepts both layouts. The
control (`DS41_ENGRAM_PLANES=1`) keeps the two-plane path one environment
variable away.

## Evidence

Spark: `/tmp/l3/pack.log`, `/tmp/l3/check.log` (packer and byte check),
`/tmp/l3m/{sc-sidecar,pl-planes,pl-2,sc-2}.csv` and `.log`, per-arm server
logs at `/tmp/l3m/<tag>/server.log`, binary `/tmp/l3m` build log
`/tmp/l3m/build.log`. Sidecar
`/home/leandro/models/ds41-1017-engram/ds41-engram-interleaved.bin`,
202,758,037,504 bytes, sha256
`cd606a9016ee27f4d2b5c9ec6da1f6beefa4bcdfa3cc4d93c5bd288aeec03ade`;
768,022,850 records, 0 mismatched against the two planes. Sources: layer 0
`model-00047-of-00048.safetensors` weights at 664, scales at 98,305,579,672,
384,006,168 rows; layer 1 `model-00048-of-00048.safetensors` weights at 672,
scales at 98,308,271,264, 384,016,682 rows; head dimension 256, record 264 B.
