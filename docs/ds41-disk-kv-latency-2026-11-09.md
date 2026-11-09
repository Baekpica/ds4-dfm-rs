# DS4.1 disk-KV store and load latency (2026-11-09)

Question: what does a disk checkpoint cost at 8K, 32K and 64K, and is a
known-address prefetch worth a unit?

## Method

Instrumentation: the existing `DS41_PHASE_LOG=1` host phase log
(`crates/ds4-core/src/ds41_run.rs`) gained four timers around the disk
path, and the two export sites report the payload size:

- `disk find` — the store index lookup, before any payload read.
- `disk import` — `ds4_bridge_v41_kvreuse_import`, the native load of the
  packed checkpoint into the run's lanes.
- `disk export` — the native pack of the in-memory checkpoint into the
  host save buffer.
- `disk store` — the `V41KvDisk::store` write of that buffer.

One window on the Spark, MTP auto (`tok_per_step` 1.78-2.00), binary
`bc17baf7b12bf2a7fa9123dfe2ba07d0df1fb5afe738d4e6e2da508e0b9150c3`
(`make ds4-server CUDA_ARCH=sm_121`), server booted with
`--kv-disk-dir /mnt/models/ds41-kvtest --kv-disk-space-mb 16384
--kv-cache-min-tokens 64`. Each context was requested twice with the same
prompt (`speed-bench/ds41_spark_sweep.sh`, 16 generated tokens): the first
request is cold and writes the checkpoint, the second is a full hit.
Clocks 2405-2457 MHz. The live server (`139aef25...`) was restored after
the window, pid 179063 on 8008.

## Results

Per-context request rows (cold a, hit b):

| ctx | prefill tokens | checkpoint | find | import | TTFT cold | TTFT hit | prefill cold | decode cold | decode hit |
|---|---|---|---|---|---|---|---|---|---|
| 8192 | 8449 | 25.3 MB | 0.6 ms | 4.7 ms | 7.826 s | 0.080 s | 1080.8 tok/s | 19.0 tok/s | 24.1 tok/s |
| 32768 | 32749 | 46.4 MB | 2.3 ms | 7.7 ms | 30.297 s | 0.110 s | 1082.1 tok/s | 15.2 tok/s | 25.3 tok/s |
| 65536 | 66608 | 75.7 MB | 4.0 ms | 75.8 ms | 60.211 s | 0.212 s | 1107.3 tok/s | 13.1 tok/s | 23.0 tok/s |

Store side, per newly written checkpoint (export + store):

| checkpoint | payload | export | store | total | export share |
|---|---|---|---|---|---|
| 8K prompt | 20.2 MB | 84.5 ms | 114.2 ms | 198.7 ms | 42% |
| 8K turn (x2) | 25.3 / 24.8 MB | 38.5 / 38.1 ms | 77.9 / 77.9 ms | 116.4 ms each | 33% |
| 32K prompt | 46.4 MB | 160.2 ms | 233.8 ms | 394.0 ms | 41% |
| 32K prompt, rewarm | 45.9 MB | 69.3 ms | 142.1 ms | 211.4 ms | 33% |
| 64K prompt | 75.7 MB | 130.3 ms | 253.3 ms | 383.6 ms | 34% |
| 64K prompt, rewarm | 75.2 MB | 125.7 ms | 251.3 ms | 377.0 ms | 33% |
| 32K / 64K turn | 45.6 / 74.7 MB | none | 2.3-5.2 ms | 2.3-5.2 ms | dedupe, already on disk |

## Reading

- The hit path removes the prefill: TTFT falls from 7.83 / 30.30 / 60.21 s
  to 0.080 / 0.110 / 0.212 s, a 97x / 275x / 285x cut. The load itself is
  4.7 / 7.7 / 75.8 ms for 25.3 / 46.4 / 75.7 MB, i.e. 5.4 / 6.0 / 1.0 GB/s.
- The 64K import is the outlier: 1.0 GB/s against 5-6 GB/s at 32K, ten
  times the cost for 1.6x the bytes. One observation only (one hit per
  context); the cause is not isolated. Candidates are a cold page cache
  after the fresh write and per-token host work inside the import. Next
  probe, if this matters: split the import timer native vs host and read
  the same checkpoint twice in a row.
- The store side is the larger cost and it runs on the request that
  produces the checkpoint: 199 / 394 / 384 ms for 20 / 46 / 76 MB, of
  which export (host pack + readback) is a third to 42%. A turn
  checkpoint that is already on disk dedupes to 2-5 ms.
- Decode looks lower in the cold row (19.0 / 15.2 / 13.1 vs 24.1 / 25.3 /
  23.0 tok/s) because the cold request also pays the checkpoint store
  inside the same window and because the hit row starts from a loaded
  state; the hit row is the honest decode figure at each context.

## Prefetch verdict

A known-address prefetch has no earlier point than the current find. The
checkpoint address is the prompt digest, so it exists only after the
request's prompt arrives; before that the server cannot name the file.
What remains is import speed, not prefetch: a parallel or mmap copy for
the 64K outlier, and `posix_fadvise`/`POSIX_FADV_WILLNEED` on the file
just written so the next read finds it warm. Neither is a unit on this
evidence: at 64K the load is 76 ms of a 212 ms TTFT and 0.13% of the
60 s prefill it replaces.

The lever that does show up is the store: 0.2-0.4 s paid synchronously by
the producing request. Moving `disk store` to a background writer removes
it from the request path; it does not change the bytes. That is the
candidate unit if disk-KV latency is picked up again.

## Evidence

Spark: `/tmp/l4/server.log` (phase lines), `/tmp/l4/sweep_{8192,32768,65536}_{a,b}.csv`,
`/tmp/l4/ds4-server.l4` (binary, sha256 `bc17baf7...`), `/tmp/l4/build.log`;
store directory `/mnt/models/ds41-kvtest` (316 MB, seven `.kv` files at
21-80 MB). Instrumentation source: `crates/ds4-core/src/ds41_run.rs`.

## Verdict

Disk KV is not a latency problem at 8K-64K: a hit costs 5-76 ms and
removes 8-60 s of prefill. The measured cost worth acting on is the
synchronous store (0.2-0.4 s at 20-76 MB, a third to 42% of it export),
which belongs on a background writer. The 64K import outlier (1.0 GB/s)
is recorded as unexplained and single-sample, not as a defect.
