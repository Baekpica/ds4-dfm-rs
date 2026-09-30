# Naive near-capacity serving gates

DGX Spark/GB10, MQ87 revision `65b235a4`, shared VMM owner, main-only
workers, eager CUDA, chunk 2048, GPU clocks 300–2200 MHz. Active samples
record 2190 MHz. [Receipts](long-context-evidence.json) identify the binary,
fixture hashes, quotes, guard samples and failures. This is one distant-fact
retrieval fixture, not a broad long-context quality score.

## 256K, two banks

The cold seed processes 261888 input tokens and retrieves the exact access
phrase at token offset 44547. It emits 13 tokens and stops normally after
1271.889 seconds. A second bank serves a short arithmetic request while
that long request is pending before and after the probe. Neither request
falls back or reports a governor fault. The seed timing includes this
functional overlap probe and is not an isolated throughput measurement.

| Fresh-process request | Input | Cached | Output | Finish |
| --- | ---: | ---: | ---: | --- |
| Seed, streaming | 261888 | 0 | 13 | stop |
| Buffered follow from disk | 261924 | 261901 | 65 | stop |
| Second disk restart, extended conversation | 262011 | 261901 | 65 | stop |

Both buffered requests return the exact phrase as `message.content` and
preserve separate `reasoning_content`. They restore the retained seed
ancestor, not an exact latest-generation history. Reuse is a real native
fork and the newly computed suffix is positive. Output budgets are 192
and 128; the final request plus its budget remains within context.

The quote records 6785528832 bytes per bank, 1536004352 shared scratch,
204472320 checkpoint pool and a 4-GiB floor: 19606501632 bytes total.
The owner remains resident and its weights receive shared-import credit.
Disk capacity is 32 GiB with a 1024-token persistence threshold; disk KV
does not replace active bank memory. Minimum available host memory over
the complete initial worker lifetime is 17.679 GiB. PSI reaches 3.72
including disk persistence; the memory guard stays live.

## Preserved output failure

The original streaming follow uses a 64-token budget. The model emits
unframed reasoning, `</think>`, and the answer; the warm run needs 65
tokens, so that request ends at `length` with a truncated phrase. It fails
the original exact-content/normal-stop gate. It has not been relabeled.

With the same messages and a 192-token budget, warm and fresh cold streams
both stop and recover the complete fact. Their only raw-text difference
is a leading newline: 65 versus 64 completion tokens. This is not
byte-identical generation or a full-logit parity proof. Width-dependent
native arithmetic is a possible explanation; these outputs do not isolate
it. The fresh cold comparison overlaps host checks and its elapsed time
is excluded from performance claims.

The buffered path already separates reasoning at `</think>`. No native
cache or generated-message parser was changed for this gate. The original
stream request, failed receipt, extended cold/warm outputs and both disk
continuations remain in the evidence. This qualifies the recorded buffered
continuation scope, not arbitrary disabled-thinking streams or budgets.

## 512K, one bank

The buffered cold seed processes 523264 input tokens and retrieves the
phrase at token offset 88983. It stops normally after 4056.039 seconds.
The follow and fresh disk restart also return the complete phrase and
stop normally. [Receipts](long-512k-evidence.json).

| Request | Input | Cached | Output | Finish |
| --- | ---: | ---: | ---: | --- |
| Cold seed | 523264 | 0 | 13 | stop |
| Warm follow | 523300 | 523277 | 119 | stop |
| Disk restart, extended conversation | 523441 | 523277 | 13 | stop |

The follow preserves separate reasoning. The restart restores the seed
ancestor and computes a 164-token suffix; it does not restore the latest
generated reasoning history. Both reuse traces report `exact` for that
ancestor. Content differs only in leading whitespace; this is not a
long-context full-logit or cold/warm generation parity proof. No request
reports a governor fault, fallback or speculation.

The quote records 13136753664 bytes per bank, 1636667648 shared scratch,
204472320 checkpoint pool and a 4-GiB floor: 19272860928 bytes total.
Disk capacity is 32 GiB with a 1024-token persistence threshold. Complete
worker lifetimes record minimum available memory of 17.656 GiB and
19.200 GiB, and maximum PSI of 4.16 and 4.48 respectively. Both workers
exit cleanly. An early restart attempted during the first worker's disk
flush is refused by the single-worker lock before model allocation; the
successful restart begins after verified cleanup.

Host tests overlap early seed prefill. Its elapsed time is functional
evidence, not a matched throughput result. This qualifies one buffered
retrieval fixture, continuation and disk restoration with MTP off.

## 1M, stopped incomplete

The first one-bank boot is stopped by the memory guard before readiness:
exit 75 at 9.714 GiB available memory and PSI 64.56. Its cause was not
established. A fresh boot with the same 29/28-GiB cgroup limits and
6-GiB reserve/trip limits reaches readiness. Sampled cgroup high, max and
OOM counters remain zero; cgroup accounting does not capture all CUDA UMA
allocation, so the whole-host guard remains necessary.

The buffered request submits 1047552 input tokens. The user stops the gate
after 210.9 monitored minutes of prefill, before any answer. Shutdown
returns empty content, zero output tokens and `finish_reason=error`.
Reported prompt/cache-write usage does not prove completed native prefill.
No retrieval, follow or disk-restored request passes at 1M.
[Receipts](long-1m-evidence.json).

The live quote is 25839203328 bytes per bank, 1837994240 scratch,
204472320 checkpoint pool and a 4-GiB floor: 32176637184 bytes total.
The second worker's complete lifetime records minimum available memory
7.233 GiB and maximum PSI 1.99; bounded host checks overlap early prefill.
The last live sample records zero governor/census faults and 2190 MHz.
The worker and guard exit cleanly after the requested stop. Port 8002 is
empty, only the shared owner remains on the GPU, and available memory
recovers to 35.493 GiB. This is incomplete functional evidence; the
qualified limits remain 512K/one bank and 256K/two banks with MTP off.
