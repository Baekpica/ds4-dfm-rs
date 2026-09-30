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

512K and 1M results will be recorded only after their actual gates finish.
