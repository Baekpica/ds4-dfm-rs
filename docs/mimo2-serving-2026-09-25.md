# MiMo serving on DGX Spark, 2026-09-25

This gate uses `MQ-IQ2-XXS-XS-Q8-MM-BF16` on one GB10, CUDA, and the
300–2200 MHz clock range. MTP and DFlash are off. The VMM owner serves
the base GGUF with `--no-repack-q8-aligned`; routed IQ2 aligned weights
remain enabled. The worker uses `DS4_MIMO2_PREFILL_CHUNK=2048`.

The owner runs `ds4_weight_server --base <first mixed shard> --backend vmm
--scope base --no-repack-q8-aligned --manifest <manifest>`. The worker imports
that manifest and runs `ds4-server --cuda -m <first mixed shard> --vision
<BF16 projector> -c 262144 --max-seqs 2 --prefix-reuse partial --kv-disk-dir
<directory> --mtp-mode off`.

| Requested shape | Result |
|---|---|
| 512K, two text banks, projector, partial reuse, disk KV | Rejected by the memory fit with the default Q8-aligned owner. |
| 256K, same default owner and 4096 chunk | Rejected by the memory fit. |
| 256K, two text banks, projector, partial reuse, disk KV, Q8 repack off, 2048 chunk | Booted and completed the gates below. |
| 1M text, two banks, Q8 repack off | Rejected by the memory fit. |
| 1M text, one bank, Q8 repack off | Booted; a 540,022-token prompt returned the correct arithmetic answer. The near-1M frontier remains open. |

The 256K plan quoted 30.82 GiB against 34.10 GiB available at boot,
including two bank KV sets, shared scratch, a separate serial media graph,
checkpoint pool, and headroom. The owner imported 82.68 GiB of base and
aligned IQ2 weights; the projector added 2.56 GiB. Q8-aligned weights are
optional and additive; disabling that repack retains the raw Q8 path.

The mixed server advertised context 262144 and two banks. An image of Earth
returned “Earth”; a two-frame Earth video and a joint video/audio request
also returned “Earth”. A one-second silent WAV was accepted but answered
“No” to a silence question, so audio semantic quality is unproven. Two
simultaneous 300-token text requests reached two in-flight jobs, both
completed through the continuous lane. Their forwards share scratch and run
one at a time. Text still answered 9+1 after media requests. Batch failure,
memory census, and governor fault counters stayed at zero.

For two 21,622-token prompts sharing the opening text, the second request
reused 12,288 tokens from a live partial checkpoint and answered 3+3 as 6.
After a worker restart, a conversation continuing a stored 21,623-token
turn loaded that disk KV and computed only 23 prompt tokens before answering
2+3 as 5. Editing the prior user prompt after restart took the cold path;
disk partial reuse of edited prompts is not claimed. Media requests use the
serial lane, and their KV is not stored by the text-bank disk path.

The 1M worker answered `2+2` with `4` after 540,022 uncached prompt tokens
in 1,876.9 seconds. This single run crosses the prior 512K text boundary;
it is not a throughput sample. A same-worker continuation toward 1M is
pending. This gate does not establish full 256K mixed request depth,
throughput parity, or the 1M frontier. The prior 512K serial-text and
256K serial-media/DFlash gates remain separate workloads.
