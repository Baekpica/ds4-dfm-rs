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
| 512K, two text banks, projector, partial reuse, disk KV | Rejected by the memory fit with both the default Q8-aligned owner and the Q8-repack-off owner. |
| 256K, same default owner and 4096 chunk | Rejected by the memory fit. |
| 256K, two text banks, projector, partial reuse, disk KV, Q8 repack off, 2048 chunk | Booted and completed the gates below. |
| 1M text, two banks, Q8 repack off | Rejected by the memory fit. |
| 1M text, one bank, Q8 repack off | Booted; 540,022- and 1,040,506-token prompts returned the correct arithmetic answers. |

The final 256K plan quoted 30.82 GB (28.71 GiB) against 34.68 GB
(32.30 GiB) available at boot, including two bank KV sets, shared scratch,
a separate serial media graph,
checkpoint pool, and headroom. The owner imported 82.68 GiB of base and
aligned IQ2 weights; the projector added 2.56 GiB. Q8-aligned weights are
optional and additive; disabling that repack retains the raw Q8 path.

The mixed server advertised context 262144 and two banks. An image of Earth
returned “Earth”; a two-frame Earth video and a joint video/audio request
also returned “Earth”. A silent WAV and a 440 Hz tone WAV both answered
“Yes” to whether a tone was present. Audio input worked, but this contrast
failed semantic discrimination. Two simultaneous 300-token text requests
reached two in-flight jobs and completed through the continuous lane. Their
forwards share scratch and run
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
in 1,876.9 seconds. A second request on that worker answered `5+5` with
`10` after 1,040,506 uncached prompt tokens in 6,232.0 seconds. The second
request had zero cache hits, so it proves the near-1M single-request path,
not long-context reuse. Both ran through the continuous lane with zero
request failures, memory census faults, and governor faults. These are
functional runs, not throughput samples. They do not establish full 256K
mixed request depth or throughput parity. The prior 512K serial-text and
256K serial-media/DFlash gates remain separate workloads.

The long text runs used binary SHA256
`2059d6befa1dec0ae50a109c887a921dd1f37b9f3c267c597a9d8d4c4c7a24fa`.
The final binary SHA256
`fecfc601864231ae17398086466a8e05dfc59a0211c2f1faac346704fd38c2cf`
booted at 1M with one bank and answered a short request. It also passed
the 256K two-bank gates above: two 300-token requests reached two in-flight
jobs, 12,288 of 21,622 prompt tokens hit a live partial checkpoint, and a
post-restart continuation loaded 21,623 disk-KV tokens and computed 23.
Both final-binary plans had zero memory census and governor faults. Sampled
busy SM clocks during the long text run were 2145–2190 MHz.

Qwen's documented GB10 profile has two persistent 256K banks, qualified
partial reuse and disk KV, still images, embedded MTP, and FP8 SSD-PLE.
MiMo now covers the two-bank text/cache workflow at 256K with MTP off and
serial image, video, and audio inputs. This is a serving-contract comparison,
not a paired throughput or quality result.
