# Tencent Hy4-flash MXQ

The CUDA family loads the `MQ-Q5-SSD-ENGRAM-FP8` or `MQ-Q5-Q8-SSD-ENGRAM-FP8` release directory or its first GGUF shard. Keep the four shards, the BF16 token embedding, the Engram directory and the template assets together. The loader checks the release manifest, the original model revision and tensor layouts; the manifest digest selects the recipe. MQ-Q5-Q8 derives from MQ-Q5: its BF16 dense projections and output head are Q8_0 and its routers BF16, everything else byte-identical. Decode reads 2.28 GB less per token. To verify the weight contents, run `sha256sum -c SHA256SUMS` inside the release directory. [MQ-Q5](https://huggingface.co/Baekpica/Hy4-flash-Mixed-Quant-GGUF/tree/94cdace56aaa0afc463311e31a2bbc3e613dcce2/MQ-Q5-SSD-ENGRAM-FP8) · [MQ-Q5-Q8](https://huggingface.co/Baekpica/Hy4-flash-Mixed-Quant-GGUF/tree/9a32b58fc7cc32d3e0488ddb916efd05cdf33cd3/MQ-Q5-Q8-SSD-ENGRAM-FP8) · [Tencent source](https://huggingface.co/tencent/Hy4-flash/tree/d88c9dfd96ba4f2e6b92adba0b8e8f1c8696f959)

## Weight placement

| Part | Size | Access |
|---|---:|---|
| Four Q4_K/Q5_K/Q8_0 and original-precision trunk shards | 98.91 GiB (MQ-Q5), 96.79 GiB (MQ-Q5-Q8) | Resident on the GPU |
| Four Engram tables | 61.41 GiB | SSD page reads at layers 1, 11, 21 and 31 |
| BF16 token embedding | 0.69 GiB | Host reads the sampled and prompt rows |
| SigLIP2 vision encoder and projector | 0.87 GiB | Host mapping, weights staged per layer during visual requests |
| DFly2 Q8_0 / BF16 | 0.95 / 1.58 GiB | One optional `--mtp` file |

Each Engram lookup hashes the current token and its two preceding tokens through Tencent's vocabulary map. Each layer reads 16 records of 96 E4M3 values and a BF16 scale. A 4 KiB page holds 41 records. On Linux the reader reads each distinct record of a prefill chunk once, through one io_uring round that keeps 512 aligned reads in flight. A prefill that starts with an idle graph reads each table at its layer, overlapping the later reads with GPU work. Consecutive chunks can cover a bulk read of all four tables. Token history crosses chunk and turn boundaries.

## Serving

```sh
model=MQ-Q5-Q8-SSD-ENGRAM-FP8   # or MQ-Q5-SSD-ENGRAM-FP8
./ds4-server --cuda -m "$model" --check-config
./ds4-server --cuda -m "$model" --host 127.0.0.1
```

The source context is 1,048,576 tokens. Twelve layers keep a 512-value sparse latent, 64 rotary values and one 128-value IndexPool key per four tokens. The other 30 layers keep a sliding window of 512. The default sparse cache uses E4M3 plus a power-of-two scale per latent row, with BF16 rotary and pooled key values. `DS4_HY4_KV=bf16` selects the BF16 latent reference. The prefill workspace is bounded by the chunk size. The engine evaluates IndexPool over tiles of 1,024 groups.

Text banks use the common `--max-seqs`, `--prefix-reuse`, `--kv-disk-dir` and `--kv-disk-space` options. Disk snapshots include each bank's committed token history, sparse rows, sliding window, incomplete IndexPool group and DFly2 frontier. A restored window can serve an extension, and a live bank can rewind while its ring still holds the requested history. Visual requests use the serial lane.

DFly2 shares the target embedding, output head and the caches of target layers 25, 29, 33, 37 and 41. The drafter proposes up to five tokens. The target verifies them and retains the accepted prefix. Select the Q8_0 or BF16 file with `--mtp` and set the server's `--mtp-mode on --mtp-draft N` for a forced run. Draft tokens rank their candidates through a Q8_0 copy of the output head that the engine builds at load (0.37 GiB, priced in the plan); the verify keeps the BF16 head. `DS4_HY4_DRAFT_HEAD_Q8=0` drafts on the BF16 head without the copy. MQ-Q5-Q8's head is Q8_0: drafts and verify read it directly.

The shipped Jinja template controls thinking and grouped tool calls. The server frames tool calls for the client. Image and MP4 video inputs use base64 payloads, the release's 16-pixel patches, 2×2 spatial merge, learned row terminator and interleaved temporal, height and width rotary axes. With the shipped video configuration every decoded frame is used. Linux video input requires FFmpeg and FFprobe with the `fd` input protocol.

## DGX Spark serving checks

The fa23eddb build served one native-context history with Q8_0 DFly2 at depth two, disk KV and 2,048-row chunks. A 1,048,542-token HTTP request returned the exact reference string planted around token 252,065. Its cold prefill averaged 798.6 tok/s (1,314 s) and its eight output tokens 21.9 tok/s; at least 5.29 GiB stayed available, with memory PSI at 0. On the previous binary a restarted server read 2,048 of 3,467 prompt tokens from disk KV and answered in 1.7 s. On GB10 this plan needs `DS4_CUDA_NO_Q8_ALIGNED=1`: beside the 4.13 GiB of aligned Q8 projection copies the history left ~3 GiB free and memory pressure stalled requests. Without them it opens at the default 4 GB floor and keeps ~5.3 GiB free beside the 0.37 GiB draft head; the throughput records use the aligned copies at 64K context.

MQ-Q5-Q8 passed the same 11 items with the 2,904-token prompt on the 2f6f75ca build. Its 1M serial plan quotes 120.13 GB against MQ-Q5's 122.80 GB without the aligned copies, which grow to 5.76 GiB.

At 8,192 context with two text banks, a client received English and Korean answers, distinguished red and blue stills and identified a red two-frame MP4. A streamed weather tool call contained the function name and Seoul argument. The client returned a tool result and received an answer using its temperature value. An identical 1,414-token request reused 1,413 tokens. Its continuation reused 1,415.

On the release binary, `ds4 --serving-check` with a 2,904-token prompt passed all 11 items at 4,096 context: prefix, rewind and continuation reuse, session disk and snapshot restores, warm, forked, partial and disk-restored banks, two banks at once, and the DFly2 target rows. With the verify rows' BF16 projections on the one-row arithmetic, the drafted greedy and seeded 16-token streams both equal one-row decoding. The prompt must exceed one 2,048-row chunk. A 718-token prompt splits at 359, off the prefill grid, and the bank items, which compare greedy streams, then measure two chunkings (6 of 16 tokens agreed).

At 4,096 context, the earlier 718-token checklist with 512-row chunks exercised common-prefix rewind, re-rendered continuation, session and bank disk restores, snapshots, a fork and two simultaneous banks. The Q8_0 DFly2 accepted three-token blocks at draft depth two. Replaying the same verified target rows reproduced 16 greedy draws and eight seeded draws with their RNG handoff. The one-row greedy stream matched. The one-row seeded stream diverged on the sixth token in the original fixture. At that frontier both RNG states agreed and the logits differed by up to 0.9204. Use the [quality report](hy4-optimization-2026-11-19.md) for comparisons across arithmetic paths.

## Reproducing the checks

`make gate-hy4` runs the CUDA attention and rotary arithmetic check, processor tests and output-protocol fixtures. Set `HY4_MODEL_PATH` to either release directory for `cargo test -p ds4-core --test hy4 -- --ignored` and `cargo test -p ds4-core --lib hy4 -- --ignored`. These tests read the released template, tokenizer, weight metadata and selected Engram records.

`make tests/hy4_quality` builds the teacher-forcing gate. Run `tests/hy4_quality "$model" cases.json result.jsonl` under the [memory guard](host-memory-guard.md). Each case has a name in `case`, an array of token `ids` and sorted `positions`. Each position specifies the zero-based index of the last consumed token, the reference `top1` next-token ID and the reference `nll` for the actual next token. The output retains both reference and measured values for each position.

An optional fourth argument names a DFly2 sidecar and a fifth the draft depth (default two). On consecutive positions the gate feeds the reference token into a trial and commits that token alone. `verify_rows` records the target width. For serving, `ds4 --serving-check` also checks the sampled target rows, RNG handoff, snapshot and stop inside an accepted block, and reports greedy and seeded agreement with one-row decoding. [Measured throughput and quality](hy4-optimization-2026-11-19.md).
