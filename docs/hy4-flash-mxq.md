# Tencent Hy4-flash MXQ

The CUDA family loads the `MQ-Q5-SSD-ENGRAM-FP8` release directory or its first GGUF shard. Keep the four shards, the BF16 token embedding, the Engram directory and the template assets together. The loader checks the release manifest, the original model revision and tensor layouts. To verify the weight contents, run `sha256sum -c SHA256SUMS` inside the release directory. [Mixed-quant artifact](https://huggingface.co/Baekpica/Hy4-flash-Mixed-Quant-GGUF/tree/94cdace56aaa0afc463311e31a2bbc3e613dcce2/MQ-Q5-SSD-ENGRAM-FP8) · [Tencent source](https://huggingface.co/tencent/Hy4-flash/tree/d88c9dfd96ba4f2e6b92adba0b8e8f1c8696f959)

## Weight placement

| Part | Size | Access |
|---|---:|---|
| Four Q4_K/Q5_K/Q8_0 and original-precision trunk shards | 98.91 GiB | Resident on the GPU |
| Four Engram tables | 61.41 GiB | SSD page reads at layers 1, 11, 21 and 31 |
| BF16 token embedding | 0.69 GiB | Host reads the sampled and prompt rows |
| SigLIP2 vision encoder and projector | 0.87 GiB | Host mapping, weights staged per layer during visual requests |
| DFly2 Q8_0 / BF16 | 0.95 / 1.58 GiB | One optional `--mtp` file |

Each Engram lookup hashes the current token and its two preceding tokens through Tencent's vocabulary map. Each layer reads 16 records of 96 E4M3 values and a BF16 scale. A 4 KiB page holds 41 records. On Linux the reader coalesces duplicate pages across a prefill chunk and submits aligned reads through io_uring. Token history crosses chunk and turn boundaries.

## Serving

```sh
model=MQ-Q5-SSD-ENGRAM-FP8
./ds4-server --cuda -m "$model" --check-config
./ds4-server --cuda -m "$model" --host 127.0.0.1
```

The source context is 1,048,576 tokens. Twelve layers keep a 512-value sparse latent, 64 rotary values and one 128-value IndexPool key per four tokens. The other 30 layers keep a sliding window of 512. The default sparse cache uses E4M3 plus a power-of-two scale per latent row, with BF16 rotary and pooled key values. `DS4_HY4_KV=bf16` selects the BF16 latent reference. The prefill workspace is bounded by the chunk size. The engine evaluates IndexPool over tiles of 1,024 groups.

Text banks use the common `--max-seqs`, `--prefix-reuse`, `--kv-disk-dir` and `--kv-disk-space` options. Disk snapshots include each bank's committed token history, sparse rows, sliding window, incomplete IndexPool group and DFly2 frontier. A restored window can serve an extension, and a live bank can rewind while its ring still holds the requested history. Visual requests use the serial lane.

DFly2 shares the target embedding, output head and the caches of target layers 25, 29, 33, 37 and 41. The drafter proposes up to five tokens. The target verifies them and retains the accepted prefix. Select the Q8_0 or BF16 file with `--mtp` and set the server's `--mtp-mode on --mtp-draft N` for a forced run.

The shipped Jinja template controls thinking and grouped tool calls. The server frames tool calls for the client. Image and MP4 video inputs use base64 payloads, the release's 16-pixel patches, 2×2 spatial merge, learned row terminator and interleaved temporal, height and width rotary axes. With the shipped video configuration every decoded frame is used. Linux video input requires FFmpeg and FFprobe with the `fd` input protocol.

## Reproducing the checks

`make gate-hy4` runs the CUDA attention and rotary arithmetic check, processor tests and output-protocol fixtures. Set `HY4_MODEL_PATH` to the release directory for `cargo test -p ds4-core --test hy4 -- --ignored` and `cargo test -p ds4-core --lib hy4 -- --ignored`. These tests read the released template, tokenizer, weight metadata and selected Engram records.

`make tests/hy4_quality` builds the teacher-forcing gate. Run `tests/hy4_quality "$model" cases.json result.jsonl` under the [memory guard](host-memory-guard.md). Each case has a name in `case`, an array of token `ids` and sorted `positions`. Each position specifies the zero-based index of the last consumed token, the reference `top1` next-token ID and the reference `nll` for the actual next token. The output retains both reference and measured values for each position.

An optional fourth argument names a DFly2 sidecar. On consecutive positions the gate feeds the reference token into a depth-two trial and commits that token alone. `verify_rows` records the target width. For serving, `ds4 --serving-check` also checks the sampled target rows, RNG handoff, snapshot and stop inside an accepted block, and reports greedy and seeded agreement with one-row decoding. [Measured throughput and quality](hy4-optimization-2026-11-19.md).
