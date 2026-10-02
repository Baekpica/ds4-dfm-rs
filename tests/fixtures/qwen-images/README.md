# Qwen image fixtures

Fixed inputs for `tests/bench_qwen_images.py` and the full-model vision gate.
These are a small reproducible regression suite, not a general vision-quality
benchmark.

| File | Pixels | Content |
| --- | --- | --- |
| `small.png` | 256 × 256 | Synthetic task dashboard |
| `screen.png` | 1024 × 768 | Queued 12, Running 7, Failed 3 |
| `screen-changed.png` | 1024 × 768 | Same screenshot with Failed 9, for cache identity |
| `document.png` | 1536 × 1024 | Synthetic invoice: subtotal $350, tax $35, total $385 |
| `photo.jpg` | 512 × 507 | Apollo 17 photograph of Earth |
| `large.png` | 1920 × 1080 | Synthetic task dashboard |

The PNGs were generated for this repository using Pillow and DejaVu Sans;
they contain no private application data. `photo.jpg` is the NASA Apollo 17
image AS17-148-22727, public domain, copied unchanged from the upstream ds4
`tests/vision-fixtures/glm53/earth.jpg` fixture. Original photograph:
[Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Earth_apollo17.jpg).
Its SHA-256 is
`a48278513a38768ff92247972e60bc873eb361a8527cb8686a3e19c3ae1bdf9d`.

The `multi` case supplies small, screen, photo, and document in that order.
Do not rescale these files for A/B comparisons. The benchmark records their
SHA-256 hashes with every response.

## Rust-tokenized full-model gate

`test-qwen-vision-host` accepts whole-prompt IDs from the Rust tokenizer.
Prepare one image placeholder per source image, in the same order. This fixed
single-image regression prompt uses the Qwen/Darwin chat grammar:

```sh
MODEL=/path/to/model-00001-of-00003.gguf
GATE_DIR=$(mktemp -d)
cat > "$GATE_DIR/rendered.txt" <<'EOF'
<|im_start|>user
Briefly describe the visible content and any task counts.
<|vision_start|><|image_pad|><|vision_end|><|im_end|>
<|im_start|>assistant
<think>

</think>

EOF
./ds4 -m "$MODEL" --dump-tokens --prompt-file "$GATE_DIR/rendered.txt" > "$GATE_DIR/dump.txt"
head -n 1 "$GATE_DIR/dump.txt" > "$GATE_DIR/tokens.txt"
make CUDA_ARCH=sm_121 tests/test_qwen_vision_host
./tests/test_qwen_vision_host "$MODEL" "$GATE_DIR/tokens.txt" tests/fixtures/qwen-images/screen.png
```

`--dump-tokens` tokenizes text as written and exits before GPU model loading;
it does not render Jinja. An official rendered prompt can replace this fixed
prompt. Tokenize the complete text once, using the same GGUF as the gate.
For multiple images, repeat the vision placeholder triplet and pass each image
in order. Total pre-merge patches must be at least 512. Reuse the existing
weight-owner IPC environment when supplied.

The manual gate uses one 262144-token bank, 8192-token prefill and plain decode.
Four reset-bank admissions compare pair/pair/quad/quad exactly, selected by
`DS4_QWEN_VISION_QUAD=0/0/1/1`:
features, 32 raw full-vocabulary logits, 32 greedy IDs, payload and M-RoPE.
For packed K/V parity, prefix the gate command with
`DS4_QWEN_VISION_GATE_CONTROL=DS4_QWEN_VISION_PACK`. All four admissions then
use Quad and compare packing `0/0/1/1`. Use a new output directory.
EOS and Qwen EOT are excluded from sampling; saved logits precede exclusion.
The saved frontier is the prompt plus 31 consumed outputs; output 32 is pending.
Payload logits are zero-filled; `logits.f32` holds the real frontiers. Separate
M-RoPE data includes image metadata and PLE hash state. Files are created
exclusively beside `tokens.txt` and streamed in one-MiB chunks. Use a new directory
for each run; binary artifacts belong to the same native build ABI.
This is a numerical gate. MTP, natural stopping, API/template processing,
restore, cached image reuse and performance require their separate serving gates.
