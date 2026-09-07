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
