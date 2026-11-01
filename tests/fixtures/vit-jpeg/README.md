# JPEG decode fixtures

Small JPEGs that pin `ds4-core::vit_image::decode` to PIL's pixels
(`vit_image::tests::jpeg_matches_pil`). Written with Pillow 12.1.1 from crops
of `../vision/homer.png` and `../qwen-images/large.png`:

| File | Size | Save options |
|---|---|---|
| `odd420.jpg` | 61x37 | quality 90, 4:2:0 (sizes off the MCU grid) |
| `h2v1.jpg` | 96x64 | quality 85, 4:2:2 |
| `progressive.jpg` | 120x90 | quality 80, 4:2:0, progressive |
| `gray.jpg` | 77x53 | quality 88, grayscale |

The expected hashes are SHA-256 of `Image.open(f).convert("RGB").tobytes()`
under Pillow 12.1.1 (libjpeg-turbo 3.1.3).
