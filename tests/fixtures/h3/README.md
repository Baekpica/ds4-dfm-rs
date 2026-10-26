# H3 media fixture

`test-pattern.png` is a synthetic 64x32 RGBA image made for this repository.
For zero-based pixel `(x, y)`, red is `x * 255 // 63`, green is
`y * 255 // 31`, blue is 255 when `(x < 32) != (y < 16)` and zero otherwise,
and alpha is 255. It contains no third-party artwork or personal data.

The asymmetric ramps and blocks exercise probing, channel order, scaling,
cropping and clip decoding. They are not a generation-quality reference.
