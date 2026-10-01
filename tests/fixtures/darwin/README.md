# Darwin source input

`tokenizer-splits.json` comes from
`FINAL-Bench/Darwin-180B-RSI@bc3c7b0410b40c085b78084e13f01c12df31087b`.
The source tokenizer SHA-256 is recorded in the fixture. `tokenizers==0.23.2`
loads `tokenizer.json` directly, applies its NFC normalizer, and records the
original pre-tokenizer's Unicode spans.

The corpus covers combining marks, Indic/Arabic text, joiners, emoji, number
categories, control text and whitespace. It qualifies input processing only.
GGUFs declaring `tokenizer.ggml.pre=qwen4exp` and
`tokenizer.ggml.normalizer=nfc` select this Rust-host input contract.
