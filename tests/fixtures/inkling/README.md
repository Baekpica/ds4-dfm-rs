# Inkling catalog fixtures

Source model: [thinkingmachines/Inkling-Small](https://huggingface.co/thinkingmachines/Inkling-Small/tree/8cc5877b44d343f88b92086aa1fb72897950f06a),
revision `8cc5877b44d343f88b92086aa1fb72897950f06a`, Apache-2.0.

`mq85gb.tsv` and `mtp-bf16.tsv` contain every tensor's original name, GGML
type, safetensors dimension order and payload size. They are extracted from
the published conversion plans, independently of the runtime layout code:

- [MQ85GB plan](https://huggingface.co/Baekpica/Inkling-Small-Mixed-Quant-GGUF/blob/4a5b2db7de3a294af660053284337bd5424cd3e4/MQ85GB/tensor-plan.json)
- [MTP-BF16 plan](https://huggingface.co/Baekpica/Inkling-Small-GGUF/blob/01e829c5acf2c9aa8026f5b8157d980e1c20730c/MTP-BF16/tensor-plan.json)

Each TSV header records the original plan's SHA-256. Config and processor
JSON are verbatim source assets, also retained in `ds4-core/src/inkling`:

| Asset | SHA-256 |
|---|---|
| config.json | `dcb5b1d587bce2f1e6b29833d739a724d05b4bfaa2dc1164fbe679330478ba53` |
| processor_config.json | `b4a3962ea5f7ec39f40b5cf14e57ce99776c3dcce4756a110f7a169809e3a04c` |

These fixtures prove catalog and precision contracts. They contain no
reference logits and do not establish numerical execution parity.
