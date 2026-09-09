# Inkling reference fixtures

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

`tokenizer-vectors.json` contains 654 ordinary/rendered-chat encoding vectors
from the source tokenizer using Hugging Face `tokenizers` 0.23.2. It covers
multilingual text, all 60 special strings, whitespace, controls and seeded
Unicode cases. `chat-vectors.json` contains four basic conversations rendered
by the source Jinja template with Jinja2 3.1.2. Both record input SHA-256;
their adjacent generators reproduce them without loading weights:

```sh
python make_tokenizer_vectors.py /path/to/MQ85GB/tokenizer.json tokenizer-vectors.json
python make_chat_vectors.py /path/to/MQ85GB/chat_template.jinja chat-vectors.json
```

`server-vectors.json` adds 32 source-template cases covering all four host
effort levels, system/history messages, tool declarations/results, sorted
JSON and interleaved text/image/audio placeholders. Regenerate with:

```sh
python make_server_vectors.py /path/to/MQ85GB/chat_template.jinja server-vectors.json
```

`moe-vectors.h` contains CPU PyTorch 2.14.0 reference values for eight router
cases, plain/shared-weighted interleaved SwiGLU and routed/shared output
combination. The generator follows the published precision contract and
pinned SGLang equations; its JSON sidecar records revisions and fixture hash.
It uses synthetic inputs and no model weights:

```sh
python make_moe_vectors.py moe-vectors.h
```

`make_media_vectors.py` extracts only the ten BF16 media tensors from local
MQ85GB shards and writes CPU image/audio encoder references to an ignored
directory. The manifest records every extracted tensor's source offset and
SHA-256. It follows the SGLang CUDA norm contract and uses FP64 matmul/sums
before BF16 stores; CUDA reduction differences are measured by the gate.

```sh
python make_media_vectors.py /path/to/MQ85GB /path/to/scratch/media-reference
```

Run `tests/test_inkling_encoders /path/to/scratch/media-reference` through
the host memory guard. This checks all image stages, audio features, and
item order across workspace chunks. It does not cover preprocessing,
full-model logits, serving or API parity.

`image-normalize.json` records FP32 output bits for raw RGB values -1..255,
including padding. `make_image_vectors.py` reproduces the pinned
[TorchvisionBackend fused normalization](https://github.com/huggingface/transformers/blob/cbc1651a032b923da7f4b44b3d0e6f68e6ba6b55/src/transformers/image_processing_backends.py).
Rust preprocessing tests cover source patch geometry, both temporal copies,
PNG alpha removal, JPEG EXIF rotation, and token-budget rejection. The file
decoder accepts PNG/JPEG up to 32 MiB, 32768 pixels per edge, 128 MiB decoded
storage, and 8192 patches per image; each request can impose a smaller budget.

`audio-vectors.json` runs the pinned Transformers feature extractor and
processor methods on CPU using `make_audio_vectors.py SOURCE_DIR OUTPUT`.
The manifest records source hashes and dependency versions. Six waveforms
cover silence, impulses, tones, noise, hop edges and 66 frames. Tests require
exact dMel codes and bounded mel energy error. FFT rounding in nearly silent
tonal bands can amplify log-space error; log-mel values are not bit-exact.
The current reader accepts 16 kHz PCM/float WAV, up to 8 channels (mean
downmix), 32 MiB encoded and 8192 frames, subject to a smaller token budget.
Resampling and compressed audio codecs are not implemented.
