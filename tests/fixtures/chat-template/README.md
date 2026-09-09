# Chat-template JSON oracle

`json-vectors.json` freezes 25 small Jinja renders using Python `json.dumps`.
The JSON options follow the
[Transformers filter](https://github.com/huggingface/transformers/blob/cbc1651a032b923da7f4b44b3d0e6f68e6ba6b55/src/transformers/utils/chat_template_utils.py#L444):
`ensure_ascii=False` by default, with explicit `sort_keys`, `separators`,
`indent` and `ensure_ascii` exercised in template source.

Cases cover insertion order, nested values, empty containers, Unicode and
control characters, integer versus float spelling, signed zero and decimal
notation boundaries near `1e-4` and `1e16`. Invalid keyword, indent and separator
cases must report a rendering error; the Rust error wording need not match
Python. Expected strings are compared byte for byte through the local adapter.

Regenerate with Python and Jinja2, independently of Rust:

```sh
python3 tests/fixtures/chat-template/make_json_vectors.py
cargo test -p ds4-core --test chat_json --locked
```

The fixture records the Python and Jinja2 versions. The generator binds the
filter to Python JSON with the Transformers ASCII default; it contains no
serializer implementation and does not use Rust output as an oracle.

`models/*/chat_template.jinja` and `model-vectors.json` cover nine official
artifacts with 76 Python renders. Each adjacent `provenance.json` records
its source revision, template hash and special-token context. Source bytes
are unmodified. DeepSeek V4 provides a Python encoder instead of Jinja and
is an explicit exception.

```sh
python3 tests/fixtures/chat-template/make_model_vectors.py
cargo test -p ds4-core --test chat_corpus --locked
```

The corpus includes generation blocks, tools/results, reasoning options,
ordered media placeholders, and source-defined errors. These are renderer
compatibility checks, not evidence of GPU or media encoder support.

Third-party templates retain their source licenses, not the host's MIT
license. Adjacent license files are copied unchanged when present at the
pinned revision. The Inkling and K2 source cards declare Apache-2.0; the
pinned Motif card declares MIT but has no separate license file.
