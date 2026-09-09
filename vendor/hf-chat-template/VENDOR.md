# hf-chat-template

Published crate 1.0.0, MIT OR Apache-2.0, from
[GregoryBolshakov/hf-chat-template](https://github.com/GregoryBolshakov/hf-chat-template).
Crates.io archive SHA-256:
`ed4c5f2a8f34e7371a60a73bf129b13f698db5f9e0ba4cd583ff25bae285e166`.

Source, published tests, documentation and licenses are retained. Cargo.toml
is the published Cargo.toml.orig. Registry metadata and the upstream lockfile
are omitted; the workspace lockfile pins dependencies.

Local patches: `src/json.rs` honors Python `tojson` sorting, separators, ASCII
escaping and integer/string indentation, rejects unsupported options, and
matches Python float spelling. Indentation allocation is capped at 4096.
`src/engine.rs` applies the same float spelling to ordinary output and the
`string` filter; K2's official tool schema uses ordinary numeric output.
The host enables serde_json `float_roundtrip` to preserve parsed float values.
Independent Python fixtures live in `tests/fixtures/chat-template` at the
workspace root; official Inkling vectors exercise these options together.

To update, replace from a pinned published crate, reapply only patches still
needed, and run the adapter JSON, official-template and upstream tests before
changing the version. Model templates must remain unmodified.

The resolved MiniJinja 2.24 prints booleans as Python `True` / `False`.
Four published tests expected the older lowercase spelling; their five
assertions are corrected against Python Jinja2 3.1.2 (no runtime change).
JSON booleans remain lowercase, covered by the independent JSON oracle.
