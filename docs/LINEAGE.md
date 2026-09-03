# Project lineage

`ds4-dfm-rs` is an independent repository, not an independent invention. It
continues the same Git history and MIT-licensed code line:

```text
antirez/ds4
    ↓ CUDA and batched serving
Entrpi/ds4
    ↓ DFM model families and DGX Spark specialization
Baekpica/ds4
    ↓ Rust host promotion
Baekpica/ds4-dfm-rs
```

No history filter, squash, or clean-room relabel was used for the split. Git
authors and ancestry remain intact, including the original fork point
`e16ead1e29c81a67bbb64e5b001117679cf9ce6e`.

## Contributions carried forward

- [`antirez/ds4`](https://github.com/antirez/ds4) established DwarfStar: a
  self-contained, deliberately narrow native engine built around a few useful
  model and hardware combinations. It also established the project's explicit
  debt to llama.cpp and GGML.
- [`Entrpi/ds4`](https://github.com/Entrpi/ds4) developed the CUDA/Linux and
  batched multi-request serving line used here, with DGX Spark / GB10 as a
  reference target.
- [`Baekpica/ds4`](https://github.com/Baekpica/ds4) added the DFM line: explicit
  Solar Open2, K-EXAONE, Motif-3, dots3-note, and Qwen3.8 family paths alongside
  DeepSeek, plus their persistent state, serving, and CUDA work.
- [`Baekpica/ds4-dfm-rs`](https://github.com/Baekpica/ds4-dfm-rs) continues that
  line with the host and control plane promoted to Rust over the same narrow
  native backend.

The Rust split changes ownership boundaries and the release lifecycle. It does
not erase the native backend, change the origin of inherited code, or claim
that ds4 was rewritten from scratch.

## Immutable split references

| Ref | Commit | Meaning |
|---|---|---|
| `v0.6.5-dfm` | `d02e2a4777a34a9f52fd987453b3ea1801fac52e` | Entrpi v0.6.5 reconciled into the DFM line; immutable C baseline. |
| Qwen freeze | `4d40d97f1e575400237a6e5cef21d7f74404a38d` | Last frozen Qwen C behavior used by the resumed Rust campaign. |
| Rust implementation cut | `d126e56877390e9522dde333a34f0d582c3e246c` | Promoted Rust-host implementation after the final server fix. |
| `ds4-dfm-rs-genesis` | `fe7733fb4f7e18204b6ea0a00fe3b136d2029b17` | Annotated split point after `SPLIT_READINESS.md` became green. |
| Post-genesis fix | `7ae5257de96e7bae807e6c406e45e17ba347e52f` | Preserves the closed-hop error through a distributed enqueue race. |

The original destination repository contained a small rename scaffold at
`b01d1fa4172a5c957fe1232774629a192493efe4`. It remains recoverable through the
annotated `pre-genesis-scaffold-b01d1fa` tag; the default branch now follows the
continuous DwarfStar ancestry above.

The annotated `v0.1.0-rc.1` tag names the first independent Rust-host release
candidate containing this document.

`v0.1.0-rc.2` retains the same runtime behavior and sequences two mutating C
oracle calls explicitly so host parity is deterministic across x86 and ARM C
compilers.

`v0.1.0-rc.3` continues the same native backend and model-family lineage while
hardening the shared Rust serving plane for agent concurrency, streaming,
continuations, admission, cache accounting, and shutdown. Its Qwen Q5+Sidecar
one- and two-bank runs validate those host changes; they do not redefine the
`v0.6.5-dfm` golden or claim a new model/kernel release.

## Versioning after the split

`v0.6.5-dfm` remains the behavioral baseline, not the new repository's version.
`ds4-dfm-rs` starts at `v0.1.0-rc.1` so Rust-host releases can advance without
implying a new Entrpi or antirez release. Old tags remain available as
provenance and comparison points.

## Upstream port policy

The repositories no longer have an automatic merge relationship. Changes from
antirez, Entrpi, or Baekpica ds4 are reviewed as selective semantic ports:

1. record the source repository and exact commit;
2. preserve original authorship and license notices;
3. port only the behavior needed by a currently supported path;
4. adapt it at the Rust/native ownership boundary without importing a parallel
   host stack;
5. rerun the affected C/Rust parity, model, and performance gates.

Likewise, a generally useful fix developed here should stay small and close to
the original native style so it can be proposed upstream without dragging in
the Rust host or DFM-only policy.

## Third-party code

The repository remains MIT licensed; see [`../LICENSE`](../LICENSE). Existing
copyright notices are preserved. Selected quantization definitions, CPU paths,
and native kernels derive from or were adapted from
[`llama.cpp`](https://github.com/ggml-org/llama.cpp) and GGML under the MIT
license. The MMQ pin and local patch policy are recorded in
[`../cuda/mmq/VENDOR.md`](../cuda/mmq/VENDOR.md).
