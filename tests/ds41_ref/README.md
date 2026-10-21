# The DS41 DSML reference leg

The conformance oracle for the DS41 DSML carriage, and the harness that drives
it.  Read `docs/ds41-dsml-reference-matrix.md` first: it is the artifact this
directory exists to feed, and it pairs every requirement with the case that
exercises it.

## The oracle

`oracle/encoding.py` — the official DeepSeek-V4.1 text/vision encoding
implementation, operator-supplied and vendored here on 2026-10-18.  It is the
implementation that teaches the format to the model, so it decides the prompt
bytes and asserts the completion shape.  It is the carriage's single
authority: the C implementation that once served as a second reference is not
consulted, and where the two disagreed the disagreement is recorded in
`docs/ds41-dsml-reference-matrix.md` part D with both citations.

`ref_drive.py` pins the vendored file by sha256
(`13231c592e958fb39e2af6001dae69d90e7a36b265702b9b390236fb387599e2`),
so a later edit of the reference cannot move the oracle silently; see
`oracle/README.md` for the provenance and the refresh procedure.

## Layout

| Path | What it is |
| --- | --- |
| `ref_drive.py` | the driver: `encode CASE.json MODE` prints the prompt bytes, `parse COMPLETION.txt MODE` prints the parsed message as JSON.  Refuses a moved oracle |
| `cases/*.json` | encode cases in the oracle's own case format (messages plus a case-level `tools`) |
| `decode/*.txt` | completion bytes, built by `mk_decode_input.py` from the oracle's own templates so each one is canonical by construction |
| `mk_decode_input.py` | authors `decode/*.txt` and has the oracle parse each; aborts if the oracle rejects its own output |
| `regen_fixtures.sh` | regenerates `crates/ds4-server/tests/fixtures/ds41-ref/` from the oracle and copies the decode inputs in |

## Running

```bash
bash tests/ds41_ref/regen_fixtures.sh          # only when the oracle or a case changes
cargo test -p ds4-server --test ref_encoding   # encode side
cargo test -p ds4-server --test ref_decode     # decode side
```

Regeneration is an explicit act: a fixture that silently tracks the oracle
pins nothing.  The same reason the harness lives under `tests/` and not in the
release build path — like the C oracles in `tests/parity/`, this is an
analysis-host oracle.

## Traps measured on this host (python 3.13.5)

- A case-level `tools` key is dropped unless the case goes through
  `load_cases`, which moves it into `messages[0]` (encoding.py:847-869).
  `encode_case` alone therefore encodes a case with no tools, silently.
- `thinking_mode` accepts only `chat` or `thinking` (encoding.py:596); any
  other value reaches an assert.
- The thinking-mode completion must NOT carry the opening `<think>` token: the
  oracle asserts no special token in content or reasoning (encoding.py:982),
  so it takes the generation, not the prompt tail.
- Never write a raw DSML block into a chat message or a prose file: this
  session's harness parsed a literal block out of an assistant message as a
  real tool invocation and failed the turn twice.  Build such bytes from the
  oracle's constants (as `mk_decode_input.py` does) or from the Rust constants
  (`DSML_TOOL_CALLS_START_SPACED` and friends).
