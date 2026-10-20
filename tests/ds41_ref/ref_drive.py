"""Drive the DS41 DSML conformance oracle (oracle/encoding.py, beside us).

The oracle is the operator-supplied DeepSeek-V4.1 text/vision encoding
implementation, vendored into this repo on 2026-10-18 so the carriage's
single authority travels with the tests instead of sitting in an external
tree.  It is the reference for the DSML carriage: tag spellings, the tools
preamble, the parameter type attribute, the effort header, and the decode
side's acceptance shape.

Two subcommands, both printing the oracle's own bytes so a fixture records
what the oracle produced and not what we believe it produces:

    ref_drive.py encode CASE.json MODE        -> the prompt bytes on stdout
    ref_drive.py parse  COMPLETION.txt MODE   -> the parsed message as JSON

A third subcommand drives a case that carries images, which the two above
refuse by construction (their `assert not images` is what keeps the text-only
legs honest):

    ref_drive.py encode-media CASE.json MODE  -> the prompt bytes on stdout,
                                                 the image records on stderr

MODE is chat or thinking.  ORACLE_SHA256 pins the vendored copy, so a later
edit of the file cannot move the oracle silently: a mismatch fails the run
loudly, and the answer is to re-read the reference, re-pin the constant and
regenerate the fixtures in the same act.

Traps measured on this host (2026-10-18, python 3.13.5), recorded so the
next reader does not re-learn them:

  * a case-level "tools" key is dropped unless the case goes through
    load_cases, which moves it into messages[0] (encoding.py:847-869);
  * thinking_mode accepts only "chat" or "thinking" (encoding.py:596), so a
    typo produces the operator-visible "error all the time" symptom;
  * encode_case alone does not clear the case's own thinking_mode key, which
    wins over the command-line mode (encoding.py:878).
"""

import hashlib
import importlib.util
import json
import pathlib
import sys

REF = pathlib.Path(__file__).resolve().parent / "oracle" / "encoding.py"
ORACLE_SHA256 = "13231c592e958fb39e2af6001dae69d90e7a36b265702b9b390236fb387599e2"

MODES = ("chat", "thinking")


def load_ref() -> object:
    """Import the vendored oracle, refusing any revision but the pinned one."""
    digest = hashlib.sha256(REF.read_bytes()).hexdigest()
    if digest != ORACLE_SHA256:
        raise SystemExit(
            f"oracle hash moved: {REF} is {digest}, pinned {ORACLE_SHA256}\n"
            "re-read the reference, re-pin this constant and regenerate the "
            "fixtures it produced before trusting a verdict"
        )
    spec = importlib.util.spec_from_file_location("encoding_dsv41_ref", REF)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def encode(ref: object, case_path: str, mode: str) -> str:
    """Render one case through load_cases, the path that keeps its tools."""
    cases = ref.load_cases(case_path)
    assert len(cases) == 1, f"{case_path} holds {len(cases)} cases; pass one at a time"
    prompt, images = ref.encode_case(cases[0], thinking_mode=mode)
    assert not images, f"{case_path} carries {len(images)} images; text-only here"
    return prompt


def encode_media(ref: object, case_path: str, mode: str) -> str:
    """Render one case that carries images, reporting its records on stderr.

    The image records are the oracle's own `_extract_image` output in prompt
    order; they go to stderr so stdout stays the prompt bytes a fixture holds.
    """
    cases = ref.load_cases(case_path)
    assert len(cases) == 1, f"{case_path} holds {len(cases)} cases; pass one at a time"
    prompt, images = ref.encode_case(cases[0], thinking_mode=mode)
    kinds = ", ".join(record.get("type", "?") for record in images)
    print(f"image records: {len(images)} ({kinds})", file=sys.stderr)
    return prompt


def parse(ref: object, completion_path: str, mode: str) -> str:
    """Parse a completion with the oracle's own parser."""
    text = pathlib.Path(completion_path).read_text()
    msg = ref.parse_message_from_completion_text(text, thinking_mode=mode)
    return json.dumps(msg, ensure_ascii=False, indent=2)


def main() -> None:
    if len(sys.argv) != 4 or sys.argv[1] not in ("encode", "encode-media", "parse"):
        raise SystemExit(__doc__.strip().splitlines()[0] + "\nusage: ref_drive.py {encode|encode-media|parse} FILE MODE")
    what, path, mode = sys.argv[1], sys.argv[2], sys.argv[3]
    if mode not in MODES:
        raise SystemExit(f"mode must be one of {MODES}, got {mode!r}")

    ref = load_ref()
    if what == "encode":
        out = encode(ref, path, mode)
    elif what == "encode-media":
        out = encode_media(ref, path, mode)
    else:
        out = parse(ref, path, mode)
    sys.stdout.write(out)


if __name__ == "__main__":
    main()
