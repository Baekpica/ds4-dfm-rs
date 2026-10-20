"""Author the decode-side completion texts with the oracle's own templates.

The decode side needs completion bytes: what the model emits.  Those bytes
must be the oracle's own production, so this script builds each completion
from the reference's templates (tool_call_template, tool_calls_template,
encode_arguments_to_dsml) instead of from a hand-written literal, then makes
the oracle parse its own output.  A completion this script writes is
therefore canonical by construction: if the oracle rejects one, the oracle
pinned here is not the revision this harness was written against, and the
script says so instead of leaving a bad fixture behind.

Beware the literal: the DSML markers are assembled from the oracle's own
constants (missile-shaped text in a chat message is parsed as a real tool
call by the harness).  Same reason the oracle is loaded here, never
paraphrased.

Usage: python3 mk_decode_input.py OUTDIR
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

# The oracle's loader carries the hash pin, so both authors and the driver
# read one revision of the reference and one place that says which.
from ref_drive import load_ref  # noqa: E402


def invoke(ref: object, name: str, args: dict) -> str:
    """One invoke element, rendered by the reference's own template."""
    return ref.tool_call_template.format(
        dsml_token=ref.dsml_token,
        tool_call_tag_name=ref.tool_call_tag_name,
        name=name,
        arguments=ref.encode_arguments_to_dsml({"name": name, "arguments": args}),
    )


def block(ref: object, invokes: list) -> str:
    """One tool-calls wrapper around the given invokes."""
    return ref.tool_calls_template.format(
        dsml_token=ref.dsml_token,
        tc_block_name=ref.tool_calls_block_name,
        tool_calls="\n".join(invokes),
    )


def completions(ref: object) -> dict:
    """Every decode case: prose, the block the oracle renders, its EOS."""
    marker = f"<{ref.dsml_token}{ref.tool_calls_block_name}>"
    return {
        # Canonical single call: one string child and one integer child.
        "dec-one-call": "Reading it now."
        + "\n\n"
        + block(ref, [invoke(ref, "read_file", {"target_file": "/tmp/x.txt", "limit": 40})])
        + ref.eos_token,
        # Two invokes in one block, the shape TP-03 exercises on the wire.
        "dec-two-calls": "Checking both."
        + "\n\n"
        + block(ref, [
            invoke(ref, "read_file", {"target_file": "/tmp/a.txt"}),
            invoke(ref, "read_file", {"target_file": "/tmp/b.txt"}),
        ])
        + ref.eos_token,
        # Every JSON child kind the type attribute distinguishes.
        "dec-complex-args": ""
        + "\n\n"
        + block(ref, [invoke(ref, "set_opts", {
            "label": "tea",
            "seconds": 90,
            "force": True,
            "ratio": 1.5,
            "note": None,
            "deps": ["a", "b"],
            "opts": {"k": 1},
        })])
        + ref.eos_token,
        # Non-ASCII in a string child: the oracle's to_json is ensure_ascii
        # False, so the value stays raw.
        "dec-unicode": ""
        + "\n\n"
        + block(ref, [invoke(ref, "read_file", {"target_file": "/tmp/smörgås.txt"})])
        + ref.eos_token,
        # A marker-looking string inside a value: structure must not be read
        # out of a parameter body (inventory G9's parse-side rule).
        "dec-marker-in-value": ""
        + "\n\n"
        + block(ref, [invoke(ref, "echo", {"body": f"quote the {marker} tag"})])
        + ref.eos_token,
        # Thinking mode: reasoning before the close, prose after it.  No
        # opening <think>: driving the oracle proved it asserts that no
        # special token sits in content or reasoning (encoding.py:982), so
        # its completion text starts after the header the prompt wrote --
        # the generation, not the prompt tail.
        "dec-thinking": "The user wants /tmp/x.txt read."
        + ref.thinking_end_token
        + "Reading it now."
        + "\n\n"
        + block(ref, [invoke(ref, "read_file", {"target_file": "/tmp/x.txt"})])
        + ref.eos_token,
    }


def main() -> None:
    out = pathlib.Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)

    ref = load_ref()
    for name, text in completions(ref).items():
        mode = "thinking" if name == "dec-thinking" else "chat"
        # The oracle must accept its own output, on this mode.
        parsed = ref.parse_message_from_completion_text(text, thinking_mode=mode)
        calls = parsed["tool_calls"]
        path = out / f"{name}.txt"
        path.write_text(text)
        print(f"{path.name}: {len(text)} bytes, {len(calls)} call(s), "
              f"names={[c['function']['name'] for c in calls]}")


if __name__ == "__main__":
    main()
