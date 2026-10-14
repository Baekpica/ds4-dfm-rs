#!/usr/bin/env python3
"""Run the actual V4.1 generation driver against bounded host tensors."""

import os
from pathlib import Path
import re
import shlex
import subprocess
import tempfile


def function(source: str, name: str) -> tuple[str, int]:
    tokens = re.compile(r'//[^\n]*|/\*.*?\*/|"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'', re.S)
    masked = tokens.sub(lambda match: " " * len(match.group()), source)
    start = re.search(rf"^[^\n;{{}}]*\b{re.escape(name)}\s*\([^;{{}}]*\)\s*{{", masked, re.M)
    if start is None:
        raise RuntimeError(f"missing native function: {name}")
    opening = masked.index("{", start.start())
    depth = 0
    for pos in range(opening, len(masked)):
        if masked[pos] == "{":
            depth += 1
        elif masked[pos] == "}":
            depth -= 1
            if depth == 0:
                return source[start.start() : pos + 1], source[: start.start()].count("\n") + 1
    raise RuntimeError(f"unterminated native function: {name}")


def structure(source: str, name: str) -> str:
    end = source.index(f"}} {name};") + len(f"}} {name};")
    start = source.rfind("typedef struct {", 0, end)
    return source[start:end]


def named_struct(source: str, name: str) -> str:
    """A plain `struct NAME { ... };` (no typedef), e.g. ds4_v41_images."""
    start = source.index(f"struct {name} {{")
    end = source.index("};", start) + 2
    return source[start:end]


def main() -> int:
    root = Path(__file__).resolve().parent.parent
    forward_path = root / "ds4_ds41_forward.inc"
    draft_path = root / "ds4_ds41_draft.inc"
    forward = forward_path.read_text()
    draft = draft_path.read_text()
    with tempfile.TemporaryDirectory(prefix="ds41-generate-boundary-") as directory:
        temp = Path(directory)
        (temp / "ds41_boundary_types.inc").write_text(
            "typedef ds41_engram_feed v41_engram_feed;\n"
            + structure(forward, "v41_image") + "\n"
            + named_struct(forward, "ds4_v41_images") + "\n"
            + structure(forward, "ds4_v41_state") + "\n"
            + structure(draft, "ds4_v41_draft") + "\n"
            + forward[forward.index("struct ds4_v41_feed {") : forward.index("static int v41_feed_tramp")]
        )
        groups = {
            "ds41_boundary_alloc.inc": (forward_path, forward, ["v41_state_alloc"]),
            "ds41_boundary_feed.inc": (forward_path, forward, ["v41_gen_cap", "v41_feed_tramp", "v41_feed_wait_tramp", "v41_feed_next_tramp", "ds4_v41_feed_close", "ds4_v41_feed_open"]),
            "ds41_boundary_forward.inc": (forward_path, forward, ["v41_forward", "v41_chunk_nc", "v41_prefill_chunk"]),
            "ds41_boundary_generate.inc": (draft_path, draft, ["ds4_v41_generate_argmax"]),
        }
        for filename, (path, source, names) in groups.items():
            pieces = []
            for name in names:
                body, line = function(source, name)
                pieces.append(f'#line {line} "{path}"\n{body}\n')
            (temp / filename).write_text("\n".join(pieces))
        binary = temp / "test_ds41_generate_boundary"
        subprocess.run(
            shlex.split(os.environ.get("CC", "cc"))
            + ["-std=c99", "-O0", "-Wall", "-Wextra", "-Wno-unused-parameter", "-I", str(temp),
               str(root / "tests/test_ds41_generate_boundary.c"), "-lm", "-o", str(binary)],
            check=True,
        )
        return subprocess.run([str(binary)], check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
