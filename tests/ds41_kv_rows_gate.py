#!/usr/bin/env python3
"""Exercise the KV shell gates' actual row checks without a model or server."""

from pathlib import Path
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parent.parent
VOCAB = 129_280
TOKENS = 31
FALSE_VOCAB = VOCAB * TOKENS // (TOKENS + 1)
GATES = ["ds41_kv_disk_gate.sh", "ds41_kv_reuse_gate.sh"]


def row_source(source: str) -> tuple[str, str]:
    start = source.index("row_size() {")
    end = source.index("\n}\n", start) + 3
    infer = source.index("\nv_a=0\n")
    stop = source.index('\n[ "$v_a" -gt 0 ]', infer)
    return source[start:end], source[infer:stop]


class KvRows(unittest.TestCase):
    def test_exact_row_width(self):
        for name in GATES:
            with self.subTest(gate=name), tempfile.TemporaryDirectory() as directory:
                path = Path(directory)
                (path / "a.ids").write_text("1\n" * TOKENS)
                for filename in ["a.rows", "a.rows.snap"]:
                    with (path / filename).open("wb") as f:
                        f.truncate(TOKENS * VOCAB * 4)
                function, infer = row_source((ROOT / "tests" / name).read_text())
                program = f'set -u\nLOG="$1"\nn_a={TOKENS}\n{function}\n{infer}\nprintf "%s\\n" "$v_a"\n'
                result = subprocess.run(
                    ["/bin/sh", "-s", "--", str(path)], input=program,
                    text=True, capture_output=True, check=True,
                )
                self.assertEqual(int(result.stdout), VOCAB)

    def test_rejects_extra_row(self):
        for name in GATES:
            with self.subTest(gate=name), tempfile.TemporaryDirectory() as directory:
                path = Path(directory)
                (path / "a.ids").write_text("1\n" * TOKENS)
                rows = path / "a.rows"
                with rows.open("wb") as f:
                    f.truncate(TOKENS * VOCAB * 4)
                function, _ = row_source((ROOT / "tests" / name).read_text())
                # The same bytes are 32 rows at a false width. The current
                # penalty route must contain exactly 31 emitted-token rows.
                program = f'{function}\nrow_size "$1" "$2" {FALSE_VOCAB}\n'
                result = subprocess.run(
                    ["/bin/sh", "-s", "--", str(rows), str(path / "a.ids")],
                    input=program, text=True, capture_output=True,
                )
                self.assertEqual(result.returncode, 1)

    def test_reuse_three_args(self):
        source = (ROOT / "tests/ds41_kv_reuse_gate.sh").read_text()
        prefix = source[:source.index('\nROOT=')]
        result = subprocess.run(
            ["/bin/sh", "-s", "--", "server", "model", "engram"],
            input=prefix + '\nprintf "ready:%s\\n" "$EXTRA"\n',
            text=True, capture_output=True,
        )
        self.assertEqual(result.returncode, 0)
        self.assertEqual(result.stdout, "ready:\n")


if __name__ == "__main__":
    unittest.main()
