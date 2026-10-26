"""Check CLI media sizing before VAE/model loading; needs a built h3-rs."""

import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
BINARY = Path(os.environ.get("H3_BINARY", str(ROOT / "h3-rs"))).resolve()


class ConditioningCanvasTests(unittest.TestCase):
    def test_decode_uses_render_canvas(self):
        with tempfile.TemporaryDirectory(prefix="h3-cli-canvas-") as directory:
            root = Path(directory)
            for partition in ["FL2VA", "Ref2VA"]:
                tokenizer = root / partition / "tokenizer/tokenizer.json"
                tokenizer.parent.mkdir(parents=True)
                shutil.copyfile(ROOT / "tests/h3/fixtures/tokenizer_portable_fixture.json",
                                tokenizer)
            probe = root / "ffprobe"
            probe.write_text("#!/bin/sh\nprintf '512x512\\n'\n")
            probe.chmod(0o755)
            decode = root / "ffmpeg"
            decode.write_text("""#!/usr/bin/env python3
import json, os, pathlib, sys
pathlib.Path(os.environ['DECODE_ARGS']).write_text(json.dumps(sys.argv[1:]))
sys.stderr.write('geometry-probe-stop')
sys.exit(7)
""")
            decode.chmod(0o755)
            args_file = root / "decode.json"
            for flag in ["--first-frame", "--last-frame", "--ref-image"]:
                for internal in [None, 256]:
                    with self.subTest(flag=flag, internal=internal):
                        args_file.unlink(missing_ok=True)
                        args = [str(BINARY), "-d", str(root), "-p", "A",
                                "--width", "512", "--height", "512", flag, "unused.png"]
                        if internal:
                            args += ["--render-width", str(internal),
                                     "--render-height", str(internal)]
                        result = subprocess.run(args, env={
                            **os.environ, "CUDA_VISIBLE_DEVICES": "",
                            "H3_FFMPEG": str(decode), "H3_FFPROBE": str(probe),
                            "DECODE_ARGS": str(args_file),
                        }, capture_output=True, text=True, timeout=10)
                        self.assertEqual(result.returncode, 2, result.stderr)
                        self.assertIn("geometry-probe-stop", result.stderr)
                        decoded = json.loads(args_file.read_text())
                        size = internal or 512
                        self.assertIn(f"scale={size}:{size}:", decoded[decoded.index("-vf") + 1])


if __name__ == "__main__":
    unittest.main()
