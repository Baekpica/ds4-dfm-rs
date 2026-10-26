"""Model-free checks for the reference gate's failure and comparison rules."""

import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
GATE = ROOT / "tools/minimax_h3/ref_gate.sh"
IMAGE = ROOT / "tests/fixtures/h3/test-pattern.png"


@unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "needs FFmpeg")
class ReferenceGateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.scratch = tempfile.TemporaryDirectory(prefix="h3-ref-gate-")
        cls.root = Path(cls.scratch.name)
        cls.addClassCleanup(cls.scratch.cleanup)
        cls.model = cls.root / "model"
        (cls.model / "Ref2VA/transformer").mkdir(parents=True)
        for name, color, sound in [
            ("base", "red", "anullsrc=r=32000:cl=stereo"),
            ("picture", "blue", "anullsrc=r=32000:cl=stereo"),
            ("sound", "red", "sine=frequency=440:sample_rate=32000"),
        ]:
            subprocess.run([
                "ffmpeg", "-v", "error", "-f", "lavfi", "-i",
                f"color=c={color}:s=32x32:r=24", "-f", "lavfi", "-i", sound,
                "-t", "0.25", "-c:v", "libx264", "-threads", "1", "-ac", "2",
                "-c:a", "aac", str(cls.root / f"{name}.mp4"),
            ], check=True)
        subprocess.run([
            "ffmpeg", "-v", "error", "-i", str(cls.root / "base.mp4"),
            "-map", "0", "-c", "copy", "-metadata", "comment=different container",
            str(cls.root / "remux.mp4"),
        ], check=True)
        cls.renderer = cls.root / "renderer"
        cls.renderer.write_text("""#!/usr/bin/env python3
import os, pathlib, shutil, sys
args = sys.argv
if '--ref-video' in args and int(args[args.index('--frames') + 1]) < 56:
    sys.exit(42)
if os.environ.get('STUB_EXIT'):
    sys.exit(int(os.environ['STUB_EXIT']))
output = pathlib.Path(args[args.index('-o') + 1])
if os.environ.get('STUB_BAD_OUTPUT'):
    output.write_text('invalid media')
else:
    shutil.copyfile(os.environ['STUB_MEDIA'], output)
""")
        cls.renderer.chmod(0o755)
        cls.bin = cls.root / "bin"
        cls.bin.mkdir()
        date = cls.bin / "date"
        date.write_text("""#!/usr/bin/env python3
import os, pathlib
state = pathlib.Path(os.environ['STUB_CLOCK'])
value = int(state.read_text()) if state.exists() else 100
print(value)
state.write_text(str(value + int(os.environ.get('STUB_TICK', '1'))))
""")
        date.chmod(0o755)

    def gate(self, name, *, media="base", case="fl2va-first", compare=None, **env):
        out = self.root / self._testMethodName / name
        out.mkdir(parents=True)
        args = ["sh", str(GATE), "--binary", str(self.renderer), "--model",
                str(self.model), "--out", str(out), "--clip", str(self.root / "base.mp4"),
                "--image", str(IMAGE), "--case", case]
        if compare:
            args += ["--compare", str(compare)]
        result = subprocess.run(args, env={
            **os.environ, "PATH": f"{self.bin}:{os.environ['PATH']}",
            "STUB_MEDIA": str(self.root / f"{media}.mp4"),
            "STUB_CLOCK": str(out / "clock"), **env,
        }, capture_output=True, text=True, timeout=30)
        return result, out

    def test_exit_code(self):
        result, out = self.gate("failed", STUB_EXIT="7")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("FAIL fl2va-first exit=7", (out / "manifest.txt").read_text())

    def test_video_audio_duration(self):
        result, _ = self.gate("video", case="ref2va-video")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_media_only_comparison(self):
        first, base = self.gate("base")
        self.assertEqual(first.returncode, 0, first.stderr)
        second, out = self.gate("remux", media="remux", compare=base, STUB_TICK="9")
        self.assertNotEqual((base / "manifest.txt").read_text(),
                            (out / "manifest.txt").read_text())
        self.assertEqual(second.returncode, 0, second.stdout + second.stderr)

    def test_changed_tracks(self):
        first, base = self.gate("base")
        self.assertEqual(first.returncode, 0, first.stderr)
        for media in ["picture", "sound"]:
            with self.subTest(media=media):
                result, _ = self.gate(media, media=media, compare=base)
                self.assertNotEqual(result.returncode, 0, result.stdout)

    def test_invalid_media(self):
        result, out = self.gate("invalid", STUB_BAD_OUTPUT="1")
        self.assertNotEqual(result.returncode, 0, result.stdout)
        self.assertNotIn("PASS", (out / "manifest.txt").read_text())

    def test_unknown_case(self):
        result, _ = self.gate("unknown", case="typo")
        self.assertNotEqual(result.returncode, 0, result.stdout)


if __name__ == "__main__":
    unittest.main()
