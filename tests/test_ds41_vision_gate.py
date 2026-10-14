#!/usr/bin/env python3
"""Exercise the real vision gate with fake tools; no model, GPU, or listener."""

import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import textwrap
import unittest


ROOT = Path(__file__).resolve().parents[1]
GATE = ROOT / "tests/ds41_vision_gate_spark.sh"
FIXTURES = (
    "review-market-panel.png", "golden_panel.bin",
    "bug.jpeg", "golden_bug.bin", "homer.png", "golden_homer.bin",
)

# One executable behind each fake tool name. HTTP runs the gate's actual
# embedded Python with urlopen replaced, so status/body checks remain tested.
MOCK_TOOLS = r'''
import io, json, os, signal, sys, time, urllib.error, urllib.request
from pathlib import Path

name = Path(sys.argv[0]).name
args = sys.argv[1:]
task_dir = Path(os.environ["GATE_TEST_DIR"])
case = os.environ["GATE_TEST_CASE"]
state = task_dir / "server.json"

def event(value):
    with (task_dir / "events").open("a") as out:
        out.write(value + "\n")

if name == "pgrep":
    sys.exit(1)
if name == "awk":
    print(200000)
    sys.exit(0)
if name == "sleep":
    time.sleep(0.01)
    sys.exit(0)
if name == "curl":
    sys.exit(0 if state.exists() else 1)
if name == "gate-server":
    mode = "vision" if "--vision-gguf" in args else "no-sidecar"
    def stop(*unused):
        state.unlink(missing_ok=True)
        event("stop:" + mode)
        sys.exit(0)
    signal.signal(signal.SIGTERM, stop)
    event("start:" + mode)
    state.write_text(json.dumps({"mode": mode, "pid": os.getpid()}))
    while True:
        signal.pause()
if name == "gate-cli":
    if "--vision-golden" in args:
        fixture = Path(args[args.index("--image") + 1]).name
        event("golden:" + fixture)
        print("[v41-golden] mock comparison")
        sys.exit(7 if case == "golden_error" else 0)
    if "--dump-tokens" in args:
        event("tokens")
        print("[100, 101]")
        sys.exit(0)
    ids = Path(args[args.index("--gen-ids") + 1])
    if ids.name == "admission_ids.txt":
        event("admission")
        print("unrelated failure" if case == "admission_wrong_error"
              else "prefill needs 999999 MB peak (admission 2x)")
        sys.exit(0 if case == "admission_success" else 1)
    mode = "vision" if "--vision-gguf" in args else "base"
    event("text:" + mode)
    if case == "text_" + mode + "_error":
        sys.exit(7)
    print("different" if case == "text_mismatch" and mode == "vision"
          else "Marginal cost is the cost of another unit.")
    sys.exit(0)

assert name == "python3", name
if args[0] == "-c":
    exec(args[1])
    sys.exit(0)

class Response(io.BytesIO):
    status = 200

def urlopen(request, *unused, **kwargs):
    url = request if isinstance(request, str) else request.full_url
    if url.endswith("/v1/models"):
        return Response(b'{"data":[{"id":"mock-ds41"}]}')
    mode = json.loads(state.read_text())["mode"]
    event("http:" + mode)
    if mode == "vision":
        if case == "image_http_error":
            raise urllib.error.URLError("mock image request failure")
        answer = "" if case == "image_empty" else "A labeled market panel."
        return Response(json.dumps({"choices":[{"message":{"content":answer}}]}).encode())
    if case == "no_sidecar_success":
        return Response(b'{"choices":[]}')
    status = 500 if case == "no_sidecar_status" else 400
    message = ("unrelated failure" if case == "no_sidecar_message"
               else "image input requires the DeepSeek V4.1 vision sidecar (--vision-gguf)")
    body = io.BytesIO(json.dumps({"error":{"message":message}}).encode())
    raise urllib.error.HTTPError(url, status, "mock refusal", {}, body)

urllib.request.urlopen = urlopen
sys.argv = ["-", *args[1:]]
exec(compile(sys.stdin.read(), "<gate-http>", "exec"), {"__name__":"__main__"})
'''


class VisionGateTest(unittest.TestCase):
    def run_gate(self, case="ok", missing=None):
        with tempfile.TemporaryDirectory(prefix="ds41-vision-gate-") as temp:
            work = Path(temp)
            tools = work / "tools"
            tools.mkdir()
            mock = tools / "mock"
            mock.write_text(f"#!{sys.executable}\n" + textwrap.dedent(MOCK_TOOLS))
            mock.chmod(0o755)
            for name in ("gate-cli", "gate-server", "pgrep", "awk", "curl", "sleep", "python3"):
                (tools / name).symlink_to(mock)
            for name in ("main.gguf", "vision.gguf", *FIXTURES):
                if name != missing:
                    (work / name).write_bytes(b"fixture")
            (work / "zchain").mkdir()
            env = {
                **os.environ,
                "PATH": str(tools) + os.pathsep + os.environ["PATH"],
                "BIN": str(tools / "gate-cli"), "SERVER": str(tools / "gate-server"),
                "IMG_DIR": str(work), "GOLD_DIR": str(work),
                "ENGRAM_DIR": str(work / "engram"), "OUT": str(work / "out"),
                "GATE_TEST_DIR": str(work), "GATE_TEST_CASE": case,
            }
            proc = subprocess.Popen(
                ["bash", str(GATE), str(work / "main.gguf"), str(work / "vision.gguf"),
                 str(work / "zchain")],
                env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, start_new_session=True,
            )
            try:
                output, _ = proc.communicate(timeout=20)
            finally:
                # Only the process group this test created can be cleaned up.
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                proc.wait()
            event_file = work / "events"
            events = event_file.read_text().splitlines() if event_file.exists() else []
            return proc.returncode, output, events

    def assert_failure(self, case, next_step, missing=None):
        rc, output, events = self.run_gate(case, missing)
        self.assertNotEqual(rc, 0, output)
        self.assertIn("GATE FAIL:", output)
        self.assertNotIn(next_step, output)
        starts = [event[6:] for event in events if event.startswith("start:")]
        stops = [event[5:] for event in events if event.startswith("stop:")]
        self.assertEqual(starts, stops, "gate left its fake server running")
        return events

    def test_happy_path(self):
        rc, output, events = self.run_gate()
        self.assertEqual(rc, 0, output)
        self.assertIn("[gate] done", output)
        self.assertEqual(sum(e.startswith("golden:") for e in events), 3)
        self.assertEqual(events.count("http:vision"), 3)
        self.assertIn("http:no-sidecar", events)
        self.assertIn("admission", events)

    def test_missing_fixture(self):
        for missing in FIXTURES:
            with self.subTest(missing=missing):
                events = self.assert_failure("ok", "== 1.", missing)
                self.assertEqual(events, [], "missing fixture loaded a model")

    def test_golden_error(self):
        self.assert_failure("golden_error", "== 2.")

    def test_text_failures(self):
        for case in ("text_base_error", "text_vision_error", "text_mismatch"):
            with self.subTest(case=case):
                self.assert_failure(case, "== 3.")

    def test_image_request_failures(self):
        for case in ("image_http_error", "image_empty"):
            with self.subTest(case=case):
                self.assert_failure(case, "== 5.")

    def test_no_sidecar_contract(self):
        for case in ("no_sidecar_success", "no_sidecar_status", "no_sidecar_message"):
            with self.subTest(case=case):
                self.assert_failure(case, "== 6.")

    def test_admission_failures(self):
        for case in ("admission_success", "admission_wrong_error"):
            with self.subTest(case=case):
                self.assert_failure(case, "[gate] done")


if __name__ == "__main__":
    unittest.main()
