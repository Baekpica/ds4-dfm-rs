"""Replay the live gate without loading a model."""
import hashlib
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
from threading import Thread
import unittest

GATE = Path(__file__).with_name("naive_long_live.py")
ANSWER = "ORCHID-74-COPPER-29"
CONTEXT = 32768
INPUT_TOKENS = 100


class LongGateTest(unittest.TestCase):
    def test_output_modes(self):
        for mode in ("stream", "buffered"):
            with self.subTest(mode=mode), TemporaryDirectory() as tmp:
                self.replay(Path(tmp), mode)

    def replay(self, root, mode):
        bodies = []
        message = {"role": "assistant", "content": ANSWER, "reasoning_content": "retrieve"}

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def send(self, value):
                self.send_response(HTTPStatus.OK)
                self.end_headers()
                self.wfile.write(json.dumps(value).encode())

            def do_GET(self):
                if self.path == "/v1/models":
                    self.send({"data": [{"context_length": CONTEXT}]})
                    return
                self.send({"serving": {"effective": {"ctx": CONTEXT, "max_seqs": 1,
                                                     "mtp_mode": "off"}},
                           "governor": {"faults": 0}, "last_request": {
                               "effective_lane": "continuous", "speculation_active": False}})

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                bodies.append(body)
                cached = INPUT_TOKENS if len(bodies) > 1 else 0
                usage = {"prompt_tokens": INPUT_TOKENS + cached, "completion_tokens": 1,
                         "prompt_tokens_details": {"cached_tokens": cached}}
                if not body["stream"]:
                    self.send({"choices": [{"message": message, "finish_reason": "stop"}],
                               "usage": usage})
                    return
                self.send_response(HTTPStatus.OK)
                self.end_headers()
                event = {"choices": [{"delta": message, "finish_reason": "stop"}], "usage": usage}
                self.wfile.write(b"data: " + json.dumps(event).encode() + b"\n\ndata: [DONE]\n\n")

        fixture = root / "fixture"
        fixture.mkdir()
        raw = json.dumps({"messages": [{"role": "user", "content": "retrieve"}],
                          "stream": mode == "stream"}).encode()
        (fixture / "request.json").write_bytes(raw)
        (fixture / "fixture.json").write_text(json.dumps({"context": CONTEXT,
            "input_tokens": INPUT_TOKENS, "expected_answer": ANSWER,
            "request_sha256": hashlib.sha256(raw).hexdigest()}))
        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            for phase in ("seed", "follow", "restored"):
                result = subprocess.run([sys.executable, str(GATE), phase, "--fixture", str(fixture),
                    "--out", str(root / "out"), "--banks", "1", "--url",
                    f"http://127.0.0.1:{server.server_port}"], capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
            self.assertEqual(bodies[1]["messages"][-2], message)
            self.assertEqual(bodies[2]["messages"][-2], message)
        finally:
            server.shutdown()
            server.server_close()
            thread.join()


if __name__ == "__main__":
    unittest.main()
