"""Raw lifecycle fixtures must opt out of default thinking to test greedy KV."""
import contextlib
import copy
import io
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import urllib.error

import glm53_bounds_live
import k2_lifecycle_live


class RawFixtureTests(unittest.TestCase):
    def assert_greedy(self, body):
        self.assertEqual(body.get("temperature"), 0)
        self.assertEqual(body.get("reasoning_effort"), "none")
        self.assertEqual(body.get("seed"), 1)

    def test_k2_all_lifecycle_requests_disable_thinking(self):
        for name, body in k2_lifecycle_live.fixture("k2", 64).items():
            with self.subTest(case=name):
                self.assert_greedy(body)

    def test_k2_rejects_cont_before_io(self):
        argv = ["k2_lifecycle_live.py", "seed", "--pid", "123",
                "--output", "/unused", "--artifact-manifest", "/unused.json",
                "--lane", "continuous"]
        stderr = io.StringIO()
        with contextlib.ExitStack() as stack:
            stack.enter_context(patch("sys.argv", argv))
            stack.enter_context(contextlib.redirect_stderr(stderr))
            boundaries = [(Path, "mkdir"), (Path, "open"),
                          (k2_lifecycle_live, "process_identity"),
                          (k2_lifecycle_live, "request"),
                          (k2_lifecycle_live, "write_json")]
            mocks = [stack.enter_context(patch.object(
                target, name, side_effect=AssertionError(f"unexpected {name} access")))
                for target, name in boundaries]
            with self.assertRaises(SystemExit) as error:
                k2_lifecycle_live.main()
            self.assertEqual(error.exception.code, 2)
            for mock in mocks:
                mock.assert_not_called()
        self.assertIn("invalid choice: 'continuous'", stderr.getvalue())

    def test_k2_serial_reuse_contract(self):
        cases = k2_lifecycle_live.fixture("k2", 64)
        with self.subTest(order="fresh restart exercises append first"):
            self.assertEqual(list(cases), ["append", "fork", "exact", "edit"])
        for name, cached, kind, accepted in [
                ("append", 546, "exact", True), ("fork", 546, "exact", True),
                ("exact", 0, "cold", True), ("edit", 0, "cold", True),
                ("append", 0, "cold", False), ("exact", 546, "exact", False)]:
            with self.subTest(case=name, cached=cached), TemporaryDirectory() as directory:
                args = SimpleNamespace(phase="restored", output=Path(directory),
                                       url="http://127.0.0.1:1", lane="serial", context=32768)
                response = {"choices": [{"text": "", "finish_reason": "stop"}],
                            "usage": {"prompt_tokens": 546 if name == "exact" else 561,
                                      "completion_tokens": 0,
                                      "prompt_tokens_details": {"cached_tokens": cached}}}
                stats = {"last_request": {"effective_lane": "serial", "speculation_active": False,
                                         "fallback_reason": None, "reuse_kind": kind},
                         "serving": {"effective": {"ctx": 32768, "max_seqs": 1, "mtp_mode": "off",
                                                   "disk": True, "prefix_reuse": "exact"}}}
                with patch.object(k2_lifecycle_live, "request", side_effect=[response, stats]), \
                     contextlib.redirect_stdout(io.StringIO()):
                    if accepted:
                        k2_lifecycle_live.run_case(args, name, cases[name])
                    else:
                        with self.assertRaises(AssertionError):
                            k2_lifecycle_live.run_case(args, name, cases[name])

    def test_glm_short_and_overcap_requests_disable_thinking(self):
        bodies = []
        def request(url, path, body=None):
            if path == "/v1/stats":
                return {"serving": {"effective": {"ctx": 2048}},
                        "last_request": {"effective_lane": "serial"}}
            bodies.append(copy.deepcopy(body))
            if len(bodies) == 2:
                raise urllib.error.HTTPError(url, 400, "context", {}, io.BytesIO(b'{"error":"context"}'))
            return {"usage": {"completion_tokens": 1,
                              "prompt_tokens_details": {"cached_tokens": 0}}}

        with TemporaryDirectory() as directory:
            args = SimpleNamespace(pid=123, output=Path(directory), model_id="glm",
                                   url="http://127.0.0.1:1")
            with patch.object(glm53_bounds_live, "request", request), patch.object(
                    glm53_bounds_live, "process_identity", return_value={"pid": 123}), contextlib.redirect_stdout(io.StringIO()):
                glm53_bounds_live.http(args)
        self.assertEqual(len(bodies), 2)
        for body in bodies:
            self.assert_greedy(body)


if __name__ == "__main__":
    unittest.main()
