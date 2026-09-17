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
