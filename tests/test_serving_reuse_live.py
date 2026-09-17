"""Model-free checks for the HTTP gate's evidence and failure predicates."""
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
import contextlib
import io
import json

import serving_reuse_live as gate


class ReuseRunnerTests(unittest.TestCase):
    def config(self, family="qwen"):
        return {"family": family, "model": "fixture", "context": 2048,
                "banks": 2, "native_chunk": 64, "mtp_mode": "off",
                "expect_speculation": False, "mtp_draft": None,
                "lane": "continuous", "padding_lines": 64}

    def response(self, text="5", cached=300):
        return {"choices": [{"message": {"role": "assistant", "content": text},
                             "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 320, "completion_tokens": 1,
                          "prompt_tokens_details": {"cached_tokens": cached}}}

    def stats(self, kind="fork"):
        return {"last_request": {"effective_lane": "continuous", "reuse_kind": kind,
                                 "speculation_active": False, "fallback_reason": None}}

    def test_history_keeps_actual_whitespace(self):
        body = {"model": "x", "messages": [{"role": "user", "content": "2+2"}]}
        reply = self.response(" \n4\n")
        follow = gate.follow_body(body, reply, "4+1")
        self.assertEqual(follow["messages"][1]["content"], " \n4\n")
        self.assertEqual(len(body["messages"]), 1)

    def test_math_failure_is_not_cache_parity(self):
        response = self.response("4")
        errors = gate.inspect_case(self.config(), "warm", "append", 5,
                                   response, self.stats(), response)
        self.assertTrue(any("arithmetic" in error for error in errors), errors)
        self.assertFalse(any("cold comparison" in error for error in errors), errors)

    def test_edit_requires_family_specific_mechanism(self):
        for family, kind, okay in [("qwen", "partial", True), ("solar", "fork", False),
                                  ("motif", "partial", True), ("deepseek", "fork", True),
                                  ("deepseek", "partial", False)]:
            with self.subTest(family=family, kind=kind):
                errors = gate.inspect_case(self.config(family), "warm", "edit", 6,
                                           self.response("6"), self.stats(kind))
                self.assertEqual(not errors, okay, errors)

    def test_cold_requires_zero_cache_and_exact_output(self):
        warm = self.response("5")
        cold = self.response(" 5", 0)
        errors = gate.inspect_case(self.config(), "cold", "append", 5,
                                   cold, self.stats("cold"), warm)
        self.assertTrue(any("cold comparison" in error for error in errors), errors)
        cold = self.response("5", 2)
        errors = gate.inspect_case(self.config(), "cold", "append", 5,
                                   cold, self.stats("cold"), warm)
        self.assertTrue(any("cached" in error for error in errors), errors)

    def test_declared_mtp_activity_is_checked(self):
        config = self.config()
        config.update(mtp_mode="on", expect_speculation=True, mtp_draft=2)
        errors = gate.inspect_case(config, "warm", "fork", 8,
                                   self.response("8"), self.stats())
        self.assertTrue(any("speculation" in error for error in errors), errors)

    def test_plan_refuses_wrong_mode_or_fitted_width(self):
        config = self.config()
        plan = {"family": "qwen4exp", "requested": {
                    "lane": "auto", "mtp_mode": "off", "prefix_reuse": "partial",
                    "ctx": 2048, "max_seqs": "2", "backend": "cuda"},
                "effective": {"ctx": 2048, "max_seqs": 2, "native_chunk": 64, "sched_chunk": 64, "sched_chunk_live": 64,
                              "mtp_mode": "off", "mtp_draft": None,
                              "prefix_reuse": "partial", "disk": True,
                              "bank_persist_min_tokens": 1, "disk_min_tokens": 1},
                "issues": []}
        self.assertEqual(gate.plan_errors(config, "warm", plan), [])
        plan["effective"]["max_seqs"] = 1
        self.assertTrue(gate.plan_errors(config, "warm", plan))
        plan["effective"]["max_seqs"] = 2
        self.assertTrue(gate.plan_errors(config, "cold", plan))

    def test_modified_fixture_is_rejected_between_phases(self):
        with TemporaryDirectory() as directory:
            output = Path(directory)
            fixture = output / "fixture.json"
            fixture.write_text('{"padding": "original"}')
            gate.write_json(output / "seed.result.json", {"fixture_sha256": gate.digest(fixture)})
            fixture.write_text('{"padding": "changed"}')
            with self.assertRaisesRegex(RuntimeError, "fixture changed"):
                gate.verify_fixture(output, "warm")

    def test_cold_request_must_match_recorded_warm_body(self):
        with TemporaryDirectory() as directory:
            output = Path(directory)
            gate.write_json(output / "warm.append.request.json", {"messages": ["original"]})
            with self.assertRaisesRegex(RuntimeError, "cold request differs"):
                gate.verify_cold_body(output, "append", {"messages": ["changed"]})

    def test_scheduler_chunks_are_pinned(self):
        config = self.config()
        plan = {"family": "qwen4exp", "requested": {
                    "lane": "auto", "mtp_mode": "off", "prefix_reuse": "partial",
                    "ctx": 2048, "max_seqs": "2", "backend": "cuda"},
                "effective": {"ctx": 2048, "max_seqs": 2, "native_chunk": 64,
                              "sched_chunk": 32, "sched_chunk_live": 16,
                              "mtp_mode": "off", "mtp_draft": None,
                              "prefix_reuse": "partial", "disk": True,
                              "bank_persist_min_tokens": 1, "disk_min_tokens": 1},
                "issues": []}
        errors = gate.plan_errors(config, "warm", plan)
        self.assertTrue(any("sched_chunk:" in error for error in errors), errors)
        self.assertTrue(any("sched_chunk_live:" in error for error in errors), errors)

    def test_all_four_phases_preserve_fixture_and_compare(self):
        with TemporaryDirectory() as directory:
            output = Path(directory) / "evidence"
            manifest = Path(directory) / "artifacts.json"
            manifest.write_text('{"synthetic": true}')
            state = {"phase": "seed", "count": 0, "pid": 100, "kind": "cold"}

            def identity(pid):
                return {"pid": pid, "start_ticks": str(pid), "boot_id": "test",
                        "executable_sha256": "fixed-binary"}

            def stats():
                cold = state["phase"] == "cold"
                return {"routes": {"chat": state["count"]},
                        "last_request": self.stats(state["kind"])["last_request"],
                        "serving": {"family": "qwen4exp", "requested": {
                            "lane": "auto", "mtp_mode": "off", "ctx": 2048,
                            "max_seqs": "2", "prefix_reuse": "off" if cold else "partial",
                            "backend": "cuda"}, "effective": {
                            "ctx": 2048, "max_seqs": 2, "native_chunk": 64,
                            "sched_chunk": 64, "sched_chunk_live": 64, "mtp_mode": "off", "mtp_draft": None,
                            "prefix_reuse": "off" if cold else "partial", "disk": not cold,
                            "bank_persist_min_tokens": 1, "disk_min_tokens": 1}, "issues": []}}

            def request(url, path, body=None):
                if path == "/v1/stats":
                    return stats()
                state["count"] += 1
                question = body["messages"][-1]["content"]
                answer = next(n for expression, n in [("2 + 2", 4), ("4 + 1", 5),
                              ("4 + 2", 6), ("5 + 3", 8), ("8 + 1", 9)] if expression in question)
                state["kind"] = ("cold" if state["phase"] in ("seed", "cold")
                                 else "partial" if answer == 6 else "fork")
                cached = 0 if state["kind"] == "cold" else 300
                return self.response(f" \n{answer}\n", cached)

            common = ["--url", "http://127.0.0.1:1", "--output", str(output),
                      "--artifact-manifest", str(manifest)]
            with patch.object(gate, "process_identity", identity), patch.object(gate, "request", request), contextlib.redirect_stdout(io.StringIO()):
                for phase, pid in [("seed", 100), ("warm", 100), ("restored", 200), ("cold", 300)]:
                    state.update(phase=phase, pid=pid)
                    if phase != "warm":
                        state["count"] = 0
                    argv = ["runner", phase, "--pid", str(pid), *common]
                    if phase == "seed":
                        argv += ["--family", "qwen", "--model", "fixture", "--context", "2048",
                                 "--banks", "2", "--native-chunk", "64", "--mtp-mode", "off",
                                 "--expect-speculation", "off", "--lane", "continuous"]
                    with patch("sys.argv", argv):
                        self.assertEqual(gate.main(), 0, phase)
                    self.assertTrue(json.loads((output / f"{phase}.result.json").read_text())["passed"])
            fixture = json.loads((output / "fixture.json").read_text())
            self.assertEqual(fixture["cases"]["append"]["body"]["messages"][1]["content"], " \n4\n")
            self.assertEqual(len(fixture["cases"]["restart"]["body"]["messages"]), 7)


if __name__ == "__main__":
    unittest.main()
