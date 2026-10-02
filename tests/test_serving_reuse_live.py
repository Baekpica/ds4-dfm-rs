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
                "lane": "continuous", "padding_lines": 64,
                "reasoning_effort": "none", "max_tokens": 32}

    def case(self, name):
        template = gate.read_json(gate.FIXTURE)[name]
        return {key: template[key] for key in ("answer", "accepted_forms")}

    def response(self, text="5", cached=300):
        return {"choices": [{"message": {"role": "assistant", "content": text},
                             "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 320, "completion_tokens": 1,
                          "prompt_tokens_details": {"cached_tokens": cached}}}

    def stats(self, kind="fork"):
        return {"last_request": {"effective_lane": "continuous", "reuse_kind": kind,
                                 "speculation_active": False, "fallback_reason": None}}

    def test_mtp_one_token_stop(self):
        config = self.config()
        config.update(mtp_mode="on", mtp_draft=2, expect_speculation=True)
        errors = gate.inspect_case(config, "seed", "seed", self.case("seed"),
                                   self.response("4", 0), self.stats("cold"))
        self.assertEqual(errors, [])

    def test_mtp_activity_stays_strict(self):
        config = self.config()
        config.update(mtp_mode="on", mtp_draft=2, expect_speculation=True)
        for tokens, finish, active in [(2, "stop", False), (1, "length", False),
                                       (1, "stop", None)]:
            with self.subTest(tokens=tokens, finish=finish, active=active):
                response = self.response("4", 0)
                response["usage"]["completion_tokens"] = tokens
                response["choices"][0]["finish_reason"] = finish
                stats = self.stats("cold")
                stats["last_request"]["speculation_active"] = active
                errors = gate.inspect_case(config, "seed", "seed", self.case("seed"),
                                           response, stats)
                self.assertTrue(any("speculation" in error for error in errors), errors)

        stats = self.stats("cold")
        stats["last_request"]["speculation_active"] = True
        errors = gate.inspect_case(self.config(), "seed", "seed", self.case("seed"),
                                   self.response("4", 0), stats)
        self.assertTrue(any("speculation" in error for error in errors), errors)

    def test_all_declared_literal_answer_forms(self):
        forms = {
            "seed": (4, "2 + 2"), "append": (5, "4 + 1"),
            "edit": (6, "4 + 2"), "fork": (8, "5 + 3"),
            "restart": (9, "8 + 1"),
        }
        for name, (answer, expression) in forms.items():
            accepted = [str(answer), f"{answer}.", f"{expression} = {answer}",
                        f"{expression} = {answer}."]
            case = {"answer": answer, "accepted_forms": accepted}
            self.assertEqual(self.case(name), case)
            for text in accepted:
                with self.subTest(name=name, text=text):
                    phase = "seed" if name == "seed" else "warm"
                    kind = "cold" if name == "seed" else "partial" if name == "edit" else "fork"
                    errors = gate.inspect_case(self.config(), phase, name, case,
                                               self.response(text, 0 if name == "seed" else 300),
                                               self.stats(kind))
                    self.assertEqual(errors, [])

    def test_literal_equation_rejects_wrong_operands_result_and_prose(self):
        case = {"answer": 4, "accepted_forms": ["4", "4.", "2 + 2 = 4", "2 + 2 = 4."]}
        for text in ["5", "2 + 2 = 5.", "1 + 3 = 4.", "3 + 1 = 4.",
                     "The answer is 4.", "2 + 2 = 4. Done.", "2+2=4", "4.0", "4!"]:
            with self.subTest(text=text):
                errors = gate.inspect_case(self.config(), "seed", "seed", case,
                                           self.response(text, 0), self.stats("cold"))
                self.assertTrue(any("arithmetic" in error for error in errors), errors)

    def test_equivalent_accepted_forms_still_require_cold_byte_parity(self):
        case = {"answer": 5, "accepted_forms": ["5", "5.", "4 + 1 = 5", "4 + 1 = 5."]}
        errors = gate.inspect_case(self.config(), "cold", "append", case,
                                   self.response("4 + 1 = 5.", 0), self.stats("cold"),
                                   self.response("5"))
        self.assertFalse(any("arithmetic" in error for error in errors), errors)
        self.assertTrue(any("cold comparison" in error for error in errors), errors)

    def test_literal_forms_keep_reasoning_tools_and_finish_strict(self):
        for field, value in [("reasoning_content", "explanation"),
                             ("tool_calls", [{"id": "tool"}]), ("finish_reason", "length")]:
            with self.subTest(field=field):
                response = self.response("2 + 2 = 4.", 0)
                if field == "finish_reason":
                    response["choices"][0][field] = value
                else:
                    response["choices"][0]["message"][field] = value
                errors = gate.inspect_case(self.config(), "seed", "seed", self.case("seed"),
                                           response, self.stats("cold"))
                self.assertTrue(errors)
                self.assertFalse(any("arithmetic" in error for error in errors), errors)

    def test_old_answer_contract_is_refused_before_http(self):
        with TemporaryDirectory() as directory:
            output = Path(directory)
            fixture = output / "fixture.json"
            gate.write_json(fixture, {"schema": "serving-reuse-live-v1"})
            gate.write_json(output / "seed.result.json", {"fixture_sha256": gate.digest(fixture)})
            manifest = output / "artifacts.json"
            manifest.write_text('{"synthetic": true}')
            argv = ["runner", "warm", "--url", "http://127.0.0.1:1", "--pid", "200",
                    "--output", str(output), "--artifact-manifest", str(manifest)]
            with patch("sys.argv", argv), patch.object(gate, "process_identity", return_value={}), \
                    patch.object(gate, "request") as request:
                with self.assertRaisesRegex(RuntimeError, "fixture contract changed"):
                    gate.main()
                request.assert_not_called()

    def test_history_keeps_actual_whitespace(self):
        body = {"model": "x", "messages": [{"role": "user", "content": "2+2"}]}
        reply = self.response(" \n4\n")
        follow = gate.follow_body(body, reply, "4+1")
        self.assertEqual(follow["messages"][1]["content"], " \n4\n")
        self.assertEqual(len(body["messages"]), 1)

    def test_high_reasoning_contract(self):
        config = self.config()
        config["reasoning_effort"] = "high"
        response = self.response("4", 0)
        for reasoning in [None, "", "  \n", [], 42]:
            with self.subTest(reasoning=reasoning):
                response["choices"][0]["message"]["reasoning_content"] = reasoning
                errors = gate.inspect_case(config, "seed", "seed", self.case("seed"),
                                           response, self.stats("cold"))
                self.assertTrue(any("reasoning" in error for error in errors), errors)
        response["choices"][0]["message"]["reasoning_content"] = "  Compute 2 + 2.\n"
        self.assertEqual(gate.inspect_case(config, "seed", "seed", self.case("seed"),
                                         response, self.stats("cold")), [])
        for field, value in [("content", "4</think>"), ("content", "The answer is 4."),
                             ("tool_calls", [{"id": "tool"}]), ("finish_reason", "length")]:
            with self.subTest(field=field, value=value):
                changed = json.loads(json.dumps(response))
                target = changed["choices"][0] if field == "finish_reason" else changed["choices"][0]["message"]
                target[field] = value
                self.assertTrue(gate.inspect_case(config, "seed", "seed", self.case("seed"),
                                                 changed, self.stats("cold")))

    def test_high_cold_parity(self):
        config = self.config()
        config["reasoning_effort"] = "high"
        warm = self.response("5")
        warm["choices"][0]["message"]["reasoning_content"] = "4 + 1.\n"
        cold = self.response("5", 0)
        cold["choices"][0]["message"]["reasoning_content"] = "4 + 1."
        errors = gate.inspect_case(config, "cold", "append", self.case("append"),
                                   cold, self.stats("cold"), warm)
        self.assertTrue(any("cold comparison" in error for error in errors), errors)
        cold["choices"][0]["message"]["reasoning_content"] += "\n"
        self.assertEqual(gate.inspect_case(config, "cold", "append", self.case("append"),
                                         cold, self.stats("cold"), warm), [])

    def test_history_keeps_reasoning(self):
        body = {"messages": [{"role": "user", "content": "2+2"}]}
        reply = self.response("4")
        reply["choices"][0]["message"]["reasoning_content"] = "  Compute.\n"
        follow = gate.follow_body(body, reply, "4+1")
        self.assertEqual(follow["messages"][1], reply["choices"][0]["message"])
        follow["messages"][1]["reasoning_content"] = "changed"
        self.assertEqual(reply["choices"][0]["message"]["reasoning_content"], "  Compute.\n")

    def test_old_resume_refused(self):
        with TemporaryDirectory() as directory:
            output = Path(directory)
            fixture = output / "fixture.json"
            manifest = output / "artifacts.json"
            manifest.write_text('{"synthetic": true}')
            argv = ["runner", "warm", "--url", "http://127.0.0.1:1", "--pid", "200",
                    "--output", str(output), "--artifact-manifest", str(manifest)]
            for version in (5, 6):
                gate.write_json(fixture, {"schema": f"serving-reuse-live-v{version}",
                                          "answer_contract": "literal-arithmetic-v2"})
                gate.write_json(output / "seed.result.json", {"fixture_sha256": gate.digest(fixture)})
                with self.subTest(version=version), patch("sys.argv", argv), \
                        patch.object(gate, "process_identity", return_value={}), \
                        patch.object(gate, "request") as request:
                    with self.assertRaisesRegex(RuntimeError, "fixture contract changed"):
                        gate.main()
                    request.assert_not_called()

    def test_seed_only_controls(self):
        with TemporaryDirectory() as directory:
            output = Path(directory)
            manifest = output / "artifacts.json"
            manifest.write_text('{"synthetic": true}')
            common = ["--url", "http://127.0.0.1:1", "--pid", "200",
                      "--output", str(output), "--artifact-manifest", str(manifest)]
            for flag, value in [("--reasoning-effort", "none"), ("--max-tokens", "32")]:
                with self.subTest(flag=flag), patch("sys.argv", ["runner", "warm", *common, flag, value]), \
                        patch.object(gate, "process_identity", return_value={}), patch.object(gate, "request") as request:
                    with self.assertRaisesRegex(RuntimeError, "frozen seed configuration"):
                        gate.main()
                    request.assert_not_called()
            for value in ["0", "-1"]:
                argv = ["runner", "seed", *common, "--max-tokens", value]
                with self.subTest(value=value), patch("sys.argv", argv), \
                        patch.object(gate, "process_identity", return_value={}), patch.object(gate, "request") as request:
                    with self.assertRaisesRegex(RuntimeError, "max-tokens must be positive"):
                        gate.main()
                    request.assert_not_called()

    def test_math_failure_is_not_cache_parity(self):
        response = self.response("4")
        errors = gate.inspect_case(self.config(), "warm", "append", self.case("append"),
                                   response, self.stats(), response)
        self.assertTrue(any("arithmetic" in error for error in errors), errors)
        self.assertFalse(any("cold comparison" in error for error in errors), errors)

    def test_edit_requires_family_specific_mechanism(self):
        for family, kind, okay in [("qwen", "partial", True), ("solar", "fork", False),
                                  ("motif", "partial", True), ("deepseek", "fork", True),
                                  ("naive", "partial", True), ("naive", "fork", False),
                                  ("iquest", "partial", True), ("iquest", "fork", False),
                                  ("deepseek", "partial", False)]:
            with self.subTest(family=family, kind=kind):
                errors = gate.inspect_case(self.config(family), "warm", "edit", self.case("edit"),
                                           self.response("6"), self.stats(kind))
                self.assertEqual(not errors, okay, errors)

    def test_motif_history_continuation_accepts_partial(self):
        for name, answer in [("append", "5"), ("fork", "8")]:
            for family, okay in [("motif", True), ("qwen", False)]:
                with self.subTest(name=name, family=family):
                    errors = gate.inspect_case(self.config(family), "warm", name, self.case(name),
                                               self.response(answer), self.stats("partial"))
                    self.assertEqual(not errors, okay, errors)

    def test_qwen_restored_partial(self):
        for mode in ("off", "on"):
            with self.subTest(mode=mode):
                config = self.config()
                config.update(mtp_mode=mode, expect_speculation=mode == "on")
                errors = gate.inspect_case(config, "restored", "restart", self.case("restart"),
                                           self.response("9"), self.stats("partial"))
                self.assertEqual(errors, [])

    def test_qwen_restore_stays_strict(self):
        for cached, fallback, reference in [(0, None, None), (320, None, None),
                                            (300, "fixture", None),
                                            (300, None, self.response("9."))]:
            with self.subTest(cached=cached, fallback=fallback, reference=reference):
                stats = self.stats("partial")
                stats["last_request"]["fallback_reason"] = fallback
                errors = gate.inspect_case(self.config(), "restored", "restart",
                                           self.case("restart"), self.response("9", cached),
                                           stats, reference)
                self.assertTrue(errors)

    def test_naive_stop_partial(self):
        for phase, name, answer in [("warm", "fork", "8"), ("restored", "restart", "9")]:
            for family, mode, okay in [("naive", "on", True), ("naive", "off", False),
                                       ("qwen", "on", phase == "restored")]:
                with self.subTest(phase=phase, family=family, mode=mode):
                    config = self.config(family)
                    config.update(mtp_mode=mode, expect_speculation=mode == "on")
                    stats = self.stats("partial")
                    stats["last_request"]["speculation_active"] = mode == "on"
                    errors = gate.inspect_case(config, phase, name, self.case(name),
                                               self.response(answer), stats)
                    self.assertEqual(not errors, okay, errors)
        self.assertFalse(gate.has_warm_fork("naive", ["partial"], []))

    def test_iquest_stop_partial(self):
        for phase, name, answer in [("warm", "append", "5"), ("warm", "fork", "8"),
                                     ("restored", "restart", "9")]:
            for mode, okay in [("on", True), ("off", False)]:
                with self.subTest(phase=phase, name=name, mode=mode):
                    config = self.config("iquest")
                    config.update(mtp_mode=mode, expect_speculation=mode == "on")
                    stats = self.stats("partial")
                    stats["last_request"]["speculation_active"] = mode == "on"
                    errors = gate.inspect_case(config, phase, name, self.case(name),
                                               self.response(answer), stats)
                    self.assertEqual(not errors, okay, errors)
        self.assertFalse(gate.has_warm_fork("iquest", ["partial"], []))
        self.assertFalse(gate.has_warm_fork("iquest", ["fork"], []))

    def test_iquest_format_failure(self):
        config = self.config("iquest")
        config.update(mtp_mode="on", expect_speculation=True)
        stats = self.stats("partial")
        stats["last_request"]["speculation_active"] = True
        text = "6\n\nI do not see a question in your message that requires reasoning."
        errors = gate.inspect_case(config, "warm", "edit", self.case("edit"),
                                   self.response(text), stats)
        self.assertTrue(any("arithmetic answer form" in error for error in errors), errors)

    def test_old_trace_refused(self):
        with TemporaryDirectory() as directory:
            output = Path(directory)
            fixture = output / "fixture.json"
            gate.write_json(fixture, {"schema": "serving-reuse-live-v3",
                                      "answer_contract": "literal-arithmetic-v2"})
            gate.write_json(output / "seed.result.json", {"fixture_sha256": gate.digest(fixture)})
            manifest = output / "artifacts.json"
            manifest.write_text('{"synthetic": true}')
            argv = ["runner", "warm", "--url", "http://127.0.0.1:1", "--pid", "200",
                    "--output", str(output), "--artifact-manifest", str(manifest)]
            with patch("sys.argv", argv), patch.object(gate, "process_identity", return_value={}), \
                    patch.object(gate, "request") as request:
                with self.assertRaisesRegex(RuntimeError, "contract changed"):
                    gate.main()
                request.assert_not_called()

    def test_partial_fork_requires_preserved_native_frontiers(self):
        line = ("ds4: Motif-3 bank reuse source=0 target=1 cached=300 partial=1 "
                "source_before=320 source_after=320 target_after=300")
        events = gate.native_forks(line, 300, 2)
        self.assertEqual(len(events), 1)
        self.assertTrue(gate.has_warm_fork("motif", ["partial"], events))
        self.assertFalse(gate.has_warm_fork("motif", ["fork"], []))
        self.assertFalse(gate.has_warm_fork("qwen", ["partial"], events))
        for bad in [line.replace("target=1", "target=0"),
                    line.replace("target=1", "target=2"),
                    line.replace("source_after=320", "source_after=300"),
                    line.replace("target_after=300", "target_after=299"),
                    line.replace("cached=300", "cached=299"), "noise " + line]:
            with self.subTest(line=bad):
                self.assertEqual(gate.native_forks(bad, 300, 2), [])

    def test_iquest_native_fork_proof(self):
        line = ("ds4: IQuest-Q1 bank reuse source=0 target=1 cached=300 partial=1 "
                "source_before=320 source_after=320 target_after=300")
        for partial in ("0", "1"):
            good = line.replace("partial=1", f"partial={partial}")
            events = gate.native_forks(good, 300, 2, "iquest")
            self.assertEqual(len(events), 1)
            self.assertTrue(gate.has_warm_fork("iquest", ["partial"], events))
            self.assertFalse(gate.has_warm_fork("motif", ["partial"], events))
        for bad in [line.replace("target=1", "target=0"),
                    line.replace("target=1", "target=2"),
                    line.replace("source_after=320", "source_after=300"),
                    line.replace("source_before=320 source_after=320", "source_before=299 source_after=299"),
                    line.replace("target_after=300", "target_after=299"),
                    line.replace("cached=300", "cached=299"),
                    line.replace("IQuest-Q1", "Motif-3"), "noise " + line]:
            with self.subTest(line=bad):
                self.assertEqual(gate.native_forks(bad, 300, 2, "iquest"), [])

    def test_iquest_v4_refused(self):
        with TemporaryDirectory() as directory:
            output = Path(directory)
            fixture = output / "fixture.json"
            gate.write_json(fixture, {"schema": "serving-reuse-live-v4",
                                      "answer_contract": "literal-arithmetic-v2"})
            gate.write_json(output / "seed.result.json", {"fixture_sha256": gate.digest(fixture)})
            manifest = output / "artifacts.json"
            manifest.write_text('{"synthetic": true}')
            argv = ["runner", "warm", "--url", "http://127.0.0.1:1", "--pid", "200",
                    "--output", str(output), "--artifact-manifest", str(manifest)]
            with patch("sys.argv", argv), patch.object(gate, "process_identity", return_value={}), \
                    patch.object(gate, "request") as request:
                with self.assertRaisesRegex(RuntimeError, "contract changed"):
                    gate.main()
                request.assert_not_called()

    def test_iquest_trace_env(self):
        with TemporaryDirectory() as directory:
            stderr = Path(directory) / "stderr.log"
            environ = Path(directory) / "environ"
            stderr.write_bytes(b"before request\n")
            paths = {"/proc/200/fd/2": stderr, "/proc/200/environ": environ}
            with patch.object(gate, "Path", side_effect=paths.__getitem__):
                for value in (b"DS4_MOTIF3_BATCH_TRACE=1\0", b"DS4_IQUEST_BATCH_TRACE=0\0"):
                    environ.write_bytes(value)
                    with self.assertRaisesRegex(RuntimeError, "DS4_IQUEST_BATCH_TRACE=1"):
                        gate.native_log_start(200, "iquest")
                environ.write_bytes(b"DS4_IQUEST_BATCH_TRACE=1\0")
                mark = gate.native_log_start(200, "iquest")
                self.assertEqual(mark[0], stderr)
                self.assertEqual(mark[2], stderr.stat().st_size)

    def test_native_log_rejects_stale_or_replaced_evidence(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "server.log"
            path.write_bytes(b"old request\n")
            info = path.stat()
            mark = path, (info.st_dev, info.st_ino), info.st_size
            with path.open("ab") as handle:
                handle.write(b"new request\n")
            self.assertEqual(gate.native_log_read(mark), b"new request\n")
            path.write_bytes(b"")
            with self.assertRaisesRegex(RuntimeError, "truncated"):
                gate.native_log_read(mark)
            path.rename(path.with_suffix(".old"))
            path.write_bytes(b"replacement\n")
            with self.assertRaisesRegex(RuntimeError, "file changed"):
                gate.native_log_read(mark)

    def test_cold_requires_zero_cache_and_exact_output(self):
        warm = self.response("5")
        cold = self.response(" 5", 0)
        errors = gate.inspect_case(self.config(), "cold", "append", self.case("append"),
                                   cold, self.stats("cold"), warm)
        self.assertTrue(any("cold comparison" in error for error in errors), errors)
        cold = self.response("5", 2)
        errors = gate.inspect_case(self.config(), "cold", "append", self.case("append"),
                                   cold, self.stats("cold"), warm)
        self.assertTrue(any("cached" in error for error in errors), errors)

    def test_mtp_activity_multiple(self):
        config = self.config()
        config.update(mtp_mode="on", expect_speculation=True, mtp_draft=2)
        response = self.response("8.")
        response["usage"]["completion_tokens"] = 2
        errors = gate.inspect_case(config, "warm", "fork", self.case("fork"),
                                   response, self.stats())
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
        self.run_four_phases(fork_on_append=True)

    def test_all_phases_record_literal_forms_and_preserve_equation_bytes(self):
        self.run_four_phases(fork_on_append=True, equations=True)

    def test_warm_requires_an_observed_bank_fork(self):
        self.run_four_phases(fork_on_append=False)

    def test_high_four_phases(self):
        self.run_four_phases(fork_on_append=True, reasoning_effort="high", max_tokens=256)

    def test_mtp_probe_is_executed(self):
        self.run_four_phases(fork_on_append=True, mtp_mode="on")

    def test_mtp_lifecycle_needs_work(self):
        for tokens in (1, 10):
            with self.subTest(tokens=tokens):
                self.run_four_phases(fork_on_append=True, mtp_mode="on",
                                     probe_active=False, probe_tokens=tokens)

    def test_mtp_probe_needs_tokens(self):
        self.run_four_phases(fork_on_append=True, mtp_mode="on", probe_tokens=1)

    def test_mtp_probe_missing_activity(self):
        self.run_four_phases(fork_on_append=True, mtp_mode="on", probe_active=None)

    def test_probe_text_is_exact(self):
        config = self.config()
        config.update(mtp_mode="on", expect_speculation=True, mtp_draft=2)
        stats = self.stats("cold")
        stats["last_request"]["speculation_active"] = True
        for text in (" 1234567890", "1234567890 ", "1234567890\n", "\n1234567890"):
            with self.subTest(text=text):
                response = self.response(text, 0)
                response["usage"]["completion_tokens"] = 2
                errors = gate.inspect_case(config, "cold", gate.MTP_PROBE,
                                           self.case(gate.MTP_PROBE), response, stats)
                self.assertTrue(any("MTP probe answer form" in error for error in errors), errors)

    def test_mtp_on_requires_activity(self):
        with TemporaryDirectory() as directory:
            output = Path(directory)
            manifest = output / "artifacts.json"
            manifest.write_text('{"synthetic": true}')
            argv = ["runner", "seed", "--url", "http://127.0.0.1:1", "--pid", "100",
                    "--output", str(output), "--artifact-manifest", str(manifest),
                    "--family", "qwen", "--model", "fixture", "--context", "2048",
                    "--banks", "2", "--native-chunk", "64", "--lane", "continuous",
                    "--mtp-mode", "on", "--mtp-draft", "2", "--expect-speculation", "off"]
            with patch("sys.argv", argv), patch.object(gate, "process_identity", return_value={}), \
                    patch.object(gate, "request", side_effect=AssertionError("unexpected HTTP")) as request:
                with self.assertRaisesRegex(RuntimeError, "on mode must expect speculation"):
                    gate.main()
                request.assert_not_called()

    def run_four_phases(self, fork_on_append, equations=False, reasoning_effort="none", max_tokens=32,
                        mtp_mode="off", probe_active=True, probe_tokens=10):
        with TemporaryDirectory() as directory:
            output = Path(directory) / "evidence"
            manifest = Path(directory) / "artifacts.json"
            manifest.write_text('{"synthetic": true}')
            state = {"phase": "seed", "count": 0, "pid": 100, "kind": "cold",
                     "active": False, "probe_seen": False}

            def identity(pid):
                return {"pid": pid, "start_ticks": str(pid), "boot_id": "test",
                        "executable_sha256": "fixed-binary"}

            def stats():
                cold = state["phase"] == "cold"
                trace = self.stats(state["kind"])["last_request"]
                trace["speculation_active"] = state["active"]
                return {"routes": {"chat": state["count"]},
                        "last_request": trace,
                        "serving": {"family": "qwen4exp", "requested": {
                            "lane": "auto", "mtp_mode": mtp_mode, "ctx": 2048,
                            "max_seqs": "2", "prefix_reuse": "off" if cold else "partial",
                            "backend": "cuda"}, "effective": {
                            "ctx": 2048, "max_seqs": 2, "native_chunk": 64,
                            "sched_chunk": 64, "sched_chunk_live": 64, "mtp_mode": mtp_mode,
                            "mtp_draft": 2 if mtp_mode == "on" else None,
                            "prefix_reuse": "off" if cold else "partial", "disk": not cold,
                            "bank_persist_min_tokens": 1, "disk_min_tokens": 1}, "issues": []}}

            def request(url, path, body=None):
                if path == "/v1/stats":
                    return stats()
                self.assertEqual(body["reasoning_effort"], reasoning_effort)
                self.assertEqual(body["max_tokens"], max_tokens)
                state["count"] += 1
                question = body["messages"][-1]["content"]
                if "1234567890" in question:
                    self.assertEqual(state["phase"], "cold")
                    self.assertEqual(mtp_mode, "on")
                    state.update(kind="cold", active=probe_active, probe_seen=True)
                    response = self.response("1234567890", 0)
                    response["usage"]["completion_tokens"] = probe_tokens
                    if reasoning_effort == "high":
                        response["choices"][0]["message"]["reasoning_content"] = "Copy the digits."
                    return response
                expression, answer = next((expression, n) for expression, n in [("2 + 2", 4), ("4 + 1", 5),
                              ("4 + 2", 6), ("5 + 3", 8), ("8 + 1", 9)] if expression in question)
                state["kind"] = ("cold" if state["phase"] in ("seed", "cold")
                                 else "partial" if answer == 6
                                 else "fork" if answer == 5 and fork_on_append
                                 else "exact")
                cached = 0 if state["kind"] == "cold" else 300
                # All arithmetic stops precede the first draft, even with MTP on.
                state["active"] = False
                text = f"{expression} = {answer}." if equations else str(answer)
                response = self.response(f" \n{text}\n", cached)
                if reasoning_effort == "high":
                    response["choices"][0]["message"]["reasoning_content"] = f"  Compute {expression}.\n"
                return response

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
                                 "--banks", "2", "--native-chunk", "64", "--mtp-mode", mtp_mode,
                                 "--expect-speculation", mtp_mode, "--lane", "continuous"]
                        if mtp_mode == "on":
                            argv += ["--mtp-draft", "2"]
                        if reasoning_effort == "high":
                            argv += ["--reasoning-effort", reasoning_effort, "--max-tokens", str(max_tokens)]
                    with patch("sys.argv", argv):
                        status = gate.main()
                    if phase == "warm" and not fork_on_append:
                        self.assertEqual(status, 1)
                        errors = json.loads((output / "warm.result.json").read_text())["errors"]
                        self.assertTrue(any("no bank fork" in error for error in errors), errors)
                        return
                    if phase == "cold" and mtp_mode == "on" and (probe_active is not True or probe_tokens < 2):
                        self.assertEqual(status, 1)
                        receipt = gate.read_json(output / "cold.result.json")
                        self.assertFalse(receipt["passed"])
                        self.assertTrue(any("mtp_probe" in error for error in receipt["errors"]), receipt)
                        self.assertTrue(state["probe_seen"])
                        return
                    self.assertEqual(status, 0, phase)
                    self.assertTrue(json.loads((output / f"{phase}.result.json").read_text())["passed"])
            self.assertEqual(state["probe_seen"], mtp_mode == "on")
            if mtp_mode == "on":
                self.assertEqual(gate.read_json(output / "cold.mtp_probe.request.json"),
                                 gate.read_json(output / "seed.mtp_probe.request.json"))
            fixture = json.loads((output / "fixture.json").read_text())
            self.assertEqual(fixture["schema"], "serving-reuse-live-v7")
            self.assertEqual(fixture["config"]["reasoning_effort"], reasoning_effort)
            self.assertEqual(fixture["config"]["max_tokens"], max_tokens)
            seed_text = " \n2 + 2 = 4.\n" if equations else " \n4\n"
            self.assertEqual(fixture["cases"]["append"]["body"]["messages"][1]["content"], seed_text)
            if reasoning_effort == "high":
                self.assertEqual(fixture["cases"]["append"]["body"]["messages"][1]["reasoning_content"],
                                 "  Compute 2 + 2.\n")
            self.assertEqual(fixture["answer_contract"], "literal-arithmetic-v2")
            for name, case in fixture["cases"].items():
                self.assertEqual(case["accepted_forms"], self.case(name)["accepted_forms"])
                phase = "cold" if name == "mtp_probe" else gate.reference_phase(name)
                receipt = gate.read_json(output / f"{phase}.{name}.summary.json")
                self.assertEqual(receipt["accepted_forms"], case["accepted_forms"])
            self.assertEqual(len(fixture["cases"]["restart"]["body"]["messages"]), 7)


if __name__ == "__main__":
    unittest.main()
