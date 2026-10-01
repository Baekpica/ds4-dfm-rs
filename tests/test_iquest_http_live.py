"""Synthetic wire checks only; never opens HTTP or loads a model."""
import contextlib
import copy
import io
import json
from pathlib import Path
import tempfile
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import iquest_http_live as gate


def answer(surface, text="42"):
    if surface in ("chat", "completion"):
        return {"choices": [{"finish_reason": "stop", "message": {"content": text}, "text": text}],
                "usage": {"prompt_tokens": 20, "completion_tokens": 2}}
    if surface == "responses":
        return {"status": "completed", "output": [{"id": "msg_fixture", "type": "message",
                "content": [{"type": "output_text", "text": text}]}],
                "usage": {"input_tokens": 20, "output_tokens": 2}}
    return {"stop_reason": "end_turn", "content": [{"type": "text", "text": text}],
            "usage": {"input_tokens": 20, "output_tokens": 2}}


def events(surface):
    if surface in ("chat", "completion"):
        choice = {"delta": {"content": "42"}} if surface == "chat" else {"text": "42"}
        return [{"choices": [choice]}, {"choices": [{"finish_reason": "stop"}]},
                {"choices": [], "usage": answer(surface)["usage"]}, "[DONE]"]
    if surface == "responses":
        return [{"type": "response.output_text.delta", "delta": "42"},
                {"type": "response.completed", "response": answer(surface)}]
    return [{"type": "message_start", "message": {"usage": {"input_tokens": 20}}},
            {"type": "content_block_delta", "delta": {"type": "text_delta", "text": "42"}},
            {"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 2}},
            {"type": "message_stop"}]


def wire(values):
    return "".join("data: " + (value if isinstance(value, str) else json.dumps(value)) + "\n\n"
                   for value in values).encode()


def snapshot():
    return {"stats": {"memory": {"census_faults": 0, "observation": {"errors": 0}},
                      "governor": {"faults": 0}, "last_request": {"speculation_active": False}},
            "metrics": dict.fromkeys((*gate.FAULT_METRICS, "ds4_spec_drafts_total", "ds4_spec_hits_total"), 0)}


def reasoned(surface, thought="19 plus 23 equals 42."):
    response = answer(surface)
    if surface == "chat":
        response["choices"][0]["message"]["reasoning_content"] = thought
    elif surface == "responses":
        response["output"].insert(0, {"id": "reason_fixture", "type": "reasoning", "status": "completed",
                                      "summary": [{"type": "summary_text", "text": thought}]})
        response["usage"]["output_tokens_details"] = {"reasoning_tokens": 6}
    else:
        response["content"].insert(0, {"type": "thinking", "thinking": thought, "signature": "fixture"})
    return response


def reasoning_events(surface):
    records = events(surface)
    thought = "19 plus 23 equals 42."
    if surface == "chat":
        records.insert(0, {"choices": [{"delta": {"reasoning_content": thought}}]})
    elif surface == "responses":
        records.insert(0, {"type": "response.reasoning_summary_text.delta", "delta": thought})
        records[-1]["response"] = reasoned(surface)
    else:
        records.insert(1, {"type": "content_block_start", "index": 0,
                           "content_block": {"type": "thinking", "thinking": ""}})
        records.insert(2, {"type": "content_block_delta", "index": 0,
                           "delta": {"type": "thinking_delta", "thinking": thought}})
    return records


class WireChecks(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.g = object.__new__(gate.Gate)
        self.g.out = Path(self.temp.name)
        self.g.model = "fixture-only"
        self.g.report = {"cases": []}
        self.g.args = SimpleNamespace(keep_going=False, mode="basic", reasoning_effort="none")

    def test_captured_anthropic_usage(self):
        # Captured MTP-off response: Anthropic reports cache creation separately.
        response = {
            "role": "assistant", "stop_reason": "end_turn",
            "content": [
                {"type": "thinking", "thinking": "The user wants an integer answer without explanation. Simple addition: 19+23=42.", "signature": ""},
                {"type": "text", "text": "42"},
            ],
            "usage": {"input_tokens": 0, "output_tokens": 21,
                      "cache_read_input_tokens": 0, "cache_creation_input_tokens": 22},
        }
        result = self.g.answer("anthropic", response, "42", gate.Reasoning.REQUIRED)
        self.assertEqual(result["usage"]["input_tokens"], 0)
        self.assertEqual(result["usage"]["cache_creation_input_tokens"], 22)

    def test_anthropic_input_parts(self):
        for usage in ({"input_tokens": 0, "cache_read_input_tokens": 20, "output_tokens": 2},
                      {"input_tokens": 0, "cache_creation_input_tokens": 20, "output_tokens": 2}):
            response = answer("anthropic")
            response["usage"] = usage
            self.g.answer("anthropic", response, "42")
        for usage in ({"input_tokens": 0, "output_tokens": 2},
                      {"input_tokens": -1, "cache_read_input_tokens": 20, "output_tokens": 2},
                      {"input_tokens": 0, "cache_creation_input_tokens": True, "output_tokens": 2},
                      {"input_tokens": 0, "cache_creation_input_tokens": 20, "output_tokens": 0}):
            response = answer("anthropic")
            response["usage"] = usage
            with self.subTest(usage=usage), self.assertRaises(RuntimeError):
                self.g.answer("anthropic", response, "42")

    def test_keep_going_records_all(self):
        self.g.args.keep_going = True
        self.g.startup = Mock()
        self.g.before = snapshot()
        self.g.snapshot = Mock(return_value=snapshot())
        self.g.no_faults = Mock()
        def cases():
            self.g.case("bad_first", Mock(side_effect=RuntimeError("strict answer failed")))
            self.g.case("good", lambda: {"text": "42"})
            self.g.case("bad_last", Mock(side_effect=RuntimeError("terminal failed")))
        self.g.basic = cases
        with contextlib.redirect_stdout(io.StringIO()), self.assertRaisesRegex(RuntimeError, "2 case"):
            self.g.run()
        self.assertEqual([v["passed"] for v in self.g.report["cases"]], [False, True, False])
        self.assertFalse(self.g.report["passed"])
        self.assertTrue(self.g.report["global_fault_check"]["passed"])
        self.g.snapshot.assert_any_call("after")
        self.g.no_faults.assert_called_once()
        self.assertIn("strict answer failed", (self.g.out / "bad_first.result.json").read_text())

    def test_keep_going_global_faults(self):
        self.g.args.keep_going = True
        self.g.startup = Mock()
        self.g.before = snapshot()
        self.g.snapshot = Mock(return_value=snapshot())
        self.g.no_faults = Mock(side_effect=RuntimeError("fault counter changed"))
        self.g.basic = lambda: self.g.case("good", lambda: {})
        with contextlib.redirect_stdout(io.StringIO()), self.assertRaisesRegex(RuntimeError, "fault counter changed"):
            self.g.run()
        self.assertFalse(self.g.report["passed"])
        self.assertFalse(self.g.report["global_fault_check"]["passed"])

    def test_case_default_fails_fast(self):
        self.g.snapshot = Mock(return_value=snapshot())
        with contextlib.redirect_stdout(io.StringIO()), self.assertRaisesRegex(RuntimeError, "strict"):
            self.g.case("bad", Mock(side_effect=RuntimeError("strict")))
        self.assertFalse(self.g.report["cases"][0]["passed"])

    def test_startup_always_fatal(self):
        self.g.args.keep_going = True
        self.g.startup = Mock(side_effect=RuntimeError("startup"))
        self.g.basic = Mock()
        with self.assertRaisesRegex(RuntimeError, "startup"):
            self.g.run()
        self.g.basic.assert_not_called()
        self.assertFalse(self.g.report["passed"])

    def test_cli_keep_going(self):
        for extra, expected in (([], False), (["--keep-going"], True)):
            with patch("sys.argv", ["gate", "basic", "--output", "unused", *extra]), \
                    patch.object(gate, "Gate") as constructor:
                gate.main()
                self.assertEqual(constructor.call_args.args[0].keep_going, expected)

    def test_request_controls(self):
        for surface in gate.PATHS:
            body = self.g.body(surface, gate.ARITHMETIC, "sse")
            self.assertEqual(body["temperature"], 0)
            self.assertTrue(body["stream"])
        self.assertEqual(self.g.body("chat", "X")["reasoning_effort"], "none")
        self.assertEqual(self.g.body("responses", "X")["reasoning"], {"effort": "none"})
        self.assertEqual(self.g.body("anthropic", "X")["thinking"], {"type": "disabled"})
        self.assertEqual(self.g.body("completion", "X")["prompt"],
                         "<|iquest_user|>X<|iquest_end|><|iquest_assistant|><think></think>")

    def test_nonfinite_rejected(self):
        for raw in (b'{"x":NaN}', b'{"x":Infinity}', b'{"x":1e999}'):
            with self.assertRaises((RuntimeError, ValueError)):
                gate.decode_json(raw)

    def test_number_contract_strict(self):
        for surface in gate.PATHS:
            self.assertEqual(self.g.answer(surface, answer(surface), "42")["text"], "42")
            for text in ("42. Sorry.", "19 + 23 = 42", "42\n\nI do not see a question.", "<think></think>42"):
                with self.subTest(surface=surface, text=text), self.assertRaises(RuntimeError):
                    self.g.answer(surface, answer(surface, text), "42")

    def test_sse_surfaces(self):
        for surface in gate.PATHS:
            self.g.exchange = Mock(return_value=({"status": 200, "streaming": True}, wire(events(surface))))
            response, _ = self.g.generate(surface, surface, {"stream": True})
            self.assertEqual(self.g.answer(surface, response, "42")["text"], "42")

    def test_sse_terminal_required(self):
        for surface in gate.PATHS:
            self.g.exchange = Mock(return_value=({"status": 200, "streaming": True}, wire(events(surface)[:-1])))
            with self.subTest(surface=surface), self.assertRaises(RuntimeError):
                self.g.generate(surface, surface, {"stream": True})

    def test_responses_delta_matches(self):
        records = events("responses")
        records[0]["delta"] = "43"
        self.g.exchange = Mock(return_value=({"status": 200, "streaming": True}, wire(records)))
        with self.assertRaisesRegex(RuntimeError, "delta/final mismatch"):
            self.g.generate("mismatch", "responses", {"stream": True})

    def test_tool_ids_and_history(self):
        call = {"id": "call_fixture_chat", "type": "function", "function": {
            "name": "add", "arguments": '{"a":19,"b":23}'}}
        message = {"role": "assistant", "content": None, "tool_calls": [call]}
        output = [{"id": "fc_fixture", "call_id": "call_fixture_response", "type": "function_call",
                   "status": "completed", "name": "add", "arguments": '{"a":19,"b":23}'}]
        generated = [{"choices": [{"finish_reason": "tool_calls", "message": message}]}, answer("chat"),
                     {"status": "completed", "output": output}, answer("responses")]
        self.g.generate = Mock(side_effect=[(response, {}) for response in generated])
        self.g.snapshot = Mock(side_effect=lambda _: snapshot())
        with contextlib.redirect_stdout(io.StringIO()):
            self.g.tools()
        calls = self.g.generate.call_args_list
        chat_follow, response_follow = calls[1].args[2], calls[3].args[2]
        self.assertEqual(chat_follow["messages"][-2], message)
        self.assertEqual(chat_follow["messages"][-1], {"role": "tool", "tool_call_id": call["id"], "content": "42"})
        self.assertEqual(response_follow["input"][-2], output[0])
        self.assertEqual(response_follow["input"][-1], {"type": "function_call_output", "call_id": output[0]["call_id"], "output": "42"})
        self.assertNotIn("previous_response_id", response_follow)
        for index in (0, 2):
            self.assertEqual(calls[index].args[2]["tool_choice"], "required")
            self.assertEqual(len(calls[index].args[2]["tools"]), 1)
        self.assertTrue(all(item["passed"] for item in self.g.report["cases"]))

    def test_sampling_no_speculation(self):
        self.g.snapshot = Mock(side_effect=lambda _: snapshot())
        self.g.generate = Mock(return_value=(answer("chat"), {}))
        with contextlib.redirect_stdout(io.StringIO()):
            self.g.sampling()
        self.assertGreater(self.g.generate.call_args.args[2]["temperature"], 0)
        self.assertTrue(self.g.report["cases"][0]["passed"])

    def test_workload_reasoning_controls(self):
        for effort, expected in (("none", gate.Reasoning.DISABLED), ("high", gate.Reasoning.REQUIRED)):
            self.g.args.reasoning_effort = effort
            for mode, budget in (("buffered", 32), ("sse", 128)):
                body, reasoning = self.g.workload_body("fixed prompt", mode, budget)
                self.assertEqual(reasoning, expected)
                self.assertEqual(body["messages"][0]["content"], "fixed prompt")
                self.assertEqual(body["reasoning_effort"], effort)
                self.assertEqual(body["max_tokens"], 256 if effort == "high" else budget)
                self.assertEqual(body["stream"], mode == "sse")
        self.assertEqual(self.g.body("chat", "basic prompt")["reasoning_effort"], "none")
        self.assertEqual(self.g.body("responses", "tool prompt")["reasoning"], {"effort": "none"})

    def test_sampling_high_reasoning(self):
        self.g.args.reasoning_effort = "high"
        self.g.snapshot = Mock(side_effect=lambda _: snapshot())
        self.g.generate = Mock(return_value=(reasoned("chat"), {}))
        with contextlib.redirect_stdout(io.StringIO()):
            self.g.sampling()
        call = self.g.generate.call_args
        self.assertEqual(call.args[3], gate.Reasoning.REQUIRED)
        self.assertEqual(call.args[2]["max_tokens"], 256)
        self.assertEqual(call.args[2]["reasoning_effort"], "high")
        self.assertEqual(call.args[2]["temperature"], 0.7)
        self.assertEqual(self.g.report["cases"][0]["text"], "42")
        for response in (answer("chat"), reasoned("chat")):
            if "reasoning_content" in response["choices"][0]["message"]:
                response["choices"][0]["message"]["content"] = "The answer is 42."
            self.g.generate = Mock(return_value=(response, {}))
            with contextlib.redirect_stdout(io.StringIO()), self.assertRaises(RuntimeError):
                self.g.sampling()

    def test_concurrency_high_reasoning(self):
        self.g.args.reasoning_effort = "high"
        polled = threading.Event()
        def state(name):
            value = snapshot()
            value["stats"].update(queue_depth=0, routes={"openai_chat_continuous": 2 if name.endswith("after") else 0})
            value["metrics"].update(ds4_requests_inflight=2, ds4_banks_live=2)
            if ".poll." in name:
                polled.set()
            return value
        def generate(name, surface, body, reasoning=gate.Reasoning.DISABLED):
            self.assertTrue(polled.wait(2))
            self.assertEqual(reasoning, gate.Reasoning.REQUIRED)
            self.assertEqual(body["reasoning_effort"], "high")
            self.assertEqual(body["max_tokens"], 256)
            self.assertEqual(body["temperature"], 0)
            self.assertTrue(body["stream"])
            self.assertEqual(body["stream_options"], {"include_usage": True})
            response = reasoned("chat")
            response["choices"][0]["message"]["content"] = gate.SEQUENCE
            return response, {"start_monotonic": 1, "first_text_monotonic": 2, "end_monotonic": 3}
        self.g.snapshot = Mock(side_effect=state)
        self.g.generate = Mock(side_effect=generate)
        with contextlib.redirect_stdout(io.StringIO()):
            self.g.concurrency()
        self.assertTrue(self.g.report["cases"][0]["passed"])
        self.assertEqual(self.g.generate.call_count, 2)
        for item in self.g.report["cases"][0]["workers"].values():
            self.assertEqual(item["text"], gate.SEQUENCE)
            self.assertTrue(item["reasoning"])

    def test_cli_workload_reasoning(self):
        for extra, effort in (([], "none"), (["--reasoning-effort", "high"], "high")):
            with patch("sys.argv", ["gate", "sampling", "--output", "unused", *extra]), \
                    patch.object(gate, "Gate") as constructor:
                gate.main()
                self.assertEqual(constructor.call_args.args[0].reasoning_effort, effort)

    def test_fault_delta_rejected(self):
        before = snapshot()
        after = copy.deepcopy(before)
        after["metrics"][gate.FAULT_METRICS[0]] += 1
        with self.assertRaisesRegex(RuntimeError, "fault counter changed"):
            self.g.no_faults(before, after)

    def test_reasoning_controls(self):
        for surface in ("chat", "responses", "anthropic"):
            body = self.g.reasoning_body(surface, "sse")
            budget = body["max_output_tokens" if surface == "responses" else "max_tokens"]
            self.assertEqual(budget, 256)
            self.assertTrue(body["stream"])
        self.assertEqual(self.g.reasoning_body("chat", "buffered")["reasoning_effort"], "high")
        self.assertEqual(self.g.reasoning_body("responses", "buffered")["reasoning"],
                         {"effort": "high", "summary": "auto"})
        self.assertEqual(self.g.reasoning_body("anthropic", "buffered")["thinking"], {"type": "enabled"})

    def test_reasoning_buffers(self):
        for surface in ("chat", "responses", "anthropic"):
            result = self.g.answer(surface, reasoned(surface), "42", gate.Reasoning.REQUIRED)
            self.assertTrue(result["reasoning"].strip())
            with self.assertRaises(RuntimeError):
                self.g.answer(surface, reasoned(surface), "42")

    def test_reasoning_sse(self):
        for surface in ("chat", "responses", "anthropic"):
            raw = wire(reasoning_events(surface))
            self.g.exchange = Mock(return_value=({"status": 200, "streaming": True}, raw))
            response, _ = self.g.generate(surface, surface, {"stream": True}, gate.Reasoning.REQUIRED)
            self.assertEqual(self.g.answer(surface, response, "42", gate.Reasoning.REQUIRED)["reasoning"],
                             "19 plus 23 equals 42.")
            with self.assertRaises(RuntimeError):
                self.g.generate(surface, surface, {"stream": True})

    def test_reasoning_strict(self):
        for surface in ("chat", "responses", "anthropic"):
            for thought in ("", " \n", "<think>19+23", "19+23</think>"):
                with self.subTest(surface=surface, thought=thought), self.assertRaises(RuntimeError):
                    self.g.answer(surface, reasoned(surface, thought), "42", gate.Reasoning.REQUIRED)
        response = reasoned("responses")
        response["output"][0]["status"] = "incomplete"
        with self.assertRaises(RuntimeError):
            self.g.answer("responses", response, "42", gate.Reasoning.REQUIRED)

    def test_startup_mtp_off(self):
        self.g.args = SimpleNamespace(ctx=8192, banks=2, mtp_mode="off", model=None)
        self.g.exchange = Mock(return_value=({"status": 404}, b""))
        self.g.get_json = Mock(return_value={"data": [{"id": "fixture-only", "context_length": 8192}]})
        state = snapshot()
        state["stats"]["serving"] = {"family": "iquest_q1", "effective": {
            "ctx": 8192, "max_seqs": 2, "mtp_mode": "off"}, "issues": []}
        self.g.snapshot = Mock(return_value=state)
        self.g.startup()
        state["stats"]["serving"]["effective"]["mtp_mode"] = "on"
        with self.assertRaisesRegex(RuntimeError, "mtp_mode"):
            self.g.startup()

    def test_cli_mtp_modes(self):
        for extra, expected in (([], "on"), (["--mtp-mode", "off"], "off")):
            with patch("sys.argv", ["gate", "reasoning", "--output", "unused", *extra]), \
                    patch.object(gate, "Gate") as constructor:
                gate.main()
                self.assertEqual(constructor.call_args.args[0].mtp_mode, expected)
                self.assertEqual(constructor.call_args.args[0].mode, "reasoning")


if __name__ == "__main__":
    unittest.main()
