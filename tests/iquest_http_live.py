#!/usr/bin/env python3
"""Bounded IQuest HTTP gates against an externally managed localhost server.

No server lifecycle or GPU control. Each output directory is single use. Fixed
prompts, raw wire bodies, timings, stats, and failures remain reviewable.
"""
# Frozen harness v3 source SHA256: 25fce13bc872ee3c55671853cae405fdd0cfc397e1bd9fba07013d53bdce969a

import argparse
import copy
from enum import Enum
import hashlib
import json
import math
from pathlib import Path
import re
import sys
import threading
import time
import traceback
import urllib.error
import urllib.parse
import urllib.request

HTTP_OK = 200
HTTP_NOT_FOUND = 404
REASONING_BUDGET = 256
ARITHMETIC = "What is 19 + 23? Reply with only the integer answer, without explanation."
TOOL_PROMPT = ("Call the add tool exactly once with a=19 and b=23. Do not calculate "
               "the answer yourself. After the tool result, reply with only the integer result.")
SEQUENCE = " ".join(str(n) for n in range(1, 25))
SCHEMA = {"type": "object", "properties": {"a": {"type": "integer"},
          "b": {"type": "integer"}}, "required": ["a", "b"], "additionalProperties": False}
DELIMITERS = ("<|iquest_", "<iquest_tool_call", "</iquest_tool_call", "<arg_key",
              "</arg_key", "<arg_value", "</arg_value", "<think", "</think",
              "<|im_", "<|end", "<|begin")
PATHS = {"chat": "/v1/chat/completions", "responses": "/v1/responses",
         "anthropic": "/v1/messages", "completion": "/v1/completions"}
FAULT_METRICS = ("ds4_memory_census_faults_total", "ds4_memory_governor_faults_total",
                 "ds4_cont_batch_failures_total", "ds4_requests_total{outcome=\"failed\"}",
                 "ds4_requests_total{outcome=\"canceled\"}")


class Reasoning(Enum):
    DISABLED = "disabled"
    REQUIRED = "required"


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2) + "\n")


def decode_json(raw):
    def bad_constant(value):
        raise ValueError(f"nonfinite JSON constant: {value}")
    value = json.loads(raw, parse_constant=bad_constant)
    finite(value)
    return value


def finite(value):
    if isinstance(value, float):
        require(math.isfinite(value), "nonfinite response number")
    elif isinstance(value, dict):
        for child in value.values():
            finite(child)
    elif isinstance(value, list):
        for child in value:
            finite(child)


def clean(text):
    require(isinstance(text, str), "visible text is not a string")
    require(not any(marker in text for marker in DELIMITERS), f"internal delimiter leaked: {text!r}")


def metric_values(raw):
    values = {}
    for line in raw.decode().splitlines():
        if not line or line.startswith("#"):
            continue
        key, value = line.rsplit(" ", 1)
        values[key] = float(value)
        require(math.isfinite(values[key]), f"nonfinite metric: {key}")
    return values


def sse_records(raw):
    records = []
    for block in raw.decode().replace("\r\n", "\n").split("\n\n"):
        data = "\n".join(line[5:].lstrip(" ") for line in block.splitlines() if line.startswith("data:"))
        if not data:
            continue
        event = next((line[6:].strip() for line in block.splitlines() if line.startswith("event:")), None)
        records.append({"event": event, "data": "[DONE]" if data == "[DONE]" else decode_json(data)})
    return records


def semantic_delta(value):
    if not isinstance(value, dict):
        return ""
    choices = value.get("choices", [])
    if choices:
        return choices[0].get("delta", {}).get("content") or choices[0].get("text") or ""
    if value.get("type") == "response.output_text.delta":
        return value.get("delta", "")
    if value.get("type") == "content_block_delta":
        return value.get("delta", {}).get("text", "")
    return ""


def reasoning_delta(value):
    choices = value.get("choices", [])
    if choices:
        return choices[0].get("delta", {}).get("reasoning_content") or ""
    if value.get("type") == "response.reasoning_summary_text.delta":
        return value.get("delta", "")
    if value.get("type") == "content_block_delta":
        return value.get("delta", {}).get("thinking", "")
    return ""


class Gate:
    def __init__(self, args):
        self.args = args
        self.out = args.output
        require(not self.out.exists() or not any(self.out.iterdir()), "output directory must be new or empty")
        self.out.mkdir(parents=True, exist_ok=True)
        self.opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        self.report = {"version": 3, "scope": "actual localhost HTTP workload; no source-model quality claim",
                       "arguments": {key: str(value) if isinstance(value, Path) else value
                                     for key, value in vars(args).items()}, "cases": []}
        self.report["script_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        write_json(self.out / "fixed-contracts.json", {"arithmetic_prompt": ARITHMETIC,
                   "arithmetic_answer": "42", "tool_prompt": TOOL_PROMPT, "tool_schema": SCHEMA,
                   "tool_arguments": {"a": 19, "b": 23}, "tool_result": 42,
                   "concurrency_answer": SEQUENCE, "reasoning_prompt": ARITHMETIC,
                   "reasoning_answer": "42", "reasoning_output_budget": REASONING_BUDGET})

    def exchange(self, name, path, body=None):
        write_json(self.out / f"{name}.request.json", {"path": path, "body": body})
        payload = None if body is None else json.dumps(body, allow_nan=False).encode()
        request = urllib.request.Request(self.args.url + path, data=payload,
                    headers={"Content-Type": "application/json", "anthropic-version": "2023-06-01"})
        timing = {"start_wall": time.time(), "start_monotonic": time.monotonic()}
        response = None
        raw = bytearray()
        try:
            try:
                response = self.opener.open(request, timeout=self.args.timeout)
            except urllib.error.HTTPError as error:
                response = error
            timing.update(status=response.status, headers=dict(response.headers),
                          headers_monotonic=time.monotonic())
            streaming = "text/event-stream" in response.headers.get("Content-Type", "")
            with (self.out / f"{name}.response.raw").open("wb") as output:
                while True:
                    piece = response.readline() if streaming else response.read(64 * 1024)
                    if not piece:
                        break
                    raw.extend(piece)
                    output.write(piece)
                    output.flush()
                    if streaming and piece.startswith(b"data:") and "first_text_monotonic" not in timing:
                        fragment = piece[5:].strip()
                        if fragment and fragment != b"[DONE]" and semantic_delta(decode_json(fragment)):
                            timing["first_text_monotonic"] = time.monotonic()
            timing["streaming"] = streaming
            return timing, bytes(raw)
        except Exception as error:
            timing["error"] = str(error)
            raise
        finally:
            if response is not None:
                response.close()
            timing.update(end_monotonic=time.monotonic(), end_wall=time.time(), bytes=len(raw),
                          sha256=hashlib.sha256(raw).hexdigest())
            timing["elapsed_seconds"] = timing["end_monotonic"] - timing["start_monotonic"]
            write_json(self.out / f"{name}.transport.json", timing)

    def get_json(self, name, path):
        timing, raw = self.exchange(name, path)
        require(timing["status"] == HTTP_OK, f"{name}: HTTP {timing['status']}")
        value = decode_json(raw)
        write_json(self.out / f"{name}.json", value)
        return value

    def snapshot(self, name):
        stats = self.get_json(f"{name}.stats", "/v1/stats")
        timing, raw = self.exchange(f"{name}.metrics", "/metrics")
        require(timing["status"] == HTTP_OK, "metrics unavailable")
        metrics = metric_values(raw)
        require(stats["memory"]["census_supported"], "native memory census unavailable")
        return {"stats": stats, "metrics": metrics}

    def no_faults(self, before, after):
        for key in FAULT_METRICS:
            require(key in before["metrics"] and key in after["metrics"], f"missing metric {key}")
            require(before["metrics"][key] == after["metrics"][key], f"fault counter changed: {key}")
        for path in (("memory", "census_faults"), ("governor", "faults"),
                     ("memory", "observation", "errors")):
            left, right = before["stats"], after["stats"]
            for key in path:
                left, right = left[key], right[key]
            require(left == right, f"fault counter changed: {'.'.join(path)}")

    def startup(self):
        health, raw = self.exchange("health", "/health")
        require(health["status"] in (HTTP_OK, HTTP_NOT_FOUND), f"unexpected health HTTP {health['status']}")
        self.report["health"] = {"http_status": health["status"], "supported": health["status"] == HTTP_OK,
                                 "note": "Current production router has no /health; models and inference are readiness gates."}
        models = self.get_json("models", "/v1/models")
        require(models["data"], "empty model list")
        self.model = self.args.model or models["data"][0]["id"]
        advertised = next((m for m in models["data"] if m["id"] == self.model), None)
        require(advertised is not None, "requested model absent")
        require(advertised["context_length"] == self.args.ctx, "advertised context mismatch")
        self.before = self.snapshot("before")
        plan = self.before["stats"]["serving"]
        require(plan["family"] == "iquest_q1", "server is not IQuest-Q1")
        effective = plan["effective"]
        for key, expected in (("ctx", self.args.ctx), ("max_seqs", self.args.banks), ("mtp_mode", self.args.mtp_mode)):
            require(effective[key] == expected, f"effective {key}: {effective[key]} != {expected}")
        require(not any(issue.get("level") == "error" for issue in plan.get("issues", [])), "serving plan errors")
        self.report.update(model=self.model, serving=plan)

    def body(self, surface, prompt, mode="buffered"):
        body = {"model": self.model, "temperature": 0, "seed": 17, "stream": mode == "sse"}
        if surface == "responses":
            body.update(input=[{"role": "user", "content": prompt}], max_output_tokens=32,
                        reasoning={"effort": "none"})
        elif surface == "anthropic":
            body.update(messages=[{"role": "user", "content": prompt}], max_tokens=32,
                        thinking={"type": "disabled"})
        elif surface == "completion":
            # Completions accepts raw text. This is the pinned official
            # template's single-user, reasoning-off form, checked in
            # crates/ds4-core/tests/iquest_tokenizer.rs.
            raw = f"<|iquest_user|>{prompt}<|iquest_end|><|iquest_assistant|><think></think>"
            body.update(prompt=raw, max_tokens=32, reasoning_effort="none")
        else:
            body.update(messages=[{"role": "user", "content": prompt}], max_tokens=32,
                        reasoning_effort="none")
        return body

    def reasoning_body(self, surface, mode):
        body = self.body(surface, ARITHMETIC, mode)
        if surface == "responses":
            body.update(max_output_tokens=REASONING_BUDGET, reasoning={"effort": "high", "summary": "auto"})
        else:
            body.update(max_tokens=REASONING_BUDGET, reasoning_effort="high")
            if surface == "anthropic":
                # This server honors thinking.type; budget_tokens is ignored.
                # Bound total output with max_tokens instead.
                body["thinking"] = {"type": "enabled"}
            elif mode == "sse":
                body["stream_options"] = {"include_usage": True}
        return body

    def workload_body(self, prompt, mode, plain_budget):
        # Only sampling/concurrency opt into this control; other fixtures stay fixed.
        body = self.body("chat", prompt, mode)
        reasoning = Reasoning.REQUIRED if self.args.reasoning_effort == "high" else Reasoning.DISABLED
        body.update(reasoning_effort=self.args.reasoning_effort,
                    max_tokens=REASONING_BUDGET if reasoning == Reasoning.REQUIRED else plain_budget)
        return body, reasoning

    def generate(self, name, surface, body, reasoning=Reasoning.DISABLED):
        timing, raw = self.exchange(name, PATHS[surface], body)
        require(timing["status"] == HTTP_OK, f"{name}: HTTP {timing['status']}; inspect raw response")
        require(timing["streaming"] == body.get("stream", False), f"{name}: stream mode mismatch")
        if not timing["streaming"]:
            response = decode_json(raw)
            write_json(self.out / f"{name}.response.json", response)
            return response, timing
        records = sse_records(raw)
        write_json(self.out / f"{name}.events.json", records)
        values = [event["data"] for event in records if isinstance(event["data"], dict)]
        require(values and not any(v.get("error") or v.get("type") in ("error", "response.failed", "response.incomplete") for v in values), "SSE error or incomplete event")
        thought = "".join(reasoning_delta(value) for value in values)
        if reasoning == Reasoning.DISABLED:
            require(not thought, "unexpected SSE reasoning")
        if surface in ("chat", "completion"):
            require(records[-1]["data"] == "[DONE]", "missing final [DONE]")
            chunks = [v["choices"][0] for v in values if v.get("choices")]
            stops = [v["finish_reason"] for v in chunks if v.get("finish_reason")]
            require(stops == ["stop"], f"SSE finish reasons: {stops}")
            require(not any(v.get("delta", {}).get("tool_calls") for v in chunks), "unexpected SSE tool call")
            text = "".join(semantic_delta(v) for v in values)
            usage = next((v["usage"] for v in reversed(values) if v.get("usage")), None)
            response = {"choices": [{"finish_reason": "stop", "message": {"content": text, "reasoning_content": thought}, "text": text}], "usage": usage}
        elif surface == "responses":
            terminal = [v for v in values if v.get("type") == "response.completed"]
            require(len(terminal) == 1 and values[-1] == terminal[0], "missing/followed Responses terminal event")
            response = terminal[0]["response"]
            require("".join(semantic_delta(v) for v in values) == self.text(surface, response), "Responses delta/final mismatch")
            require(thought == self.reasoning_text(surface, response), "Responses reasoning delta/final mismatch")
            if reasoning == Reasoning.DISABLED:
                require(not any(v.get("type", "").startswith("response.reasoning") for v in values), "unexpected reasoning SSE")
        else:
            require(values[-1].get("type") == "message_stop", "missing final message_stop")
            stops = [v["delta"].get("stop_reason") for v in values if v.get("type") == "message_delta"]
            require(stops == ["end_turn"], f"Anthropic SSE stop reasons: {stops}")
            require(not any(v.get("content_block", {}).get("type") == "tool_use" for v in values), "unexpected tool SSE")
            if reasoning == Reasoning.DISABLED:
                require(not any(v.get("content_block", {}).get("type") == "thinking" or v.get("delta", {}).get("type") == "thinking_delta" for v in values), "unexpected reasoning SSE")
            initial = next(v["message"] for v in values if v.get("type") == "message_start")
            usage = dict(initial.get("usage", {}))
            for value in values:
                usage.update(value.get("usage", {}))
            response = {"content": [{"type": "text", "text": "".join(semantic_delta(v) for v in values)}], "stop_reason": "end_turn", "usage": usage}
            if thought:
                response["content"].insert(0, {"type": "thinking", "thinking": thought})
        write_json(self.out / f"{name}.assembled.json", response)
        return response, timing

    def text(self, surface, response):
        if surface == "chat":
            return response["choices"][0]["message"].get("content") or ""
        if surface == "completion":
            return response["choices"][0]["text"]
        if surface == "responses":
            return "".join(c["text"] for item in response["output"] if item["type"] == "message"
                           for c in item["content"] if c["type"] == "output_text")
        return "".join(c["text"] for c in response["content"] if c["type"] == "text")

    def reasoning_text(self, surface, response):
        if surface == "chat":
            return response["choices"][0]["message"].get("reasoning_content") or ""
        if surface == "responses":
            return "".join(part["text"] for item in response["output"] if item["type"] == "reasoning"
                           for part in item["summary"] if part["type"] == "summary_text")
        if surface == "anthropic":
            return "".join(item["thinking"] for item in response["content"] if item["type"] == "thinking")
        return ""

    def answer(self, surface, response, expected, reasoning=Reasoning.DISABLED):
        finite(response)
        if surface in ("chat", "completion"):
            choice = response["choices"][0]
            require(choice["finish_reason"] == "stop", f"finish_reason={choice['finish_reason']}")
            if surface == "chat":
                require(not choice["message"].get("tool_calls"), "unexpected tools")
        elif surface == "responses":
            require(response["status"] == "completed", "Responses incomplete")
            allowed = ("message", "reasoning") if reasoning == Reasoning.REQUIRED else ("message",)
            require(all(item["type"] in allowed for item in response["output"]), "unexpected reasoning/tools")
            require(all(item.get("status") == "completed" for item in response["output"]
                        if item["type"] == "reasoning"), "incomplete reasoning item")
        else:
            require(response["stop_reason"] == "end_turn", "Anthropic did not end turn")
            allowed = ("text", "thinking") if reasoning == Reasoning.REQUIRED else ("text",)
            require(all(item["type"] in allowed for item in response["content"]), "unexpected reasoning/tools")
        thought = self.reasoning_text(surface, response)
        clean(thought)
        if reasoning == Reasoning.REQUIRED:
            require(thought.strip(), "missing generated reasoning text")
        else:
            require(not thought, "unexpected reasoning")
        text = self.text(surface, response)
        clean(text)
        require(text.strip() == expected, f"answer {text!r} != fixed contract {expected!r}")
        usage = response.get("usage")
        require(isinstance(usage, dict), "missing usage")
        if surface == "anthropic":
            # Anthropic excludes cache reads and writes from input_tokens.
            inputs = [usage.get("input_tokens"), usage.get("cache_creation_input_tokens", 0),
                      usage.get("cache_read_input_tokens", 0)]
            require(all(type(value) is int and value >= 0 for value in inputs), "invalid input usage")
            require(sum(inputs) > 0 and type(usage.get("output_tokens")) is int
                    and usage["output_tokens"] > 0, "missing/nonpositive usage")
        else:
            names = ("input_tokens", "output_tokens") if surface == "responses" else ("prompt_tokens", "completion_tokens")
            require(all(type(usage.get(key)) is int and usage[key] > 0 for key in names), "missing/nonpositive usage")
        details = usage.get("output_tokens_details", usage.get("completion_tokens_details", {}))
        if reasoning == Reasoning.DISABLED:
            require(details.get("reasoning_tokens", 0) == 0, "reasoning tokens with reasoning off")
        elif "reasoning_tokens" in details:
            require(type(details["reasoning_tokens"]) is int and details["reasoning_tokens"] > 0,
                    "nonempty reasoning has no reported reasoning tokens")
        return {"text": text, "reasoning": thought, "usage": usage}

    def case(self, name, action):
        entry = {"name": name, "passed": False}
        self.report["cases"].append(entry)
        try:
            entry.update(action())
            entry["passed"] = True
        except Exception as error:
            entry.update(error=str(error), traceback=traceback.format_exc())
            try:
                self.snapshot(f"{name}.failure")
            except Exception as snapshot_error:
                entry["failure_snapshot_error"] = str(snapshot_error)
            if not self.args.keep_going:
                raise
        finally:
            write_json(self.out / f"{name}.result.json", entry)
            write_json(self.out / "report.json", self.report)
            print(json.dumps(entry, ensure_ascii=False), flush=True)

    def basic(self):
        for surface in PATHS:
            for mode in ("buffered", "sse"):
                name = f"basic.{surface}.{mode}"
                def action(surface=surface, mode=mode, name=name):
                    before = self.snapshot(f"{name}.before")
                    body = self.body(surface, ARITHMETIC, mode)
                    if surface in ("chat", "completion") and mode == "sse":
                        body["stream_options"] = {"include_usage": True}
                    response, timing = self.generate(name, surface, body)
                    result = self.answer(surface, response, "42")
                    after = self.snapshot(f"{name}.after")
                    self.no_faults(before, after)
                    return dict(result, timing=timing, last_request=after["stats"].get("last_request"))
                self.case(name, action)

    def tools(self):
        for surface in ("chat", "responses"):
            def action(surface=surface):
                name = f"tools.{surface}"
                before = self.snapshot(f"{name}.before")
                body = self.body(surface, TOOL_PROMPT)
                tool = {"name": "add", "description": "Return the sum of integer a and integer b.", "parameters": SCHEMA}
                body["tools"] = [{"type": "function", "function": tool}] if surface == "chat" else [dict(tool, type="function")]
                body["tool_choice"] = "required"
                body["max_tokens" if surface == "chat" else "max_output_tokens"] = 192
                response, timing = self.generate(f"{name}.call", surface, body)
                clean(self.text(surface, response))
                if surface == "chat":
                    choice = response["choices"][0]
                    require(choice["finish_reason"] == "tool_calls", "expected actual generated tool call")
                    message = choice["message"]
                    require(not message.get("reasoning_content"), "tool generation emitted reasoning")
                    calls = message.get("tool_calls", [])
                    require(len(calls) == 1, "expected exactly one tool call")
                    call = calls[0]
                    require(call.get("type") == "function", "tool type mismatch")
                    call_id, function = call["id"], call["function"]
                else:
                    require(response["status"] == "completed", "tool response incomplete")
                    require(not any(item["type"] == "reasoning" for item in response["output"]), "tool generation emitted reasoning")
                    calls = [item for item in response["output"] if item["type"] == "function_call"]
                    require(len(calls) == 1, "expected exactly one tool call")
                    function = calls[0]
                    call_id = function["call_id"]
                    require(function.get("id") and function.get("status") == "completed", "missing completed function item ID")
                require(isinstance(call_id, str) and call_id, "empty generated call ID")
                require(function["name"] == "add", "wrong generated function name")
                arguments = decode_json(function["arguments"])
                require(arguments == {"a": 19, "b": 23} and all(type(v) is int for v in arguments.values()), "generated arguments violate fixed integer schema")
                result = arguments["a"] + arguments["b"]
                require(result == 42, "tool implementation result mismatch")
                follow = copy.deepcopy(body)
                follow["tool_choice"] = "none"
                follow["max_tokens" if surface == "chat" else "max_output_tokens"] = 32
                if surface == "chat":
                    follow["messages"].extend([copy.deepcopy(message), {"role": "tool", "tool_call_id": call_id, "content": str(result)}])
                else:
                    follow["input"].extend(copy.deepcopy(response["output"]))
                    follow["input"].append({"type": "function_call_output", "call_id": call_id, "output": str(result)})
                completed, follow_timing = self.generate(f"{name}.follow", surface, follow)
                final = self.answer(surface, completed, "42")
                after = self.snapshot(f"{name}.after")
                self.no_faults(before, after)
                return dict(final, generated_call_id=call_id, generated_arguments=arguments,
                            tool_result=result, call_timing=timing, follow_timing=follow_timing,
                            history="returned structured assistant/output items replayed unchanged")
            self.case(f"tools.{surface}", action)

    def reasoning(self):
        for surface in ("chat", "responses", "anthropic"):
            for mode in ("buffered", "sse"):
                name = f"reasoning.{surface}.{mode}"
                def action(surface=surface, mode=mode, name=name):
                    before = self.snapshot(f"{name}.before")
                    body = self.reasoning_body(surface, mode)
                    response, timing = self.generate(name, surface, body, Reasoning.REQUIRED)
                    result = self.answer(surface, response, "42", Reasoning.REQUIRED)
                    after = self.snapshot(f"{name}.after")
                    self.no_faults(before, after)
                    return dict(result, timing=timing, last_request=after["stats"].get("last_request"))
                self.case(name, action)

    def sampling(self):
        def action():
            before = self.snapshot("sampling.before")
            body, reasoning = self.workload_body(ARITHMETIC, "buffered", 32)
            body.update(temperature=0.7, top_p=0.9, seed=29)
            response, timing = self.generate("sampling", "chat", body, reasoning)
            result = self.answer("chat", response, "42", reasoning)
            after = self.snapshot("sampling.after")
            self.no_faults(before, after)
            trace = after["stats"]["last_request"]
            require(trace["speculation_active"] is False, "positive-temperature sampling used greedy MTP")
            for key in ("ds4_spec_drafts_total", "ds4_spec_hits_total"):
                require(before["metrics"][key] == after["metrics"][key], f"sampling changed {key}")
            return dict(result, timing=timing, last_request=trace, mtp_fallback="ordinary sampled decode; zero speculative counter delta")
        self.case("sampling", action)

    def concurrency(self):
        def action():
            before = self.snapshot("concurrency.before")
            barrier = threading.Barrier(3)
            results, errors, samples = {}, {}, []
            def worker(index):
                name = f"concurrency.{index}"
                prompt = (f"Request label {index}. Write the integers from 1 through 24, increasing by one, "
                          "separated by single spaces. Output only that sequence; do not output the label.")
                body, reasoning = self.workload_body(prompt, "sse", 128)
                body["stream_options"] = {"include_usage": True}
                try:
                    barrier.wait(timeout=10)
                    response, timing = self.generate(name, "chat", body, reasoning)
                    results[index] = dict(self.answer("chat", response, SEQUENCE, reasoning), timing=timing)
                except Exception as error:
                    errors[index] = {"error": str(error), "traceback": traceback.format_exc()}
            threads = [threading.Thread(target=worker, args=(index,)) for index in range(2)]
            for thread in threads:
                thread.start()
            barrier.wait(timeout=10)
            while any(thread.is_alive() for thread in threads):
                snap = self.snapshot(f"concurrency.poll.{len(samples):04d}")
                samples.append({"monotonic": time.monotonic(), "inflight": snap["metrics"]["ds4_requests_inflight"],
                                "banks_live": snap["metrics"]["ds4_banks_live"], "queue_depth": snap["stats"]["queue_depth"]})
                time.sleep(0.25)
            for thread in threads:
                thread.join()
            write_json(self.out / "concurrency.workers.json", {"results": results, "errors": errors, "samples": samples})
            require(not errors and len(results) == 2, f"concurrent request failed: {errors}")
            after = self.snapshot("concurrency.after")
            self.no_faults(before, after)
            timings = [item["timing"] for item in results.values()]
            require(all("first_text_monotonic" in item for item in timings), "missing content-delta timing")
            first_finish = min(item["end_monotonic"] for item in timings)
            require(max(item["start_monotonic"] for item in timings) < first_finish, "requests did not overlap")
            require(max(item["first_text_monotonic"] for item in timings) < first_finish, "both requests did not emit text before either completed")
            require(any(sample["inflight"] >= 2 for sample in samples), "did not observe two inflight requests")
            routes = "openai_chat_continuous"
            require(after["stats"]["routes"][routes] - before["stats"]["routes"][routes] == 2, "requests did not both complete on continuous lane")
            return {"workers": results, "samples": samples, "max_inflight": max(s["inflight"] for s in samples),
                    "max_reported_banks_live": max(s["banks_live"] for s in samples),
                    "scope": "two inflight requests with interleaved visible generation on continuous lane; native active-bank gauge is recorded separately"}
        self.case("concurrency", action)

    def run(self):
        try:
            self.startup()
            modes = ("basic", "tools", "sampling", "concurrency") if self.args.mode == "all" else (self.args.mode,)
            for mode in modes:
                getattr(self, mode)()
            # Collect the global fault result even when individual cases failed.
            self.report["global_fault_check"] = {"passed": False}
            try:
                after = self.snapshot("after")
                self.no_faults(self.before, after)
                self.report["global_fault_check"]["passed"] = True
            except Exception as error:
                self.report["global_fault_check"]["error"] = str(error)
                raise
            failed = [entry["name"] for entry in self.report["cases"] if not entry["passed"]]
            require(not failed, f"{len(failed)} case(s) failed: {', '.join(failed)}")
            self.report["passed"] = True
        except Exception as error:
            self.report.update(passed=False, error=str(error), traceback=traceback.format_exc())
            raise
        finally:
            write_json(self.out / "report.json", self.report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("basic", "tools", "sampling", "concurrency", "reasoning", "all"))
    parser.add_argument("--url", default="http://127.0.0.1:8002")
    parser.add_argument("--model", help="Exact advertised model ID; default first /v1/models entry")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ctx", type=int, default=8192)
    parser.add_argument("--banks", type=int, default=2)
    parser.add_argument("--mtp-mode", choices=("on", "off"), default="on")
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument("--reasoning-effort", choices=("none", "high"), default="none",
                        help="sampling/concurrency only: high requires generated reasoning and uses a 256-token budget")
    parser.add_argument("--keep-going", action="store_true",
                        help="record every case failure and check final faults; overall failure remains nonzero")
    args = parser.parse_args()
    url = urllib.parse.urlparse(args.url)
    require(url.scheme == "http" and url.hostname in ("127.0.0.1", "localhost", "::1") and not url.username and not url.password and url.path in ("", "/"), "only plain localhost HTTP is accepted")
    args.url = args.url.rstrip("/")
    require(args.ctx > 0 and args.banks == 2 and args.timeout > 0, "positive context/timeout and exactly two banks required")
    Gate(args).run()


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"GATE FAILED: {error}", file=sys.stderr)
        sys.exit(1)
