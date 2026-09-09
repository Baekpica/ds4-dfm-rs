#!/usr/bin/env python3
"""Bounded template/API smoke against an already running local server.

Save exact requests and responses; this script never starts or loads a model.
"""
import argparse
import base64
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier
import json
from pathlib import Path
import time
import urllib.request

PATHS = {"openai": "/v1/chat/completions", "anthropic": "/v1/messages", "responses": "/v1/responses"}
SCHEMA = {"name": "get_weather", "description": "Get weather for a city.", "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}
QUESTION = "Call get_weather for Seoul. After receiving the result, state the temperature in one short sentence."


def payload(api, text):
    if api == "responses":
        return {"input": [{"role": "user", "content": text}], "max_output_tokens": 128, "reasoning": {"effort": "none"}}
    body = {"messages": [{"role": "user", "content": text}], "max_tokens": 128}
    body.update({"thinking": {"type": "disabled"}} if api == "anthropic" else {"reasoning_effort": "none"})
    return body


def tool_payload(api):
    body = payload(api, QUESTION)
    if api == "responses":
        body["tools"] = [{"type": "function", **SCHEMA}]
    elif api == "anthropic":
        body["tools"] = [{"name": SCHEMA["name"], "description": SCHEMA["description"], "input_schema": SCHEMA["parameters"]}]
    else:
        body["tools"] = [{"type": "function", "function": SCHEMA}]
    return body


def unpack(api, response):
    if isinstance(response, list):
        if api == "responses":
            completed = [e["response"] for e in response if e["type"] == "response.completed"]
            assert len(completed) == 1, response
            return unpack(api, completed[0])
        if api == "anthropic":
            blocks = {}
            finish = None
            for event in response:
                if event["type"] == "content_block_start":
                    blocks[event["index"]] = dict(event["content_block"])
                elif event["type"] == "content_block_delta":
                    block = blocks[event["index"]]
                    delta = event["delta"]
                    for key in ("text", "thinking", "partial_json"):
                        if key in delta:
                            block[key] = block.get(key, "") + delta[key]
                elif event["type"] == "message_delta":
                    finish = event["delta"]["stop_reason"]
            for block in blocks.values():
                if block["type"] == "tool_use":
                    block["input"] = json.loads(block.pop("partial_json", "{}"))
            return unpack(api, {"content": list(blocks.values()), "stop_reason": finish})
        message = {"role": "assistant", "content": "", "reasoning_content": ""}
        calls = {}
        finish = None
        for event in response:
            for choice in event.get("choices", []):
                delta = choice.get("delta", {})
                for key in ("content", "reasoning_content"):
                    message[key] += delta.get(key) or ""
                for call in delta.get("tool_calls", []):
                    target = calls.setdefault(call["index"], {"id": "", "type": "function", "function": {"name": "", "arguments": ""}})
                    if call.get("id"):
                        target["id"] = call["id"]
                    for key in ("name", "arguments"):
                        target["function"][key] += call.get("function", {}).get(key, "")
                finish = choice.get("finish_reason") or finish
        if calls:
            message["tool_calls"] = list(calls.values())
        return unpack(api, {"choices": [{"message": message, "finish_reason": finish}]})
    if api == "openai":
        message = response["choices"][0]["message"]
        calls = [{"id": c["id"], "name": c["function"]["name"], "args": json.loads(c["function"]["arguments"])} for c in message.get("tool_calls", [])]
        return message.get("content") or "", calls, response["choices"][0]["finish_reason"]
    if api == "anthropic":
        text = "".join(c.get("text", "") for c in response["content"])
        calls = [{"id": c["id"], "name": c["name"], "args": c["input"]} for c in response["content"] if c["type"] == "tool_use"]
        return text, calls, response["stop_reason"]
    text = "".join(c.get("text", "") for b in response["output"] for c in b.get("content", []))
    calls = [{"id": c["call_id"], "name": c["name"], "args": json.loads(c["arguments"])} for c in response["output"] if c["type"] == "function_call"]
    return text, calls, response["status"]


class Probe:
    def __init__(self, args):
        self.args = args
        self.records = []
        args.output.mkdir(parents=True, exist_ok=True)

    def request(self, api, name, body):
        body = {"model": self.args.model, "temperature": 0, **body}
        stem = self.args.output / (api + "-" + name)
        data = json.dumps(body, ensure_ascii=False, indent=2).encode()
        stem.with_suffix(".request.json").write_bytes(data)
        start = time.monotonic()
        req = urllib.request.Request(self.args.url + PATHS[api], data=data, headers={"Content-Type": "application/json"})
        try:
            response = urllib.request.urlopen(req, timeout=180)
        except urllib.error.HTTPError as error:
            stem.with_suffix(".response.txt").write_bytes(error.read())
            raise
        raw = response.read()
        stem.with_suffix(".response.txt").write_bytes(raw)
        assert response.status == 200
        assert "<|" not in raw.decode(), raw
        if body.get("stream"):
            decoded = [json.loads(line[6:]) for line in raw.decode().splitlines() if line.startswith("data: ") and line != "data: [DONE]"]
        else:
            decoded = json.loads(raw)
        self.records.append({"api": api, "case": name, "seconds": round(time.monotonic() - start, 3)})
        print(api, name, unpack(api, decoded), flush=True)
        return unpack(api, decoded)

    def text(self, api):
        body = payload(api, "What is 2 + 2? Reply with just the number.")
        text, calls, finish = self.request(api, "text", body)
        assert text.strip() == "4" and not calls and finish in ("stop", "end_turn", "completed")
        field = "input" if api == "responses" else "messages"
        # Deliberately omit hidden reasoning: normal clients may only retain text.
        body[field] += [{"role": "assistant", "content": text}, {"role": "user", "content": "Add one to that number. Reply with just the number."}]
        text, calls, finish = self.request(api, "followup", body)
        assert text.strip() == "5" and not calls and finish in ("stop", "end_turn", "completed")

    def tools(self, api, mode):
        body = tool_payload(api)
        body["stream"] = mode == "stream"
        _, calls, finish = self.request(api, "tool-" + mode, body)
        assert len(calls) == 1 and calls[0]["name"] == SCHEMA["name"], calls
        assert calls[0]["args"] == {"city": "Seoul"}, calls
        assert finish in ("tool_calls", "tool_use", "completed"), finish
        result = '{"city":"Seoul","temperature_c":21,"condition":"Sunny"}'
        reply = payload(api, "")
        reply["stream"] = mode == "stream"
        if api == "responses":
            reply["input"] = [{"type": "function_call_output", "call_id": calls[0]["id"], "output": result}]
        elif api == "anthropic":
            reply["messages"] = [{"role": "user", "content": [{"type": "tool_result", "tool_use_id": calls[0]["id"], "content": result}]}]
        else:
            assistant = {"role": "assistant", "content": "", "tool_calls": [{"id": calls[0]["id"], "type": "function", "function": {"name": calls[0]["name"], "arguments": json.dumps(calls[0]["args"])}}]}
            reply["messages"] = body["messages"] + [assistant, {"role": "tool", "tool_call_id": calls[0]["id"], "content": result}]
            reply["tools"] = body["tools"]
        # Messages/Responses recover history and schemas from a live frontier.
        # Chat Completions retains its existing full-history replay contract.
        text, calls, finish = self.request(api, "tool-result-" + mode, reply)
        assert "21" in text and not calls and finish in ("stop", "end_turn", "completed"), (text, calls, finish)

    def images(self, api):
        for color in ("red", "blue"):
            data = base64.b64encode((self.args.images / (color + ".png")).read_bytes()).decode()
            body = payload(api, "")
            question = "What color is this image? Reply with one color word."
            if api == "responses":
                body["input"][0]["content"] = [{"type": "input_image", "image_url": "data:image/png;base64," + data}, {"type": "input_text", "text": question}]
            elif api == "anthropic":
                body["messages"][0]["content"] = [{"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": data}}, {"type": "text", "text": question}]
            else:
                body["messages"][0]["content"] = [{"type": "image_url", "image_url": {"url": "data:image/png;base64," + data}}, {"type": "text", "text": question}]
            text, calls, finish = self.request(api, "image-" + color, body)
            assert color in text.lower() and not calls and finish in ("stop", "end_turn", "completed"), (text, finish)

    def parallel_tools(self):
        barrier = Barrier(2)
        def call(city):
            body = tool_payload("responses")
            # Retained bank tool turns use the supported streaming lane.
            body["stream"] = True
            body["input"][0]["content"] = f"Call get_weather for {city}. After the result, state its city and temperature."
            barrier.wait(timeout=10)
            _, calls, finish = self.request("responses", "concurrent-call-" + city, body)
            assert finish == "completed" and len(calls) == 1 and calls[0]["args"] == {"city": city}, calls
            return city, calls[0]["id"]
        with ThreadPoolExecutor(max_workers=2) as pool:
            pending = list(pool.map(call, ["Paris", "Seoul"]))
        assert pending[0][1] != pending[1][1]
        def reply(item):
            city, call_id = item
            temperature = 17 if city == "Paris" else 21
            body = payload("responses", "")
            body["stream"] = True
            body["input"] = [{"type": "function_call_output", "call_id": call_id, "output": json.dumps({"city": city, "temperature_c": temperature})}]
            barrier.wait(timeout=10)
            text, calls, finish = self.request("responses", "concurrent-result-" + city, body)
            assert finish == "completed" and not calls and city.lower() in text.lower() and str(temperature) in text, text
        with ThreadPoolExecutor(max_workers=2) as pool:
            list(pool.map(reply, pending))

    def run(self):
        for api in PATHS:
            self.text(api)
            for mode in ("buffered", "stream"):
                self.tools(api, mode)
            if self.args.images:
                self.images(api)
        if self.args.concurrent:
            self.parallel_tools()
        (self.args.output / "summary.json").write_text(json.dumps(self.records, indent=2) + "\n")
        stats = urllib.request.urlopen(self.args.url + "/v1/stats", timeout=10).read()
        (self.args.output / "stats.txt").write_bytes(stats)
        print("PASS", len(self.records), "requests", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:18085")
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--images", type=Path, help="Directory containing fixed red.png and blue.png fixtures")
    parser.add_argument("--concurrent", action="store_true", help="Require two banks and check independent live tool frontiers")
    Probe(parser.parse_args()).run()


if __name__ == "__main__":
    main()
