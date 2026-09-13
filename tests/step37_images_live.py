#!/usr/bin/env python3
"""Bounded Step image-answer/continuation gate; never starts a server.

Reuse the original repository images without resizing. Save exact HTTP
requests, responses, fixture hashes and timings; this is not a TPS benchmark.
"""
import argparse
import base64
import hashlib
import json
from pathlib import Path
import re

from chat_template_live import Probe, payload


def image_part(api, path):
    data = base64.b64encode(path.read_bytes()).decode()
    mime = "image/jpeg" if path.suffix == ".jpg" else "image/png"
    if api == "anthropic":
        return {"type": "image", "source": {"type": "base64", "media_type": mime, "data": data}}
    uri = f"data:{mime};base64,{data}"
    if api == "responses":
        return {"type": "input_image", "image_url": uri}
    return {"type": "image_url", "image_url": {"url": uri}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:18037")
    parser.add_argument("--model", default="step-3.7-flash-mq83")
    parser.add_argument("--fixtures", type=Path, default=Path(__file__).parent / "fixtures/qwen-images")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.reasoning_effort = "none"
    args.max_tokens = 128
    probe = Probe(args)
    hashes = {}

    def request(api, name, files, question, stream=False):
        body = payload(api, "")
        field = "input" if api == "responses" else "messages"
        parts = []
        for file in files:
            path = args.fixtures / file
            hashes[file] = hashlib.sha256(path.read_bytes()).hexdigest()
            parts.append(image_part(api, path))
        parts.append({"type": "input_text" if api == "responses" else "text", "text": question})
        body[field][0]["content"] = parts
        body["stream"] = stream
        text, calls, finish = probe.request(api, name, body)
        assert not calls and finish in ("stop", "end_turn", "completed"), (name, text, finish)
        return text, body

    question = 'Read the Queued, Running, and Failed counts. Reply only as JSON with keys "queued", "running", "failed".'
    for name, failed in (("small", 3), ("screen", 3), ("screen-changed", 9), ("large", 3)):
        text, _ = request("openai", name, [name + ".png"], question)
        match = re.search(r"\{[^{}]+\}", text)
        assert match, (name, text)
        counts = json.loads(match.group())
        assert {key.lower(): int(value) for key, value in counts.items()} == {
            "queued": 12, "running": 7, "failed": failed}, (name, text)

    text, invoice = request("openai", "document", ["document.png"],
                            "Read the invoice number, customer name and total due. Reply briefly.")
    assert "2048" in text and "mina park" in text.lower() and "385" in text, text
    invoice["messages"] += [{"role": "assistant", "content": text}, {
        "role": "user", "content": "What is the tax on that invoice? Reply with just the amount."}]
    text, calls, finish = probe.request("openai", "document-followup", invoice)
    assert "35" in text and not calls and finish == "stop", (text, finish)

    text, photo = request("responses", "photo", ["photo.jpg"], "What planet is shown? Reply with its name.")
    assert "earth" in text.lower(), text
    # The runtime's Responses contract requires replaying complete input.
    photo["input"] += [{"role": "assistant", "content": text}, {
        "role": "user", "content": "What planet did I show you? Reply with its name."}]
    text, calls, finish = probe.request("responses", "photo-followup", photo)
    assert "earth" in text.lower() and not calls and finish == "completed", (text, finish)

    text, _ = request("anthropic", "multi-stream", ["small.png", "screen.png", "photo.jpg", "document.png"],
                      "Describe the main subject of each of the four images in order. Use a short numbered list.", True)
    lower = text.lower()
    assert lower.count("dashboard") >= 2 and "earth" in lower and "invoice" in lower, text
    assert lower.index("earth") < lower.index("invoice"), text
    (args.output / "summary.json").write_text(json.dumps({"requests": probe.records, "sha256": hashes}, indent=2))
    print("PASS", len(probe.records), "balanced image/continuation requests", flush=True)


if __name__ == "__main__":
    main()
