#!/usr/bin/env python3
"""Measure a fixed image suite against a running Qwen server (stdlib only).

Use a fresh worker and empty KV directory for each suite. This client neither
restarts the server nor clears its caches. It rejects cached timed prompts.
"""
import argparse
import base64
import hashlib
import json
from pathlib import Path
import time
import urllib.request


CASES = {
    "small": ["small.png"],
    "screen": ["screen.png"],
    "document": ["document.png"],
    "photo": ["photo.jpg"],
    "large": ["large.png"],
    "multi": ["small.png", "screen.png", "photo.jpg", "document.png"],
}
PROMPT = "Briefly describe the visible content and any task counts. /no_think"


def request_body(case, fixtures, model, max_tokens):
    parts = [{"type": "text", "text": PROMPT}]
    for name in CASES[case]:
        path = fixtures / name
        mime = "jpeg" if path.suffix == ".jpg" else "png"
        uri = f"data:image/{mime};base64," + base64.b64encode(path.read_bytes()).decode()
        parts.append({"type": "image_url", "image_url": {"url": uri}})
    return {
        "model": model,
        "messages": [{"role": "user", "content": parts}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "thinking": {"type": "disabled"},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8002")
    parser.add_argument("--model", default="Qwen3.8-Flash-Next-Mixed-Quant")
    parser.add_argument("--fixtures", type=Path,
                        default=Path(__file__).parent / "fixtures/qwen-images")
    parser.add_argument("--cases", default=",".join(CASES))
    parser.add_argument("--max-tokens", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cases = args.cases.split(",")
    if args.max_tokens < 1 or any(case not in CASES for case in cases):
        parser.error("positive --max-tokens and known --cases required")
    # Exclusive creation avoids silently mixing runs with different cache state.
    with args.output.open("x") as output:
        for case in cases:
            body = request_body(case, args.fixtures, args.model, args.max_tokens)
            data = json.dumps(body).encode()
            req = urllib.request.Request(
                args.base_url.rstrip("/") + "/v1/chat/completions", data=data,
                headers={"Content-Type": "application/json"})
            start = time.perf_counter()
            with urllib.request.urlopen(req, timeout=1800) as response:
                result = json.load(response)
            elapsed = (time.perf_counter() - start) * 1000
            cached = result["usage"]["prompt_tokens_details"]["cached_tokens"]
            if cached != 0:
                raise RuntimeError("timed prompt reused KV; restart with an empty KV directory")
            row = {
                "case": case, "wall_ms": elapsed, "response": result,
                "fixture_sha256": {
                    name: hashlib.sha256((args.fixtures / name).read_bytes()).hexdigest()
                    for name in CASES[case]
                },
            }
            output.write(json.dumps(row) + "\n")
            output.flush()
            print(json.dumps({"case": case, "wall_ms": elapsed,
                              "cached_tokens": cached, "timings": result.get("timings")}),
                  flush=True)


if __name__ == "__main__":
    main()
