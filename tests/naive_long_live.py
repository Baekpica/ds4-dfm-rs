#!/usr/bin/env python3
"""Gate a running Naive server with a hashed near-capacity retrieval fixture.

Run seed, follow, and optionally restored after restarting the same disk store.
Readiness is checked separately from actual generation and prefix reuse.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time
import urllib.error
import urllib.request

FOLLOW = "Repeat the access phrase for depot Aster again, with no explanation."
REQUEST_TIMEOUT = 43200
STATE_TIMEOUT = 20


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("seed", "follow", "restored"))
    parser.add_argument("--url", default="http://127.0.0.1:8002")
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--banks", type=int, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    assert not (args.out / f"{args.phase}.response.json").exists(), "refusing to overwrite evidence"
    receipt = json.loads((args.fixture / "fixture.json").read_text())

    def save(name, data):
        (args.out / name).write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")

    def get(path):
        with urllib.request.urlopen(args.url + path, timeout=STATE_TIMEOUT) as response:
            return json.load(response)

    models = get("/v1/models")
    before = get("/v1/stats")
    save(args.phase + ".models.json", models)
    save(args.phase + ".before.json", before)
    effective = before["serving"]["effective"]
    assert models["data"][0]["context_length"] == receipt["context"]
    assert effective["ctx"] == receipt["context"]
    assert effective["max_seqs"] == args.banks
    assert effective["mtp_mode"] == "off"

    raw = (args.fixture / "request.json").read_bytes()
    assert hashlib.sha256(raw).hexdigest() == receipt["request_sha256"]
    body = json.loads(raw)
    if args.phase != "seed":
        previous = "follow" if args.phase == "restored" else "seed"
        prior = json.loads((args.out / (previous + ".response.json")).read_text())
        assert prior["passed"], "the previous request must pass before continuation"
        if args.phase == "restored":
            body = json.loads((args.out / "follow.request.json").read_text())
        body["messages"] += [{"role": "assistant", "content": prior["text"]},
                             {"role": "user", "content": FOLLOW}]
    save(args.phase + ".request.json", body)

    started = time.monotonic()
    first = None
    text = ""
    usage = None
    finish = None
    events = 0
    request = urllib.request.Request(args.url + "/v1/chat/completions", json.dumps(body).encode(),
                                     {"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT) as response, \
                (args.out / (args.phase + ".sse.jsonl")).open("w") as log:
            for raw in response:
                line = raw.decode().strip()
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    break
                data = json.loads(payload)
                log.write(json.dumps(data, ensure_ascii=False) + "\n")
                log.flush()
                events += 1
                for choice in data.get("choices", []):
                    delta = choice.get("delta", {}).get("content")
                    if delta:
                        if first is None:
                            first = time.monotonic() - started
                        text += delta
                    if choice.get("finish_reason") is not None:
                        finish = choice["finish_reason"]
                if data.get("usage") is not None:
                    usage = data["usage"]
    except urllib.error.HTTPError as error:
        (args.out / (args.phase + ".error.txt")).write_bytes(error.read())
        raise

    elapsed = time.monotonic() - started
    after = get("/v1/stats")
    save(args.phase + ".after.json", after)
    trace = after.get("last_request", {})
    errors = []
    if text.strip() != receipt["expected_answer"]:
        errors.append("distant fact mismatch")
    if finish != "stop":
        errors.append("not a normal stop")
    if not usage:
        errors.append("missing usage")
    if args.phase == "seed" and usage:
        if usage["prompt_tokens"] != receipt["input_tokens"]:
            errors.append("official tokenizer count mismatch")
        if usage.get("prompt_tokens_details", {}).get("cached_tokens", 0) != 0:
            errors.append("seed was not cold")
    if after["governor"]["faults"] != 0 or before["governor"]["faults"] != 0:
        errors.append("governor fault")
    if trace.get("effective_lane") != "continuous" or trace.get("fallback_reason"):
        errors.append("lane fallback")
    if trace.get("speculation_active"):
        errors.append("unexpected speculation")
    if args.phase != "seed" and usage:
        cached = usage.get("prompt_tokens_details", {}).get("cached_tokens", 0)
        if not 0 < cached < usage["prompt_tokens"]:
            errors.append("continuation did not reuse a proper KV prefix")
    result = {"context": receipt["context"], "banks": args.banks, "phase": args.phase,
              "text": text, "usage": usage, "finish_reason": finish, "events": events,
              "first_content_seconds": first, "total_seconds": elapsed, "trace": trace,
              "passed": not errors, "errors": errors, "fixture": receipt}
    save(args.phase + ".response.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return bool(errors)


if __name__ == "__main__":
    raise SystemExit(main())
