#!/usr/bin/env python3
"""Step history-frontier Chat restart gate against a managed local server.

Run seed, restart with the same disk cache, run restored, then start a fresh
server with an empty cache and run cold. Each phase records the actual process,
request, response, stats and time. This script never starts or stops a server.
For a short bank gate set DS4_SERVER_PERSIST_MIN_TOKENS=1 and
--kv-cache-min-tokens 1; the production persistence default remains 8192.
"""

import argparse
import hashlib
import json
from pathlib import Path
import time
import urllib.error
import urllib.request


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def process_identity(pid):
    proc = Path("/proc") / str(pid)
    stat = (proc / "stat").read_text().rsplit(")", 1)[1].split()
    command = (proc / "cmdline").read_bytes().replace(b"\0", b" ").decode()
    assert "ds4-server" in command, command
    return {
        "pid": pid,
        "start_ticks": stat[19],
        "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
        "command": command,
        "executable_sha256": hashlib.sha256((proc / "exe").read_bytes()).hexdigest(),
    }


def request(url, path, body=None):
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(url + path, data=data,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as response:
        return json.load(response)


def fingerprint(process):
    return process["boot_id"], process["pid"], process["start_ticks"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=["seed", "restored", "cold"])
    parser.add_argument("--url", default="http://127.0.0.1:18037")
    parser.add_argument("--pid", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="step-3.7-flash-mq83")
    parser.add_argument("--lane", choices=["serial", "continuous"], default="continuous")
    parser.add_argument("--padding-lines", type=int, default=32)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    process = process_identity(args.pid)
    write_json(args.output / f"{args.phase}.process.json", process)

    if args.phase == "seed":
        padding = "".join(f"Inventory item {i}: blue square.\n" for i in range(args.padding_lines))
        body = {
            "model": args.model, "temperature": 0, "seed": 1,
            "max_tokens": 32, "reasoning_effort": "none",
            "messages": [{"role": "user", "content": padding +
                          "What is 2 + 2? Reply with just the number."}],
        }
    else:
        seed = json.loads((args.output / "seed.process.json").read_text())
        assert fingerprint(process) != fingerprint(seed), "server was not restarted"
        body = json.loads((args.output / "follow.request.json").read_text())
    write_json(args.output / f"{args.phase}.request.json", body)

    start = time.monotonic()
    try:
        response = request(args.url, "/v1/chat/completions", body)
    except urllib.error.HTTPError as error:
        (args.output / f"{args.phase}.error.txt").write_bytes(error.read())
        raise
    seconds = time.monotonic() - start
    write_json(args.output / f"{args.phase}.response.json", response)
    stats = request(args.url, "/v1/stats")
    write_json(args.output / f"{args.phase}.stats.json", stats)
    message = response["choices"][0]["message"]
    text = message.get("content", "")
    assert not message.get("reasoning_content") and not message.get("tool_calls"), message
    assert response["choices"][0]["finish_reason"] == "stop", response
    cached = response["usage"]["prompt_tokens_details"]["cached_tokens"]
    trace = stats["last_request"]
    assert trace["effective_lane"] == args.lane, trace
    summary = {"phase": args.phase, "seconds": seconds, "cached_tokens": cached,
               "prompt_tokens": response["usage"]["prompt_tokens"], "text": text,
               "trace": trace}
    write_json(args.output / f"{args.phase}.summary.json", summary)

    if args.phase == "seed":
        assert text.strip() == "4" and cached == 0, summary
        body["messages"] += [{"role": "assistant", "content": text},
                             {"role": "user", "content": "Add one. Reply with just the number."}]
        write_json(args.output / "follow.request.json", body)
    elif args.phase == "restored":
        assert text.strip() == "5" and cached > 0, summary
        assert trace["reuse_kind"] in ("exact", "fork", "partial"), trace
    else:
        warm_process = json.loads((args.output / "restored.process.json").read_text())
        assert fingerprint(process) != fingerprint(warm_process), "cold control needs a fresh process"
        warm = json.loads((args.output / "restored.response.json").read_text())
        assert message == warm["choices"][0]["message"] and cached == 0, summary
        assert trace["reuse_kind"] == "cold", trace
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
