#!/usr/bin/env python3
"""K2 serial raw-completion disk lifecycle; never starts or stops a server.

Run seed on an empty-cache server, restart with that cache, run restored,
then run cold on a third process with --prefix-reuse off and no disk cache.
The newline after the seed's IFM delimiter keeps suffixes token-prefix stable.
Raw completions use the serial disk-cache path.
Strict append and sibling requests reuse a stored prefix; identical requests
and early edits currently replay cold under K2's zero-rewind policy.
This exercises /v1/completions; it does not qualify Chat thinking/history.
"""

import argparse
import hashlib
import json
from pathlib import Path
import time
import urllib.error

from step37_history_live import fingerprint, process_identity, request, write_json


def fixture(model, lines):
    padding = "".join(f"Inventory item {i}: blue square.\n" for i in range(lines))
    base = ("<|ifm|begin_of_text|><|ifm|im_start|>user\n" + padding
            + "What is 2 + 2?<|ifm|im_end|>\n")
    suffix = "<|ifm|im_start|>assistant\n<ifm|think_faster>\n</ifm|think_faster>"

    def body(prompt, budget=8):
        return {"model": model, "prompt": prompt, "max_tokens": budget,
                "temperature": 0, "seed": 1, "reasoning_effort": "none"}

    return {"append": body(base + suffix + "The answer is"),
            "fork": body(base + suffix + "4. Add one to get"),
            "exact": body(base, 1),
            "edit": body(base.replace("blue square", "red circle", 1) + suffix + "The answer is")}


def run_case(args, name, body):
    key = f"{args.phase}.{name}"
    write_json(args.output / f"{key}.request.json", body)
    started = time.monotonic()
    try:
        response = request(args.url, "/v1/completions", body)
    except urllib.error.HTTPError as error:
        (args.output / f"{key}.error.txt").write_bytes(error.read())
        raise
    seconds = time.monotonic() - started
    stats = request(args.url, "/v1/stats")
    write_json(args.output / f"{key}.response.json", response)
    write_json(args.output / f"{key}.stats.json", stats)
    trace = stats["last_request"]
    assert trace["effective_lane"] == args.lane, trace
    assert not trace["speculation_active"], trace
    assert not trace.get("fallback_reason"), trace
    plan = stats["serving"]
    assert plan["effective"]["ctx"] == args.context, plan
    assert plan["effective"]["max_seqs"] == 1, plan
    assert plan["effective"]["mtp_mode"] == "off", plan
    if args.phase == "cold":
        assert not plan["effective"]["disk"], plan
        assert plan["effective"]["prefix_reuse"] == "off", plan
    else:
        assert plan["effective"]["disk"], plan
        assert plan["effective"]["prefix_reuse"] == "exact", plan
    cached = response["usage"]["prompt_tokens_details"]["cached_tokens"]
    summary = {"case": name, "seconds": seconds, "cached_tokens": cached,
               "usage": response["usage"], "choice": response["choices"][0], "trace": trace}
    write_json(args.output / f"{key}.summary.json", summary)
    if args.phase == "restored" and name in ("append", "fork"):
        assert cached > 0 and trace["reuse_kind"] in ("exact", "fork"), summary
        assert cached < response["usage"]["prompt_tokens"], summary
    else:
        assert cached == 0 and trace["reuse_kind"] == "cold", summary
    if args.phase == "cold":
        warm = json.loads((args.output / f"restored.{name}.response.json").read_text())
        for field in ("text", "finish_reason"):
            assert response["choices"][0][field] == warm["choices"][0][field], summary
        assert response["usage"]["completion_tokens"] == warm["usage"]["completion_tokens"], summary
    print(json.dumps(summary, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=["seed", "restored", "cold"])
    parser.add_argument("--url", default="http://127.0.0.1:18038")
    parser.add_argument("--pid", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--artifact-manifest", type=Path, required=True)
    parser.add_argument("--model", default="k2-horizon")
    parser.add_argument("--context", type=int, default=32768)
    parser.add_argument("--lane", choices=["serial"], default="serial")
    parser.add_argument("--padding-lines", type=int, default=64)
    args = parser.parse_args()
    assert args.padding_lines > 0
    args.output.mkdir(parents=True, exist_ok=True)
    process = process_identity(args.pid)
    process["artifacts_sha256"] = hashlib.sha256(args.artifact_manifest.read_bytes()).hexdigest()
    write_json(args.output / f"{args.phase}.process.json", process)
    if args.phase == "seed":
        (args.output / "artifacts.json").write_bytes(args.artifact_manifest.read_bytes())
        cases = fixture(args.model, args.padding_lines)
        write_json(args.output / "fixture.json", cases)
        run_case(args, "exact", cases["exact"])
    else:
        seed = json.loads((args.output / "seed.process.json").read_text())
        assert fingerprint(seed) != fingerprint(process), "need a restarted server"
        assert seed["executable_sha256"] == process["executable_sha256"], "binary changed"
        assert seed["artifacts_sha256"] == process["artifacts_sha256"], "artifacts changed"
        if args.phase == "cold":
            warm = json.loads((args.output / "restored.process.json").read_text())
            assert fingerprint(warm) != fingerprint(process), "cold control needs a fresh process"
        cases = json.loads((args.output / "fixture.json").read_text())
        for name, body in cases.items():
            run_case(args, name, body)


if __name__ == "__main__":
    main()
