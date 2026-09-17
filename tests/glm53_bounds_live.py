#!/usr/bin/env python3
"""GLM configuration/HTTP boundary gates; no server is started or stopped.

config runs metadata-only --check-config subprocesses. Run on an idle host so
an unrelated resident model cannot cause the accepted configuration to fail
memory admission. http checks a separately managed ctx=2048 serial worker.
"""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import urllib.error

from step37_history_live import process_identity, request, write_json


def config(args):
    server = str(args.server.resolve())
    identity = {"server": server,
                "server_sha256": hashlib.sha256(args.server.read_bytes()).hexdigest(),
                "model": str(args.model.resolve()), "model_bytes": args.model.stat().st_size}
    write_json(args.output / "config.identity.json", identity)
    for context in (2048, 2049):
        command = [server, "--cuda", "-m", str(args.model), "--ctx", str(context),
                   "--max-seqs", "1", "--prefix-reuse", "off", "--mtp-mode", "off",
                   "--check-config"]
        result = subprocess.run(command, capture_output=True, text=True, timeout=180)
        (args.output / f"config-{context}.stdout").write_text(result.stdout)
        (args.output / f"config-{context}.stderr").write_text(result.stderr)
        write_json(args.output / f"config-{context}.command.json", command)
        plan = json.loads(result.stdout)
        write_json(args.output / f"config-{context}.json", plan)
        codes = [issue["code"] for issue in plan["issues"]]
        assert plan["effective"]["ctx"] == context, plan
        assert not plan["effective"]["disk"], plan
        if context == 2048:
            assert result.returncode == 0 and "ctx_unavailable" not in codes, plan
        else:
            assert result.returncode != 0 and "ctx_unavailable" in codes, plan
    print("GLM metadata config: ctx=2048 accepted; ctx=2049 rejected")


def http(args):
    assert args.pid is not None, "http needs --pid"
    write_json(args.output / "http.process.json", process_identity(args.pid))
    body = {"model": args.model_id, "prompt": "The capital of France is",
            "max_tokens": 2, "temperature": 0, "seed": 1, "reasoning_effort": "none"}
    write_json(args.output / "http-short.request.json", body)
    response = request(args.url, "/v1/completions", body)
    stats = request(args.url, "/v1/stats")
    write_json(args.output / "http-short.response.json", response)
    write_json(args.output / "http-short.stats.json", stats)
    assert stats["serving"]["effective"]["ctx"] == 2048, stats
    assert stats["last_request"]["effective_lane"] == "serial", stats
    assert response["usage"]["completion_tokens"] > 0, response
    assert response["usage"]["prompt_tokens_details"]["cached_tokens"] == 0, response
    body["prompt"] = " context-boundary" * 4096
    body["max_tokens"] = 1
    write_json(args.output / "http-overcap.request.json", body)
    try:
        response = request(args.url, "/v1/completions", body)
    except urllib.error.HTTPError as error:
        raw = error.read()
        (args.output / "http-overcap.error.json").write_bytes(raw)
        write_json(args.output / "http-overcap.status.json", {"status": error.code})
        # The existing serial native-error projection is HTTP 500. Record it;
        # a context-specific 400 is also a valid future projection.
        assert error.code in (400, 500), error.code
        assert b"context" in raw.lower() or b"ctx" in raw.lower(), raw
    else:
        write_json(args.output / "http-overcap.response.json", response)
        raise AssertionError("GLM accepted an over-cap prompt")
    print("GLM HTTP: serial ctx=2048 short inference passed; over-cap prompt rejected")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=["config", "http"])
    parser.add_argument("--server", type=Path, default=Path("target/release/ds4-server-rs"))
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--url", default="http://127.0.0.1:18039")
    parser.add_argument("--pid", type=int)
    parser.add_argument("--model-id", default="glm-5.3-flash")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    (config if args.phase == "config" else http)(args)


if __name__ == "__main__":
    main()
