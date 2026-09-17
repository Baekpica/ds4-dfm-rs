#!/usr/bin/env python3
"""Chat append/edit/fork/restart/cold gate against an externally managed server.

Run seed, then warm on that process; restart with the same disk cache and run
restored; start a third process with reuse/disk off and run cold. This runner
only sends requests and writes evidence. See serving-reuse-live.md for flags.
"""

import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
import time
import urllib.error

from step37_history_live import fingerprint, process_identity, request, write_json

PROFILES = {
    "qwen": {"names": ["qwen4exp"], "reuse": "partial"},
    "solar": {"names": ["solar-open2"], "reuse": "partial"},
    "motif": {"names": ["motif3"], "reuse": "partial"},
    "deepseek": {"names": ["deepseek4-flash", "deepseek4-pro"], "reuse": "exact"},
}
FIXTURE = Path(__file__).parent / "fixtures" / "serving-reuse.json"


def read_json(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def reference_phase(name):
    return "seed" if name == "seed" else "restored" if name == "restart" else "warm"


def verify_fixture(output, phase):
    previous = {"warm": "seed", "restored": "warm", "cold": "restored"}[phase]
    receipt = read_json(output / f"{previous}.result.json")
    require(digest(output / "fixture.json") == receipt["fixture_sha256"],
            "fixture changed since previous phase")


def verify_cold_body(output, name, body):
    original = read_json(output / f"{reference_phase(name)}.{name}.request.json")
    require(body == original, f"cold request differs from recorded body: {name}")


def follow_body(body, response, user):
    # Preserve the generated text, including leading/trailing whitespace.
    # Thinking and tool output are rejected by inspect_case, not synthesized.
    follow = copy.deepcopy(body)
    message = response["choices"][0]["message"]
    follow["messages"].extend([
        {"role": "assistant", "content": message["content"]},
        {"role": "user", "content": user},
    ])
    return follow


def plan_errors(config, phase, plan):
    errors = []
    requested, effective = plan.get("requested", {}), plan.get("effective", {})
    profile = PROFILES[config["family"]]
    reuse = "off" if phase == "cold" else profile["reuse"]
    if plan.get("family") not in profile["names"]:
        errors.append(f"family: {plan.get('family')} != {profile['names']}")
    expected = {"ctx": config["context"], "max_seqs": config["banks"],
                "native_chunk": config["native_chunk"], "mtp_mode": config["mtp_mode"],
                "sched_chunk": config["native_chunk"], "sched_chunk_live": config["native_chunk"],
                "mtp_draft": config["mtp_draft"], "prefix_reuse": reuse,
                "disk": phase != "cold"}
    for key, value in expected.items():
        if effective.get(key) != value:
            errors.append(f"effective {key}: {effective.get(key)!r} != {value!r}")
    expected_requested = {"lane": "auto", "mtp_mode": config["mtp_mode"],
                          "ctx": config["context"], "max_seqs": str(config["banks"]),
                          "prefix_reuse": reuse, "backend": "cuda"}
    for key, value in expected_requested.items():
        if requested.get(key) != value:
            errors.append(f"requested {key}: {requested.get(key)!r} != {value!r}")
    if phase != "cold":
        for key in ("bank_persist_min_tokens", "disk_min_tokens"):
            if effective.get(key) != 1:
                errors.append(f"short fixture requires {key}=1")
    if any(issue.get("level") == "error" for issue in plan.get("issues", [])):
        errors.append("serving plan has errors")
    return errors


def inspect_case(config, phase, name, case, response, stats, reference=None):
    errors = []
    choice = response["choices"][0]
    message = choice["message"]
    text = message.get("content")
    # Compare only the frozen literal forms; never extract a number from prose.
    if not isinstance(text, str) or text.strip() not in case["accepted_forms"]:
        errors.append(f"arithmetic answer form: {text!r} not in {case['accepted_forms']!r}")
    if message.get("reasoning_content") or message.get("tool_calls"):
        errors.append("fixture requires plain visible output without reasoning/tools")
    if choice.get("finish_reason") != "stop":
        errors.append(f"finish_reason: {choice.get('finish_reason')!r} != stop")
    usage = response["usage"]
    cached = usage["prompt_tokens_details"]["cached_tokens"]
    prompt = usage["prompt_tokens"]
    trace = stats.get("last_request") or {}
    if trace.get("effective_lane") != config["lane"]:
        errors.append(f"lane: {trace.get('effective_lane')!r} != {config['lane']}")
    if trace.get("speculation_active") is not config["expect_speculation"]:
        errors.append(f"speculation: {trace.get('speculation_active')!r} != {config['expect_speculation']}")
    if trace.get("fallback_reason"):
        errors.append(f"fallback: {trace['fallback_reason']}")
    if phase == "cold" or name == "seed":
        kinds = {"cold"}
        if cached != 0:
            errors.append(f"cold cached tokens: {cached} != 0")
    else:
        if not 0 < cached < prompt:
            errors.append(f"cached tokens: expected 0 < {cached} < {prompt}")
        if name == "edit":
            kinds = {"partial"} if PROFILES[config["family"]]["reuse"] == "partial" else {"exact", "fork"}
        else:
            kinds = {"exact", "fork"}
    if trace.get("reuse_kind") not in kinds:
        errors.append(f"reuse: {trace.get('reuse_kind')!r} not in {sorted(kinds)}")
    if reference is not None:
        other = reference["choices"][0]
        if (message != other["message"] or choice.get("finish_reason") != other.get("finish_reason")
                or usage["completion_tokens"] != reference["usage"]["completion_tokens"]):
            errors.append("cold comparison: message, finish_reason or completion count differs")
    return errors


def route_count(stats):
    return sum(stats["routes"].values())


def run_case(args, config, name, case, previous):
    key = f"{args.phase}.{name}"
    if args.phase == "cold":
        verify_cold_body(args.output, name, case["body"])
    write_json(args.output / f"{key}.request.json", case["body"])
    start = time.monotonic()
    try:
        response = request(args.url, "/v1/chat/completions", case["body"])
    except urllib.error.HTTPError as error:
        (args.output / f"{key}.error.txt").write_bytes(error.read())
        raise
    elapsed = time.monotonic() - start
    write_json(args.output / f"{key}.response.json", response)
    stats = request(args.url, "/v1/stats")
    write_json(args.output / f"{key}.stats.json", stats)
    reference = None
    if args.phase == "cold":
        phase = reference_phase(name)
        reference = read_json(args.output / f"{phase}.{name}.response.json")
    errors = inspect_case(config, args.phase, name, case, response, stats, reference)
    errors.extend(plan_errors(config, args.phase, stats.get("serving", {})))
    if route_count(stats) != route_count(previous) + 1:
        errors.append("route count changed by other than one; concurrent traffic invalidates this trace")
    summary = {"phase": args.phase, "case": name, "seconds": elapsed,
               "configured": config, "expected_answer": case["answer"],
               "accepted_forms": case["accepted_forms"],
               "message": response["choices"][0]["message"], "usage": response["usage"],
               "trace": stats.get("last_request"), "errors": errors,
               "passed": not errors}
    write_json(args.output / f"{key}.summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    return response, stats, errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=["seed", "warm", "restored", "cold"])
    parser.add_argument("--url", required=True)
    parser.add_argument("--pid", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--artifact-manifest", type=Path, required=True)
    parser.add_argument("--family", choices=PROFILES)
    parser.add_argument("--model")
    parser.add_argument("--context", type=int)
    parser.add_argument("--banks", type=int, choices=[2])
    parser.add_argument("--native-chunk", type=int)
    parser.add_argument("--mtp-mode", choices=["off", "on"])
    parser.add_argument("--mtp-draft", type=int)
    parser.add_argument("--expect-speculation", choices=["off", "on"])
    parser.add_argument("--lane", choices=["continuous"])
    parser.add_argument("--padding-lines", type=int, default=64)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    phase_path = args.output / f"{args.phase}.process.json"
    require(not phase_path.exists(), f"refusing to overwrite phase evidence: {phase_path}")
    process = process_identity(args.pid)
    process["artifacts_sha256"] = digest(args.artifact_manifest)
    process["url"] = args.url
    fixture_path = args.output / "fixture.json"
    if args.phase == "seed":
        required = ("family", "model", "context", "banks", "native_chunk", "mtp_mode",
                    "expect_speculation", "lane")
        require(all(getattr(args, key) is not None for key in required), "seed requires all shape/mode flags")
        require(args.padding_lines >= 64, "use >=64 padding lines to pass native partial minimum")
        require(args.context > 0 and args.native_chunk > 0, "context and chunk must be positive")
        require(args.mtp_mode == "on" or args.mtp_draft is None, "off mode must not specify a draft")
        require(args.mtp_mode == "off" or args.mtp_draft is not None and args.mtp_draft > 0,
                "on mode requires an explicit positive draft")
        require(args.mtp_mode == "on" or args.expect_speculation == "off", "off mode cannot speculate")
        require(args.family not in ("solar", "motif") or args.mtp_mode == "off", "this family has no MTP")
        config = {key: getattr(args, key) for key in (*required, "mtp_draft", "padding_lines")}
        config["expect_speculation"] = args.expect_speculation == "on"
        config["requested_lane"] = "auto"
        templates = read_json(FIXTURE)
        padding = "".join(templates["padding"].format(i=i) for i in range(args.padding_lines))
        body = {"model": args.model, "temperature": 0, "seed": 1, "max_tokens": 32,
                "reasoning_effort": "none",
                "messages": [{"role": "user", "content": padding + templates["seed"]["user"]}]}
        cases = {"seed": {"body": body, "answer": templates["seed"]["answer"],
                          "accepted_forms": templates["seed"]["accepted_forms"]}}
        fixture = {"schema": "serving-reuse-live-v2", "answer_contract": templates["answer_contract"],
                   "config": config, "templates": templates,
                   "source_fixture_sha256": digest(FIXTURE), "cases": cases}
        (args.output / "artifacts.json").write_bytes(args.artifact_manifest.read_bytes())
    else:
        require(not any(getattr(args, key) is not None for key in (
            "family", "model", "context", "banks", "native_chunk", "mtp_mode", "mtp_draft",
            "expect_speculation", "lane")), "later phases use the frozen seed configuration")
        verify_fixture(args.output, args.phase)
        fixture = read_json(fixture_path)
        require(fixture.get("schema") == "serving-reuse-live-v2"
                and fixture.get("answer_contract") == "literal-arithmetic-v2",
                "answer-form contract changed; start a new evidence directory")
        config, templates, cases = fixture["config"], fixture["templates"], fixture["cases"]
        seed = read_json(args.output / "seed.process.json")
        require((fingerprint(process) == fingerprint(seed)) == (args.phase == "warm"),
                "warm needs seed process; restored/cold need a fresh process")
        require(process["executable_sha256"] == seed["executable_sha256"], "binary changed")
        require(process["artifacts_sha256"] == seed["artifacts_sha256"], "artifact manifest changed")
        if args.phase == "cold":
            restored = read_json(args.output / "restored.process.json")
            require(fingerprint(process) != fingerprint(restored), "cold needs a third process")
    stats = request(args.url, "/v1/stats")
    write_json(args.output / f"{args.phase}.before.json", stats)
    errors = plan_errors(config, args.phase, stats.get("serving", {}))
    require(not errors, "; ".join(errors))
    require(route_count(stats) == (1 if args.phase == "warm" else 0),
            "unexpected prior traffic; use a dedicated fresh server")
    write_json(phase_path, process)
    write_json(fixture_path, fixture)
    all_errors = []
    warm_kinds = []
    names = {"seed": ["seed"], "warm": ["append", "edit", "fork"],
             "restored": ["restart"], "cold": ["seed", "append", "edit", "fork", "restart"]}[args.phase]
    for name in names:
        response, stats, errors = run_case(args, config, name, cases[name], stats)
        all_errors.extend(f"{name}: {error}" for error in errors)
        if args.phase == "warm":
            warm_kinds.append((stats.get("last_request") or {}).get("reuse_kind"))
        if args.phase == "seed":
            for follow in ("append", "edit"):
                cases[follow] = {"body": follow_body(cases[name]["body"], response, templates[follow]["user"]),
                                 "answer": templates[follow]["answer"],
                                 "accepted_forms": templates[follow]["accepted_forms"]}
        elif args.phase == "warm" and name in ("append", "fork"):
            follow = "fork" if name == "append" else "restart"
            cases[follow] = {"body": follow_body(cases[name]["body"], response, templates[follow]["user"]),
                             "answer": templates[follow]["answer"],
                             "accepted_forms": templates[follow]["accepted_forms"]}
        write_json(fixture_path, fixture)
    # A later branch can extend a parent that is still in its original bank.
    # Require a real bank copy somewhere in this phase, without prescribing
    # which eligible continuation the scheduler assigns to the other bank.
    if args.phase == "warm" and "fork" not in warm_kinds:
        all_errors.append("warm: no bank fork observed")
    require(fingerprint(process_identity(args.pid)) == fingerprint(process), "server changed during phase")
    receipt = {"passed": not all_errors, "errors": all_errors, "fixture_sha256": digest(fixture_path),
               "process": process, "answer_contract": fixture["answer_contract"], "files": []}
    for path in sorted(args.output.glob(f"{args.phase}.*")):
        receipt["files"].append({"path": path.name, "sha256": digest(path)})
    write_json(args.output / f"{args.phase}.result.json", receipt)
    return 1 if all_errors else 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (RuntimeError, KeyError, ValueError, OSError) as error:
        sys.exit(str(error))
