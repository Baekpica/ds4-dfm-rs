#!/usr/bin/env python3
"""ds41_http_suite.py -- the five-case HTTP A/B harness for the warm-path unit.

Provenance
  The recorded comparison (2026-10-13) was produced by a harness named suite.py
  that is in neither tree: the port's record says so in its own Limits
  (docs/benchmarks/ds41-spark-2026-10-13/README.md) and /data/YoungAi/bench/
  carries only its outputs. This file is a re-authoring of it against the
  recorded contract, not a recovery: the case BYTES are gone (only their
  prompt_sha survived), so a run of this harness compares the port and the
  engine on identical bytes within one window - it does not reproduce the
  archived table case for case. Read the recorded table as the motivation for
  the unit, and this harness's own run as the measurement.

Protocol (the recorded one, kept so the numbers stay comparable in shape)
  Five cases over three content types at two prompt lengths; one streamed
  /v1/chat/completions request per case, temperature 0, greedy, three samples
  per case, where rep 0 is cold and reps 1..2 repeat the identical bytes so
  they ride the prefix-reuse checkpoint. The CSV columns are the contract that
  /data/YoungAi/bench/summarize.py reads:
      engine,case,rep,cache,prompt_tokens,completion_tokens,ttft_s,total_s,
      sse_deltas,prefill_tps,decode_tps,e2e_tps,finish,prompt_sha
  and this harness appends four of its own, so a run proves its own warm rows:
  server_ttft_ms, server_prefill_ms, cached_tokens (the server's own reuse
  account), reuse_line (the native's [v41] KV reuse line, when the server log
  is given). Column order leaves the canonical fifteen first and in order.

  The wire is the CHAT one on purpose: a /v1/completions request renders its
  prompt untemplated and the ds41 artifact answers such a prompt with an
  immediate stop (measured 2026-10-13: 0 tokens, finish "stop"), so a raw
  completion cannot carry the recorded 512-token cases. The recorded harness's
  own wire is unknown; this is the shape that generates.

  stream_options.include_usage is requested because the port emits the usage
  block only when asked (stream.rs:594, sse_usage_chunk), and the columns
  above need the server's own token counts. If a server answers without usage,
  prompt_tokens stays 0 and decode_tps falls back to the delta count.

  Derived fields follow the recorded arithmetic: prefill_tps = prompt/ttft_s,
  decode_tps = (completion-1)/(total-ttft_s), e2e_tps = completion/total_s.
  prompt_sha is sha256(case bytes) hex, first 16 - the archived recipe is
  unknown, so the column's job here is proving byte-identity across samples
  and sides WITHIN a run, which it does whatever the recipe was.

usage
  ds41_http_suite.py --url http://spark:8099 --engine ds4-dfm-rs --out raw.csv
  ds41_http_suite.py --url http://spark:8098 --engine youngai --out raw.csv --append
  then: cd <dir with raw.csv> && python3 /data/YoungAi/bench/summarize.py

  --penalty adds frequency_penalty 1e-30, the engine's own trick for taking the
  host penalty route (sampling and speculation off). Off by default: the
  recorded five-case run measured the drafted regime, so the default here is
  the default server configuration. Turn it on only to compare against the
  reuse gate's byte-exact regime (tests/ds41_kv_reuse_gate.sh).
"""

import argparse
import csv
import hashlib
import http.client
import json
import statistics as st
import sys
import time
import urllib.parse

# case -> (content kind, target characters, max_tokens). The recorded run used
# 54/54/43 prompt tokens and 512 completion tokens for the three short cases
# and ~10.36k prompt tokens with 32/256 completion tokens for the two long
# ones; characters are the only offline-proxy for tokens, so the recorded token
# counts are re-reported from the server, not assumed.
CASES = [
    ("prose_short", "prose", 220, 512),
    ("code_short", "code", 220, 512),
    ("table_short", "table", 190, 512),
    ("prose_long", "prose", 41500, 32),
    ("table_long", "table", 41500, 256),
]

CANON = [
    "engine", "case", "rep", "cache", "prompt_tokens", "completion_tokens",
    "ttft_s", "total_s", "sse_deltas", "prefill_tps", "decode_tps", "e2e_tps",
    "finish", "prompt_sha",
]
EXTRA = ["server_ttft_ms", "server_prefill_ms", "cached_tokens", "reuse_line"]


def grow(build, target):
    """Repeat a block builder until the text reaches the character target."""
    out = []
    size = 0
    i = 0
    while size < target:
        piece = build(i)
        out.append(piece)
        size += len(piece)
        i += 1
    return "".join(out)[:target]


def prose_block(i):
    return (
        "The engineer kept a written record of every change, because a change "
        "that leaves no trace is a change nobody can undo. Round %d restated "
        "the same problem in different words: the cache must hold what the "
        "next request will read, or the next request repeats the work.\n"
    ) % i


def code_block(i):
    return (
        "def step_%d(state, rows):\n"
        "    total = 0\n"
        "    for row in rows:\n"
        "        if row.valid:\n"
        "            total += row.weight * state.rate\n"
        "    return total\n\n"
    ) % i


def table_block(i):
    return (
        "| step | rows | rate | total |\n"
        "| ---: | ---: | ---: | ---: |\n"
        "| %d | %d | %.2f | %.3f |\n"
    ) % (i, 7 + i, 1.0 + i * 0.25, (7 + i) * (1.0 + i * 0.25))


BUILDERS = {"prose": prose_block, "code": code_block, "table": table_block}


def build_cases():
    cases = []
    for name, kind, chars, max_tokens in CASES:
        text = grow(BUILDERS[kind], chars)
        cases.append({
            "name": name,
            "text": text,
            "max_tokens": max_tokens,
            "sha": hashlib.sha256(text.encode("utf-8")).hexdigest()[:16],
        })
    return cases


def post_stream(url, prompt, max_tokens, timeout, key, penalty, model):
    """One streamed completion. Returns the sample's measurements."""
    parsed = urllib.parse.urlsplit(url)
    body = {
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if penalty:
        body["frequency_penalty"] = 1e-30
    if model:
        body["model"] = model
    payload = json.dumps(body).encode("utf-8")

    conn = http.client.HTTPConnection(parsed.hostname, parsed.port, timeout=timeout)
    headers = {"Content-Type": "application/json", "Connection": "close"}
    if key:
        headers["Authorization"] = "Bearer " + key

    t0 = time.perf_counter()
    conn.request("POST", "/v1/chat/completions", payload, headers)
    resp = conn.getresponse()
    if resp.status != 200:
        detail = resp.read(400).decode("utf-8", "replace")
        conn.close()
        raise RuntimeError("HTTP %d: %s" % (resp.status, detail))

    t_first = None
    deltas = 0
    finish = ""
    usage = {}
    timings = {}
    for raw in resp:
        line = raw.decode("utf-8", "replace").strip()
        if not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if data == "[DONE]":
            break
        try:
            chunk = json.loads(data)
        except ValueError:
            continue
        if chunk.get("usage"):
            usage = chunk["usage"]
        if chunk.get("timings"):
            timings = chunk["timings"]
        for choice in chunk.get("choices") or []:
            delta = choice.get("delta") or {}
            text = delta.get("content")
            if text is None:
                text = choice.get("text")
            if text:
                deltas += 1
                if t_first is None:
                    t_first = time.perf_counter()
            if choice.get("finish_reason"):
                finish = choice["finish_reason"]
    t_end = time.perf_counter()
    conn.close()

    return {
        "ttft_s": (t_first - t0) if t_first is not None else 0.0,
        "total_s": t_end - t0,
        "sse_deltas": deltas,
        "finish": finish or "none",
        "prompt_tokens": int(usage.get("prompt_tokens") or 0),
        "completion_tokens": int(usage.get("completion_tokens") or 0),
        "cached_tokens": int(
            (usage.get("prompt_tokens_details") or {}).get("cached_tokens") or 0
        ),
        "server_ttft_ms": timings.get("ttft_ms", ""),
        "server_prefill_ms": timings.get("prefill_ms", ""),
    }


def derive(row):
    ttft = row["ttft_s"]
    total = row["total_s"]
    prompt = row["prompt_tokens"]
    done = row["completion_tokens"] or row["sse_deltas"]
    row["prefill_tps"] = "%.1f" % (prompt / ttft) if ttft > 0 else ""
    row["decode_tps"] = (
        "%.2f" % ((done - 1) / (total - ttft)) if done > 1 and total > ttft else ""
    )
    row["e2e_tps"] = "%.2f" % (done / total) if total > 0 else ""
    row["ttft_s"] = "%.4f" % ttft
    row["total_s"] = "%.4f" % total
    return row


def summarize(rows):
    by = {}
    for row in rows:
        by.setdefault(row["case"], []).append(row)
    print("case                 cold ttft_s   cold prefill t/s   warm ttft ms   decode t/s   cached")
    for case, _kind, _chars, _max in CASES:
        rs = by.get(case)
        if not rs:
            continue
        cold = next((r for r in rs if r["cache"] == "cold"), rs[0])
        warm = [float(r["ttft_s"]) for r in rs if r["cache"] == "warm"]
        dec = [float(r["decode_tps"]) for r in rs if r["decode_tps"]]
        warm_ms = "%.1f" % (st.median(warm) * 1000) if warm else "-"
        print("%-20s %-12s %-18s %-14s %-12s %s" % (
            case, cold["ttft_s"], cold["prefill_tps"], warm_ms,
            "%.2f" % st.median(dec) if dec else "-", cold["cached_tokens"],
        ))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--url", required=True, help="base URL of a running server")
    ap.add_argument("--engine", required=True, help="label for the engine column")
    ap.add_argument("--out", required=True, help="CSV path (raw samples)")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--timeout", type=float, default=900.0)
    ap.add_argument("--key", default="", help="bearer token, when the server wants one")
    ap.add_argument("--model", default="", help="model field, when the server routes by it")
    ap.add_argument("--penalty", action="store_true", help="add frequency_penalty 1e-30")
    ap.add_argument("--append", action="store_true", help="append to --out instead of writing it")
    ap.add_argument("--cases", default="", help="comma-separated case names to run")
    args = ap.parse_args()

    wanted = [c.strip() for c in args.cases.split(",") if c.strip()]
    cases = [c for c in build_cases() if not wanted or c["name"] in wanted]
    if not cases:
        sys.exit("no cases selected")

    for case in cases:
        print("# case %s: %d chars, sha %s" % (case["name"], len(case["text"]), case["sha"]))

    rows = []
    for case in cases:
        for rep in range(args.reps):
            cache = "cold" if rep == 0 else "warm"
            try:
                sample = post_stream(
                    args.url, case["text"], case["max_tokens"], args.timeout,
                    args.key, args.penalty, args.model,
                )
            except (OSError, RuntimeError) as error:
                print("!! %s rep %d failed: %s" % (case["name"], rep, error), file=sys.stderr)
                break
            row = {
                "engine": args.engine,
                "case": case["name"],
                "rep": rep,
                "cache": cache,
                "prompt_sha": case["sha"],
                "reuse_line": "",
            }
            row.update(sample)
            row = derive(row)
            rows.append(row)
            print("   rep %d %-5s ttft %ss total %ss deltas %s cached %s" % (
                rep, cache, row["ttft_s"], row["total_s"], row["sse_deltas"],
                row["cached_tokens"],
            ))

    mode = "a" if args.append else "w"
    with open(args.out, mode, newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=CANON + EXTRA, extrasaction="ignore")
        if not args.append:
            writer.writeheader()
        writer.writerows(rows)

    if rows:
        summarize(rows)
    print("wrote %d samples to %s" % (len(rows), args.out))


if __name__ == "__main__":
    main()
