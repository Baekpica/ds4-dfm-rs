#!/usr/bin/env python3
"""glm53_http_suite.py -- the five-case warm-path HTTP harness for the GLM-5.3 VQ
family (phase 7a of ds4-dfm-rs-handoff.md).

Provenance
  Shape from tests/ds41_http_suite.py, the ds41 line's own five-case warm-path
  harness; the phases it serves are named there. This file re-authors that shape
  for GLM-5.3 VQ with GLM cases: the recorded ds41 case bytes are English
  prose/code/table of the same kind, so the growth builders are kept and the
  case list is replaced. No number here is comparable across the two files.

Deviation from the ds41 shape, named and measured
  ds41_http_suite.py streams, because its columns were read off SSE deltas. The
  Rust server emits its `timings` block only on the non-streamed final body
  (crates/ds4-server/src/stream.rs:398, attached in openai_final / anthropic_final
  / responses_final), and `tok_per_step` -- the MTP column this unit needs -- is
  in that block and nowhere else (verified live 2026-10-17: a streamed
  /v1/chat/completions answered usage and no timings; the non-streamed body of
  the same request carried ttft_ms 935.9, prefill_tokens 16,
  prefill_cached_tokens 0, prefill_tok_s 17.1, decode_tok_s 23.9,
  tok_per_step 1.03). So this harness is non-streamed and takes ttft from the
  server. total_s is the client wall clock, which is now the honest bound on a
  cold request: it contains the fixed per-request cost this unit is hunting.

Protocol
  One non-streamed /v1/chat/completions per sample, temperature 0. Per case the
  first rep is cold and the remaining reps repeat the identical bytes, so they
  ride whatever prefix reuse the server has (harness probe 2026-10-17: with a
  clean prefix they do not -- cached_tokens stays 0 and prefill_tokens equals
  the whole prompt -- which is the finding 7c has to explain, not a bug here).

  Columns: engine,case,rep,cache,prompt_tokens,completion_tokens,ttft_s,
  total_s,prefill_tps,decode_tps,e2e_tps,finish,prompt_sha hold the ds41
  canonical fourteen minus sse_deltas (there are no deltas on this wire); the
  server_* group is the Rust server's own account, its names taken verbatim from
  stream.rs (ttft_ms, prefill_tokens, prefill_cached_tokens, prefill_tok_s,
  decode_tok_s, tok_per_step). tool_args is the raw JSON string a tool sample
  returned, so the argument-typing fix is visible in the row itself.

usage
  tests/glm53_http_suite.py --url http://192.168.1.91:8009 \
      --engine glm53vq-f738059 --out raw.csv
  then the same run with --cases to re-take one case, or --append to add samples.
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

# Measured live 2026-10-17 on the Spark artifact: 10590 chars of the prose block
# below rendered 2219 prompt tokens (glm-5.3-flash-uncensored-vq, the chat
# template's own render), i.e. 4.772 chars/token. The 64k case's character
# target is derived from that ratio and the case records the token count the
# server actually reports; the constant is a sizing aid, never a claim.
CHARS_PER_TOKEN = 4.772
CTX_CASE_TOKENS = 64 * 1024
LONG_CHARS = int(CTX_CASE_TOKENS * CHARS_PER_TOKEN)

PROSE = ("The engineer kept a written record of every change, because a change "
         "that leaves no trace is a change nobody can undo. Round %d restated "
         "the same problem in different words: the cache must hold what the "
         "next request will read, or the next request repeats the work.\n")
CODE = ("def step_%d(state, rows):\n"
        "    total = 0\n"
        "    for row in rows:\n"
        "        if row.valid:\n"
        "            total += row.weight * state.rate\n"
        "    return total\n\n")

# The tool schema is the shape the unit-1 fix was measured on: a property the
# client typed as u8 and one schemars renders as the list ["integer","null"]
# (Option<usize>), so a row shows whether the argument arrives as a number.
TOOLS = [{
    "type": "function",
    "function": {
        "name": "lookup",
        "description": "Look up a ledger record by code.",
        "parameters": {
            "type": "object",
            "properties": {
                "code": {"type": "string", "description": "Record code."},
                "limit": {"type": "u8", "description": "Rows to return."},
                "max_results": {"type": ["integer", "null"],
                                "description": "Upper bound."},
            },
            "required": ["code"],
        },
    },
}]

TOOL_PROMPT = ("Call the lookup tool once with code QZ5813, limit 3 and "
               "max_results 15. Do not answer in prose.")

# case -> (kind, characters, max_tokens, reps). The last case repeats the first
# case's bytes after every other case has run, so its rows measure whether the
# fixed cost decays over a session; the interleave case alternates two
# conversations so a bank re-keying shows up as a miss on the returning one.
CASES = [
    ("prose_short", "prose", 220, 128, 3),
    ("tool_short", "tool", 0, 512, 3),
    ("prose_64k", "prose", LONG_CHARS, 32, 2),
    ("interleave", "interleave", 220, 128, 3),
    ("prose_short_repeat", "repeat", 220, 128, 3),
]

CANON = ["engine", "case", "rep", "cache", "prompt_tokens", "completion_tokens",
         "ttft_s", "total_s", "prefill_tps", "decode_tps", "e2e_tps", "finish",
         "prompt_sha"]
EXTRA = ["server_ttft_ms", "server_prefill_tokens", "server_prefill_cached_tokens",
         "server_prefill_tok_s", "server_decode_tok_s", "tok_per_step",
         "cached_tokens", "cache_write_tokens", "tool_args", "reuse_line"]


def grow(block, target):
    """Repeat a block until the text reaches the character target."""
    out, size, i = [], 0, 0
    while size < target:
        piece = block % i
        out.append(piece)
        size += len(piece)
        i += 1
    return "".join(out)[:target]


def prose_text(target):
    return grow(PROSE, target)


def code_text(target):
    return grow(CODE, target)


def post(url, messages, max_tokens, timeout, tools=None, key="", model=""):
    """One non-streamed completion; returns the sample's measurements."""
    parsed = urllib.parse.urlsplit(url)
    body = {"messages": messages, "max_tokens": max_tokens, "temperature": 0}
    if tools:
        body["tools"] = tools
    if model:
        body["model"] = model
    payload = json.dumps(body).encode("utf-8")

    headers = {"Content-Type": "application/json", "Connection": "close"}
    if key:
        headers["Authorization"] = "Bearer " + key

    conn = http.client.HTTPConnection(parsed.hostname, parsed.port, timeout=timeout)
    t0 = time.perf_counter()
    conn.request("POST", "/v1/chat/completions", payload, headers)
    resp = conn.getresponse()
    raw = resp.read()
    t_end = time.perf_counter()
    if resp.status != 200:
        conn.close()
        raise RuntimeError("HTTP %d: %s" % (resp.status, raw[:400].decode("utf-8", "replace")))
    conn.close()

    doc = json.loads(raw)
    choice = (doc.get("choices") or [{}])[0]
    message = choice.get("message") or {}
    usage = doc.get("usage") or {}
    details = usage.get("prompt_tokens_details") or {}
    timings = doc.get("timings") or {}
    calls = message.get("tool_calls") or []
    args = calls[0].get("function", {}).get("arguments", "") if calls else ""

    return {
        "total_s": t_end - t0,
        "prompt_tokens": int(usage.get("prompt_tokens") or 0),
        "completion_tokens": int(usage.get("completion_tokens") or 0),
        "cached_tokens": int(details.get("cached_tokens") or 0),
        "cache_write_tokens": int(details.get("cache_write_tokens") or 0),
        "finish": choice.get("finish_reason") or "none",
        "tool_args": args,
        "server_ttft_ms": timings.get("ttft_ms", ""),
        "server_prefill_tokens": timings.get("prefill_tokens", ""),
        "server_prefill_cached_tokens": timings.get("prefill_cached_tokens", ""),
        "server_prefill_tok_s": timings.get("prefill_tok_s", ""),
        "server_decode_tok_s": timings.get("decode_tok_s", ""),
        "tok_per_step": timings.get("tok_per_step", ""),
    }


def derive(row):
    """Fill the canonical derived columns from the server's own ttft account."""
    ttft = float(row["server_ttft_ms"]) / 1e3 if row["server_ttft_ms"] != "" else row["total_s"]
    total, prompt = row["total_s"], row["prompt_tokens"]
    done = row["completion_tokens"]
    row["ttft_s"] = "%.4f" % ttft
    row["total_s"] = "%.4f" % total
    row["prefill_tps"] = "%.1f" % (prompt / ttft) if ttft > 0 else ""
    row["decode_tps"] = ("%.2f" % ((done - 1) / (total - ttft))
                         if done > 1 and total > ttft else "")
    row["e2e_tps"] = "%.2f" % (done / total) if total > 0 else ""
    return row


def sample(args, name, messages, max_tokens, rep, cache, sha, tools=None):
    """One measured sample, or None when the request failed."""
    try:
        out = post(args.url, messages, max_tokens, args.timeout, tools,
                   args.key, args.model)
    except (OSError, RuntimeError, ValueError) as error:
        print("!! %s rep %d failed: %s" % (name, rep, error), file=sys.stderr)
        return None
    row = {"engine": args.engine, "case": name, "rep": rep, "cache": cache,
           "prompt_sha": sha, "reuse_line": ""}
    row.update(out)
    row = derive(row)
    print("   %-19s rep %d %-5s prompt %-7s ttft %ss total %ss tok/step %s cached %s %s"
          % (name, rep, cache, row["prompt_tokens"], row["ttft_s"], row["total_s"],
             row["tok_per_step"], row["cached_tokens"], row["tool_args"][:40]))
    return row


def run_case(args, name, kind, chars, max_tokens, reps):
    """The samples of one case, in the order the case defines."""
    rows = []
    if kind in ("prose", "tool"):
        messages = [{"role": "user", "content": TOOL_PROMPT if kind == "tool"
                     else prose_text(chars)}]
        sha = hashlib.sha256(json.dumps(messages).encode()).hexdigest()[:16]
        tools = TOOLS if kind == "tool" else None
        for rep in range(reps):
            row = sample(args, name, messages, max_tokens, rep,
                         "cold" if rep == 0 else "warm", sha, tools)
            if row is None:
                break
            rows.append(row)
        return rows

    if kind == "repeat":
        # The first case's exact bytes, re-sent late in the run.
        messages = [{"role": "user", "content": prose_text(chars)}]
        return run_case(args, name, "prose", chars, max_tokens, reps)

    if kind == "interleave":
        # Two live conversations, alternating, each turn appending to itself.
        base = [{"role": "user", "content": prose_text(chars)},
                {"role": "user", "content": code_text(chars)}]
        for rep in range(reps):
            for index, opening in enumerate(base):
                history = [opening]
                for turn in range(1 + rep):
                    history = history + [{"role": "assistant", "content": "ok"},
                                         {"role": "user", "content": "Again."}]
                sha = hashlib.sha256(json.dumps(history).encode()).hexdigest()[:16]
                row = sample(args, "%s_%s" % (name, "ab"[index]), history, max_tokens,
                             rep, "cold" if rep == 0 else "warm", sha)
                if row is None:
                    return rows
                rows.append(row)
        return rows

    raise ValueError(kind)


def summarize(rows):
    by = {}
    for row in rows:
        by.setdefault(row["case"], []).append(row)
    print("\n%-22s %-10s %-11s %-11s %-10s %s"
          % ("case", "prompt", "server ttft", "prefill t/s", "decode t/s", "cached"))
    for name in dict.fromkeys(r["case"] for r in rows):
        rs = by[name]
        ttft = st.median([float(r["server_ttft_ms"]) for r in rs
                          if r["server_ttft_ms"] != ""] or [0.0])
        hit = st.median([r["cached_tokens"] for r in rs])
        print("%-22s %-10s %-11s %-11s %-10s %s"
              % (name, rs[0]["prompt_tokens"], "%.1f ms" % ttft,
                 rs[0]["server_prefill_tok_s"] if rs[0]["server_prefill_tok_s"] != "" else "-",
                 rs[0]["server_decode_tok_s"] if rs[0]["server_decode_tok_s"] != "" else "-",
                 hit))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--url", required=True, help="base URL of a running server")
    ap.add_argument("--engine", required=True, help="label for the engine column")
    ap.add_argument("--out", required=True, help="CSV path (raw samples)")
    ap.add_argument("--timeout", type=float, default=900.0)
    ap.add_argument("--key", default="", help="bearer token, when the server wants one")
    ap.add_argument("--model", default="", help="model field, when the server routes by it")
    ap.add_argument("--cases", default="", help="comma-separated case names to run")
    ap.add_argument("--append", action="store_true", help="append to --out instead of writing it")
    args = ap.parse_args()

    wanted = [c.strip() for c in args.cases.split(",") if c.strip()]
    cases = [c for c in CASES if not wanted or c[0] in wanted]
    if not cases:
        sys.exit("no cases selected")

    for name, kind, chars, max_tokens, reps in cases:
        print("# case %s (%s, %d chars, max %d, %d reps)"
              % (name, kind, chars, max_tokens, reps))

    rows = []
    for name, kind, chars, max_tokens, reps in cases:
        rows.extend(run_case(args, name, kind, chars, max_tokens, reps))

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
