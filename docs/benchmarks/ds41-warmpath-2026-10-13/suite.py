#!/usr/bin/env python3
"""Client-side serving benchmark: identical cases, identical timing method, any engine.

For each case it sends one cold streaming request (fresh unique text) and N-1 warm
repeats, then records prompt/completion tokens, time-to-first-token and total time.
Everything is measured at the client, so it is engine-agnostic.
"""
import argparse, csv, json, os, sys, time, urllib.request, urllib.error

def model_id(url, timeout=20):
    with urllib.request.urlopen(url + "/v1/models", timeout=timeout) as r:
        data = json.loads(r.read().decode())
    return data["data"][0]["id"]

def stream_once(url, body, timeout):
    req = urllib.request.Request(url + "/v1/chat/completions",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json",
                                          "Accept": "text/event-stream"})
    t0 = time.perf_counter()
    ttft = None; total = None; deltas = 0; usage = None; finish = None
    with urllib.request.urlopen(req, timeout=timeout) as r:
        for raw in r:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                total = time.perf_counter() - t0
                break
            try:
                obj = json.loads(payload)
            except Exception:
                continue
            choices = obj.get("choices") or [{}]
            ch = choices[0]
            content = (ch.get("delta") or {}).get("content")
            if content:
                deltas += 1
                if ttft is None:
                    ttft = time.perf_counter() - t0
            if ch.get("finish_reason"):
                finish = ch["finish_reason"]
            if obj.get("usage"):
                usage = obj["usage"]
    if total is None:
        total = time.perf_counter() - t0
    return ttft, total, deltas, usage, finish

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--engine", required=True)
    ap.add_argument("--cases", required=True)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--reps", type=int, default=3, help="1 cold + (reps-1) warm")
    ap.add_argument("--timeout", type=float, default=1800.0)
    a = ap.parse_args()

    cases = json.load(open(a.cases))
    mid = model_id(a.url)
    new = not os.path.exists(a.csv)
    f = open(a.csv, "a", newline="")
    w = csv.writer(f)
    if new:
        w.writerow(["engine","case","rep","cache","prompt_tokens","completion_tokens",
                    "ttft_s","total_s","sse_deltas","prefill_tps","decode_tps","e2e_tps","finish","prompt_sha"])
    for c in cases:
        for rep in range(a.reps):
            tag = f"{c['name']}-{rep}"
            body = {"model": mid, "temperature": 0, "max_tokens": c["max_tokens"],
                    "stream": True, "stream_options": {"include_usage": True},
                    "messages": [{"role": "user", "content": c["prompt"]}]}
            try:
                ttft, total, deltas, usage, finish = stream_once(a.url, body, a.timeout)
            except Exception as e:
                print(f"  {tag}: ERROR {e}", flush=True)
                w.writerow([a.engine, c["name"], rep, "error" if rep==0 else "error", "", "", "", "", "", "", "", "", str(e)[:60], c["prompt_sha256"]])
                f.flush()
                continue
            pt = (usage or {}).get("prompt_tokens", "")
            ct = (usage or {}).get("completion_tokens", "")
            prefill = (pt / ttft) if (isinstance(pt, int) and ttft) else ""
            decode = ((ct - 1) / (total - ttft)) if (isinstance(ct, int) and ct > 1 and ttft and total > ttft) else ""
            e2e = (ct / total) if (isinstance(ct, int) and total) else ""
            w.writerow([a.engine, c["name"], rep, "cold" if rep == 0 else "warm",
                        pt, ct, f"{ttft:.4f}", f"{total:.4f}", deltas,
                        f"{prefill:.1f}" if prefill != "" else "",
                        f"{decode:.2f}" if decode != "" else "",
                        f"{e2e:.2f}" if e2e != "" else "", finish or "", c["prompt_sha256"]])
            f.flush()
            print(f"  {tag}: prompt={pt} completion={ct} ttft={ttft:.3f}s total={total:.3f}s "
                  f"decode={decode if decode=='' else round(decode,2)} t/s", flush=True)
    f.close()

if __name__ == "__main__":
    main()
