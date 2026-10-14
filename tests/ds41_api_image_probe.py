#!/usr/bin/env python3
"""Live API check: one image request through every wire surface that carries
images, and the answers compared.

The port renders every surface into the same V4.1 DSML prompt
(crates/ds4-server/src/render.rs: the head from the official encoding.py,
DSML_IMAGE per image part, the V4.1 branch gated on an image part at :1327),
so at temperature 0 the greedy answer must be identical across surfaces.  The
image parts parse per shape: OpenAI image_url requires a base64 data URI
(parse.rs:900/1013), Anthropic image source{type,media_type,data}
(parse.rs:1173-1210 -> :1299), Responses input_image (:2150-2161); all land in
one image list that crates/ds4-server/src/generate.rs:3006 expand_v41_images
turns into the ds41 image span.  Without the sidecar every surface is refused
with HTTP 400 by name.

usage: ds41_api_image_probe.py [base-url] [image]
       (default http://127.0.0.1:8000 tests/fixtures/vision/homer.png)

exit 0 when every surface answered and all answers are identical; 1 otherwise.
"""
import base64
import json
import sys
import urllib.error
import urllib.request

base = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8000"
img = sys.argv[2] if len(sys.argv) > 2 else "tests/fixtures/vision/homer.png"
mime = "image/jpeg" if img.lower().endswith((".jpg", ".jpeg")) else "image/png"
prompt = "What is in this image? List all the text and numbers you can see."
b64 = base64.b64encode(open(img, "rb").read()).decode()

mid = json.load(urllib.request.urlopen(base + "/v1/models"))["data"][0]["id"]


def post(path, body):
    req = urllib.request.Request(base + path, json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    try:
        return json.load(urllib.request.urlopen(req, timeout=900))
    except urllib.error.HTTPError as e:
        print(f"{path}: HTTP {e.code}: {e.read().decode()[:200]}")
        return None


answers = {}

oai = post("/v1/chat/completions", {
    "model": mid, "temperature": 0, "max_tokens": 64,
    "messages": [{"role": "user", "content": [
        {"type": "text", "text": prompt},
        {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}}]}]})
if oai:
    answers["openai /v1/chat/completions"] = oai["choices"][0]["message"]["content"].strip()

ant = post("/v1/messages", {
    "model": mid, "temperature": 0, "max_tokens": 64,
    "messages": [{"role": "user", "content": [
        {"type": "text", "text": prompt},
        {"type": "image", "source": {"type": "base64", "media_type": mime, "data": b64}}]}]})
if ant:
    answers["anthropic /v1/messages"] = "".join(
        b.get("text", "") for b in ant.get("content", []) if b.get("type") == "text").strip()

res = post("/v1/responses", {
    "model": mid, "temperature": 0, "max_output_tokens": 64,
    "input": [{"role": "user", "content": [
        {"type": "input_text", "text": prompt},
        {"type": "input_image", "image_url": f"data:{mime};base64,{b64}"}]}]})
if res:
    text = "".join(
        c.get("text", "") for item in res.get("output", [])
        for c in item.get("content", []) if c.get("type") == "output_text")
    answers["responses /v1/responses"] = text.strip()

for name, answer in answers.items():
    print(f"=== {name} ===")
    print(answer)

values = [a for a in answers.values() if a]
ok = len(answers) == 3 and len(values) == 3 and len(set(values)) == 1
print(f"=== surfaces answered: {len(answers)}/3; identical: {len(set(values)) == 1}")
sys.exit(0 if ok else 1)
