#!/usr/bin/env python3
"""Prepare an exact native-tokenized synthetic GLM retrieval fixture; no GPU."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

FIXTURES = Path(__file__).parent / "fixtures/chat-template"
TEMPLATE = FIXTURES / "models/glm-uncensored/chat_template.jinja"
RECORDS = (("quartz", "QZ5813", .01), ("cedar", "CD7426", .50), ("harbor", "HB9632", .95))
INTRO = "This ledger stores three key/code records. Ignore filler. Preserve every code exactly.\n"
QUERY = '\nReturn the codes for quartz, cedar and harbor as a JSON object. Use only the recorded codes.'
MODEL_CTX = 1048576
OUTPUT_RESERVE = 64


def stamp(path):
    s = path.stat()
    return dict(path=str(path.resolve()), bytes=s.st_size, inode=s.st_ino,
                device=s.st_dev, mtime_ns=s.st_mtime_ns, ctime_ns=s.st_ctime_ns)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--tokens", type=int, default=6147)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.tokens < 4096 or args.tokens > MODEL_CTX - OUTPUT_RESERVE:
        parser.error(f"tokens must be within 4096..{MODEL_CTX - OUTPUT_RESERVE}")
    args.output.mkdir(parents=True, exist_ok=True)
    before = stamp(args.model)

    # Use the independent source oracle and the shared none-mode closing tag.
    sys.path.insert(0, str(FIXTURES))
    from make_model_vectors import CLOCK, Generation, fail, loopcontrols, python_json
    from jinja2.sandbox import ImmutableSandboxedEnvironment
    env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True,
                                       extensions=[loopcontrols, Generation])
    env.filters["tojson"] = python_json
    env.globals.update(raise_exception=fail, strftime_now=CLOCK.strftime)
    source = TEMPLATE.read_bytes()
    template = env.from_string(source.decode())
    prompt_file = args.output / "prompt.txt"
    token_file = args.output / "prompt.i32"

    def tokenize(text, stem):
        text_file = args.output / (stem + ".txt")
        ids = args.output / (stem + ".i32")
        text_file.write_text(text)
        result = subprocess.run([str(args.tokenizer.resolve()), str(args.model.resolve()),
                                 str(text_file), str(ids)], capture_output=True, text=True, check=True)
        (args.output / (stem + ".stderr")).write_text(result.stderr)
        return json.loads(result.stdout)["tokens"]

    counts = [max(0, int(args.tokens * RECORDS[0][2]) - 32)]
    counts += [int(args.tokens * (RECORDS[i][2] - RECORDS[i - 1][2])) - 16 for i in (1, 2)]
    counts += [int(args.tokens * (1 - RECORDS[-1][2])) - 48]
    for attempt in range(12):
        content = INTRO
        for i, (key, code, _) in enumerate(RECORDS):
            content += " note" * counts[i] + f"\nRecord {key}: {code}.\n"
        content += " note" * counts[-1] + QUERY
        messages = [{"role": "user", "content": content}]
        rendered = template.render(messages=messages, tools=[], add_generation_prompt=True,
                                   reasoning_effort="none") + "</think>"
        count = tokenize(rendered, "prompt")
        if count == args.tokens:
            break
        counts[-1] += args.tokens - count
        if counts[-1] < 0:
            raise RuntimeError("tail filler cannot absorb the token correction")
    else:
        raise RuntimeError("exact token length did not converge")

    positions = []
    for key, code, fraction in RECORDS:
        offset = rendered.index(f"Record {key}: {code}.")
        position = tokenize(rendered[:offset], "prefix-" + key)
        prefix = (args.output / ("prefix-" + key + ".i32")).read_bytes()
        if not token_file.read_bytes().startswith(prefix):
            raise RuntimeError("record prefix retokenized across its boundary")
        positions.append(dict(key=key, code=code, requested_fraction=fraction,
                              token_position=position, actual_fraction=position/count))
    assert stamp(args.model) == before, "model identity changed"
    manifest = dict(schema="glm53-synthetic-retrieval-v1", input_tokens=count,
                    synthetic_filler="repeated ASCII note, not natural-text throughput",
                    records=positions, messages=messages, model_stat=before,
                    template_sha256=hashlib.sha256(source).hexdigest(),
                    prompt_sha256=hashlib.sha256(prompt_file.read_bytes()).hexdigest(),
                    token_sha256=hashlib.sha256(token_file.read_bytes()).hexdigest(),
                    tokenizer_sha256=hashlib.sha256(args.tokenizer.read_bytes()).hexdigest())
    (args.output / "fixture.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({k: manifest[k] for k in ("schema", "input_tokens", "records")}), flush=True)


if __name__ == "__main__":
    main()
