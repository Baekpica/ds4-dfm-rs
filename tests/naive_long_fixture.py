#!/usr/bin/env python3
"""Build a near-capacity archive retrieval request with the official template.

Requires tokenizers/Jinja2 and the pinned source tokenizer.json; no weights.
"""
import argparse
import hashlib
import json
from pathlib import Path
import struct

from jinja2 import Environment
from tokenizers import Tokenizer

CONTEXTS = (32768, 262144, 524288, 1048576)
PROMPT_MARGIN = 256
MAX_OUTPUT = 64
NEEDLE_PERCENT = 17
ANSWER = "ORCHID-74-COPPER-29"
ARCHIVE_LINE = "Archive note: this page records routine shipments and inventory checks.\n"
START = "Read this archive. A named depot has a unique access phrase in its records.\n"
NEEDLE = f"IMPORTANT: The access phrase for depot Aster is {ANSWER}.\n"
QUESTION = ("\nEnd of archive. What is the access phrase for depot Aster? "
            "Reply with the exact phrase only.")
SYSTEM = "Retrieve the requested archive fact exactly. Return only its value."


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--context", type=int, choices=CONTEXTS, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--template", type=Path, default=Path(__file__).resolve().parent /
                        "fixtures/chat-template/models/naive/chat_template.jinja")
    parser.add_argument("--model", default="naive-n05-flash")
    parser.add_argument("--prompt-margin", type=int, default=PROMPT_MARGIN)
    parser.add_argument("--max-output", type=int, default=MAX_OUTPUT)
    parser.add_argument("--output-mode", choices=("stream", "buffered"), default="stream")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if not 0 < args.max_output <= args.prompt_margin < args.context:
        parser.error("output budget must fit the positive prompt margin")
    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    template = Environment().from_string(args.template.read_text())

    def encode(text):
        return tokenizer.encode(text, add_special_tokens=False).ids

    def render(content):
        messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": content}]
        text = template.render(messages=messages, add_generation_prompt=True,
                               enable_thinking=False, reasoning_effort="none", tools=[])
        return messages, text

    assert len(encode(" x")) == 1
    target = args.context - args.prompt_margin
    unit = len(encode(ARCHIVE_LINE * 2)) - len(encode(ARCHIVE_LINE))
    count = (target - len(encode(render(START + NEEDLE + QUESTION)[1]))) // unit - 2
    cut = count * NEEDLE_PERCENT // 100
    content = START + ARCHIVE_LINE * cut + NEEDLE + ARCHIVE_LINE * (count - cut)
    gap = target - len(encode(render(content + QUESTION)[1]))
    assert gap >= 0
    # Boundary tokenization can change a padding row by one token.
    for _ in range(4):
        messages, text = render(content + " x" * gap + QUESTION)
        ids = encode(text)
        if len(ids) == target:
            break
        gap += target - len(ids)
    assert len(ids) == target, (len(ids), target)

    args.out.mkdir(parents=True, exist_ok=False)
    body = {"model": args.model, "messages": messages, "temperature": 0, "seed": 1,
            "max_tokens": args.max_output, "reasoning_effort": "none",
            "stream": args.output_mode == "stream"}
    if body["stream"]:
        body["stream_options"] = {"include_usage": True}
    (args.out / "request.json").write_text(json.dumps(body, ensure_ascii=False) + "\n")
    (args.out / "prompt.txt").write_text(text)
    (args.out / "prompt.i32").write_bytes(struct.pack(f"<{len(ids)}i", *ids))
    receipt = {"context": args.context, "input_tokens": len(ids), "expected_answer": ANSWER,
               "needle_fraction": NEEDLE_PERCENT / 100,
               "needle_token_offset": len(encode(render(START + ARCHIVE_LINE * cut)[1])),
               "request_sha256": digest(args.out / "request.json"),
               "prompt_ids_sha256": digest(args.out / "prompt.i32"),
               "template_sha256": digest(args.template), "tokenizer_sha256": digest(args.tokenizer),
               "request_bytes": (args.out / "request.json").stat().st_size,
               "prompt_margin": args.prompt_margin, "max_output": args.max_output,
               "output_mode": args.output_mode}
    (args.out / "fixture.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
