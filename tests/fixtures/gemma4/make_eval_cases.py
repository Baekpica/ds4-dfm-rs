#!/usr/bin/env python3
"""Regenerate eval-cases.json from the pinned checkpoint's own tokenizer.

Reference stack (the one the other family fixtures record):
  python3 -m venv /data/tmp/gemma4-ref-venv
  /data/tmp/gemma4-ref-venv/bin/pip install tokenizers jinja2
  /data/tmp/gemma4-ref-venv/bin/python tests/fixtures/gemma4/make_eval_cases.py \
      /data/models/gemma-4-26B-A4B-it-hf

`tokenizers` reads tokenizer.json exactly as Hugging Face ships it (the
normalizer, the Split(" ") stage, the BPE with byte_fallback and the added
tokens). jinja2 renders chat_template.jinja, the same engine HF's
apply_chat_template drives, with the raise_exception global HF supplies.
The ids therefore come from the reference pipeline, not from this tree's
Rust code.
"""
import json
import sys
from pathlib import Path

import jinja2
from tokenizers import Tokenizer

MODEL = Path(sys.argv[1] if len(sys.argv) > 1 else "/data/models/gemma-4-26B-A4B-it-hf")
OUT = Path(__file__).with_name("eval-cases.json")

# Plain text: whitespace forms the normalizer and the split stage see,
# unicode outside the merge table, digits, code and the turn controls.
TEXTS = [
    ("ascii", "The quick brown fox jumps over the lazy dog."),
    ("leading_space", " leading space and  double  spaces"),
    ("newlines", "line one\nline two\n\nline four"),
    ("tabs", "a\tb\t\tc"),
    ("digits", "1234567890 3.14159 1,000,000"),
    ("code", "fn main() { println!(\"hi\"); } // \u00e9"),
    ("unicode", "caf\u00e9 na\u00efve \u00c5ngstr\u00f6m \u00df"),
    ("cjk", "\u4f60\u597d\u4e16\u754c"),
    ("emoji", "ok \U0001F600\U0001F680 done"),
    ("controls", "<|turn>user\nhi<turn|>\n<|think|><|channel>thought\n<channel|>"),
    ("punct", "!@#$%^&*()[]{}<>?/\\|~`"),
    ("long", "word " * 60),
]

CONVERSATIONS = [
    (
        "en_single",
        {
            "messages": [{"role": "user", "content": "What is the capital of France? Answer in one word."}],
            "add_generation_prompt": True,
            "enable_thinking": True,
        },
    ),
    (
        "system_no_think",
        {
            "messages": [
                {"role": "system", "content": "You are a terse assistant."},
                {"role": "user", "content": "Name three prime numbers."},
            ],
            "add_generation_prompt": True,
            "enable_thinking": False,
        },
    ),
    (
        "multi_turn",
        {
            "messages": [
                {"role": "user", "content": "Hi"},
                {"role": "assistant", "content": "Hello! How can I help?"},
                {"role": "user", "content": "Write one line of Rust that prints 42."},
            ],
            "add_generation_prompt": True,
            "enable_thinking": True,
        },
    ),
]


def env(template_text: str) -> jinja2.Environment:
    def raise_exception(message):
        raise jinja2.exceptions.TemplateError(message)

    # HF renders in an ImmutableSandboxedEnvironment with the default
    # undefined, and supplies the tokenizer's special tokens as globals.
    e = jinja2.Environment()
    e.globals["raise_exception"] = raise_exception
    e.globals["strftime_now"] = lambda fmt: "2026-01-01"
    return e.from_string(template_text)


def main() -> None:
    tok = Tokenizer.from_file(str(MODEL / "tokenizer.json"))
    template_text = (MODEL / "chat_template.jinja").read_text()
    template = env(template_text)
    config = json.loads((MODEL / "tokenizer_config.json").read_text())
    special = {
        key: config[key]
        for key in ("bos_token", "eos_token", "unk_token", "pad_token")
        if key in config
    }

    cases = []
    for case_id, text in TEXTS:
        cases.append({"id": case_id, "text": text, "ids": tok.encode(text).ids})
    for case_id, context in CONVERSATIONS:
        text = template.render(**special, **context)
        cases.append(
            {"id": case_id, "context": context, "text": text, "ids": tok.encode(text).ids}
        )

    doc = {
        "source": "tokenizers %s + jinja2 %s on %s"
        % (__import__("tokenizers").__version__, jinja2.__version__, MODEL.name),
        "cases": cases,
    }
    OUT.write_text(json.dumps(doc, indent=1, ensure_ascii=False) + "\n")
    print(f"wrote {OUT} with {len(cases)} cases")


if __name__ == "__main__":
    main()
