#!/usr/bin/env python3
"""Regenerate eval-cases.json for the Spark-X2.6-4B-Base input adapter.

Reference stack (the one the other family fixtures record):
  python3 -m venv <venv>
  <venv>/bin/pip install tokenizers transformers jinja2
  <venv>/bin/python tests/fixtures/spark2_5/make_eval_cases.py \
      <hf-dir>

`tokenizers` reads tokenizer.json exactly as Hugging Face ships it: the
normalizer Sequence (empty), the three Split stages, Digits(individual_digits)
and ByteLevel, with the 113 added tokens matched atomically. The rendered
cases come from Transformers' own Jinja environment (`_compile_jinja_template`
in transformers/utils/chat_template_utils.py: trim_blocks, lstrip_blocks,
loopcontrols, the Python `tojson`, `namespace`, `raise_exception`), and are
cross-checked against AutoTokenizer.apply_chat_template. The ids and bytes
therefore come from the reference pipeline, not from this tree's Rust code.
"""
import json
import random
import sys
from pathlib import Path

from tokenizers import Tokenizer
from transformers.utils.chat_template_utils import _compile_jinja_template

if len(sys.argv) < 2:
    raise SystemExit("usage: make_eval_cases.py <hf-dir> [fuzz-cases]")
MODEL = Path(sys.argv[1])
FUZZ = int(sys.argv[2]) if len(sys.argv) > 2 else 120
OUT = Path(__file__).with_name("eval-cases.json")
TEMPLATE = Path(__file__).with_name("chat_template.jinja")

TEXTS = [
    ("empty", ""),
    ("ascii", "The quick brown fox jumps over the lazy dog."),
    ("leading_space", " leading space"),
    ("trailing_space", "trailing spaces   "),
    ("double_space", "double  spaces and   more"),
    ("whitespace_only", "   \t \n  "),
    ("newlines", "line one\nline two\n\nline four"),
    ("crlf", "line one\r\nline two\r\n"),
    ("tabs", "a\tb\t\tc"),
    ("digits_1_3", "1 12 123"),
    ("digits_run", "1234 12345 1234567890"),
    ("digits_letters", "1a a1 2b3 4 5x"),
    ("digits_punct", "3.14159 1,000,000 +7 (555) 010-9999"),
    ("punctuation_chain", "!@#$%^&*()[]{}<>?/\\|~`"),
    ("punct_leading_word", "!!!hello ---world ???yes"),
    ("punct_only", "!!! ... ??? ,,, ;;;"),
    ("contractions", "it's I'll we've they're can't won't"),
    ("code", "fn main() { println!(\"hi\"); } // \u00e9"),
    ("underscore_ident", "snake_case_name and UPPER_CASE"),
    ("unicode_latin", "caf\u00e9 na\u00efve \u00c5ngstr\u00f6m \u00df"),
    ("combining_marks", "e\u0301 a\u0308 n\u0303"),
    ("cjk_han", "\u4f60\u597d\u4e16\u754c"),
    ("hiragana", "\u3053\u3093\u306b\u3061\u306f"),
    ("katakana", "\u30ab\u30bf\u30ab\u30ca"),
    ("hangul", "\uc548\ub155\ud558\uc138\uc694"),
    ("mixed_cjk_latin", "abc\u4f60\u597ddef \u3053\u3093 abc"),
    ("emoji", "ok \U0001F600\U0001F680 done \u2764\ufe0f"),
    ("symbols", "\u2192 \u2211 \u221e \u00b1 \u20ac \u00a3 \u00a5"),
    ("hyphen_words", "state-of-the-art well-known"),
    ("nbsp", "a\u00a0b\u2007c"),
    ("zero_width", "a\u200bb\u200dc\ufeffd"),
    ("fullwidth", "\uff21\uff22\uff23\uff11\uff12"),
    ("markers_bos_eos", "<\uff5cstart\u2581of\u2581sentence\uff5c><|System|>\nhi"
                        "<\uff5cend\u2581of\u2581sentence\uff5c>"),
    ("markers_roles", "<|System|><|User|><|Tool|><|Bot|><|Developer|>"),
    ("markers_think", "<think>plan</think>answer"),
    ("tool_grammar", "<tool_call>get_weather<arg_key>city</arg_key>"
                     "<arg_value>Paris</arg_value></tool_call>"),
    ("tools_text_marker", "<tools>{\"name\":\"x\"}</tools>"),
    ("reserved_slot", "<||unused2||> and <||unused87||>"),
    ("end_of_text", "<\uff5cstart\u2581of\u2581text\uff5c><\uff5cend\u2581of\u2581text\uff5c>"),
    ("plain_role_words", "<User><System><Bot>"),
    ("control_chars", "a\u0001b\u0002c"),
    ("long_mixed", ("word 123 \u4f60\u597d !!! it's\n" * 40) + "<|Bot|>tail"),
]


def fuzz_texts(count: int = 120) -> list[tuple[str, str]]:
    """Deterministic strings over one pool that hits every split class."""
    pool = list(
        "abcXYZ 0123456789.,;:!?-_()[]{}<>/\\'\"`~@#$%^&*+=|"
        "\n\r\t\u00a0\u2581\uff5c"
        "\u00e9\u00df\u0301\u0308"
        "\u4f60\u597d\u4e16\u754c\u3053\u3093\u30ab\u30bf"
        "\uc548\ub155"
        "\U0001F600\u2192\u2211\u20ac"
    )
    rng = random.Random(20261107)
    out = []
    for i in range(count):
        n = rng.randint(1, 48)
        out.append((f"fuzz_{i:03d}", "".join(rng.choice(pool) for _ in range(n))))
    return out


TOOL = {
    "type": "function",
    "function": {
        "name": "lookup",
        "description": "Find places: \uc11c\uc6b8, caf\u00e9, \u6f22\u5b57.",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string"},
                "labels": {"type": "array", "items": {"type": "string"}},
                "active": {"type": "boolean"},
                "scale": {"type": "number", "default": 1e-5},
                "limit": {"type": "integer"},
            },
            "required": ["city"],
        },
    },
}

ARGUMENTS = {
    "city": "\uc11c\uc6b8",
    "labels": ["\uc911\uad6c", "caf\u00e9"],
    "active": True,
    "scale": 1e-5,
    "limit": 2,
    "note": "quo\"te and\nnewline",
}

USER = {"role": "user", "content": "What is the capital of France? One word."}

CONVERSATIONS = [
    (
        "plain_user",
        {
            "messages": [USER],
            "add_generation_prompt": True,
        },
    ),
    (
        "plain_no_generation",
        {
            "messages": [USER],
            "add_generation_prompt": False,
        },
    ),
    (
        "system_user",
        {
            "messages": [
                {"role": "system", "content": "You are a terse assistant."},
                {"role": "user", "content": "Name three prime numbers."},
            ],
            "add_generation_prompt": True,
        },
    ),
    (
        "system_user_no_think",
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
        "thinking_low",
        {
            "messages": [USER],
            "add_generation_prompt": True,
            "reasoning_effort": "low",
        },
    ),
    (
        "thinking_high",
        {
            "messages": [USER],
            "add_generation_prompt": True,
            "reasoning_effort": "high",
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
        },
    ),
    (
        "assistant_reasoning",
        {
            "messages": [
                {"role": "user", "content": "Why is the sky blue?"},
                {
                    "role": "assistant",
                    "content": "Rayleigh scattering.",
                    "reasoning_content": "The short wavelengths scatter more.",
                },
                {"role": "user", "content": "One sentence, please."},
            ],
            "add_generation_prompt": True,
        },
    ),
    (
        "second_system",
        {
            "messages": [
                {"role": "system", "content": "First system."},
                {"role": "user", "content": "Hi"},
                {"role": "system", "content": "Later system."},
                {"role": "user", "content": "Again"},
            ],
            "add_generation_prompt": True,
        },
    ),
    (
        "assistant_tool_calls",
        {
            "messages": [
                {"role": "user", "content": "\uc11c\uc6b8\uc758 \uc7a5\uc18c\ub97c \uc54c\ub824 \uc918."},
                {
                    "role": "assistant",
                    "content": "",
                    "reasoning_content": "\ub3c4\uad6c\ub85c \ud655\uc778\ud55c\ub2e4.",
                    "tool_calls": [
                        {
                            "id": "call_01",
                            "type": "function",
                            "function": {"name": "lookup", "arguments": ARGUMENTS},
                        }
                    ],
                },
            ],
            "tools": [TOOL],
            "add_generation_prompt": True,
        },
    ),
    (
        "tool_result",
        {
            "messages": [
                {"role": "user", "content": "Find it."},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call_01",
                            "type": "function",
                            "function": {"name": "lookup", "arguments": {"city": "\uc11c\uc6b8"}},
                        }
                    ],
                },
                {"role": "tool", "content": "{\"name\":\"Seoul\",\"ok\":true}"},
            ],
            "tools": [TOOL],
            "add_generation_prompt": True,
        },
    ),
    (
        "consecutive_tools",
        {
            "messages": [
                {"role": "user", "content": "Find two."},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call_01",
                            "type": "function",
                            "function": {"name": "lookup", "arguments": {"city": "A"}},
                        },
                        {
                            "id": "call_02",
                            "type": "function",
                            "function": {"name": "lookup", "arguments": {"city": "B"}},
                        },
                    ],
                },
                {"role": "tool", "content": "first"},
                {"role": "tool", "content": "second"},
            ],
            "tools": [TOOL],
            "add_generation_prompt": True,
        },
    ),
    (
        "tools_defined",
        {
            "messages": [USER],
            "tools": [TOOL],
            "add_generation_prompt": True,
        },
    ),
    (
        "text_blocks",
        {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "first block "},
                        {"type": "text", "text": "\u4f60\u597d"},
                    ],
                }
            ],
            "add_generation_prompt": True,
        },
    ),
]

ERRORS = [
    ("no_messages", {"messages": [], "add_generation_prompt": True},
     "No messages provided."),
    ("developer_role", {"messages": [{"role": "developer", "content": "x"}],
                        "add_generation_prompt": True},
     "Unsupported message role: developer"),
    ("bad_effort", {"messages": [USER], "add_generation_prompt": True,
                    "reasoning_effort": "medium"},
     "Unexpected reasoning effort medium"),
    ("string_arguments", {
        "messages": [
            {"role": "user", "content": "x"},
            {"role": "assistant", "content": "", "tool_calls": [
                {"id": "c", "type": "function",
                 "function": {"name": "f", "arguments": "{\"a\":1}"}}]},
        ],
        "tools": [TOOL],
        "add_generation_prompt": True,
     }, "tool_call.function.arguments must be a dictionary"),
    ("image_block", {
        "messages": [{"role": "user", "content": [
            {"type": "image", "image": "x"}]}],
        "add_generation_prompt": True,
     }, "Unsupported user content block type: image"),
]


def render(template, context: dict) -> str:
    return template.render(**context)


def main() -> None:
    tok = Tokenizer.from_file(str(MODEL / "tokenizer.json"))
    template = _compile_jinja_template(TEMPLATE.read_text())

    cases = []
    for case_id, text in TEXTS + fuzz_texts(FUZZ):
        cases.append({"id": case_id, "text": text, "ids": tok.encode(text).ids})

    for case_id, context in CONVERSATIONS:
        text = render(template, context)
        cases.append(
            {"id": case_id, "context": context, "text": text, "ids": tok.encode(text).ids}
        )

    errors = []
    for case_id, context, needle in ERRORS:
        try:
            render(template, context)
        except Exception as error:  # noqa: BLE001 - the message is the fixture
            errors.append({"id": case_id, "context": context, "error": str(error)})
        else:
            raise SystemExit(f"{case_id}: the pinned template accepted it")

    doc = {
        "source": "tokenizers %s + transformers %s on %s"
        % (__import__("tokenizers").__version__,
           __import__("transformers").__version__, MODEL.name),
        "cases": cases,
        "errors": errors,
    }
    OUT.write_text(json.dumps(doc, indent=1, ensure_ascii=False) + "\n")
    print(f"wrote {OUT} with {len(cases)} cases and {len(errors)} errors")


if __name__ == "__main__":
    main()
