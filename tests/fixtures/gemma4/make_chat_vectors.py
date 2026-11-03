#!/usr/bin/env python3
"""Regenerate server-vectors.json for the official Gemma-4 chat template.

Reference stack (the one the Gemma-4 tokenizer fixture records):
  python3 -m venv <ref-venv>
  <ref-venv>/bin/pip install jinja2
  <ref-venv>/bin/python tests/fixtures/gemma4/make_chat_vectors.py

jinja2 renders chat_template.jinja with the same environment Hugging Face's
apply_chat_template builds (sandboxed, loopcontrols, Python json filters), so
the expected bytes come from the reference engine, not from this tree's Rust
code. The template file is copied verbatim from the pinned checkpoint
(revision 4d7ae4984b7db7de8f8457170b3f1a419ee76d52) and its sha256 is recorded
in provenance.json.
"""
import copy
import hashlib
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import jinja2
from jinja2.ext import loopcontrols
from jinja2.sandbox import ImmutableSandboxedEnvironment

ROOT = Path(__file__).parent
CLOCK = datetime.fromtimestamp(0, timezone.utc)

TOOL = {
    "type": "function",
    "function": {
        "name": "lookup",
        "description": "Find places: 서울, café, 漢字.",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "object",
                    "properties": {
                        "city": {"type": "string"},
                        "labels": {"type": "array", "items": {"type": "string"}},
                        "active": {"type": "boolean"},
                    },
                    "required": ["city"],
                },
                "scale": {"type": "number", "default": 1e-5},
                "limit": {"type": "integer"},
            },
            "required": ["location"],
        },
    },
}
ARGUMENTS = {
    "location": {"city": "서울", "labels": ["중구", "café"], "active": True},
    "scale": 1e-5,
    "limit": 2,
}
USER = {"role": "user", "content": "  서울의 장소를 알려 줘. café도 좋아.\n"}
ASSISTANT = {
    "role": "assistant",
    "content": "서울에서 찾을 수 있어요.",
    "reasoning_content": "질문의 장소를 확인한다.",
}
MULTITURN = [
    {"role": "system", "content": "Be concise. 한국어로 답해."},
    USER,
    ASSISTANT,
    {"role": "user", "content": "두 곳만 골라 줘."},
]
TOOL_HISTORY = [
    USER,
    {
        "role": "assistant",
        "content": "",
        "reasoning_content": "도구로 위치를 확인한다.",
        "tool_calls": [
            {"id": "call_01", "type": "function", "function": {"name": "lookup", "arguments": ARGUMENTS}}
        ],
    },
    {
        "role": "tool",
        "tool_call_id": "call_01",
        "content": '{"city": "서울", "items": ["café", "漢字"], "count": 2}',
    },
]


def python_json(value, **kwargs):
    return json.dumps(value, **{"ensure_ascii": False, **kwargs})


def fail(message):
    raise jinja2.TemplateError(message)


def case(name, messages, thinking, preserve=False, tools=None, add_generation_prompt=True):
    return {
        "name": name,
        "context": {
            "messages": copy.deepcopy(messages),
            "tools": copy.deepcopy(tools or []),
            "add_generation_prompt": add_generation_prompt,
            "enable_thinking": thinking,
            "preserve_thinking": preserve,
            "bos_token": "<bos>",
        },
    }


def cases():
    out = [
        case("user_thinking_off", [USER], False),
        case("user_thinking_on", [USER], True),
        case("system_multiturn", MULTITURN, True),
        case("tool_schema", [USER], True, tools=[TOOL]),
        case("tool_result", TOOL_HISTORY, True, tools=[TOOL]),
        case("no_generation", [USER, ASSISTANT], True, add_generation_prompt=False),
        case("preserve_tool_thinking", TOOL_HISTORY, True, preserve=True, tools=[TOOL]),
        case(
            "media_placeholders",
            [{"role": "user", "content": [
                {"type": "text", "text": "입력을 설명해 줘."},
                {"type": "image"},
                {"type": "audio"},
                {"type": "video"},
            ]}],
            False,
        ),
        case("tool_thinking_off", TOOL_HISTORY, False, tools=[TOOL]),
    ]
    # The template rejects string arguments by design; the server must
    # deserialize them before rendering (chat_template.jinja:258).
    bad = copy.deepcopy(TOOL_HISTORY)
    bad[1]["tool_calls"][0]["function"]["arguments"] = json.dumps(ARGUMENTS, ensure_ascii=False)
    out.append(case("reject_string_arguments", bad, True, tools=[TOOL]))
    return out


def main():
    env = ImmutableSandboxedEnvironment(
        trim_blocks=True, lstrip_blocks=True, extensions=[loopcontrols]
    )
    env.filters["tojson"] = python_json
    env.globals["raise_exception"] = fail
    env.globals["strftime_now"] = CLOCK.strftime

    provenance = json.loads((ROOT / "provenance.json").read_text(encoding="utf-8"))
    source = (ROOT / "chat_template.jinja").read_bytes()
    digest = hashlib.sha256(source).hexdigest()
    assert digest == provenance["template_sha256"], "template drifted from the pin"
    template = env.from_string(source.decode("utf-8"))

    vectors = []
    for row in cases():
        rejection = row["name"] == "reject_string_arguments"
        try:
            row["expected"] = template.render(**row["context"])
        except (jinja2.TemplateError, TypeError, ValueError) as error:
            assert rejection, f"{row['name']}: {error}"
            row["error_type"] = type(error).__name__
            row["error_message"] = str(error)
        else:
            assert not rejection, f"{row['name']}: expected a rejection"
        vectors.append(row)
        print(f"{row['name']}: {'rejected' if rejection else 'rendered'}")

    result = {
        "oracle": "Python Jinja2 with the Hugging Face rendering environment",
        "python_version": platform.python_version(),
        "jinja2_version": jinja2.__version__,
        "template_sha256": digest,
        "clock_unix_seconds": 0,
        "scope": "Template bytes only; media placeholders do not qualify model media support.",
        "vectors": vectors,
    }
    (ROOT / "server-vectors.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
