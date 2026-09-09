#!/usr/bin/env python3
"""Render pinned, unmodified model templates with an independent Python oracle."""

import copy
import hashlib
import json
import platform
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path

import jinja2
from jinja2 import nodes
from jinja2.ext import Extension, loopcontrols
from jinja2.sandbox import ImmutableSandboxedEnvironment


ROOT = Path(__file__).parent
FAMILIES = ("solar", "exaone", "motif", "dots", "qwen", "qwen-uncensored", "glm", "k2", "inkling")
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


class Thinking(Enum):
    OFF = "off"
    ON = "on"


class Generation(Extension):
    # HF's generation blocks alter token-mask tracking, not rendered text.
    tags = {"generation"}

    def parse(self, parser):
        line = next(parser.stream).lineno
        body = parser.parse_statements(["name:endgeneration"], drop_needle=True)
        return nodes.CallBlock(self.call_method("emit"), [], [], body).set_lineno(line)

    def emit(self, caller):
        return caller()


def python_json(value, **kwargs):
    return json.dumps(value, **{"ensure_ascii": False, **kwargs})


def fail(message):
    raise jinja2.TemplateError(message)


def render_options(family, mode):
    if family in ("motif", "exaone", "dots"):
        return {"enable_thinking": mode is Thinking.ON}
    if family in ("qwen", "qwen-uncensored"):
        return {
            "enable_thinking": mode is Thinking.ON,
            "reasoning_effort": "xhigh" if mode is Thinking.ON else "none",
        }
    return {"reasoning_effort": "high" if mode is Thinking.ON else "none"}


def model_cases(family):
    enabled = render_options(family, Thinking.ON)

    def case(name, messages, options=None, tools=None, reject=None):
        context = {
            **enabled,
            "add_generation_prompt": True,
            "messages": copy.deepcopy(messages),
            "tools": copy.deepcopy(tools or []),
            **(options or {}),
        }
        return {"name": name, "context": context, "reject": reject}

    off_name = "user_none_effort" if family in ("glm", "k2") else "user_thinking_off"
    # K2 rejects none; GLM maps it to Max and still opens a thinking block.
    off_reject = "Unsupported reasoning_effort" if family == "k2" else None
    cases = [
        case(off_name, [USER], render_options(family, Thinking.OFF), reject=off_reject),
        case("user_thinking_on", [USER]),
        case("system_multiturn", MULTITURN),
        case("tool_schema", [USER], tools=[TOOL]),
        case("tool_result", TOOL_HISTORY, tools=[TOOL]),
        case("no_generation", [USER, ASSISTANT], {"add_generation_prompt": False}),
    ]

    if family in ("qwen", "qwen-uncensored"):
        cases.extend([
            case("preserve_history", MULTITURN, {"preserve_thinking": True}),
            case("drop_history_thinking", MULTITURN, {"preserve_thinking": False}),
            case("reject_high_effort", [USER], {"reasoning_effort": "high"}, reject="Unexpected reasoning effort"),
        ])
        images = [{"role": "user", "content": [
            {"type": "text", "text": "두 이미지를 비교해 줘."},
            {"type": "image"},
            {"type": "image_url", "image_url": {"url": "fixture://second-image"}},
        ]}]
        cases.append(case("image_placeholders", images, {"add_vision_id": True}))

    if family in ("dots", "glm"):
        cases.extend([
            case("keep_history_thinking", MULTITURN, {"clear_thinking": False}),
            case("clear_history_thinking", MULTITURN, {"clear_thinking": True}),
        ])

    if family in ("solar", "glm"):
        pair = copy.deepcopy(TOOL_HISTORY)
        second = copy.deepcopy(pair[1]["tool_calls"][0])
        second["id"] = "call_02"
        second["function"]["arguments"]["location"]["city"] = "Zürich"
        pair[1]["tool_calls"].append(second)
        pair.insert(2, {"role": "tool", "tool_call_id": "call_02", "content": "Zürich: café"})
        cases.append(case("reordered_tool_results", pair, tools=[TOOL]))

    if family == "solar":
        cases.extend([
            case("preserved_thinking", MULTITURN, {"think_render_option": "preserved"}),
            case("provider_fixed_clock", [USER], {"provider_system_prompt": True}),
        ])

    if family == "exaone":
        cases.append(case("skip_tool_thinking", TOOL_HISTORY, {"skip_think": True}, [TOOL]))

    if family == "glm":
        # This published template emits inability reminders for media parts.
        media = [{"role": "user", "content": [
            {"type": "text", "text": "입력을 설명해 줘."},
            {"type": "image"},
            {"type": "input_audio"},
        ]}]
        cases.append(case("media_reminders", media))

    if family == "k2":
        cases.extend([
            case("user_low_effort", [USER], {"reasoning_effort": "low"}),
            case("missing_thinking_field", [USER, {"role": "assistant", "content": "Reply."}], reject="missing a thinking field"),
        ])

    if family == "inkling":
        media = [{"role": "user", "content": [
            {"type": "input_image"},
            {"type": "input_audio"},
            {"type": "text", "text": "색과 내용을 알려 줘."},
        ]}]
        cases.append(case("image_audio_placeholders", media))
        bad_call = copy.deepcopy(TOOL_HISTORY)
        bad_call[1]["tool_calls"][0]["function"]["arguments"] = json.dumps(ARGUMENTS, ensure_ascii=False)
        cases.append(case("reject_string_arguments", bad_call, tools=[TOOL], reject="arguments must be a parsed object"))

    return cases


def main():
    env = ImmutableSandboxedEnvironment(
        trim_blocks=True, lstrip_blocks=True, extensions=[loopcontrols, Generation]
    )
    env.filters["tojson"] = python_json
    env.globals["raise_exception"] = fail
    env.globals["strftime_now"] = CLOCK.strftime

    models = []
    for family in FAMILIES:
        folder = ROOT / "models" / family
        provenance = json.loads((folder / "provenance.json").read_text(encoding="utf-8"))
        source = (folder / "chat_template.jinja").read_bytes()
        digest = hashlib.sha256(source).hexdigest()
        assert digest == provenance["template_sha256"], family
        template = env.from_string(source.decode("utf-8"))
        vectors = []
        for row in model_cases(family):
            rejection = row.pop("reject")
            row["context"] = {**provenance["special_tokens"], **row["context"]}
            try:
                row["expected"] = template.render(**row["context"])
            except (jinja2.TemplateError, TypeError, ValueError) as error:
                if rejection is None or rejection not in str(error):
                    raise RuntimeError(f"{family}/{row['name']}: {error}") from error
                row["error_type"] = type(error).__name__
                row["error_message"] = str(error)
            else:
                assert rejection is None, f"{family}/{row['name']}: expected rejection"
            vectors.append(row)
        models.append({"name": family, "template_sha256": digest, "vectors": vectors})
        print(f"{family}: {len(vectors)} cases")

    result = {
        "oracle": "Python Jinja2 with Transformers rendering options and Python json.dumps",
        "python_version": platform.python_version(),
        "jinja2_version": jinja2.__version__,
        "clock_unix_seconds": 0,
        "scope": "Template bytes only; media placeholders/reminders do not qualify model media support.",
        "models": models,
    }
    (ROOT / "model-vectors.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
