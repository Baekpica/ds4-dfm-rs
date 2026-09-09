#!/usr/bin/env python3
"""Independent HTTP prompt fixtures from the pinned source Jinja template."""
import hashlib
import json
import sys
from pathlib import Path

from jinja2.sandbox import ImmutableSandboxedEnvironment

template_path, output_path = map(Path, sys.argv[1:])
source = template_path.read_bytes()
env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)


def fail(message):
    raise ValueError(message)


env.globals["raise_exception"] = fail
env.filters["tojson"] = lambda value, **kw: json.dumps(value, ensure_ascii=False, **kw)
template = env.from_string(source.decode())
tools = [{"type": "function", "function": {
    "name": "weather", "description": "서울 날씨", "parameters": {
        "type": "object", "properties": {"city": {"type": "string"}},
        "required": ["city"],
    },
}}]
cases = [
    ("no_system", [{"role": "user", "content": "Hi"}], []),
    ("systems", [{"role": "system", "content": "A"},
                 {"role": "system", "content": " B\n"},
                 {"role": "user", "content": " Hi "}], []),
    ("empty", [], []),
    ("history", [{"role": "user", "content": "2 + 2?"},
                 {"role": "assistant", "reasoning_content": "Check.", "content": "4"},
                 {"role": "user", "content": "Continue."}], []),
    ("tool_result", [{"role": "system", "content": "Be concise."},
                     {"role": "user", "content": "Weather?"},
                     {"role": "assistant", "content": None, "tool_calls": [
                         {"id": "call_1", "function": {"name": "weather", "arguments": {
                             "z": [1e-7, 1e20, 1.0, -0.0], "city": "서울",
                             "nested": {"z": False, "a": None},
                         }}},
                     ]},
                     {"role": "tool", "tool_call_id": "call_1", "content": "Sunny"},
                     {"role": "user", "content": "Thanks"}], tools),
    ("named_result", [{"role": "tool", "name": "weather", "content": "Sunny"}], tools),
    ("parts", [{"role": "user", "content": [
        {"type": "text", "text": "Before"}, {"type": "image"},
        {"type": "text", "text": "After"},
    ]}], []),
]
vectors = []
for name, messages, functions in cases:
    for effort in ["none", "low", "high", "max"]:
        vectors.append({"name": name, "messages": messages, "tools": functions,
                        "effort": effort, "rendered": template.render(
                            messages=messages, tools=functions, reasoning_effort=effort,
                            add_generation_prompt=True,
                        )})
output_path.write_text(json.dumps({
    "template_sha256": hashlib.sha256(source).hexdigest(), "vectors": vectors,
}, ensure_ascii=False, indent=2) + "\n")
