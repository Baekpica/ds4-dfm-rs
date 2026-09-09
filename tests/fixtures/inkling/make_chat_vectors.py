#!/usr/bin/env python3
"""Render the published template with the Transformers Jinja environment."""
import argparse
import hashlib
import json
from pathlib import Path

import jinja2
from jinja2.sandbox import ImmutableSandboxedEnvironment


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("template", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    data = args.template.read_bytes()
    env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)

    def fail(message):
        raise ValueError(message)

    env.globals["raise_exception"] = fail
    env.filters["tojson"] = lambda value, **kw: json.dumps(value, ensure_ascii=False, **kw)
    template = env.from_string(data.decode())
    messages = [
        dict(role="system", content="Be concise."),
        dict(role="user", content="안녕하세요"),
        dict(role="assistant", content="Hello."),
        dict(role="tool", content="result"),
        dict(role="user", content="Continue."),
    ]
    rows = [dict(effort=effort, messages=messages, rendered=template.render(
        messages=messages, tools=[], reasoning_effort=effort, add_generation_prompt=True,
    )) for effort in ["none", "low", "high", "max"]]
    args.output.write_text(json.dumps({
        "template_sha256": hashlib.sha256(data).hexdigest(),
        "jinja2_version": jinja2.__version__, "vectors": rows,
    }, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
