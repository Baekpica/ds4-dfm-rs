#!/usr/bin/env python3
"""Render the artifact's embedded GLM grammar with the independent Jinja oracle."""
import hashlib
import json
import platform
from pathlib import Path

import jinja2
from jinja2.sandbox import ImmutableSandboxedEnvironment

from make_model_vectors import CLOCK, Generation, fail, loopcontrols, model_cases, python_json


ROOT = Path(__file__).parent
MODEL = ROOT / "models" / "glm-uncensored"


def main():
    source = (MODEL / "chat_template.jinja").read_bytes()
    provenance = json.loads((MODEL / "provenance.json").read_text())
    assert hashlib.sha256(source).hexdigest() == provenance["template_sha256"]
    env = ImmutableSandboxedEnvironment(
        trim_blocks=True, lstrip_blocks=True, extensions=[loopcontrols, Generation]
    )
    env.filters["tojson"] = python_json
    env.globals.update(raise_exception=fail, strftime_now=CLOCK.strftime)
    template = env.from_string(source.decode())
    cases = model_cases("glm")
    for mode in ("low", "high", "max", "none"):
        cases.append({"name": f"embedded_history_{mode}", "adapter": mode, "context": {
            "messages": [
                {"role": "user", "content": "Compute 2 + 2."},
                {"role": "assistant", "content": "<think>Two pairs make four.</think>4"},
                {"role": "user", "content": "Now add 1."},
            ],
            "tools": [], "add_generation_prompt": True, "reasoning_effort": mode,
        }})
    cases.append({"name": "ordered_images", "adapter": "high", "context": {
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": "Compare the images."}, {"type": "image"},
            {"type": "image_url", "image_url": {"url": "fixture://second"}},
        ]}], "tools": [], "add_generation_prompt": True, "reasoning_effort": "high",
    }})
    for row in cases:
        assert row.pop("reject", None) is None
        row["expected"] = template.render(**row["context"])
        if "adapter" in row:
            suffix = "</think>" if row["adapter"] == "none" else ""
            row["adapter_expected"] = row["expected"] + suffix
    result = {
        "oracle": "Python Jinja2 and Transformers-compatible Python json.dumps",
        "python_version": platform.python_version(), "jinja2_version": jinja2.__version__,
        "template_sha256": provenance["template_sha256"], "vectors": cases,
    }
    (ROOT / "glm53-vectors.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(f"GLM Uncensored: {len(cases)} independent renders")


if __name__ == "__main__":
    main()
