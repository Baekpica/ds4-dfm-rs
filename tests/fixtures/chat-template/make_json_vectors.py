#!/usr/bin/env python3
"""Freeze Python JSON bytes; never derive expected output from Rust."""

import json
import platform
from pathlib import Path

import jinja2
from jinja2.sandbox import ImmutableSandboxedEnvironment


def python_json(value, **kwargs):
    # Transformers uses JSON text, without Jinja's HTML escaping or ASCII default.
    return json.dumps(value, **{"ensure_ascii": False, **kwargs})


def main():
    nested = {"z": [True, None, -7], "a": {"y": {}, "b": []}}
    unicode_value = {"한글": "é / 😀", "lines": "\u2028\u2029\x7f"}
    cases = [
        ("nested_default", nested, "{{ value | tojson }}"),
        (
            "source_map_order",
            None,
            "{{ {'z': 1, 'a': 2, 'm': {'y': 3, 'b': 4}} | tojson }}",
        ),
        ("sort_keys_recursive", nested, "{{ value | tojson(sort_keys=true) }}"),
        ("preserve_input_order", nested, "{{ value | tojson(sort_keys=false) }}"),
        (
            "compact_separators",
            nested,
            "{{ value | tojson(separators=(',', ':')) }}",
        ),
        (
            "custom_separators",
            nested,
            "{{ value | tojson(separators=(' | ', ' => ')) }}",
        ),
        ("unicode_default", unicode_value, "{{ value | tojson }}"),
        (
            "ascii_escape",
            unicode_value,
            "{{ value | tojson(ensure_ascii=true, sort_keys=true) }}",
        ),
        (
            "string_escape",
            "\"\\/\b\f\n\r\t\x00\x01\x1f <tag>&'",
            "{{ value | tojson }}",
        ),
        ("indent_two", nested, "{{ value | tojson(indent=2) }}"),
        ("indent_tab", nested, "{{ value | tojson(indent='\t') }}"),
        ("indent_zero", nested, "{{ value | tojson(indent=0) }}"),
        ("indent_negative", nested, "{{ value | tojson(indent=-2) }}"),
        (
            "indent_compact",
            nested,
            "{{ value | tojson(indent=2, separators=(',', ':'), sort_keys=true) }}",
        ),
        ("empty_containers", [[], {}, [[], {}]], "{{ value | tojson(indent=2) }}"),
        ("integral_floats", [1.0, -2.0, 1000.0, 1, -2], "{{ value | tojson }}"),
        ("signed_zero", [0.0, -0.0, 0], "{{ value | tojson }}"),
        (
            "small_exponents",
            [1e-5, 1e-6, 1e-7, -1e-5],
            "{{ value | tojson }}",
        ),
        (
            "lower_threshold",
            [0.0001, 0.00009999999999999999, 0.00010000000000000002],
            "{{ value | tojson }}",
        ),
        (
            "upper_threshold",
            [9999999999999998.0, 1e16, 1.0000000000000002e16, -1e16],
            "{{ value | tojson }}",
        ),
        ("large_exponents", [1e20, 1e100, -1e100], "{{ value | tojson }}"),
        ("unknown_keyword", nested, "{{ value | tojson(unknown_option=true) }}"),
        ("invalid_indent", nested, "{{ value | tojson(indent=1.5) }}"),
        ("short_separators", nested, "{{ value | tojson(separators=[',']) }}"),
        ("invalid_separator", nested, "{{ value | tojson(separators=(',', 1)) }}"),
    ]
    env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)
    env.filters["tojson"] = python_json

    vectors = []
    for name, value, source in cases:
        row = {"name": name, "template": source, "context": {"value": value}}
        try:
            row["expected"] = env.from_string(source).render(value=value)
        except (TypeError, ValueError) as error:
            row["error_type"] = type(error).__name__
            row["error_message"] = str(error)
        vectors.append(row)

    output = {
        "oracle": "Python json.dumps; ensure_ascii=False unless specified",
        "python_version": platform.python_version(),
        "jinja2_version": jinja2.__version__,
        "vectors": vectors,
    }
    path = Path(__file__).with_name("json-vectors.json")
    path.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
