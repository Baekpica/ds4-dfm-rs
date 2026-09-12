#!/usr/bin/env python3
"""Pinned official Step template, independently rendered with Python Jinja."""
import hashlib
import json
from pathlib import Path
import jinja2
from jinja2.sandbox import ImmutableSandboxedEnvironment

root = Path(__file__).parent
source = (root / 'chat_template.jinja').read_bytes()
env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)
env.filters['tojson'] = lambda value, **kw: json.dumps(value, **kw)
env.filters['fromjson'] = json.loads
template = env.from_string(source.decode())
tools = [{'type': 'function', 'function': {'name': 'weather', 'description': '서울 날씨',
    'parameters': {'type': 'object', 'properties': {'city': {'type': 'string'}}, 'required': ['city']}}}]
cases = [
    ('plain', [{'role': 'user', 'content': 'Hello'}], []),
    ('history', [{'role': 'user', 'content': 'Hi'},
        {'role': 'assistant', 'content': 'Hello', 'reasoning_content': 'Think'},
        {'role': 'user', 'content': 'Continue'}], []),
    ('image', [{'role': 'user', 'content': [{'type': 'text', 'text': 'Read'},
        {'type': 'image'}, {'type': 'text', 'text': 'this'}, {'type': 'image'}]}], []),
    ('tool_return', [{'role': 'system', 'content': 'Be concise'},
        {'role': 'user', 'content': 'Weather?'},
        {'role': 'assistant', 'content': None, 'reasoning_content': 'Check', 'tool_calls': [
            {'function': {'name': 'weather', 'arguments': '{"city":"서울","days":[1,2]}'}}]},
        {'role': 'tool', 'content': '맑음'}, {'role': 'tool', 'content': '20 °C'}], tools),
    ('observation', [{'role': 'user', 'content': 'Read'},
        {'role': 'system', 'name': 'observation', 'content': {'value': 'Found'}}], tools),
]
vectors = []
for name, messages, functions in cases:
    for effort in ['none', 'low', 'high', 'max']:
        context = dict(bos_token='<｜begin▁of▁sentence｜>', messages=messages, tools=functions,
                       reasoning_effort=effort, add_generation_prompt=True)
        vectors.append(dict(name=name, context=context, rendered=template.render(**context)))
(root / 'chat-vectors.json').write_text(json.dumps(dict(
    source_revision='5f6244077ac62e04eec3f320501ff8c2b293373a',
    template_sha256=hashlib.sha256(source).hexdigest(), jinja2_version=jinja2.__version__,
    vectors=vectors), ensure_ascii=False, indent=2) + '\n')
