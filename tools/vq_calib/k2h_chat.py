#!/usr/bin/env python3
"""K2-Horizon chat rendering for reference prompts: the checkpoint's chat_template.jinja through jinja2 with
the pieces Transformers' apply_chat_template adds (sandboxed environment, trim/lstrip blocks, loop controls,
`raise_exception`, `strftime_now`, `tojson`, and the `generation` block tag as a pass-through)."""
import datetime
import json

from jinja2 import nodes
from jinja2.ext import Extension
from jinja2.sandbox import ImmutableSandboxedEnvironment

BOS, EOS = '<|ifm|begin_of_text|>', '<|ifm|endoftext|>'


class Generation(Extension):
    tags = {'generation'}

    def parse(self, parser):
        lineno = next(parser.stream).lineno
        body = parser.parse_statements(['name:endgeneration'], drop_needle=True)
        return nodes.CallBlock(self.call_method('_pass'), [], [], body).set_lineno(lineno)

    def _pass(self, caller):
        return caller()


def _raise(message):
    raise ValueError(message)


def render(template_path, messages, date=datetime.datetime(2026, 11, 24), **kw):
    env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True, extensions=[Generation, 'jinja2.ext.loopcontrols'])
    env.globals['raise_exception'] = _raise
    env.globals['strftime_now'] = lambda f: date.strftime(f)
    env.filters['tojson'] = lambda x, ensure_ascii=False, indent=None, separators=None, sort_keys=False: json.dumps(
        x, ensure_ascii=ensure_ascii, indent=indent, separators=separators, sort_keys=sort_keys)
    template = env.from_string(open(template_path, encoding='utf-8').read())
    return template.render(messages=messages, bos_token=BOS, eos_token=EOS, **kw)
