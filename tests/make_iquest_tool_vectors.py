#!/usr/bin/env python3
"""Regenerate offline vectors from an explicitly supplied pinned official parser."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace


SOURCE_URL = 'https://github.com/IQuestLab/vllm-iquest-q1/blob/1f305c04f6d63b936e118e5aa4a9eb6b20e38a32/src/vllm_iquest_q1/tool_parser.py'
SOURCE_SHA256 = '4b6de273cf973808a98879ff258a98a68c2cc1415e491ba653aaf03d5c3789d0'


def main():
    parser_args = argparse.ArgumentParser(description=__doc__, epilog='Source: ' + SOURCE_URL)
    parser_args.add_argument('--source', type=Path, required=True)
    parser_args.add_argument('--out', type=Path, required=True)
    args = parser_args.parse_args()
    raw = args.source.read_bytes()
    if hashlib.sha256(raw).hexdigest() != SOURCE_SHA256:
        parser_args.error('source does not match the pinned official parser SHA256')
    source = ast.parse(raw)
    cls = next(x for x in source.body if isinstance(x, ast.ClassDef) and x.name == 'IQuestQ1ToolParser')
    wanted = {'_tools_enabled', '_decode_value', '_parameter_is_string',
              '_parse_call_body', 'extract_tool_calls', '_partial_start_length',
              'extract_tool_calls_streaming'}
    cls.body = [x for x in cls.body if isinstance(x, ast.Assign)
                or isinstance(x, ast.FunctionDef) and x.name in wanted]
    cls.bases = []
    module = ast.Module(body=[ast.ImportFrom(module='__future__',
                         names=[ast.alias(name='annotations')], level=0), cls], type_ignores=[])
    scope = {'json':json, 'FunctionCall':SimpleNamespace,
             'ToolCall':SimpleNamespace, 'ExtractedToolCallInformation':SimpleNamespace,
             'DeltaMessage':SimpleNamespace, 'DeltaToolCall':SimpleNamespace,
             'DeltaFunctionCall':SimpleNamespace, 'make_tool_call_id':lambda:'oracle-id'}
    exec(compile(ast.fix_missing_locations(module), str(args.source), 'exec'), scope)
    parser = scope[cls.name]()
    parameters = {'type':'object','properties':{'text':{'type':'string'},
                  'count':{'type':'integer'}, 'active':{'type':'boolean'}, 'data':{'type':'object'}}}
    request = SimpleNamespace(tools=[SimpleNamespace(function=SimpleNamespace(name='lookup',parameters=parameters))],
                              tool_choice='auto')
    def call(name='lookup', **args):
        body = name + ''.join(f'<arg_key>{key}</arg_key><arg_value>{value}</arg_value>' for key,value in args.items())
        return '<iquest_tool_call>'+body+'</iquest_tool_call>'
    cases = [
        ('typed',call(text=' 123 \n',count='2',active='true',data='{"city":"서울","items":[1,null]}')),
        ('literal-strings',call(text='true &amp; <tag> "quoted" café')),
        ('string-json',call(text='{"raw":true}')),
        ('unknown-properties',call(value=' null ',list='[1,"漢字",false]',bad=' 0123 ')),
        ('float',call(count='1e-05',active='false')),
        ('arbitrary-integer',call(count='18446744073709551617',data='{"negative":-18446744073709551617}')),
        ('no-arguments',call(' \u2003ping\n')),
        ('empty-string',call(text='')),
        ('duplicate-key','<iquest_tool_call>lookup<arg_key>count</arg_key><arg_value>1</arg_value><arg_key>count</arg_key><arg_value>2</arg_value></iquest_tool_call>'),
        ('parallel-content','前'+call(text='123')+'中間'+call('ping')+'後'),
        ('malformed','<iquest_tool_call>lookup<arg_key>text</arg_key><arg_value>broken</iquest_tool_call>'),
        ('malformed-then-valid','<iquest_tool_call>lookup<arg_key>text</arg_key><arg_value>broken</iquest_tool_call>'+call('ping')),
        ('empty-key','<iquest_tool_call>lookup<arg_key> </arg_key><arg_value>1</arg_value></iquest_tool_call>'),
        ('truncated',call('ping')+'after<iquest_tool_call>lookup<arg_key>text'),
        ('unicode-space','<iquest_tool_call>\u2003lookup\n<arg_key>\u2003text </arg_key>\u2003<arg_value> raw </arg_value>\u2003</iquest_tool_call>'),
        ('python-control-space','<iquest_tool_call>\u001clookup\u001d<arg_key>\u001etext\u001f</arg_key>\u001c<arg_value> raw </arg_value>\u001d</iquest_tool_call>'),
        ('disabled',call(text='123')),
    ]
    vectors = []
    for name,text in cases:
        request.tool_choice = 'none' if name=='disabled' else 'auto'
        result = parser.extract_tool_calls(text,request)
        streaming = scope[cls.name]()
        streaming._stream_buffer = ''
        streaming._inside_tool_call = False
        streaming._next_tool_index = 0
        streaming.prev_tool_call_arr = []
        streaming.streamed_args_for_tool = []
        stream_content, stream_calls, previous = [], [], ''
        for char in text:
            current = previous + char
            delta = streaming.extract_tool_calls_streaming(previous,current,char,[],[],[],request)
            if delta:
                stream_content.append(getattr(delta,'content',None) or '')
                stream_calls.extend({'index':x.index, 'name':x.function.name,
                    'arguments':json.loads(x.function.arguments)}
                    for x in getattr(delta,'tool_calls',None) or [])
            previous = current
        vectors.append({'name':name,'text':text,'tools_enabled':name!='disabled',
                        'content':result.content or '',
                        'calls':[{'name':x.function.name,'arguments':json.loads(x.function.arguments)} for x in result.tool_calls],
                        'stream_content_before_final':''.join(stream_content),
                        'stream_calls_before_final':stream_calls})
    out = args.out
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps({'source_url':SOURCE_URL,
                              'source_sha256':SOURCE_SHA256,
                              'scope':'Extracted official buffered and streaming parser methods, complete Unicode input streamed one codepoint at a time; no model inference. Native stream finalization additionally preserves unfinished XML as raw text.',
                              'parameters':parameters,'cases':vectors},ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'cases':len(vectors),'out':str(out),'sha256':hashlib.sha256(out.read_bytes()).hexdigest()}))


if __name__=='__main__':
    main()
