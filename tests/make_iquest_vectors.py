#!/usr/bin/env python3
"""Regenerate a weight-free GGUF and offline HF tokenizer/Jinja oracle vectors."""
import argparse
import hashlib
import json
from pathlib import Path

SOURCE_REPO = 'IQuestLab/IQuest-Q1'
SOURCE_REVISION = '5c21b0630586ef77d38cff1094b8cf37a417fcd8'
SOURCE_HASHES = {
    'tokenizer.json': 'e79745512446814ada2d414200d233912d854e56c60e1ffcb12e76463b08b847',
    'chat_template.jinja': 'b3c99aab1b3f17a2513d134ac8da39303da965f73924e35057f0cd6f10eaa06d',
}


def main():
    p = argparse.ArgumentParser(description=__doc__, epilog=f'Source: {SOURCE_REPO}@{SOURCE_REVISION}; requires gguf, tokenizers and transformers.')
    p.add_argument('--model', type=Path, required=True, help='directory containing pinned tokenizer.json and chat_template.jinja')
    p.add_argument('--gguf', type=Path, required=True)
    p.add_argument('--vectors', type=Path, required=True)
    args = p.parse_args()
    for name, expected in SOURCE_HASHES.items():
        if hashlib.sha256((args.model / name).read_bytes()).hexdigest() != expected:
            p.error(f'{name} does not match {SOURCE_REPO}@{SOURCE_REVISION}')
    import gguf
    from tokenizers import Tokenizer
    from transformers import PreTrainedTokenizerFast
    from transformers.utils.chat_template_utils import _compile_jinja_template

    raw = json.loads((args.model / 'tokenizer.json').read_text())
    tokens = [''] * 160000
    types = [gguf.TokenType.NORMAL] * len(tokens)
    for word, idx in raw['model']['vocab'].items():
        tokens[idx] = word
    for item in raw['added_tokens']:
        tokens[item['id']] = item['content']
        types[item['id']] = gguf.TokenType.CONTROL if item['special'] else gguf.TokenType.USER_DEFINED
    args.gguf.parent.mkdir(parents=True, exist_ok=True)
    writer = gguf.GGUFWriter(args.gguf, 'iquest_q1')
    writer.add_tokenizer_model('gpt2')
    writer.add_tokenizer_pre('iquest-q1')
    writer.add_token_list(tokens)
    writer.add_token_types(types)
    writer.add_token_merges([' '.join(x) for x in raw['model']['merges']])
    writer.add_eos_token_id(0)
    writer.add_add_bos_token(False)
    writer.add_add_eos_token(False)
    writer.add_string('iquest_q1.tokenizer.json', (args.model / 'tokenizer.json').read_text())
    template = (args.model / 'chat_template.jinja').read_text()
    writer.add_chat_template(template)
    writer.write_header_to_file(); writer.write_kv_data_to_file(); writer.close()
    tok = Tokenizer.from_file(str(args.model / 'tokenizer.json'))
    fast = PreTrainedTokenizerFast(tokenizer_file=str(args.model / 'tokenizer.json'), chat_template=template)
    # The released IQuest template calls fromjson for string tool arguments.
    # Match the serving image's JSON parser without changing the source grammar.
    _compile_jinja_template(template).environment.filters['fromjson'] = json.loads
    texts = [
        ('nfc_composed', 'café é'), ('nfc_decomposed', 'cafe\u0301 e\u0301'),
        ('korean', '안녕하세요. 코드를 검토해주세요.'), ('korean_jamo', '\u1100\u1161 \u1102\u1161'),
        ('cjk_ascii', 'ABC中文xyzかなカナ1234567890'), ('numbers', '1 12 123 1234 12345 123456 １２３４５６'),
        ('code', 'def f(x):\n\treturn x + 1 # café\n'), ('spaces', '   abc  \n\t abc\r\n'),
        ('symbols', '!Hello .world _snake +plus 中文，かな。'), ('combining', 'a\u0308\u0323 z\u0301 é'),
        ('emoji', '🙂🚀 👨\u200d👩\u200d👧\u200d👦'), ('empty', ''),
    ]
    vectors = [{'name': name, 'mode': 'text', 'text': text, 'ids': tok.encode(text, add_special_tokens=False).ids}
               for name, text in texts]
    for text in ['<|iquest_end|>', '<think>plan</think>', '<iquest_tool_call>run</iquest_tool_call>',
                 '<|im_start|>assistant\né<|im_end|>', '<tool_response>result</tool_response>']:
        for mode in ('text', 'rendered'):
            vectors.append({'name': text, 'mode': mode, 'text': text,
                            'ids': tok.encode(text, add_special_tokens=False).ids})
    for item in raw['added_tokens']:
        vectors.append({'name': 'source_added_' + str(item['id']), 'mode': 'text', 'text': item['content'],
                        'ids': tok.encode(item['content'], add_special_tokens=False).ids})
    chats = []
    tool = {'type': 'function', 'function': {'name': 'weather', 'description': 'Get weather',
            'parameters': {'type': 'object', 'properties': {'city': {'type': 'string', 'description': 'City'}}, 'required': ['city']}}}
    for name, messages, tools in [
        ('user', [{'role': 'user', 'content': 'Hello café 中文'}], []),
        ('system_followup', [{'role': 'system', 'content': 'Use Korean.'}, {'role': 'user', 'content': '안녕'},
                            {'role': 'assistant', 'content': '안녕하세요.'}, {'role': 'user', 'content': '다음 단계'}], []),
        ('tool_schema', [{'role': 'user', 'content': 'Weather in 서울?'}], [tool]),
        ('tool_result', [{'role': 'user', 'content': 'Weather in 서울?'},
                         {'role': 'assistant', 'content': '', 'tool_calls': [{'id': 'call_1', 'type': 'function',
                           'function': {'name': 'weather', 'arguments': '{"city":"서울"}'}}]},
                         {'role': 'tool', 'name': 'weather', 'tool_call_id': 'call_1', 'content': 'Sunny'}], [tool]),
    ]:
        context = {'messages': messages, 'tools': tools, 'add_generation_prompt': True}
        rendered = fast.apply_chat_template(messages, tools=tools, add_generation_prompt=True, tokenize=False)
        chats.append({'name': name, 'context': context, 'rendered': rendered,
                      'ids': tok.encode(rendered, add_special_tokens=False).ids})
    report = {'source_repository': SOURCE_REPO, 'source_revision': SOURCE_REVISION,
              'source_tokenizer_sha256': hashlib.sha256((args.model / 'tokenizer.json').read_bytes()).hexdigest(),
              'source_template_sha256': hashlib.sha256(template.encode()).hexdigest(),
              'normalizer': raw['normalizer'], 'vectors': vectors, 'chats': chats,
              'source_template': template,
              'python_filter_extension': 'fromjson=json.loads for released IQuest string tool arguments'}
    args.vectors.parent.mkdir(parents=True, exist_ok=True)
    args.vectors.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps({'gguf': str(args.gguf), 'vectors': str(args.vectors), 'text_cases': len(vectors), 'chat_cases': len(chats)}))


if __name__ == '__main__':
    main()
