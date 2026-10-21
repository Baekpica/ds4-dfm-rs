#!/usr/bin/env python3
"""DeepSeek-V4.1-Flash-1017 calibration and held-out token sets for the VQ imatrix collector.

  fetch  stream bounded samples of every source into calib/raw/<source>/ (data only)
  build  render chat records natively with the V4.1 encoding (encoding/encoding.py:
         thinking / chat mode, earlier-turn reasoning dropped by the encoding, DSML
         tool calls), tokenize with the V4.1 tokenizer, reserve the held-out set first,
         budget by domain and length, drop calibration rows that share a 64-token
         window with a held-out case, and write calib(.safetensors|_manifest.json) and
         heldout(.safetensors|_manifest.json): `input_ids` int32 [N] + `offsets` int64
         [n+1] and a per-row JSON manifest
  check  id range, BOS, decoded samples per domain (calib/samples_decoded.txt)

Port of build_calib.py (A.X-K2, seed 20261021) to DeepSeek-V4.1: same structure
(seeded, pinned revisions, bounded sampling of big JSONL / parquet by byte offsets /
row groups, render -> tokenize -> budget -> filter -> write, held-out reserved first
+ 64-token-window overlap check). Every row starts with BOS (id 0): chat rows because
encode_messages() starts every conversation with <｜begin▁of▁sentence｜>, raw-text rows
because the builder prepends it the same way. Downloaded bytes are parsed as JSON /
parquet data only.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import random
import re
import time

import numpy as np

ROOT = Path('/work/ivq/calib')
RAW = ROOT / 'raw'
MODEL = Path('/models/deepseek-ai/DeepSeek-V4.1-Flash-1017')
MODEL_REPO = 'deepseek-ai/DeepSeek-V4.1-Flash-1017'
MODEL_REV = '1bae0e1088993b9b82de08f59de5be5a34e42170'
TOKENIZER = Path('/work/ivq/tok/tokenizer.json')
DS4_PROMPTS = Path('/ds4-dfm-rs/gguf-tools/imatrix/dataset/prompts.jsonl')
DS4_REV = '32a4168332615a4d3c9b01b579623197caf7b6c8'   # local clone HEAD of Baekpica/ds4-dfm-rs
SEED = 20261022
MAX_LEN = 32768
VOCAB = 129280
BOS_ID, EOS_ID = 0, 1

REV = {
    'HuggingFaceFW/fineweb-edu': '87f09149ef4734204d70ed1d046ddc9ca3f2b8f9',
    'HuggingFaceFW/fineweb-2': 'af9c13333eb981300149d5ca60a8e9d659b276b9',
    'HuggingFaceTB/finemath': 'e92b25a616738fe95dc186b64dfb19f9c8525594',
    'codeparrot/github-code-clean': 'c48d40f9e70f0196f8236901ee35807f7d6c44c0',
    'nvidia/Nemotron-SFT-Instruction-Following-Chat-v2': '1a9454ed054b8544503ab8d8c0a519d141a44c5b',
    'nvidia/Nemotron-Cascade-SFT-Stage-1': 'e59a356b107737562f064ec252e6adacbc737899',
    'nvidia/Nemotron-Cascade-SFT-Stage-2': '54cfa23ba28c9d94aeae15142e73bde6f8d14d8b',
    'nvidia/Nemotron-SFT-Multilingual-v2': '971a252224b75414b1b67c55dbe0446d8b6606a0',
    'nvidia/Nemotron-SFT-Multilingual-v1': '22c86505762a7c595abee309d720084351c9f4ba',
    'nvidia/Nemotron-SpecializedDomains-Finance-v1': '5a21b106168facb96ced11b883c2a9b4788ee939',
    'nvidia/Nemotron-SFT-SWE-v2': 'bd151f3f2d89c4804dda0083d912bd9f6a0a9fb7',
    'nvidia/Nemotron-SFT-Competitive-Programming-v2': '778afc98a9e027e10b3cd78020c120e93e142ef2',
    'nvidia/Nemotron-SFT-Agentic-v2': '7c804833427f633ccd53b582dbf02525fd680f78',
    'Congliu/Chinese-DeepSeek-R1-Distill-data-110k': '8520b649430617c2be4490f424d251d09d835ed3',
    'm-a-p/COIG-CQIA': '8b55868c6168adf86c30e7ca0f782cca1c514297',
    'Salesforce/wikitext': 'b08601e04326c79dfdd32d625aee71d232d685c3',
}
IF2 = 'nvidia/Nemotron-SFT-Instruction-Following-Chat-v2'
C1, C2 = 'nvidia/Nemotron-Cascade-SFT-Stage-1', 'nvidia/Nemotron-Cascade-SFT-Stage-2'
ML1, ML2 = 'nvidia/Nemotron-SFT-Multilingual-v1', 'nvidia/Nemotron-SFT-Multilingual-v2'
FW2 = 'HuggingFaceFW/fineweb-2'

# JSONL sources: name -> (repo, [paths], segments per path, lines per segment)
JSONL = {
    'if_on': (IF2, ['data/reasoning_on.jsonl'], 24, 12),
    'if_off': (IF2, ['data/reasoning_off.jsonl'], 24, 12),
    'casc1_math': (C1, ['math/math_1.jsonl', 'math/math_4.jsonl'], 12, 8),
    'casc1_science': (C1, ['science.jsonl'], 16, 8),
    'casc1_code': (C1, ['code/code_1.jsonl'], 16, 6),
    'casc2_general': (C2, ['general/general_1.jsonl', 'general/general_3.jsonl'], 12, 8),
    'casc2_if': (C2, ['instruction-following.jsonl'], 16, 8),
    'ko_math': (ML2, ['ultra-v3_math_ko_translated_final.jsonl'], 20, 8),
    'ko_stem': (ML2, ['ultra-v3_stem_ko_translated_postedit_final.jsonl'], 20, 8),
    'ko_code': (ML2, ['ultra-v3_code_ko_translated_final.jsonl'], 20, 6),
    'ja_chat': (ML2, ['ultra-v3_stem_ja_translated_postedit_final.jsonl', 'ultra-v3_math_ja_translated_final.jsonl'], 10, 6),
    'zh_chat': (ML1, ['data/super-v3_stem_zh_translated_postedit_final.jsonl', 'data/super-v3_math_zh_translated_final.jsonl',
                      'data/super-v3_code_zh_translated_final.jsonl'], 14, 6),
    'zh_r1': ('Congliu/Chinese-DeepSeek-R1-Distill-data-110k', ['distill_r1_110k.jsonl'], 40, 8),
    'zh_cqia': ('m-a-p/COIG-CQIA', ['COIG-CQIA-full.jsonl'], 32, 12),
    'es_chat': (ML1, ['data/super-v3_stem_es_translated_postedit_final.jsonl'], 16, 6),
    'finance': ('nvidia/Nemotron-SpecializedDomains-Finance-v1', ['data/train.jsonl'], 24, 4),
    'swe': ('nvidia/Nemotron-SFT-SWE-v2', ['data/agentless.jsonl'], 24, 6),
    'cp_py': ('nvidia/Nemotron-SFT-Competitive-Programming-v2', ['data/competitive_programming_python_00.jsonl'], 20, 6),
    'cp_cpp': ('nvidia/Nemotron-SFT-Competitive-Programming-v2', ['data/competitive_programming_cpp_00.jsonl'], 20, 6),
    'tool': ('nvidia/Nemotron-SFT-Agentic-v2', ['data/tool_calling.jsonl'], 40, 8),
}
# parquet sources: name -> (repo, [paths], row groups per path (seeded), text column)
PARQUET = {
    'en_web': ('HuggingFaceFW/fineweb-edu', ['sample/10BT/003_00000.parquet', 'sample/10BT/009_00000.parquet'], 4, 'text'),
    'ko_web': (FW2, ['data/kor_Hang/test/000_00000.parquet'], 6, 'text'),
    'zh_web': (FW2, ['data/cmn_Hani/test/000_00000.parquet'], 6, 'text'),
    'ja_web': (FW2, ['data/jpn_Jpan/test/000_00000.parquet'], 2, 'text'),
    'es_web': (FW2, ['data/spa_Latn/test/000_00000.parquet'], 2, 'text'),
    'math_web': ('HuggingFaceTB/finemath', ['finemath-4plus/train-00007-of-00064.parquet'], 3, 'text'),
    'code_web': ('codeparrot/github-code-clean', ['data/train-00011-of-00880.parquet', 'data/train-00412-of-00880.parquet'], 1, 'code'),
    'wikitext2': ('Salesforce/wikitext', ['wikitext-2-raw-v1/test-00000-of-00001.parquet'], 99, 'text'),
}
JSONL_BLOCK = 2 ** 20


def log(*a):
    print(time.strftime('%H:%M:%S'), *a, flush=True)


def fs():
    from huggingface_hub import HfFileSystem
    return HfFileSystem()


def fetch_jsonl(name, spec):
    repo, paths, segments, lines = spec
    out = RAW / name / 'records.jsonl'
    if out.exists():
        return name, 'cached'
    out.parent.mkdir(parents=True, exist_ok=True)
    f_s = fs()
    rng = random.Random(f'{SEED}:{name}')
    got = []
    for path in paths:
        full = f'datasets/{repo}@{REV[repo]}/{path}'
        size = f_s.info(full)['size']
        with f_s.open(full, 'rb', block_size=JSONL_BLOCK) as f:
            for s in range(segments):
                off = int(size * (s + rng.random() * 0.9) / segments)
                f.seek(off)
                if off:
                    f.readline()          # partial line after the seek
                for _ in range(lines):
                    pos = f.tell()
                    line = f.readline()
                    if not line:
                        break
                    if len(line) > 4_000_000:
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    rec['_src'] = {'repo': repo, 'revision': REV[repo], 'path': path, 'byte_offset': pos, 'segment': s}
                    got.append(rec)
    tmp = out.with_suffix('.partial')
    with tmp.open('w') as w:
        for r in got:
            w.write(json.dumps(r, ensure_ascii=False) + '\n')
    tmp.rename(out)
    return name, len(got)


def fetch_parquet(name, spec):
    import pyarrow.parquet as pq
    repo, paths, groups, column = spec
    out = RAW / name / 'records.jsonl'
    if out.exists():
        return name, 'cached'
    out.parent.mkdir(parents=True, exist_ok=True)
    f_s = fs()
    rng = random.Random(f'{SEED}:{name}')
    got = []
    for path in paths:
        full = f'datasets/{repo}@{REV[repo]}/{path}'
        with f_s.open(full, 'rb', block_size=2 ** 23) as f:
            pf = pq.ParquetFile(f)
            n = pf.num_row_groups
            picks = list(range(n)) if groups >= n else sorted(rng.sample(range(n), groups))
            cols = [column] + [c for c in ('language', 'path', 'repo_name', 'id', 'url') if c in pf.schema_arrow.names and c != column]
            for g in picks:
                t = pf.read_row_group(g, columns=cols).to_pylist()
                for i, row in enumerate(t):
                    row['text'] = row.pop(column)
                    row['_src'] = {'repo': repo, 'revision': REV[repo], 'path': path, 'row_group': g, 'row': i}
                    got.append(row)
    tmp = out.with_suffix('.partial')
    with tmp.open('w') as w:
        for r in got:
            w.write(json.dumps(r, ensure_ascii=False) + '\n')
    tmp.rename(out)
    return name, len(got)


def fetch(only=None):
    jobs = [(fetch_jsonl, n, s) for n, s in JSONL.items()] + [(fetch_parquet, n, s) for n, s in PARQUET.items()]
    if only:
        jobs = [j for j in jobs if j[1] in only]
    with ThreadPoolExecutor(len(jobs)) as ex:
        futs = {ex.submit(fn, n, s): n for fn, n, s in jobs}
        for fu in futs:
            n = futs[fu]
            try:
                log('fetched', *fu.result())
            except Exception as e:  # a slow or gated source is dropped, not fatal
                log('FETCH FAILED', n, repr(e)[:300])


# ---------------------------------------------------------------- rendering

THINK_RE = re.compile(r'^\s*<think>(.*?)</think>\s*', re.S)
# DS4 (V4) DSML as stored in prompts.jsonl `dsml` fields; re-emitted by the V4.1 encoder
DSML_INVOKE = re.compile(r'<｜DSML｜invoke name="([^"]+)">(.*?)</｜DSML｜invoke>', re.S)
DSML_PARAM = re.compile(r'<｜DSML｜parameter name="([^"]+)" string="(true|false)">(.*?)</｜DSML｜parameter>', re.S)
# text that would tokenize to V4.1 control tokens if it appeared inside document / message text
SPECIAL_STRINGS = ('<｜begin▁of▁sentence｜>', '<｜end▁of▁sentence｜>', '<｜User｜>', '<｜Assistant｜>', '<｜System｜>',
                   '｜DSML｜', '<｜latest_reminder｜>', '<｜deepseek_image｜>', '<think>', '</think>', '<｜▁pad▁｜>')


def has_special(s):
    return isinstance(s, str) and any(t in s for t in SPECIAL_STRINGS)


def as_text(c):
    if c is None:
        return ''
    if isinstance(c, str):
        return c
    if isinstance(c, list):
        return ''.join(p.get('text', '') for p in c if isinstance(p, dict))
    return json.dumps(c, ensure_ascii=False)


def split_think(msg):
    """-> (reasoning or None, content) from reasoning_content / reasoning / inline <think>."""
    content = as_text(msg.get('content'))
    r = msg.get('reasoning_content') or msg.get('reasoning')
    if not r:
        m = THINK_RE.match(content)
        if m:
            r, content = m.group(1), content[m.end():]
    return (r.strip() if isinstance(r, str) and r.strip() else None), content.lstrip('\n')


def parse_dsml_v4(dsml):
    calls = []
    for name, body in DSML_INVOKE.findall(dsml or ''):
        args = {}
        for p, is_str, v in DSML_PARAM.findall(body):
            v = v.replace('&lt;/｜DSML｜parameter>', '</｜DSML｜parameter>')
            if is_str == 'true':
                args[p] = v
            else:
                try:
                    args[p] = json.loads(v)
                except json.JSONDecodeError:
                    args[p] = v
        calls.append({'type': 'function', 'function': {'name': name, 'arguments': args}})
    return calls


def clean_messages(msgs, think):
    """Source messages -> V4.1 encoding messages (OpenAI style); None if unusable.

    Reasoning is kept as `reasoning_content` only for think renders (the encoding then
    drops it before the last user turn unless tools are declared, as at inference)."""
    out = []
    for m in msgs:
        if not isinstance(m, dict):
            return None
        role = m.get('role')
        if role not in ('system', 'user', 'assistant', 'tool'):
            continue
        if role == 'assistant':
            r, c = split_think(m)
            if has_special(c) or has_special(r):
                return None
            n = {'role': 'assistant', 'content': c}
            if think and r:
                n['reasoning_content'] = r
            calls = []
            for tc in m.get('tool_calls') or []:
                fn = tc.get('function', tc) if isinstance(tc, dict) else None
                if not isinstance(fn, dict) or not fn.get('name'):
                    continue
                args = fn.get('arguments')
                call = {'type': 'function', 'function': {'name': fn['name'], 'arguments': {} if args is None else args}}
                if tc.get('id'):
                    call['id'] = tc['id']
                calls.append(call)
            calls += parse_dsml_v4(m.get('dsml'))
            if calls:
                n['tool_calls'] = calls
            out.append(n)
        else:
            c = as_text(m.get('content'))
            if has_special(c):
                return None
            if role == 'system' and not c.strip():
                continue
            n = {'role': role, 'content': c}
            if role == 'tool' and m.get('tool_call_id'):
                n['tool_call_id'] = m['tool_call_id']
            out.append(n)
    return out


def has_reasoning(msgs):
    return any(isinstance(m, dict) and m.get('role') == 'assistant' and split_think(m)[0] for m in msgs)


def norm_tools(tools):
    if not tools:
        return None
    if isinstance(tools, str):
        try:
            tools = json.loads(tools)
        except json.JSONDecodeError:
            return None
    out = []
    for t in tools if isinstance(tools, list) else []:
        if not isinstance(t, dict):
            continue
        fn = t.get('function', t)
        if not isinstance(fn, dict) or not fn.get('name'):
            continue
        out.append({'type': 'function', 'function': fn})
    return out or None


def infer_tools(msgs):
    """Tool schemas for records that carry calls but no `tools` list."""
    seen = {}
    for m in msgs:
        for tc in m.get('tool_calls') or []:
            fn = tc['function']
            args = fn.get('arguments') or {}
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {}
            props = seen.setdefault(fn['name'], {})
            for k, v in (args.items() if isinstance(args, dict) else []):
                kind = ('boolean' if isinstance(v, bool) else 'integer' if isinstance(v, int) else
                        'number' if isinstance(v, float) else 'object' if isinstance(v, dict) else
                        'array' if isinstance(v, list) else 'string')
                props[k] = {'type': kind}
    return [{'type': 'function', 'function': {'name': n, 'description': f'{n} tool',
             'parameters': {'type': 'object', 'properties': p, 'required': sorted(p)}}} for n, p in seen.items()]


def ds4_tools(rendered):
    """Tool schemas of a DS4 record, read back from its rendered V4 system prompt (one JSON per line)."""
    a = rendered.find('### Available Tool Schemas\n\n')
    if a < 0:
        return None
    a += len('### Available Tool Schemas\n\n')
    b = rendered.find('\n\nYou MUST strictly follow', a)
    tools = []
    for line in rendered[a:b].split('\n'):
        line = line.strip()
        if line:
            tools.append({'type': 'function', 'function': json.loads(line)})
    return tools or None


def record_messages(name, r):
    """-> (messages, tools) of a raw record in OpenAI style (source dialects normalized)."""
    if name == 'zh_r1':
        return [{'role': 'user', 'content': r.get('input') or ''},
                {'role': 'assistant', 'content': r.get('content') or '', 'reasoning_content': r.get('reasoning_content')}], None
    if name == 'zh_cqia':
        q = (r.get('instruction') or '').strip()
        if (r.get('input') or '').strip():
            q += '\n\n' + r['input'].strip()
        return [{'role': 'user', 'content': q}, {'role': 'assistant', 'content': r.get('output') or ''}], None
    msgs = r.get('messages') or r.get('conversations') or []
    tools = norm_tools(r.get('tools'))
    if msgs and isinstance(msgs[0], dict) and not tools:
        tools = norm_tools(msgs[0].get('tools'))
    return msgs, tools


class Renderer:
    def __init__(self):
        spec = importlib.util.spec_from_file_location('dsv41_encoding', str(MODEL / 'encoding' / 'encoding.py'))
        self.enc = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.enc)
        from tokenizers import Tokenizer
        self.tok = Tokenizer.from_file(str(TOKENIZER))
        assert self.tok.token_to_id(self.enc.bos_token) == BOS_ID and self.tok.token_to_id(self.enc.eos_token) == EOS_ID

    def chat(self, msgs, think, tools=None):
        """Native V4.1 rendering: thinking mode (reasoning effort default "high") or chat mode.
        Tools are declared on the leading system message (an empty one is added if needed)."""
        msgs = [dict(m) for m in msgs]
        if tools:
            if not msgs or msgs[0]['role'] != 'system':
                msgs.insert(0, {'role': 'system', 'content': ''})
            msgs[0]['tools'] = tools
        return self.enc.encode_messages(msgs, thinking_mode='thinking' if think else 'chat')

    def encode(self, texts):
        return [e.ids for e in self.tok.encode_batch(list(texts), add_special_tokens=False)]

    def raw(self, texts):
        """Raw text rows begin with BOS exactly like an encoded conversation."""
        return [[BOS_ID] + ids for ids in self.encode(texts)]

    def decode(self, ids):
        return self.tok.decode(list(map(int, ids)), skip_special_tokens=False)


def load_raw(name):
    p = RAW / name / 'records.jsonl'
    if not p.exists():
        return []
    return [json.loads(l) for l in p.open()]


def load_ds4():
    recs = [json.loads(l) for l in open(DS4_PROMPTS)]
    for r in recs:
        r['_src'] = {'repo': 'Baekpica/ds4-dfm-rs', 'revision': DS4_REV, 'path': 'gguf-tools/imatrix/dataset/prompts.jsonl',
                     'id': r['id'], 'category': r['category'], 'ds4_source': r.get('source')}
    return recs


def ds4_twin_key(r):
    """DS4 records come in think / nothink twins with the same turns: key without reasoning."""
    return hashlib.sha256(json.dumps([(m.get('role'), m.get('content'), m.get('dsml')) for m in r['messages']],
                                     ensure_ascii=False).encode()).hexdigest()


def text_hash(s):
    return hashlib.sha256(s.encode('utf-8', 'replace')).hexdigest()


def code_ok(text):
    """Skip minified / generated files (very long lines)."""
    lines = text.split('\n')
    return max(len(l) for l in lines) <= 1000 and len(text) / max(1, len(lines)) <= 150


# ---------------------------------------------------------------- budgets (tokens)

RAW_BUDGET = {
    # name: (domain, lang, token budget, max tokens per doc incl. BOS)
    'en_web': ('web_en', 'en', 450_000, 16384),
    'ko_web': ('web_ko', 'ko', 255_000, 16384),
    'zh_web': ('web_zh', 'zh', 210_000, 16384),
    'code_web': ('code_raw', 'code', 345_000, 16384),
    'ja_web': ('web_ja', 'ja', 40_000, 8192),
    'es_web': ('web_es', 'es', 40_000, 8192),
    'math_web': ('math_raw', 'en', 135_000, 8192),
}
CHAT_BUDGET = {
    # name: (domain, lang, token budget, both-modes probability, max tokens per sequence)
    'if_on': ('chat_if', 'en', 170_000, 0.25, 8192),
    'if_off': ('chat_if', 'en', 125_000, 0.0, 8192),
    'casc2_if': ('chat_if', 'en', 55_000, 0.25, 8192),
    'casc1_math': ('reason_math', 'en', 150_000, 0.25, 16384),
    'casc1_science': ('reason_science', 'en', 95_000, 0.25, 16384),
    'casc1_code': ('reason_code', 'en', 75_000, 0.25, 16384),
    'casc2_general': ('reason_general', 'en', 105_000, 0.25, 16384),
    'ko_math': ('chat_ko', 'ko', 95_000, 0.25, 8192),
    'ko_stem': ('chat_ko', 'ko', 95_000, 0.25, 8192),
    'ko_code': ('chat_ko', 'ko', 80_000, 0.25, 8192),
    'zh_r1': ('chat_zh', 'zh', 110_000, 0.25, 16384),
    'zh_chat': ('chat_zh', 'zh', 90_000, 0.25, 8192),
    'zh_cqia': ('chat_zh', 'zh', 55_000, 0.0, 8192),
    'ja_chat': ('chat_ja', 'ja', 35_000, 0.25, 8192),
    'es_chat': ('chat_es', 'es', 35_000, 0.25, 8192),
    'tool': ('agent_tool', 'en', 210_000, 0.25, 16384),
    'swe': ('agent_swe', 'en', 135_000, 0.0, 16384),
    'cp_py': ('code_chat', 'code', 65_000, 0.25, 16384),
    'cp_cpp': ('code_chat', 'code', 65_000, 0.25, 16384),
    'finance': ('long_finance', 'en', 190_000, 0.0, MAX_LEN),
    'ds4': ('ds4_prompts', 'mixed', 185_000, 0.0, MAX_LEN),
}
NOTHINK_ONLY = {'if_off', 'zh_cqia'}

# tokens each bucket may spend on sequences longer than LONG (the rest is <= LONG): targets
# ~15% of all tokens in 4K-32K sequences (finance, SWE, long web/code, long reasoning, DS4)
LONG = 4096
LONG_ALLOW = {'finance': 190_000, 'swe': 70_000, 'en_web': 70_000, 'ko_web': 35_000, 'zh_web': 35_000,
              'code_web': 60_000, 'math_web': 10_000, 'ja_web': 6_000, 'es_web': 6_000, 'ds4': 45_000,
              'zh_r1': 25_000, 'zh_chat': 10_000, 'zh_cqia': 5_000,
              'if_on': 8_000, 'if_off': 8_000, 'casc2_if': 8_000, 'ko_math': 10_000, 'ko_stem': 10_000, 'ko_code': 10_000,
              'ja_chat': 5_000, 'es_chat': 5_000}
LONG_DEFAULT = 20_000

# held-out picks: (case name, source, kind, target tokens incl. BOS; wikitext windows are BOS + 2048)
HELDOUT = [
    ('wikitext2_a', 'wikitext2', 'stream', 2048), ('wikitext2_b', 'wikitext2', 'stream', 2048),
    ('en_web', 'en_web', 'doc', 2048), ('ko_web_a', 'ko_web', 'doc', 2048), ('ko_web_b', 'ko_web', 'doc', 2048),
    ('zh_web_a', 'zh_web', 'doc', 1536), ('zh_web_b', 'zh_web', 'doc', 1536),
    ('ja_web', 'ja_web', 'doc', 1536), ('es_web', 'es_web', 'doc', 1536),
    ('math_web', 'math_web', 'doc', 2048), ('code_python', 'code_web', 'doc_py', 2048), ('code_c', 'code_web', 'doc_c', 2048),
    ('ko_chat_nothink', 'ko_stem', 'chat_nothink', 3072), ('ko_math_think', 'ko_math', 'chat_think', 3072),
    ('en_reason_think', 'casc2_general', 'chat_think', 3072), ('math_think', 'casc1_math', 'chat_think', 3072),
    ('if_chat_nothink', 'if_off', 'chat_nothink', 3072), ('tool_call_dsml', 'tool', 'chat_tool', 4096),
    ('swe_agentless', 'swe', 'chat_nothink', 4096), ('code_chat_think', 'cp_py', 'chat_think', 4096),
    ('zh_chat_think', 'zh_r1', 'chat_think', 3072), ('zh_chat_nothink', 'zh_chat', 'chat_nothink', 3072),
    ('ja_chat_nothink', 'ja_chat', 'chat_nothink', 3072), ('ds4_agent_dsml', 'ds4', 'ds4_agent', 4096),
    ('long_en_doc', 'en_web', 'long_doc', 8192), ('long_finance', 'finance', 'chat_nothink_long', 8192),
]


def family_of(src):
    if src == 'wikitext2':
        return 'wikitext2'
    if src in RAW_BUDGET:
        return RAW_BUDGET[src][0]
    return CHAT_BUDGET[src][0]


def build():
    rng = random.Random(SEED)
    R = Renderer()
    samples, used_hashes = [], set()
    held = []
    t0 = time.monotonic()
    ds4_recs = load_ds4()
    random.Random(f'{SEED}:ds4').shuffle(ds4_recs)

    # ---- held-out first (documents reserved before any calibration pick)
    reserved = set()
    reserved_ds4 = set()
    pools = {}

    def pool(name):
        if name not in pools:
            recs = load_raw(name)
            random.Random(f'{SEED}:pool:{name}').shuffle(recs)
            pools[name] = recs
        return pools[name]

    for case, src, kind, target in HELDOUT:
        if kind == 'stream':
            ordered = load_raw(src)   # original row order (HF perplexity recipe: '\n\n'.join(test['text']))
            ids = R.encode(['\n\n'.join(r['text'] for r in ordered)])[0]
            start = 0 if case.endswith('_a') else len(ids) // 2
            piece = [BOS_ID] + ids[start:start + target]
            held.append({'case': case, 'family': 'wikitext2', 'domain': 'wikitext2', 'lang': 'en',
                         'source': ordered[0]['_src'] | {'row_group': None, 'row': None}, 'format': 'raw', 'mode': None,
                         'ids': piece, 'note': f'BOS + wikitext-2-raw test contiguous tokens [{start}, {start + target}) '
                                               f'of the {len(ids)}-token stream'})
            continue
        if kind == 'ds4_agent':
            chosen = None
            for r in ds4_recs:
                if r['category'] != 'agent' or not any(m.get('dsml') for m in r['messages']):
                    continue
                think = r['mode'] == 'think'
                msgs = clean_messages(r['messages'], think)
                tools = ds4_tools(r['rendered'])
                if not msgs or not tools:
                    continue
                ids = R.encode([R.chat(msgs, think, tools)])[0]
                if len(ids) < 1024:
                    continue
                chosen = (r, ids[:target], think)
                break
            if chosen is None:
                log('heldout: no suitable record', case)
                continue
            r, ids, think = chosen
            reserved_ds4.add(ds4_twin_key(r))
            held.append({'case': case, 'family': 'ds4_prompts', 'domain': 'ds4_prompts', 'lang': 'mixed', 'source': r['_src'],
                         'format': 'chat', 'mode': 'think' if think else 'nothink', 'tools': True, 'ids': ids,
                         'doc_sha256': ds4_twin_key(r)})
            continue
        recs = pool(src)
        if not recs:
            log('heldout source missing', case, src)
            continue
        chosen = None
        for j, r in enumerate(recs):
            if (src, j) in reserved:
                continue
            if kind.startswith('doc') or kind == 'long_doc':
                path = str(r.get('path', ''))
                if kind == 'doc_py' and not path.endswith('.py'):
                    continue
                if kind == 'doc_c' and not path.endswith(('.c', '.h')):
                    continue
                text = r.get('text') or ''
                if len(text) < 200 or has_special(text) or (src == 'code_web' and not code_ok(text)):
                    continue
                ids = R.raw([text])[0]
                lo = target * 0.6 if kind != 'long_doc' else target
                if len(ids) < lo:
                    continue
                chosen = (j, ids[:target], 'raw', None, r, None, text_hash(text))
                break
            raw_msgs, tools = record_messages(src, r)
            if kind == 'chat_tool':
                think = has_reasoning(raw_msgs)
            else:
                think = 'think' in kind and 'nothink' not in kind
            msgs = clean_messages(raw_msgs, think)
            if not msgs or msgs[-1]['role'] != 'assistant':
                continue
            if think and not has_reasoning(raw_msgs):
                continue
            if tools is None and any(m.get('tool_calls') for m in msgs):
                tools = infer_tools(msgs)
            if kind == 'chat_tool' and not (tools and any(m.get('tool_calls') for m in msgs)):
                continue
            try:
                text = R.chat(msgs, think, tools)
            except Exception:
                continue
            ids = R.encode([text])[0]
            lo = 512 if 'long' not in kind else target
            if len(ids) < lo:
                continue
            # teacher-forced measurement: a long record is cut to the target length
            chosen = (j, ids[:target], 'chat', 'think' if think else 'nothink', r, bool(tools),
                      text_hash(json.dumps(raw_msgs, ensure_ascii=False)))
            break
        if chosen is None:
            log('heldout: no suitable record', case)
            continue
        j, ids, fmt, mode, r, tools, digest = chosen
        reserved.add((src, j))
        h = {'case': case, 'family': family_of(src), 'domain': family_of(src),
             'lang': RAW_BUDGET[src][1] if src in RAW_BUDGET else CHAT_BUDGET[src][1],
             'source': r['_src'], 'format': fmt, 'mode': mode, 'ids': ids, 'doc_sha256': digest}
        if fmt == 'chat':
            h['tools'] = tools
        held.append(h)
    log('heldout cases', len(held), 'tokens', sum(len(h['ids']) for h in held), f'{time.monotonic() - t0:.0f}s')
    held_hashes = {h.get('doc_sha256') for h in held}

    long_used = {}
    overlap_report = {}
    skipped = {}

    def add(ids, meta):
        if not ids:
            return 0
        ids = ids[:MAX_LEN]
        h = hashlib.sha256(np.asarray(ids, np.int32).tobytes()).hexdigest()
        if h in used_hashes:
            return 0
        used_hashes.add(h)
        meta['n_tokens'] = len(ids)
        samples.append((ids, meta))
        return len(ids)

    # ---- raw text buckets (BOS + document)
    for name, (domain, lang, budget, cap) in RAW_BUDGET.items():
        recs = pool(name)
        got = 0
        cands = [(j, r) for j, r in enumerate(recs) if (name, j) not in reserved]
        for c0 in range(0, len(cands), 64):
            if got >= budget:
                break
            chunk = []
            for j, r in cands[c0:c0 + 64]:
                text = r.get('text') or ''
                if len(text) < 200 or text_hash(text) in held_hashes or has_special(text):
                    skipped[name] = skipped.get(name, 0) + 1
                    continue
                if name == 'code_web' and not code_ok(text):
                    skipped[name] = skipped.get(name, 0) + 1
                    continue
                chunk.append((j, r, text))
            for (j, r, text), ids in zip(chunk, R.raw([t for _, _, t in chunk])):
                if got >= budget:
                    break
                if len(ids) < 64:
                    continue
                n = min(len(ids), cap)
                if n > LONG:
                    if long_used.get(name, 0) + n > LONG_ALLOW.get(name, LONG_DEFAULT):
                        n = LONG          # over the allowance: cut to 4,096
                    else:
                        long_used[name] = long_used.get(name, 0) + n
                src = dict(r['_src'])
                for k in ('id', 'url', 'path', 'repo_name', 'language'):
                    if r.get(k) is not None:
                        src[k] = r[k]
                got += add(ids[:n], {'domain': domain, 'lang': lang, 'format': 'raw', 'mode': None, 'source': src,
                                     'truncated': len(ids) > n, 'doc_tokens': len(ids)})
        log('raw', name, got, 'tokens', f'{time.monotonic() - t0:.0f}s')

    # ---- chat buckets
    for name, (domain, lang, budget, both_p, max_len) in CHAT_BUDGET.items():
        if name == 'ds4':
            continue
        recs = pool(name)
        if name == 'finance':
            # prefer whole records (question and answer inside 32K); longer ones are cut at 32K
            lens = {j: len(json.dumps(r.get('messages'), ensure_ascii=False)) for j, r in enumerate(recs)}
            order = sorted(range(len(recs)), key=lambda j: (lens[j] > 110_000, random.Random(f'{SEED}:fin:{j}').random()))
        else:
            order = range(len(recs))
        got = 0
        for j in order:
            r = recs[j]
            if got >= budget:
                break
            if (name, j) in reserved:
                continue
            raw_msgs, tools = record_messages(name, r)
            if not raw_msgs or text_hash(json.dumps(raw_msgs, ensure_ascii=False)) in held_hashes:
                continue
            reasoning = has_reasoning(raw_msgs)
            if name in NOTHINK_ONLY:
                modes = ['nothink']
            elif name in ('swe', 'finance'):
                modes = ['think'] if reasoning and rng.random() < 0.5 else ['nothink']
            elif reasoning:
                u = rng.random()
                modes = ['think', 'nothink'] if u < both_p else (['think'] if u < both_p + (1 - both_p) * 0.5 else ['nothink'])
            else:
                modes = ['nothink']
            for mode in modes:
                msgs = clean_messages(raw_msgs, mode == 'think')
                if not msgs or not any(m['role'] == 'assistant' for m in msgs):
                    skipped[name] = skipped.get(name, 0) + 1
                    continue
                tl = tools
                if tl is None and any(m.get('tool_calls') for m in msgs):
                    tl = infer_tools(msgs)
                try:
                    text = R.chat(msgs, mode == 'think', tl)
                except Exception:
                    skipped[name] = skipped.get(name, 0) + 1
                    continue
                ids = R.encode([text])[0]
                if len(ids) < 32:
                    continue
                if len(ids) > max_len and name != 'finance':
                    continue
                n = min(len(ids), MAX_LEN)
                if n > LONG:
                    if long_used.get(name, 0) + n > LONG_ALLOW.get(name, LONG_DEFAULT):
                        continue
                    long_used[name] = long_used.get(name, 0) + n
                got += add(ids, {'domain': domain, 'lang': lang, 'format': 'chat', 'mode': mode, 'source': r['_src'],
                                 'truncated': len(ids) > MAX_LEN, 'tools': bool(tl), 'paired': len(modes) == 2,
                                 'turns': len(msgs)})
        log('chat', name, got, 'tokens', f'{time.monotonic() - t0:.0f}s')

    # ---- DS4 imatrix prompts rendered natively with the V4.1 encoding (local file)
    name = 'ds4'
    domain, lang, budget, _, _ = CHAT_BUDGET[name]
    got = 0
    for r in ds4_recs:
        if got >= budget:
            break
        if ds4_twin_key(r) in reserved_ds4:
            continue
        think = r.get('mode') == 'think'
        msgs = clean_messages(r['messages'], think)
        if not msgs:
            skipped[name] = skipped.get(name, 0) + 1
            continue
        tools = ds4_tools(r['rendered'])
        if tools is None and any(m.get('tool_calls') for m in msgs):
            tools = infer_tools(msgs)
        gen = msgs[-1]['role'] != 'assistant'
        try:
            text = R.chat(msgs, think, tools)
        except Exception:
            skipped[name] = skipped.get(name, 0) + 1
            continue
        ids = R.encode([text])[0]
        if len(ids) < 32:
            continue
        if len(ids) > LONG:
            if long_used.get(name, 0) + len(ids) > LONG_ALLOW.get(name, LONG_DEFAULT):
                continue
            long_used[name] = long_used.get(name, 0) + len(ids)
        got += add(ids, {'domain': domain, 'lang': lang, 'format': 'chat', 'mode': 'think' if think else 'nothink',
                         'source': r['_src'], 'generation_prompt': gen, 'tools': bool(tools), 'truncated': False})
    log('chat', name, got, 'tokens', 'skipped (control-token text / unusable)', skipped)

    # ---- held-out disjointness: drop calibration rows that share document text with a
    # held-out case. A 64-token window counts as boilerplate (chat template, reasoning-effort
    # prefix, tool schemas, task instructions) when it occurs in >= 3 calibration rows; any
    # other shared window removes the calibration row.
    W = 64

    def windows(ids):
        b = np.asarray(ids, np.int32).tobytes()
        return [hashlib.blake2b(b[4 * k:4 * (k + W)], digest_size=8).digest() for k in range(0, len(ids) - W + 1)]
    hw = {}
    for hc, hcase in enumerate(held):
        for d in windows(hcase['ids']):
            hw.setdefault(d, set()).add(hc)
    rows_per = {}
    for r, (ids, _) in enumerate(samples):
        for d in set(windows(ids)):
            if d in hw:
                rows_per.setdefault(d, set()).add(r)
    drop = set()
    for d, rs in rows_per.items():
        if len(rs) < 3:
            drop |= rs
    dropped = [{'domain': samples[r][1]['domain'], 'n_tokens': samples[r][1]['n_tokens'], 'source': samples[r][1]['source'],
                'heldout_cases': sorted({held[c]['case'] for d in windows(samples[r][0]) if d in hw for c in hw[d]})}
               for r in sorted(drop)]
    boiler = sum(1 for rs in rows_per.values() if len(rs) >= 3)
    log('heldout overlap: dropped calibration rows', len(dropped), sum(x['n_tokens'] for x in dropped), 'tokens;',
        'boilerplate windows', boiler)
    samples = [sm for r, sm in enumerate(samples) if r not in drop]
    overlap_report.update({'window_tokens': W, 'boilerplate_min_rows': 3, 'dropped_rows': dropped,
                           'dropped_tokens': sum(x['n_tokens'] for x in dropped),
                           'boilerplate_windows': boiler,
                           'shared_windows_total': len(rows_per)})

    # ---- write
    order = list(range(len(samples)))
    random.Random(f'{SEED}:order').shuffle(order)
    samples = [samples[i] for i in order]
    write_set(ROOT / 'calib', samples, extra={'heldout_overlap_filter': overlap_report, 'long_used': long_used})
    hs = [(h['ids'], {k: v for k, v in h.items() if k != 'ids'}) for h in held]
    write_set(ROOT / 'heldout', hs)
    log('done', f'{time.monotonic() - t0:.0f}s')


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 22), b''):
            h.update(b)
    return h.hexdigest()


def write_set(stem, samples, extra=None):
    from safetensors.numpy import save_file
    lens = [len(ids) for ids, _ in samples]
    flat = np.concatenate([np.asarray(ids, np.int32) for ids, _ in samples])
    if flat.max() >= VOCAB or flat.min() < 0 or min(lens) <= 0:
        raise ValueError('token id range / empty row')
    offsets = np.zeros(len(samples) + 1, np.int64)
    offsets[1:] = np.cumsum(lens)
    assert offsets[-1] == len(flat)
    if not all(flat[offsets[i]] == BOS_ID for i in range(len(samples))):
        raise ValueError('row without leading BOS')
    rows = {hashlib.sha256(flat[offsets[i]:offsets[i + 1]].tobytes()).hexdigest() for i in range(len(samples))}
    if len(rows) != len(samples):
        raise ValueError('duplicate rows')
    path = stem.with_suffix('.safetensors')
    meta = {'tokenizer': f'{MODEL_REPO}@{MODEL_REV}', 'bos_prepended': 'true', 'bos_id': str(BOS_ID),
            'seed': str(SEED), 'vocab_size': str(VOCAB)}
    save_file({'input_ids': flat, 'offsets': offsets}, str(path), metadata=meta)
    metas = []
    for i, (ids, m) in enumerate(samples):
        m = dict(m)
        m['i'] = i
        m['n_tokens'] = len(ids)
        metas.append(m)
    by_domain, by_lang, by_mode = {}, {}, {}
    for m in metas:
        by_domain[m.get('domain')] = by_domain.get(m.get('domain'), 0) + m['n_tokens']
        by_lang[m.get('lang')] = by_lang.get(m.get('lang'), 0) + m['n_tokens']
        key = f"{m.get('format')}:{m.get('mode')}"
        by_mode[key] = by_mode.get(key, 0) + m['n_tokens']
    total = int(offsets[-1])
    hist_edges = [0, 128, 512, 1024, 2048, 4096, 8192, 16384, 32769]
    hist = np.histogram(lens, bins=hist_edges)
    tok_hist = [int(sum(l for l in lens if lo <= l < hi)) for lo, hi in zip(hist_edges[:-1], hist_edges[1:])]
    totals = {'sequences': len(samples), 'tokens': total,
              'length': {'min': int(min(lens)), 'p50': int(np.median(lens)), 'p90': int(np.quantile(lens, .9)),
                         'max': int(max(lens)), 'mean': float(np.mean(lens))},
              'length_histogram': {'edges': hist_edges, 'sequences': hist[0].tolist(), 'tokens': tok_hist},
              'tokens_ge_4096': int(sum(l for l in lens if l >= 4096)),
              'tokens_by_domain': dict(sorted(by_domain.items(), key=lambda x: -x[1])),
              'share_by_domain': {k: v / total for k, v in sorted(by_domain.items(), key=lambda x: -x[1])},
              'tokens_by_lang': dict(sorted(by_lang.items(), key=lambda x: -x[1])),
              'tokens_by_format_mode': by_mode,
              'safetensors_sha256': sha256_file(path),
              'bos_prepended': True, 'bos_id': BOS_ID,
              'tokenizer': f'{MODEL_REPO}@{MODEL_REV} tokenizer.json sha256 {sha256_file(TOKENIZER)}',
              'chat_template': f'{MODEL_REPO}@{MODEL_REV} encoding/encoding.py encode_messages() '
                               f'(sha256 {sha256_file(MODEL / "encoding" / "encoding.py")}); thinking_mode thinking '
                               '(reasoning effort default "high" = 75) or chat; drop_thinking default',
              'ds4_prompts_sha256': sha256_file(DS4_PROMPTS),
              'seed': SEED, 'max_len': MAX_LEN}
    if extra:
        totals.update(extra)
    stem.with_name(stem.name + '_manifest.json').write_text(json.dumps({'samples': metas, 'totals': totals}, ensure_ascii=False, indent=0) + '\n')
    log('wrote', path, json.dumps({k: totals[k] for k in ('sequences', 'tokens', 'length', 'tokens_ge_4096')}))


# ---------------------------------------------------------------- sanity checks

def check(per_domain=2):
    from safetensors.numpy import load_file
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(str(TOKENIZER))
    out = []
    for stem in ('calib', 'heldout'):
        t = load_file(str(ROOT / f'{stem}.safetensors'))
        ids, off = t['input_ids'], t['offsets']
        man = json.loads((ROOT / f'{stem}_manifest.json').read_text())
        n = len(off) - 1
        assert ids.dtype == np.int32 and off.dtype == np.int64 and off[0] == 0 and off[-1] == len(ids)
        assert len(man['samples']) == n
        starts = ids[off[:-1]]
        bos_inside = int((ids == BOS_ID).sum()) - n
        lens = np.diff(off)
        assert all(int(lens[i]) == man['samples'][i]['n_tokens'] for i in range(n))
        log(stem, 'rows', n, 'tokens', len(ids), 'ids in [', int(ids.min()), ',', int(ids.max()), '] < VOCAB',
            bool(ids.max() < VOCAB), 'rows starting with BOS', int((starts == BOS_ID).sum()), 'BOS inside rows', bos_inside,
            'EOS', int((ids == EOS_ID).sum()), 'max len', int(lens.max()))
        seen = {}
        for i, m in enumerate(man['samples']):
            key = (m.get('case') or m['domain'], m.get('mode'), bool(m.get('tools')))
            if seen.get(key, 0) >= (1 if stem == 'heldout' else per_domain):
                continue
            seen[key] = seen.get(key, 0) + 1
            x = ids[off[i]:off[i + 1]].tolist()
            s = tok.decode(x, skip_special_tokens=False)
            head, tail = s[:900], s[-500:]
            out.append(f'===== {stem} row {i} {key} n_tokens={len(x)} source={json.dumps(m.get("source"), ensure_ascii=False)[:300]}\n'
                       f'--- head\n{head}\n--- tail\n{tail}\n')
    (ROOT / 'samples_decoded.txt').write_text('\n'.join(out))
    log('wrote', ROOT / 'samples_decoded.txt', len(out), 'samples')


def main():
    global ROOT, RAW, MODEL, TOKENIZER, DS4_PROMPTS
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('stage', choices=['fetch', 'build', 'check', 'all'])
    p.add_argument('--only', nargs='*')
    p.add_argument('--out', type=Path, default=ROOT, help='calibration directory (downloads under OUT/raw)')
    p.add_argument('--model', type=Path, default=MODEL, help='DeepSeek-V4.1 directory (encoding/encoding.py)')
    p.add_argument('--tokenizer', type=Path, default=TOKENIZER, help='DeepSeek-V4.1 tokenizer.json')
    p.add_argument('--ds4-prompts', type=Path, default=DS4_PROMPTS)
    a = p.parse_args()
    ROOT, RAW, MODEL, TOKENIZER, DS4_PROMPTS = a.out, a.out / 'raw', a.model, a.tokenizer, a.ds4_prompts
    if a.stage in ('fetch', 'all'):
        fetch(a.only)
    if a.stage in ('build', 'all'):
        build()
    if a.stage in ('check', 'all'):
        check()


if __name__ == '__main__':
    main()
