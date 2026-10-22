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
  freeze (--mix v2) copy calib.* to frozen/calib2-<sha8>.* + SHA256SUMS (never overwrites)

--mix v1 (default) reproduces the frozen calib-b57558b7 / heldout-c6b5c017 build unchanged.
--mix v2 (/work/ivq/calib2): even domain spread and stronger multilingual coverage; reuses the v1
raw cache read-only, fetches only new sources, and measures against the frozen v1 held-out set
(no held-out file is written). See the `mix v2` section below.

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
    if name == 'ml_reason':      # v2: lightblue multilingual R1 reasoning (inline <think> in the target language)
        return [{'role': 'user', 'content': r.get('translated_prompt') or ''},
                {'role': 'assistant', 'content': r.get('response') or ''}], None
    if name == 'ru_chat':        # v2: GrandMaster-PRO-MAX (role / content turns)
        return [dict(m) for m in r.get('conversation') or []], None
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


RAW_FALLBACK = []   # --mix v2: read-only raw caches of earlier builds (v1: empty)


def load_raw(name):
    for d in [RAW] + RAW_FALLBACK:
        p = d / name / 'records.jsonl'
        if p.exists():
            return [json.loads(l) for l in p.open()]
    return []


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


def write_set(stem, samples, extra=None, meta_extra=None):
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
    meta.update(meta_extra or {})
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

def check(per_domain=2, stems=('calib', 'heldout'), by_lang=False):
    from safetensors.numpy import load_file
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(str(TOKENIZER))
    out = []
    for stem in stems:
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
            if by_lang:
                key = (m['domain'], m.get('lang'), m.get('mode'), bool(m.get('tools')))
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


# ================================================================ mix v2 (calib2)
# `--mix v2` (v1 stays the default; its code path and constants above are unchanged): the same
# renderer, rendering rules, row metadata and held-out overlap rule, with domains spread more
# evenly (no domain above ~8%) and much stronger multilingual coverage (raw web text and
# think / non-think chat per language). v2 reads v1's cached raw samples read-only (--v1-raw),
# fetches only its new sources into OUT/raw, and is measured on the FROZEN v1 held-out set,
# which it never rebuilds or writes: held-out documents are excluded by source position and
# document hash, and calibration rows sharing a non-boilerplate 64-token window with a
# held-out case are dropped (v1 rule: a window in >= 3 calibration rows is boilerplate).

V2_OUT = Path('/work/ivq/calib2')
V1_RAW = Path('/work/ivq/calib/raw')
V1_HELDOUT = Path('/work/ivq/calib/frozen/heldout-c6b5c017.safetensors')
V1_HELDOUT_SHA256 = 'c6b5c0172c627f56bf11e636e903b8fdbf7efa1354218d7b90e6f9345cb65cc5'
LB = 'lightblue/reasoning-multilingual-R1-Llama-70B-train'
GM = 'Vikhrmodels/GrandMaster-PRO-MAX'
REV.update({LB: '242ea5e1d94e47357b3bd756175e277d16996202', GM: 'de9cc765d834f6d14f03155fe9c78b0b6c992b4c'})


def _ml_paths(prefix, lang, kinds):
    return [f'{prefix}_{k}_{lang}_translated{"_postedit" if k == "stem" else ""}_final.jsonl' for k in kinds]


# new JSONL sources (Nemotron multilingual, translated prompts / answers, English reasoning):
# name -> (repo, [paths], segments per path, lines per segment)
JSONL_V2 = {
    'ja_chat_ml1': (ML1, _ml_paths('data/super-v3', 'ja', ('stem', 'math', 'code')), 10, 6),
    'es_chat_ml1': (ML1, _ml_paths('data/super-v3', 'es', ('math', 'code')), 8, 6),
    'fr_chat': (ML1, _ml_paths('data/super-v3', 'fr', ('stem', 'math', 'code')), 10, 6),
    'de_chat': (ML1, _ml_paths('data/super-v3', 'de', ('stem', 'math', 'code')), 10, 6),
    'it_chat': (ML1, _ml_paths('data/super-v3', 'it', ('stem', 'math')), 8, 6),
    'pt_chat': (ML2, _ml_paths('ultra-v3', 'pt', ('stem', 'math', 'code')), 8, 6),
}
# new FineWeb-2 per-language test files: name -> (repo, [paths], seeded row groups (~1,000 docs each), column)
PARQUET_V2 = {
    'fr_web': (FW2, ['data/fra_Latn/test/000_00000.parquet'], 2, 'text'),
    'de_web': (FW2, ['data/deu_Latn/test/000_00000.parquet'], 2, 'text'),
    'ru_web': (FW2, ['data/rus_Cyrl/test/000_00000.parquet'], 2, 'text'),
    'pt_web': (FW2, ['data/por_Latn/test/000_00000.parquet'], 1, 'text'),
    'it_web': (FW2, ['data/ita_Latn/test/000_00000.parquet'], 1, 'text'),
    'vi_web': (FW2, ['data/vie_Latn/test/000_00000.parquet'], 1, 'text'),
    'id_web': (FW2, ['data/ind_Latn/test/000_00000.parquet'], 1, 'text'),
    'ar_web': (FW2, ['data/arb_Arab/test/000_00000.parquet'], 1, 'text'),
}
# small chat parquet files read whole: name -> (repo, [paths], columns)
ROWS_V2 = {
    'ml_reason': (LB, ['data/train-00000-of-00001.parquet'],
                  ['language', 'translated_prompt', 'response', 'row_id', 'dataset_name', 'res_finish_reason',
                   'is_valid_prompt', 'is_valid_think', 'is_valid_res', 'prompt_think_lang_correct']),
    'ru_chat': (GM, ['data/test-00000-of-00001.parquet'], ['source', 'conversation', 'prompt_lang', 'answer_lang']),
}
LB_LANG = {'Chinese': 'zh', 'Japanese': 'ja', 'Korean': 'ko', 'Spanish': 'es', 'French': 'fr', 'German': 'de',
           'Russian': 'ru', 'Portuguese': 'pt', 'Italian': 'it', 'Vietnamese': 'vi', 'Indonesian': 'id', 'Arabic': 'ar'}

# raw buckets: (source, domain, lang, token budget, max tokens per doc incl. BOS, tokens allowed in rows >= 4,096)
V2_RAW = [
    ('en_web', 'web_en', 'en', 152_000, 16384, 55_000),
    ('math_web', 'math_raw', 'en', 38_000, 8192, 8_000),
    ('code_web', 'code_raw', 'code', 285_000, 16384, 70_000),
    ('zh_web', 'web_zh', 'zh', 296_000, 16384, 55_000),
    ('ko_web', 'web_ko', 'ko', 185_000, 16384, 48_000),
    ('ja_web', 'web_ja', 'ja', 148_000, 16384, 25_000),
    ('es_web', 'web_es', 'es', 55_000, 8192, 8_000),
    ('fr_web', 'web_fr', 'fr', 48_000, 8192, 8_000),
    ('de_web', 'web_de', 'de', 48_000, 8192, 8_000),
    ('ru_web', 'web_ru', 'ru', 37_000, 8192, 6_000),
    ('pt_web', 'web_other', 'pt', 9_500, 8192, 0),
    ('it_web', 'web_other', 'it', 9_500, 8192, 0),
    ('vi_web', 'web_other', 'vi', 9_500, 8192, 0),
    ('id_web', 'web_other', 'id', 9_500, 8192, 0),
    ('ar_web', 'web_other', 'ar', 9_500, 8192, 0),
]
# chat buckets: (bucket, source, domain, lang, token budget, both-modes probability, think probability of a
# single-mode render (None: non-think only), max tokens per row, tokens allowed in rows >= 4,096,
# lightblue languages (rows of the shared `ml_reason` source; row lang from the record))
V2_CHAT = [
    ('if_on', 'if_on', 'chat_if', 'en', 110_000, 0.25, 0.5, 8192, 12_000, None),
    ('if_off', 'if_off', 'chat_if', 'en', 75_000, 0.0, None, 8192, 5_000, None),
    ('casc2_if', 'casc2_if', 'chat_if', 'en', 37_000, 0.25, 0.5, 8192, 5_000, None),
    ('casc1_math', 'casc1_math', 'reason_math', 'en', 120_000, 0.25, 0.5, 16384, 26_000, None),
    ('casc1_science', 'casc1_science', 'reason_science', 'en', 95_000, 0.25, 0.5, 16384, 26_000, None),
    ('casc1_code', 'casc1_code', 'reason_code', 'en', 85_000, 0.25, 0.5, 16384, 26_000, None),
    ('casc2_general', 'casc2_general', 'reason_general', 'en', 105_000, 0.25, 0.5, 16384, 26_000, None),
    ('tool', 'tool', 'agent_tool', 'en', 222_000, 0.25, 0.5, 16384, 50_000, None),
    ('swe', 'swe', 'agent_swe', 'en', 111_000, 0.0, 0.5, 16384, 55_000, None),
    ('finance', 'finance', 'long_finance', 'en', 115_000, 0.0, 0.5, MAX_LEN, 115_000, None),
    ('cp_py', 'cp_py', 'code_chat', 'code', 58_000, 0.25, 0.5, 16384, 20_000, None),
    ('cp_cpp', 'cp_cpp', 'code_chat', 'code', 58_000, 0.25, 0.5, 16384, 20_000, None),
    ('zh_r1', 'zh_r1', 'chat_zh', 'zh', 118_000, 0.25, 0.5, 16384, 18_000, None),
    ('zh_cqia', 'zh_cqia', 'chat_zh', 'zh', 70_000, 0.0, None, 8192, 5_000, None),
    ('zh_chat', 'zh_chat', 'chat_zh', 'zh', 90_000, 0.25, 0.5, 8192, 10_000, None),
    ('lb_zh', 'ml_reason', 'chat_zh', 'zh', 15_000, 0.25, 0.5, 8192, 0, ('Chinese',)),
    ('ko_math', 'ko_math', 'chat_ko', 'ko', 95_000, 0.25, 0.5, 8192, 12_000, None),
    ('ko_stem', 'ko_stem', 'chat_ko', 'ko', 95_000, 0.25, 0.5, 8192, 12_000, None),
    ('ko_code', 'ko_code', 'chat_ko', 'ko', 90_000, 0.25, 0.5, 8192, 12_000, None),
    ('lb_ko', 'ml_reason', 'chat_ko', 'ko', 10_000, 0.25, 0.5, 8192, 0, ('Korean',)),
    ('ja_chat', 'ja_chat', 'chat_ja', 'ja', 58_000, 0.25, 0.5, 8192, 10_000, None),
    ('ja_chat_ml1', 'ja_chat_ml1', 'chat_ja', 'ja', 75_000, 0.25, 0.5, 8192, 10_000, None),
    ('lb_ja', 'ml_reason', 'chat_ja', 'ja', 15_000, 0.25, 0.5, 8192, 0, ('Japanese',)),
    ('es_chat', 'es_chat', 'chat_es', 'es', 35_000, 0.25, 0.5, 8192, 4_500, None),
    ('es_chat_ml1', 'es_chat_ml1', 'chat_es', 'es', 28_000, 0.25, 0.5, 8192, 4_500, None),
    ('lb_es', 'ml_reason', 'chat_es', 'es', 12_000, 0.25, 0.5, 8192, 0, ('Spanish',)),
    ('fr_chat', 'fr_chat', 'chat_fr', 'fr', 55_000, 0.25, 0.5, 8192, 5_000, None),
    ('lb_fr', 'ml_reason', 'chat_fr', 'fr', 10_000, 0.25, 0.5, 8192, 0, ('French',)),
    ('de_chat', 'de_chat', 'chat_de', 'de', 58_000, 0.25, 0.5, 8192, 5_000, None),
    ('lb_de', 'ml_reason', 'chat_de', 'de', 7_000, 0.25, 0.5, 8192, 0, ('German',)),
    ('ru_chat', 'ru_chat', 'chat_ru', 'ru', 20_000, 0.0, None, 8192, 0, None),
    ('lb_ru', 'ml_reason', 'chat_ru', 'ru', 20_000, 0.25, 0.5, 8192, 0, ('Russian',)),
    ('it_chat', 'it_chat', 'chat_other', 'it', 12_000, 0.25, 0.5, 8192, 0, None),
    ('pt_chat', 'pt_chat', 'chat_other', 'pt', 14_000, 0.25, 0.5, 8192, 0, None),
    ('lb_other', 'ml_reason', 'chat_other', None, 24_000, 0.25, 0.5, 8192, 0,
     ('Portuguese', 'Italian', 'Vietnamese', 'Indonesian', 'Arabic')),
    ('ds4', 'ds4', 'ds4_prompts', 'mixed', 148_000, None, None, MAX_LEN, 40_000, None),   # mode from the record
]


def fetch_rows(name, spec):
    """Small chat parquet file read whole (every row group), all listed columns kept."""
    import pyarrow.parquet as pq
    repo, paths, cols = spec
    out = RAW / name / 'records.jsonl'
    if out.exists():
        return name, 'cached'
    out.parent.mkdir(parents=True, exist_ok=True)
    f_s = fs()
    got = []
    for path in paths:
        full = f'datasets/{repo}@{REV[repo]}/{path}'
        with f_s.open(full, 'rb', block_size=2 ** 23) as f:
            pf = pq.ParquetFile(f)
            for g in range(pf.num_row_groups):
                for i, row in enumerate(pf.read_row_group(g, columns=cols).to_pylist()):
                    row['_src'] = {'repo': repo, 'revision': REV[repo], 'path': path, 'row_group': g, 'row': i}
                    got.append(row)
    tmp = out.with_suffix('.partial')
    with tmp.open('w') as w:
        for r in got:
            w.write(json.dumps(r, ensure_ascii=False) + '\n')
    tmp.rename(out)
    return name, len(got)


def fetch_v2(only=None):
    jobs = ([(fetch_jsonl, n, s) for n, s in JSONL_V2.items()] + [(fetch_parquet, n, s) for n, s in PARQUET_V2.items()]
            + [(fetch_rows, n, s) for n, s in ROWS_V2.items()])
    if only:
        jobs = [j for j in jobs if j[1] in only]
    new = set(JSONL_V2) | set(PARQUET_V2) | set(ROWS_V2)
    reused = sorted(({b[0] for b in V2_RAW} | {b[1] for b in V2_CHAT if b[1] != 'ds4'}) - new)
    missing = [n for n in reused if not any((d / n / 'records.jsonl').exists() for d in [RAW] + RAW_FALLBACK)]
    if missing:
        log('v1 raw cache missing', missing, '-> `fetch --only ... --out <this dir>` (v1 specs) first')
    with ThreadPoolExecutor(max(1, len(jobs))) as ex:
        futs = {ex.submit(fn, n, s): n for fn, n, s in jobs}
        for fu in futs:
            n = futs[fu]
            try:
                log('fetched', *fu.result())
            except Exception as e:
                log('FETCH FAILED', n, repr(e)[:300])


def v2_src_key(s):
    s = s or {}
    return (s.get('repo'), s.get('path'), s.get('row_group'), s.get('row'), s.get('byte_offset'),
            s.get('id') if s.get('repo') == 'Baekpica/ds4-dfm-rs' else None)


def load_v1_heldout():
    """The frozen v1 held-out set shared by every variant (read-only): cases with ids, content sha256."""
    from safetensors.numpy import load_file
    if sha256_file(V1_HELDOUT) != V1_HELDOUT_SHA256:
        raise SystemExit(f'{V1_HELDOUT}: sha256 mismatch')
    t = load_file(str(V1_HELDOUT))
    man = json.loads(V1_HELDOUT.with_name(V1_HELDOUT.stem + '_manifest.json').read_text())
    ids, off = t['input_ids'], t['offsets']
    held = [dict(m, ids=ids[off[i]:off[i + 1]].tolist()) for i, m in enumerate(man['samples'])]
    return held, hashlib.sha256(ids.tobytes() + off.tobytes()).hexdigest()


def heldout_overlap_filter(samples, held, W=64, boiler_rows=3):
    """v1 rule (build()): drop every calibration row that shares a 64-token window with a held-out
    case, unless the window occurs in >= 3 calibration rows (template / effort prefix / tool schemas)."""
    def windows(ids):
        b = np.asarray(ids, np.int32).tobytes()
        return [hashlib.blake2b(b[4 * k:4 * (k + W)], digest_size=8).digest() for k in range(0, len(ids) - W + 1)]
    hw = {}
    for hc, h in enumerate(held):
        for d in windows(h['ids']):
            hw.setdefault(d, set()).add(hc)
    rows_per = {}
    for r, (ids, _) in enumerate(samples):
        for d in set(windows(ids)):
            if d in hw:
                rows_per.setdefault(d, set()).add(r)
    drop = set()
    for d, rs in rows_per.items():
        if len(rs) < boiler_rows:
            drop |= rs
    dropped = [{'domain': samples[r][1]['domain'], 'bucket': samples[r][1].get('bucket'), 'record': samples[r][1].get('_rec'),
                'n_tokens': samples[r][1]['n_tokens'], 'source': samples[r][1]['source'],
                'heldout_cases': sorted({held[c]['case'] for d in windows(samples[r][0]) if d in hw for c in hw[d]})}
               for r in sorted(drop)]
    report = {'window_tokens': W, 'boilerplate_min_rows': boiler_rows, 'dropped_rows': dropped,
              'dropped_tokens': sum(x['n_tokens'] for x in dropped),
              'boilerplate_windows': sum(1 for rs in rows_per.values() if len(rs) >= boiler_rows),
              'shared_windows_total': len(rows_per)}
    return [sm for r, sm in enumerate(samples) if r not in drop], report


def lb_ok(r):
    return bool(r.get('is_valid_prompt') and r.get('is_valid_think') and r.get('is_valid_res')
                and r.get('prompt_think_lang_correct') and r.get('res_finish_reason') == 'stop')


def build_v2():
    R = Renderer()
    t0 = time.monotonic()
    held, held_content = load_v1_heldout()
    held_hashes = {h['doc_sha256'] for h in held if h.get('doc_sha256')}
    held_src = {v2_src_key(h.get('source')) for h in held if h.get('family') != 'wikitext2'}
    log('frozen v1 held-out', V1_HELDOUT.name, len(held), 'cases', sum(len(h['ids']) for h in held), 'tokens; content',
        held_content)
    pools = {}
    ds4_recs = load_ds4()
    random.Random(f'{SEED}:ds4').shuffle(ds4_recs)

    def pool(name):
        if name not in pools:
            recs = load_raw(name)
            if not recs:
                raise SystemExit(f'no raw samples for {name}: run `fetch --mix v2` (and the v1 cache under --v1-raw)')
            random.Random(f'{SEED}:pool:{name}').shuffle(recs)
            pools[name] = recs
        return pools[name]

    def select(exclude):
        """One deterministic selection pass; `exclude` holds records dropped by an earlier
        overlap filter (their bucket is refilled from the next records)."""
        samples, used_hashes, used_recs = [], set(), set()
        st = {'long_used': {}, 'skipped': {}, 'excluded': {}, 'got': {}}
        long_used, skipped, excluded = st['long_used'], st['skipped'], st['excluded']

        def is_held(name, key, r, digest):
            if key in exclude:
                return True
            if v2_src_key(r.get('_src')) in held_src or digest in held_hashes:
                excluded[name] = excluded.get(name, 0) + 1
                return True
            return False

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

        def skip(name):
            skipped[name] = skipped.get(name, 0) + 1

        # ---- raw text buckets (BOS + document); a document over its bucket's long allowance is
        # cut to 4,095 tokens, so rows >= 4,096 are long by intent only
        for name, domain, lang, budget, cap, allow in V2_RAW:
            recs = pool(name)
            got = 0
            for c0 in range(0, len(recs), 64):
                if got >= budget:
                    break
                chunk = []
                for j, r in enumerate(recs[c0:c0 + 64], c0):
                    text = r.get('text') or ''
                    if is_held(name, (name, j), r, text_hash(text)):
                        continue
                    if len(text) < 200 or has_special(text) or (name == 'code_web' and not code_ok(text)):
                        skip(name)
                        continue
                    chunk.append((j, r, text))
                for (j, r, text), ids in zip(chunk, R.raw([t for _, _, t in chunk])):
                    if got >= budget:
                        break
                    if len(ids) < 64:
                        continue
                    n = min(len(ids), cap)
                    if n >= LONG:
                        if long_used.get(name, 0) + n > allow:
                            n = LONG - 1
                        else:
                            long_used[name] = long_used.get(name, 0) + n
                    src = dict(r['_src'])
                    for k in ('id', 'url', 'path', 'repo_name', 'language'):
                        if r.get(k) is not None:
                            src[k] = r[k]
                    used_recs.add((name, j))
                    got += add(ids[:n], {'domain': domain, 'lang': lang, 'format': 'raw', 'mode': None, 'source': src,
                                         'truncated': len(ids) > n, 'doc_tokens': len(ids), 'bucket': name,
                                         '_rec': (name, j)})
            st['got'][name] = got

        # ---- chat buckets (v1 rendering; per-bucket seeded mode draws)
        for bucket, name, domain, lang, budget, both_p, p_think, max_len, allow, lb_langs in V2_CHAT:
            if name == 'ds4':
                continue
            recs = pool(name)
            rng = random.Random(f'{SEED}:v2:mode:{bucket}')
            if name == 'finance':
                lens = {j: len(json.dumps(r.get('messages'), ensure_ascii=False)) for j, r in enumerate(recs)}
                order = sorted(range(len(recs)),
                               key=lambda j: (lens[j] > 110_000, random.Random(f'{SEED}:fin:{j}').random()))
            else:
                order = range(len(recs))
            got = 0
            for j in order:
                if got >= budget:
                    break
                r = recs[j]
                if (name, j) in used_recs:
                    continue
                if lb_langs is not None and (r.get('language') not in lb_langs or not lb_ok(r)):
                    continue
                if name == 'ru_chat' and not (r.get('prompt_lang') == 'ru' and r.get('answer_lang') == 'ru'):
                    continue
                raw_msgs, tools = record_messages(name, r)
                if not raw_msgs or is_held(name, (name, j), r, text_hash(json.dumps(raw_msgs, ensure_ascii=False))):
                    continue
                used_recs.add((name, j))
                reasoning = has_reasoning(raw_msgs)
                if p_think is None:
                    modes = ['nothink']
                elif name in ('swe', 'finance'):
                    modes = ['think'] if reasoning and rng.random() < p_think else ['nothink']
                elif reasoning:
                    u = rng.random()
                    modes = (['think', 'nothink'] if u < both_p else
                             ['think'] if u < both_p + (1 - both_p) * p_think else ['nothink'])
                else:
                    modes = ['nothink']
                row_lang = LB_LANG[r['language']] if lb_langs else lang
                for mode in modes:
                    msgs = clean_messages(raw_msgs, mode == 'think')
                    if not msgs or not any(m['role'] == 'assistant' for m in msgs):
                        skip(name)
                        continue
                    tl = tools
                    if tl is None and any(m.get('tool_calls') for m in msgs):
                        tl = infer_tools(msgs)
                    try:
                        text = R.chat(msgs, mode == 'think', tl)
                    except Exception:
                        skip(name)
                        continue
                    ids = R.encode([text])[0]
                    if len(ids) < 32:
                        continue
                    if len(ids) > max_len and name != 'finance':
                        continue
                    n = min(len(ids), MAX_LEN)
                    if n >= LONG:
                        if long_used.get(bucket, 0) + n > allow:
                            continue
                        long_used[bucket] = long_used.get(bucket, 0) + n
                    got += add(ids, {'domain': domain, 'lang': row_lang, 'format': 'chat', 'mode': mode,
                                     'source': r['_src'], 'truncated': len(ids) > MAX_LEN, 'tools': bool(tl),
                                     'paired': len(modes) == 2, 'turns': len(msgs), 'bucket': bucket,
                                     '_rec': (name, j)})
            st['got'][bucket] = got

        # ---- DS4 imatrix prompts (as v1: native V4.1 re-render, record mode, held-out twins excluded)
        _, name, domain, lang, budget, _, _, _, allow, _ = next(b for b in V2_CHAT if b[1] == 'ds4')
        got = 0
        for r in ds4_recs:
            if got >= budget:
                break
            key = ('ds4', ds4_twin_key(r))
            if key in exclude:
                continue
            if key[1] in held_hashes:
                excluded[name] = excluded.get(name, 0) + 1
                continue
            think = r.get('mode') == 'think'
            msgs = clean_messages(r['messages'], think)
            if not msgs:
                skip(name)
                continue
            tools = ds4_tools(r['rendered'])
            if tools is None and any(m.get('tool_calls') for m in msgs):
                tools = infer_tools(msgs)
            gen = msgs[-1]['role'] != 'assistant'
            try:
                text = R.chat(msgs, think, tools)
            except Exception:
                skip(name)
                continue
            ids = R.encode([text])[0]
            if len(ids) < 32:
                continue
            if len(ids) >= LONG:
                if long_used.get(name, 0) + len(ids) > allow:
                    continue
                long_used[name] = long_used.get(name, 0) + len(ids)
            got += add(ids, {'domain': domain, 'lang': lang, 'format': 'chat', 'mode': 'think' if think else 'nothink',
                             'source': r['_src'], 'generation_prompt': gen, 'tools': bool(tools), 'truncated': False,
                             'bucket': name, '_rec': key})
        st['got'][name] = got
        return samples, st

    # ---- select, filter against the frozen v1 held-out set, refill: a record whose row shares a
    # non-boilerplate 64-token window with a held-out case is excluded and its bucket refilled from
    # the next records; repeated until the filter drops nothing (the last filter always applies)
    exclude, passes = set(), []
    for it in range(6):
        samples, st = select(exclude)
        n_sel, t_sel = len(samples), sum(len(ids) for ids, _ in samples)
        samples, overlap = heldout_overlap_filter(samples, held)
        log(f'pass {it + 1}: selected {n_sel} rows {t_sel} tokens; overlap filter dropped',
            len(overlap['dropped_rows']), 'rows', overlap['dropped_tokens'], 'tokens; boilerplate windows',
            overlap['boilerplate_windows'], 'of', overlap['shared_windows_total'], f'{time.monotonic() - t0:.0f}s')
        passes.append({'pass': it + 1, 'selected_rows': n_sel, 'selected_tokens': t_sel,
                       'dropped_rows': overlap['dropped_rows'], 'dropped_tokens': overlap['dropped_tokens'],
                       'boilerplate_windows': overlap['boilerplate_windows'],
                       'shared_windows_total': overlap['shared_windows_total']})
        if not overlap['dropped_rows']:
            break
        exclude |= {tuple(d['record']) for d in overlap['dropped_rows']}
    for k, v in st['got'].items():
        log('bucket', k, v, 'tokens (before the last filter)')
    log('skipped (control-token text / unusable)', st['skipped'], '; held-out records excluded', st['excluded'])

    # ---- write (calibration only; the held-out set is v1's frozen file)
    for _, m in samples:
        m.pop('_rec', None)
    for ps in passes:
        for d in ps['dropped_rows']:
            d['record'] = list(d['record'])
    order = list(range(len(samples)))
    random.Random(f'{SEED}:v2:order').shuffle(order)
    samples = [samples[i] for i in order]
    buckets = {b[0]: {'source': b[0], 'domain': b[1], 'lang': b[2], 'budget': b[3], 'max_doc_tokens': b[4],
                      'long_allow': b[5], 'tokens_selected': st['got'].get(b[0], 0)} for b in V2_RAW}
    buckets.update({b[0]: {'source': b[1], 'domain': b[2], 'lang': b[3] or 'from record', 'budget': b[4],
                           'both_modes_p': b[5], 'think_p': b[6], 'max_row_tokens': b[7], 'long_allow': b[8],
                           'lightblue_languages': b[9], 'tokens_selected': st['got'].get(b[0], 0)}
                    for b in V2_CHAT})
    write_set(ROOT / 'calib', samples, meta_extra={'mix': 'v2', 'heldout': V1_HELDOUT.stem},
              extra={'mix': 'v2', 'heldout': {'file': str(V1_HELDOUT), 'sha256': V1_HELDOUT_SHA256,
                                              'content_sha256': held_content, 'cases': len(held),
                                              'tokens': sum(len(h['ids']) for h in held)},
                     'heldout_overlap_filter': overlap | {'refill_passes': passes,
                                                          'records_excluded_by_overlap': len(exclude)},
                     'heldout_records_excluded': st['excluded'], 'skipped_control_or_unusable': st['skipped'],
                     'long_used': st['long_used'], 'buckets': buckets})
    from safetensors.numpy import load_file
    t = load_file(str(ROOT / 'calib.safetensors'))
    content = hashlib.sha256(t['input_ids'].tobytes() + t['offsets'].tobytes()).hexdigest()
    mp = ROOT / 'calib_manifest.json'
    man = json.loads(mp.read_text())
    man['totals']['content_sha256'] = content
    mp.write_text(json.dumps(man, ensure_ascii=False, indent=0) + '\n')
    log('content sha256 (input_ids + offsets)', content, 'done', f'{time.monotonic() - t0:.0f}s')


def freeze_v2():
    """Copy calib(.safetensors|_manifest.json) to frozen/calib2-<sha8>.*; never overwrites."""
    import shutil
    src, man = ROOT / 'calib.safetensors', ROOT / 'calib_manifest.json'
    digest = sha256_file(src)
    fz = ROOT / 'frozen'
    fz.mkdir(exist_ok=True)
    dst, dman = fz / f'calib2-{digest[:8]}.safetensors', fz / f'calib2-{digest[:8]}_manifest.json'
    if dst.exists() or dman.exists():
        raise SystemExit(f'{dst} already exists: not overwritten')
    shutil.copy2(src, dst)
    shutil.copy2(man, dman)
    if sha256_file(dst) != digest:
        raise SystemExit('copy verification failed')
    with (fz / 'SHA256SUMS').open('a') as w:
        for p in (dst, dman):
            w.write(f'{sha256_file(p)}  {p.name}\n')
    log('frozen', dst, digest)


def main():
    global ROOT, RAW, MODEL, TOKENIZER, DS4_PROMPTS
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('stage', choices=['fetch', 'build', 'check', 'freeze', 'all'])
    p.add_argument('--mix', choices=['v1', 'v2'], default='v1',
                   help='v1 (default): the frozen calib-b57558b7 recipe; v2: calib2 (even spread, multilingual)')
    p.add_argument('--only', nargs='*')
    p.add_argument('--out', type=Path, default=None,
                   help=f'calibration directory (downloads under OUT/raw); default {ROOT} (v1) or {V2_OUT} (v2)')
    p.add_argument('--model', type=Path, default=MODEL, help='DeepSeek-V4.1 directory (encoding/encoding.py)')
    p.add_argument('--tokenizer', type=Path, default=None,
                   help=f'DeepSeek-V4.1 tokenizer.json; default {TOKENIZER} (v1) or MODEL/tokenizer.json (v2)')
    p.add_argument('--ds4-prompts', type=Path, default=DS4_PROMPTS)
    p.add_argument('--v1-raw', type=Path, default=V1_RAW, help='v2: read-only raw cache of the v1 build')
    a = p.parse_args()
    out = a.out or (V2_OUT if a.mix == 'v2' else ROOT)
    tok = a.tokenizer or (a.model / 'tokenizer.json' if a.mix == 'v2' else TOKENIZER)
    ROOT, RAW, MODEL, TOKENIZER, DS4_PROMPTS = out, out / 'raw', a.model, tok, a.ds4_prompts
    if a.mix == 'v2':
        RAW_FALLBACK.append(a.v1_raw)
        if a.stage in ('fetch', 'all'):
            fetch_v2(a.only)
        if a.stage in ('build', 'all'):
            build_v2()
        if a.stage in ('check', 'all'):
            check(stems=('calib',), by_lang=True)
        if a.stage == 'freeze':
            freeze_v2()
        return
    if a.stage == 'freeze':
        raise SystemExit('v1 is frozen already (/work/ivq/calib/frozen)')
    if a.stage in ('fetch', 'all'):
        fetch(a.only)
    if a.stage in ('build', 'all'):
        build()
    if a.stage in ('check', 'all'):
        check()


if __name__ == '__main__':
    main()
