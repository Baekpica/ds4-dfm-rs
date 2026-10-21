#!/usr/bin/env python3
"""A.X-K2 calibration and held-out token sets for the VQ imatrix collector.

  fetch  stream bounded samples of every source into calib/raw/<source>/ (data only)
  build  render chat records with the A.X-K2 chat template, tokenize, budget by
         domain and length, and write calib(.safetensors|_manifest.json) and
         heldout(.safetensors|_manifest.json): `input_ids` int32 [N] + `offsets` int64 [n+1]
         and a per-sample JSON manifest (docs/vq-calibration.md, Pipeline)

Every remote file is pinned to a revision. Large JSONL files are read at
seeded random byte offsets (a partial line is skipped after each seek), large
parquet files by row group, so no source is downloaded whole. Downloaded
bytes are parsed as JSON / parquet data only.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
import json
import os
from pathlib import Path
import random
import re
import sys
import time

import numpy as np

ROOT = Path('/workspace/axk2/calib')
RAW = ROOT / 'raw'
MODEL = '/models/A.X-K2'
DS4_PROMPTS = Path(__file__).resolve().parents[2] / 'gguf-tools/imatrix/dataset/prompts.jsonl'
SEED = 20261021
MAX_LEN = 32768

# (name, repo, revision, path, kind, fetch budget)
REV = {
    'HuggingFaceFW/fineweb-edu': '87f09149ef4734204d70ed1d046ddc9ca3f2b8f9',
    'HuggingFaceFW/fineweb-2': 'af9c13333eb981300149d5ca60a8e9d659b276b9',
    'HuggingFaceTB/finemath': 'e92b25a616738fe95dc186b64dfb19f9c8525594',
    'codeparrot/github-code-clean': 'c48d40f9e70f0196f8236901ee35807f7d6c44c0',
    'nvidia/Nemotron-Pretraining-Code-v2': '7b1a453d43e3c0df9749834b04f1a9510c0f5e5b',
    'nvidia/Nemotron-SFT-Instruction-Following-Chat-v2': '1a9454ed054b8544503ab8d8c0a519d141a44c5b',
    'nvidia/Nemotron-Cascade-SFT-Stage-1': 'e59a356b107737562f064ec252e6adacbc737899',
    'nvidia/Nemotron-Cascade-SFT-Stage-2': '54cfa23ba28c9d94aeae15142e73bde6f8d14d8b',
    'nvidia/Nemotron-SFT-Multilingual-v2': '971a252224b75414b1b67c55dbe0446d8b6606a0',
    'nvidia/Nemotron-SFT-Multilingual-v1': '22c86505762a7c595abee309d720084351c9f4ba',
    'nvidia/Nemotron-SpecializedDomains-Finance-v1': '5a21b106168facb96ced11b883c2a9b4788ee939',
    'nvidia/Nemotron-SFT-SWE-v2': 'bd151f3f2d89c4804dda0083d912bd9f6a0a9fb7',
    'nvidia/Nemotron-SFT-Competitive-Programming-v2': '778afc98a9e027e10b3cd78020c120e93e142ef2',
    'nvidia/Nemotron-SFT-Agentic-v2': '7c804833427f633ccd53b582dbf02525fd680f78',
    'nvidia/Nemotron-SFT-Safety-v1': None,
    'Salesforce/wikitext': 'b08601e04326c79dfdd32d625aee71d232d685c3',
}

# JSONL sources: name -> (repo, [paths], segments, lines per segment)
JSONL = {
    'if_on': ('nvidia/Nemotron-SFT-Instruction-Following-Chat-v2', ['data/reasoning_on.jsonl'], 24, 12),
    'if_off': ('nvidia/Nemotron-SFT-Instruction-Following-Chat-v2', ['data/reasoning_off.jsonl'], 24, 12),
    'casc1_math': ('nvidia/Nemotron-Cascade-SFT-Stage-1', ['math/math_1.jsonl', 'math/math_4.jsonl'], 12, 8),
    'casc1_science': ('nvidia/Nemotron-Cascade-SFT-Stage-1', ['science.jsonl'], 16, 8),
    'casc1_code': ('nvidia/Nemotron-Cascade-SFT-Stage-1', ['code/code_1.jsonl'], 16, 6),
    'casc2_general': ('nvidia/Nemotron-Cascade-SFT-Stage-2', ['general/general_1.jsonl', 'general/general_3.jsonl'], 12, 8),
    'casc2_if': ('nvidia/Nemotron-Cascade-SFT-Stage-2', ['instruction-following.jsonl'], 16, 8),
    'ko_math': ('nvidia/Nemotron-SFT-Multilingual-v2', ['ultra-v3_math_ko_translated_final.jsonl'], 20, 8),
    'ko_stem': ('nvidia/Nemotron-SFT-Multilingual-v2', ['ultra-v3_stem_ko_translated_postedit_final.jsonl'], 20, 8),
    'ko_code': ('nvidia/Nemotron-SFT-Multilingual-v2', ['ultra-v3_code_ko_translated_final.jsonl'], 20, 6),
    'ja_chat': ('nvidia/Nemotron-SFT-Multilingual-v2', ['ultra-v3_stem_ja_translated_postedit_final.jsonl', 'ultra-v3_math_ja_translated_final.jsonl'], 10, 6),
    'zh_chat': ('nvidia/Nemotron-SFT-Multilingual-v1', ['data/super-v3_stem_zh_translated_postedit_final.jsonl'], 16, 6),
    'es_chat': ('nvidia/Nemotron-SFT-Multilingual-v1', ['data/super-v3_stem_es_translated_postedit_final.jsonl'], 16, 6),
    'finance': ('nvidia/Nemotron-SpecializedDomains-Finance-v1', ['data/train.jsonl'], 24, 4),
    'swe': ('nvidia/Nemotron-SFT-SWE-v2', ['data/agentless.jsonl'], 24, 6),
    'cp_py': ('nvidia/Nemotron-SFT-Competitive-Programming-v2', ['data/competitive_programming_python_00.jsonl'], 20, 6),
    'cp_cpp': ('nvidia/Nemotron-SFT-Competitive-Programming-v2', ['data/competitive_programming_cpp_00.jsonl'], 20, 6),
    'tool': ('nvidia/Nemotron-SFT-Agentic-v2', ['data/tool_calling.jsonl'], 40, 8),
}
# parquet sources: name -> (repo, path, row groups to read (seeded), text column)
PARQUET = {
    'en_web': ('HuggingFaceFW/fineweb-edu', ['sample/10BT/003_00000.parquet', 'sample/10BT/009_00000.parquet'], 1, 'text'),
    'ko_web': ('HuggingFaceFW/fineweb-2', ['data/kor_Hang/test/000_00000.parquet'], 99, 'text'),
    'ja_web': ('HuggingFaceFW/fineweb-2', ['data/jpn_Jpan/test/000_00000.parquet'], 2, 'text'),
    'zh_web': ('HuggingFaceFW/fineweb-2', ['data/cmn_Hani/test/000_00000.parquet'], 2, 'text'),
    'es_web': ('HuggingFaceFW/fineweb-2', ['data/spa_Latn/test/000_00000.parquet'], 2, 'text'),
    'math_web': ('HuggingFaceTB/finemath', ['finemath-4plus/train-00007-of-00064.parquet'], 1, 'text'),
    'code_web': ('codeparrot/github-code-clean', ['data/train-00011-of-00880.parquet', 'data/train-00412-of-00880.parquet'], 1, 'code'),
    'wikitext2': ('Salesforce/wikitext', ['wikitext-2-raw-v1/test-00000-of-00001.parquet'], 99, 'text'),
}


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
        with f_s.open(full, 'rb', block_size=2 ** 21) as f:
            for s in range(segments):
                off = int(size * (s + rng.random() * 0.9) / segments)
                f.seek(off)
                if off:
                    f.readline()
                for _ in range(lines):
                    line = f.readline()
                    if not line:
                        break
                    if len(line) > 4_000_000:
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    rec['_src'] = {'repo': repo, 'revision': REV[repo], 'path': path, 'offset_segment': s}
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
            cols = [column] + [c for c in ('language', 'lang', 'path', 'repo_name', 'id', 'url') if c in pf.schema_arrow.names and c != column]
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
DSML_INVOKE = re.compile(r'<｜DSML｜invoke name="([^"]+)">(.*?)</｜DSML｜invoke>', re.S)
DSML_PARAM = re.compile(r'<｜DSML｜parameter name="([^"]+)" string="(true|false)">(.*?)</｜DSML｜parameter>', re.S)


def split_think(msg):
    """-> (reasoning or None, content) from reasoning_content / reasoning / inline <think>."""
    content = msg.get('content') or ''
    if not isinstance(content, str):
        content = ''.join(p.get('text', '') for p in content if isinstance(p, dict)) if isinstance(content, list) else str(content)
    r = msg.get('reasoning_content') or msg.get('reasoning')
    if not r:
        m = THINK_RE.match(content)
        if m:
            r, content = m.group(1), content[m.end():]
    return (r.strip() if r and r.strip() else None), content


def clean_messages(msgs, think):
    """Normalize roles/fields; drop empty system turns; keep or strip reasoning."""
    out = []
    for m in msgs:
        role = m.get('role')
        if role not in ('system', 'user', 'assistant', 'tool'):
            continue
        if role == 'assistant':
            r, c = split_think(m)
            n = {'role': 'assistant', 'content': c}
            if think and r:
                n['reasoning_content'] = r
            if m.get('tool_calls'):
                calls = []
                for tc in m['tool_calls']:
                    fn = tc.get('function', tc)
                    if 'name' not in fn:
                        continue
                    calls.append({'type': 'function', 'function': {'name': fn['name'], 'arguments': fn.get('arguments', {})}})
                if calls:
                    n['tool_calls'] = calls
            if m.get('dsml'):
                calls = []
                for name, body in DSML_INVOKE.findall(m['dsml']):
                    args = {}
                    for p, is_str, v in DSML_PARAM.findall(body):
                        if is_str == 'true':
                            args[p] = v
                        else:
                            try:
                                args[p] = json.loads(v)
                            except json.JSONDecodeError:
                                args[p] = v
                    calls.append({'type': 'function', 'function': {'name': name, 'arguments': args}})
                if calls:
                    n['tool_calls'] = n.get('tool_calls', []) + calls
            out.append(n)
        else:
            c = m.get('content')
            if isinstance(c, list):
                c = ''.join(p.get('text', '') for p in c if isinstance(p, dict))
            elif c is not None and not isinstance(c, str):
                c = json.dumps(c, ensure_ascii=False)
            c = c or ''
            if role == 'system' and not c.strip():
                continue
            out.append({'role': role, 'content': c})
    return out


def has_reasoning(msgs):
    return any(m.get('role') == 'assistant' and split_think(m)[0] for m in msgs)


def infer_tools(msgs):
    """Tool schemas for records that carry calls but no `tools` list (DS4 prompts)."""
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
            for k, v in args.items():
                kind = ('boolean' if isinstance(v, bool) else 'integer' if isinstance(v, int) else
                        'number' if isinstance(v, float) else 'object' if isinstance(v, dict) else
                        'array' if isinstance(v, list) else 'string')
                props[k] = {'type': kind}
    return [{'type': 'function', 'function': {'name': n, 'description': f'{n} tool',
             'parameters': {'type': 'object', 'properties': p, 'required': sorted(p)}}} for n, p in seen.items()]


class Renderer:
    def __init__(self):
        from transformers import AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(MODEL)

    def chat(self, msgs, tools=None, gen_prompt=False, think=False):
        kw = {}
        if tools:
            kw['tools'] = tools
        if gen_prompt:
            kw['add_generation_prompt'] = True
            kw['enable_thinking'] = think
        return self.tok.apply_chat_template(msgs, tokenize=False, **kw)

    def encode(self, texts):
        return self.tok(texts, add_special_tokens=False)['input_ids']


def load_raw(name):
    p = RAW / name / 'records.jsonl'
    if not p.exists():
        return []
    return [json.loads(l) for l in p.open()]


def text_hash(s):
    return hashlib.sha256(s.encode('utf-8', 'replace')).hexdigest()


# ---------------------------------------------------------------- budgets (tokens)

RAW_BUDGET = {
    # name: (domain, lang, token budget, max tokens per doc)
    'en_web': ('web_en', 'en', 560_000, 16384),
    'ko_web': ('web_ko', 'ko', 420_000, 16384),
    'code_web': ('code_raw', 'code', 300_000, 16384),
    'ja_web': ('web_ja', 'ja', 40_000, 8192),
    'zh_web': ('web_zh', 'zh', 40_000, 8192),
    'es_web': ('web_es', 'es', 40_000, 8192),
    'math_web': ('math_raw', 'en', 150_000, 8192),
}
CHAT_BUDGET = {
    # name: (domain, lang, token budget, both-modes probability, max tokens per sequence)
    'if_on': ('chat_if', 'en', 200_000, 0.25, 8192),
    'if_off': ('chat_if', 'en', 160_000, 0.0, 8192),
    'casc1_math': ('reason_math', 'en', 180_000, 0.25, 16384),
    'casc1_science': ('reason_science', 'en', 110_000, 0.25, 16384),
    'casc1_code': ('reason_code', 'en', 90_000, 0.25, 16384),
    'casc2_general': ('reason_general', 'en', 120_000, 0.25, 16384),
    'casc2_if': ('chat_if', 'en', 80_000, 0.25, 8192),
    'ko_math': ('chat_ko', 'ko', 150_000, 0.25, 8192),
    'ko_stem': ('chat_ko', 'ko', 150_000, 0.25, 8192),
    'ko_code': ('chat_ko', 'ko', 110_000, 0.25, 8192),
    'ja_chat': ('chat_ja', 'ja', 35_000, 0.25, 8192),
    'zh_chat': ('chat_zh', 'zh', 35_000, 0.25, 8192),
    'es_chat': ('chat_es', 'es', 35_000, 0.25, 8192),
    'tool': ('agent_tool', 'en', 250_000, 0.25, 16384),
    'swe': ('agent_swe', 'en', 150_000, 0.0, 16384),
    'cp_py': ('code_chat', 'code', 100_000, 0.25, 16384),
    'cp_cpp': ('code_chat', 'code', 90_000, 0.25, 16384),
    'finance': ('long_finance', 'en', 220_000, 0.0, 32768),
    'ds4': ('ds4_prompts', 'mixed', 200_000, 0.0, 32768),
}

# tokens each bucket may spend on sequences longer than LONG (the rest is <= LONG):
# targets ~12-15% of all tokens in 4K-32K sequences (finance and SWE carry most of it)
LONG = 4096
LONG_ALLOW = {'finance': 200_000, 'swe': 70_000, 'en_web': 60_000, 'ko_web': 40_000, 'code_web': 60_000,
              'math_web': 10_000, 'ja_web': 6_000, 'zh_web': 6_000, 'es_web': 6_000, 'ds4': 50_000,
              'if_on': 8_000, 'if_off': 8_000, 'casc2_if': 8_000, 'ko_math': 12_000, 'ko_stem': 12_000, 'ko_code': 12_000,
              'ja_chat': 5_000, 'zh_chat': 5_000, 'es_chat': 5_000}
LONG_DEFAULT = 20_000

# held-out picks: (case name, source, kind, target tokens)
HELDOUT = [
    ('wikitext2_a', 'wikitext2', 'stream', 2048), ('wikitext2_b', 'wikitext2', 'stream', 2048),
    ('en_web', 'en_web', 'doc', 2048), ('ko_web_a', 'ko_web', 'doc', 2048), ('ko_web_b', 'ko_web', 'doc', 2048),
    ('ja_web', 'ja_web', 'doc', 1536), ('zh_web', 'zh_web', 'doc', 1536), ('es_web', 'es_web', 'doc', 1536),
    ('math_web', 'math_web', 'doc', 2048), ('code_python', 'code_web', 'doc_py', 2048), ('code_c', 'code_web', 'doc_c', 2048),
    ('ko_chat_nothink', 'ko_stem', 'chat_nothink', 3072), ('ko_math_think', 'ko_math', 'chat_think', 3072),
    ('en_reason_think', 'casc2_general', 'chat_think', 3072), ('math_think', 'casc1_math', 'chat_think', 3072),
    ('if_chat_nothink', 'if_off', 'chat_nothink', 3072), ('tool_call', 'tool', 'chat_think', 4096),
    ('swe_agentless', 'swe', 'chat_nothink', 4096), ('code_chat_think', 'cp_py', 'chat_think', 4096),
    ('ja_chat', 'ja_chat', 'chat_nothink', 3072),
    ('long_en_doc', 'en_web', 'long_doc', 8192), ('long_finance', 'finance', 'chat_nothink_long', 8192),
]


def build(v0=False):
    rng = random.Random(SEED)
    R = Renderer()
    samples, used_hashes = [], set()
    held = []
    t0 = time.monotonic()
    scale = 0.1 if v0 else 1.0

    # ---- held-out first (disjoint documents are reserved before calibration picks)
    reserved = set()
    pools = {}

    def pool(name):
        if name not in pools:
            recs = load_raw(name) if name != 'ds4' else []
            r2 = random.Random(f'{SEED}:pool:{name}')
            r2.shuffle(recs)
            pools[name] = recs
        return pools[name]

    if not v0:
        for case, src, kind, target in HELDOUT:
            recs = pool(src)
            if not recs:
                log('heldout source missing', case, src)
                continue
            if kind == 'stream':
                ordered = load_raw(src)   # original row order (HF perplexity recipe: '\n\n'.join(test['text']))
                text = '\n\n'.join(r['text'] for r in ordered)
                ids = R.encode([text])[0]
                start = 0 if case.endswith('_a') else len(ids) // 2
                piece = ids[start:start + target]
                held.append({'case': case, 'domain': 'wikitext2', 'source': recs[0]['_src'], 'format': 'raw', 'mode': None,
                             'ids': piece, 'note': f'wikitext-2-raw test contiguous tokens [{start}, {start + len(piece)})'})
                reserved.add(f'wikitext2:{case}')
                continue
            chosen = None
            for j, r in enumerate(recs):
                key = (src, j)
                if key in reserved:
                    continue
                if kind.startswith('doc') or kind == 'long_doc':
                    if kind == 'doc_py' and not str(r.get('path', '')).endswith('.py'):
                        continue
                    if kind == 'doc_c' and not str(r.get('path', '')).endswith(('.c', '.h')):
                        continue
                    ids = R.encode([r['text']])[0]
                    lo = target * 0.6 if kind != 'long_doc' else target * 0.9
                    if len(ids) < lo:
                        continue
                    chosen = (j, ids[:target], 'raw', None, r)
                    break
                else:
                    think = 'think' in kind and 'nothink' not in kind
                    msgs = clean_messages(r.get('messages', []), think)
                    if not msgs or msgs[-1]['role'] != 'assistant':
                        continue
                    if think and not has_reasoning(r.get('messages', [])):
                        continue
                    tools = r.get('tools') or None
                    try:
                        text = R.chat(msgs, tools=tools)
                    except Exception:
                        continue
                    ids = R.encode([text])[0]
                    lo = 512 if 'long' not in kind else target * 0.9
                    if len(ids) < lo:
                        continue
                    # teacher-forced measurement: a long record is cut to the target length
                    chosen = (j, ids[:target], 'chat', 'think' if think else 'nothink', r)
                    break
            if chosen is None:
                log('heldout: no suitable record', case)
                continue
            j, ids, fmt, mode, r = chosen
            reserved.add((src, j))
            held.append({'case': case, 'domain': src, 'source': r['_src'], 'format': fmt, 'mode': mode, 'ids': ids,
                         'doc_sha256': text_hash(r.get('text') or json.dumps(r.get('messages'), ensure_ascii=False))})
        log('heldout cases', len(held), 'tokens', sum(len(h['ids']) for h in held))
    held_hashes = {h.get('doc_sha256') for h in held}

    long_used = {}
    overlap_report = {}

    def add(ids, meta):
        if not ids:
            return 0
        ids = ids[:MAX_LEN]
        h = hashlib.sha256(np.asarray(ids, np.int32).tobytes()).hexdigest()
        if h in used_hashes:
            return 0
        used_hashes.add(h)
        meta['tokens'] = len(ids)
        samples.append((ids, meta))
        return len(ids)

    # ---- raw text buckets
    for name, (domain, lang, budget, cap) in RAW_BUDGET.items():
        budget = int(budget * scale)
        recs = pool(name)
        got = 0
        for j, r in enumerate(recs):
            if got >= budget:
                break
            if (name, j) in reserved:
                continue
            text = r['text']
            if not text or len(text) < 200 or text_hash(text) in held_hashes:
                continue
            ids = R.encode([text])[0]
            if len(ids) < 64:
                continue
            n = min(len(ids), cap)
            if n > LONG:
                if long_used.get(name, 0) + n > LONG_ALLOW.get(name, LONG_DEFAULT):
                    n = LONG
                else:
                    long_used[name] = long_used.get(name, 0) + n
            got += add(ids[:n], {'domain': domain, 'source': r['_src'], 'lang': lang, 'format': 'raw', 'mode': None,
                                  'truncated': len(ids) > n})
        log('raw', name, got, 'tokens', f'{time.monotonic() - t0:.0f}s')

    # ---- chat buckets
    for name, (domain, lang, budget, both_p, max_len) in CHAT_BUDGET.items():
        budget = int(budget * scale)
        if name == 'ds4':
            continue
        recs = pool(name)
        got = 0
        for j, r in enumerate(recs):
            if got >= budget:
                break
            if (name, j) in reserved:
                continue
            raw_msgs = r.get('messages', [])
            reasoning = has_reasoning(raw_msgs)
            if name == 'if_off':
                modes = ['nothink']
            elif name == 'swe' or name == 'finance':
                modes = ['think'] if reasoning and rng.random() < 0.5 else ['nothink']
            elif reasoning:
                u = rng.random()
                modes = ['think', 'nothink'] if u < both_p else (['think'] if u < both_p + (1 - both_p) * 0.5 else ['nothink'])
            else:
                modes = ['nothink']
            tools = r.get('tools') or None
            for mode in modes:
                msgs = clean_messages(raw_msgs, mode == 'think')
                if not msgs or not any(m['role'] == 'assistant' for m in msgs):
                    continue
                if tools is None and any(m.get('tool_calls') for m in msgs):
                    tools = infer_tools(msgs)
                try:
                    text = R.chat(msgs, tools=tools)
                except Exception as e:
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
                got += add(ids, {'domain': domain, 'source': r['_src'], 'lang': lang, 'format': 'chat', 'mode': mode,
                                 'truncated': len(ids) > MAX_LEN, 'tools': bool(tools)})
        log('chat', name, got, 'tokens', f'{time.monotonic() - t0:.0f}s')

    # ---- DS4 imatrix prompts re-rendered with the A.X-K2 template (local file)
    name = 'ds4'
    domain, lang, budget, _, _ = CHAT_BUDGET[name]
    budget = int(budget * scale)
    recs = [json.loads(l) for l in open(DS4_PROMPTS)]
    random.Random(f'{SEED}:ds4').shuffle(recs)
    got = 0
    for r in recs:
        if got >= budget:
            break
        think = r.get('mode') == 'think'
        msgs = clean_messages(r['messages'], think)
        if not msgs:
            continue
        tools = infer_tools(msgs) if any(m.get('tool_calls') for m in msgs) else None
        gen = msgs[-1]['role'] != 'assistant'
        try:
            text = R.chat(msgs, tools=tools, gen_prompt=gen, think=think)
        except Exception:
            continue
        ids = R.encode([text])[0]
        if len(ids) < 32:
            continue
        if len(ids) > LONG:
            if long_used.get(name, 0) + len(ids) > LONG_ALLOW.get(name, LONG_DEFAULT):
                continue
            long_used[name] = long_used.get(name, 0) + len(ids)
        got += add(ids, {'domain': domain, 'source': {'repo': 'Baekpica/ds4-dfm-rs', 'path': 'gguf-tools/imatrix/dataset/prompts.jsonl',
                                                       'id': r['id'], 'category': r['category']},
                         'lang': lang, 'format': 'chat', 'mode': 'think' if think else 'nothink', 'generation_prompt': gen,
                         'tools': bool(tools)})
    log('chat', name, got, 'tokens')

    # ---- held-out disjointness: drop calibration rows that share document text with a
    # held-out case. A 64-token window counts as boilerplate (chat/tool system prompts,
    # task instructions) when it occurs in >= 3 calibration rows; any other shared window
    # removes the calibration row (Finance-v1 asks several questions over one 10-K passage).
    if held:
        W = 64

        def win(x, k):
            return hashlib.blake2b(np.asarray(x[k:k + W], np.int32).tobytes(), digest_size=8).digest()
        hw = {}
        for hc, hcase in enumerate(held):
            x = hcase['ids']
            for k in range(0, len(x) - W + 1):
                hw.setdefault(win(x, k), set()).add(hc)
        rows_per = {}
        for r, (ids, _) in enumerate(samples):
            for k in range(0, len(ids) - W + 1):
                d = win(ids, k)
                if d in hw:
                    rows_per.setdefault(d, set()).add(r)
        drop = set()
        for d, rs in rows_per.items():
            if len(rs) < 3:
                drop |= rs
        dropped = [(samples[r][1]['domain'], samples[r][1]['tokens']) for r in sorted(drop)]
        log('heldout overlap: dropped calibration rows', dropped,
            'boilerplate windows', sum(1 for rs in rows_per.values() if len(rs) >= 3))
        samples = [sm for r, sm in enumerate(samples) if r not in drop]
        overlap_report.update({'window_tokens': W, 'boilerplate_min_rows': 3, 'dropped_rows': dropped,
                               'boilerplate_windows': sum(1 for rs in rows_per.values() if len(rs) >= 3)})

    # ---- write
    order = list(range(len(samples)))
    random.Random(f'{SEED}:order').shuffle(order)
    samples = [samples[i] for i in order]
    write_set(ROOT / ('calib_v0' if v0 else 'calib'), samples, extra={'heldout_overlap_filter': overlap_report} if not v0 else None)
    if not v0:
        hs = [(h['ids'], {k: v for k, v in h.items() if k != 'ids'}) for h in held]
        write_set(ROOT / 'heldout', hs)
    log('done', f'{time.monotonic() - t0:.0f}s')


def write_set(stem, samples, extra=None):
    import torch
    from safetensors.torch import save_file
    lens = [len(ids) for ids, _ in samples]
    flat = np.concatenate([np.asarray(ids, np.int32) for ids, _ in samples])
    if flat.max() >= 163840 or flat.min() < 0 or min(lens) <= 0:
        raise ValueError('token id range / empty row')
    offsets = np.zeros(len(samples) + 1, np.int64)
    offsets[1:] = np.cumsum(lens)
    assert offsets[-1] == len(flat)
    rows = {hashlib.sha256(flat[offsets[i]:offsets[i + 1]].tobytes()).hexdigest() for i in range(len(samples))}
    if len(rows) != len(samples):
        raise ValueError('duplicate rows')
    path = stem.with_suffix('.safetensors')
    save_file({'input_ids': torch.from_numpy(flat), 'offsets': torch.from_numpy(offsets)}, str(path),
              metadata={'tokenizer': 'skt/A.X-K2@d0805cfd1fc647b4078c0d9f232c80d34ee0c893', 'bos_prepended': 'false',
                        'seed': str(SEED)})
    metas = []
    for i, (ids, m) in enumerate(samples):
        m = dict(m)
        m['i'] = i
        m['tokens'] = len(ids)
        metas.append(m)
    by_domain, by_lang, by_mode = {}, {}, {}
    for m in metas:
        by_domain[m.get('domain')] = by_domain.get(m.get('domain'), 0) + m['tokens']
        by_lang[m.get('lang')] = by_lang.get(m.get('lang'), 0) + m['tokens']
        key = f"{m.get('format')}:{m.get('mode')}"
        by_mode[key] = by_mode.get(key, 0) + m['tokens']
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
              'tokens_by_lang': by_lang, 'tokens_by_format_mode': by_mode,
              'safetensors_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
              'bos_prepended': False, 'tokenizer': 'skt/A.X-K2@d0805cfd1fc647b4078c0d9f232c80d34ee0c893',
              'chat_template': 'A.X-K2 chat_template.jinja via tokenizer.apply_chat_template'}
    if extra:
        totals.update(extra)
    stem.with_name(stem.name + '_manifest.json').write_text(json.dumps({'samples': metas, 'totals': totals}, ensure_ascii=False, indent=0) + '\n')
    log('wrote', path, json.dumps({k: totals[k] for k in ('sequences', 'tokens', 'length', 'tokens_ge_4096')}))


# ---------------------------------------------------------------- held-out 2

# Independent second held-out set (held-out 1 was used to choose between release
# recipes). (family, source pools, kind, cases, min tokens, max tokens); each case
# draws its own target length in [min, max] and is cut to it (teacher-forced
# measurement only). Wikitext-2 windows avoid the two used by held-out 1.
HELDOUT2 = [
    ('wikitext2', ['wikitext2'], 'stream', 6, 2048, 2048),
    ('en_web', ['en_web'], 'doc', 5, 1024, 3072),
    ('en_web_long', ['en_web'], 'doc', 1, 8192, 10240),
    ('ko_web', ['ko_web'], 'doc', 5, 1024, 3072),
    ('ko_web_long', ['ko_web'], 'doc', 1, 8192, 10240),
    ('ko_chat_nothink', ['ko_stem', 'ko_math', 'ko_code'], 'chat_nothink', 5, 768, 3072),
    ('ko_think', ['ko_math', 'ko_stem'], 'chat_think', 5, 1024, 3072),
    ('en_reason_think', ['casc2_general', 'casc1_science'], 'chat_think', 5, 1024, 3072),
    ('math_think', ['casc1_math'], 'chat_think', 5, 1024, 3072),
    ('if_chat', ['if_off', 'if_on', 'casc2_if'], 'chat_mixed', 5, 768, 3072),
    ('tool_call', ['tool'], 'chat_mixed', 5, 1024, 3072),
    ('swe_agentless', ['swe'], 'chat_nothink', 4, 1024, 3072),
    ('swe_long', ['swe'], 'chat_nothink', 1, 8192, 10240),
    ('competitive_prog', ['cp_py', 'cp_cpp'], 'chat_think', 5, 1024, 3072),
    ('code_python', ['code_web'], 'doc_py', 4, 1024, 3072),
    ('code_c', ['code_web'], 'doc_c', 4, 1024, 3072),
    ('ja_web', ['ja_web'], 'doc', 4, 768, 3072),
    ('zh_web', ['zh_web'], 'doc', 4, 768, 3072),
    ('es_web', ['es_web'], 'doc', 4, 768, 3072),
    ('finance', ['finance'], 'chat_nothink', 4, 2048, 3072),
    ('finance_long', ['finance'], 'chat_nothink', 1, 8192, 10240),
]
HELDOUT2_CAP = 16384
FROZEN_CALIB = ROOT / 'frozen' / 'calib-b244872e.safetensors'
FROZEN_HELDOUT = ROOT / 'frozen' / 'heldout-81681f74.safetensors'


def build_heldout2():
    from safetensors.numpy import load_file
    t0 = time.monotonic()
    R = Renderer()
    W = 64

    def win(x, k):
        return hashlib.blake2b(np.asarray(x[k:k + W], np.int32).tobytes(), digest_size=8).digest()

    # windows of the frozen calibration set: rows per window (capped at 3 = boilerplate)
    calib = load_file(str(FROZEN_CALIB))
    ci, co = calib['input_ids'], calib['offsets']
    crow = {}
    for r in range(len(co) - 1):
        x = ci[co[r]:co[r + 1]]
        seen = set()
        for k in range(0, len(x) - W + 1):
            d = win(x, k)
            if d in seen:
                continue
            seen.add(d)
            c = crow.get(d, 0)
            if c < 3:
                crow[d] = c + 1
    boiler = {d for d, c in crow.items() if c >= 3}
    used = {d for d, c in crow.items() if c < 3}
    h1 = load_file(str(FROZEN_HELDOUT))
    hi_, ho_ = h1['input_ids'], h1['offsets']
    for r in range(len(ho_) - 1):
        x = hi_[ho_[r]:ho_[r + 1]]
        for k in range(0, len(x) - W + 1):
            d = win(x, k)
            if d not in boiler:
                used.add(d)
    log('heldout2: frozen windows', len(crow), 'boilerplate', len(boiler), f'{time.monotonic() - t0:.0f}s')

    formatting = {}

    def is_formatting(ids, k, d):
        # windows that are pure layout (empty markdown table cells, separators) are not document text
        if d not in formatting:
            text = R.tok.decode(list(map(int, ids[k:k + W])))
            formatting[d] = sum(ch.isalnum() for ch in text) < 16
        return formatting[d]

    pool_boiler = {}

    def source_boiler(name):
        # Instruction text repeated across a source's own records (Finance-v1's task prompt and
        # its variants) is boilerplate even when few rows of that source are in calibration: a
        # window in >= 8 distinct pool records spans several byte-offset segments, while one
        # 10-K passage is only shared by the consecutive records of one segment (<= 4).
        if name not in pool_boiler:
            cnt = {}
            for r in load_raw(name):
                msgs = clean_messages(r.get('messages', []), False)
                try:
                    ids = R.encode([R.chat(msgs)])[0][:HELDOUT2_CAP]
                except Exception:
                    continue
                for d in {win(ids, k) for k in range(0, len(ids) - W + 1)}:
                    cnt[d] = cnt.get(d, 0) + 1
            pool_boiler[name] = {d for d, c in cnt.items() if c >= 8}
            log('heldout2: pool boilerplate windows', name, len(pool_boiler[name]))
        return pool_boiler[name]

    extra_boiler = set()

    def clean(ids):
        """-> (ok, windows): no document window shared with calib, held-out 1 or chosen cases."""
        ws = []
        for k in range(0, len(ids) - W + 1):
            d = win(ids, k)
            if d in boiler or d in extra_boiler:
                continue
            if d in used:
                if is_formatting(ids, k, d):
                    continue
                return False, None
            ws.append(d)
        return True, ws

    rng = random.Random(f'{SEED}:heldout2')
    pools = {}

    def pool(name):
        if name not in pools:
            recs = load_raw(name)
            random.Random(f'{SEED}:heldout2:pool:{name}').shuffle(recs)
            pools[name] = [recs, 0]
        return pools[name]

    cases, rejected = [], {}
    wt_ids = None
    for family, sources, kind, n, lo, hi in HELDOUT2:
        got = 0
        if kind == 'stream':
            ordered = load_raw('wikitext2')
            wt_ids = R.encode(['\n\n'.join(r['text'] for r in ordered)])[0]
            L = len(wt_ids)
            used_ranges = [(0, 2048), (L // 2, L // 2 + 2048)]      # held-out 1
            starts = [int(L * f) for f in (0.07, 0.21, 0.35, 0.62, 0.76, 0.90)]
            for st in starts:
                if any(a < st + hi and st < b for a, b in used_ranges):
                    continue
                piece = wt_ids[st:st + hi]
                ok, ws = clean(piece)
                if not ok:
                    rejected[family] = rejected.get(family, 0) + 1
                    continue
                used.update(ws)
                cases.append((piece, {'case': f'{family}_{got}', 'domain': 'wikitext2', 'source': ordered[0]['_src'],
                                      'format': 'raw', 'mode': None,
                                      'note': f'wikitext-2-raw test contiguous tokens [{st}, {st + len(piece)})'}))
                got += 1
            log('heldout2', family, got)
            continue
        extra_boiler = source_boiler('finance') if 'finance' in sources else set()
        si = 0
        attempts = 0
        while got < n and attempts < 4000:
            attempts += 1
            src = sources[si % len(sources)]
            si += 1
            st = pool(src)
            recs, pos = st
            if pos >= len(recs):
                if all(pool(s)[1] >= len(pool(s)[0]) for s in sources):
                    break
                continue
            r = recs[pos]
            st[1] = pos + 1
            target = rng.randint(lo, hi)
            if kind.startswith('doc'):
                path = str(r.get('path', ''))
                if kind == 'doc_py' and not path.endswith('.py'):
                    continue
                if kind == 'doc_c' and not path.endswith(('.c', '.h')):
                    continue
                text = r.get('text') or ''
                if len(text) < 200:
                    continue
                ids = R.encode([text])[0]
                if len(ids) < lo:
                    continue
                ids = ids[:min(target, HELDOUT2_CAP)]
                fmt, mode = 'raw', None
                digest = text_hash(text)
            else:
                raw_msgs = r.get('messages', [])
                reasoning = has_reasoning(raw_msgs)
                if kind == 'chat_think':
                    think = True
                elif kind == 'chat_nothink':
                    think = False
                else:
                    think = reasoning and got % 2 == 1
                if think and not reasoning:
                    continue
                msgs = clean_messages(raw_msgs, think)
                if not msgs or not any(m['role'] == 'assistant' for m in msgs):
                    continue
                tools = r.get('tools') or None
                if tools is None and any(m.get('tool_calls') for m in msgs):
                    tools = infer_tools(msgs)
                if family == 'tool_call' and not tools:
                    continue
                try:
                    text = R.chat(msgs, tools=tools)
                except Exception:
                    continue
                ids = R.encode([text])[0]
                if len(ids) < lo:
                    continue
                ids = ids[:min(target, HELDOUT2_CAP)]
                fmt, mode = 'chat', 'think' if think else 'nothink'
                digest = text_hash(json.dumps(raw_msgs, ensure_ascii=False))
            ok, ws = clean(ids)
            if not ok:
                rejected[family] = rejected.get(family, 0) + 1
                continue
            used.update(ws)
            meta = {'case': f'{family}_{got}', 'domain': family, 'source': r['_src'], 'format': fmt, 'mode': mode,
                    'doc_sha256': digest}
            if fmt == 'chat':
                meta['tools'] = bool(tools)
            cases.append((ids, meta))
            got += 1
        log('heldout2', family, got, 'cases', 'rejected (overlap)', rejected.get(family, 0))
    write_set(ROOT / 'heldout2', cases, extra={'overlap_check': {
        'window_tokens': W, 'against': [FROZEN_CALIB.name, FROZEN_HELDOUT.name],
        'boilerplate_rule': 'window in >= 3 frozen-calibration rows',
        'formatting_rule': 'a shared window with < 16 alphanumeric characters (layout only) is ignored',
        'source_boilerplate_rule': 'finance only: window in >= 8 distinct Finance-v1 pool records (task prompt variants)',
        'rejected_candidates': rejected,
        'cap_tokens': HELDOUT2_CAP}})
    log('heldout2 done', f'{time.monotonic() - t0:.0f}s')


def main():
    global ROOT, RAW, MODEL, DS4_PROMPTS, FROZEN_CALIB, FROZEN_HELDOUT
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=['fetch', 'build', 'build-v0', 'build-heldout2', 'all'])
    p.add_argument('--only', nargs='*')
    p.add_argument('--out', type=Path, default=ROOT, help='calibration directory (downloads under OUT/raw)')
    p.add_argument('--model', default=MODEL, help='A.X-K2 directory (tokenizer, chat template)')
    p.add_argument('--ds4-prompts', type=Path, default=DS4_PROMPTS)
    a = p.parse_args()
    ROOT, RAW, MODEL, DS4_PROMPTS = a.out, a.out / 'raw', a.model, a.ds4_prompts
    FROZEN_CALIB = ROOT / 'frozen' / FROZEN_CALIB.name
    FROZEN_HELDOUT = ROOT / 'frozen' / FROZEN_HELDOUT.name
    if a.stage in ('fetch', 'all'):
        fetch(a.only)
    if a.stage == 'build-v0':
        build(v0=True)
    if a.stage in ('build', 'all'):
        build()
    if a.stage == 'build-heldout2':
        build_heldout2()


if __name__ == '__main__':
    main()
