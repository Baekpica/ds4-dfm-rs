#!/usr/bin/env python3
"""Inkling-2-Small calibration and held-out token sets for the hVQ imatrix / Hessian collector.

  fetch         stream bounded samples of every source into calib/raw/<source>/ (data only)
  build         render chat records natively with Inkling-2's chat template (chat_template.jinja
                through transformers' apply_chat_template: thinking effort header, reasoning as
                <|content_thinking|> blocks, JSON tool calls), tokenize with the Inkling-2
                tokenizer, reserve the held-out set first, budget by domain and length, drop
                calibration rows that share a 64-token window with a held-out case, and write
                calib(.safetensors|_manifest.json) and heldout(.safetensors|_manifest.json):
                `input_ids` int32 [N] + `offsets` int64 [n+1] and a per-row JSON manifest
  check         id range, row starts, held-out / calibration window disjointness, image
                placeholders, decoded samples per domain (calib/samples_decoded.txt)
  reprefix      rewrite calib/heldout for another --raw-prefix without refetching or re-tokenizing
                (the selection does not depend on the prefix: the files equal a `build` with it)
  build-images  bounded fetch of four pinned image datasets (photos, receipts, charts, screenshots)
                and images(.safetensors|_manifest.json) + images_heldout.*: one user turn (image +
                question) and the dataset's answer as a non-thinking assistant turn, the image
                placeholder expanded by the Inkling-2 processor, pixel_values as it produces them

Port of build_calib_ds41.py (DeepSeek-V4.1, --mix v1) to Inkling-2-Small: same structure (seeded,
pinned revisions, bounded sampling of big JSONL / parquet by byte offsets / row groups, render ->
tokenize -> budget -> filter -> write, held-out reserved first + 64-token-window overlap check),
same sources, revisions, seed, domains, budgets and held-out cases. What the model changes:
  * chat rows have no BOS: they start with <|message_system|> (tool_declare when tools are
    declared, else a system prompt or the `Thinking effort level: X` header);
    think rows use reasoning_effort "high" (0.9), non-think rows "none" (0);
  * every assistant turn whose source carries reasoning keeps it in think renders (the template
    renders reasoning_content on every assistant message; there is no earlier-turn drop as in
    DeepSeek's encoder); non-think renders drop all reasoning; no reasoning is synthesized;
  * tool calls are OpenAI-style with parsed-dict arguments (rendered as <|content_invoke_tool_json|>),
    tool results are role "tool" with the called function's `name`; DS4 records' DSML is parsed
    back into calls;
  * raw-text rows (web, code, math, wikitext, long documents) start with --raw-prefix
    (default begin_of_text = <|begin_of_text|> 200028, or none); selection, budgets and the overlap
    filter are computed with the prefix, so `reprefix` switches it without changing the selection;
  * safetensors files are written with sorted metadata (byte-identical on re-runs).
Downloaded bytes are parsed as JSON / parquet / image data only; run with `python3 -I`.
CPU only: CUDA is hidden and the process pins itself to --threads CPUs (default 32).
"""
import os

os.environ['CUDA_VISIBLE_DEVICES'] = ''          # CPU only: never touch a GPU
for _v in ('OMP_NUM_THREADS', 'RAYON_NUM_THREADS', 'RAYON_RS_NUM_CPUS', 'TOKIO_WORKER_THREADS'):
    os.environ.setdefault(_v, '32')
for _v in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):   # numpy's BLAS is not used: no idle pool
    os.environ.setdefault(_v, '1')
os.environ.setdefault('HF_XET_HIGH_PERFORMANCE', '0')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'true')

import argparse
from concurrent.futures import ThreadPoolExecutor
import copy
import hashlib
import io
import json
from pathlib import Path
import random
import re
import struct
import time

import numpy as np

ROOT = Path('/work/inkling2/calib')
RAW = ROOT / 'raw'
MODEL = Path('/work/inkling2/src-small')
MODEL_REPO = 'thinkingmachines/Inkling-2-Small'
MODEL_REV = '086a047e8b7ff2ac902dd52355328ddb66b16914'   # src-small/.cache/huggingface (download metadata)
DS4_PROMPTS = Path('/ds4-dfm-rs/gguf-tools/imatrix/dataset/prompts.jsonl')
DS4_REV = '32a4168332615a4d3c9b01b579623197caf7b6c8'   # as DS41; prompts.jsonl is unchanged since (sha256 266fa7d4...)
SEED = 20261022                                        # DS41's: the same documents are fetched and pooled
MAX_LEN = 32768
VOCAB = 201024                                         # embedding rows (padded)
REAL_VOCAB = 200058                                    # text_config.unpadded_vocab_size: every id must be below
EOS_ID = 200006                                        # <|content_model_end_sampling|>, ends every assistant turn
MSG_SYSTEM_ID = 200002                                 # <|message_system|>: first token of every chat row
EOT_ID = 199999                                        # <|endoftext|>
IMAGE_ID = 200054                                      # <|unused_200054|>, one per image token
RAW_PREFIXES = {'begin_of_text': 200028, 'none': None}
RAW_PREFIX = 'begin_of_text'
CANON_PREFIX = RAW_PREFIXES['begin_of_text']           # raw rows are selected / filtered with this prefix
THINK_EFFORT, NOTHINK_EFFORT = 'high', 'none'

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


def limit_threads(n):
    """Pin this process (every existing thread, and so every pool it starts later) to n CPUs."""
    try:
        cpus = sorted(os.sched_getaffinity(0))
        if len(cpus) > n:
            for tid in os.listdir('/proc/self/task'):
                try:
                    os.sched_setaffinity(int(tid), cpus[:n])
                except OSError:
                    pass
    except (AttributeError, OSError):
        pass
    for v in ('OMP_NUM_THREADS', 'RAYON_NUM_THREADS', 'RAYON_RS_NUM_CPUS', 'TOKIO_WORKER_THREADS'):
        os.environ[v] = str(n)


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
    with ThreadPoolExecutor(max(1, len(jobs))) as ex:
        futs = {ex.submit(fn, n, s): n for fn, n, s in jobs}
        for fu in futs:
            n = futs[fu]
            try:
                log('fetched', *fu.result())
            except Exception as e:  # a slow or gated source is dropped, not fatal
                log('FETCH FAILED', n, repr(e)[:300])


# ---------------------------------------------------------------- rendering

THINK_RE = re.compile(r'^\s*<think>(.*?)</think>\s*', re.S)
# DS4 (V4) DSML as stored in prompts.jsonl `dsml` fields: parsed back into OpenAI-style tool calls
DSML_INVOKE = re.compile(r'<｜DSML｜invoke name="([^"]+)">(.*?)</｜DSML｜invoke>', re.S)
DSML_PARAM = re.compile(r'<｜DSML｜parameter name="([^"]+)" string="(true|false)">(.*?)</｜DSML｜parameter>', re.S)
# text that would tokenize to Inkling-2 control tokens if it appeared inside document / message text:
# the content of every added token of tokenizer.json (all of the form <|...|>), filled by load_specials()
SPECIAL_STRINGS = ()
_SPECIAL_RE = re.compile(r'(?!)')


def load_specials():
    global SPECIAL_STRINGS, _SPECIAL_RE
    added = json.loads((MODEL / 'tokenizer.json').read_text())['added_tokens']
    SPECIAL_STRINGS = tuple(sorted(a['content'] for a in added))
    if not all(s.startswith('<|') for s in SPECIAL_STRINGS):
        raise SystemExit('unexpected added-token form (has_special() assumes <|...|>)')
    _SPECIAL_RE = re.compile('|'.join(re.escape(s) for s in SPECIAL_STRINGS))


def has_special(s):
    return isinstance(s, str) and '<|' in s and _SPECIAL_RE.search(s) is not None


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
    """Source messages -> Inkling-2 template messages (OpenAI style); None if unusable.

    Think renders keep `reasoning_content` on every assistant turn whose source carries reasoning (the
    template renders all of them: unlike DeepSeek's encoder nothing is dropped before the last user
    turn); non-think renders drop it. Tool-call arguments become parsed objects (the template refuses
    JSON strings) and each tool result is named after its call (`tool_call_id`, else position)."""
    out = []
    prev_calls, k = [], 0
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
                if args is None or args == '':
                    args = {}
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except json.JSONDecodeError:
                        return None
                if not isinstance(args, dict) or not isinstance(fn['name'], str):
                    return None
                call = {'type': 'function', 'function': {'name': fn['name'], 'arguments': args}}
                if tc.get('id'):
                    call['id'] = tc['id']
                calls.append(call)
            calls += parse_dsml_v4(m.get('dsml'))
            if any(has_special(json.dumps(x['function'], ensure_ascii=False)) for x in calls):
                return None
            if calls:
                n['tool_calls'] = calls
            prev_calls, k = calls, 0
            out.append(n)
        else:
            c = as_text(m.get('content'))
            if has_special(c):
                return None
            if role == 'system' and not c.strip():
                continue
            n = {'role': role, 'content': c}
            if role == 'tool':
                name = m.get('name') if isinstance(m.get('name'), str) and m.get('name') else None
                tid = m.get('tool_call_id')
                if name is None and tid:
                    name = next((x['function']['name'] for x in prev_calls if x.get('id') == tid), None)
                if name is None and k < len(prev_calls):
                    name = prev_calls[k]['function']['name']
                k += 1
                if name:
                    n['name'] = name
                if tid:
                    n['tool_call_id'] = tid
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
        if isinstance(fn.get('parameters'), str):
            try:
                fn = dict(fn, parameters=json.loads(fn['parameters']))
            except json.JSONDecodeError:
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
        from tokenizers import Tokenizer
        from transformers import AutoTokenizer
        self.hf = AutoTokenizer.from_pretrained(str(MODEL))          # chat_template.jinja
        self.tok = Tokenizer.from_file(str(MODEL / 'tokenizer.json'))  # same tokenizer, batched
        load_specials()
        for s, i in (('<|content_model_end_sampling|>', EOS_ID), ('<|message_system|>', MSG_SYSTEM_ID),
                     ('<|begin_of_text|>', CANON_PREFIX), ('<|endoftext|>', EOT_ID), ('<|unused_200054|>', IMAGE_ID)):
            if self.tok.token_to_id(s) != i:
                raise SystemExit(f'{MODEL}: {s} is not id {i}')

    def chat(self, msgs, think, tools=None):
        """Native Inkling-2 rendering: reasoning effort "high" (think) or "none"; tools as tool_declare."""
        if tools and has_special(json.dumps(tools, ensure_ascii=False)):
            raise ValueError('control-token text in a tool schema')
        return self.hf.apply_chat_template(copy.deepcopy(msgs), tools=tools or None, tokenize=False,
                                           reasoning_effort=THINK_EFFORT if think else NOTHINK_EFFORT)

    def encode(self, texts):
        return [e.ids for e in self.tok.encode_batch(list(texts), add_special_tokens=False)]

    def raw(self, texts):
        """Raw text rows are selected with the canonical <|begin_of_text|> prefix (stripped at write for none)."""
        return [[CANON_PREFIX] + ids for ids in self.encode(texts)]

    def decode(self, ids):
        return self.tok.decode(list(map(int, ids)), skip_special_tokens=False)


def load_raw(name):
    p = RAW / name / 'records.jsonl'
    return [json.loads(l) for l in p.open()] if p.exists() else []


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


# ---------------------------------------------------------------- budgets (tokens; raw rows incl. the prefix)

RAW_BUDGET = {
    # name: (domain, lang, token budget, max tokens per doc incl. the raw prefix)
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

# held-out picks: (case name, source, kind, target tokens incl. the raw prefix; wikitext windows are prefix + 2048).
# DS41's list; its two DSML cases are the same picks rendered as Inkling-2 JSON tool calls.
HELDOUT = [
    ('wikitext2_a', 'wikitext2', 'stream', 2048), ('wikitext2_b', 'wikitext2', 'stream', 2048),
    ('en_web', 'en_web', 'doc', 2048), ('ko_web_a', 'ko_web', 'doc', 2048), ('ko_web_b', 'ko_web', 'doc', 2048),
    ('zh_web_a', 'zh_web', 'doc', 1536), ('zh_web_b', 'zh_web', 'doc', 1536),
    ('ja_web', 'ja_web', 'doc', 1536), ('es_web', 'es_web', 'doc', 1536),
    ('math_web', 'math_web', 'doc', 2048), ('code_python', 'code_web', 'doc_py', 2048), ('code_c', 'code_web', 'doc_c', 2048),
    ('ko_chat_nothink', 'ko_stem', 'chat_nothink', 3072), ('ko_math_think', 'ko_math', 'chat_think', 3072),
    ('en_reason_think', 'casc2_general', 'chat_think', 3072), ('math_think', 'casc1_math', 'chat_think', 3072),
    ('if_chat_nothink', 'if_off', 'chat_nothink', 3072), ('tool_call_json', 'tool', 'chat_tool', 4096),
    ('swe_agentless', 'swe', 'chat_nothink', 4096), ('code_chat_think', 'cp_py', 'chat_think', 4096),
    ('zh_chat_think', 'zh_r1', 'chat_think', 3072), ('zh_chat_nothink', 'zh_chat', 'chat_nothink', 3072),
    ('ja_chat_nothink', 'ja_chat', 'chat_nothink', 3072), ('ds4_agent_json', 'ds4', 'ds4_agent', 4096),
    ('long_en_doc', 'en_web', 'long_doc', 8192), ('long_finance', 'finance', 'chat_nothink_long', 8192),
]
# manifest totals that describe the selection (copied unchanged by `reprefix`)
SELECTION_KEYS = ('raw_prefix_selection', 'heldout_overlap_filter', 'long_used', 'skipped_control_or_unusable', 'buckets')
W_OVERLAP, BOILER_ROWS = 64, 3


def family_of(src):
    if src == 'wikitext2':
        return 'wikitext2'
    if src in RAW_BUDGET:
        return RAW_BUDGET[src][0]
    return CHAT_BUDGET[src][0]


def windows(ids, W=W_OVERLAP):
    b = np.asarray(ids, np.int32).tobytes()
    return [hashlib.blake2b(b[4 * k:4 * (k + W)], digest_size=8).digest() for k in range(0, len(ids) - W + 1)]


def apply_prefix(ids, meta):
    """Canonical row (raw rows start with <|begin_of_text|>) -> the row written for RAW_PREFIX."""
    if meta['format'] == 'raw':
        if ids[0] != CANON_PREFIX:
            raise ValueError('raw row without the canonical prefix')
        pid = RAW_PREFIXES[RAW_PREFIX]
        return ([pid] + list(ids[1:]) if pid is not None else list(ids[1:])), meta
    return ids, meta


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
            piece = [CANON_PREFIX] + ids[start:start + target]
            held.append({'case': case, 'family': 'wikitext2', 'domain': 'wikitext2', 'lang': 'en',
                         'source': ordered[0]['_src'] | {'row_group': None, 'row': None}, 'format': 'raw', 'mode': None,
                         'ids': piece, 'note': f'raw prefix + wikitext-2-raw test contiguous tokens [{start}, {start + target}) '
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
                try:
                    ids = R.encode([R.chat(msgs, think, tools)])[0]
                except Exception:
                    continue
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
    skipped = {}
    got_by = {}

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

    # ---- raw text buckets (prefix + document)
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
                                     'truncated': len(ids) > n, 'doc_tokens': len(ids) - 1})
        got_by[name] = got
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
        got_by[name] = got
        log('chat', name, got, 'tokens', f'{time.monotonic() - t0:.0f}s')

    # ---- DS4 imatrix prompts rendered natively with the Inkling-2 template (local file)
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
    got_by[name] = got
    log('chat', name, got, 'tokens', 'skipped (control-token text / unusable)', skipped)

    # ---- held-out disjointness: drop calibration rows that share document text with a held-out
    # case. A 64-token window counts as boilerplate (chat template, thinking-effort header, tool
    # schemas, task instructions) when it occurs in >= 3 calibration rows; any other shared window
    # removes the calibration row. The boilerplate windows are recorded (blake2b-8 of the window's
    # int32 bytes) so `check` can verify the written files against this rule.
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
        if len(rs) < BOILER_ROWS:
            drop |= rs
    dropped = [{'domain': samples[r][1]['domain'], 'n_tokens': samples[r][1]['n_tokens'], 'source': samples[r][1]['source'],
                'heldout_cases': sorted({held[c]['case'] for d in windows(samples[r][0]) if d in hw for c in hw[d]})}
               for r in sorted(drop)]
    boiler = sorted(d.hex() for d, rs in rows_per.items() if len(rs) >= BOILER_ROWS)
    log('heldout overlap: dropped calibration rows', len(dropped), sum(x['n_tokens'] for x in dropped), 'tokens;',
        'boilerplate windows', len(boiler))
    samples = [sm for r, sm in enumerate(samples) if r not in drop]
    overlap_report = {'window_tokens': W_OVERLAP, 'boilerplate_min_rows': BOILER_ROWS, 'dropped_rows': dropped,
                      'dropped_tokens': sum(x['n_tokens'] for x in dropped),
                      'boilerplate_windows': len(boiler), 'shared_windows_total': len(rows_per),
                      'note': 'computed on rows with the canonical <|begin_of_text|> raw prefix (a superset of the '
                              'windows of either written prefix); dropped n_tokens include it',
                      'boilerplate_window_digests': boiler}
    buckets = {n: {'domain': v[0], 'lang': v[1], 'budget': v[2], 'max_doc_tokens': v[3],
                   'long_allow': LONG_ALLOW.get(n, LONG_DEFAULT), 'tokens_selected': got_by.get(n, 0)}
               for n, v in RAW_BUDGET.items()}
    buckets.update({n: {'domain': v[0], 'lang': v[1], 'budget': v[2], 'both_modes_p': v[3], 'max_row_tokens': v[4],
                        'long_allow': LONG_ALLOW.get(n, LONG_DEFAULT), 'tokens_selected': got_by.get(n, 0)}
                    for n, v in CHAT_BUDGET.items()})
    selection = ('raw rows selected, budgeted and overlap-filtered with the <|begin_of_text|> prefix; the written '
                 'prefix only changes their first token (`reprefix` switches it)')

    # ---- write
    order = list(range(len(samples)))
    random.Random(f'{SEED}:order').shuffle(order)
    samples = [samples[i] for i in order]
    write_set(ROOT / 'calib', [apply_prefix(ids, m) for ids, m in samples],
              extra={'raw_prefix_selection': selection, 'heldout_overlap_filter': overlap_report, 'long_used': long_used,
                     'skipped_control_or_unusable': skipped, 'buckets': buckets})
    hs = [apply_prefix(h['ids'], {k: v for k, v in h.items() if k != 'ids'}) for h in held]
    write_set(ROOT / 'heldout', hs, extra={'raw_prefix_selection': selection})
    write_sums()
    log('done', f'{time.monotonic() - t0:.0f}s')


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 22), b''):
            h.update(b)
    return h.hexdigest()


_ST_DTYPES = {'int32': 'I32', 'int64': 'I64', 'float32': 'F32', 'float16': 'F16', 'uint8': 'U8'}


def save_st(path, tensors, metadata):
    """safetensors file with a deterministic header (sorted metadata, tensors in the given order):
    safetensors' own writer orders the metadata map differently on every run."""
    header, off, arrs = {'__metadata__': {k: str(metadata[k]) for k in sorted(metadata)}}, 0, []
    for name, a in tensors:
        a = np.ascontiguousarray(a)
        if a.dtype.byteorder == '>':
            a = a.astype(a.dtype.newbyteorder('<'))
        header[name] = {'dtype': _ST_DTYPES[a.dtype.name], 'shape': list(a.shape), 'data_offsets': [off, off + a.nbytes]}
        off += a.nbytes
        arrs.append(a)
    h = json.dumps(header, separators=(',', ':')).encode()
    h += b' ' * (-len(h) % 8)
    tmp = path.with_name(path.name + '.partial')
    with open(tmp, 'wb') as f:
        f.write(struct.pack('<Q', len(h)))
        f.write(h)
        for a in arrs:
            f.write(a.reshape(-1).view(np.uint8).data)
    os.replace(tmp, path)


_INFO = {}


def model_info():
    if not _INFO:
        import importlib.metadata as md
        ver = md.version('transformers')
        try:
            commit = json.loads(md.distribution('transformers').read_text('direct_url.json') or '{}')['vcs_info']['commit_id']
        except Exception:
            commit = None
        tf = f'transformers {ver}' + (f' @ {commit}' if commit else '')
        _INFO['transformers'] = tf
        _INFO['tokenizer'] = f'{MODEL_REPO}@{MODEL_REV} tokenizer.json sha256 {sha256_file(MODEL / "tokenizer.json")}'
        _INFO['chat_template'] = (f'{MODEL_REPO}@{MODEL_REV} chat_template.jinja (sha256 {sha256_file(MODEL / "chat_template.jinja")}) '
                                  f'via {tf} apply_chat_template; reasoning_effort "{THINK_EFFORT}" (0.9) for think rows, '
                                  f'"{NOTHINK_EFFORT}" (0) for non-think rows; think rows keep reasoning_content on every '
                                  'assistant turn whose source has it, non-think rows none; no generation prompt (the '
                                  'template has none)')
        _INFO['processor'] = (f'{MODEL_REPO}@{MODEL_REV} processor_config.json (sha256 '
                              f'{sha256_file(MODEL / "processor_config.json")}): Inkling2Processor / Inkling2ImageProcessor '
                              f'via {tf}')
    return _INFO


def write_set(stem, samples, extra=None, meta_extra=None):
    lens = [len(ids) for ids, _ in samples]
    flat = np.concatenate([np.asarray(ids, np.int32) for ids, _ in samples])
    if flat.max() >= REAL_VOCAB or flat.min() < 0 or min(lens) <= 0:
        raise ValueError('token id range / empty row')
    offsets = np.zeros(len(samples) + 1, np.int64)
    offsets[1:] = np.cumsum(lens)
    assert offsets[-1] == len(flat)
    pid = RAW_PREFIXES[RAW_PREFIX]
    fmts = [m['format'] for _, m in samples]
    starts = flat[offsets[:-1]]
    if any(int(s) != MSG_SYSTEM_ID for s, f in zip(starts, fmts) if f == 'chat'):
        raise ValueError('chat row not starting with <|message_system|>')
    if pid is not None and any(int(s) != pid for s, f in zip(starts, fmts) if f == 'raw'):
        raise ValueError('raw row without its prefix')
    if int((flat == CANON_PREFIX).sum()) != (fmts.count('raw') if pid is not None else 0):
        raise ValueError('<|begin_of_text|> inside a row')
    rows = {hashlib.sha256(flat[offsets[i]:offsets[i + 1]].tobytes()).hexdigest() for i in range(len(samples))}
    if len(rows) != len(samples):
        raise ValueError('duplicate rows')
    path = stem.with_suffix('.safetensors')
    info = model_info()
    meta = {'tokenizer': f'{MODEL_REPO}@{MODEL_REV}', 'raw_prefix': RAW_PREFIX, 'raw_prefix_id': '' if pid is None else pid,
            'chat_row_start_id': MSG_SYSTEM_ID, 'eos_id': EOS_ID, 'seed': SEED, 'vocab_size': VOCAB,
            'real_vocab_size': REAL_VOCAB}
    meta.update(meta_extra or {})
    save_st(path, [('input_ids', flat), ('offsets', offsets)], meta)
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
              'content_sha256': hashlib.sha256(flat.tobytes() + offsets.tobytes()).hexdigest(),
              'raw_prefix': RAW_PREFIX, 'raw_prefix_id': pid,
              'chat_row_start': f'<|message_system|> ({MSG_SYSTEM_ID}): tool_declare, a system prompt or the thinking-effort '
                                'header; no BOS',
              'eos_id': EOS_ID, 'vocab_size': VOCAB, 'real_vocab_size': REAL_VOCAB,
              'doc_tokens': 'raw rows: document tokens without the prefix',
              'tokenizer': info['tokenizer'], 'chat_template': info['chat_template'],
              'ds4_prompts_sha256': sha256_file(DS4_PROMPTS),
              'seed': SEED, 'max_len': MAX_LEN}
    if extra:
        totals.update(extra)
    stem.with_name(stem.name + '_manifest.json').write_text(json.dumps({'samples': metas, 'totals': totals}, ensure_ascii=False, indent=0) + '\n')
    log('wrote', path, json.dumps({k: totals[k] for k in ('sequences', 'tokens', 'length', 'tokens_ge_4096')}))


OUT_FILES = ('calib.safetensors', 'calib_manifest.json', 'heldout.safetensors', 'heldout_manifest.json',
             'images.safetensors', 'images_manifest.json', 'images_heldout.safetensors', 'images_heldout_manifest.json',
             'samples_decoded.txt')


def write_sums():
    lines = [f'{sha256_file(ROOT / f)}  {f}\n' for f in OUT_FILES if (ROOT / f).exists()]
    (ROOT / 'SHA256SUMS').write_text(''.join(lines))
    log('wrote', ROOT / 'SHA256SUMS', len(lines), 'files')


def reprefix():
    """Rewrite calib / heldout for --raw-prefix from the written sets (no refetch, no re-tokenization):
    raw rows get their first token swapped / added / removed, everything else is copied."""
    from safetensors.numpy import load_file
    pid = RAW_PREFIXES[RAW_PREFIX]
    for stem in ('calib', 'heldout'):
        t = load_file(str(ROOT / f'{stem}.safetensors'))
        man = json.loads((ROOT / f'{stem}_manifest.json').read_text())
        old = man['totals']['raw_prefix']
        if old == RAW_PREFIX:
            log(stem, 'already has raw prefix', old)
            continue
        old_id = RAW_PREFIXES[old]
        ids, off = t['input_ids'], t['offsets']
        rows = []
        for i, m in enumerate(man['samples']):
            x = ids[off[i]:off[i + 1]].tolist()
            if m['format'] == 'raw':
                if old_id is not None:
                    if x[0] != old_id:
                        raise SystemExit(f'{stem} row {i}: raw row without its prefix')
                    x = x[1:]
                if pid is not None:
                    x = [pid] + x
            rows.append((x, m))
        write_set(ROOT / stem, rows, extra={k: man['totals'][k] for k in SELECTION_KEYS if k in man['totals']})
    write_sums()


# ---------------------------------------------------------------- sanity checks

def decode_row(tok, x):
    """Decoded row with each run of image placeholders collapsed to `<|unused_200054|>xN`."""
    segs, cur, k = [], [], 0
    x = [int(v) for v in x]
    while k < len(x):
        if x[k] == IMAGE_ID:
            j = k
            while j < len(x) and x[j] == IMAGE_ID:
                j += 1
            if cur:
                segs.append(tok.decode(cur, skip_special_tokens=False))
                cur = []
            segs.append(f'<|unused_200054|>x{j - k}')
            k = j
        else:
            cur.append(x[k])
            k += 1
    if cur:
        segs.append(tok.decode(cur, skip_special_tokens=False))
    return ''.join(segs)


def excerpt(s, head=900, tail=500):
    parts = [f'--- head\n{s[:head]}']
    for marker in ('<|content_thinking|>', '<|content_invoke_tool_json|>', '<|message_tool|>', '<|content_image|>'):
        p = s.find(marker)
        if head <= p < len(s) - tail:
            parts.append(f'--- first {marker}\n{s[max(head, p - 150):p + 450]}')
    parts.append(f'--- tail\n{s[-tail:]}')
    return '\n'.join(parts)


def check(per_domain=2):
    from safetensors.numpy import load_file
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(str(MODEL / 'tokenizer.json'))
    out = []
    loaded = {}
    for stem in ('calib', 'heldout'):
        t = load_file(str(ROOT / f'{stem}.safetensors'))
        ids, off = t['input_ids'], t['offsets']
        man = json.loads((ROOT / f'{stem}_manifest.json').read_text())
        n = len(off) - 1
        assert ids.dtype == np.int32 and off.dtype == np.int64 and off[0] == 0 and off[-1] == len(ids)
        assert len(man['samples']) == n and man['totals']['safetensors_sha256'] == sha256_file(ROOT / f'{stem}.safetensors')
        lens = np.diff(off)
        assert (lens > 0).all() and all(int(lens[i]) == man['samples'][i]['n_tokens'] for i in range(n))
        assert int(ids.min()) >= 0 and int(ids.max()) < REAL_VOCAB, 'token id out of range'
        pid = RAW_PREFIXES[man['totals']['raw_prefix']]
        fmt = [m['format'] for m in man['samples']]
        starts = ids[off[:-1]]
        bad_chat = sum(1 for i in range(n) if fmt[i] == 'chat' and starts[i] != MSG_SYSTEM_ID)
        bad_raw = sum(1 for i in range(n) if fmt[i] == 'raw' and pid is not None and starts[i] != pid)
        n_bot = int((ids == CANON_PREFIX).sum())
        assert bad_chat == 0 and bad_raw == 0 and n_bot == (fmt.count('raw') if pid is not None else 0)
        rows = {hashlib.sha256(ids[off[i]:off[i + 1]].tobytes()).hexdigest() for i in range(n)}
        assert len(rows) == n, 'duplicate rows'
        log(stem, 'rows', n, '(raw', fmt.count('raw'), 'chat', fmt.count('chat'), ') tokens', len(ids), 'ids in [',
            int(ids.min()), ',', int(ids.max()), '] <', REAL_VOCAB, '; raw prefix', man['totals']['raw_prefix'],
            '; chat rows starting with <|message_system|>', fmt.count('chat') - bad_chat, '; <|begin_of_text|>', n_bot,
            '; EOS', int((ids == EOS_ID).sum()), '; <|endoftext|>', int((ids == EOT_ID).sum()),
            '; image placeholders', int((ids == IMAGE_ID).sum()), '; max len', int(lens.max()))
        assert int((ids == IMAGE_ID).sum()) == 0
        loaded[stem] = (ids, off, man)
        seen = {}
        for i, m in enumerate(man['samples']):
            key = (m.get('case') or m['domain'], m.get('mode'), bool(m.get('tools')))
            if seen.get(key, 0) >= (1 if stem == 'heldout' else per_domain):
                continue
            seen[key] = seen.get(key, 0) + 1
            x = ids[off[i]:off[i + 1]]
            out.append(f'===== {stem} row {i} {key} n_tokens={len(x)} source={json.dumps(m.get("source"), ensure_ascii=False)[:300]}\n'
                       f'{excerpt(decode_row(tok, x))}\n')
    # held-out / calibration disjointness on the written files: every 64-token window a held-out case
    # shares with a calibration row must be one of the build's boilerplate windows (in >= 3 calibration
    # rows when the filter ran; recorded in the calibration manifest)
    hids, hoff, _ = loaded['heldout']
    cids, coff, cman = loaded['calib']
    boiler = {bytes.fromhex(x) for x in cman['totals']['heldout_overlap_filter']['boilerplate_window_digests']}
    hw = set()
    for i in range(len(hoff) - 1):
        hw.update(windows(hids[hoff[i]:hoff[i + 1]]))
    cnt = {}
    for i in range(len(coff) - 1):
        for d in set(windows(cids[coff[i]:coff[i + 1]])):
            if d in hw:
                cnt[d] = cnt.get(d, 0) + 1
    bad = sum(1 for d in cnt if d not in boiler)
    log('held-out windows shared with calibration rows:', len(cnt), 'of', len(hw), '(in >=', BOILER_ROWS, 'written rows:',
        sum(1 for c in cnt.values() if c >= BOILER_ROWS), '); not boilerplate:', bad)
    assert bad == 0, 'held-out text leaks into the calibration set'
    # image sets
    img_keys = {}
    for stem in ('images', 'images_heldout'):
        p = ROOT / f'{stem}.safetensors'
        if not p.exists():
            continue
        t = load_file(str(p))
        man = json.loads((ROOT / f'{stem}_manifest.json').read_text())
        ids, off, pv, grid, cnt_i = t['input_ids'], t['offsets'], t['pixel_values'], t['image_grid_thw'], t['image_counts']
        n = len(off) - 1
        assert man['totals']['safetensors_sha256'] == sha256_file(p) and len(man['samples']) == n
        assert ids.dtype == np.int32 and off.dtype == np.int64 and pv.dtype == np.float32 and grid.dtype == np.int64
        assert cnt_i.dtype == np.int64 and cnt_i.shape == (n,) and int(cnt_i.sum()) == grid.shape[0] and grid.shape[1] == 3
        assert pv.shape[1:] == (40, 40, 3) and int(grid.prod(1).sum()) == pv.shape[0]
        assert off[0] == 0 and off[-1] == len(ids) and (np.diff(off) > 0).all()
        assert int(ids.min()) >= 0 and int(ids.max()) < REAL_VOCAB, 'token id out of range'
        assert float(pv.min()) >= -1.0079 and float(pv.max()) <= 1.0
        gi = 0
        for i in range(n):
            x = ids[off[i]:off[i + 1]]
            g = grid[gi:gi + cnt_i[i]]
            gi += int(cnt_i[i])
            assert x[0] == MSG_SYSTEM_ID and int((x == IMAGE_ID).sum()) == int(g.prod(1).sum())
            assert int((x == 200005).sum()) == int(cnt_i[i])     # one <|content_image|> per image
            assert int((x == EOS_ID).sum()) == 1 and x[-1] == EOS_ID
            m = man['samples'][i]
            assert m['n_image_tokens'] == int(g.prod(1).sum()) and m['n_tokens'] == len(x)
            img_keys.setdefault(stem, set()).update(im['sha256'] for im in m['images'])
        log(stem, 'samples', n, 'images', grid.shape[0], 'tokens', len(ids), 'image tokens', pv.shape[0],
            'ids in [', int(ids.min()), ',', int(ids.max()), '] <', REAL_VOCAB, 'pixel_values', tuple(pv.shape),
            '[', float(pv.min()), ',', float(pv.max()), ']')
        seen = set()
        for i, m in enumerate(man['samples']):
            key = m.get('case') or m['domain']
            if key in seen:
                continue
            seen.add(key)
            x = ids[off[i]:off[i + 1]]
            out.append(f'===== {stem} row {i} {key} n_tokens={len(x)} image_tokens={m["n_image_tokens"]} '
                       f'images={json.dumps(m["images"])[:300]} source={json.dumps(m.get("source"), ensure_ascii=False)[:300]}\n'
                       f'{excerpt(decode_row(tok, x), head=1400)}\n')
    if 'images' in img_keys and 'images_heldout' in img_keys:
        assert not img_keys['images'] & img_keys['images_heldout'], 'held-out image in the calibration image set'
        log('image sets disjoint (image sha256)')
    (ROOT / 'samples_decoded.txt').write_text('\n'.join(out))
    log('wrote', ROOT / 'samples_decoded.txt', len(out), 'samples')
    write_sums()


# ================================================================ images (build-images)
# Pinned image datasets with permissive licences (photos, documents, charts, screenshots). Each sample is
# one user turn [image, question] and the dataset's answer as a non-thinking assistant turn; the
# Inkling-2 processor expands the image placeholder to ceil(h/40) * ceil(w/40) tokens (one per 40x40
# patch, padded with -1 on the 0-255 scale; no resizing of its own). Images above MAX_IMAGE_TOKENS
# are downscaled (aspect kept, PIL bicubic) to the largest size within the cap.

IMG_REV = {
    'lmms-lab/textvqa': '9c0699cd19768ac5ab97568f6b3cbac4c0062884',
    'naver-clova-ix/cord-v2': '7f0115a4b758a71d6473b8d085751692da2fef98',
    'HuggingFaceM4/the_cauldron': '847a98a779b1652d65111daf20c972dfcd333605',
    'rootsautomation/ScreenSpot': '0be08781e2e188582f6131625ae1598d443b4d5d',
}
IMG_LICENCE = {
    'lmms-lab/textvqa': {
        'licence': 'CC BY 4.0 (TextVQA questions / answers); images: Open Images (Flickr), listed as CC BY 2.0',
        'evidence': 'facebook/textvqa@30a47cfa557c996f64903f01250799d453be6215 README (license: cc-by-4.0; "Licensing '
                    'Information: CC by 4.0"); Open Images factsfigures_v7: "The images are listed as having a CC BY 2.0 '
                    'license"; lmms-lab/textvqa is a parquet repackaging of TextVQA (its card states no licence)'},
    'naver-clova-ix/cord-v2': {
        'licence': 'CC BY 4.0',
        'evidence': 'naver-clova-ix/cord-v2@7f0115a4b758a71d6473b8d085751692da2fef98 README (license: cc-by-4.0)'},
    'HuggingFaceM4/the_cauldron': {
        'licence': 'CC BY 4.0 (PlotQA subset); the Cauldron prompts CC BY 4.0',
        'evidence': 'github.com/NiteshMethani/PlotQA README: "All the datasets created as part of this work are released '
                    'under a CC-BY-4.0 license"; HuggingFaceM4/the_cauldron@847a98a779b1652d65111daf20c972dfcd333605 '
                    'README: sub-datasets keep their licences, "To the extent we have any rights in the prompts, these '
                    'are licensed under CC-BY-4.0"'},
    'rootsautomation/ScreenSpot': {
        'licence': 'Apache-2.0',
        'evidence': 'rootsautomation/ScreenSpot@0be08781e2e188582f6131625ae1598d443b4d5d README (license: apache-2.0; '
                    '"License: Apache 2.0")'},
}
# image sources: name -> (repo, [paths], row groups per path (seeded, 100 rows each), domain, lang)
IMAGES = {
    'img_photo': ('lmms-lab/textvqa', ['data/train-00007-of-00020.parquet'], 2, 'image_photo', 'en'),
    'img_doc': ('naver-clova-ix/cord-v2', ['data/train-00002-of-00004-688fe1305a55e5cc.parquet'], 1, 'image_document', 'id'),
    'img_chart': ('HuggingFaceM4/the_cauldron', ['plotqa/train-00005-of-00016-948e185c5d91409f.parquet'], 2, 'image_chart', 'en'),
    'img_screen': ('rootsautomation/ScreenSpot', ['data/test-00000-of-00003.parquet', 'data/test-00001-of-00003.parquet',
                                                  'data/test-00002-of-00003.parquet'], 2, 'image_screenshot', 'en'),
}
# name -> (calibration images, calibration image-token budget, held-out images)
IMG_BUDGET = {'img_photo': (80, 42_000, 2), 'img_doc': (50, 40_000, 2), 'img_chart': (60, 30_000, 2),
              'img_screen': (60, 38_000, 2)}
MAX_IMAGE_TOKENS = 1024          # language-model image tokens per image
IMAGE_TOKENS_TOTAL = 150_000     # cap on the calibration set's image tokens
PATCH = 40                       # processor_config size: one image token per 40x40 patch
CORD_Q = 'Extract the line items and totals of this receipt as JSON.'
SCREEN_Q = ('Find the UI element for this instruction: "{}". Answer with its bounding box [x1, y1, x2, y2] in coordinates '
            'normalized to 0-1.')


def image_tokens(w, h):
    return -(-w // PATCH) * -(-h // PATCH)


def fetch_image_source(name, spec):
    """Seeded row groups of a pinned parquet file; image bytes to raw/<name>/images/*.bin (data only)."""
    import pyarrow.parquet as pq
    repo, paths, groups = spec[:3]
    d = RAW / name
    out = d / 'records.jsonl'
    if out.exists():
        return name, 'cached'
    (d / 'images').mkdir(parents=True, exist_ok=True)
    f_s = fs()
    rng = random.Random(f'{SEED}:{name}')
    got = []
    for path in paths:
        full = f'datasets/{repo}@{IMG_REV[repo]}/{path}'
        with f_s.open(full, 'rb', block_size=2 ** 23) as f:
            pf = pq.ParquetFile(f)
            n = pf.num_row_groups
            picks = list(range(n)) if groups >= n else sorted(rng.sample(range(n), groups))
            for g in picks:
                for i, row in enumerate(pf.read_row_group(g).to_pylist()):
                    if 'images' in row:
                        ims = row.pop('images') or []
                        im = ims[0] if len(ims) == 1 else None
                    else:
                        im = row.pop('image', None)
                    if not isinstance(im, dict) or not im.get('bytes'):
                        continue
                    fn = f'images/{len(got):05d}.bin'
                    (d / fn).write_bytes(im['bytes'])
                    row['_image_file'] = fn
                    row['_image_sha256'] = hashlib.sha256(im['bytes']).hexdigest()
                    row['_src'] = {'repo': repo, 'revision': IMG_REV[repo], 'path': path, 'row_group': g, 'row': i}
                    got.append(row)
    tmp = out.with_suffix('.partial')
    with tmp.open('w') as w:
        for r in got:
            w.write(json.dumps(r, ensure_ascii=False) + '\n')
    tmp.rename(out)
    return name, len(got)


def image_qa(name, r):
    """-> (question, the dataset's answer) or None."""
    if name == 'img_photo':        # TextVQA: question, the most frequent of the annotators' answers
        q = (r.get('question') or '').strip()
        answers = [a.strip() for a in r.get('answers') or [] if isinstance(a, str) and a.strip()]
        if not q or not answers:
            return None
        counts = {}
        for a in answers:
            counts[a] = counts.get(a, 0) + 1
        return q, max(counts, key=lambda a: (counts[a], -answers.index(a)))
    if name == 'img_doc':          # CORD v2: the receipt's ground-truth parse
        try:
            parse = json.loads(r.get('ground_truth') or '{}').get('gt_parse')
        except json.JSONDecodeError:
            return None
        return (CORD_Q, json.dumps(parse, ensure_ascii=False)) if parse else None
    if name == 'img_chart':        # PlotQA (the Cauldron): one of the row's question / answer pairs (seeded)
        pairs = [t for t in r.get('texts') or [] if isinstance(t, dict) and t.get('user') and t.get('assistant')]
        if not pairs:
            return None
        s = r['_src']
        t = pairs[random.Random(f'{SEED}:qa:{name}:{s["path"]}:{s["row_group"]}:{s["row"]}').randrange(len(pairs))]
        return t['user'].strip(), t['assistant'].strip()
    if name == 'img_screen':       # ScreenSpot: instruction -> normalized bounding box
        bb, ins = r.get('bbox'), (r.get('instruction') or '').strip()
        if not ins or not isinstance(bb, list) or len(bb) != 4:
            return None
        return SCREEN_Q.format(ins), '[' + ', '.join(f'{float(v):.3f}' for v in bb) + ']'
    raise KeyError(name)


def prepare_image(path, cap=MAX_IMAGE_TOKENS):
    """Decoded image, downscaled (aspect kept) when it would exceed `cap` image tokens."""
    from PIL import Image
    im = Image.open(io.BytesIO(path.read_bytes()))
    im.load()
    w, h = im.size
    info = {'orig_size': [w, h], 'orig_mode': im.mode, 'resized': False}
    if image_tokens(w, h) > cap:
        if im.mode not in ('RGB', 'RGBA', 'L'):
            im = im.convert('RGBA' if im.has_transparency_data else 'RGB')
        s = (cap * PATCH * PATCH / (w * h)) ** 0.5
        while True:
            w2, h2 = max(1, round(w * s)), max(1, round(h * s))
            if image_tokens(w2, h2) <= cap:
                break
            s *= 0.995
        im = im.resize((w2, h2), Image.Resampling.BICUBIC)
        info['resized'] = True
    w, h = im.size
    info.update(size=[w, h], grid_thw=[1, -(-h // PATCH), -(-w // PATCH)], image_tokens=image_tokens(w, h))
    return im, info


def render_image_item(proc, it):
    msgs = [{'role': 'user', 'content': [{'type': 'image'}, {'type': 'text', 'text': it['q']}]},
            {'role': 'assistant', 'content': it['a']}]
    text = proc.tokenizer.apply_chat_template(msgs, tokenize=False, reasoning_effort=NOTHINK_EFFORT)
    enc = proc(text=[text], images=[it['img']], return_tensors='pt')
    ids = enc['input_ids'][0].numpy().astype(np.int32)
    pv = enc['pixel_values'].numpy().astype(np.float32)
    grid = enc['image_grid_thw'].numpy().astype(np.int64)
    n_img = int(grid.prod(1).sum())
    if (int((ids == IMAGE_ID).sum()) != n_img or n_img != it['info']['image_tokens'] or pv.shape[0] != n_img
            or list(grid[0]) != it['info']['grid_thw']):
        raise ValueError('image placeholder expansion mismatch')
    if int(ids.max()) >= REAL_VOCAB or int(ids[0]) != MSG_SYSTEM_ID:
        raise ValueError('token id range / row start')
    r = it['r']
    src = dict(r['_src'])
    for k in ('question_id', 'image_id', 'file_name', 'data_source', 'data_type', 'set_name'):
        if r.get(k) is not None:
            src[k] = r[k]
    meta = {}
    if it.get('case'):
        meta['case'] = it['case']
    meta.update({'domain': it['domain'], 'lang': it['lang'], 'format': 'chat_image', 'mode': 'nothink', 'source': src,
                 'licence': IMG_LICENCE[src['repo']]['licence'],
                 'images': [dict(it['info'], sha256=r['_image_sha256'])],
                 'question': it['q'], 'answer': it['a'],
                 'n_image_tokens': n_img, 'n_text_tokens': len(ids) - n_img, 'n_tokens': len(ids)})
    return ids, pv, grid, meta


def write_image_set(stem, items, extra=None):
    lens = [len(x[0]) for x in items]
    flat = np.concatenate([x[0] for x in items]).astype(np.int32)
    offsets = np.zeros(len(items) + 1, np.int64)
    offsets[1:] = np.cumsum(lens)
    pv = np.concatenate([x[1] for x in items]).astype(np.float32)
    grid = np.concatenate([x[2] for x in items]).astype(np.int64)
    counts = np.asarray([len(x[2]) for x in items], np.int64)
    if flat.max() >= REAL_VOCAB or flat.min() < 0 or min(lens) <= 0 or int(grid.prod(1).sum()) != pv.shape[0]:
        raise ValueError('token id range / empty row / patch count')
    path = stem.with_suffix('.safetensors')
    info = model_info()
    meta = {'tokenizer': f'{MODEL_REPO}@{MODEL_REV}', 'image_token_id': IMAGE_ID, 'patch_size': PATCH, 'eos_id': EOS_ID,
            'seed': SEED, 'vocab_size': VOCAB, 'real_vocab_size': REAL_VOCAB,
            'layout': 'input_ids int32 [N] + offsets int64 [n+1]; pixel_values float32 [image tokens, 40, 40, 3]; '
                      'image_grid_thw int64 [images, 3]; image_counts int64 [n] (images of each row, in order)'}
    save_st(path, [('input_ids', flat), ('offsets', offsets), ('pixel_values', pv), ('image_grid_thw', grid),
                   ('image_counts', counts)], meta)
    metas = []
    for i, x in enumerate(items):
        m = dict(x[3])
        m['i'] = i
        metas.append(m)
    by_domain = {}
    for m in metas:
        d = by_domain.setdefault(m['domain'], {'samples': 0, 'images': 0, 'image_tokens': 0, 'tokens': 0, 'resized': 0})
        d['samples'] += 1
        d['images'] += len(m['images'])
        d['image_tokens'] += m['n_image_tokens']
        d['tokens'] += m['n_tokens']
        d['resized'] += sum(im['resized'] for im in m['images'])
    it_tok = [im['image_tokens'] for m in metas for im in m['images']]
    totals = {'samples': len(items), 'images': int(grid.shape[0]), 'tokens': int(offsets[-1]),
              'image_tokens': int(pv.shape[0]), 'text_tokens': int(offsets[-1]) - int(pv.shape[0]),
              'image_tokens_per_image': {'min': int(min(it_tok)), 'p50': int(np.median(it_tok)), 'max': int(max(it_tok)),
                                         'mean': float(np.mean(it_tok))},
              'by_domain': by_domain,
              'sources': {n: {'repo': s[0], 'revision': IMG_REV[s[0]], 'paths': s[1], 'row_groups_per_path': s[2],
                              'domain': s[3], 'budget': dict(zip(('calib_images', 'calib_image_tokens', 'heldout_images'),
                                                                 IMG_BUDGET[n])), **IMG_LICENCE[s[0]]}
                          for n, s in IMAGES.items()},
              'safetensors_sha256': sha256_file(path),
              'content_sha256': hashlib.sha256(flat.tobytes() + offsets.tobytes() + grid.tobytes()).hexdigest(),
              'max_image_tokens': MAX_IMAGE_TOKENS,
              'resize': 'images above max_image_tokens are downscaled with PIL bicubic, aspect kept, to the largest size '
                        'with ceil(h/40)*ceil(w/40) <= max_image_tokens (modes other than RGB/RGBA/L converted first); '
                        'others are passed as decoded',
              'pixel_values': 'as Inkling2ImageProcessor produces them: RGB (alpha composited on its checkerboard), padded '
                              'to multiples of 40 with -1 on the 0-255 scale, rescaled 1/255, normalized mean 0.5 std 0.5; '
                              'one 40x40x3 patch per image token in raster order',
              'rendering': 'user [image, question], assistant answer; chat_template.jinja with reasoning_effort "none"; the '
                           'processor expands each <|unused_200054|> to prod(image_grid_thw) placeholders',
              'tokenizer': info['tokenizer'], 'chat_template': info['chat_template'], 'processor': info['processor'],
              'seed': SEED}
    if extra:
        totals.update(extra)
    stem.with_name(stem.name + '_manifest.json').write_text(json.dumps({'samples': metas, 'totals': totals}, ensure_ascii=False, indent=0) + '\n')
    log('wrote', path, json.dumps({k: totals[k] for k in ('samples', 'images', 'tokens', 'image_tokens', 'text_tokens')}))


def build_images():
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'    # one short text per call: no rayon pool next to torch's
    import torch
    from transformers import AutoProcessor
    torch.set_num_threads(int(os.environ['OMP_NUM_THREADS']))
    torch.set_num_interop_threads(1)
    load_specials()
    t0 = time.monotonic()
    for name, spec in IMAGES.items():
        log('fetched', *fetch_image_source(name, spec))
    proc = AutoProcessor.from_pretrained(str(MODEL))
    if proc.image_token_id != IMAGE_ID:
        raise SystemExit(f'image token id {proc.image_token_id} != {IMAGE_ID}')
    held, calib = [], []
    for name, (repo, paths, groups, domain, lang) in IMAGES.items():
        recs = load_raw(name)
        random.Random(f'{SEED}:pool:{name}').shuffle(recs)
        n_cal, budget, n_held = IMG_BUDGET[name]
        seen, got_tok, n_c, n_h = set(), 0, 0, 0
        for r in recs:      # held-out images first, then calibration images (one sample per image)
            if n_h >= n_held and (n_c >= n_cal or got_tok >= budget):
                break
            keys = {r['_image_sha256']} | {f'{k}:{r[k]}' for k in ('image_id', 'file_name') if r.get(k)}
            if keys & seen:
                continue
            qa = image_qa(name, r)
            if qa is None or has_special(qa[0]) or has_special(qa[1]):
                continue
            img, info = prepare_image(RAW / name / r['_image_file'])
            if n_h < n_held and info['image_tokens'] < 64:
                continue
            seen |= keys
            item = {'domain': domain, 'lang': lang, 'r': r, 'q': qa[0], 'a': qa[1], 'img': img, 'info': info}
            if n_h < n_held:
                item['case'] = f'{domain}_{"ab"[n_h]}'
                held.append(item)
                n_h += 1
            else:
                calib.append(item)
                n_c += 1
                got_tok += info['image_tokens']
        log('images', name, 'calibration', n_c, 'images', got_tok, 'image tokens; held-out', n_h, f'{time.monotonic() - t0:.0f}s')
    total = sum(it['info']['image_tokens'] for it in calib)
    if total > IMAGE_TOKENS_TOTAL:
        raise SystemExit(f'calibration image tokens {total} > {IMAGE_TOKENS_TOTAL}')
    order = list(range(len(calib)))
    random.Random(f'{SEED}:img:order').shuffle(order)
    calib = [calib[i] for i in order]
    write_image_set(ROOT / 'images', [render_image_item(proc, it) for it in calib])
    write_image_set(ROOT / 'images_heldout', [render_image_item(proc, it) for it in held])
    write_sums()
    log('done', f'{time.monotonic() - t0:.0f}s')


def main():
    global ROOT, RAW, MODEL, DS4_PROMPTS, RAW_PREFIX
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('stage', choices=['fetch', 'build', 'check', 'reprefix', 'build-images', 'all'])
    p.add_argument('--only', nargs='*')
    p.add_argument('--out', type=Path, default=ROOT, help=f'calibration directory (downloads under OUT/raw); default {ROOT}')
    p.add_argument('--model', type=Path, default=MODEL, help='Inkling-2 directory (tokenizer, chat_template.jinja, processor)')
    p.add_argument('--ds4-prompts', type=Path, default=DS4_PROMPTS)
    p.add_argument('--raw-prefix', choices=sorted(RAW_PREFIXES), default=RAW_PREFIX,
                   help='first token of raw-text rows: begin_of_text (<|begin_of_text|> 200028, default) or none')
    p.add_argument('--threads', type=int, default=32, help='CPUs this process may use (affinity and thread pools)')
    a = p.parse_args()
    limit_threads(a.threads)
    ROOT, RAW, MODEL, DS4_PROMPTS, RAW_PREFIX = a.out, a.out / 'raw', a.model, a.ds4_prompts, a.raw_prefix
    if a.stage in ('fetch', 'all'):
        fetch(a.only)
    if a.stage in ('build', 'all'):
        build()
    if a.stage == 'reprefix':
        reprefix()
    if a.stage == 'build-images':
        build_images()
    if a.stage in ('check', 'all'):
        check()


if __name__ == '__main__':
    main()
