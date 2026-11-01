"""Rebuild the measured cases: MODEL_DIR PROMPT_FILE OUTPUT_JSON."""
import hashlib
import json
from pathlib import Path
import sys
from tokenizers import Tokenizer

model, prompt, output = map(Path, sys.argv[1:])
seeds = json.loads(Path(__file__).with_name('quality-seeds.json').read_text())
original = {c['seed']: c['tokens'] for c in seeds['cases']}
tokenizer = Tokenizer.from_file(str(model / 'tokenizer.json'))
filler = tokenizer.encode(prompt.read_text(), add_special_tokens=False).ids
separator = tokenizer.encode('\n\n', add_special_tokens=False).ids
assert all(x not in (200019, 200020) for x in filler)
cases = []
for frontier, seed in [(8192, 1), (8192, 2), (8192, 3), (32768, 4), (65536, 1), (65536, 2), (65536, 3)]:
    ids = original[seed]
    assert ids[183:186] == [200019, 3995, 10]
    needed = frontier + 256 - len(ids) - len(separator)
    assert 0 < needed <= len(filler)
    tokens = ids[:186] + filler[:needed] + separator + ids[186:]
    assert len(tokens) == frontier + 256
    cases.append(dict(name=f'{frontier}-seed{seed}', frontier=frontier, seed=seed, tokens=tokens))
output.write_text(json.dumps(dict(
    source_reference_sha256=seeds['source_sha256'],
    prompt_sha256=hashlib.sha256(prompt.read_bytes()).hexdigest(),
    tokenizer_sha256=hashlib.sha256((model / 'tokenizer.json').read_bytes()).hexdigest(),
    cases=cases)) + '\n')
