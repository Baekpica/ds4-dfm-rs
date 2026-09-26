import csv, io, json, statistics
from pathlib import Path
root = Path(__file__).resolve().parent
samples = {}
reference = None
all_exact = True
for arm in ('original', 'off', 'on'):
    samples[arm] = []
    for repeat in range(3):
        stem = root / f'swiglu6-{repeat}-{arm}-sample'
        text = stem.with_suffix('.csv').read_text()
        rows = list(csv.DictReader(io.StringIO(text[text.index('ctx_tokens,'):])))
        assert len(rows) == 1
        row = rows[0]
        assert int(row['ctx_tokens']) == 8192 and int(row['gen_tokens']) == 128
        samples[arm].append({k:float(row[k]) for k in ('prefill_tps','gen_tps')})
        guard = json.loads(stem.with_suffix('.memory.jsonl').read_text().splitlines()[-1])
        assert guard['event'] == 'exit' and guard['status'] == 0 and guard['payload_status'] == 0
        folder = Path(str(stem) + '-proof')
        logits = json.loads((folder / 'frontier_008192.logits.json').read_text())['logits']
        tokens = json.loads((folder / 'tokens-8192.json').read_text())
        assert len(logits) == 152576 and len(tokens) == 128
        proof = (logits,tokens)
        if reference is None: reference = proof
        all_exact &= proof == reference
medians = {a:{k:statistics.median(v[k] for v in values) for k in ('prefill_tps','gen_tps')} for a,values in samples.items()}
gains = {a:{k:100*(medians[a][k]/medians['original'][k]-1) for k in ('prefill_tps','gen_tps')} for a in ('off','on')}
result = {'samples':samples, 'medians':medians, 'gain_vs_original_pct':gains, 'all_full_logits_and_tokens_exact':all_exact}
(root/'swiglu6-result.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
