#!/usr/bin/env python3
"""Write the locked text token streams and whitespace-separated case manifest."""
import argparse
import json
from pathlib import Path

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('directory', type=Path)
p.add_argument('--label', choices=('native', 'oracle', 'precise'), default='native')
a = p.parse_args()
root = a.directory.resolve()
if any(c.isspace() for c in str(root)):
    p.error('the native test manifest requires paths without whitespace')
inputs = root / 'fixtures'
outputs = root / a.label
inputs.mkdir(parents=True, exist_ok=True)
outputs.mkdir(parents=True, exist_ok=True)
fixture = json.loads(Path(__file__).with_name('text-inputs.json').read_text())
manifest = []
for case in fixture['cases']:
    ids = case['ids']
    assert ids[0] == 0 and ids.count(0) == 1
    path = inputs / (case['id'] + '.tokens')
    path.write_text(' '.join(map(str, ids)) + '\n')
    manifest.append(f"{path} {outputs / (case['id'] + '.f32')}\n")
(root / (a.label + '.cases')).write_text(''.join(manifest))

# Structural ring fixture: repeat the existing user payload, with one BOS and
# the unchanged assistant prefix. This is a cache proof, not a quality prompt.
summary = next(c['ids'] for c in fixture['cases'] if c['id'] == 'summary')
end = summary.index(128007)
body, tail = summary[4:end] + [201], summary[end:]
count = 832 - 4 - len(tail)
ring = summary[:4] + (body * ((count + len(body) - 1) // len(body)))[:count] + tail
assert len(ring) == 832 and ring.count(0) == 1
(inputs / 'ring832.tokens').write_text(' '.join(map(str, ring)) + '\n')
print(root / (a.label + '.cases'))
