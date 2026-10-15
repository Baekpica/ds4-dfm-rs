#!/usr/bin/env python3
"""Exercise native pointer binding using the pinned public tensor manifest."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

MANIFEST_SHA = 'c3c35fe33c2ac7075f2b46ef234179b0521e2a3b590d71e3ba1e367917d201cf'

parser = argparse.ArgumentParser()
parser.add_argument('--binary', type=Path, required=True)
parser.add_argument('--model', type=Path, required=True)
args = parser.parse_args()
raw = (args.model / 'tensor-manifest.json').read_bytes()
assert hashlib.sha256(raw).hexdigest() == MANIFEST_SHA
tensors = json.loads(raw)['tensors']
lines = [str(len(tensors))]
for t in tensors:
    dims = list(reversed(t['stored_shape']))
    lines.append(' '.join(str(x) for x in [t['name'], t['type_id'], len(dims), t['payload_bytes'], *dims]))
subprocess.run([str(args.binary.resolve())], input='\n'.join(lines), text=True, check=True)
