#!/usr/bin/env python3
"""Real local BASE producer and raw fallback, isolated per cached control."""
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
BASE_ENV = {k: v for k, v in os.environ.items() if not k.startswith('DS4_')}
MANIFEST = 'DS4_CUDA_WEIGHT_IPC_MANIFEST'
SCOPE = 'DS4_CUDA_WEIGHT_IPC_SCOPE'
CASES = [
    ('local', {}, 1),
    ('empty-manifest', {MANIFEST: ''}, 1),
    ('mtp-only', {MANIFEST: 'fixture.ipc', SCOPE: 'mtp'}, 1),
    ('base', {MANIFEST: 'fixture.ipc', SCOPE: 'base'}, 0),
    ('both', {MANIFEST: 'fixture.ipc', SCOPE: 'both'}, 0),
    ('default', {MANIFEST: 'fixture.ipc'}, 0),
    ('empty-scope', {MANIFEST: 'fixture.ipc', SCOPE: ''}, 0),
    ('invalid-scope', {MANIFEST: 'fixture.ipc', SCOPE: 'invalid'}, 0),
    ('build-disabled', {MANIFEST: 'fixture.ipc', SCOPE: 'mtp', 'DS4_CUDA_BUILD_ARTIFACTS': '0'}, 0),
    ('derived-disabled', {MANIFEST: 'fixture.ipc', SCOPE: 'mtp', 'DS4_CUDA_NO_DERIVED_WEIGHTS': '1'}, 0),
]
for name, controls, count in CASES:
    print(name, flush=True)
    subprocess.run([str(ROOT / 'tests/test_cuda_artifact_scope'), str(count)],
                   cwd=ROOT, env={**BASE_ENV, **controls}, check=True)
print('PASS all artifact producer scopes and raw fallbacks', flush=True)
