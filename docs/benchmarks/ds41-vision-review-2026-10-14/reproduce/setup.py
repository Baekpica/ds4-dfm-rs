#!/usr/bin/env python3
"""Restore the recorded harness paths; never replace different local files."""
import hashlib
import json
from pathlib import Path

here = Path(__file__).resolve().parent
root = here.parents[3]
manifest = json.loads((here / "manifest.json").read_text())
for source, entry in manifest.items():
    data = (here / source).read_bytes()
    if hashlib.sha256(data).hexdigest() != entry["sha256"]:
        raise SystemExit(f"archive hash mismatch: {source}")
    target = root / entry["target"]
    if target.exists() and target.read_bytes() != data:
        raise SystemExit(f"refusing to replace different file: {target}")
for source, entry in manifest.items():
    target = root / entry["target"]
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes((here / source).read_bytes())
print(f"Restored {len(manifest)} recorded inputs; no build or GPU work started.")
