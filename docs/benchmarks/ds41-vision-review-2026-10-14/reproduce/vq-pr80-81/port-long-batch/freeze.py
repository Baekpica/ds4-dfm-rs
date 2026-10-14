#!/usr/bin/env python3
"""Hash frozen link inputs; never compile or execute production inference."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

mode, output, *names = sys.argv[1:]
paths = [Path(name).resolve() for name in names]
if not paths or any(not path.is_file() for path in paths):
    raise SystemExit("missing frozen input; refuse production rebuild")

def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

receipt = {
    "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
    "inputs": {str(path): sha(path) for path in paths},
    "model_or_GPU_execution": False,
}
dest = Path(output)
if mode == "before":
    dest.write_text(json.dumps(receipt, indent=2) + "\n")
elif mode == "after":
    if json.loads(dest.read_text()) != receipt:
        raise SystemExit("frozen native/Rust link inputs changed")
    print(f"PASS frozen link inputs: {len(paths)}")
else:
    raise SystemExit("expected before or after")
