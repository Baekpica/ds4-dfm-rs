#!/usr/bin/env python3
"""Score native full-vocabulary rows against the pinned MQ87 BF16 reference.

Requires NumPy and the private handoff. Start a shared weight owner first.
This reports arithmetic agreement; representative generation is a separate gate.
"""
import argparse
import hashlib
import io
import json
from pathlib import Path
import struct
import subprocess
import tarfile

import numpy as np

MANIFEST_SHA = "bf17afc89b8ad49803a35915de2190c9d93f0db2a2a6be9e6d6a925763a81a49"
REFERENCE_SHA = "5f439581b8f65961cdc582d10ba8d82379cbb217878a8cfd6714f1835c84620a"
VOCAB = 152576


def log_probs(x):
    x = x.astype(np.float64)
    x -= x.max(axis=-1, keepdims=True)
    return x - np.log(np.exp(x).sum(axis=-1, keepdims=True))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True)
    p.add_argument("--handoff", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--runner", default="tests/naive_gate")
    p.add_argument("--ids", nargs="+", default=["eval-clean-v2-00000", "eval-clean-v2-00032"])
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=False)
    manifest_path = a.handoff / "calibration/adaptive-inputs/clean-eval-v2/manifest.json"
    manifest_bytes = manifest_path.read_bytes()
    assert hashlib.sha256(manifest_bytes).hexdigest() == MANIFEST_SHA
    records = {r["id"]: r for r in json.loads(manifest_bytes)["holdout"]}
    ref_path = a.handoff / "calibration/main-gguf-quality/rank-0/holdout-gguf-logits.safetensors"
    raw = ref_path.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == REFERENCE_SHA
    head_size = struct.unpack_from("<Q", raw)[0]
    head = json.loads(raw[8:8 + head_size])
    data = memoryview(raw)[8 + head_size:]

    def tensor(name, dtype):
        entry = head[name]
        start, end = entry["data_offsets"]
        return np.frombuffer(data[start:end], dtype=dtype).reshape(entry["shape"])

    cases = []
    with tarfile.open(a.handoff / "calibration/adaptive-inputs.tar.gz") as archive:
        for name in a.ids:
            record = records[name]
            stream = archive.extractfile("adaptive-inputs/clean-eval-v2/" + record["token_file"])
            tokens = np.load(io.BytesIO(stream.read()))
            # The manifest hashes the I32 token array, excluding its NPY header.
            assert tokens.dtype == np.dtype("<i4")
            assert hashlib.sha256(tokens.tobytes()).hexdigest() == record["tokens_sha256"]
            assert len(tokens) == record["tokens"]
            positions = tensor(name + ".positions", "<i8")
            assert np.array_equal(tokens[positions + 1], tensor(name + ".targets", "<i8"))
            cases.append({"id": name, "tokens": tokens.tolist(), "positions": positions.tolist()})
    case_path = a.out / "cases.json"
    case_path.write_text(json.dumps(cases))
    with (a.out / "native.log").open("w") as log:
        subprocess.run([a.runner, a.model, str(case_path), str(a.out / "rows")],
                       stdout=log, stderr=subprocess.STDOUT, check=True)

    scores = []
    for case in cases:
        name = case["id"]
        native = np.fromfile(a.out / "rows" / (name + ".f32"), dtype="<f4")
        native = native.reshape(len(case["positions"]), VOCAB)
        reference = (tensor(name + ".logits", "<u2").astype(np.uint32) << 16).view(np.float32)
        assert reference.shape == native.shape and np.isfinite(native).all()
        lp = log_probs(reference)
        lq = log_probs(native)
        targets = tensor(name + ".targets", "<i8")
        rows = np.arange(len(targets))
        delta = native.astype(np.float64) - reference
        kl = (np.exp(lp) * (lp - lq)).sum(axis=-1)
        agreement = native.argmax(axis=-1) == reference.argmax(axis=-1)
        scores.append({"id": name, "prompt_tokens": len(case["tokens"]),
                       "positions": case["positions"], "finite": True,
                       "relative_l2": float(np.linalg.norm(delta) / np.linalg.norm(reference)),
                       "max_abs": float(np.abs(delta).max()),
                       "mean_kl": float(kl.mean()), "max_kl": float(kl.max()),
                       "top1_agree": int(agreement.sum()), "rows": len(targets),
                       "reference_nll": float(-lp[rows, targets].mean()),
                       "native_nll": float(-lq[rows, targets].mean())})
    report = {"manifest_sha256": MANIFEST_SHA, "reference_sha256": REFERENCE_SHA,
              "reference": "canonical GGUF decode to BF16; native MMQ arithmetic differs",
              "scores": scores}
    (a.out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
