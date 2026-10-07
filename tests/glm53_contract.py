#!/usr/bin/env python3
"""Sequential GLM artifact contract runner/comparator; never manages servers.

Run only after the central CUDA build and owner/memory inspection. A supplied
artifact hash reuses the caller's verified download identity instead of hashing
94 GB again. This harness records numeric differences; nonexact cross-width
arithmetic still needs the model-family numerical/answer review.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import struct
import subprocess
import time

import numpy as np

FIXTURE = Path(__file__).parent / "fixtures/glm53/contract.json"
MTP_TAGS = {0x354D4347, 0x354D4547}
COMPACT_TAGS = {0x35434C47, 0x354D4347}
EXPANDED_TAGS = {0x35454C47, 0x354D4547}
CHUNK = 262144
IDENTITY_KEYS = ("binary_sha256", "model_sha256", "model_stat", "fixture_sha256",
                 "mtp", "vision", "vision_sha256", "vision_stat", "source_manifest_sha256")


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def small_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def file_stat(path):
    stat = path.stat()
    return {"bytes": stat.st_size, "inode": stat.st_ino, "device": stat.st_dev,
            "mtime_ns": stat.st_mtime_ns, "ctime_ns": stat.st_ctime_ns}


def run(args, fixture):
    model = args.model.resolve()
    expected = fixture["artifact"]
    assert model.stat().st_size == expected["bytes"], "artifact byte count differs"
    assert args.model_sha256 == expected["sha256"], "supplied artifact identity differs"
    vision = args.vision.resolve() if args.vision else None
    if vision:
        assert vision.stat().st_size == fixture["vision_artifact"]["bytes"], "Vision byte count differs"
        assert args.vision_sha256 == fixture["vision_artifact"]["sha256"], "Vision identity differs"
    if args.artifact_stat:
        receipt = json.loads(args.artifact_stat.read_text())["files"]
        for name, path, digest in (("main", model, args.model_sha256),
                                   ("vision", vision, args.vision_sha256)):
            if path is None:
                continue
            assert Path(receipt[name]["path"]).resolve() == path
            assert receipt[name]["sha256"] == digest
            assert all(receipt[name][key] == value for key, value in file_stat(path).items()), "verified file changed"
    binary = args.binary.resolve()
    identity = {"binary": str(binary), "binary_sha256": small_hash(binary),
                "model": str(model), "model_bytes": model.stat().st_size,
                "model_sha256": args.model_sha256,
                "model_stat": file_stat(model),
                "model_sha256_source": "caller supplied previously verified download identity",
                "artifact": expected, "fixture_sha256": small_hash(args.fixture),
                "source_head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                "source_dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], text=True)),
                "mtp": not args.no_mtp, "vision": str(vision) if vision else None,
                "vision_sha256": args.vision_sha256, "vision_stat": file_stat(vision) if vision else None,
                "source_manifest_sha256": small_hash(args.source_manifest) if args.source_manifest else None}
    identity_path = args.output / "identity.json"
    if identity_path.exists():
        old = json.loads(identity_path.read_text())
        for key in IDENTITY_KEYS:
            assert old[key] == identity[key], f"{key} changed; use a fresh evidence directory"
    else:
        write_json(identity_path, identity)
    (args.output / "prompt.txt").write_text(fixture["prompt"])
    subprocess.run(["nvidia-smi", "--query-gpu=name,uuid,clocks.sm,clocks.mem,memory.used,memory.free",
                    "--format=csv"], stdout=(args.output / "gpu-before.csv").open("w"), check=True)
    arms = args.arms or [f"{mode}-r{rows}" for mode in fixture["modes"] for rows in fixture["prefill_rows"]]
    for arm in arms:
        mode, raw_rows = arm.split("-r")
        rows = int(raw_rows)
        assert (mode in fixture["modes"] and rows in fixture["prefill_rows"]) or (mode == "expanded" and rows == 1)
        target = args.output / arm
        target.mkdir()
        write_json(target / "identity.json", identity)
        env = os.environ.copy()
        env.update({"DS4_GLM53_PREFILL_ROWS": str(rows), "DS4_GLM53_DSA_EXPANDED": str(int(mode == "expanded")),
                    "DS4_SESSION_LAZY_GRAPH": "1", "DS4_SERVER_FORK_PARTIAL": "1",
                    "DS4_GLM53_CONTRACT_SSD": str(int(mode == "ssd")),
                    "DS4_GLM53_CONTRACT_MTP": str(int(not args.no_mtp)),
                    "DS4_GLM53_CONTRACT_OUT": str(target.resolve()),
                    "DS4_GLM53_CONTRACT_PROMPT": str((args.output / "prompt.txt").resolve())})
        if args.no_mtp:
            env["DS4_MTP_SPEC_DISABLE"] = "1"
        else:
            env.pop("DS4_MTP_SPEC_DISABLE", None)
        command = [str(binary), str(model)] + ([str(args.vision.resolve())] if args.vision else [])
        write_json(target / "command.json", {"argv": command, "env": {k: v for k, v in env.items()
                   if k.startswith("DS4_") and not any(x in k for x in ("TOKEN", "KEY", "SECRET"))}})
        start = time.monotonic()
        with (target / "stdout.log").open("w") as out, (target / "stderr.log").open("w") as err:
            result = subprocess.run(command, env=env, stdout=out, stderr=err, timeout=args.timeout)
        write_json(target / "process.json", {"returncode": result.returncode,
                   "wall_seconds": time.monotonic() - start, "profiling_scope": "contract verification, not speed proof"})
        if result.returncode:
            raise RuntimeError(f"{arm} failed; inspect {target / 'stderr.log'}")
        assert file_stat(model) == identity["model_stat"], "model changed during run"
        assert vision is None or file_stat(vision) == identity["vision_stat"], "Vision changed during run"
        summary = json.loads((target / "contract.json").read_text())
        assert summary["prefill_rows"] == rows and summary["ssd"] == (mode == "ssd"), summary
    subprocess.run(["nvidia-smi", "--query-gpu=name,uuid,clocks.sm,clocks.mem,memory.used,memory.free",
                    "--format=csv"], stdout=(args.output / "gpu-after.csv").open("w"), check=True)


def view(path):
    mapped = np.memmap(path, mode="r", dtype=np.uint8)
    header = struct.unpack_from("<13I", mapped)
    assert header[0] == 0x34565344 and header[1] == 3
    assert header[5] in COMPACT_TAGS | EXPANDED_TAGS, "unknown GLM cache representation"
    compact = header[5] in COMPACT_TAGS
    ctx, latent, dsa, row_bytes, n, layers, conv, pooldim, vocab, state = (
        header[2], header[3], header[4], header[6], header[7], header[8],
        header[9], header[10], header[11], header[12])
    assert ctx == 2048 and latent == 512 and dsa == 11
    assert row_bytes == (1024 if compact else 65536)
    assert layers == 45 and conv == 4 and pooldim == 128 and vocab == 154880
    off = 52
    tokens = np.ndarray((n,), dtype="<u4", buffer=mapped, offset=off); off += n * 4
    regions = {"logits": np.ndarray((vocab,), dtype="<f4", buffer=mapped, offset=off)}; off += vocab * 4
    cursors = (0, 0)
    if header[5] in MTP_TAGS:
        cursors = struct.unpack_from("<2I", mapped, off); off += 8
        assert cursors[1] <= cursors[0] < n
    state_start = off
    for layer in range(layers):
        if layer % 4 != 3:
            for name, width in (("recurrent", 64 * 128 * 128), ("q_conv", 64 * 128 * conv),
                                ("k_conv", 64 * 128 * conv), ("v_conv", 64 * 128 * conv)):
                regions[f"layer{layer}/{name}"] = np.ndarray((width,), dtype="<f4", buffer=mapped, offset=off)
                off += width * 4
        else:
            for name in ("pool_tail_k", "pool_tail_gate"):
                regions[f"layer{layer}/{name}"] = np.ndarray((4 * pooldim,), dtype="<f4", buffer=mapped, offset=off)
                off += 4 * pooldim * 4
    regions["last_hidden"] = np.ndarray((4096,), dtype="<f4", buffer=mapped, offset=off); off += 4096 * 4
    regions["mtp_hidden"] = np.ndarray((4096,), dtype="<f4", buffer=mapped, offset=off); off += 4096 * 4
    assert off - state_start == state, "state layout differs; update source schema deliberately"
    for layer in range(3, layers, 4):
        name = "latent" if compact else "expanded_kv"
        regions[f"layer{layer}/{name}"] = np.ndarray((n * row_bytes // 2,), dtype="<f2", buffer=mapped, offset=off)
        off += n * row_bytes
        if compact:
            regions[f"layer{layer}/pool"] = np.ndarray(((n // 4) * pooldim,), dtype="<f2", buffer=mapped, offset=off)
            off += (n // 4) * pooldim * 2
    if header[5] in MTP_TAGS:
        width = (cursors[0] - cursors[1]) * latent
        regions["mtp_kv"] = np.ndarray((width,), dtype="<f2", buffer=mapped, offset=off); off += width * 2
    assert off == mapped.size, "payload byte count differs from full frontier"
    return {"header": header, "tokens": tokens, "cursors": cursors, "regions": regions,
            "mapped": mapped, "compact": compact}


def metrics(a, b):
    assert a.shape == b.shape
    max_abs = sum_sq = norm_sq = 0.0
    exact = True
    for off in range(0, a.size, CHUNK):
        x = a[off:off + CHUNK].astype(np.float64)
        y = b[off:off + CHUNK].astype(np.float64)
        assert np.isfinite(x).all() and np.isfinite(y).all(), "nonfinite state/logits"
        delta = x - y
        exact = exact and np.array_equal(x, y)
        max_abs = max(max_abs, float(np.max(np.abs(delta), initial=0.0)))
        sum_sq += float(delta @ delta); norm_sq += float(x @ x)
    return {"values": int(a.size), "exact": exact, "max_abs": max_abs,
            "rms": (sum_sq / max(a.size, 1)) ** 0.5,
            "relative_l2": (sum_sq / max(norm_sq, 1e-300)) ** 0.5}


def compare_pair(left, right):
    common = sorted(p.name for p in left.glob("*.payload.bin") if (right / p.name).exists())
    result = {}
    for name in common:
        a, b = view(left / name), view(right / name)
        matched = a["header"][7] == b["header"][7] and a["cursors"] == b["cursors"] and np.array_equal(a["tokens"], b["tokens"])
        item = {"same_frontier_and_tokens": bool(matched), "regions": {}}
        if matched:
            for region in a["regions"].keys() & b["regions"].keys():
                if a["compact"] != b["compact"] and "pool_tail" in region:
                    continue
                item["regions"][region] = metrics(a["regions"][region], b["regions"][region])
            item["prefill_argmax_same"] = bool(np.argmax(a["regions"]["logits"]) == np.argmax(b["regions"]["logits"]))
        result[name] = item
    for name in ("greedy.tokens.txt", "sampled.tokens.txt", "vision-greedy.tokens.txt"):
        if (left / name).exists() and (right / name).exists():
            result[name] = {"tokens_identical": (left / name).read_bytes() == (right / name).read_bytes()}
    return result


def compare(args, fixture):
    arms = [f"{mode}-r{rows}" for mode in fixture["modes"] for rows in fixture["prefill_rows"]]
    present = [arm for arm in arms if (args.output / arm / "contract.json").exists()]
    if (args.output / "expanded-r1/contract.json").exists():
        present.append("expanded-r1")
    report = {"artifact": fixture["artifact"], "automated_status": "incomplete",
              "qualification_status": "pending numerical and answer review",
              "long_context_status": fixture["long_context"]["status"], "arms": {}, "pairs": {}}
    for arm in present:
        directory = args.output / arm
        arm_identity = json.loads((directory / "identity.json").read_text())
        base_identity = json.loads((args.output / "identity.json").read_text())
        for key in IDENTITY_KEYS:
            assert arm_identity[key] == base_identity[key], f"unmatched identity in {arm}"
        report["arms"][arm] = json.loads((directory / "contract.json").read_text())
        for path in directory.glob("*.payload.bin"):
            data = view(path)
            for array in data["regions"].values():
                for off in range(0, array.size, CHUNK):
                    assert np.isfinite(array[off:off + CHUNK]).all(), path
        for path in directory.glob("*.logits.f32"):
            values = np.memmap(path, mode="r", dtype="<f4")
            assert values.size == 154880 and np.isfinite(values).all(), path
    pairs = [("resident-r1", "ssd-r1"), ("resident-r128", "ssd-r128"),
             ("resident-r1", "resident-r128"), ("ssd-r1", "ssd-r128"),
             ("resident-r1", "expanded-r1")]
    for left, right in pairs:
        if left in present and right in present:
            report["pairs"][f"{left}__{right}"] = compare_pair(args.output / left, args.output / right)
    if all(arm in present for arm in arms):
        report["automated_status"] = "all native contract arms passed; numeric comparisons recorded"
    report["answer_review"] = fixture["answer_review"]
    write_json(args.output / "comparison.json", report)
    print(report["automated_status"] + "; qualification remains review-bounded")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("run", "compare", "plan-long"))
    parser.add_argument("--fixture", type=Path, default=FIXTURE)
    parser.add_argument("--binary", type=Path, default=Path("tests/test_glm53_session"))
    parser.add_argument("--model", type=Path)
    parser.add_argument("--model-sha256", help="reuse a verified download identity; required for run")
    parser.add_argument("--vision", type=Path)
    parser.add_argument("--vision-sha256", help="reuse the verified Vision sidecar identity")
    parser.add_argument("--artifact-stat", type=Path, help="guard the file identity captured after SHA256 verification")
    parser.add_argument("--source-manifest", type=Path, help="exact dirty source manifest for this build")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--arms", nargs="+")
    parser.add_argument("--no-mtp", action="store_true", help="diagnostic partial scope")
    parser.add_argument("--timeout", type=int, default=1800)
    args = parser.parse_args()
    fixture = json.loads(args.fixture.read_text())
    args.output.mkdir(parents=True, exist_ok=True)
    if args.phase == "run":
        if not args.model or not args.model_sha256:
            parser.error("run needs --model and --model-sha256")
        if args.vision and not args.vision_sha256:
            parser.error("--vision requires --vision-sha256")
        run(args, fixture)
    elif args.phase == "compare":
        compare(args, fixture)
    else:
        write_json(args.output / "long-context-plan.json", fixture["long_context"])


if __name__ == "__main__":
    main()
