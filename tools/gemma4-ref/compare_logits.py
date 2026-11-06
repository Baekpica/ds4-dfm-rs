"""Compare the port's next-token logits against the reference dump.

The port side is the JSON written by `ds4 --dump-logits`; the reference side is
the float32 .npy written by ref_logits.py. Both carry the full vocabulary and
both are post-softcap, so the comparison is direct.

Usage:
  python compare_logits.py --ref FILE.npy --port FILE.json
"""

import argparse
import json

import numpy as np


def softmax(x):
    z = x - x.max()
    e = np.exp(z)
    return e / e.sum()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ref", required=True)
    parser.add_argument("--port", required=True)
    args = parser.parse_args()

    ref = np.load(args.ref).astype(np.float64)
    with open(args.port, "r", encoding="utf-8") as fp:
        dump = json.load(fp)
    port = np.asarray(dump["logits"], dtype=np.float64)
    if ref.shape != port.shape:
        raise SystemExit(f"shape mismatch: ref {ref.shape} port {port.shape}")

    ref_top = np.argsort(-ref)[:10]
    port_top = np.argsort(-port)[:10]
    overlap = len(set(ref_top.tolist()) & set(port_top.tolist()))
    port_rank_of_ref = int((port > port[ref_top[0]]).sum())

    p_ref = softmax(ref)
    p_port = softmax(port)
    kl_ref_port = float(np.sum(p_ref * (np.log(p_ref) - np.log(p_port))))
    kl_port_ref = float(np.sum(p_port * (np.log(p_port) - np.log(p_ref))))
    cosine = float(np.dot(ref, port) / (np.linalg.norm(ref) * np.linalg.norm(port)))

    delta = port - ref
    top100 = np.argsort(-ref)[:100]

    print(f"vocab            {ref.shape[0]}")
    print(f"ref argmax       {int(ref_top[0])}  logit {ref[ref_top[0]]:.4f}")
    print(f"port argmax      {int(port_top[0])}  logit {port[port_top[0]]:.4f}")
    print(f"argmax match     {int(ref_top[0]) == int(port_top[0])}")
    print(f"port rank of ref argmax  {port_rank_of_ref}")
    print(f"top10 overlap    {overlap}/10")
    print(f"cosine           {cosine:.6f}")
    print(f"KL ref||port     {kl_ref_port:.6f}")
    print(f"KL port||ref     {kl_port_ref:.6f}")
    print(f"max |delta|      {float(np.abs(delta).max()):.6f}")
    print(f"rms delta (all)  {float(np.sqrt(np.mean(delta**2))):.6f}")
    print(f"rms delta (top100) {float(np.sqrt(np.mean(delta[top100]**2))):.6f}")
    print("ref top10:  " + " ".join(f"{int(i)}:{ref[i]:.3f}" for i in ref_top))
    print("port top10: " + " ".join(f"{int(i)}:{port[i]:.3f}" for i in port_top))


if __name__ == "__main__":
    main()
