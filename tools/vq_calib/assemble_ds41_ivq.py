#!/usr/bin/env python3
"""Assemble the DeepSeek-V4.1-Flash iVQ base: the published `vq8sh14-q4k-mtpnative` base
with its 40 routed-expert DQVL v3 blobs replaced by convert_layer_ds41.py's, nothing else.

0. check that the 40 receipts come from one variant (mode, gain refit, Lloyd iterations,
   weighting) and one calibration run (receipts that record none, written before run ids,
   only with --allow-unverified-calibration; listed in assemble-receipt.json);
1. copy the verified published base;
2. patch `blk.{L}.ffn_exps_vq.blob` for L = 0..39 in place (each blob's SHA-256 must match
   its receipt and its DQVL header the published one; sizes are unchanged);
3. check every byte outside the 40 blobs (header, skeleton, MTP towers, padding) against
   the published base, and every patched blob against its blob file;
4. cut the 40 parts at the published byte ranges (24 x 2,838,915,986 B, then
   16 x 2,838,915,985 B) and write SHA256SUMS for the base, the parts, the vision sidecar
   and the Engram shards (whose files are the published release's, unchanged).
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import shutil

import numpy as np

import ds41_vq_format as fmt

NAME = 'DeepSeek-V4.1-Flash-vq8sh14-q4k-mtpnative.gguf'
PART_BASE, PART_EXTRA, NPART = 2838915985, 24, 40
LAYERS = 40
VARIANT_KEYS = ('mode', 'gain_refit', 'codebook_scope')
WEIGHTING_KEYS = ('mode', 'shrink_tokens', 'col_floor', 'col_shrink', 'col_power')    # shrink_units is per layer
UNCHANGED = {  # published SHA256SUMS
    'DeepSeek-V4.1-Flash.vision.gguf': '3a8b194a011e630e8e49a51b5f4e9821e48a9e5674e678d52c0302d6b164698c',
    'deepseek-engram/model-00047-of-00048.safetensors': '76a05106ece259be83d1bfea0dcb54f23377864ee60fd321690f35e6698a2af5',
    'deepseek-engram/model-00048-of-00048.safetensors': '9dc51433e195bc78b5ad846b68643cc0351d3b174e5e4309222d3b7402c83afd',
}
MB = 1 << 20


def sha_range(path, off, n, chunk=64 * MB):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        f.seek(off)
        left = n
        while left:
            b = f.read(min(chunk, left))
            if not b:
                raise IOError(f'{path}: short read at {off + n - left}')
            h.update(b)
            left -= len(b)
    return h.hexdigest()


def same_range(a, b, off, n, chunk=64 * MB):
    with open(a, 'rb') as fa, open(b, 'rb') as fb:
        fa.seek(off)
        fb.seek(off)
        left = n
        while left:
            k = min(chunk, left)
            if fa.read(k) != fb.read(k):
                return False
            left -= k
    return True


def variant_of(r):
    hist = (r.get('training') or {}).get('history')
    w = r.get('weighting') or {}
    return {**{k: r.get(k) for k in VARIANT_KEYS}, **{f'weighting.{k}': w.get(k) for k in WEIGHTING_KEYS},
            'lloyd_iterations': len(hist) if isinstance(hist, list) else None}


def load_receipts(blobs, allow_unverified):
    """-> receipts of layers 0..39, their one variant and their one calibration run (with the layers recording none)."""
    receipts, variants, runs = {}, {}, {}
    for L in range(LAYERS):
        r = json.loads((blobs / f'receipt-{L:02d}.json').read_text())
        if r.get('layer') != L or r.get('tensor') != f'blk.{L}.ffn_exps_vq.blob':
            raise SystemExit(f'receipt-{L:02d}.json is for layer {r.get("layer")} / {r.get("tensor")}')
        receipts[L] = r
        variants.setdefault(json.dumps(variant_of(r), sort_keys=True), []).append(L)
        runs.setdefault((r.get('calibration_run_id'), r.get('calibration_set_sha256')), []).append(L)
    if len(variants) != 1:
        raise SystemExit(f'layers come from different variants (variant: layers): {variants}')
    unverified = runs.pop((None, None), [])
    if len(runs) > 1:
        raise SystemExit(f'layers come from different calibration runs ((run id, calibration set): layers): {runs}')
    if unverified and not allow_unverified:
        raise SystemExit(f'layers {unverified} record no calibration run (converted before run ids); '
                         'pass --allow-unverified-calibration to assemble them anyway')
    (run_id, calib_sha), = runs or [(None, None)]
    return receipts, json.loads(next(iter(variants))), {'run_id': run_id, 'calibration_set_sha256': calib_sha,
                                                        'unverified_layers': unverified}


def part_ranges(total):
    out, off = [], 0
    for i in range(NPART):
        n = PART_BASE + (1 if i < PART_EXTRA else 0)
        out.append((off, n))
        off += n
    assert off == total, (off, total)
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base', required=True, help='assembled published base (SHA-256 checked)')
    p.add_argument('--base-sha256', default='9f3afe6a46ff8de711911dfad5994519a4e5dd57b163f88a5bf4111c308d0e6f')
    p.add_argument('--blobs', type=Path, required=True, help='directory with blob-LL.bin + receipt-LL.json')
    p.add_argument('--out', type=Path, required=True, help='publish directory')
    p.add_argument('--threads', type=int, default=16)
    p.add_argument('--skip-base-hash', action='store_true')
    p.add_argument('--allow-unverified-calibration', action='store_true',
                   help='accept receipts that record no calibration run (listed in assemble-receipt.json)')
    a = p.parse_args()
    receipts_in, variant, calibration = load_receipts(a.blobs, a.allow_unverified_calibration)
    a.out.mkdir(parents=True, exist_ok=True)
    dst = a.out / NAME
    g = fmt.open_base(a.base)
    total = g.size
    log = {'base': a.base, 'blobs': str(a.blobs), 'variant': variant, 'calibration_run': calibration}
    if not a.skip_base_hash:
        h = sha_range(a.base, 0, total)
        assert h == a.base_sha256, (h, a.base_sha256)
    log['base_sha256'] = a.base_sha256
    print(json.dumps({'phase': 'copy', 'bytes': total}), flush=True)
    shutil.copyfile(a.base, dst)
    spans, receipts = [], {}
    for L in range(LAYERS):
        name = f'blk.{L}.ffn_exps_vq.blob'
        r = receipts_in[L]
        blob = np.fromfile(a.blobs / f'blob-{L:02d}.bin', dtype=np.uint8)
        hb = hashlib.sha256(blob.tobytes()).hexdigest()
        assert hb == r['sha256'] and r['tensor'] == name, (L, hb, r['sha256'])
        off = fmt.write_blob(str(dst), name, blob, gguf=g, fsync=False)
        spans.append((off, blob.size, hb))
        receipts[L] = {'sha256': hb, 'offset': off, 'bytes': int(blob.size), 'K': r['K'],
                       'weight_error': r['weight_error']}
        print(json.dumps({'phase': 'patch', 'layer': L, 'offset': off}), flush=True)
    os.sync()
    # bytes outside the 40 blobs must equal the published base; patched blobs their files
    spans.sort()
    gaps, cur = [], 0
    for off, n, _ in spans:
        if off > cur:
            gaps.append((cur, off - cur))
        cur = off + n
    if cur < total:
        gaps.append((cur, total - cur))
    with ThreadPoolExecutor(a.threads) as ex:
        same = list(ex.map(lambda s: same_range(a.base, str(dst), *s), gaps))
        assert all(same), 'bytes outside the blobs differ from the published base'
        back = list(ex.map(lambda s: sha_range(str(dst), s[0], s[1]) == s[2], spans))
        assert all(back), 'a patched blob does not read back'
    log['unchanged_bytes'] = sum(n for _, n in gaps)
    log['patched_bytes'] = sum(n for _, n, _ in spans)
    print(json.dumps({'phase': 'verified', 'unchanged_bytes': log['unchanged_bytes'],
                      'patched_bytes': log['patched_bytes']}), flush=True)
    # parts at the published byte ranges, hashed while written
    ranges = part_ranges(total)

    def cut(i):
        off, n = ranges[i]
        path = a.out / f'{NAME}.part{i + 1:02d}-of-{NPART}'
        h = hashlib.sha256()
        with open(dst, 'rb') as f, open(path, 'wb') as o:
            f.seek(off)
            left = n
            while left:
                b = f.read(min(64 * MB, left))
                h.update(b)
                o.write(b)
                left -= len(b)
        return path.name, h.hexdigest()
    with ThreadPoolExecutor(a.threads) as ex:
        parts = list(ex.map(cut, range(NPART)))
        full = ex.submit(sha_range, str(dst), 0, total).result()
    sums = [(full, NAME)] + [(h, n) for n, h in parts]
    sums += [(h, n) for n, h in UNCHANGED.items()]
    (a.out / 'SHA256SUMS').write_text(''.join(f'{h}  {n}\n' for h, n in sums))
    log.update({'sha256': full, 'parts': dict((n, h) for n, h in parts), 'layers': receipts})
    (a.out.parent / 'assemble-receipt.json').write_text(json.dumps(log, indent=1) + '\n')
    print(json.dumps({'phase': 'done', 'sha256': full}), flush=True)


if __name__ == '__main__':
    main()
