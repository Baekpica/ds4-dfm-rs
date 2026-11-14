#!/usr/bin/env python3
"""The iVQ baseline of an Inkling-2 hVQ conversion: the same codebooks and column weights D, t = 0.

At t = 0, H_0 = D (no off-diagonal term, no error feedback), and ds41_hvq's two passes reduce to the iVQ
conversion bit for bit (tests/test_ds41_hvq.py): column-weighted nearest codewords and least-squares row
gains under D. This reads the layer's stored codebook (`CONVERT/layer-LL/codebook.bin`) and diagonal
statistics (`STATS/layer-LL.safetensors`), so it needs no second moments and no calibration pass, and writes
the same layout as run_hvq_inkling2.py (one rank file), so assemble_experts_inkling2.py builds its parts.

    convert_t0_inkling2.py --source $SRC --convert $W/convert --stats $W/stats \
        --out $W/convert-ivq --layers 0-47 [--gpu-rank 0 --gpu-world 4]
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
from safetensors.torch import load_file

import inkling2_ref as R
import inkling2_vq_format as fmt
from run_hvq_inkling2 import GROUPS, SHAPES, factor_safe, pack12, parse_layers


def convert(l, a, dev, hvq, src):
    import ds41_hvq as hv
    from axk2_weighting import Weighting
    rec = json.loads((a.convert / f'layer-{l:02d}' / 'receipt.json').read_text())
    book = (a.convert / f'layer-{l:02d}' / 'codebook.bin').read_bytes()
    cb = torch.from_numpy(np.frombuffer(book, np.uint8).copy()).to(dev).view(torch.float8_e4m3fn).float().reshape(-1, 8)
    stats = {k: v.to(dev) for k, v in load_file(str(a.stats / f'layer-{l:02d}.safetensors')).items()}
    w8 = rec['weighting']
    wt = Weighting('imatrix', stats, device=dev, shrink_tokens=w8['shrink_tokens'], col_floor=w8['col_floor'],
                   col_power=w8['col_power'])
    lw = R.LayerWeights(l, src.read(src.layer_names(l)), dev)
    out_dir = a.out / f'layer-{l:02d}'
    out_dir.mkdir(parents=True, exist_ok=True)
    bid = fmt.book_id(l)
    payloads = {}
    t0 = time.monotonic()
    path = out_dir / 'rank-0.bin'
    with (out_dir / 'rank-0.bin.tmp').open('wb') as f:
        for group, projs in GROUPS.items():
            cols = SHAPES[projs[0]][1]
            for b0 in range(0, R.E, 16):
                es = list(range(b0, min(R.E, b0 + 16)))
                w = torch.stack([(lw.expert(e)[0] if group == 'x' else lw.expert(e)[1]).float() for e in es]).contiguous()
                col = torch.stack([c if (c := wt.col(e, projs[0])) is not None else torch.ones(cols, device=dev) for e in es])
                eye = torch.eye(cols, dtype=torch.float64, device=dev).expand(len(es), cols, cols)
                h = hv.hessian(col, eye, 0.0, rec['damping'])
                v, _, _ = factor_safe(h)
                ids, g = hvq.quantize(w, cb, h, v)
                row0 = 0
                for p in projs:
                    rows = SHAPES[p][0]
                    for i, e in enumerate(es):
                        ip, gp = ids[i, row0:row0 + rows].contiguous(), g[i, row0:row0 + rows].contiguous()
                        payload = fmt.assemble_vq(gp.cpu().numpy().astype('<f2').tobytes(), pack12(ip).cpu().numpy().tobytes(),
                                                  None, rows, cols, 4096, book, bid)
                        payloads[fmt.matrix_name(l, e, p)] = (f.tell(), len(payload), hashlib.sha256(payload).hexdigest())
                        f.write(payload)
                    row0 += rows
                del w, col, h, v, ids, g
    (out_dir / 'rank-0.bin.tmp').rename(path)
    index = [{'name': n, 'offset': payloads[n][0], 'payload_bytes': payloads[n][1], 'payload_sha256': payloads[n][2]}
             for n, _, _ in fmt.expert_inventory(l)]
    with path.open('rb') as f:
        sha = hashlib.file_digest(f, 'sha256').hexdigest()
    (out_dir / 'rank-0.json').write_text(json.dumps({'rank': 0, 'experts': [0, R.E - 1], 'file': path.name,
                                                     'file_sha256': sha, 'index': index}) + '\n')
    (out_dir / 'codebook.bin').write_bytes(book)
    rec2 = {**rec, 't': 0.0, 'method': 'ivq (hvq at t = 0: same codebook, same D)',
            'baseline_of': 'hvq receipt ' + hashlib.sha256((a.convert / f'layer-{l:02d}' / 'receipt.json').read_bytes()).hexdigest(),
            'ranks': None, 'seconds': {'t0_conversion': round(time.monotonic() - t0, 1)}}
    (out_dir / 'receipt.json').write_text(json.dumps(rec2, indent=1) + '\n')
    print(json.dumps({'layer': l, 'seconds': rec2['seconds']}), flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--source', type=Path, required=True,
                    help='the source checkpoint (thinkingmachines/Inkling-2-Small-MXFP8 at R.SOURCE_REV)')
    ap.add_argument('--convert', type=Path, required=True)
    ap.add_argument('--stats', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--layers', default='0-47')
    ap.add_argument('--gpu-rank', type=int, default=0)
    ap.add_argument('--gpu-world', type=int, default=1)
    a = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    import ds41_hvq as hv
    dev = torch.device('cuda')
    hvq, src = hv.Hvq(), R.Source(a.source)
    for l in parse_layers(a.layers):
        if l % a.gpu_world != a.gpu_rank or (a.out / f'layer-{l:02d}' / 'receipt.json').exists():
            continue
        convert(l, a, dev, hvq, src)


if __name__ == '__main__':
    main()
