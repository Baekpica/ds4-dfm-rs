#!/usr/bin/env python3
"""Engram rows of a calibration set, gathered once (DeepSeek-V4.1-Flash, layers 1 and 14).

A layer's Engram table is 98 GiB (384M rows x 256 B) and a forward needs (ngram - 1) x heads
= 24 rows per token. ds41_ref.Engram.rows gathers them through a memmap, which needs the
table in the page cache; a container whose memory limit also holds the residual streams of
the whole calibration set cannot give that, and every chunk then waits on the disk.

build() makes one sequential pass over the table and writes the rows the set needs in the
order the collector reads them (shard by shard, token by token): 6.3 KiB per token.
EngramCache.rows() returns a chunk as ds41_ref.Engram.rows does.

  ds41_engram_cache.py --calib calib.safetensors --world 8 --out DIR [--layer 1]
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
import torch

BF = torch.bfloat16
BLOCK_BYTES = 1 << 30
SUFFIX = {'weight': 'w', 'scale': 's'}


def emit(v):
    print(json.dumps(v), flush=True)


def table_span(src, name):
    """-> (file, byte offset of the tensor data, rows, bytes per row) of a 2-D uint8-like tensor."""
    f = src.loc[name]
    n, header = src.hdr[f]
    a, b = header[name]['data_offsets']
    rows = header[name]['shape'][0]
    return f, 8 + n + a, rows, (b - a) // rows


def gather_sorted(path, base, rows, row_bytes, ids, block_bytes=BLOCK_BYTES):
    """Rows `ids` [n] (any order, repeats allowed) of a table stored at `base` in `path` -> uint8 [n, row_bytes],
    read in ascending blocks; the page cache of each block is dropped behind the read."""
    order = np.argsort(ids, kind='stable')
    sorted_ids = ids[order]
    if len(ids) and (sorted_ids[0] < 0 or sorted_ids[-1] >= rows):
        raise ValueError(f'row ids outside [0, {rows})')
    out = np.empty((len(ids), row_bytes), np.uint8)
    block_rows = max(1, block_bytes // row_bytes)
    buf = bytearray(block_rows * row_bytes)
    view = memoryview(buf)
    fd = os.open(path, os.O_RDONLY)
    try:
        os.posix_fadvise(fd, base, rows * row_bytes, os.POSIX_FADV_SEQUENTIAL)
        for r0 in range(0, rows, block_rows):
            lo, hi = np.searchsorted(sorted_ids, (r0, r0 + block_rows))
            if lo == hi:
                continue
            want = min(block_rows, rows - r0) * row_bytes
            got = 0
            while got < want:
                k = os.preadv(fd, [view[got:want]], base + r0 * row_bytes + got)
                if k <= 0:
                    raise IOError(f'{path}: short read at {base + r0 * row_bytes + got}')
                got += k
            block = np.frombuffer(buf, np.uint8, want).reshape(-1, row_bytes)
            out[order[lo:hi]] = block[sorted_ids[lo:hi] - r0]
            os.posix_fadvise(fd, base + r0 * row_bytes, want, os.POSIX_FADV_DONTNEED)
    finally:
        os.close(fd)
    return out


class EngramCache:
    """Engram rows of every token of a set, in collector order; one pair of files per Engram layer."""

    def __init__(self, directory, key):
        """key: what the rows belong to (calibration content, shards, token cap); a cache built for another
        key is not used."""
        self.dir, self.key = Path(directory), key
        self.fd = {}

    def path(self, L, kind):
        return self.dir / f'engram-{L:02d}.{SUFFIX[kind]}.u8'

    def meta_path(self, L):
        return self.dir / f'engram-{L:02d}.json'

    def ready(self, L):
        try:
            meta = json.loads(self.meta_path(L).read_text())
        except OSError:
            return None
        if meta.get('key') != self.key:
            return None
        for kind in SUFFIX:
            if not self.path(L, kind).exists() or self.path(L, kind).stat().st_size != meta['bytes'][kind]:
                return None
        return meta

    def build(self, src, L, ids):
        """ids int64 [tokens, cols]: the table rows of every token, in collector order."""
        t0 = time.monotonic()
        self.dir.mkdir(parents=True, exist_ok=True)
        flat = np.ascontiguousarray(ids.reshape(-1))
        meta = {'key': self.key, 'layer': L, 'tokens': int(ids.shape[0]), 'cols': int(ids.shape[1]),
                'ids_sha256': hashlib.sha256(flat).hexdigest(), 'bytes': {}, 'row_bytes': {}}
        for kind in SUFFIX:
            f, base, rows, row_bytes = table_span(src, f'layers.{L}.engram.embed.{kind}')
            out = gather_sorted(f, base, rows, row_bytes, flat)
            tmp = self.path(L, kind).with_suffix('.tmp')
            with open(tmp, 'wb') as fh:
                fh.write(memoryview(out).cast('B'))
                fh.flush()
                os.fsync(fh.fileno())
                os.posix_fadvise(fh.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
            tmp.rename(self.path(L, kind))
            meta['bytes'][kind], meta['row_bytes'][kind] = int(out.size), int(row_bytes)
            del out
            emit({'phase': 'engram-cache', 'layer': L, 'table': kind, 'seconds': round(time.monotonic() - t0, 1)})
        self.meta_path(L).write_text(json.dumps(meta) + '\n')
        return meta

    def open(self, L):
        meta = self.ready(L)
        if meta is None:
            raise FileNotFoundError(f'no Engram cache for layer {L} of this set under {self.dir}')
        self.fd[L] = {kind: os.open(self.path(L, kind), os.O_RDONLY) for kind in SUFFIX}
        self.meta = meta
        return meta

    def close(self, L):
        for fd in self.fd.pop(L, {}).values():
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
            os.close(fd)

    def read(self, L, kind, start, stop):
        """Rows of tokens [start, stop) -> uint8 [(stop - start) * cols, row_bytes]."""
        cols, rb = self.meta['cols'], self.meta['row_bytes'][kind]
        n = (stop - start) * cols * rb
        out = np.empty(n, np.uint8)
        view = memoryview(out)
        got, off = 0, start * cols * rb
        while got < n:
            k = os.preadv(self.fd[L][kind], [view[got:]], off + got)
            if k <= 0:
                raise IOError(f'{self.path(L, kind)}: short read at {off + got}')
            got += k
        return out.reshape(-1, rb)

    def rows(self, L, start, stop, dev):
        """BF16 [stop - start, cols * 256] on the device, as ds41_ref.Engram.rows."""
        n, c = stop - start, self.meta['cols']
        v = torch.from_numpy(self.read(L, 'weight', start, stop)).to(dev).view(torch.float8_e4m3fn).float()
        s = torch.from_numpy(self.read(L, 'scale', start, stop)).to(dev).view(torch.float8_e8m0fnu).float()
        x = v.view(n * c, -1, 32) * s.unsqueeze(-1)
        return x.view(n, -1).to(BF)


def set_key(content_sha256, world, max_tokens):
    return {'calib_content_sha256': content_sha256, 'world': world, 'max_tokens': max_tokens,
            'order': 'shards 0..world-1 of run_ds41.shard, rows ascending, tokens in row order'}


def shard_tokens(ids, off, world, max_tokens=0):
    """-> per shard: (rows, [token tensors])."""
    from run_ds41 import shard
    out = []
    for r in range(world):
        rows = shard(off, r, world, max_tokens or None)
        out.append((rows, [ids[int(off[i]):int(off[i + 1])] for i in rows]))
    return out


def content_sha256(data):
    """sha256(input_ids bytes + offsets bytes): a set's identity (file bytes vary with metadata order)."""
    return hashlib.sha256(data['input_ids'].numpy().tobytes() + data['offsets'].numpy().tobytes()).hexdigest()


def main():
    from safetensors.torch import load_file
    from ds41_ref import EngramHash, Source
    from run_ds41 import ROOT
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--source', default=ROOT)
    p.add_argument('--tok', default=ROOT)
    p.add_argument('--calib', type=Path, required=True)
    p.add_argument('--world', type=int, default=8)
    p.add_argument('--max-tokens', type=int, default=0)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--layer', type=int, action='append', help='Engram layer(s); default: all')
    a = p.parse_args()
    src = Source(a.source)
    args = src.args
    data = load_file(str(a.calib))
    cache = EngramCache(a.out, set_key(content_sha256(data), a.world, a.max_tokens))
    layers = [L for L in args['engram_layer_ids'] if not a.layer or L in a.layer]
    todo = [L for L in layers if cache.ready(L) is None]
    if not todo:
        emit({'phase': 'engram-cache', 'ready': layers})
        return
    hasher = EngramHash(args, a.tok, 'cpu')
    hashes = torch.cat([hasher(t) for _, toks in shard_tokens(data['input_ids'], data['offsets'], a.world, a.max_tokens)
                        for t in toks]).numpy()                               # [tokens, engram layers, cols]
    for L in todo:
        cache.build(src, L, hashes[:, args['engram_layer_ids'].index(L)])


if __name__ == '__main__':
    main()
