"""ds41_vq_format.py -- reader / exact writer / dequantizer for the routed-expert
VQ blobs (YoungAi "DQVL v3") inside the DeepSeek-V4.1-Flash-1017
``vq8sh14-q4k-mtpnative`` base GGUF, plus an FP4 loader for the original
checkpoint.  numpy / torch only (safetensors headers are parsed by hand so a
shard that is still downloading is simply skipped).

Producer: YoungAi @ 7fb26cc ``v41_quantize --skel q4k --mtp-vq --vq-nc 4096
--vq-nc-layers 0:14=8192 --vq-shared-cb --vq-cb-fp8`` then ``v41_to_gguf``
(blob bytes: gguf-tools/quantize/v41_to_gguf_vq3.inc.c gen_vqblob_v3,
13-bit split: v41_vq3_bits.c vq3_split_row).  Consumer: ds4-dfm-rs
crates/ds4-core/src/vq.rs, ds41_vq_fmt.h.

GGUF CONTAINER
--------------
GGUF v3, little endian, ``general.alignment`` = 32.  Tensor offsets in the
tensor-info table are relative to data0 = align32(end of tensor-info table);
every tensor's data is zero-padded to a multiple of 32.  The published file is
40 byte-range parts (parts 1-24: 2,838,915,986 B, 25-40: 2,838,915,985 B;
113,556,639,424 B total); :class:`GGUFFile` reads them as one virtual file.

Expert tensors: exactly 43 tensors of ggml type 42 (``DS4_GGT_VQBLOB``),
n_dims = 1, ne[0] = byte count (opaque bytes):
    blk.{L}.ffn_exps_vq.blob   L = 0..39   (nexp = 384)
    mtp.{T}.ffn_exps_vq.blob   T = 0..2    (nexp = 128, MTP/DSpark towers)
There are no separate codebook / gain tensors.  (Shared experts are separate
q4_K/F8 tensors ``*_shexp*``; not handled here.)

BLOB LAYOUT (all integers little endian; offsets relative to blob start)
-----------------------------------------------------------------------
  0   u32 magic 0x4C565144 'DQVL'
  4   u32 ver = 3
  8   u32 L      (layer index; tower index T for mtp blobs)
  12  u32 nexp   (384 / 128)
  16  u64 slot[nexp][3]: payload offset of (expert e, matrix w); w = 0: w1
      (gate, [2304, 5120]), 1: w3 (up, [2304, 5120]), 2: w2 (down,
      [5120, 2304]); shapes are [rows = out, cols = in].
  cb_off = align16(16 + 24 * nexp)          (9232 for nexp 384, 3088 for 128)
  cb_off: K * 8 bytes, E4M3FN codebook [K][8] row-major (codeword k = bytes
      8k .. 8k+7).  ONE codebook per blob, shared by every expert and all
      three matrices.  E4M3FN: sign bit 7, exp bits 6..3 (bias 7), mant bits
      2..0, exp 0 = subnormal m * 2^-9, 0x7f/0xff = NaN; zero always 0x00.
  body = align16(cb_off + 8 * K)            (all padding bytes are zero)
  body: payloads back to back, order e = 0..nexp-1 outer, w = w1, w3, w2
      inner; NO alignment between payloads.  Payload:
        +0   u32 magic 0x33565144 'DQV3'
        +4   u16 dim = 8
        +6   u16 nc  = K (4096 or 8192)
        +8   u32 rows
        +12  u32 cols
        +16  u32 flags: bit0 = 1 (codebook is E4M3), bit1 = 13th-bit plane
             present (K = 8192)          -> 3 on layers 0-13, 1 elsewhere
        +20  u32 main_nbit = 12
        +24  u64 cb_off (blob relative)
        +32  f16 gain[rows]              (one per output row)
        then main stream  rows * mrow bytes, mrow = ceil(12 * nidx / 8)
        then bit plane    rows * prow bytes, prow = ceil(nidx / 8)  (if bit1)
        then 8 zero bytes
      nidx = cols / 8 indices per row; index k of row r codes the 8
      consecutive weights W[r, 8k : 8k+8] (consecutive along the input dim).
      Main stream: per row an LSB-first bit stream, index k occupies bits
      [12k, 12k+12) -> for a pair (a, b) = (idx[2j], idx[2j+1]):
          byte 3j = a & 0xff, 3j+1 = (a >> 8) | (b & 0xf) << 4, 3j+2 = b >> 4.
      Plane: bit 12 of index k at byte k >> 3, bit k & 7 (LSB first).
      (The quantizer's safetensors store 13-bit indices as one direct 13-bit
      stream; v41_to_gguf splits it.  12-bit layers are byte-identical to it.)
  Decode (f32): W[r, 8k + d] = e4m3(cb[idx[r, k]][d]) * f32(gain[r]).

Sizes (bytes): payload = 32 + 2*rows + rows*mrow + rows*prow + 8
  K=8192: w1/w3 2,400,808, w2 2,406,440 -> blob 2,767,968,272 (L 0-13)
  K=4096: w1/w3 2,216,488, w2 2,222,120 -> blob 2,555,598,864 (L 14-39)
  mtp (K=4096, 128 experts): 851,888,144

Original checkpoint (deepseek-ai/DeepSeek-V4.1-Flash-1017 safetensors):
``layers.{L}.ffn.experts.{e}.w{1,3,2}.weight`` I8 [rows, cols/2] packed FP4
e2m1 (low nibble = even column), ``.scale`` F8_E8M0 [rows, cols/32];
value = fp4 * 2^(s - 127) (YoungAi treats s = 255 as 0, s = 0 as 2^-127).
"""
from __future__ import annotations

import glob
import json
import os
import re
import struct
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np

try:  # torch is only needed for dequant()
    import torch
except Exception:  # pragma: no cover
    torch = None

# --------------------------------------------------------------------------
# constants
# --------------------------------------------------------------------------
BASE_DIR = "/models/Baekpica/DeepSeek-V4.1-Flash-1017-VQ-GGUF"
BASE_NAME = "DeepSeek-V4.1-Flash-vq8sh14-q4k-mtpnative.gguf"
BASE_SIZE = 113_556_639_424
BASE_SHA256 = "9f3afe6a46ff8de711911dfad5994519a4e5dd57b163f88a5bf4111c308d0e6f"
ORIG_DIR = "/models/deepseek-ai/DeepSeek-V4.1-Flash-1017"

GGML_TYPE_VQBLOB = 42
GGML_TYPE_FP4X32 = 43
GGML_TYPE_FP8_32X32 = 44

DQVL_MAGIC = 0x4C565144  # 'DQVL'
DQV3_MAGIC = 0x33565144  # 'DQV3'
DQVL_VER = 3
PAYLOAD_HDR = 32
MAIN_NBIT = 12
FLAG_E4M3 = 1
FLAG_PLANE = 2
TRAILER = 8

PROJS = ("w1", "w3", "w2")  # slot order (which = 0, 1, 2)
PROJ_ALIASES = {"w1": "w1", "w3": "w3", "w2": "w2", "gate": "w1", "up": "w3", "down": "w2",
                0: "w1", 1: "w3", 2: "w2"}


def canon_proj(p) -> str:
    try:
        return PROJ_ALIASES[p]
    except KeyError:
        raise KeyError(f"unknown projection {p!r}; use w1/w3/w2 or gate/up/down") from None


def _align(x: int, a: int) -> int:
    return (x + a - 1) // a * a


def vq_tensor_name(L: Optional[int] = None, mtp: Optional[int] = None) -> str:
    """GGUF tensor name of a routed-expert blob: layer L or MTP tower ``mtp``."""
    if mtp is not None:
        return f"mtp.{int(mtp)}.ffn_exps_vq.blob"
    return f"blk.{int(L)}.ffn_exps_vq.blob"


def default_parts(base_dir: str = BASE_DIR, base_name: str = BASE_NAME) -> List[str]:
    parts = sorted(glob.glob(os.path.join(base_dir, base_name + ".part*-of-*")))
    if not parts:
        raise FileNotFoundError(f"no parts {base_name}.partNN-of-MM under {base_dir}")
    m = re.search(r"\.part(\d+)-of-(\d+)$", parts[0])
    n = int(m.group(2))
    want = [os.path.join(base_dir, f"{base_name}.part{i:02d}-of-{n:02d}") for i in range(1, n + 1)]
    if parts != want:
        raise FileNotFoundError(f"incomplete part set: have {len(parts)}, want {n}")
    return want


# --------------------------------------------------------------------------
# virtual file over byte-range parts
# --------------------------------------------------------------------------
_MAXIO = 1 << 30  # per-syscall chunk (Linux caps a single read at ~2 GiB)


class VirtualFile:
    """Read-only concatenation of one or more files (e.g. the 40 parts)."""

    def __init__(self, paths: Union[str, Sequence[str]]):
        if isinstance(paths, (str, os.PathLike)):
            paths = [str(paths)]
        self.paths = [str(p) for p in paths]
        self.sizes = [os.path.getsize(p) for p in self.paths]
        self.starts = np.concatenate([[0], np.cumsum(self.sizes)]).astype(np.int64)
        self.size = int(self.starts[-1])
        self._fds = [os.open(p, os.O_RDONLY) for p in self.paths]

    def close(self):
        for fd in self._fds:
            try:
                os.close(fd)
            except OSError:
                pass
        self._fds = []

    def __del__(self):
        self.close()

    def part_spans(self, offset: int, n: int) -> List[Tuple[int, str, int, int]]:
        """[(part_index, part_path, offset_in_part, length)] covering [offset, offset+n)."""
        if offset < 0 or offset + n > self.size:
            raise ValueError(f"range [{offset}, {offset + n}) outside file of {self.size} B")
        out = []
        pos, end = offset, offset + n
        i = int(np.searchsorted(self.starts, offset, side="right") - 1)
        while pos < end:
            pe = int(self.starts[i + 1])
            take = min(end, pe) - pos
            if take > 0:
                out.append((i, self.paths[i], pos - int(self.starts[i]), take))
            pos += take
            i += 1
        return out

    def readinto(self, offset: int, out: np.ndarray) -> np.ndarray:
        """Fill the C-contiguous uint8 array ``out`` from virtual offset ``offset``."""
        mv = memoryview(out).cast("B")
        done = 0
        for i, _, loc, ln in self.part_spans(offset, mv.nbytes):
            got = 0
            while got < ln:
                k = os.preadv(self._fds[i], [mv[done + got: done + got + min(_MAXIO, ln - got)]], loc + got)
                if k <= 0:
                    raise IOError(f"short read in {self.paths[i]} at {loc + got}")
                got += k
            done += ln
        return out

    def pread(self, offset: int, n: int) -> np.ndarray:
        return self.readinto(offset, np.empty(n, np.uint8))


# --------------------------------------------------------------------------
# GGUF reader
# --------------------------------------------------------------------------
# ggml type -> (block elements, block bytes); None = special-cased
GGML_BLOCK = {
    0: (1, 4), 1: (1, 2), 2: (32, 18), 3: (32, 20), 6: (32, 22), 7: (32, 24), 8: (32, 34),
    9: (32, 36), 10: (256, 84), 11: (256, 110), 12: (256, 144), 13: (256, 176), 14: (256, 210),
    15: (256, 292), 16: (256, 66), 17: (256, 74), 18: (256, 98), 19: (256, 50), 20: (32, 18),
    21: (256, 110), 22: (256, 82), 23: (256, 136), 24: (1, 1), 25: (1, 2), 26: (1, 4),
    27: (1, 8), 28: (1, 8), 29: (256, 56), 30: (1, 2),
    GGML_TYPE_VQBLOB: (1, 1), GGML_TYPE_FP4X32: (32, 17),
}
GGML_TYPE_NAME = {0: "F32", 1: "F16", 8: "Q8_0", 10: "Q2_K", 12: "Q4_K", 16: "IQ2_XXS", 24: "I8",
                  25: "I16", 26: "I32", 27: "I64", 28: "F64", 30: "BF16",
                  GGML_TYPE_VQBLOB: "VQBLOB", GGML_TYPE_FP4X32: "FP4X32",
                  GGML_TYPE_FP8_32X32: "FP8_32X32"}


def ggml_nbytes(gtype: int, ne: Sequence[int]) -> int:
    """Byte size of a tensor; ``ne`` in GGUF order (innermost first)."""
    if gtype == GGML_TYPE_FP8_32X32:  # ne = [cols, rows]: e4m3 plane + ceil/32 x ceil/32 ue8m0
        cols, rows = int(ne[0]), int(ne[1])
        return rows * cols + ((rows + 31) // 32) * ((cols + 31) // 32)
    blk, tsz = GGML_BLOCK[gtype]
    n = 1
    for d in ne:
        n *= int(d)
    if n % blk:
        raise ValueError(f"type {gtype}: {n} elements not a multiple of block {blk}")
    return n // blk * tsz


@dataclass
class TensorInfo:
    name: str
    ggml_type: int
    ne: Tuple[int, ...]          # GGUF order (innermost first)
    rel_offset: int              # relative to data0
    abs_offset: int              # absolute offset in the (virtual) file
    nbytes: int

    @property
    def shape(self) -> Tuple[int, ...]:
        """numpy/torch order (outermost first)."""
        return tuple(reversed(self.ne))

    @property
    def type_name(self) -> str:
        return GGML_TYPE_NAME.get(self.ggml_type, str(self.ggml_type))


class _Cursor:
    def __init__(self, vf: VirtualFile, chunk: int = 8 << 20):
        self.vf, self.chunk = vf, chunk
        self.buf = b""
        self.pos = 0

    def _need(self, n: int):
        if self.pos + n > len(self.buf):
            grow = max(self.chunk, self.pos + n - len(self.buf))
            grow = min(grow, self.vf.size - len(self.buf))
            self.buf += self.vf.pread(len(self.buf), grow).tobytes()
            if self.pos + n > len(self.buf):
                raise EOFError("GGUF header truncated")

    def take(self, n: int) -> bytes:
        self._need(n)
        b = self.buf[self.pos:self.pos + n]
        self.pos += n
        return b

    def u(self, fmt: str):
        n = struct.calcsize(fmt)
        return struct.unpack("<" + fmt, self.take(n))[0]

    def string(self) -> str:
        return self.take(self.u("Q")).decode("utf-8", errors="surrogateescape")


_GGUF_SCALAR = {0: ("B", np.uint8), 1: ("b", np.int8), 2: ("H", np.uint16), 3: ("h", np.int16),
                4: ("I", np.uint32), 5: ("i", np.int32), 6: ("f", np.float32), 7: ("?", np.bool_),
                10: ("Q", np.uint64), 11: ("q", np.int64), 12: ("d", np.float64)}


class GGUFFile:
    """GGUF v2/v3 reader over a single file or the list of byte-range parts.

    Attributes: ``version``, ``kv`` (dict key -> python value / numpy array),
    ``kv_types`` (key -> gguf value type), ``tensors`` (list of TensorInfo in
    table order), ``data0``, ``alignment``, ``size``.
    """

    def __init__(self, source: Union[None, str, Sequence[str]] = None):
        if source is None:
            source = default_parts()
        self.vf = VirtualFile(source)
        self.size = self.vf.size
        c = _Cursor(self.vf)
        if c.take(4) != b"GGUF":
            raise ValueError("not a GGUF file")
        self.version = c.u("I")
        nt, nkv = c.u("Q"), c.u("Q")
        self.kv, self.kv_types = {}, {}
        for _ in range(nkv):
            k = c.string()
            t = c.u("I")
            self.kv_types[k] = t
            self.kv[k] = self._value(c, t)
        self.alignment = int(self.kv.get("general.alignment", 32))
        raw = []
        for _ in range(nt):
            name = c.string()
            nd = c.u("I")
            ne = tuple(c.u("Q") for _ in range(nd))
            gt = c.u("I")
            off = c.u("Q")
            raw.append((name, gt, ne, off))
        self.header_end = c.pos
        self.data0 = _align(c.pos, self.alignment)
        self.tensors: List[TensorInfo] = []
        for name, gt, ne, off in raw:
            nb = ggml_nbytes(gt, ne)
            self.tensors.append(TensorInfo(name, gt, ne, off, self.data0 + off, nb))
        self._by_name = {t.name: t for t in self.tensors}
        self._check_spans()

    @staticmethod
    def _value(c: _Cursor, t: int):
        if t == 8:
            return c.string()
        if t == 9:
            et, n = c.u("I"), c.u("Q")
            if et == 8:
                return [c.string() for _ in range(n)]
            if et in _GGUF_SCALAR:
                fmt, dt = _GGUF_SCALAR[et]
                return np.frombuffer(c.take(n * np.dtype(dt).itemsize), dtype=dt).copy()
            return [GGUFFile._value(c, et) for _ in range(n)]
        fmt, _ = _GGUF_SCALAR[t]
        return c.u(fmt)

    def _check_spans(self):
        ts = sorted(self.tensors, key=lambda t: t.rel_offset)
        for a, b in zip(ts, ts[1:] + [None]):
            end = (b.abs_offset if b is not None else self.size)
            if a.abs_offset + a.nbytes > end or (b is not None and _align(a.abs_offset + a.nbytes, self.alignment) != b.abs_offset):
                raise ValueError(f"tensor {a.name}: computed size {a.nbytes} does not fit the offset table")

    def close(self):
        self.vf.close()

    def tensor(self, name: str) -> TensorInfo:
        return self._by_name[name]

    def __contains__(self, name: str) -> bool:
        return name in self._by_name

    def read(self, name_or_info: Union[str, TensorInfo]) -> np.ndarray:
        """Raw tensor bytes as a uint8 array."""
        t = self.tensor(name_or_info) if isinstance(name_or_info, str) else name_or_info
        return self.vf.pread(t.abs_offset, t.nbytes)

    def vq_tensors(self) -> List[TensorInfo]:
        return [t for t in self.tensors if t.ggml_type == GGML_TYPE_VQBLOB]


_DEFAULT_GGUF: Optional[GGUFFile] = None


def open_base(source: Union[None, str, Sequence[str]] = None) -> GGUFFile:
    """Open the base GGUF (default: the 40 published parts as one virtual file).
    The first call without ``source`` is cached and reused by read_vq_layer()."""
    global _DEFAULT_GGUF
    if source is None:
        if _DEFAULT_GGUF is None:
            _DEFAULT_GGUF = GGUFFile(None)
        return _DEFAULT_GGUF
    return GGUFFile(source)


# --------------------------------------------------------------------------
# small numeric formats
# --------------------------------------------------------------------------
def _e4m3_decode_table() -> np.ndarray:
    t = np.zeros(256, np.float32)
    for x in range(256):
        a = x & 0x7F
        s = -1.0 if x & 0x80 else 1.0
        if a == 0:
            v = 0.0
        elif a == 0x7F:
            v = float("nan")
        else:
            e, m = (x >> 3) & 0xF, x & 7
            v = m * 2.0 ** -9 if e == 0 else (1.0 + m / 8.0) * 2.0 ** (e - 7)
        t[x] = s * v
    return t


E4M3_TABLE = _e4m3_decode_table()                                  # byte -> f32
_E4M3_MAG = np.array([abs(E4M3_TABLE[i]) for i in range(127)], np.float32)  # index 0..126


def e4m3_decode(b: np.ndarray) -> np.ndarray:
    return E4M3_TABLE[np.asarray(b, np.uint8)]


def e4m3_round(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """YoungAi ds4_e4m3fn_round (src/common/ds4_fp8.h:69): nearest E4M3FN value,
    saturate at 448, ties to even mantissa.  Returns (f32 value, magnitude index)."""
    x = np.asarray(x, np.float32)
    sign = np.where(x < 0, np.float32(-1), np.float32(1))
    ax = np.fmin(np.abs(x), np.float32(448.0))
    lo = np.clip(np.searchsorted(_E4M3_MAG, ax, side="right") - 1, 0, 126)
    hi = np.minimum(lo + 1, 126)
    bd = np.abs(ax - _E4M3_MAG[lo])
    nd = np.abs(ax - _E4M3_MAG[hi])
    up = (lo < 126) & ((nd < bd) | ((nd == bd) & ((hi & 1) == 0) & ((lo & 1) == 1)))
    best = np.where(up, hi, lo)
    return (sign * _E4M3_MAG[best]).astype(np.float32), best


def e4m3_encode(x: np.ndarray) -> np.ndarray:
    """f32 -> E4M3FN byte exactly like ds4_e4m3fn_f32_to_byte (zero -> 0x00)."""
    v, best = e4m3_round(x)
    return np.where((v < 0) & (best != 0), best | 0x80, best).astype(np.uint8)


def codebook_to_e4m3_like_youngai(c_f32: np.ndarray) -> np.ndarray:
    """Quantizer + converter rounding of trained f32 codewords: f32 -> f16 (RNE,
    v41_vq.cu:297) -> E4M3 (v41_vq_rate.inc.cu:63) -> byte (v41_to_gguf_vq3.inc.c:123)."""
    return e4m3_encode(np.asarray(c_f32, np.float32).astype(np.float16).astype(np.float32))


FP4_TABLE = np.array([0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6], np.float32)


def e8m0_to_f32(s: np.ndarray) -> np.ndarray:
    """YoungAi v41_dq_fp4_kernel convention: 255 -> 0, 0 -> 2^-127, else 2^(s-127)."""
    s = np.asarray(s, np.uint8)
    bits = np.where(s == 0, np.uint32(0x00400000), s.astype(np.uint32) << np.uint32(23)).astype(np.uint32)
    v = bits.view(np.float32)
    return np.where(s == 255, np.float32(0), v).astype(np.float32)


def fp4_dequant(w_i8: np.ndarray, scale_e8m0: np.ndarray) -> np.ndarray:
    """Packed FP4 [rows, cols/2] + E8M0 [rows, cols/32] -> f32 [rows, cols]."""
    w = np.asarray(w_i8).view(np.uint8)
    rows, half = w.shape
    out = np.empty((rows, half * 2), np.float32)
    out[:, 0::2] = FP4_TABLE[w & 0x0F]
    out[:, 1::2] = FP4_TABLE[w >> 4]
    sc = e8m0_to_f32(scale_e8m0)
    out.reshape(rows, -1, 32)[...] *= sc[:, :, None]
    return out


# --------------------------------------------------------------------------
# 12-bit stream / 1-bit plane
# --------------------------------------------------------------------------
def unpack12(main: np.ndarray, nidx: int, out: Optional[np.ndarray] = None, dtype=np.int32) -> np.ndarray:
    """[..., mrow] uint8 (per-row LSB-first 12-bit stream) -> [..., nidx] indices."""
    lead = main.shape[:-1]
    if out is None:
        out = np.empty(lead + (nidx,), dtype)
    npair = nidx // 2
    trip = main[..., :3 * npair].reshape(lead + (npair, 3))
    b0 = trip[..., 0].astype(np.int32)
    b1 = trip[..., 1].astype(np.int32)
    b2 = trip[..., 2].astype(np.int32)
    out[..., 0:2 * npair:2] = b0 | ((b1 & 0x0F) << 8)
    out[..., 1:2 * npair:2] = (b1 >> 4) | (b2 << 4)
    if nidx & 1:
        out[..., -1] = main[..., 3 * npair].astype(np.int32) | ((main[..., 3 * npair + 1].astype(np.int32) & 0x0F) << 8)
    return out


def pack12(idx: np.ndarray) -> np.ndarray:
    """Inverse of unpack12: [..., nidx] (low 12 bits used) -> [..., ceil(12*nidx/8)] uint8."""
    nidx = idx.shape[-1]
    lead = idx.shape[:-1]
    mrow = (nidx * MAIN_NBIT + 7) // 8
    if nidx & 1:
        idx = np.concatenate([idx, np.zeros(lead + (1,), idx.dtype)], axis=-1)
    a = idx[..., 0::2].astype(np.int32) & 0xFFF
    b = idx[..., 1::2].astype(np.int32) & 0xFFF
    out = np.empty(lead + (a.shape[-1], 3), np.uint8)
    out[..., 0] = a & 0xFF
    out[..., 1] = (a >> 8) | ((b & 0x0F) << 4)
    out[..., 2] = b >> 4
    return out.reshape(lead + (-1,))[..., :mrow]


def unpack_plane(plane: np.ndarray, nidx: int) -> np.ndarray:
    return np.unpackbits(plane, axis=-1, bitorder="little")[..., :nidx]


def pack_plane(bits: np.ndarray) -> np.ndarray:
    return np.packbits(np.asarray(bits, np.uint8), axis=-1, bitorder="little")


# --------------------------------------------------------------------------
# blob layout
# --------------------------------------------------------------------------
@dataclass
class BlobLayout:
    nexp: int
    K: int
    dim: int
    bits: int
    shapes: Dict[str, Tuple[int, int]]       # proj -> (rows, cols)
    cb_off: int
    body: int
    pay_off: Dict[str, int]                  # proj -> offset inside one expert's group
    pay_size: Dict[str, int]
    mrow: Dict[str, int]
    prow: Dict[str, int]
    expert_stride: int
    total: int

    def slot(self, e: int, proj: str) -> int:
        return self.body + e * self.expert_stride + self.pay_off[canon_proj(proj)]


def index_bits(K: int) -> int:
    b = 1
    while (1 << b) < K:
        b += 1
    return b


def blob_layout(nexp: int, K: int, shapes: Dict[str, Tuple[int, int]], dim: int = 8) -> BlobLayout:
    """Byte layout produced by gen_vqblob_v3 (v41_to_gguf_vq3.inc.c:98-159)."""
    bits = index_bits(K)
    if bits not in (12, 13):
        raise ValueError("DQV3 supports 12- or 13-bit indices only")
    cb_off = _align(16 + nexp * 3 * 8, 16)
    body = _align(cb_off + K * dim, 16)
    pay_off, pay_size, mrow, prow = {}, {}, {}, {}
    o = 0
    for p in PROJS:
        rows, cols = shapes[p]
        nidx = cols // dim
        mrow[p] = (nidx * MAIN_NBIT + 7) // 8
        prow[p] = (nidx + 7) // 8 if bits > MAIN_NBIT else 0
        pay_size[p] = PAYLOAD_HDR + rows * 2 + rows * mrow[p] + rows * prow[p] + TRAILER
        pay_off[p] = o
        o += pay_size[p]
    return BlobLayout(nexp, K, dim, bits, dict(shapes), cb_off, body, pay_off, pay_size, mrow, prow, o,
                      body + nexp * o)


# --------------------------------------------------------------------------
# layer data
# --------------------------------------------------------------------------
@dataclass
class VQLayer:
    name: str                                  # GGUF tensor name
    L: int                                     # header L field (tower index for mtp)
    nexp: int
    K: int
    bits: int                                  # 12 or 13
    dim: int
    cb_bytes: np.ndarray                       # uint8 [K, dim] (E4M3FN)
    cb: np.ndarray                             # float32 [K, dim]
    gains: Dict[str, np.ndarray]               # proj -> float16 [nexp, rows]
    idx: Dict[str, np.ndarray]                 # proj -> int32 [nexp, rows, cols/dim]
    shapes: Dict[str, Tuple[int, int]]         # proj -> (rows, cols)
    abs_offset: int = -1                       # absolute file offset of the blob
    nbytes: int = -1
    _dev_cache: dict = field(default_factory=dict, repr=False)

    @property
    def layout(self) -> BlobLayout:
        return blob_layout(self.nexp, self.K, self.shapes, self.dim)


def _parse_header(hdr: np.ndarray) -> Tuple[int, int, int, int]:
    magic, ver, L, nexp = struct.unpack_from("<4I", hdr.tobytes(), 0)
    if magic != DQVL_MAGIC:
        raise ValueError(f"bad blob magic {magic:#x}")
    if ver != DQVL_VER:
        raise ValueError(f"DQVL ver {ver} (only v3 supported)")
    return magic, ver, L, nexp


def decode_vq_blob(blob: np.ndarray, name: str = "?", idx_dtype=np.int32, chunk: int = 64) -> VQLayer:
    """Parse a whole DQVL v3 blob (uint8 array).  Validates every payload header,
    the slot table, zero padding and trailers against the canonical layout."""
    blob = np.asarray(blob, np.uint8)
    _, _, L, nexp = _parse_header(blob[:16])
    slots = blob[16:16 + nexp * 24].view(np.uint64).reshape(nexp, 3).astype(np.int64)
    # first payload headers give K / dim / shapes
    shapes, K, dim, cb_off = {}, None, None, None
    for w, p in enumerate(PROJS):
        o = int(slots[0, w])
        mg, d, nc, rows, cols, flags, mnb, cbo = struct.unpack_from("<IHHIIIIQ", blob[o:o + 32].tobytes())
        if mg != DQV3_MAGIC:
            raise ValueError(f"{name}: payload magic {mg:#x} at {o}")
        K, dim, cb_off = (nc, d, cbo) if K is None else (K, dim, cb_off)
        if (nc, d, cbo) != (K, dim, cb_off):
            raise ValueError(f"{name}: inconsistent payload headers")
        shapes[p] = (rows, cols)
    lay = blob_layout(nexp, K, shapes, dim)
    if lay.cb_off != cb_off or lay.total != blob.size:
        raise ValueError(f"{name}: layout mismatch (cb_off {cb_off} vs {lay.cb_off}, size {blob.size} vs {lay.total})")
    exp_slots = np.array([[lay.slot(e, p) for p in PROJS] for e in range(nexp)], np.int64)
    if not np.array_equal(slots, exp_slots):
        raise ValueError(f"{name}: slot table is not the canonical back-to-back layout")
    if blob[16 + nexp * 24:cb_off].any() or blob[cb_off + K * dim:lay.body].any():
        raise ValueError(f"{name}: non-zero header padding")
    cb_bytes = blob[cb_off:cb_off + K * dim].reshape(K, dim).copy()
    E = blob[lay.body:].reshape(nexp, lay.expert_stride)
    gains, idx = {}, {}
    for w, p in enumerate(PROJS):
        rows, cols = shapes[p]
        nidx = cols // dim
        o = lay.pay_off[p]
        flags = (FLAG_E4M3 | (FLAG_PLANE if lay.bits > MAIN_NBIT else 0))
        hdr = np.frombuffer(struct.pack("<IHHIIIIQ", DQV3_MAGIC, dim, K, rows, cols, flags, MAIN_NBIT, cb_off), np.uint8)
        if not (E[:, o:o + 32] == hdr).all():
            raise ValueError(f"{name}: payload header mismatch for {p}")
        g0 = o + PAYLOAD_HDR
        gains[p] = np.ascontiguousarray(E[:, g0:g0 + rows * 2]).view(np.float16).reshape(nexp, rows)
        m0 = g0 + rows * 2
        m1 = m0 + rows * lay.mrow[p]
        p1 = m1 + rows * lay.prow[p]
        if E[:, p1:p1 + TRAILER].any():
            raise ValueError(f"{name}: non-zero payload trailer for {p}")
        out = np.empty((nexp, rows, nidx), idx_dtype)
        for e0 in range(0, nexp, chunk):
            e1 = min(nexp, e0 + chunk)
            main = E[e0:e1, m0:m1].reshape(e1 - e0, rows, lay.mrow[p])
            v = unpack12(main, nidx)
            if lay.prow[p]:
                pl = E[e0:e1, m1:p1].reshape(e1 - e0, rows, lay.prow[p])
                v |= unpack_plane(pl, nidx).astype(np.int32) << MAIN_NBIT
            out[e0:e1] = v
        idx[p] = out
    return VQLayer(name, L, nexp, K, lay.bits, dim, cb_bytes, e4m3_decode(cb_bytes), gains, idx, shapes,
                   nbytes=int(blob.size))


def read_vq_blob(name: str, gguf: Optional[GGUFFile] = None, idx_dtype=np.int32) -> VQLayer:
    g = gguf if gguf is not None else open_base()
    t = g.tensor(name)
    if t.ggml_type != GGML_TYPE_VQBLOB:
        raise ValueError(f"{name} is ggml type {t.ggml_type}, not {GGML_TYPE_VQBLOB}")
    lay = decode_vq_blob(g.read(t), name, idx_dtype)
    lay.abs_offset, lay.nbytes = t.abs_offset, t.nbytes
    return lay


def read_vq_layer(L: int, gguf: Optional[GGUFFile] = None, mtp: Optional[int] = None,
                  idx_dtype=np.int32) -> VQLayer:
    """Routed experts of layer ``L`` (or MTP tower ``mtp`` when given).

    Returns a VQLayer: cb_bytes uint8 [K,8], cb f32 [K,8], gains[proj] f16
    [nexp, rows], idx[proj] int32 [nexp, rows, cols/8], K, bits, shapes."""
    return read_vq_blob(vq_tensor_name(L, mtp), gguf, idx_dtype)


def pack_vq_layer(layer: VQLayer = None, *, L: int = None, cb_bytes: np.ndarray = None,
                  gains: Dict[str, np.ndarray] = None, idx: Dict[str, np.ndarray] = None,
                  dim: int = 8, chunk: int = 64) -> np.ndarray:
    """Exact DQVL v3 blob bytes (uint8 array) from codebook bytes + gains + indices.

    Either pass a VQLayer, or keyword arguments L, cb_bytes ([K,8] uint8 E4M3),
    gains {proj: f16 [nexp, rows]}, idx {proj: int [nexp, rows, cols/8]}.
    Byte-identical to v41_to_gguf for the same inputs."""
    if layer is not None:
        L, cb_bytes, gains, idx, dim = layer.L, layer.cb_bytes, layer.gains, layer.idx, layer.dim
    gains = {canon_proj(k): v for k, v in gains.items()}
    idx = {canon_proj(k): v for k, v in idx.items()}
    cb_bytes = np.ascontiguousarray(cb_bytes, np.uint8).reshape(-1, dim)
    K = cb_bytes.shape[0]
    nexp = idx["w1"].shape[0]
    shapes = {p: (idx[p].shape[1], idx[p].shape[2] * dim) for p in PROJS}
    lay = blob_layout(nexp, K, shapes, dim)
    out = np.zeros(lay.total, np.uint8)
    out[:16] = np.frombuffer(struct.pack("<4I", DQVL_MAGIC, DQVL_VER, int(L), nexp), np.uint8)
    slots = np.array([[lay.slot(e, p) for p in PROJS] for e in range(nexp)], np.uint64)
    out[16:16 + nexp * 24] = slots.reshape(-1).view(np.uint8)
    out[lay.cb_off:lay.cb_off + K * dim] = cb_bytes.reshape(-1)
    E = out[lay.body:].reshape(nexp, lay.expert_stride)
    for p in PROJS:
        rows, cols = shapes[p]
        nidx = cols // dim
        ix = idx[p]
        g = np.ascontiguousarray(gains[p], np.float16)
        if g.shape != (nexp, rows) or ix.shape != (nexp, rows, nidx):
            raise ValueError(f"{p}: gains {g.shape} / idx {ix.shape} do not match ({nexp}, {rows}, {nidx})")
        o = lay.pay_off[p]
        flags = FLAG_E4M3 | (FLAG_PLANE if lay.bits > MAIN_NBIT else 0)
        E[:, o:o + 32] = np.frombuffer(struct.pack("<IHHIIIIQ", DQV3_MAGIC, dim, K, rows, cols, flags,
                                                   MAIN_NBIT, lay.cb_off), np.uint8)
        g0 = o + PAYLOAD_HDR
        E[:, g0:g0 + rows * 2] = g.view(np.uint8).reshape(nexp, rows * 2)
        m0 = g0 + rows * 2
        m1 = m0 + rows * lay.mrow[p]
        p1 = m1 + rows * lay.prow[p]
        for e0 in range(0, nexp, chunk):
            e1 = min(nexp, e0 + chunk)
            v = np.asarray(ix[e0:e1])
            if v.min() < 0 or v.max() >= K:
                raise ValueError(f"{p}: index out of range [0, {K})")
            E[e0:e1, m0:m1] = pack12(v).reshape(e1 - e0, -1)
            if lay.prow[p]:
                E[e0:e1, m1:p1] = pack_plane((v >> MAIN_NBIT) & 1).reshape(e1 - e0, -1)
        # trailer stays zero
    return out


# --------------------------------------------------------------------------
# dequantization
# --------------------------------------------------------------------------
def dequant(layer: VQLayer, e: int, proj, device: Union[str, "torch.device"] = "cpu") -> "torch.Tensor":
    """W_hat [rows, cols] float32 for expert ``e`` of ``proj`` (w1/w3/w2 or gate/up/down).
    Exactly the engine/quantizer decode: e4m3(cb[idx]) * f32(f16 gain), f32 multiply."""
    return dequant_experts(layer, [e], proj, device)[0]


def dequant_experts(layer: VQLayer, experts: Iterable[int], proj, device="cpu") -> "torch.Tensor":
    """Batched dequant -> float32 [len(experts), rows, cols]."""
    p = canon_proj(proj)
    dev = torch.device(device)
    key = ("cb", str(dev))
    if key not in layer._dev_cache:
        layer._dev_cache[key] = torch.from_numpy(np.ascontiguousarray(layer.cb)).to(dev)
    cb = layer._dev_cache[key]
    ex = np.asarray(list(experts), np.int64)
    rows, cols = layer.shapes[p]
    ix = torch.from_numpy(np.ascontiguousarray(layer.idx[p][ex])).to(dev).long()      # [n, rows, nidx]
    g = torch.from_numpy(layer.gains[p][ex].astype(np.float32)).to(dev)               # [n, rows]
    w = cb[ix] * g[:, :, None, None]                                                   # [n, rows, nidx, dim]
    return w.reshape(len(ex), rows, cols)


# --------------------------------------------------------------------------
# patching
# --------------------------------------------------------------------------
def write_blob(dst_path: str, name: str, blob: np.ndarray, gguf: Optional[GGUFFile] = None,
               fsync: bool = True, check_header: bool = True) -> int:
    """Overwrite tensor ``name`` in the assembled GGUF copy ``dst_path`` with ``blob``
    (same size required; with check_header the 16-byte DQVL header already in the
    file must equal the new one).  Offsets come from ``gguf`` (default: the
    published parts, whose layout the copy must share).  Returns the absolute
    offset written.  The 32-byte GGUF padding after the tensor is not touched."""
    g = gguf if gguf is not None else open_base()
    t = g.tensor(name)
    blob = np.ascontiguousarray(blob, np.uint8)
    if blob.size != t.nbytes:
        raise ValueError(f"{name}: new blob {blob.size} B != tensor {t.nbytes} B")
    if os.path.getsize(dst_path) != g.size:
        raise ValueError(f"{dst_path}: size {os.path.getsize(dst_path)} != base {g.size}")
    fd = os.open(dst_path, os.O_RDWR)
    try:
        old = os.pread(fd, 16, t.abs_offset)
        o_hdr, n_hdr = struct.unpack("<4I", old), struct.unpack("<4I", blob[:16].tobytes())
        if check_header and o_hdr != n_hdr:
            raise ValueError(f"{name}: blob header {n_hdr} != file header {o_hdr} at {t.abs_offset}")
        mv = memoryview(blob)
        done = 0
        while done < blob.size:
            done += os.pwrite(fd, mv[done:done + _MAXIO], t.abs_offset + done)
        if fsync:
            os.fsync(fd)
    finally:
        os.close(fd)
    return t.abs_offset


# --------------------------------------------------------------------------
# original checkpoint (FP4) access
# --------------------------------------------------------------------------
class OriginalWeights:
    """Index of the original safetensors shards (no index json; headers scanned).
    Tensors whose bytes are not fully on disk yet are skipped."""

    def __init__(self, model_dir: str = ORIG_DIR):
        self.dir = model_dir
        self.ent: Dict[str, Tuple[str, str, Tuple[int, ...], int, int]] = {}
        for f in sorted(glob.glob(os.path.join(model_dir, "*.safetensors"))):
            sz = os.path.getsize(f)
            with open(f, "rb") as fh:
                head = fh.read(8)
                if len(head) < 8:
                    continue
                n = struct.unpack("<Q", head)[0]
                try:
                    h = json.loads(fh.read(n))
                except Exception:
                    continue
            h.pop("__metadata__", None)
            for k, v in h.items():
                a, b = v["data_offsets"]
                if 8 + n + b <= sz:
                    self.ent[k] = (f, v["dtype"], tuple(v["shape"]), 8 + n + a, b - a)

    def has(self, name: str) -> bool:
        return name in self.ent

    def raw(self, name: str) -> np.ndarray:
        f, dt, shape, off, nb = self.ent[name]
        with open(f, "rb") as fh:
            fh.seek(off)
            buf = np.frombuffer(fh.read(nb), np.uint8)
        return buf.reshape(shape)

    def expert_fp4(self, L: int, e: int, proj, mtp: Optional[int] = None) -> np.ndarray:
        """Dequantized original expert matrix f32 [rows, cols] (exactly the
        quantizer's v41_dq_fp4_kernel values)."""
        p = canon_proj(proj)
        pre = f"mtp.{mtp}" if mtp is not None else f"layers.{L}"
        w = self.raw(f"{pre}.ffn.experts.{e}.{p}.weight")
        s = self.raw(f"{pre}.ffn.experts.{e}.{p}.scale")
        return fp4_dequant(w, s)


# --------------------------------------------------------------------------
# YoungAi numerics (for replicating the unweighted baseline)
# --------------------------------------------------------------------------
def youngai_row_gain(W: np.ndarray) -> np.ndarray:
    """f16 row gains exactly as v41_vq.cu row_gain_kernel (:33-50) + round_f16
    (:209-215, :321): 256 'threads', thread t sums w[i]^2 for i = t, t+256, ...
    sequentially in f32, then a shared-memory tree (o = 128..1), sqrtf(s / cols),
    0 -> 1, then f16 RNE.  Built with -fmad=false (no FMA)."""
    W = np.asarray(W, np.float32)
    rows, cols = W.shape
    T = 256
    pad = (-cols) % T
    if pad:
        W = np.concatenate([W, np.zeros((rows, pad), np.float32)], axis=1)
    sq = (W * W).reshape(rows, -1, T)
    s = np.zeros((rows, T), np.float32)
    for j in range(sq.shape[1]):
        s = s + sq[:, j, :]
    o = T // 2
    while o:
        s[:, :o] = s[:, :o] + s[:, o:2 * o]
        o //= 2
    g = np.sqrt(s[:, 0] / np.float32(cols)).astype(np.float32)
    g = np.where(g > 0, g, np.float32(1.0))
    return g.astype(np.float16)


def youngai_pool_stride(nexp: int, shapes: Dict[str, Tuple[int, int]], K: int, dim: int = 8,
                        global_nc: int = 4096) -> Tuple[int, int]:
    """Shared-codebook pool sampling (v41_shcb.inc.c:27-40): want = 1000 * nc
    (nc = max(K, --vq-nc 4096)), capped at 16,777,216; ps = floor(total_vectors / want).
    Returns (ps, per-expert sample count floor(nv_expert / ps))."""
    nv = sum(r * c for r, c in shapes.values()) // dim
    want = min(1000 * max(K, global_nc), 16_777_216)
    ps = max(1, (nexp * nv) // want)
    return ps, nv // ps


def youngai_expert_vectors(Ws: Sequence[np.ndarray], gains_f16: Sequence[np.ndarray], dim: int = 8):
    """Normalized vectors of one expert in YoungAi order: concat(w1, w3, w2) each
    row-major divided by its f16 row gain (f32 division), viewed as [nv, dim]."""
    parts = [(np.asarray(W, np.float32) / np.asarray(g, np.float16).astype(np.float32)[:, None]).reshape(-1, dim)
             for W, g in zip(Ws, gains_f16)]
    return np.concatenate(parts, 0)


def youngai_lloyd(pool: "torch.Tensor", K: int, iters: int = 8, chunk: int = 16384, topk: int = 6):
    """Unweighted Lloyd exactly as vq_fit_assign (v41_vq.cu:252-299) for the shared
    codebook (stride 1 over the pool): init C[k] = pool[k*n//K]; each iteration
    assigns with fp32 sum_d (v_d - c_d)^2 (d sequential, no FMA, strict '<' so the
    lowest index wins ties), then C[k] = f32(double sum / count), empty clusters
    keep their codeword.  No rounding inside the loop.  Returns f32 C [K, dim]
    BEFORE the final f16 -> E4M3 rounding (see codebook_to_e4m3_like_youngai).
    Assignment uses an fp32 matmul shortlist (top-``topk``) re-ranked with the
    exact distance."""
    torch.backends.cuda.matmul.allow_tf32 = False
    n, dim = pool.shape
    dev = pool.device
    C = pool[(torch.arange(K, device=dev, dtype=torch.int64) * n) // K].clone()
    idx = torch.empty(n, dtype=torch.int64, device=dev)
    for _ in range(iters):
        idx = youngai_assign(pool, C, chunk, topk, out=idx)
        s = torch.zeros(K, dim, dtype=torch.float64, device=dev).index_add_(0, idx, pool.double())
        c = torch.bincount(idx, minlength=K)
        nz = c > 0
        C[nz] = (s[nz] / c[nz, None].double()).float()
    return C


def youngai_assign(V: "torch.Tensor", C: "torch.Tensor", chunk: int = 16384, topk: int = 6, out=None):
    """argmin_k fp32 sum_d (v_d - C[k,d])^2 with YoungAi's tie rule (lowest k)."""
    torch.backends.cuda.matmul.allow_tf32 = False
    n = V.shape[0]
    out = torch.empty(n, dtype=torch.int64, device=V.device) if out is None else out
    cn = (C * C).sum(1)
    big = torch.iinfo(torch.int64).max
    for i in range(0, n, chunk):
        v = V[i:i + chunk]
        cand = torch.topk(cn[None, :] - 2.0 * (v @ C.T), topk, dim=1, largest=False).indices
        e = v[:, None, :] - C[cand]
        dist = e[..., 0] * e[..., 0]
        for d in range(1, e.shape[-1]):
            dist = dist + e[..., d] * e[..., d]
        dmin = dist.min(1, keepdim=True).values
        out[i:i + chunk] = torch.where(dist == dmin, cand, torch.full_like(cand, big)).min(1).values
    return out
