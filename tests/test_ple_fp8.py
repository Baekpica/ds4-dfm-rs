#!/usr/bin/env python3
"""Model-free PLE format, page-boundary and source BF16 decode contracts."""

import copy
import ctypes as C
import json
import math
from pathlib import Path
import struct
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
ROWS = 2500012
VOCABS = [20000003, 20000023, 20000033, 20000047, 20000059, 20000063,
          20000069, 20000077, 20000081, 20000093, 20000107, 20000147,
          20000153, 20000159, 20000161, 20000171]


def manifest(fp8=True):
    stride = 160 if fp8 else 320
    payload = ROWS * stride
    aligned = (payload + 4095) // 4096 * 4096
    names = [f"part-{i}.bin" for i in range(4)]
    offsets = [sum(VOCABS[:i]) for i in range(16)]
    result = dict(
        format_version=2 if fp8 else 1,
        artifact_variant="PLE-FP8" if fp8 else "MQ-Q5-SSD-PLE-BF16",
        byte_order="little", storage="ssd_backed_bounded_page_cache",
        storage_dtype="FP8_E4M3FN" if fp8 else "BF16",
        alignment_bytes=4096, row_stride_bytes=stride,
        embedding_row_dimension=160, ngram_size=3, heads_per_ngram=8,
        number_of_ngram_heads=16, number_of_ple_layers=1,
        logical_shard_count=128, physical_file_count=4,
        usable_vocabulary_rows=sum(VOCABS), padded_vocabulary_rows=128 * ROWS,
        total_payload_bytes=128 * payload,
        total_file_bytes_including_alignment=128 * aligned,
        layer_multipliers=[23703573157769, 20109073645365, 8052911324071],
        per_head_vocabulary_sizes=VOCABS, per_head_offsets=offsets,
        hash_reference={"implementation_sha256":
                        "77fec77d87f2a0eb23b95fa04276fb5779698a7c7f523cf5061e49c118bcc459"},
        logical_parts=[dict(
            logical_part=i, physical_file_index=i // 32,
            physical_file=names[i // 32], global_row_start=i * ROWS,
            rows=ROWS, file_offset=(i % 32) * aligned, payload_bytes=payload,
            row_stride_bytes=stride, embedding_row_dimension=160,
        ) for i in range(128)],
        physical_files=[dict(index=i, path=names[i], file_bytes=32 * aligned,
                             payload_bytes=32 * payload) for i in range(4)],
    )
    if fp8:
        result["physical_paths_relative_to"] = "directory containing ple-manifest.json"
        result["quantization"] = dict(
            format="E4M3FN", scheme="per_tensor", scale_is_inverse=False,
            scale=dict(dtype="BF16", path="scale.bin", file_bytes=2,
                       bits_hex="0x3951", shape=[1]),
        )
    return result


def bf16_reference(code):
    """E4M3FN decoded as a binary rational, then round to nearest BF16."""
    sign = -1 if code & 128 else 1
    exponent, mantissa = (code >> 3) & 15, code & 7
    if exponent == 15 and mantissa == 7:
        return 0x7fc0 | ((code & 128) << 8)
    value = math.ldexp(mantissa, -9) if not exponent else math.ldexp(8 + mantissa, exponent - 10)
    scale = struct.unpack("<f", bytes.fromhex("00005139"))[0]
    value *= scale
    if not value:
        return (code & 128) << 8
    # BF16 has eight significant bits. Python round implements ties-to-even.
    _, power = math.frexp(value)
    rounded = math.ldexp(round(math.ldexp(value, 8 - power)), power - 8)
    return struct.unpack("<I", struct.pack("<f", sign * rounded))[0] >> 16


class RowView(C.Structure):
    _fields_ = [("segments", C.c_void_p * 2), ("segment_bytes", C.c_uint32 * 2),
                ("slots", C.c_uint32 * 2), ("segment_count", C.c_uint32)]


class PleFormats(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.lib = C.CDLL(str(ROOT / "tests/libds4ple_test.so"))
        cls.lib.ds4_ple_store_open.argtypes = [C.c_char_p, C.c_char_p, C.c_size_t,
                                             C.c_uint32, C.c_bool, C.c_char_p, C.c_size_t]
        cls.lib.ds4_ple_store_open.restype = C.c_void_p
        cls.lib.ds4_ple_store_close.argtypes = [C.c_void_p]
        cls.lib.ds4_ple_store_read_row.argtypes = [C.c_void_p, C.c_uint64, C.c_void_p,
                                                 C.c_size_t, C.c_char_p, C.c_size_t]
        cls.lib.ds4_ple_store_read_row.restype = C.c_bool
        cls.lib.ds4_ple_store_acquire_row.argtypes = [C.c_void_p, C.c_uint64,
                                                    C.POINTER(RowView), C.c_char_p, C.c_size_t]
        cls.lib.ds4_ple_store_acquire_row.restype = C.c_bool
        cls.lib.ds4_ple_store_release_row.argtypes = [C.c_void_p, C.POINTER(RowView)]

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.data = self.root / "sidecar"
        self.data.mkdir()
        self.error = C.create_string_buffer(512)

    def prepare(self, fp8=True):
        m = manifest(fp8)
        # Sparse files reproduce the full layout with only test rows allocated.
        for f in m["physical_files"]:
            with (self.data / f["path"]).open("wb") as handle:
                handle.truncate(f["file_bytes"])
        (self.data / "scale.bin").write_bytes(bytes.fromhex("5139"))
        if not fp8:
            # Version 1 paths are relative to the artifact root, not manifest.
            for f in m["physical_files"]:
                f["path"] = "sidecar/" + f["path"]
            for p in m["logical_parts"]:
                p["physical_file"] = "sidecar/" + p["physical_file"]
        self.save(m)
        return m

    def save(self, m):
        (self.data / "ple-manifest.json").write_text(json.dumps(m))

    def open(self, direct=False):
        self.error.value = b""
        return self.lib.ds4_ple_store_open(
            str(self.root).encode(), b"sidecar/ple-manifest.json", 65536,
            2, direct, self.error, len(self.error))

    def test_fp8_rows_and_pages(self):
        m = self.prepare()
        # Include all physical/logical boundaries and straddling 4 KiB rows.
        rows = {0, 1, 25, 26, 51, 127, m["usable_vocabulary_rows"] - 1}
        for i in range(1, 128):
            rows.update((i * ROWS - 1, i * ROWS, i * ROWS + 25))
        expected = {}
        for row in sorted(rows):
            p = m["logical_parts"][row // ROWS]
            codes = bytes((row + j) % 256 for j in range(160))
            codes = codes.replace(b"\x7f", b"\x7e").replace(b"\xff", b"\xfe")
            with (self.data / p["physical_file"]).open("r+b") as f:
                f.seek(p["file_offset"] + (row % ROWS) * 160)
                f.write(codes)
            expected[row] = b"".join(struct.pack("<H", bf16_reference(c)) for c in codes)
        for direct in (False, True):
            with self.subTest(direct=direct):
                store = self.open(direct)
                self.assertTrue(store, self.error.value.decode())
                try:
                    for row in sorted(rows):
                        out = C.create_string_buffer(320)
                        self.assertTrue(self.lib.ds4_ple_store_read_row(
                            store, row, out, len(out), self.error, len(self.error)), self.error.value)
                        self.assertEqual(out.raw, expected[row], f"row {row}")
                    view = RowView()
                    self.assertTrue(self.lib.ds4_ple_store_acquire_row(
                        store, 25, C.byref(view), self.error, len(self.error)))
                    self.assertEqual(list(view.segment_bytes), [96, 64])
                    self.lib.ds4_ple_store_release_row(store, C.byref(view))
                    self.assertFalse(self.lib.ds4_ple_store_read_row(
                        store, m["usable_vocabulary_rows"], out, len(out), self.error, len(self.error)))
                finally:
                    self.lib.ds4_ple_store_close(store)

    def test_bf16_is_unchanged(self):
        self.prepare(False)
        raw = bytes(range(256)) + bytes(range(64))
        with (self.data / "part-0.bin").open("r+b") as f:
            f.seek(12 * 320)
            f.write(raw)
        store = self.open()
        self.assertTrue(store, self.error.value.decode())
        try:
            out = C.create_string_buffer(320)
            self.assertTrue(self.lib.ds4_ple_store_read_row(
                store, 12, out, len(out), self.error, len(self.error)))
            self.assertEqual(out.raw, raw)
        finally:
            self.lib.ds4_ple_store_close(store)

    def test_rejects_invalid_fp8(self):
        original = self.prepare()
        mutations = [
            lambda m: m.update(format_version=1),
            lambda m: m.update(storage_dtype="BF16"),
            lambda m: m.update(row_stride_bytes=320),
            lambda m: m.update(artifact_variant="MQ-Q5-SSD-PLE-BF16"),
            lambda m: m.pop("quantization"),
            lambda m: m.pop("physical_paths_relative_to"),
            lambda m: m["quantization"].update(format="E5M2"),
            lambda m: m["quantization"].update(scheme="per_block"),
            lambda m: m["quantization"].update(scale_is_inverse=True),
            lambda m: m["quantization"]["scale"].update(path="../scale.bin"),
            lambda m: m["quantization"]["scale"].update(dtype="F32"),
            lambda m: m["quantization"]["scale"].update(shape=[128]),
            lambda m: m["quantization"]["scale"].update(bits_hex="0x3952"),
            lambda m: m["logical_parts"][3].update(row_stride_bytes=320),
        ]
        for index, mutate in enumerate(mutations):
            with self.subTest(case=index):
                m = copy.deepcopy(original)
                mutate(m)
                self.save(m)
                store = self.open()
                if store:
                    self.lib.ds4_ple_store_close(store)
                self.assertFalse(store, f"accepted invalid manifest {index}")
        self.save(original)
        for raw in (b"", b"\x51", b"\x51\x39\x00", b"\x00\x00"):
            (self.data / "scale.bin").write_bytes(raw)
            store = self.open()
            if store:
                self.lib.ds4_ple_store_close(store)
            self.assertFalse(store, f"accepted invalid scale {raw!r}")

    def test_rejects_other_fp8_contract(self):
        original = self.prepare()
        for case in ("scale", "overlap", "hash"):
            with self.subTest(case=case):
                m = copy.deepcopy(original)
                (self.data / "scale.bin").write_bytes(bytes.fromhex("5139"))
                if case == "scale":
                    m["quantization"]["scale"]["bits_hex"] = "0x3952"
                    (self.data / "scale.bin").write_bytes(bytes.fromhex("5239"))
                elif case == "overlap":
                    m["logical_parts"][1]["file_offset"] = 0
                else:
                    m["layer_multipliers"][0] += 2
                self.save(m)
                store = self.open()
                if store:
                    self.lib.ds4_ple_store_close(store)
                self.assertFalse(store, f"accepted non-reference FP8 {case}")


if __name__ == "__main__":
    unittest.main()
