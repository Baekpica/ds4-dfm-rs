#!/usr/bin/env python3
"""Model-free checks for prefix-vector validation and numerical reporting."""
import hashlib
from pathlib import Path
import tempfile
import unittest

import numpy as np

from compare_iquest_prefix import KV_ROW_BYTES, TOKENS, float_metrics, read_kv, read_stage


class PrefixCompare(unittest.TestCase):
    def test_metrics(self):
        reference = np.array([[1, 2], [3, 4]], dtype=np.float32)
        self.assertEqual(float_metrics(reference, reference)['relative_rmse'], 0)
        measured = float_metrics(reference, reference * 2)
        self.assertEqual(measured['relative_rmse'], 1)
        self.assertAlmostEqual(measured['mean_cosine'], 1)

    def test_vector_rejections(self):
        with tempfile.TemporaryDirectory(prefix='iquest-vectors-') as work:
            path = Path(work) / 'stage.f32'
            for values, shape in (([1], (2,)), ([0, 0], (2,)), ([1, np.nan], (2,))):
                np.asarray(values, dtype='<f4').tofile(path)
                with self.assertRaises(ValueError):
                    read_stage(path, shape)
            np.array([1, 2], dtype='<f4').tofile(path)
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            np.testing.assert_array_equal(read_stage(path, (2,), digest), [1, 2])
            with self.assertRaises(ValueError):
                read_stage(path, (2,), '0' * 64)

    def test_kv_rejections(self):
        with tempfile.TemporaryDirectory(prefix='iquest-kv-vectors-') as work:
            path = Path(work) / 'kv.q8_0'
            blocks = np.zeros((len(TOKENS) * KV_ROW_BYTES // 34, 34), dtype=np.uint8)
            blocks.tofile(path)
            with self.assertRaises(ValueError):
                read_kv(path)
            blocks[:, :2] = np.array([1], dtype='<f2').view(np.uint8)
            blocks[:, 2] = 1
            blocks.tofile(path)
            self.assertEqual(len(read_kv(path)), len(TOKENS) * KV_ROW_BYTES)
            blocks[0, :2] = np.array([np.nan], dtype='<f2').view(np.uint8)
            blocks.tofile(path)
            with self.assertRaises(ValueError):
                read_kv(path)


if __name__ == '__main__':
    unittest.main()
