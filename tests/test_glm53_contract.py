#!/usr/bin/env python3
"""Model-free comparison/schema regression; sparse files allocate no model."""
import math
from pathlib import Path
import struct
import tempfile
import unittest

import numpy as np
import glm53_contract as contract


class ContractTests(unittest.TestCase):
    def test_independent_metrics(self):
        a = np.array([1, 2, 3], dtype=np.float32)
        b = np.array([2, 4, 3], dtype=np.float32)
        result = contract.metrics(a, b)
        self.assertFalse(result["exact"])
        self.assertEqual(result["max_abs"], 2)
        self.assertAlmostEqual(result["rms"], math.sqrt(5 / 3))
        self.assertAlmostEqual(result["relative_l2"], math.sqrt(5 / 14))
        with self.assertRaises(AssertionError):
            contract.metrics(a, np.array([2, np.nan, 3]))
        self.assertTrue(contract.metrics(a, a)["exact"])

    def test_complete_payload_schema(self):
        n, vocab, latent, pooldim = 13, 154880, 512, 128
        state = 34 * (64 * 128 * 128 + 3 * 64 * 128 * 4) * 4 + 11 * 2 * 4 * pooldim * 4 + 2 * 4096 * 4
        header = [0x34565344, 3, 2048, latent, 11, 0x354D4347, 1024,
                  n, 45, 4, pooldim, vocab, state]
        cursors = (12, 9)
        total = 52 + n * 4 + vocab * 4 + 8 + state + 11 * (n * 1024 + (n // 4) * pooldim * 2) + 3 * latent * 2
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "frontier.bin"
            with path.open("wb") as fp:
                fp.write(struct.pack("<13I", *header))
                fp.write(struct.pack(f"<{n}I", *range(n)))
                fp.seek(52 + n * 4 + vocab * 4)
                fp.write(struct.pack("<2I", *cursors))
                fp.truncate(total)
            view = contract.view(path)
            self.assertEqual(view["cursors"], cursors)
            self.assertEqual(view["regions"]["layer43/pool"].size, 3 * pooldim)
            self.assertEqual(view["regions"]["layer43/latent"].size, n * latent)
            self.assertEqual(view["regions"]["mtp_kv"].size, 3 * latent)
            self.assertEqual(view["regions"]["layer0/recurrent"].size, 64 * 128 * 128)
            self.assertEqual(view["regions"]["last_hidden"].size, 4096)
            del view
            with path.open("r+b") as fp:
                fp.truncate(total - 1)
            with self.assertRaises((AssertionError, ValueError, TypeError)):
                contract.view(path)


if __name__ == "__main__":
    unittest.main()
