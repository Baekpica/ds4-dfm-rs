#!/usr/bin/env python3
"""CPU-only selection provenance, byte preservation and recipe regressions."""
import hashlib
from enum import Enum
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
import glm_vq_q4k as fmt

PREFIX = 'model.language_model.layers.3.self_attn.'
SELECTED = PREFIX + 'o_proj.weight'
EXISTING = PREFIX + 'q_b_proj.weight'
SCALE = PREFIX + 'o_proj.weight_scale_inv'
CONTROL = PREFIX + 'q_a_layernorm.weight'
PART = 'native.gguf'


class Artifact(Enum):
    SOURCE = 'source'
    DONOR = 'donor'


def _fixture(path, kind):
    path.mkdir()
    donor = kind is Artifact.DONOR
    # These are opaque copy fixtures, not an arithmetic quantization proof.
    original = bytes([0x38]) * (128 * 256)
    packed = bytes(128 * 144)
    weights = [(EXISTING, fmt.Q4_K, [128, 256], packed),
        (SELECTED, fmt.Q4_K if donor else fmt.FP8, [128, 256], packed if donor else original),
        (SCALE, 0, [1, 2], bytes(8)),
        (CONTROL, fmt.BF16, [128], bytes(256))]
    records, entries = [], []
    offset = 0
    for name, typ, shape, data in weights:
        record = dict(name=name, part=PART, shape=shape, stored_shape=shape,
            type_id=typ, source_dtype='F8_E4M3', payload_bytes=len(data),
            payload_sha256=hashlib.sha256(data).hexdigest())
        if donor and name == SELECTED:
            record['parent_payload_sha256'] = hashlib.sha256(original).hexdigest()
        records.append(record)
        entries.append(dict(name=name, typ=typ, dims=tuple(reversed(shape)), out_offset=offset))
        offset = fmt.align(offset + len(data))
    links = {} if donor else {SELECTED:SCALE}
    meta = {'general.alignment':(4,128), 'glm_vq.format_version':(4,1),
        'glm_vq.fp8_scales':(8,json.dumps(links))}
    with (path / PART).open('wb') as file:
        header = fmt.header(meta, entries)
        file.write(header)
        for entry, (_, _, _, data) in zip(entries, weights):
            file.write(bytes(len(header) + entry['out_offset'] - file.tell()))
            file.write(data)
    manifest = dict(source_revision='same-source', tensors=records,
        nonexpert_recipe=dict(name=fmt.POLICY, kept_tensors=[SELECTED]))
    (path / 'tensor-manifest.json').write_text(json.dumps(manifest))
    (path / 'files.json').write_text(json.dumps(dict(files=[dict(path=name,
        bytes=(path/name).stat().st_size, sha256=fmt.digest(path/name))
        for name in [PART, 'tensor-manifest.json']])))
    return {record['name']:record for record in records}


class SelectionTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.source, self.donor, self.output = (self.root / n for n in ['source','donor','output'])
        self.records = _fixture(self.source, Artifact.SOURCE)
        self.imported = _fixture(self.donor, Artifact.DONOR)

    def tearDown(self):
        self.temp.cleanup()

    def _run_select(self, name=SELECTED):
        selected = self.root / 'selected.json'
        selected.write_text(json.dumps([name]))
        return subprocess.run([sys.executable, str(ROOT / 'tools/glm_vq_select_q4.py'),
            str(self.source), str(self.donor), str(self.output), '--select', str(selected)],
            capture_output=True, text=True)

    def test_copy_preserves_data(self):
        before = fmt.digest(self.source / PART)
        result = self._run_select()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(fmt.digest(self.source / PART), before)
        manifest = json.loads((self.output / 'tensor-manifest.json').read_text())
        self.assertNotIn(SELECTED, manifest['nonexpert_recipe']['kept_tensors'])
        records = {r['name']:r for r in manifest['tensors']}
        with (self.output / PART).open('rb') as file:
            meta, entries, base = fmt.read_header(file)
            self.assertNotIn(SELECTED, json.loads(meta['glm_vq.fp8_scales'][1]))
            for entry in entries:
                name = entry['name']
                expected = self.imported[name] if name == SELECTED else self.records[name]
                self.assertEqual(records[name], expected)
                self.assertEqual(entry['typ'], expected['type_id'])
                file.seek(base + entry['offset'])
                self.assertEqual(hashlib.sha256(file.read(expected['payload_bytes'])).hexdigest(),
                    expected['payload_sha256'])
        report = json.loads((self.output / 'q4k-selection.json').read_text())
        self.assertEqual((report['changed_count'], report['total_q4']), (1, 2))

    def test_reject_bad_parent(self):
        path = self.donor / 'tensor-manifest.json'
        manifest = json.loads(path.read_text())
        next(r for r in manifest['tensors'] if r['name']==SELECTED)['parent_payload_sha256'] = 'wrong'
        path.write_text(json.dumps(manifest))
        self.assertNotEqual(self._run_select().returncode, 0)
        self.assertFalse(self.output.exists())

    def test_preserve_prior_report(self):
        report = self.source / 'q4k-selection.json'
        original = '{"previous":true}\n'
        report.write_text(original)
        path = self.source / 'files.json'
        files = json.loads(path.read_text())
        files['files'].append(dict(path=report.name, bytes=report.stat().st_size, sha256=fmt.digest(report)))
        path.write_text(json.dumps(files))
        result = self._run_select()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(report.read_text(), original)
        output = json.loads((self.output / 'files.json').read_text())
        self.assertEqual(sum(entry['path']==report.name for entry in output['files']), 1)

    def test_reject_scale_change(self):
        path = self.donor / 'tensor-manifest.json'
        manifest = json.loads(path.read_text())
        next(r for r in manifest['tensors'] if r['name']==SCALE)['payload_sha256'] = 'different-scales'
        path.write_text(json.dumps(manifest))
        result = self._run_select()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('different FP8 scales', result.stderr)
        self.assertFalse((self.output / PART).exists())

    def test_reject_control(self):
        self.assertNotEqual(self._run_select(CONTROL).returncode, 0)
        self.assertFalse(self.output.exists())


if __name__ == '__main__':
    unittest.main()
