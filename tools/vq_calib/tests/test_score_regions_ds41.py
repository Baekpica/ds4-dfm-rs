#!/usr/bin/env python3
"""CPU checks for domain/region accounting and held-out alignment."""
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch

import torch
from safetensors.torch import save_file

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import score_regions_ds41 as score


class Tokens:
    def token_to_id(self, text):
        return {'<｜Assistant｜>': 1, '<｜User｜>': 2, '<｜System｜>': 3}[text]


class Source:
    def __init__(self, _):
        pass

    def read(self, _):
        return {'head.weight': torch.eye(16)}


with tempfile.TemporaryDirectory() as directory:
    root = Path(directory)
    torch.manual_seed(41)
    teacher = {'input_ids': torch.tensor([0, 4, 5, 6, 7, 2, 8, 1, 9, 10, 2, 11, 1, 12, 3]),
               'offsets': torch.tensor([0, 5, 15]), 'final': torch.randn(15, 16)}
    save_file(teacher, str(root / 'teacher.safetensors'))
    save_file(teacher, str(root / 'student.safetensors'))
    (root / 'manifest.json').write_text(json.dumps({'samples': [
        {'domain': 'web', 'format': 'raw'}, {'domain': 'chat', 'format': 'chat'}]}))
    args = ['score_regions_ds41.py', '--source', str(root), '--device', 'cpu',
            '--teacher', str(root / 'teacher.safetensors'),
            '--student', f'same={root / "student.safetensors"}',
            '--manifest', str(root / 'manifest.json'), '--out', str(root / 'scores.json')]

    def run():
        with patch.object(sys, 'argv', args), patch.object(score, 'Source', Source), \
                patch.object(score.Tokenizer, 'from_file', return_value=Tokens()), \
                contextlib.redirect_stdout(io.StringIO()):
            score.main()

    run()
    result = json.loads((root / 'scores.json').read_text())
    assert [result[k]['same']['tokens'] for k in ('all', 'generated', 'prompt')] == [13, 9, 4]
    web, chat = result['by_domain']['web'], result['by_domain']['chat']
    assert web['all']['same']['tokens'] == web['generated']['same']['tokens'] == 4
    assert web['prompt'] == {}
    assert [chat[k]['same']['tokens'] for k in ('all', 'generated', 'prompt')] == [9, 5, 4]
    for scope in (result, web, chat):
        for region in ('all', 'generated', 'prompt'):
            if not scope[region]:
                continue
            value = scope[region]['same']
            assert abs(value['kl']) < 1e-7 and value['top1'] == 1.0
            assert value['nll'] == value['teacher_nll']
    positions = torch.tensor(list(range(4)) + list(range(5, 14)))
    expected = torch.nn.functional.cross_entropy(teacher['final'][positions], teacher['input_ids'][positions + 1])
    assert abs(result['all']['same']['nll'] - float(expected)) < 1e-6
    teacher['offsets'] = torch.tensor([0, 6, 15])
    save_file(teacher, str(root / 'student.safetensors'))
    try:
        run()
    except AssertionError:
        pass
    else:
        raise AssertionError('mismatched sequence offsets accepted')

print('domain totals, generated/prompt masks, NLL and alignment: passed')
