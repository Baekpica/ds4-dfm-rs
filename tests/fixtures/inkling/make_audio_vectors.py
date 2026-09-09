#!/usr/bin/env python3
"""Run pinned HF extraction methods on CPU; no model weights required."""
import argparse
import ast
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace
import warnings

import numpy as np
import torch
import torch.nn.functional as F


def load_functions(path, names, namespace):
    tree = ast.parse(path.read_text())
    nodes = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name in names]
    assert len(nodes) == len(names)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), namespace)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('source', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    namespace = dict(np=np, torch=torch, F=F, math=math, warnings=warnings)
    files = ['hf_audio_utils.py', 'hf_feature_extraction_inkling.py', 'hf_processing_inkling.py']
    load_functions(args.source / files[0], ['hertz_to_mel', 'mel_to_hertz', '_create_triangular_filter_bank', 'mel_filter_bank'], namespace)
    load_functions(args.source / files[1], ['_torch_extract_fbank_features'], namespace)
    load_functions(args.source / files[2], ['_extract_dmel_bins'], namespace)
    bank = namespace['mel_filter_bank'](801, 80, 0., 8000., 16000, norm='slaney', mel_scale='slaney')
    settings = SimpleNamespace(hop_length=800, n_fft=1600, window_size=1600,
        window=torch.hann_window(1600, periodic=True, dtype=torch.float32),
        mel_filters=torch.from_numpy(np.ascontiguousarray(bank.T, dtype=np.float32)),
        bin_centers=torch.linspace(-7., 2., 16, dtype=torch.float64).float(),
        dmel_min_value=-7., dmel_max_value=2.)
    cases = []
    for name, count in [('silence', 1), ('impulse', 799), ('tones', 800), ('noise', 801), ('tones', 2400), ('noise', 52001)]:
        if name == 'silence':
            wave = np.zeros(count, dtype=np.float32)
        elif name == 'impulse':
            wave = np.zeros(count, dtype=np.float32)
            wave[0], wave[-1] = 0.75, -0.25
        elif name == 'tones':
            t = np.arange(count, dtype=np.float64) / 16000
            wave = (0.2 * np.sin(2 * np.pi * 440 * t) + 0.03 * np.cos(2 * np.pi * 1923 * t)).astype(np.float32)
        else:
            wave = np.array([(((i * 1664525 + 1013904223) & 0xFFFF) - 32768) / 131072 for i in range(count)], dtype=np.float32)
        features = namespace['_torch_extract_fbank_features'](settings, torch.from_numpy(wave)[None])
        codes = namespace['_extract_dmel_bins'](settings, features)
        # Long fixture uses an exact integer waveform recipe to stay compact.
        cases.append(dict(name=name, samples=count, waveform=wave.tolist() if count < 3000 else None,
                          log_mel=features[0].tolist(), codes=codes[0].tolist()))
    middle = (settings.bin_centers[:-1] + settings.bin_centers[1:]) / 2
    quant = torch.cat([torch.tensor([-8., 3.]), settings.bin_centers, middle,
                       torch.nextafter(middle, torch.full_like(middle, -torch.inf)),
                       torch.nextafter(middle, torch.full_like(middle, torch.inf))])
    out = dict(source_revision='cbc1651a032b923da7f4b44b3d0e6f68e6ba6b55',
               sources={name: hashlib.sha256((args.source / name).read_bytes()).hexdigest() for name in files},
               torch=torch.__version__, numpy=np.__version__,
               window=settings.window.tolist(), bin_centers=settings.bin_centers.tolist(),
               quant_values=quant.tolist(), quant_codes=namespace['_extract_dmel_bins'](settings, quant).tolist(),
               cases=cases)
    args.output.write_text(json.dumps(out, separators=(',', ':')) + '\n')
    print(hashlib.sha256(args.output.read_bytes()).hexdigest())


if __name__ == '__main__':
    main()
