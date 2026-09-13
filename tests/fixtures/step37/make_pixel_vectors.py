#!/usr/bin/env python3
"""Execute the pinned official Step image processor on bounded RGB fixtures."""
import argparse
import ast
import hashlib
import json
import math
from itertools import product
from pathlib import Path
from typing import Union

import numpy as np
import PIL
from PIL import Image
import torch
import torchvision
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path, help='pinned processing_step3.py')
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    source = args.source.read_bytes()
    names = {'GPUToTensor', 'Step3VisionProcessor', 'ImagePatcher'}
    classes = [node for node in ast.parse(source).body if isinstance(node, ast.ClassDef) and node.name in names]
    assert len(classes) == len(names)
    namespace = dict(torch=torch, transforms=transforms, np=np, Image=Image, Union=Union,
                     BaseImageProcessor=object, InterpolationMode=InterpolationMode,
                     ceil=math.ceil, product=product, MAX_IMAGE_SIZE=3024)
    exec(compile(ast.Module(body=classes, type_ignores=[]), str(args.source), 'exec'), namespace)
    patcher = namespace['ImagePatcher']()
    processor = namespace['Step3VisionProcessor'](728, 'bilinear', 504)
    cases = [(33, 27), (97, 81), (727, 480), (729, 728), (31, 140),
             (1108, 600), (1109, 600), (800, 300), (3041, 777)]
    vectors = []
    for index, (width, height) in enumerate(cases):
        yy, xx = np.indices((height, width))
        channels = [(xx * 17 + yy * 31 + c * 53 + (xx // 7 % 2) * 71) % 256 for c in range(3)]
        rgb = np.stack(channels, -1).astype(np.uint8)
        image = Image.fromarray(rgb)
        name = f'case-{index}'
        image.save(args.output / (name + '.png'))
        base, patches, newlines = patcher(image)
        cropped = []
        for ci, crop in enumerate([base, *patches]):
            stem = f'{name}-{ci}'
            array = np.asarray(crop)
            array.tofile(args.output / (stem + '.rgb'))
            values = processor(crop, is_patch=ci > 0)['pixel_values'].squeeze(0).contiguous().numpy()
            values.astype('<f4').tofile(args.output / (stem + '.f32'))
            cropped.append(dict(stem=stem, input_size=list(crop.size), shape=list(values.shape),
                                rgb_sha256=hashlib.sha256(array).hexdigest(),
                                pixels_sha256=hashlib.sha256(values).hexdigest()))
        vectors.append(dict(name=name, source=[width, height], crops=cropped, newlines=newlines))
        print(name, (width, height), len(patches), flush=True)
    # Tiny interpolation vectors run without external files or large buffers.
    rgb = np.arange(7 * 5 * 3, dtype=np.uint8).reshape(5, 7, 3) * 2
    x = transforms.ToTensor()(Image.fromarray(rgb))
    tiny = []
    for width, height in [(3, 2), (13, 9), (7, 2), (3, 5)]:
        y = transforms.functional.resize(x, [height, width], InterpolationMode.BILINEAR, antialias=True)
        z = Image.fromarray(rgb).resize((width, height), Image.Resampling.BILINEAR)
        tiny.append(dict(output=[width, height], float=y.flatten().tolist(), rgb=np.asarray(z).flatten().tolist()))
    metadata = dict(source_revision='5f6244077ac62e04eec3f320501ff8c2b293373a',
                    processor_sha256=hashlib.sha256(source).hexdigest(), pillow=PIL.__version__,
                    torch=torch.__version__, torchvision=torchvision.__version__)
    (args.output / 'reference.json').write_text(json.dumps(dict(**metadata, vectors=vectors), indent=2) + '\n')
    Path(__file__).with_name('resize-vectors.json').write_text(json.dumps(dict(**metadata, vectors=tiny), indent=2) + '\n')


if __name__ == '__main__':
    main()
