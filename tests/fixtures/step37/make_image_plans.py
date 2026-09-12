#!/usr/bin/env python3
"""Extract only the pinned ImagePatcher for independent geometry fixtures."""
import ast
import hashlib
import json
import math
from pathlib import Path
import sys
from itertools import product
import numpy as np
from PIL import Image

source = Path(sys.argv[1]).read_bytes()
module = ast.parse(source)
node = next(n for n in module.body if isinstance(n, ast.ClassDef) and n.name == 'ImagePatcher')
namespace = dict(np=np, Image=Image, ceil=math.ceil, product=product, MAX_IMAGE_SIZE=3024)
exec(compile(ast.Module(body=[node], type_ignores=[]), '<official ImagePatcher>', 'exec'), namespace)
patcher = namespace['ImagePatcher']()
cases = [(1, 1), (31, 2000), (32, 2000), (728, 728), (729, 728), (728, 480),
         (728, 486), (504, 504), (1008, 504), (1108, 1008), (1109, 1008),
         (3024, 3024), (3025, 2000), (32768, 32), (32, 32768), (4000, 1800)]
cases += [(width, height) for width, height in cases.copy() if width != height for width, height in [(height, width)]]
vectors = []
for w, h in cases:
    padded = patcher.get_image_size_for_padding(w, h)
    base = patcher.get_image_size_for_preprocess(*padded)
    window = patcher.determine_window_size(max(base), min(base))
    crop = patcher.get_image_size_for_crop(*base, window) if window else base
    boxes, grid = patcher.slide_window(*crop, [(window, window)], [(window, window)]) if window else ([], (0, 0))
    count, newlines = patcher.get_num_patches(w, h)
    vectors.append(dict(source=[w, h], padded=padded, base=base, crop=crop, window=window,
        boxes=boxes, grid=grid, tokens=count * 83 + 171 + newlines))
out = Path(__file__).with_name('image-plans.json')
out.write_text(json.dumps(dict(source_revision='5f6244077ac62e04eec3f320501ff8c2b293373a',
    processor_sha256=hashlib.sha256(source).hexdigest(), vectors=vectors), indent=2) + '\n')
