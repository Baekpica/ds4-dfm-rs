#!/usr/bin/env python3
"""Execute official Step replacement methods; no model weights required."""
import ast
import hashlib
import json
from itertools import product
from math import ceil
from pathlib import Path
import sys

import numpy as np
from PIL import Image

source = Path(sys.argv[1]).read_bytes()
tokenizer = json.loads(Path(sys.argv[2]).read_text())
ids = {v['content']: int(k) for k, v in tokenizer['added_tokens_decoder'].items()}
nodes = ast.parse(source).body
patcher = next(n for n in nodes if isinstance(n, ast.ClassDef) and n.name == 'ImagePatcher')
processor = next(n for n in nodes if isinstance(n, ast.ClassDef) and n.name == 'Step3VLProcessor')
methods = {'image_token_id', '_get_patch_repl', '_get_image_repl', '_get_image_repl_features'}
reference = ast.ClassDef(name='Reference', bases=[], keywords=[], decorator_list=[],
                         body=[n for n in processor.body if isinstance(n, ast.FunctionDef) and n.name in methods])
module = ast.fix_missing_locations(ast.Module(body=[patcher, reference], type_ignores=[]))
scope = dict(np=np, Image=Image, ceil=ceil, product=product, MAX_IMAGE_SIZE=3024, Optional=__import__('typing').Optional)
exec(compile(module, str(sys.argv[1]), 'exec'), scope)

class Tokenizer:
    def get_vocab(self):
        return ids
    def convert_tokens_to_ids(self, token):
        return ids[token]

reference = scope['Reference']()
reference.tokenizer = Tokenizer()
reference.image_token = '<im_patch>'
reference.num_image_feature_size = 169
reference.num_patch_feature_size = 81
reference.image_feature_placeholder = reference.image_token * 169
reference.patch_feature_placeholder = reference.image_token * 81
vectors = []
for width, height in [(33, 27), (800, 300), (729, 728), (1108, 600), (1109, 600), (140, 31), (1200, 1200), (3041, 777)]:
    base, patches, newlines = scope['ImagePatcher']()(Image.new('RGB', (width, height)))
    _, tokens = reference._get_image_repl_features(1, len(patches), newlines)
    vectors.append(dict(source=[width, height], patches=len(patches), newlines=newlines, tokens=tokens))
output = dict(source_revision='5f6244077ac62e04eec3f320501ff8c2b293373a',
              processor_sha256=hashlib.sha256(source).hexdigest(), vectors=vectors)
Path(__file__).with_name('media-vectors.json').write_text(json.dumps(output, indent=2) + '\n')
