#!/usr/bin/env python3
"""Write the chosen expert variant into recipe.json, RECIPE.md and the model card.

usage: select_variant.py --experts-dir variants/<V>/parts [--measured measured.md] [--root /workspace/axk2]

The selection (weighting mode, codebook scope, column power, gain refit) is
read from the variant's own layer receipts, which must all agree, so the
recipe and the docs cannot name a variant other than the one assembled. Fills
@@SELECTED_MODE@@ / @@SELECTED_BOOKS@@ / @@SELECTED_POWER@@ / @@SELECTED@@ and,
with --measured, @@MEASURED@@ (the file's markdown). Writes atomically and
reports any placeholder left anywhere.
"""
import argparse
import json
import os
from pathlib import Path
import re

BOOKS = {'shared': 'one shared book per layer (gate/up/down, ID 4L)',
         'split2': 'gate+up (4L) and down (4L+2) books per layer',
         'split3': 'separate gate (4L), up (4L+3) and down (4L+2) books per layer'}
PLACEHOLDER = re.compile(r'@@[A-Z_]+@@')


def write(path, text):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(text)
    os.replace(tmp, path)


def selection(experts):
    seen = set()
    for f in sorted(experts.glob('layer-*-experts.json')):
        r = json.loads(f.read_text())
        w = r.get('weighting') or {}
        seen.add((r['mode'], r['codebook_scope'], float(w.get('col_power', 1.0)), r['gain_refit'],
                  float(w.get('shrink_tokens', 256.0)), float(w.get('col_floor', 0.02))))
    if len(seen) != 1:
        raise SystemExit(f'receipts disagree or are missing: {sorted(seen)}')
    return seen.pop()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--experts-dir', type=Path, required=True)
    p.add_argument('--measured', type=Path, help='markdown for the card section @@MEASURED@@')
    p.add_argument('--root', type=Path, default=Path('/workspace/axk2'))
    a = p.parse_args()
    mode, books, power, refit, shrink, floor = selection(a.experts_dir)
    recipe_path = a.root / 'recipe/recipe.json'
    recipe = json.loads(recipe_path.read_text())
    cal = recipe['calibration']
    for key, want in (('gain_refit', refit), ('shrink_tokens', shrink), ('col_floor', floor)):
        if cal.get(key) is not None and str(cal[key]) != str(want) and not (
                isinstance(cal[key], (int, float)) and float(cal[key]) == float(want)):
            raise SystemExit(f'recipe {key}={cal[key]} but the variant used {want}')
    cal.update(weighting_mode=mode, codebook_scope=books, col_power=power)
    recipe['codebook_scope'] = f'{books}: {BOOKS[books]} (4L+1 reserved for a residual stage)'
    write(recipe_path, json.dumps(recipe, indent=2) + '\n')
    pw = f'{power:g}'
    fills = {'@@SELECTED_MODE@@': f'`{mode}`', '@@SELECTED_BOOKS@@': f'`{books}` ({BOOKS[books]})',
             '@@SELECTED_POWER@@': pw,
             '@@SELECTED@@': f'`{mode}` column importance, `{books}` books, column power {pw}, '
                             f'{"LS gain refit" if refit == "ls" else "RMS gains"}'}
    if a.measured:
        fills['@@MEASURED@@'] = a.measured.read_text().strip()
    for path in (a.root / 'docs/release/RECIPE.md', a.root / 'docs/README.public.md'):
        text = path.read_text()
        for k, v in fills.items():
            text = text.replace(k, v)
        write(path, text)
    left = {}
    for path in [recipe_path, a.root / 'docs/README.public.md'] + sorted((a.root / 'docs/release').glob('*')):
        if path.is_file() and path.suffix in ('.md', '.json'):
            found = sorted(set(PLACEHOLDER.findall(path.read_text())))
            if found:
                left[str(path)] = found
    print(json.dumps({'selected': {'mode': mode, 'books': books, 'col_power': power, 'gain_refit': refit,
                                   'shrink_tokens': shrink, 'col_floor': floor},
                      'placeholders_left': left}, indent=1))


if __name__ == '__main__':
    main()
