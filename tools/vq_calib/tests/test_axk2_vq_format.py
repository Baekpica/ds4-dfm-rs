"""Self-test of tools/axk2_vq_format.py with random data (no model files)."""
import hashlib, json, struct, sys, tempfile
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import axk2_vq_format as f

rng = np.random.default_rng(0)

def book(k):
    # random finite E4M3 bytes (exclude NaN 0x7f/0xff)
    b = rng.integers(0, 256, size=k * 8, dtype=np.uint8)
    b[(b & 127) == 127] = 0
    return b.tobytes()

def expect_fail(fn, what):
    try:
        fn()
    except ValueError:
        return
    raise AssertionError('accepted: ' + what)

for k, rows, cols in [(512, 64, 7168), (1024, 64, 7168), (512, 96, 2048), (1024, 96, 2048)]:
    bits = f.bits_for(k)
    raw = book(k)
    gains = rng.random(rows).astype(np.float16).astype(np.float32)
    gains[3] = 0.0
    ix = rng.integers(0, k, size=(rows, cols // 8))
    payload = f.encode_vq(gains, ix, rows, cols, k, raw, 4 * 7)
    g = f.geometry(rows, cols, k)
    assert len(payload) == g['size'] and g['size'] % 128 == 0
    t = f.VqTensor(payload, raw, (rows, cols), 28)
    assert np.array_equal(t.indices(), ix), 'index round trip'
    ref = f.FP8_TABLE[np.frombuffer(raw, np.uint8)].reshape(k, 8)[ix].reshape(rows, cols) * gains[:, None]
    assert np.array_equal(t.decode_rows(), ref.astype('<f4')), 'decode'
    assert np.array_equal(t.decode_rows(5, 7), ref[5:12].astype('<f4')), 'row range'
    # explicit layout check (docs/vq-calibration.md, Container), independent of both coders
    flat = ix.reshape(-1)
    main = np.frombuffer(payload, np.uint8, len(flat), g['main'])
    assert np.array_equal(main, flat & 255)
    plane = np.frombuffer(payload, np.uint8, g['end'] - g['plane'], g['plane'])
    for n in rng.integers(0, len(flat), 200):
        hi = flat[n] >> 8
        if bits == 9:
            assert (plane[n // 8] >> (n % 8)) & 1 == hi
        else:
            assert (plane[n // 4] >> (2 * (n % 4))) & 3 == hi
    # device-style assembly from separately packed streams must match
    low = (flat & 255).astype(np.uint8).tobytes()
    high = f._pack_plane(flat.astype(np.uint16), bits)
    assert f.assemble_vq(gains.astype('<f2').tobytes(), low, high, rows, cols, k, raw, 28) == payload
    # rejections
    bad = bytearray(payload); struct.pack_into('<Q', bad, 40, g['main'] + 128)
    expect_fail(lambda: f.VqTensor(bytes(bad), raw, (rows, cols)), 'noncanonical main offset')
    bad = bytearray(payload); struct.pack_into('<H', bad, 4, 1)
    expect_fail(lambda: f.VqTensor(bytes(bad), raw, (rows, cols)), 'version 1 header')
    bad = bytearray(payload); bad[g['main_end']] = 1 if g['main_end'] < g['plane'] else bad[g['main_end']]
    if g['main_end'] < g['plane']:
        expect_fail(lambda: f.VqTensor(bytes(bad), raw, (rows, cols)), 'nonzero padding')
    bad = bytearray(payload); bad[-1] ^= 1
    expect_fail(lambda: f.VqTensor(bytes(bad), raw, (rows, cols)), 'payload checksum')
    expect_fail(lambda: f.VqTensor(payload, book(k), (rows, cols)), 'codebook checksum')
    gneg = gains.copy(); gneg[0] = -1.0
    expect_fail(lambda: f.encode_vq(gneg, ix, rows, cols, k, raw, 28), 'negative gain')
    gnan = gains.copy(); gnan[0] = np.nan
    expect_fail(lambda: f.encode_vq(gnan, ix, rows, cols, k, raw, 28), 'nan gain')
    gun = gains.copy(); gun[1] = 0.1234567
    expect_fail(lambda: f.encode_vq(gun, ix, rows, cols, k, raw, 28), 'unrounded gain')
    ixb = ix.copy(); ixb[0, 0] = k
    expect_fail(lambda: f.encode_vq(gains, ixb, rows, cols, k, raw, 28), 'index >= K')
    expect_fail(lambda: f.geometry(rows, cols + 8, k), 'cols % 64')
    expect_fail(lambda: f.geometry(rows, cols, 4096), 'K 4096 in v2')
print('matrix round trips and rejections: ok')

# GGUF part round trip with every tensor type the release uses
with tempfile.TemporaryDirectory() as d:
    d = Path(d)
    k, raw = 512, book(512)
    gains = rng.random(32).astype(np.float16).astype(np.float32)
    m1 = f.encode_vq(gains, rng.integers(0, k, (32, 7168 // 8)), 32, 7168, k, raw, 4)
    m2 = f.encode_vq(rng.random(64).astype(np.float16).astype(np.float32), rng.integers(0, k, (64, 2048 // 8)), 64, 2048, k, raw, 4)
    q8 = np.zeros((4, 1536 // 32, 34), np.uint8); q8[:, :, :2] = np.frombuffer(np.float16(0.01).tobytes(), np.uint8)
    q4 = np.zeros((2, 512 // 256, 144), np.uint8); q4[:, :, :4] = np.frombuffer(np.array([0.01, 0.0], np.float16).tobytes(), np.uint8)
    bf = (rng.standard_normal(7168).astype(np.float32).view(np.uint32) >> 16).astype(np.uint16).tobytes()
    f32 = rng.standard_normal(256).astype('<f4').tobytes()
    tensors = [('model.layers.1.mlp.vq_codebook', [k, 8], f.BOOK, raw),
               ('model.layers.1.mlp.experts.0.gate_proj.weight', [32, 7168], f.VQ, m1),
               ('model.layers.1.mlp.experts.0.down_proj.weight', [64, 2048], f.VQ, m2),
               ('model.layers.1.self_attn.indexer.wq_b.weight', [4, 1536], f.Q8_0, q8.tobytes()),
               ('model.layers.1.self_attn.kv_b_proj.weight', [2, 512], f.Q4_K, q4.tobytes()),
               ('model.layers.1.input_layernorm.norm.weight', [7168], f.BF16, bf),
               ('model.layers.1.mlp.gate.e_score_correction_bias', [256], f.F32, f32)]
    meta = f.metadata({'4': 'model.layers.1.mlp.vq_codebook'}, 'trunk',
                      {'axk2_vq.layer': 1, 'axk2_vq.index_bits': 9, 'axk2_vq.recipe_id': 'test'})
    entries = [{'name': n, 'shape': s, 'type_id': t, 'payload_bytes': len(p)} for n, s, t, p in tensors]
    with f.PartWriter(d / 'trunk.gguf', meta, entries) as w:
        for n, _, _, p in tensors:
            w.write(n, p)
    with f.PartReader(d / 'trunk.gguf') as r:
        assert set(r.entries) == {t[0] for t in tensors}
        assert np.array_equal(r.vq('model.layers.1.mlp.experts.0.down_proj.weight').decode_rows(0, 2),
                              f.VqTensor(m2, raw).decode_rows(0, 2))
        for n, _, _, p in tensors:
            assert bytes(r.raw(n)) == p
    # foreign metadata / unused book / missing link are rejected
    for name, bad_meta, ents in [
            ('foreign', {**meta, 'beam_vq.layer': 1}, entries),
            ('unused book', f.metadata({'4': 'model.layers.1.mlp.vq_codebook'}, 'trunk'), entries[:1] + entries[3:]),
            ('missing link', f.metadata({}, 'trunk'), entries)]:
        p = d / f'bad-{name.replace(" ", "_")}.gguf'
        with f.PartWriter(p, bad_meta, ents) as w:
            for e in ents:
                w.write(e['name'], dict((t[0], t[3]) for t in tensors)[e['name']])
        try:
            f.PartReader(p).close()
        except ValueError:
            continue
        raise AssertionError('reader accepted ' + name)
    # truncated / trailing bytes
    blob = (d / 'trunk.gguf').read_bytes()
    (d / 'trunc.gguf').write_bytes(blob[:-1]); (d / 'trail.gguf').write_bytes(blob + b'\0')
    for p in ('trunc.gguf', 'trail.gguf'):
        try:
            f.PartReader(d / p).close()
        except ValueError:
            continue
        raise AssertionError('reader accepted ' + p)
    # embedding part + release verification + source mapping with consumed scales
    emb = rng.standard_normal((8, 7168)).astype(np.float32)
    emb_raw = (emb.view(np.uint32) >> 16).astype(np.uint16).tobytes()
    with f.PartWriter(d / 'embed.gguf', f.metadata(part_role='embedding', extra={'axk2_vq.residency': 'ssd-rows'}),
                      [{'name': 'model.embed_tokens.weight', 'shape': [8, 7168], 'type_id': f.BF16,
                        'payload_bytes': len(emb_raw)}]) as w:
        w.write('model.embed_tokens.weight', emb_raw)
    def sha(p):
        return hashlib.sha256(p.read_bytes()).hexdigest()
    rel = {'format': f.VARIANT, 'format_version': 1, 'entrypoint': 'release.json',
           'parts': [{'path': 'trunk.gguf', 'role': 'trunk', 'required': True, 'bytes': (d / 'trunk.gguf').stat().st_size, 'sha256': sha(d / 'trunk.gguf')},
                     {'path': 'embed.gguf', 'role': 'embedding', 'required': True, 'bytes': (d / 'embed.gguf').stat().st_size, 'sha256': sha(d / 'embed.gguf')}]}
    recs = [{'name': n, 'part': 'trunk.gguf', 'stored_shape': s, 'type_id': t, 'payload_bytes': len(p),
             'payload_sha256': hashlib.sha256(p).hexdigest()} for n, s, t, p in tensors]
    recs.append({'name': 'model.embed_tokens.weight', 'part': 'embed.gguf', 'stored_shape': [8, 7168], 'type_id': f.BF16,
                 'payload_bytes': len(emb_raw), 'payload_sha256': hashlib.sha256(emb_raw).hexdigest()})
    src = [t[0] for t in tensors[1:]] + ['model.embed_tokens.weight'] + \
          [t[0] + '_scale_inv' for t in tensors[1:5]]
    disp = {f.VQ: 'consumed_vq', f.Q4_K: 'consumed_q4k', f.Q8_0: 'consumed_q8_0', f.BF16: 'preserved', f.F32: 'preserved'}
    mapping = [{'source_name': r['name'], 'target_name': r['name'], 'disposition': disp[r['type_id']]} for r in recs[1:]]
    mapping += [{'source_name': t[0] + '_scale_inv', 'target_name': t[0], 'disposition': 'consumed_scale'} for t in tensors[1:5]]
    (d / 'release.json').write_text(json.dumps(rel))
    (d / 'tensor-manifest.json').write_text(json.dumps({'tensors': recs, 'source_names': src, 'source_mapping': mapping}))
    print(json.dumps(f.verify_release(d)))
    bad_map = mapping[:-1] + [{'source_name': tensors[4][0] + '_scale_inv', 'target_name': tensors[5][0], 'disposition': 'consumed_scale'}]
    expect_fail(lambda: f.verify_source_mapping(src, bad_map, recs), 'scale consumed by the wrong tensor')
    expect_fail(lambda: f.verify_source_mapping(src, mapping[:-1], recs), 'unaccounted scale')
print('parts, release and source mapping: ok')

# Split codebooks: up to three generated books per expert part (4L, 4L+2 down, 4L+3 up)
with tempfile.TemporaryDirectory() as d:
    d = Path(d)
    L, k = 9, 512
    books = {'model.layers.9.mlp.vq_codebook': (36, book(k)), 'model.layers.9.mlp.vq_codebook_down': (38, book(k)),
             'model.layers.9.mlp.vq_codebook_up': (39, book(k))}
    mats = []
    for proj, (rows, cols), bname in (('gate', (32, 7168), 'model.layers.9.mlp.vq_codebook'),
                                      ('up', (32, 7168), 'model.layers.9.mlp.vq_codebook_up'),
                                      ('down', (64, 2048), 'model.layers.9.mlp.vq_codebook_down')):
        bid, raw = books[bname]
        mats.append((f'model.layers.9.mlp.experts.0.{proj}_proj.weight', [rows, cols],
                     f.encode_vq(rng.random(rows).astype(np.float16).astype(np.float32),
                                 rng.integers(0, k, (rows, cols // 8)), rows, cols, k, raw, bid)))
    def write(path, links):
        ents = [{'name': n, 'shape': [k, 8], 'type_id': f.BOOK, 'payload_bytes': k * 8} for n in books]
        ents += [{'name': n, 'shape': s, 'type_id': f.VQ, 'payload_bytes': len(p)} for n, s, p in mats]
        with f.PartWriter(path, f.metadata(links, 'trunk', {'axk2_vq.layer': L, 'axk2_vq.index_bits': 9}), ents) as w:
            for n, (_, raw) in books.items():
                w.write(n, raw)
            for n, _, p in mats:
                w.write(n, p)
    write(d / 'split3.gguf', {str(b): n for n, (b, _) in books.items()})
    with f.PartReader(d / 'split3.gguf') as r:
        assert len([e for e in r.entries.values() if e['type_id'] == f.BOOK]) == 3
        r.vq('model.layers.9.mlp.experts.0.up_proj.weight').decode_rows(0, 1)
    assert f.book_id(9, 'model.layers.9.mlp.vq_codebook_down') == 38
    expect_fail(lambda: f.book_id(9, 'model.layers.8.mlp.vq_codebook'), 'book of another layer')
    # a part whose metadata maps the down book to 4L+1 must be rejected (matrices then link a missing ID)
    try:
        write(d / 'badid.gguf', {'36': 'model.layers.9.mlp.vq_codebook', '37': 'model.layers.9.mlp.vq_codebook_down',
                                 '39': 'model.layers.9.mlp.vq_codebook_up'})
        f.PartReader(d / 'badid.gguf').close()
        raise AssertionError('accepted a non-conventional codebook ID')
    except ValueError:
        pass
    recs = [{'name': n, 'type_id': f.BOOK} for n in books] + [{'name': n, 'type_id': f.VQ} for n, _, _ in mats]
    src = [n for n, _, _ in mats] + [n + '_scale_inv' for n, _, _ in mats]
    mp = [{'source_name': n, 'target_name': n, 'disposition': 'consumed_vq'} for n, _, _ in mats]
    mp += [{'source_name': n + '_scale_inv', 'target_name': n, 'disposition': 'consumed_scale'} for n, _, _ in mats]
    f.verify_source_mapping(src, mp, recs)
print('split codebooks (1-3 per part): ok')
