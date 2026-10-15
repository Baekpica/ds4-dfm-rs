"""Bounded actual-weight register-MMA check; NumPy and system CUDA only."""
import ctypes as ct
import hashlib
import json
from pathlib import Path
import subprocess
import argparse
import os

import numpy as np

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--lib', type=Path, required=True)
parser.add_argument('--fixture', type=Path)
parser.add_argument('--report', type=Path)
parser.add_argument('--mapping-only', action='store_true')
args = parser.parse_args()
root = Path(__file__).resolve().parents[1]
cuda_home = Path(os.environ.get('CUDA_HOME', '/usr/local/cuda')).resolve()
p, u, z = ct.c_void_p, ct.c_uint32, ct.c_size_t


class Matrix(ct.Structure):
    _fields_ = [('main', p), ('plane', p), ('book', p), ('gain', p),
                ('rows', u), ('cols', u), ('precision', ct.c_int)]


class Source(ct.Structure):
    _fields_ = [('data', p), ('scale', p), ('vq', Matrix),
                ('rows', u), ('cols', u), ('storage', ct.c_int)]


class Item(ct.Structure):
    _fields_ = [('matrix', u * 3), ('offset', u), ('tokens', u)]


assert (ct.sizeof(Matrix), ct.sizeof(Source), ct.sizeof(Item)) == (48, 80, 20)
cuda = ct.CDLL(str(cuda_home / 'lib64/libcudart.so'))
lib = ct.CDLL(str(args.lib.resolve()))
cuda.cudaMalloc.argtypes = [ct.POINTER(p), z]
cuda.cudaMemcpy.argtypes = [p, p, z, ct.c_int]
cuda.cudaFree.argtypes = [p]
cuda.cudaMemset.argtypes = [p, ct.c_int, z]
lib.mma_resources.argtypes = [ct.c_int, ct.POINTER(ct.c_int), ct.POINTER(ct.c_int), ct.POINTER(z)]
lib.mma_launch.argtypes = [ct.c_int, p, p, u, p, p, p, p, p, u, u, ct.c_float, u]
lib.mma_ref_create.argtypes, lib.mma_ref_create.restype = [p], p
lib.mma_ref.argtypes = [p, ct.POINTER(Source), u, p, p, p, ct.c_uint64]
lib.mma_ref_free.argtypes = [p]
lib.mma_swiglu.argtypes = [p, p, p, ct.c_uint64]


def ok(status):
    assert status == 0, f'CUDA/cuBLAS status {status}'


allocations = []


def alloc(size):
    value = p()
    ok(cuda.cudaMalloc(ct.byref(value), size))
    allocations.append(value)
    return value.value


def put(array):
    array = np.ascontiguousarray(array)
    dest = alloc(array.nbytes)
    ok(cuda.cudaMemcpy(dest, array.ctypes.data, array.nbytes, 1))
    return dest


def get(src, shape, dtype=np.float32):
    array = np.empty(shape, dtype=dtype)
    ok(cuda.cudaMemcpy(array.ctypes.data, src, array.nbytes, 2))
    return array


def bf_bits(x):
    bits = np.asarray(x, dtype=np.float32).view(np.uint32)
    return ((bits + np.uint32(0x7fff) + ((bits >> 16) & 1)) >> 16).astype(np.uint16)


def bf_float(bits):
    return (bits.astype(np.uint32) << 16).view(np.float32)


def restore(fixture, slot):
    key = f'm{slot}_'
    rows, cols = map(int, fixture[key + 'shape'])
    raw = fixture[key + 'main'].reshape(-1, 3).astype(np.uint32)
    index = np.empty(raw.shape[0] * 2, np.uint32)
    index[::2] = raw[:, 0] | ((raw[:, 1] & 15) << 8)
    index[1::2] = (raw[:, 1] >> 4) | (raw[:, 2] << 4)
    high = np.unpackbits(fixture[key + 'plane'].reshape(-1), bitorder='little')
    index |= high[:index.size].astype(np.uint32) << 12
    words = fixture['book'][index].reshape(rows, cols)
    sign = np.where(words & 128, -1.0, 1.0).astype(np.float32)
    exp = ((words >> 3) & 15).astype(np.int32)
    mant = (words & 7).astype(np.float32)
    assert not np.any((exp == 15) & (mant == 7))
    values = np.where(exp == 0, mant * np.float32(2.0**-9),
                      np.ldexp(1 + mant / 8, exp - 7)).astype(np.float32) * sign
    gains = fixture[key + 'gain'].astype(np.float32).reshape(rows, 1)
    return bf_float(bf_bits(values * gains)).astype(np.float64)


def gamma(steps, unit):
    return steps * unit / (1 - steps * unit)


def numeric(actual, reference, abs_dot, cols, algorithm):
    # Products of finite BF16 operands fit FP32 exactly. Conservatively permit
    # 1 FP32 ULP in each MMA add (PTX does not prescribe its rounding order),
    # then RN FP32 partial sums. cuBLAS gets the serial-K rounding envelope.
    if algorithm == 'mma':
        inner, outer = gamma(15, 2.0**-23), gamma(cols // 16 - 1, 2.0**-24)
        factor = inner + outer + inner * outer
    else:
        factor = gamma(cols - 1, 2.0**-24)
    dot_bound = factor * abs_dot
    magnitude = np.maximum(np.abs(reference) + dot_bound, np.finfo(np.float32).tiny)
    bf_half_ulp = np.exp2(np.floor(np.log2(magnitude)) - 8)
    bound = dot_bound + bf_half_ulp
    error = np.abs(actual.astype(np.float64) - reference)
    finite = bool(np.isfinite(actual).all())
    failures = int(np.count_nonzero(error > bound))
    return dict(finite=finite, outside=failures, count=actual.size,
                max_abs=float(error.max()), max_bound_ratio=float((error / bound).max()),
                relative_rms=float(np.linalg.norm(error) / np.linalg.norm(reference)))


def mapping():
    lib.mma_fits.argtypes = [ct.POINTER(Source), u, u, p]
    lib.mma_mapping.argtypes = [p] * 7 + [u] * 3
    # The predicate only reads descriptors; no fake address is dereferenced.
    source = Source(None, None, Matrix(128, 256, 512, 1024, 2048, 4096, 1), 2048, 4096, 2)
    assert lib.mma_fits(ct.byref(source), 2048, 4096, 512)
    for field, bad in [('main', 129), ('plane', 0), ('book', 528),
                       ('gain', 1025), ('precision', 2), ('rows', 1024)]:
        prior = getattr(source.vq, field)
        setattr(source.vq, field, bad)
        assert not lib.mma_fits(ct.byref(source), 2048, 4096, 512), field
        setattr(source.vq, field, prior)
    source.storage = 1
    assert not lib.mma_fits(ct.byref(source), 2048, 4096, 512)

    rng = np.random.default_rng(5353)
    rows, width, used, experts = 33, 4096, 8, 288
    routes = np.stack([rng.choice(experts, used, replace=False) for _ in range(rows)])
    grouped = np.argsort(routes.ravel(), kind='stable').astype(np.uint32)
    inverse = np.argsort(grouped).astype(np.uint32)
    order = inverse.reshape(rows, used)[np.arange(rows)[:, None], np.argsort(routes, axis=1)]
    input = rng.normal(size=(rows, width)).astype(np.float32)
    down = bf_float(bf_bits(rng.normal(size=(rows * used, width)).astype(np.float32)))
    weights = rng.uniform(.01, 1, size=rows * used).astype(np.float32)
    gathered, output = alloc(rows * used * width * 2), alloc(rows * width * 4)
    ok(lib.mma_mapping(gathered, output, put(input), put(down), put(weights),
                       put(grouped), put(order), rows, width, used))
    expected = np.zeros((rows, width), np.float32)
    for k in range(used):
        group = order[:, k]
        contribution = bf_float(bf_bits(down[group] * weights[grouped[group], None]))
        expected = bf_float(bf_bits(expected + contribution))
    assert np.array_equal(get(gathered, (rows * used, width), np.uint16), bf_bits(input[grouped // used]))
    assert np.array_equal(get(output, (rows, width)), expected)
    print('GLMVQ_MMA_MAPPING_PASS', flush=True)


def main():
    assert args.fixture, '--fixture is required for actual-weight arithmetic'
    fixture = np.load(args.fixture)
    widths = [1, 7, 8, 9, 16, 17, 31, 32]
    offsets = np.r_[0, np.cumsum(widths[:-1])]
    total = sum(widths)
    input = np.concatenate([fixture['input'][(np.arange(n) + e) % 22]
                            for e, n in enumerate(widths)])
    input = bf_float(bf_bits(input))
    sources = (Source * 24)()
    book = put(fixture['book'])
    for slot in range(24):
        key = f'm{slot}_'
        rows, cols = map(int, fixture[key + 'shape'])
        main, plane, gain = [put(fixture[key + field]) for field in ['main', 'plane', 'gain']]
        matrix = Matrix(main, plane, book, gain, rows, cols, 1)
        sources[slot] = Source(None, None, matrix, rows, cols, 2)
    source_gpu = alloc(ct.sizeof(sources))
    ok(cuda.cudaMemcpy(source_gpu, ct.addressof(sources), ct.sizeof(sources), 1))
    items = (Item * 8)(*[Item((u * 3)(3 * e, 3 * e + 1, 3 * e + 2), int(offsets[e]), widths[e])
                        for e in range(8)])
    item_gpu = alloc(ct.sizeof(items))
    ok(cuda.cudaMemcpy(item_gpu, ct.addressof(items), ct.sizeof(items), 1))
    alias = (Item * 8)(*[Item((u * 3)(3 * e + 1, 3 * e + 1, 3 * e + 2), int(offsets[e]), widths[e])
                        for e in range(8)])
    alias_gpu = alloc(ct.sizeof(alias))
    ok(cuda.cudaMemcpy(alias_gpu, ct.addressof(alias), ct.sizeof(alias), 1))
    input_gpu = put(bf_bits(input))
    gate_gpu, pure_up_gpu = [alloc(total * 2048 * 4) for _ in range(2)]
    mid_gpu, output_gpu = alloc(total * 2048 * 2), alloc(total * 4096 * 4)
    for pointer, count in [(gate_gpu, total * 2048 * 4), (pure_up_gpu, total * 2048 * 4),
                           (mid_gpu, total * 2048 * 2), (output_gpu, total * 4096 * 4)]:
        ok(cuda.cudaMemset(pointer, 255, count))
    blocks = []
    for stage in range(3):
        count, registers, local = ct.c_int(), ct.c_int(), z()
        ok(lib.mma_resources(stage, ct.byref(count), ct.byref(registers), ct.byref(local)))
        blocks.append(count.value)
    ok(lib.mma_launch(0, source_gpu, item_gpu, 8, book, input_gpu, gate_gpu, mid_gpu,
                      output_gpu, 2048, 4096, 10, blocks[0]))
    ok(lib.mma_launch(0, source_gpu, alias_gpu, 8, book, input_gpu, pure_up_gpu, mid_gpu,
                      output_gpu, 2048, 4096, 10, blocks[0]))
    ok(lib.mma_launch(1, source_gpu, item_gpu, 8, book, input_gpu, gate_gpu, mid_gpu,
                      output_gpu, 2048, 4096, 10, blocks[1]))
    ok(lib.mma_launch(2, source_gpu, item_gpu, 8, book, mid_gpu, gate_gpu, mid_gpu,
                      output_gpu, 4096, 2048, 10, blocks[2]))
    ok(cuda.cudaDeviceSynchronize())
    gate, up = [get(ptr, (total, 2048)) for ptr in [gate_gpu, pure_up_gpu]]
    mid = bf_float(get(mid_gpu, (total, 2048), np.uint16))
    output = get(output_gpu, (total, 4096))
    fusion_gpu = alloc(total * 2048 * 4)
    ok(lib.mma_swiglu(gate_gpu, pure_up_gpu, fusion_gpu, total * 2048))
    fusion = get(fusion_gpu, (total, 2048))
    fusion_equal = bool(np.array_equal(fusion, mid))
    print(json.dumps(dict(event='fusion', exact=fusion_equal,
                          differences=int(np.count_nonzero(fusion != mid)))), flush=True)

    workspace, scratch_bytes = alloc(32 * 1024**2), 16 * 1024**2 + 32 * 6144 * 2
    scratch = alloc(scratch_bytes)
    handle = lib.mma_ref_create(workspace)
    assert handle, 'system cuBLAS handle/version'
    input_f32, mid_f32 = put(input), put(mid)
    ref_gpu = alloc(32 * 4096 * 4)
    receipt = dict(fixture_sha256=hashlib.sha256(args.fixture.read_bytes()).hexdigest(),
                   widths=widths, source_sha256=hashlib.sha256((root / 'cuda/glm_vq_mma.cuh').read_bytes()).hexdigest(),
                   binary_sha256=hashlib.sha256(args.lib.read_bytes()).hexdigest(),
                   fusion_exact=fusion_equal, cases=[],
                   numerical_contract='FP64 same BF16 operands; per-output sumabs FP32 dot bound plus BF16 half ULP; fusion checked independently',
                   scope='isolated layer3 actual weights; layer0 activations; not final model quality')
    try:
        for slot in range(24):
            e, stage = divmod(slot, 3)
            off, nt = int(offsets[e]), widths[e]
            src = sources[slot]
            values = (mid if stage == 2 else input)[off:off + nt].astype(np.float64)
            actual = [gate, up, output][stage][off:off + nt]
            input_ptr = (mid_f32 if stage == 2 else input_f32) + off * src.cols * 4
            ok(lib.mma_ref(handle, ct.byref(src), nt, input_ptr, ref_gpu, scratch, scratch_bytes))
            library = get(ref_gpu, (nt, src.rows))
            weight = restore(fixture, slot)
            reference, abs_dot = values @ weight.T, np.abs(values) @ np.abs(weight.T)
            candidate_result = numeric(actual, reference, abs_dot, src.cols, 'mma')
            library_result = numeric(library, reference, abs_dot, src.cols, 'cublas')
            case = dict(slot=slot, stage=stage, width=nt, mma=candidate_result, cublas=library_result,
                        mma_cublas_max_abs=float(np.max(np.abs(actual - library))),
                        mma_cublas_differing=int(np.count_nonzero(actual != library)))
            receipt['cases'].append(case)
            print(json.dumps(case), flush=True)
    finally:
        ok(lib.mma_ref_free(handle))
    libraries = sorted(set(line.split()[-1] for line in Path('/proc/self/maps').read_text().splitlines()
        if any(x in line for x in ['libcublas', 'libcudart'])))
    assert libraries and all(str(cuda_home) + '/' in path for path in libraries), libraries
    receipt['libraries'] = libraries
    receipt['clock'] = subprocess.check_output(['nvidia-smi', '--query-gpu=clocks.sm,clocks.mem',
                                               '--format=csv,noheader,nounits'], text=True).strip()
    # cuBLAS default may use BF16 intermediate reductions. Its pure-FP32
    # envelope is characterization, not a condition imposed on this candidate.
    receipt['pass'] = fusion_equal and all(c[k]['finite'] and not c[k]['outside']
                                           for c in receipt['cases'] for k in ['mma'])
    if args.report:
        args.report.write_text(json.dumps(receipt, indent=2))
    assert receipt['pass'], 'isolated candidate numerical contract failed'


if __name__ == '__main__':
    try:
        mapping()
        if not args.mapping_only:
            main()
    finally:
        for pointer in reversed(allocations):
            cuda.cudaFree(pointer)
