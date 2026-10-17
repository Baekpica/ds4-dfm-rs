#!/usr/bin/env python3
"""VQ KDA source arithmetic and append-state gate; no model weights."""
import argparse
import ctypes as ct
import hashlib
import json
from pathlib import Path

import torch
from transformers.models.glm5_next import modeling_glm5_next as source

SOURCE_SHA = 'a56b3f59f79c4430505bbc85557e0abdc35098561b831ab79cf14661aac1f6a5'
DIM, CONV, LOWER = 128, 4, -5.


class Kernel:
    def __init__(self, path):
        self.lib = ct.CDLL(str(path.resolve()))
        self.fn = self.lib.test_glmvq_kda
        self.fn.argtypes = [ct.c_void_p] * 15 + [ct.c_uint32] * 2 + [ct.c_float, ct.c_void_p]
        self.fn.restype = ct.c_int

    def run(self, state, tails, values, weights, a, dt):
        q, k, v, g, beta = values
        rows, heads = q.shape[:2]
        out = torch.empty_like(q)
        tensors = [out, state, *tails, q, k, v, g, beta, *weights, a, dt]
        rc = self.fn(*[t.data_ptr() for t in tensors], rows, heads, LOWER,
                     torch.cuda.current_stream().cuda_stream)
        assert rc == 0, rc
        torch.cuda.synchronize()
        return out


def rounded(*shape):
    return torch.randn(shape, device='cuda').bfloat16().float()


def reference(state, tails, values, weights, a, dt, mode):
    q, k, v, g, beta = values
    rows, heads = q.shape[:2]
    conv = []
    new_tails = []
    for raw, tail, weight in zip([q, k, v], tails, weights):
        x = raw.bfloat16().reshape(rows, heads * DIM).T.unsqueeze(0)
        history = tail.bfloat16().reshape(1, heads * DIM, CONV).clone()
        result = source.causal_conv1d_update(x, history, weight.bfloat16().reshape(heads * DIM, CONV), activation='silu')
        conv.append(result.squeeze(0).T.reshape(1, rows, heads, DIM))
        new_tails.append(history.float().reshape(heads, DIM, CONV))
    forget = LOWER * torch.sigmoid(a.exp().view(1, heads, 1) * (g + dt))
    beta = torch.sigmoid(beta.bfloat16()).unsqueeze(0)
    fn = source.recurrent_kimi_delta_attention if mode == 'recurrent' else source.chunk_kimi_delta_attention
    out, final = fn(*conv, g=forget.unsqueeze(0), beta=beta,
                    initial_state=state.unsqueeze(0).clone(), output_final_state=True,
                    use_qk_l2norm_in_kernel=True)
    return out.squeeze(0).float(), final.squeeze(0), new_tails


def compare(label, actual, expected, state, expected_state):
    error, state_error = actual - expected, state - expected_state
    report = {'case': label, 'output_max_abs': float(error.abs().max()),
              'output_rel_rms': float(error.norm() / expected.norm()),
              'state_max_abs': float(state_error.abs().max()),
              'state_rel_rms': float(state_error.norm() / expected_state.norm())}
    print(json.dumps(report), flush=True)
    assert torch.equal(actual, actual.bfloat16().float())
    assert report['output_rel_rms'] < .001
    torch.testing.assert_close(actual, expected, rtol=.008, atol=2e-6)
    torch.testing.assert_close(state, expected_state, rtol=3e-5, atol=2e-6)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lib', type=Path, required=True)
    args = parser.parse_args()
    assert hashlib.sha256(Path(source.__file__).read_bytes()).hexdigest() == SOURCE_SHA
    torch.manual_seed(5353)
    torch.backends.cuda.matmul.allow_tf32 = False
    kernel = Kernel(args.lib)
    for rows, heads in [(1, 1), (4, 64), (17, 3), (65, 3)]:
        state = torch.randn((heads, DIM, DIM), device='cuda') * .1
        tails = [rounded(heads, DIM, CONV) for _ in range(3)]
        values = [rounded(rows, heads, DIM) for _ in range(4)] + [rounded(rows, heads)]
        weights = [rounded(heads, DIM, CONV) * .25 for _ in range(3)]
        a, dt = torch.randn(heads, device='cuda') * .25, torch.randn((heads, DIM), device='cuda') * .25
        actual_state, actual_tails = state.clone(), [x.clone() for x in tails]
        actual = kernel.run(actual_state, actual_tails, values, weights, a, dt)
        for mode in ['recurrent', 'chunk']:
            expected, expected_state, expected_tails = reference(state, tails, values, weights, a, dt, mode)
            compare(f'{mode}/{rows}/{heads}', actual, expected, actual_state, expected_state)
            for got, want in zip(actual_tails, expected_tails):
                assert torch.equal(got, want), 'conv tail mismatch'
        if rows > 1:
            split = rows // 2
            append_state, append_tails = state.clone(), [x.clone() for x in tails]
            left = kernel.run(append_state, append_tails, [x[:split].contiguous() for x in values], weights, a, dt)
            right = kernel.run(append_state, append_tails, [x[split:].contiguous() for x in values], weights, a, dt)
            assert torch.equal(torch.cat([left, right]), actual), 'append output differs'
            assert torch.equal(append_state, actual_state), 'append state differs'
            for got, want in zip(append_tails, actual_tails):
                assert torch.equal(got, want), 'append conv differs'
    print('GLMVQ_KDA_PASS', flush=True)


if __name__ == '__main__':
    main()
