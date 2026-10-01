#!/usr/bin/env python3
"""Bounded public embedding API regression using real Q6_K embedding blocks.

Compiles the exact API/kernel section from ds4_cuda.cu with weight-address
resolution stubbed to a device fixture. This does not test full model binding.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess

import numpy as np


HARNESS = r'''
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <vector>
#include <cmath>
struct ds4_gpu_tensor { void *ptr; uint64_t bytes; };
static cudaStream_t ds4_current_stream() { return nullptr; }
static int ds4_tensor_device_idx(const ds4_gpu_tensor *) { return 1; }
static int cuda_ok(cudaError_t e,const char *) { return e==cudaSuccess; }
static const char *cuda_model_range_ptr(const void *p,uint64_t o,uint64_t,uint64_t) { return (const char *)p+o; }
static const char *cuda_model_range_ptr(const void *p,uint64_t o,uint64_t,const char *) { return (const char *)p+o; }
static const void *cuda_resolve_weight_ptr(const void *p,uint64_t o,uint64_t,int,const char *) { return (const char *)p+o; }
static int ds4_mmq_pq2_0_rows_f32(float *,const unsigned char *,const int32_t *,uint32_t,uint32_t,uint32_t,cudaStream_t) { return -1; }
'''

MAIN = r'''
static std::vector<unsigned char> read_bytes(const char *p) {
    FILE *f=std::fopen(p,"rb"); if(!f){std::exit(2);} std::fseek(f,0,SEEK_END);
    const long n=std::ftell(f); std::rewind(f); std::vector<unsigned char> b(n);
    if(std::fread(b.data(),1,n,f)!=(size_t)n){std::exit(2);} std::fclose(f); return b;
}
int main(int argc,char **argv) {
    if(argc!=6){return 2;} cudaSetDevice(std::atoi(argv[5]));
    const uint32_t type=std::atoi(argv[1]),vocab=7,dim=3072;
    auto raw=read_bytes(argv[2]),ref=read_bytes(argv[3]);
    const float *expected=(const float *)ref.data();
    std::vector<int32_t> tokens={0,1,2,3,4,5,6,-1,7,2147483647};
    void *weights; cudaMalloc(&weights,raw.size()); cudaMemcpy(weights,raw.data(),raw.size(),cudaMemcpyHostToDevice);
    ds4_gpu_tensor out{},ids{}; out.bytes=tokens.size()*dim*sizeof(float); ids.bytes=tokens.size()*sizeof(int32_t);
    cudaMalloc(&out.ptr,out.bytes); cudaMalloc(&ids.ptr,ids.bytes); cudaMemcpy(ids.ptr,tokens.data(),ids.bytes,cudaMemcpyHostToDevice);
    int ok=ds4_gpu_embed_tokens_quant_tensor(&out,&ids,weights,raw.size(),0,type,vocab,tokens.size(),dim);
    if(!ok){std::printf("{\"type\":%u,\"accepted\":false,\"api\":\"batch\"}\n",type);return 1;}
    std::vector<float> actual(tokens.size()*dim); cudaMemcpy(actual.data(),out.ptr,out.bytes,cudaMemcpyDeviceToHost);
    float max_error=0;
    for(unsigned r=0;r<tokens.size();r++) {
        for(unsigned d=0;d<dim;d++) {
            float truth=(tokens[r]>=0&&(unsigned)tokens[r]<vocab)?expected[tokens[r]*dim+d]:0.0f;
            max_error=fmaxf(max_error,fabsf(actual[r*dim+d]-truth));
        }
    }
    for(unsigned r=0;r<vocab;r++) {
        ok=ds4_gpu_embed_token_quant_tensor(&out,weights,raw.size(),0,type,vocab,r,dim);
        if(!ok){std::printf("{\"type\":%u,\"accepted\":false,\"api\":\"single\"}\n",type);return 1;}
        cudaMemcpy(actual.data(),out.ptr,dim*sizeof(float),cudaMemcpyDeviceToHost);
        for(unsigned d=0;d<dim;d++){max_error=fmaxf(max_error,fabsf(actual[d]-expected[r*dim+d]));}
    }
    const bool bad_range=ds4_gpu_embed_token_quant_tensor(&out,weights,raw.size()-1,0,type,vocab,0,dim)!=0;
    const bool bad_token=ds4_gpu_embed_token_quant_tensor(&out,weights,raw.size(),0,type,vocab,vocab,dim)!=0;
    const bool bad_dim=type==14u&&ds4_gpu_embed_token_quant_tensor(&out,weights,raw.size(),0,type,vocab,0,dim-1)!=0;
    std::printf("{\"type\":%u,\"accepted\":true,\"max_error\":%.9g,\"invalid_range_rejected\":%s,\"invalid_single_token_rejected\":%s,\"invalid_q6_dimension_rejected\":%s}\n",
        type,max_error,bad_range?"false":"true",bad_token?"false":"true",bad_dim?"false":"true");
    cudaFree(weights);cudaFree(out.ptr);cudaFree(ids.ptr);
    return max_error>1e-7f||bad_range||bad_token||bad_dim?1:0;
}
'''


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', type=Path, required=True, help='GGUF shard containing token_embd.weight')
    p.add_argument('--work', type=Path, default=Path('scratch/iquest/embedding'))
    p.add_argument('--device', default='0')
    p.add_argument('--arch', default='sm_121a')
    p.add_argument('--nvcc', default=os.environ.get('NVCC', '/usr/local/cuda/bin/nvcc'))
    p.add_argument('--out', type=Path)
    args = p.parse_args()
    import gguf
    from gguf.quants import dequantize, quantize
    repo = Path(__file__).resolve().parents[1]
    directory = args.work
    directory.mkdir(parents=True, exist_ok=True)
    source = (repo / 'ds4_cuda.cu').read_text()
    begin = source.index('__global__ static void embed_token_q8_0_kernel(')
    end = source.index('extern "C" int ds4_gpu_embed_token_hc_tensor(', begin)
    translation = directory / 'test_embed.cu'
    translation.write_text(HARNESS + source[begin:end] + MAIN)
    binary = directory / 'test_embed'
    subprocess.run([args.nvcc, '-O2', '-std=c++17', '--fmad=false', '-arch=' + args.arch, str(translation), '-o', str(binary)], check=True)
    ids = np.asarray([0, 1, 42, 100, 1023, 50000, 159999], dtype=np.int64)
    reader = gguf.GGUFReader(str(args.model), 'r')
    embedding = next(t for t in reader.tensors if t.name == 'token_embd.weight')
    assert embedding.tensor_type == gguf.GGMLQuantizationType.Q6_K
    assert list(embedding.shape) == [3072, 160000]
    q6 = np.ascontiguousarray(embedding.data.reshape(160000, 2520)[ids]).reshape(-1)
    reference = dequantize(q6, gguf.GGMLQuantizationType.Q6_K).reshape(7, 3072)
    cases = [(14, q6, reference),
             (0, reference.view(np.uint8).reshape(-1), reference),
             (1, reference.astype(np.float16).view(np.uint8).reshape(-1), reference.astype(np.float16).astype(np.float32))]
    bf16 = ((reference.view(np.uint32) + 0x7fff + ((reference.view(np.uint32) >> 16) & 1)) >> 16).astype(np.uint16)
    cases.append((30, bf16.view(np.uint8).reshape(-1), (bf16.astype(np.uint32) << 16).view(np.float32)))
    q8 = quantize(reference, gguf.GGMLQuantizationType.Q8_0)
    cases.append((8, q8, dequantize(q8, gguf.GGMLQuantizationType.Q8_0)))
    report = {'scope': 'Exact public embedding API/kernel section with stubbed weight-address mapping; real saved Q6 blocks; excludes full native binding',
              'source_token_ids': ids.tolist(), 'oracle': 'gguf-py dequantization',
              'q6_sample_sha256': hashlib.sha256(q6.tobytes()).hexdigest(),
              'native_api_section_sha256': hashlib.sha256(source[begin:end].encode()).hexdigest(),
              'model_shard': str(args.model.resolve()), 'cuda_arch': args.arch,
              'cases': [], 'complete': True}
    for typ, raw, truth in cases:
        data, ref = directory / f'type-{typ}.bin', directory / f'type-{typ}.f32'
        raw.tofile(data); truth.tofile(ref)
        result = subprocess.run([str(binary), str(typ), str(data), str(ref), 'unused', args.device], text=True, capture_output=True)
        print(result.stdout, end='', flush=True)
        report['cases'].append({'type': typ, 'exit_code': result.returncode, 'output': result.stdout, 'stderr': result.stderr})
        report['complete'] &= result.returncode == 0
    (args.out or directory / 'parity.json').write_text(json.dumps(report, indent=2))
    return 0 if report['complete'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
