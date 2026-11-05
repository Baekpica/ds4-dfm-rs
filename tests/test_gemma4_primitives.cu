#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <algorithm>
#include <cmath>
#include "../cuda/gemma4_primitives.cuh"
#include "../cuda/gemma4_attn.cuh"
#include "../cuda/gemma4_attn_mma.cuh"
#include "../cuda/gemma4_attn_split.cuh"
#include "../cuda/gemma4_attn_split_mma.cuh"
#include "../ds4_gemma4_plan.h"

#define CUDA_OK(call) do { cudaError_t e_=(call); if(e_!=cudaSuccess){ \
    std::fprintf(stderr,"%s: %s\n",#call,cudaGetErrorString(e_)); std::exit(2); } } while(0)
static unsigned failures=0;

static float value(unsigned i) { return std::sin((i+1)*0.117f) + 0.3f*std::cos((i+3)*0.031f); }

template<class T> static T *device(const std::vector<T>& x) {
    T *p; CUDA_OK(cudaMalloc(&p,x.size()*sizeof(T)));
    CUDA_OK(cudaMemcpy(p,x.data(),x.size()*sizeof(T),cudaMemcpyHostToDevice));
    return p;
}

/* The container stores every norm/router weight as BF16, so the device arrays
 * here are BF16 too and the reference sees the same rounded values: a float32
 * array would test a shape the artifact never has (this is the P6 nan). */
static __nv_bfloat16 *bf16_device(const std::vector<float>& x) {
    std::vector<__nv_bfloat16> packed(x.size());
    for (unsigned i=0;i<x.size();i++) { packed[i]=__float2bfloat16_rn(x[i]); }
    return device(packed);
}

static float bf16_ulp(float x) {
    uint32_t bits; std::memcpy(&bits,&x,sizeof(bits));
    bits+=0x10000u;
    float next; std::memcpy(&next,&bits,sizeof(bits));
    return std::fabs(next-x);
}

/* Exact-match count and the worst bf16-ULP distance; a rounding-boundary
 * crossing between a float32 device reduction and the double reference is
 * one ULP, not an error. */
static void check_ulps(const char *name,const std::vector<float>& expected,const float *dev,
        float allowed_ulps,unsigned long *exact=nullptr) {
    std::vector<float> actual(expected.size());
    CUDA_OK(cudaMemcpy(actual.data(),dev,actual.size()*sizeof(float),cudaMemcpyDeviceToHost));
    float maximum=0; double worst_ulps=0; unsigned long same=0; unsigned bad=0;
    for(unsigned i=0;i<actual.size();i++) {
        if(!std::isfinite(actual[i])) { std::fprintf(stderr,"%s nonfinite at %u\n",name,i); failures++; return; }
        same+=actual[i]==expected[i];
        const float delta=std::fabs(actual[i]-expected[i]);
        const float ulp=std::max(bf16_ulp(actual[i]),bf16_ulp(expected[i]));
        const double ulps=ulp?delta/ulp:0.0;
        maximum=std::max(maximum,delta);
        worst_ulps=std::max(worst_ulps,ulps);
        if(!(ulps<=allowed_ulps+1e-6)) { bad++; }
    }
    std::printf("{\"primitive\":\"%s\",\"max_error\":%.9g,\"max_ulps\":%.6f,\"exact\":%lu,\"count\":%zu,"
        "\"outside\":%u,\"allowed_ulps\":%.3f}\n",name,maximum,worst_ulps,same,actual.size(),bad,allowed_ulps);
    if (bad) {
        for (unsigned i=0,shown=0;i<actual.size()&&shown<3;i++) {
            const float ulp=std::max(bf16_ulp(actual[i]),bf16_ulp(expected[i]));
            if (!(ulp?std::fabs(actual[i]-expected[i])/ulp<=allowed_ulps+1e-6:false)) {
                std::printf("{\"primitive\":\"%s\",\"miss\":%u,\"index\":%u,\"actual\":%.9g,"
                    "\"expected\":%.9g}\n",name,shown++,i,actual[i],expected[i]);
            }
        }
    }
    if(exact) { *exact=same; }
    if(bad) { failures++; }
}

/* Absolute-error gate for the activation, whose output approaches zero from
 * both sides: there a bf16 ULP is meaningless and the float32 reference's own
 * spread against a double evaluation is ~1e-5. Any wrong formula (SiLU, a
 * missing 0.044715, a wrong constant) misses by orders of magnitude more. */
static void check_abs(const char *name,const std::vector<float>& expected,const float *dev,
        float tolerance) {
    std::vector<float> actual(expected.size());
    CUDA_OK(cudaMemcpy(actual.data(),dev,actual.size()*sizeof(float),cudaMemcpyDeviceToHost));
    float maximum=0; unsigned long same=0; unsigned bad=0;
    for(unsigned i=0;i<actual.size();i++) {
        if(!std::isfinite(actual[i])) { std::fprintf(stderr,"%s nonfinite at %u\n",name,i); failures++; return; }
        same+=actual[i]==expected[i];
        const float delta=std::fabs(actual[i]-expected[i]);
        maximum=std::max(maximum,delta);
        bad+=delta>tolerance;
    }
    std::printf("{\"primitive\":\"%s\",\"max_error\":%.9g,\"exact\":%lu,\"count\":%zu,"
        "\"outside\":%u,\"tolerance\":%.9g}\n",name,maximum,same,actual.size(),bad,tolerance);
    if(bad) { failures++; }
}

static void rms_test() {
    const unsigned rows=3, width=G4_REF_EMBED;
    std::vector<float> in(rows*width),weight(width),expected(rows*width);
    for(unsigned i=0;i<in.size();i++) { in[i]=value(i)*1.7f; }
    for(unsigned d=0;d<width;d++) { weight[d]=g4_ref_bf16(1.0f+0.25f*std::sin(d*0.017f)); }
    g4_ref_rms(expected.data(),in.data(),weight.data(),rows,width,1e-6f);
    float *di=device(in),*out;
    __nv_bfloat16 *dw=bf16_device(weight);
    CUDA_OK(cudaMalloc(&out,expected.size()*sizeof(float)));
    g4_rms_kernel<256><<<rows,256>>>(out,di,dw,rows,width,1e-6f); CUDA_OK(cudaGetLastError());
    check_ulps("rms_weight",expected,out,1.0f);
    CUDA_OK(cudaFree(dw));
    /* v_norm: no weight, per-head width. */
    const unsigned heads=G4_REF_HEADS, dim=G4_REF_HEAD_DIM;
    std::vector<float> v(heads*dim),ve(heads*dim);
    for(unsigned i=0;i<v.size();i++) { v[i]=value(i+17)*3.1f; }
    g4_ref_rms(ve.data(),v.data(),nullptr,heads,dim,1e-6f);
    float *dv=device(v),*vo;
    CUDA_OK(cudaMalloc(&vo,ve.size()*sizeof(float)));
    g4_rms_kernel<256><<<heads,256>>>(vo,dv,nullptr,heads,dim,1e-6f); CUDA_OK(cudaGetLastError());
    check_ulps("v_norm_no_weight",ve,vo,1.0f);
    CUDA_OK(cudaFree(di)); CUDA_OK(cudaFree(out)); CUDA_OK(cudaFree(dv)); CUDA_OK(cudaFree(vo));
}

static void router_input_test() {
    const unsigned rows=2, width=G4_REF_EMBED;
    const float root=1.0f/std::sqrt((float)width);
    std::vector<float> in(rows*width),scale(width),expected(rows*width);
    for(unsigned i=0;i<in.size();i++) { in[i]=value(i)*2.3f; }
    for(unsigned d=0;d<width;d++) { scale[d]=g4_ref_bf16(3.0f+0.02f*value(d)); }
    g4_ref_router_input(expected.data(),in.data(),scale.data(),rows,width,1e-6f,root);
    float *di=device(in),*out;
    __nv_bfloat16 *ds=bf16_device(scale);
    CUDA_OK(cudaMalloc(&out,expected.size()*sizeof(float)));
    g4_router_input_kernel<256><<<rows,256>>>(out,di,ds,rows,width,1e-6f,root); CUDA_OK(cudaGetLastError());
    check_ulps("router_input",expected,out,2.0f);
    CUDA_OK(cudaFree(di)); CUDA_OK(cudaFree(ds)); CUDA_OK(cudaFree(out));
}

static void rope_test() {
    const unsigned heads=G4_REF_HEADS;
    const uint32_t positions[2]={0u, 4093u};
    std::vector<uint32_t> pos(positions,positions+2);
    /* Sliding: head_dim 256, theta 1e4, 128 pairs. */
    {
        const uint32_t dim=G4_REF_HEAD_DIM;
        std::vector<float> inv(dim/2),in(2u*heads*dim),expected;
        const uint32_t pairs=g4_ref_rope_inv(inv.data(),dim,0,10000.0);
        for(unsigned i=0;i<in.size();i++) { in[i]=value(i)*1.3f; }
        expected=in;
        g4_ref_rope(expected.data(),2u,heads,dim,pos.data(),inv.data(),pairs);
        uint32_t *dp=device(pos); float *dv=device(inv),*out;
        CUDA_OK(cudaMalloc(&out,in.size()*sizeof(float)));
        CUDA_OK(cudaMemcpy(out,in.data(),in.size()*sizeof(float),cudaMemcpyHostToDevice));
        g4_rope_kernel<<<(2u*heads+127)/128,128>>>(out,dp,dv,2u,heads,dim,pairs); CUDA_OK(cudaGetLastError());
        check_ulps("rope_sliding_256",expected,out,0.0f);
        CUDA_OK(cudaFree(dp)); CUDA_OK(cudaFree(dv)); CUDA_OK(cudaFree(out));
    }
    /* Full: head_dim 512, theta 1e6, proportional 0.25 (64 of 256 pairs). */
    {
        const uint32_t dim=G4_REF_GLOBAL_HEAD_DIM;
        std::vector<float> inv(dim/2),in(2u*heads*dim),expected;
        const uint32_t pairs=g4_ref_rope_inv(inv.data(),dim,1,1000000.0);
        for(unsigned i=0;i<in.size();i++) { in[i]=value(i+53)*1.9f; }
        expected=in;
        g4_ref_rope(expected.data(),2u,heads,dim,pos.data(),inv.data(),pairs);
        uint32_t *dp=device(pos); float *dv=device(inv),*out;
        CUDA_OK(cudaMalloc(&out,in.size()*sizeof(float)));
        CUDA_OK(cudaMemcpy(out,in.data(),in.size()*sizeof(float),cudaMemcpyHostToDevice));
        g4_rope_kernel<<<(2u*heads+127)/128,128>>>(out,dp,dv,2u,heads,dim,pairs); CUDA_OK(cudaGetLastError());
        check_ulps("rope_full_proportional_512",expected,out,0.0f);
        CUDA_OK(cudaFree(dp)); CUDA_OK(cudaFree(dv)); CUDA_OK(cudaFree(out));
    }
}

static void gelu_test() {
    const unsigned count=G4_REF_FF*G4_REF_USED;
    std::vector<float> gate(count),up(count),expected(count);
    for(unsigned i=0;i<count;i++) {
        gate[i]=value(i)*4.2f; up[i]=value(i+91)*2.4f;
        expected[i]=g4_ref_gate_up(gate[i],up[i]);
    }
    float *dg=device(gate),*du=device(up),*out;
    CUDA_OK(cudaMalloc(&out,count*sizeof(float)));
    g4_gelu_kernel<<<(count+255)/256,256>>>(out,dg,du,count); CUDA_OK(cudaGetLastError());
    check_abs("gelu_tanh_product",expected,out,1e-4f);
    CUDA_OK(cudaFree(dg)); CUDA_OK(cudaFree(du)); CUDA_OK(cudaFree(out));
}

static void embed_test() {
    const unsigned count=G4_REF_EMBED;
    std::vector<float> in(count),expected(count);
    for(unsigned i=0;i<count;i++) { in[i]=value(i)*0.7f; expected[i]=g4_ref_embed(in[i]); }
    const float factor=(float)G4_REF_EMBED_SCALE_BF16;
    std::printf("{\"primitive\":\"embed_scale\",\"constant\":%.9g,\"exact_root\":%.12g,"
        "\"bf16_of_root\":%.12g}\n",factor,std::sqrt((double)G4_REF_EMBED),
        (double)g4_ref_bf16((float)std::sqrt((double)G4_REF_EMBED)));
    float *di=device(in),*out;
    CUDA_OK(cudaMalloc(&out,count*sizeof(float)));
    g4_embed_kernel<<<(count+255)/256,256>>>(out,di,count); CUDA_OK(cudaGetLastError());
    check_ulps("embed_scale_rows",expected,out,0.0f);
    CUDA_OK(cudaFree(di)); CUDA_OK(cudaFree(out));
}

static void softcap_test() {
    const unsigned count=4096;
    std::vector<float> in(count),expected(count);
    for(unsigned i=0;i<count;i++) {
        in[i]=-60.0f+(float)i*0.03f;
        expected[i]=g4_ref_softcap(in[i]);
    }
    float *d=device(in);
    g4_softcap_kernel<<<(count+255)/256,256>>>(d,count,(float)G4_REF_SOFTCAP); CUDA_OK(cudaGetLastError());
    check_ulps("softcap_30",expected,d,1.0f);
    CUDA_OK(cudaFree(d));
}

static void layer_scale_test() {
    const unsigned count=G4_REF_EMBED;
    const float scalar=1.390625f;   /* layer 0's measured value */
    std::vector<float> in(count),expected(count);
    for(unsigned i=0;i<count;i++) { in[i]=value(i)*1.1f; expected[i]=g4_ref_layer_scale(in[i],scalar); }
    float *d=device(in);
    g4_layer_scale_kernel<<<(count+255)/256,256>>>(d,count,scalar); CUDA_OK(cudaGetLastError());
    check_ulps("layer_scalar",expected,d,0.0f);
    CUDA_OK(cudaFree(d));
}

static void router_test() {
    const unsigned rows=3, experts=G4_REF_EXPERTS, k=G4_REF_USED;
    std::vector<float> logits(rows*experts),per_expert(experts);
    std::vector<uint32_t> want_index(rows*k);
    std::vector<float> want_weight(rows*k);
    for(unsigned i=0;i<logits.size();i++) { logits[i]=g4_ref_bf16(value(i*7+1)*2.0f); }
    for(unsigned e=0;e<experts;e++) { per_expert[e]=g4_ref_bf16(1.8f+0.05f*value(e+3)); }
    for(unsigned r=0;r<rows;r++) {
        g4_ref_router(want_index.data()+r*k,want_weight.data()+r*k,logits.data()+r*experts,
            per_expert.data(),experts,k);
    }
    float *dl=device(logits);
    __nv_bfloat16 *dp=bf16_device(per_expert);
    uint32_t *di; float *dw;
    CUDA_OK(cudaMalloc(&di,rows*k*sizeof(uint32_t)));
    CUDA_OK(cudaMalloc(&dw,rows*k*sizeof(float)));
    g4_router_kernel<false><<<rows,experts>>>(di,dw,dl,dp,rows,experts,k); CUDA_OK(cudaGetLastError());
    std::vector<uint32_t> got_index(rows*k); std::vector<float> got_weight(rows*k);
    CUDA_OK(cudaMemcpy(got_index.data(),di,got_index.size()*sizeof(uint32_t),cudaMemcpyDeviceToHost));
    CUDA_OK(cudaMemcpy(got_weight.data(),dw,got_weight.size()*sizeof(float),cudaMemcpyDeviceToHost));
    unsigned wrong_index=0; float worst=0;
    for(unsigned i=0;i<rows*k;i++) {
        wrong_index+=got_index[i]!=want_index[i];
        worst=std::max(worst,std::fabs(got_weight[i]-want_weight[i]));
    }
    std::printf("{\"primitive\":\"router_top8\",\"index_mismatches\":%u,\"max_weight_error\":%.9g}\n",
        wrong_index,worst);
    if(wrong_index||worst>2e-6f) { failures++; }
    CUDA_OK(cudaFree(dl)); CUDA_OK(cudaFree(dp)); CUDA_OK(cudaFree(di)); CUDA_OK(cudaFree(dw));
}

/* One attention configuration: build bf16 KV rows, run the warp kernel and
 * the double reference, and report the worst absolute difference. `ring` is
 * the cache's row count (slot = position % ring); 0 indexes by absolute
 * position. A ring case fills every slot the visible window reads, which is
 * what a wrapped cache holds. `keys` is how many keys the producer has
 * written (0 = first + rows): the drafter's query sits at `first` while the
 * trunk holds only `first` rows, so its slot at `first` is poisoned here and
 * must not be read (candidate_generator.py:1316-1327). */
static void attn_case(const char *name,unsigned rows,unsigned first,unsigned heads,
        unsigned kv_heads,unsigned dim,unsigned window,unsigned ring,unsigned keys=0u) {
    const unsigned capacity=ring?ring:first+rows;
    const uint64_t stride=(uint64_t)kv_heads*dim;
    std::vector<float> keys_(capacity*stride),values(capacity*stride);
    std::vector<float> query((uint64_t)rows*heads*dim),expected((uint64_t)rows*heads*dim);
    std::vector<__nv_bfloat16> key16(keys_.size()),value16(values.size());
    const unsigned filled=keys?keys:first+rows;
    const unsigned oldest=(ring&&filled>ring)?filled-ring:0u;
    for(unsigned p=oldest;p<filled;p++) {
        const uint64_t slot=ring?p%capacity:p;
        for(uint64_t j=0;j<stride;j++) {
            const uint64_t k=(uint64_t)p*stride+j;
            keys_[slot*stride+j]=g4_ref_bf16(value((unsigned)(k%65536))*1.1f);
            values[slot*stride+j]=g4_ref_bf16(value((unsigned)((k+7)%65536))*0.9f);
            key16[slot*stride+j]=__float2bfloat16_rn(keys_[slot*stride+j]);
            value16[slot*stride+j]=__float2bfloat16_rn(values[slot*stride+j]);
        }
    }
    if(keys) {
        /* Every key and value the producer has not written: a kernel that
         * reads slot `first` fails loudly instead of drifting. */
        const uint64_t slot=ring?first%capacity:first;
        for(uint64_t j=0;j<stride;j++) {
            keys_[slot*stride+j]=key16[slot*stride+j]=1.0e4f;
            values[slot*stride+j]=value16[slot*stride+j]=-1.0e4f;
        }
    }
    for(uint64_t i=0;i<query.size();i++) { query[i]=g4_ref_bf16(value((unsigned)((i+3)%65536))*1.4f); }
    g4_ref_attn(expected.data(),query.data(),keys_.data(),values.data(),rows,first,heads,kv_heads,dim,window,capacity,ring,keys);
    float *dq=device(query),*out;
    __nv_bfloat16 *dk=device(key16),*dv=device(value16);
    CUDA_OK(cudaMalloc(&out,expected.size()*sizeof(float)));
    const unsigned warps=4;
    const unsigned jobs=rows*heads;
    g4_attn_kernel<<<(jobs+warps-1)/warps,warps*32>>>(out,dq,dk,dv,rows,first,heads,kv_heads,dim,window,ring,keys,stride);
    CUDA_OK(cudaGetLastError());
    check_abs(name,expected,out,3e-5f);
    CUDA_OK(cudaFree(dq)); CUDA_OK(cudaFree(dk)); CUDA_OK(cudaFree(dv)); CUDA_OK(cudaFree(out));
}

/* The tensor-core prefill kernel, sliding geometry <256, 2>. Same inputs
 * and reference as attn_case; the gate is the Phase 8 parity tolerance
 * (Agy_phase8_review.md section 4.3), not bit identity. */
static void attn_mma_case(const char *name,unsigned rows,unsigned first,unsigned heads,
        unsigned kv_heads,unsigned dim,unsigned window,unsigned ring,unsigned keys=0u) {
    const unsigned capacity=ring?ring:first+rows;
    const uint64_t stride=(uint64_t)kv_heads*dim;
    std::vector<float> keys_(capacity*stride),values(capacity*stride);
    std::vector<float> query((uint64_t)rows*heads*dim),expected((uint64_t)rows*heads*dim);
    std::vector<__nv_bfloat16> key16(keys_.size()),value16(values.size());
    const unsigned filled=keys?keys:first+rows;
    const unsigned oldest=(ring&&filled>ring)?filled-ring:0u;
    for(unsigned p=oldest;p<filled;p++) {
        const uint64_t slot=ring?p%capacity:p;
        for(uint64_t j=0;j<stride;j++) {
            const uint64_t k=(uint64_t)p*stride+j;
            keys_[slot*stride+j]=g4_ref_bf16(value((unsigned)(k%65536))*1.1f);
            values[slot*stride+j]=g4_ref_bf16(value((unsigned)((k+7)%65536))*0.9f);
            key16[slot*stride+j]=__float2bfloat16_rn(keys_[slot*stride+j]);
            value16[slot*stride+j]=__float2bfloat16_rn(values[slot*stride+j]);
        }
    }
    if(keys) {
        const uint64_t slot=ring?first%capacity:first;
        for(uint64_t j=0;j<stride;j++) {
            keys_[slot*stride+j]=key16[slot*stride+j]=1.0e4f;
            values[slot*stride+j]=value16[slot*stride+j]=-1.0e4f;
        }
    }
    for(uint64_t i=0;i<query.size();i++) { query[i]=g4_ref_bf16(value((unsigned)((i+3)%65536))*1.4f); }
    g4_ref_attn(expected.data(),query.data(),keys_.data(),values.data(),rows,first,heads,kv_heads,dim,window,capacity,ring,keys);
    float *dq=device(query),*out;
    __nv_bfloat16 *dk=device(key16),*dv=device(value16);
    CUDA_OK(cudaMalloc(&out,expected.size()*sizeof(float)));
    const unsigned rtile=64;
    const unsigned shared=2u*32u*dim*sizeof(__nv_bfloat16);
    g4_attn_mma_slide_kernel<<<dim3(kv_heads,(rows+rtile-1)/rtile),256,shared>>>(
        out,dq,dk,dv,rows,first,heads,kv_heads,window,ring,keys,stride);
    CUDA_OK(cudaGetLastError());
    check_abs(name,expected,out,2.5e-3f);
    CUDA_OK(cudaFree(dq)); CUDA_OK(cudaFree(dk)); CUDA_OK(cudaFree(dv)); CUDA_OK(cudaFree(out));
}

/* The tensor-core prefill kernel, full geometry <512, 8>. The two halves
 * of each M group exchange partial scores; the gate is the same parity
 * tolerance as the sliding case. */
static void attn_mma_full_case(const char *name,unsigned rows,unsigned first,unsigned heads,
        unsigned kv_heads,unsigned dim,unsigned ring,unsigned keys=0u) {
    const unsigned capacity=ring?ring:first+rows;
    const uint64_t stride=(uint64_t)kv_heads*dim;
    std::vector<float> keys_(capacity*stride),values(capacity*stride);
    std::vector<float> query((uint64_t)rows*heads*dim),expected((uint64_t)rows*heads*dim);
    std::vector<__nv_bfloat16> key16(keys_.size()),value16(values.size());
    const unsigned filled=keys?keys:first+rows;
    const unsigned oldest=(ring&&filled>ring)?filled-ring:0u;
    for(unsigned p=oldest;p<filled;p++) {
        const uint64_t slot=ring?p%capacity:p;
        for(uint64_t j=0;j<stride;j++) {
            const uint64_t k=(uint64_t)p*stride+j;
            keys_[slot*stride+j]=g4_ref_bf16(value((unsigned)(k%65536))*1.1f);
            values[slot*stride+j]=g4_ref_bf16(value((unsigned)((k+7)%65536))*0.9f);
            key16[slot*stride+j]=__float2bfloat16_rn(keys_[slot*stride+j]);
            value16[slot*stride+j]=__float2bfloat16_rn(values[slot*stride+j]);
        }
    }
    if(keys) {
        const uint64_t slot=ring?first%capacity:first;
        for(uint64_t j=0;j<stride;j++) {
            keys_[slot*stride+j]=key16[slot*stride+j]=1.0e4f;
            values[slot*stride+j]=value16[slot*stride+j]=-1.0e4f;
        }
    }
    for(uint64_t i=0;i<query.size();i++) { query[i]=g4_ref_bf16(value((unsigned)((i+3)%65536))*1.4f); }
    g4_ref_attn(expected.data(),query.data(),keys_.data(),values.data(),rows,first,heads,kv_heads,dim,0,capacity,ring,keys);
    float *dq=device(query),*out;
    __nv_bfloat16 *dk=device(key16),*dv=device(value16);
    CUDA_OK(cudaMalloc(&out,expected.size()*sizeof(float)));
    const unsigned rtile=8;
    const unsigned shared=32u*dim*sizeof(__nv_bfloat16)+4u*2u*16u*32u*sizeof(float);
    g4_attn_mma_full_kernel<false><<<dim3(kv_heads,(rows+rtile-1)/rtile),256,shared>>>(
        out,dq,dk,dv,nullptr,rows,first,heads,kv_heads,0u,ring,keys,stride,0u);
    CUDA_OK(cudaGetLastError());
    check_abs(name,expected,out,2.5e-3f);
    CUDA_OK(cudaFree(dq)); CUDA_OK(cudaFree(dk)); CUDA_OK(cudaFree(dv)); CUDA_OK(cudaFree(out));
}

/* Split-K decode attention: the fast kernel's per-key arithmetic over
 * `splits` key ranges plus the combine. Same reference as attn_case; the
 * gate is a tolerance because the softmax combine order differs. The split
 * count is the host's: ceil(visible keys / span), capped at G4_SPLIT_MAX
 * (ds4_gemma4_gpu.cuh, g4_split_span). */
static void attn_split_case(const char *name,unsigned first,unsigned heads,
        unsigned kv_heads,unsigned dim,unsigned window,unsigned ring,unsigned keys=0u,
        unsigned span=512u) {
    const unsigned rows=1;
    const unsigned capacity=ring?ring:first+rows;
    const uint64_t stride=(uint64_t)kv_heads*dim;
    std::vector<float> keys_(capacity*stride),values(capacity*stride);
    std::vector<float> query((uint64_t)rows*heads*dim),expected((uint64_t)rows*heads*dim);
    std::vector<__nv_bfloat16> key16(keys_.size()),value16(values.size());
    const unsigned filled=keys?keys:first+rows;
    const unsigned oldest=(ring&&filled>ring)?filled-ring:0u;
    for(unsigned p=oldest;p<filled;p++) {
        const uint64_t slot=ring?p%capacity:p;
        for(uint64_t j=0;j<stride;j++) {
            const uint64_t k=(uint64_t)p*stride+j;
            keys_[slot*stride+j]=g4_ref_bf16(value((unsigned)(k%65536))*1.1f);
            values[slot*stride+j]=g4_ref_bf16(value((unsigned)((k+7)%65536))*0.9f);
            key16[slot*stride+j]=__float2bfloat16_rn(keys_[slot*stride+j]);
            value16[slot*stride+j]=__float2bfloat16_rn(values[slot*stride+j]);
        }
    }
    if(keys) {
        const uint64_t slot=ring?first%capacity:first;
        for(uint64_t j=0;j<stride;j++) {
            keys_[slot*stride+j]=key16[slot*stride+j]=1.0e4f;
            values[slot*stride+j]=value16[slot*stride+j]=-1.0e4f;
        }
    }
    for(uint64_t i=0;i<query.size();i++) { query[i]=g4_ref_bf16(value((unsigned)((i+3)%65536))*1.4f); }
    g4_ref_attn(expected.data(),query.data(),keys_.data(),values.data(),rows,first,heads,kv_heads,dim,window,capacity,ring,keys);
    float *dq=device(query),*out;
    __nv_bfloat16 *dk=device(key16),*dv=device(value16);
    CUDA_OK(cudaMalloc(&out,expected.size()*sizeof(float)));
    /* The host's count (g4_split_span + ds4_gpu_g4_attn_split): the same
     * begin/last the kernel derives, then ceil(visible / span). */
    const unsigned begin=window&&first+1u>window?first+1u-window:0u;
    const unsigned last=keys&&keys<=first?keys-1u:first;
    unsigned splits=begin<=last?(last-begin+span)/span:1u;
    if(splits<1u) { splits=1u; }
    float *dpartial,*dstats;
    CUDA_OK(cudaMalloc(&dpartial,(size_t)splits*heads*dim*sizeof(float)));
    CUDA_OK(cudaMalloc(&dstats,(size_t)splits*heads*2*sizeof(float)));
    const unsigned group=heads/kv_heads;
    const unsigned shared=2u*G4_SPLIT_TILE*dim*sizeof(__nv_bfloat16);
    g4_attn_split_kernel<G4_SPLIT_TILE><<<dim3(kv_heads,splits),group*32u,shared>>>(
        dpartial,dstats,dq,dk,dv,first,heads,kv_heads,dim,window,ring,keys,stride,splits);
    CUDA_OK(cudaGetLastError());
    g4_attn_split_combine_kernel<<<heads,256,splits*sizeof(float)>>>(out,dpartial,dstats,heads,dim,splits);
    CUDA_OK(cudaGetLastError());
    check_abs(name,expected,out,1.0e-4f);
    CUDA_OK(cudaFree(dq)); CUDA_OK(cudaFree(dk)); CUDA_OK(cudaFree(dv)); CUDA_OK(cudaFree(out));
    CUDA_OK(cudaFree(dpartial)); CUDA_OK(cudaFree(dstats));
}

/* Full-layer split-K decode on tensor cores (cuda/gemma4_attn_split_mma.cuh):
 * the same split count, scratch layout and combine as attn_split_case, so
 * the only difference is the per-split kernel. P is rounded to bf16 before
 * P V, which makes this a parity path; the gate is the prefill MMA's 2.5e-3
 * against the f32 reference. */
static void attn_split_mma_case(const char *name,unsigned first,unsigned heads,
        unsigned kv_heads,unsigned dim,unsigned ring,unsigned keys=0u,
        unsigned span=512u) {
    const unsigned rows=1;
    const unsigned capacity=ring?ring:first+rows;
    const uint64_t stride=(uint64_t)kv_heads*dim;
    std::vector<float> keys_(capacity*stride),values(capacity*stride);
    std::vector<float> query((uint64_t)rows*heads*dim),expected((uint64_t)rows*heads*dim);
    std::vector<__nv_bfloat16> key16(keys_.size()),value16(values.size());
    const unsigned filled=keys?keys:first+rows;
    const unsigned oldest=(ring&&filled>ring)?filled-ring:0u;
    for(unsigned p=oldest;p<filled;p++) {
        const uint64_t slot=ring?p%capacity:p;
        for(uint64_t j=0;j<stride;j++) {
            const uint64_t k=(uint64_t)p*stride+j;
            keys_[slot*stride+j]=g4_ref_bf16(value((unsigned)(k%65536))*1.1f);
            values[slot*stride+j]=g4_ref_bf16(value((unsigned)((k+7)%65536))*0.9f);
            key16[slot*stride+j]=__float2bfloat16_rn(keys_[slot*stride+j]);
            value16[slot*stride+j]=__float2bfloat16_rn(values[slot*stride+j]);
        }
    }
    if(keys) {
        const uint64_t slot=ring?first%capacity:first;
        for(uint64_t j=0;j<stride;j++) {
            keys_[slot*stride+j]=key16[slot*stride+j]=1.0e4f;
            values[slot*stride+j]=value16[slot*stride+j]=-1.0e4f;
        }
    }
    for(uint64_t i=0;i<query.size();i++) { query[i]=g4_ref_bf16(value((unsigned)((i+3)%65536))*1.4f); }
    g4_ref_attn(expected.data(),query.data(),keys_.data(),values.data(),rows,first,heads,kv_heads,dim,0,capacity,ring,keys);
    float *dq=device(query),*out;
    __nv_bfloat16 *dk=device(key16),*dv=device(value16);
    CUDA_OK(cudaMalloc(&out,expected.size()*sizeof(float)));
    const unsigned last=keys&&keys<=first?keys-1u:first;
    unsigned splits=(last+span)/span;
    if(splits<1u) { splits=1u; }
    float *dpartial,*dstats;
    CUDA_OK(cudaMalloc(&dpartial,(size_t)splits*heads*dim*sizeof(float)));
    CUDA_OK(cudaMalloc(&dstats,(size_t)splits*heads*2*sizeof(float)));
    const unsigned shared=(16u*dim+G4_SMMA_TK*dim+G4_SMMA_GROUP*G4_SMMA_TK)
        *sizeof(__nv_bfloat16)+2u*8u*8u*sizeof(float);
    CUDA_OK(cudaFuncSetAttribute(g4_attn_split_mma_kernel<false>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,shared));
    g4_attn_split_mma_kernel<false><<<dim3(kv_heads,splits),G4_SMMA_THREADS,shared>>>(
        dpartial,dstats,dq,dk,dv,nullptr,0u,first,heads,kv_heads,dim,0u,ring,keys,stride,splits);
    CUDA_OK(cudaGetLastError());
    g4_attn_split_combine_kernel<<<heads,256,splits*sizeof(float)>>>(out,dpartial,dstats,heads,dim,splits);
    CUDA_OK(cudaGetLastError());
    check_abs(name,expected,out,2.5e-3f);
    CUDA_OK(cudaFree(dq)); CUDA_OK(cudaFree(dk)); CUDA_OK(cudaFree(dv)); CUDA_OK(cudaFree(out));
    CUDA_OK(cudaFree(dpartial)); CUDA_OK(cudaFree(dstats));
}

/* ---- The E4M3 full-layer lane (cuda/gemma4_attn_mma.cuh, FP8 = true).
 *
 * The cache rows are built with the device encoder's own arithmetic
 * (ds4_gemma4_gpu.cuh:102-133: amax/448 rounded through f16, then the E4M3
 * round-to-nearest table search) and the reference reads the values the
 * kernel will decode, so the gate is the attention arithmetic, not the
 * quantizer: K and V reach the reference exactly as g4_fp8_decode8 produces
 * them (the exact E4M3 value times the f16 row scale, rounded to f16). */

static float e4m3_value_host(unsigned code) {
    const unsigned s=(code>>7)&1u,e=(code>>3)&15u,m=code&7u;
    float v;
    if(e==0u) { v=(float)m*(1.0f/8.0f)*0.015625f; }
    else { v=(1.0f+(float)m/8.0f)*std::ldexp(1.0f,(int)e-7); }
    return s?-v:v;
}

static unsigned char e4m3_encode_host(float x) {
    if(!std::isfinite(x)||x==0.0f) { return 0u; }
    const unsigned sign=x<0.0f?0x80u:0u;
    const float ax=std::fmin(std::fabs(x),448.0f);
    int lo=0,hi=126;
    while(lo<hi) {
        const int mid=(lo+hi+1)>>1;
        if(e4m3_value_host((unsigned)mid)<=ax) { lo=mid; } else { hi=mid-1; }
    }
    int best=lo;
    if(best<126) {
        const float bd=std::fabs(ax-e4m3_value_host((unsigned)best));
        const float nd=std::fabs(ax-e4m3_value_host((unsigned)(best+1)));
        if(nd<bd||(nd==bd&&(((best+1)&1)==0)&&((best&1)!=0))) { best++; }
    }
    return (unsigned char)(sign|(unsigned)best);
}

static float half_round_host(float x) { return __half2float(__float2half(x)); }

static void build_fp8_rows(unsigned capacity,unsigned first,unsigned rows,
        unsigned kv_heads,unsigned dim,unsigned ring,unsigned keys,
        std::vector<uint8_t> &bytes,std::vector<float> &kf,std::vector<float> &vf) {
    const uint64_t kv_dim=(uint64_t)kv_heads*dim;
    const uint64_t row_bytes=2u*kv_dim+2u*kv_heads*sizeof(uint16_t);
    bytes.assign((size_t)capacity*row_bytes,0);
    kf.assign((size_t)capacity*kv_dim,0.0f);
    vf.assign((size_t)capacity*kv_dim,0.0f);
    const unsigned filled=keys?keys:first+rows;
    const unsigned oldest=(ring&&filled>ring)?filled-ring:0u;
    auto encode=[&](unsigned p,bool poison) {
        const uint64_t slot=ring?p%capacity:p;
        uint8_t *row=&bytes[(size_t)slot*row_bytes];
        for(unsigned h=0;h<kv_heads;h++) {
            std::vector<float> kx(dim),vx(dim);
            float kmax=0.0f,vmax=0.0f;
            for(unsigned d=0;d<dim;d++) {
                if(poison) { kx[d]=1.0e4f; vx[d]=-1.0e4f; }
                else {
                    const uint64_t k=(uint64_t)p*kv_dim+(uint64_t)h*dim+d;
                    kx[d]=value((unsigned)(k%65536))*1.1f;
                    vx[d]=value((unsigned)((k+7)%65536))*0.9f;
                }
                kmax=std::fmax(kmax,std::fabs(kx[d]));
                vmax=std::fmax(vmax,std::fabs(vx[d]));
            }
            const float kscale=half_round_host(kmax/448.0f);
            const float vscale=half_round_host(vmax/448.0f);
            for(unsigned d=0;d<dim;d++) {
                const unsigned kc=e4m3_encode_host(kx[d]/kscale);
                const unsigned vc=e4m3_encode_host(vx[d]/vscale);
                row[(uint64_t)h*dim+d]=(uint8_t)kc;
                row[kv_dim+(uint64_t)h*dim+d]=(uint8_t)vc;
                kf[(size_t)slot*kv_dim+(uint64_t)h*dim+d]=half_round_host(e4m3_value_host(kc)*kscale);
                vf[(size_t)slot*kv_dim+(uint64_t)h*dim+d]=half_round_host(e4m3_value_host(vc)*vscale);
            }
            const __half hk=__float2half(kscale),hv=__float2half(vscale);
            std::memcpy(row+2u*kv_dim+h*2u,&hk,2);
            std::memcpy(row+2u*kv_dim+(kv_heads+h)*2u,&hv,2);
        }
    };
    for(unsigned p=oldest;p<filled;p++) { encode(p,false); }
    if(keys) { encode(first,true); }
}

/* The full-geometry prefill over the E4M3 rows. */
static void attn_mma_fp8_case(const char *name,unsigned rows,unsigned first,
        unsigned heads,unsigned kv_heads,unsigned dim,unsigned ring,unsigned keys=0u) {
    const unsigned capacity=ring?ring:first+rows;
    const unsigned row_bytes=2u*kv_heads*dim+2u*kv_heads*(unsigned)sizeof(uint16_t);
    std::vector<uint8_t> bytes;
    std::vector<float> kf,vf;
    build_fp8_rows(capacity,first,rows,kv_heads,dim,ring,keys,bytes,kf,vf);
    std::vector<float> query((uint64_t)rows*heads*dim),expected((uint64_t)rows*heads*dim);
    for(uint64_t i=0;i<query.size();i++) { query[i]=g4_ref_bf16(value((unsigned)((i+3)%65536))*1.4f); }
    g4_ref_attn(expected.data(),query.data(),kf.data(),vf.data(),rows,first,heads,kv_heads,dim,0,capacity,ring,keys);
    float *dq=device(query),*out;
    uint8_t *dkv=device(bytes);
    CUDA_OK(cudaMalloc(&out,expected.size()*sizeof(float)));
    const unsigned rtile=8;
    const unsigned shared=32u*dim*sizeof(__half)+4u*2u*16u*32u*sizeof(float);
    g4_attn_mma_full_kernel<true><<<dim3(kv_heads,(rows+rtile-1)/rtile),256,shared>>>(
        out,dq,nullptr,nullptr,dkv,rows,first,heads,kv_heads,0u,ring,keys,0u,row_bytes);
    CUDA_OK(cudaGetLastError());
    check_abs(name,expected,out,2.5e-3f);
    CUDA_OK(cudaFree(dq)); CUDA_OK(cudaFree(dkv)); CUDA_OK(cudaFree(out));
}

/* The split-K decode over the E4M3 rows: the same split count and combine
 * as attn_split_mma_case. */
static void attn_split_mma_fp8_case(const char *name,unsigned first,unsigned heads,
        unsigned kv_heads,unsigned dim,unsigned ring,unsigned keys=0u,unsigned span=512u) {
    const unsigned rows=1;
    const unsigned capacity=ring?ring:first+rows;
    const unsigned row_bytes=2u*kv_heads*dim+2u*kv_heads*(unsigned)sizeof(uint16_t);
    std::vector<uint8_t> bytes;
    std::vector<float> kf,vf;
    build_fp8_rows(capacity,first,rows,kv_heads,dim,ring,keys,bytes,kf,vf);
    std::vector<float> query((uint64_t)rows*heads*dim),expected((uint64_t)rows*heads*dim);
    for(uint64_t i=0;i<query.size();i++) { query[i]=g4_ref_bf16(value((unsigned)((i+3)%65536))*1.4f); }
    g4_ref_attn(expected.data(),query.data(),kf.data(),vf.data(),rows,first,heads,kv_heads,dim,0,capacity,ring,keys);
    float *dq=device(query),*out;
    uint8_t *dkv=device(bytes);
    CUDA_OK(cudaMalloc(&out,expected.size()*sizeof(float)));
    const unsigned last=keys&&keys<=first?keys-1u:first;
    unsigned splits=(last+span)/span;
    if(splits<1u) { splits=1u; }
    float *dpartial,*dstats;
    CUDA_OK(cudaMalloc(&dpartial,(size_t)splits*heads*dim*sizeof(float)));
    CUDA_OK(cudaMalloc(&dstats,(size_t)splits*heads*2*sizeof(float)));
    const unsigned shared=(16u*dim+G4_SMMA_TK*dim+G4_SMMA_GROUP*G4_SMMA_TK)
        *sizeof(__half)+2u*8u*8u*sizeof(float);
    CUDA_OK(cudaFuncSetAttribute(g4_attn_split_mma_kernel<true>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,shared));
    g4_attn_split_mma_kernel<true><<<dim3(kv_heads,splits),G4_SMMA_THREADS,shared>>>(
        dpartial,dstats,dq,nullptr,nullptr,dkv,row_bytes,first,heads,kv_heads,dim,0u,ring,
        keys,0u,splits);
    CUDA_OK(cudaGetLastError());
    g4_attn_split_combine_kernel<<<heads,256,splits*sizeof(float)>>>(out,dpartial,dstats,heads,dim,splits);
    CUDA_OK(cudaGetLastError());
    check_abs(name,expected,out,2.5e-3f);
    CUDA_OK(cudaFree(dq)); CUDA_OK(cudaFree(dkv)); CUDA_OK(cudaFree(out));
    CUDA_OK(cudaFree(dpartial)); CUDA_OK(cudaFree(dstats));
}

static void attn_test() {
    attn_case("attn_sliding_decode",1,4096,G4_REF_HEADS,G4_REF_KV_HEADS,G4_REF_HEAD_DIM,G4_REF_WINDOW,0);    attn_case("attn_sliding_prefill_window_edge",5,1020,G4_REF_HEADS,G4_REF_KV_HEADS,G4_REF_HEAD_DIM,G4_REF_WINDOW,0);
    attn_case("attn_full_decode",1,8192,G4_REF_HEADS,G4_REF_GLOBAL_KV_HEADS,G4_REF_GLOBAL_HEAD_DIM,0,0);
    attn_case("attn_full_prefill",4,100,G4_REF_HEADS,G4_REF_GLOBAL_KV_HEADS,G4_REF_GLOBAL_HEAD_DIM,0,0);
    /* Wrapped ring: a decode at 4096 in a window+cap cache reads slots
     * 3073 % 1025 .. 4096 % 1025, i.e. across the wrap. */
    attn_case("attn_sliding_ring_wrap",1,4096,G4_REF_HEADS,G4_REF_KV_HEADS,G4_REF_HEAD_DIM,
        G4_REF_WINDOW,G4_REF_WINDOW+1u);
    /* A prefill chunk of 5 rows past the window in a window+5 cache: the
     * clobbered slots are older than every visible key. */
    attn_case("attn_sliding_ring_prefill",5,1025,G4_REF_HEADS,G4_REF_KV_HEADS,G4_REF_HEAD_DIM,
        G4_REF_WINDOW,G4_REF_WINDOW+5u);
    /* The drafter's shape (candidate_generator.py:1316-1327): one query at
     * position `first`, only `first` keys written, so the query's own slot
     * is poisoned and must be skipped, while the sliding window still starts
     * at first + 1 - window. */
    attn_case("attn_shared_prefix_sliding",1,4096,G4_REF_HEADS,G4_REF_KV_HEADS,
        G4_REF_HEAD_DIM,G4_REF_WINDOW,0,4096);
    attn_case("attn_shared_prefix_sliding_ring",1,4096,G4_REF_HEADS,G4_REF_KV_HEADS,
        G4_REF_HEAD_DIM,G4_REF_WINDOW,G4_REF_WINDOW+1u,4096);
    attn_case("attn_shared_prefix_full",1,8192,G4_REF_HEADS,G4_REF_GLOBAL_KV_HEADS,
        G4_REF_GLOBAL_HEAD_DIM,0,0,8192);
    attn_mma_case("attn_mma_sliding_prefill",64,0,G4_REF_HEADS,G4_REF_KV_HEADS,
        G4_REF_HEAD_DIM,G4_REF_WINDOW,0);
    attn_mma_case("attn_mma_sliding_partial",17,0,G4_REF_HEADS,G4_REF_KV_HEADS,
        G4_REF_HEAD_DIM,G4_REF_WINDOW,0);
    attn_mma_case("attn_mma_sliding_window_edge",64,1000,G4_REF_HEADS,G4_REF_KV_HEADS,
        G4_REF_HEAD_DIM,G4_REF_WINDOW,0);
    attn_mma_case("attn_mma_sliding_ring",64,1025,G4_REF_HEADS,G4_REF_KV_HEADS,
        G4_REF_HEAD_DIM,G4_REF_WINDOW,G4_REF_WINDOW+64u);
    attn_mma_case("attn_mma_sliding_shared_prefix",64,4096,G4_REF_HEADS,G4_REF_KV_HEADS,
        G4_REF_HEAD_DIM,G4_REF_WINDOW,0,4096);
    attn_mma_full_case("attn_mma_full_16",16,0,G4_REF_HEADS,G4_REF_GLOBAL_KV_HEADS,
        G4_REF_GLOBAL_HEAD_DIM,0);
    attn_mma_full_case("attn_mma_full_prefill",64,0,G4_REF_HEADS,G4_REF_GLOBAL_KV_HEADS,
        G4_REF_GLOBAL_HEAD_DIM,0);
    attn_mma_full_case("attn_mma_full_mid",64,1000,G4_REF_HEADS,G4_REF_GLOBAL_KV_HEADS,
        G4_REF_GLOBAL_HEAD_DIM,0);
    attn_mma_full_case("attn_mma_full_shared_prefix",64,8192,G4_REF_HEADS,G4_REF_GLOBAL_KV_HEADS,
        G4_REF_GLOBAL_HEAD_DIM,0,8192);
    attn_split_case("attn_split_sliding_decode",4096,G4_REF_HEADS,G4_REF_KV_HEADS,
        G4_REF_HEAD_DIM,G4_REF_WINDOW,0);
    attn_split_case("attn_split_sliding_ring",4096,G4_REF_HEADS,G4_REF_KV_HEADS,
        G4_REF_HEAD_DIM,G4_REF_WINDOW,G4_REF_WINDOW+1u);
    attn_split_case("attn_split_full_decode",8192,G4_REF_HEADS,G4_REF_GLOBAL_KV_HEADS,
        G4_REF_GLOBAL_HEAD_DIM,0,0);
    attn_split_case("attn_split_shared_prefix",4096,G4_REF_HEADS,G4_REF_KV_HEADS,
        G4_REF_HEAD_DIM,G4_REF_WINDOW,0,4096);
    /* Finer spans (DS4_GEMMA4_ATTN_SPLIT_SPAN): the sliding window gives 8
     * and 4 splits, the 8192-key full range 64 and 32. */
    attn_split_case("attn_split_sliding_span128",4096,G4_REF_HEADS,G4_REF_KV_HEADS,
        G4_REF_HEAD_DIM,G4_REF_WINDOW,0,0u,128u);
    attn_split_case("attn_split_sliding_span256",4096,G4_REF_HEADS,G4_REF_KV_HEADS,
        G4_REF_HEAD_DIM,G4_REF_WINDOW,0,0u,256u);
    attn_split_case("attn_split_full_span128",8192,G4_REF_HEADS,G4_REF_GLOBAL_KV_HEADS,
        G4_REF_GLOBAL_HEAD_DIM,0,0,0u,128u);
    attn_split_case("attn_split_full_span256",8192,G4_REF_HEADS,G4_REF_GLOBAL_KV_HEADS,
        G4_REF_GLOBAL_HEAD_DIM,0,0,0u,256u);
    attn_split_case("attn_split_shared_prefix_span128",4096,G4_REF_HEADS,G4_REF_KV_HEADS,
        G4_REF_HEAD_DIM,G4_REF_WINDOW,0,4096,128u);
    /* Uncapped: the combine's weights are dynamic shared memory, so the
     * count is only ceil(visible / span). 128, 256 and 512 splits exercise
     * the stride-256 weight fill and the larger shared launches. */
    attn_split_case("attn_split_full_span64",8192,G4_REF_HEADS,G4_REF_GLOBAL_KV_HEADS,
        G4_REF_GLOBAL_HEAD_DIM,0,0,0u,64u);
    attn_split_case("attn_split_full_span32",8192,G4_REF_HEADS,G4_REF_GLOBAL_KV_HEADS,
        G4_REF_GLOBAL_HEAD_DIM,0,0,0u,32u);
    attn_split_case("attn_split_full_span16",8192,G4_REF_HEADS,G4_REF_GLOBAL_KV_HEADS,
        G4_REF_GLOBAL_HEAD_DIM,0,0,0u,16u);
    attn_split_case("attn_split_sliding_span16",4096,G4_REF_HEADS,G4_REF_KV_HEADS,
        G4_REF_HEAD_DIM,G4_REF_WINDOW,0,0u,16u);
    /* Full-layer split-K on tensor cores: span 512/128/96, a wrapped ring,
     * a shared prefix and the two tile edges (64 keys exactly, then 65). */
    attn_split_mma_case("attn_split_mma_full_span512",8192,G4_REF_HEADS,
        G4_REF_GLOBAL_KV_HEADS,G4_REF_GLOBAL_HEAD_DIM,0,0u,512u);
    attn_split_mma_case("attn_split_mma_full_span128",8192,G4_REF_HEADS,
        G4_REF_GLOBAL_KV_HEADS,G4_REF_GLOBAL_HEAD_DIM,0,0u,128u);
    attn_split_mma_case("attn_split_mma_full_span96",8192,G4_REF_HEADS,
        G4_REF_GLOBAL_KV_HEADS,G4_REF_GLOBAL_HEAD_DIM,0,0u,96u);
    /* The default moved to 64 on the measured rows; the finer spans keep the
     * combine's stride-256 weight fill covered on the tensor-core path. */
    attn_split_mma_case("attn_split_mma_full_span64",8192,G4_REF_HEADS,
        G4_REF_GLOBAL_KV_HEADS,G4_REF_GLOBAL_HEAD_DIM,0,0u,64u);
    attn_split_mma_case("attn_split_mma_full_span32",8192,G4_REF_HEADS,
        G4_REF_GLOBAL_KV_HEADS,G4_REF_GLOBAL_HEAD_DIM,0,0u,32u);
    attn_split_mma_case("attn_split_mma_full_span16",8192,G4_REF_HEADS,
        G4_REF_GLOBAL_KV_HEADS,G4_REF_GLOBAL_HEAD_DIM,0,0u,16u);
    attn_split_mma_case("attn_split_mma_full_ring",8192,G4_REF_HEADS,
        G4_REF_GLOBAL_KV_HEADS,G4_REF_GLOBAL_HEAD_DIM,1025,0u,128u);
    attn_split_mma_case("attn_split_mma_full_shared_prefix",8192,G4_REF_HEADS,
        G4_REF_GLOBAL_KV_HEADS,G4_REF_GLOBAL_HEAD_DIM,0,8192u,128u);
    attn_split_mma_case("attn_split_mma_single_key",0,G4_REF_HEADS,
        G4_REF_GLOBAL_KV_HEADS,G4_REF_GLOBAL_HEAD_DIM,0,0u,128u);
    attn_split_mma_case("attn_split_mma_tile_edge63",63,G4_REF_HEADS,
        G4_REF_GLOBAL_KV_HEADS,G4_REF_GLOBAL_HEAD_DIM,0,0u,128u);
    attn_split_mma_case("attn_split_mma_tile_edge64",64,G4_REF_HEADS,
        G4_REF_GLOBAL_KV_HEADS,G4_REF_GLOBAL_HEAD_DIM,0,0u,128u);
    /* The E4M3 full-layer lane (cuda/gemma4_attn_mma.cuh, FP8 = true): the
     * row counts and ring/window edges the unit gates on. */
    attn_mma_fp8_case("attn_mma_fp8_full_16",16,0,G4_REF_HEADS,
        G4_REF_GLOBAL_KV_HEADS,G4_REF_GLOBAL_HEAD_DIM,0);
    attn_mma_fp8_case("attn_mma_fp8_full_17",17,0,G4_REF_HEADS,
        G4_REF_GLOBAL_KV_HEADS,G4_REF_GLOBAL_HEAD_DIM,0);
    attn_mma_fp8_case("attn_mma_fp8_full_64",64,0,G4_REF_HEADS,
        G4_REF_GLOBAL_KV_HEADS,G4_REF_GLOBAL_HEAD_DIM,0);
    attn_mma_fp8_case("attn_mma_fp8_full_mid",64,1000,G4_REF_HEADS,
        G4_REF_GLOBAL_KV_HEADS,G4_REF_GLOBAL_HEAD_DIM,0);
    attn_mma_fp8_case("attn_mma_fp8_full_ring",64,1025,G4_REF_HEADS,
        G4_REF_GLOBAL_KV_HEADS,G4_REF_GLOBAL_HEAD_DIM,1025u+64u);
    attn_mma_fp8_case("attn_mma_fp8_full_shared_prefix",64,4096,G4_REF_HEADS,
        G4_REF_GLOBAL_KV_HEADS,G4_REF_GLOBAL_HEAD_DIM,0,4096u);
    attn_split_mma_fp8_case("attn_split_mma_fp8_span512",8192,G4_REF_HEADS,
        G4_REF_GLOBAL_KV_HEADS,G4_REF_GLOBAL_HEAD_DIM,0,0u,512u);
    attn_split_mma_fp8_case("attn_split_mma_fp8_span128",8192,G4_REF_HEADS,
        G4_REF_GLOBAL_KV_HEADS,G4_REF_GLOBAL_HEAD_DIM,0,0u,128u);
    attn_split_mma_fp8_case("attn_split_mma_fp8_span64",8192,G4_REF_HEADS,
        G4_REF_GLOBAL_KV_HEADS,G4_REF_GLOBAL_HEAD_DIM,0,0u,64u);
    attn_split_mma_fp8_case("attn_split_mma_fp8_ring",8192,G4_REF_HEADS,
        G4_REF_GLOBAL_KV_HEADS,G4_REF_GLOBAL_HEAD_DIM,1025,0u,128u);
    attn_split_mma_fp8_case("attn_split_mma_fp8_shared_prefix",8192,G4_REF_HEADS,
        G4_REF_GLOBAL_KV_HEADS,G4_REF_GLOBAL_HEAD_DIM,0,8192u,128u);
    attn_split_mma_fp8_case("attn_split_mma_fp8_single_key",0,G4_REF_HEADS,
        G4_REF_GLOBAL_KV_HEADS,G4_REF_GLOBAL_HEAD_DIM,0,0u,128u);
    attn_split_mma_fp8_case("attn_split_mma_fp8_tile_edge63",63,G4_REF_HEADS,
        G4_REF_GLOBAL_KV_HEADS,G4_REF_GLOBAL_HEAD_DIM,0,0u,128u);
    attn_split_mma_fp8_case("attn_split_mma_fp8_tile_edge64",64,G4_REF_HEADS,
        G4_REF_GLOBAL_KV_HEADS,G4_REF_GLOBAL_HEAD_DIM,0,0u,128u);
}

int main(int argc,char **argv) {
    const int ordinal=argc>1?std::atoi(argv[1]):0;
    CUDA_OK(cudaSetDevice(ordinal));
    /* The graph builds its rope tables from G4_ROT/G4_GLOBAL_ROT; keep them
     * equal to what g4_ref_rope_inv (and the reference buffers) use: every
     * pair of a sliding head, 64 of 256 pairs on a full head. */
    if (gemma4_rot(0) != G4_REF_HEAD_DIM / 2u || gemma4_rot(5) != G4_REF_ROT_PAIRS) {
        std::fprintf(stderr, "gemma4_rot: sliding %u full %u, expected %u/%u\n",
            gemma4_rot(0), gemma4_rot(5), G4_REF_HEAD_DIM / 2u, G4_REF_ROT_PAIRS);
        failures++;
    }
    rms_test(); router_input_test(); rope_test(); gelu_test(); embed_test();
    softcap_test(); layer_scale_test(); router_test(); attn_test();
    CUDA_OK(cudaDeviceSynchronize());
    std::printf("{\"complete\":%s,\"failures\":%u,\"scope\":\"Gemma-4 primitives against the pinned "
        "reference formulas; excludes the 30-layer walk and serving\"}\n",failures?"false":"true",failures);
    return failures?1:0;
}
