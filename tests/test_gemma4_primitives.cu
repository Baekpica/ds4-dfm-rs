#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <algorithm>
#include <cmath>
#include "../cuda/gemma4_primitives.cuh"
#include "../cuda/gemma4_attn.cuh"
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
    g4_router_kernel<<<rows,experts>>>(di,dw,dl,dp,rows,experts,k); CUDA_OK(cudaGetLastError());
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

static void attn_test() {
    attn_case("attn_sliding_decode",1,4096,G4_REF_HEADS,G4_REF_KV_HEADS,G4_REF_HEAD_DIM,G4_REF_WINDOW,0);
    attn_case("attn_sliding_prefill_window_edge",5,1020,G4_REF_HEADS,G4_REF_KV_HEADS,G4_REF_HEAD_DIM,G4_REF_WINDOW,0);
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
