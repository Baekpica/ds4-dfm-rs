#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <algorithm>
#include <cmath>
#include "../cuda/iquest_primitives.cuh"

#define CUDA_OK(call) do { cudaError_t e=(call); if(e!=cudaSuccess){ std::fprintf(stderr,"%s: %s\n",#call,cudaGetErrorString(e)); std::exit(2); } } while(0)
static unsigned failures=0;

static float value(unsigned i) { return std::sin((i+1)*0.117f) + 0.3f*std::cos((i+3)*0.031f); }

template<class T> static T *device(const std::vector<T>& x) {
    T *p; CUDA_OK(cudaMalloc(&p,x.size()*sizeof(T)));
    CUDA_OK(cudaMemcpy(p,x.data(),x.size()*sizeof(T),cudaMemcpyHostToDevice));
    return p;
}

static float check(const char *name,const std::vector<float>& expected,const float *dev,float tolerance) {
    std::vector<float> actual(expected.size());
    CUDA_OK(cudaMemcpy(actual.data(),dev,actual.size()*sizeof(float),cudaMemcpyDeviceToHost));
    float maximum=0; unsigned maximum_index=0;
    for(unsigned i=0;i<actual.size();i++) {
        if(!std::isfinite(actual[i])) { std::fprintf(stderr,"%s nonfinite\n",name); std::exit(1); }
        const float delta=std::fabs(actual[i]-expected[i]);
        if(delta>maximum) { maximum=delta; maximum_index=i; }
    }
    std::printf("{\"primitive\":\"%s\",\"max_error\":%.9g,\"max_index\":%u,\"tolerance\":%.9g}\n",name,maximum,maximum_index,tolerance);
    if(maximum>tolerance) { failures++; }
    return maximum;
}

static float bf16_ulp(float value) {
    uint32_t bits;
    std::memcpy(&bits,&value,sizeof(bits));
    bits+=0x10000u;
    float next;
    std::memcpy(&next,&bits,sizeof(bits));
    return std::fabs(next-value);
}

static void check_bf16(const char *name,const std::vector<float>& expected,const float *dev) {
    std::vector<float> actual(expected.size());
    CUDA_OK(cudaMemcpy(actual.data(),dev,actual.size()*sizeof(float),cudaMemcpyDeviceToHost));
    float maximum=0,maximum_ulps=0;
    unsigned unequal=0,non_bf16=0,excess_error=0;
    for(unsigned i=0;i<actual.size();i++) {
        uint32_t bits;
        std::memcpy(&bits,&actual[i],sizeof(bits));
        non_bf16+=(bits&0xffffu)!=0;
        unequal+=actual[i]!=expected[i];
        const float delta=std::fabs(actual[i]-expected[i]);
        const float ulp=std::max(bf16_ulp(actual[i]),bf16_ulp(expected[i]));
        maximum=std::max(maximum,delta);
        maximum_ulps=std::max(maximum_ulps,ulp?delta/ulp:0.0f);
        excess_error+=delta>std::max(ulp,2e-6f);
        if(!std::isfinite(actual[i])) { failures++; }
    }
    const float agreement=1.0f-(float)unequal/actual.size();
    std::printf("{\"primitive\":\"%s\",\"max_error\":%.9g,\"max_bf16_ulps\":%.9g,\"bf16_element_agreement\":%.9g,\"non_bf16_outputs\":%u,\"fp32_cancellation_atol\":2e-6}\n",
                name,maximum,maximum_ulps,agreement,non_bf16);
    // Reduction order can cross a BF16 tie; FP32 cancellation near zero also
    // needs the original attention's absolute floating-point error allowance.
    if(non_bf16||excess_error||agreement<0.99f) { failures++; }
}

static void rms_test() {
    const unsigned rows=4,width=IQ_EMBED;
    std::vector<float> x(rows*width),w(width),expected(rows*width);
    for(unsigned i=0;i<x.size();i++) { x[i]=value(i); }
    for(unsigned i=0;i<width;i++) { w[i]=0.8f+0.1f*value(i); }
    for(unsigned r=0;r<rows;r++) { iquest_rms(expected.data()+r*width,x.data()+r*width,w.data(),width); }
    float *dx=device(x),*dw=device(w),*out; CUDA_OK(cudaMalloc(&out,x.size()*sizeof(float)));
    iquest_rms_kernel<<<rows,256>>>(out,dx,dw,width); CUDA_OK(cudaGetLastError());
    check("rms3072",expected,out,2e-6f);
    CUDA_OK(cudaFree(dx)); CUDA_OK(cudaFree(dw)); CUDA_OK(cudaFree(out));
}

static void rope_test() {
    const std::vector<unsigned> positions={0,1,4095,4096,524287};
    const unsigned count=positions.size()*IQ_HEADS*IQ_HEAD;
    std::vector<float> x(count),expected(count);
    for(unsigned i=0;i<count;i++) { x[i]=value(i); }
    unsigned *dp=device(positions);
    for(float theta : {10000.0f,1000000.0f}) {
        expected=x;
        for(unsigned row=0;row<positions.size();row++) { iquest_rope(expected.data()+row*IQ_HEADS*IQ_HEAD,IQ_HEADS,positions[row],theta); }
        float *dx=device(x);
        const unsigned pairs=positions.size()*IQ_HEADS*IQ_ROT/2;
        iquest_rope_kernel<<<(pairs+255)/256,256>>>(dx,dp,IQ_HEADS,positions.size(),theta); CUDA_OK(cudaGetLastError());
        check(theta==10000.0f?"rope_swa_512k":"rope_fa_512k",expected,dx,2e-5f);
        CUDA_OK(cudaFree(dx));
    }
    CUDA_OK(cudaFree(dp));
}

static void router_test() {
    const unsigned rows=3;
    std::vector<float> logits(rows*IQ_EXPERTS),weights(rows*IQ_USED);
    std::vector<unsigned> ids(rows*IQ_USED),actual_ids(ids.size());
    for(unsigned i=0;i<logits.size();i++) { logits[i]=value(i)*40.0f; }
    std::fill(logits.begin()+2*IQ_EXPERTS,logits.end(),-1.0f);
    for(unsigned r=0;r<rows;r++) { iquest_router(ids.data()+r*IQ_USED,weights.data()+r*IQ_USED,logits.data()+r*IQ_EXPERTS); }
    float *dl=device(logits),*dw; unsigned *di;
    CUDA_OK(cudaMalloc(&dw,weights.size()*sizeof(float))); CUDA_OK(cudaMalloc(&di,ids.size()*sizeof(unsigned)));
    iquest_router_kernel<<<rows,32>>>(di,dw,dl); CUDA_OK(cudaGetLastError());
    CUDA_OK(cudaMemcpy(actual_ids.data(),di,ids.size()*sizeof(unsigned),cudaMemcpyDeviceToHost));
    if(actual_ids!=ids) { std::fprintf(stderr,"router ids mismatch\n"); std::exit(1); }
    check("top8_softmax",weights,dw,2e-7f);
    CUDA_OK(cudaFree(dl)); CUDA_OK(cudaFree(dw)); CUDA_OK(cudaFree(di));
}

static void attn_test(unsigned position,unsigned capacity,unsigned window) {
    const unsigned begin=window && position+1>window?position+1-window:0;
    std::vector<iquest_q8> cache(capacity*IQ_Q8_ROW_BLOCKS),actual_cache(cache.size());
    std::vector<float> key(IQ_KV_HEADS*IQ_HEAD),val(key.size()),query(IQ_HEADS*IQ_HEAD),sink(key.size()),expected(query.size());
    for(unsigned d=0;d<query.size();d++) { query[d]=value(d+71); }
    for(unsigned d=0;d<sink.size();d++) { sink[d]=value(d+37); }
    iquest_q8 *dc=device(cache);
    float *dk,*dv,*dq=device(query),*ds=device(sink),*out;
    unsigned *dp;
    CUDA_OK(cudaMalloc(&dk,key.size()*sizeof(float))); CUDA_OK(cudaMalloc(&dv,val.size()*sizeof(float)));
    CUDA_OK(cudaMalloc(&out,query.size()*sizeof(float))); CUDA_OK(cudaMalloc(&dp,sizeof(unsigned)));
    for(unsigned token=begin;token<=position;token++) {
        for(unsigned d=0;d<key.size();d++) { key[d]=value(d+token*3); val[d]=value(d+token*7); }
        iquest_store(cache.data(),key.data(),val.data(),token,capacity);
        CUDA_OK(cudaMemcpy(dk,key.data(),key.size()*sizeof(float),cudaMemcpyHostToDevice));
        CUDA_OK(cudaMemcpy(dv,val.data(),val.size()*sizeof(float),cudaMemcpyHostToDevice));
        CUDA_OK(cudaMemcpy(dp,&token,sizeof(token),cudaMemcpyHostToDevice));
        iquest_kv_kernel<<<IQ_Q8_ROW_BLOCKS,32>>>(dc,dk,dv,dp,1,capacity); CUDA_OK(cudaGetLastError());
    }
    CUDA_OK(cudaMemcpy(actual_cache.data(),dc,cache.size()*sizeof(iquest_q8),cudaMemcpyDeviceToHost));
    if(std::memcmp(actual_cache.data(),cache.data(),cache.size()*sizeof(iquest_q8))) { std::fprintf(stderr,"Q8 cache bytes mismatch\n"); std::exit(1); }
    iquest_attn(expected.data(),query.data(),cache.data(),sink.data(),position,capacity,window);
    iquest_attn_kernel<<<dim3(1,IQ_HEADS),IQ_HEAD>>>(out,dq,dc,ds,dp,capacity,window); CUDA_OK(cudaGetLastError());
    check_bf16(window?"q8_swa_sink":"q8_full_sink",expected,out);
    CUDA_OK(cudaFree(dc)); CUDA_OK(cudaFree(dk)); CUDA_OK(cudaFree(dv)); CUDA_OK(cudaFree(dq));
    CUDA_OK(cudaFree(ds)); CUDA_OK(cudaFree(out)); CUDA_OK(cudaFree(dp));
}

static void add_test() {
    std::vector<float> raw(IQ_EMBED),branch(IQ_EMBED),expected(IQ_EMBED);
    for(unsigned i=0;i<IQ_EMBED;i++) { raw[i]=value(i); branch[i]=value(i+77); }
    float *dr=device(raw),*db=device(branch),*out;
    CUDA_OK(cudaMalloc(&out,raw.size()*sizeof(float)));
    for(float scale:{1.0f,0.53881590608f}) {
        for(unsigned i=0;i<IQ_EMBED;i++) { expected[i]=raw[i]+scale*branch[i]; }
        iquest_add_kernel<<<(IQ_EMBED+255)/256,256>>>(out,dr,db,IQ_EMBED,scale); CUDA_OK(cudaGetLastError());
        check("scaled_residual",expected,out,2e-7f);
    }
    CUDA_OK(cudaFree(dr)); CUDA_OK(cudaFree(db)); CUDA_OK(cudaFree(out));
}

static float bf16(float x) {
    uint32_t bits;
    std::memcpy(&bits, &x, sizeof(bits));
    bits = (bits + 0x7fffu + ((bits >> 16) & 1u)) & 0xffff0000u;
    std::memcpy(&x, &bits, sizeof(bits));
    return x;
}

static void moe_sum_bf16_test() {
    const unsigned rows=3;
    std::vector<float> down(rows*IQ_USED*IQ_EMBED),weights(rows*IQ_USED),expected(rows*IQ_EMBED);
    for(unsigned r=0;r<rows;r++) {
        float total=0;
        for(unsigned k=0;k<IQ_USED;k++) { weights[r*IQ_USED+k]=0.11f+(k+1)*0.037f; total+=weights[r*IQ_USED+k]; }
        for(unsigned k=0;k<IQ_USED;k++) { weights[r*IQ_USED+k]/=total; }
        for(unsigned d=0;d<IQ_EMBED;d++) {
            float sum=0;
            for(unsigned k=0;k<IQ_USED;k++) {
                const unsigned index=(r*IQ_USED+k)*IQ_EMBED+d;
                down[index]=value(index)*2.713f+0.000137f;
                sum+=bf16(down[index])*weights[r*IQ_USED+k];
            }
            expected[r*IQ_EMBED+d]=bf16(sum);
        }
    }
    float *dd=device(down),*dw=device(weights),*out;
    CUDA_OK(cudaMalloc(&out,expected.size()*sizeof(float)));
    iquest_sum_kernel<<<(expected.size()+255)/256,256>>>(out,dd,dw,rows); CUDA_OK(cudaGetLastError());
    check("moe_down_bf16_then_fp32_weighted_sum_bf16",expected,out,1e-7f);
    CUDA_OK(cudaFree(dd)); CUDA_OK(cudaFree(dw)); CUDA_OK(cudaFree(out));
}

static void swiglu_bf16_test() {
    const unsigned count=IQ_USED*IQ_FF;
    std::vector<float> gate(count),up(count),expected(count);
    for(unsigned i=0;i<count;i++) {
        gate[i]=value(i)*5.183f+0.00037f; up[i]=value(i+107)*2.619f;
        const float g=bf16(gate[i]),u=bf16(up[i]);
        expected[i]=bf16((g/(1.0f+std::exp(-g)))*u);
    }
    float *dg=device(gate),*du=device(up),*out;
    CUDA_OK(cudaMalloc(&out,count*sizeof(float)));
    iquest_swiglu_kernel<<<(count+255)/256,256>>>(out,dg,du,count); CUDA_OK(cudaGetLastError());
    check("swiglu_bf16_inputs_fp32_activation_single_bf16_output",expected,out,1e-7f);
    CUDA_OK(cudaFree(dg)); CUDA_OK(cudaFree(du)); CUDA_OK(cudaFree(out));
}

int main(int argc,char **argv) {
    const int ordinal=argc>1?std::atoi(argv[1]):0;
    CUDA_OK(cudaSetDevice(ordinal));
    rms_test(); rope_test(); router_test();
    attn_test(12,16,0); attn_test(31,7,4); attn_test(524287,7,7); add_test(); moe_sum_bf16_test(); swiglu_bf16_test();
    CUDA_OK(cudaDeviceSynchronize());
    std::printf("{\"complete\":%s,\"failures\":%u,\"scope\":\"CPU/CUDA primitives at real head dimensions; excludes model serving\"}\n",failures?"false":"true",failures);
    return failures?1:0;
}
