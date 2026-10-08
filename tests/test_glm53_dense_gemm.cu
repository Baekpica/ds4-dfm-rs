/* Causal dense attention: independent rounded-input/probability reference. */
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_profiler_api.h>
#include <cublas_v2.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cstring>
#ifdef TEST_DENSE_GEMM
#include "../cuda/glm53_dense_attn.cuh"
#else
struct ds4_gpu_tensor { void *ptr; uint64_t bytes; };
enum { GLM53_COMPACT_THREADS = 256, GLM53_COMPACT_MAX_DIM = 1024 };
static int ds4_capture_active() { return 0; }
static cudaStream_t ds4_current_stream() { return nullptr; }
static bool cuda_ok(cudaError_t s, const char *) { return s == cudaSuccess; }
static bool glm53_compact_shape(uint32_t, uint32_t, uint32_t, uint32_t) { return true; }
static bool glm53_compact_has(const ds4_gpu_tensor *t, uint64_t n, uint64_t s) { return t && t->bytes >= n*s; }
#include "../cuda/glm53_low_attn.cuh"
#endif
#define CHECK(x) do { if (!(x)) { fprintf(stderr,"dense FAIL %d: %s\n",__LINE__,#x); exit(1); } } while (0)
enum { HEADS = 64, LATENT = 512, HEAD_DIM = 256, GUARD = 32, REPEATS = 8 };
static constexpr float SENTINEL = 12345.f;
static constexpr double ROUNDED_ABS = 2e-6;
static constexpr double ORIGINAL_ABS = 3e-4;

static void run(int rows, int pos, bool profile) {
    const int keys = pos + rows;
    std::vector<__half> kv(keys * LATENT);
    std::vector<float> q((size_t)rows * HEADS * LATENT);
    for (size_t i=0; i<kv.size(); i++) { kv[i]=__float2half_rn(sinf(i*.031f)*.7f); }
    for (size_t i=0; i<q.size(); i++) { q[i]=cosf(i*.019f)*.8f; }
    std::vector<float> out(q.size()+GUARD,SENTINEL);
    float *dq, *dy; __half *dk;
    CHECK(cudaMalloc(&dq,q.size()*sizeof(float))==cudaSuccess);
    CHECK(cudaMalloc(&dy,out.size()*sizeof(float))==cudaSuccess);
    CHECK(cudaMalloc(&dk,kv.size()*sizeof(__half))==cudaSuccess);
    CHECK(cudaMemcpy(dq,q.data(),q.size()*sizeof(float),cudaMemcpyHostToDevice)==cudaSuccess);
    CHECK(cudaMemcpy(dk,kv.data(),kv.size()*sizeof(__half),cudaMemcpyHostToDevice)==cudaSuccess);
    CHECK(cudaMemcpy(dy,out.data(),out.size()*sizeof(float),cudaMemcpyHostToDevice)==cudaSuccess);
#ifdef TEST_DENSE_GEMM
    cublasHandle_t handle;
    CHECK(cublasCreate(&handle)==CUBLAS_STATUS_SUCCESS);
    void *scratch;
    CHECK(cudaMalloc(&scratch,glm53_dense_bytes(rows,keys))==cudaSuccess);
    CHECK(glm53_dense_shape(rows,keys,HEADS,LATENT,GLM53_ATTN_ALL));
    CHECK(!glm53_dense_shape(rows,2052,HEADS,LATENT,GLM53_ATTN_ALL));
    CHECK(!glm53_dense_shape(rows,keys,HEADS,LATENT,GLM53_ATTN_SELECTED));
    auto launch = [&] { CHECK(glm53_dense_run(handle,nullptr,dy,dq,dk,scratch,
        rows,pos,HEADS,LATENT,HEAD_DIM)); };
#else
    ds4_gpu_tensor y{dy,out.size()*sizeof(float)}, x{dq,q.size()*sizeof(float)}, k{dk,kv.size()*sizeof(__half)};
    auto launch = [&] { CHECK(ds4_gpu_glm53_attn_low(&y,&x,&k,nullptr,0,rows,pos,keys,HEADS,LATENT,HEAD_DIM)); };
#endif
    launch(); CHECK(cudaDeviceSynchronize()==cudaSuccess);
    if(profile) { CHECK(cudaProfilerStart()==cudaSuccess); launch(); CHECK(cudaDeviceSynchronize()==cudaSuccess); CHECK(cudaProfilerStop()==cudaSuccess); }
    cudaEvent_t start,stop; CHECK(cudaEventCreate(&start)==cudaSuccess); CHECK(cudaEventCreate(&stop)==cudaSuccess);
    for(int n=0;n<REPEATS;n++) {
        CHECK(cudaEventRecord(start)==cudaSuccess); launch(); CHECK(cudaEventRecord(stop)==cudaSuccess);
        CHECK(cudaEventSynchronize(stop)==cudaSuccess); float ms;
        CHECK(cudaEventElapsedTime(&ms,start,stop)==cudaSuccess);
        printf("rows=%d pos=%d call=%d ms=%.6f\n",rows,pos,n,ms);
    }
    CHECK(cudaMemcpy(out.data(),dy,out.size()*sizeof(float),cudaMemcpyDeviceToHost)==cudaSuccess);
    for(float v:out) { CHECK(std::isfinite(v)); }
    for(size_t i=q.size();i<out.size();i++) { CHECK(out[i]==SENTINEL); }
    double rounded=0,original=0;
    for(int t:{0,rows/2,rows-1}) { for(int h:{0,17,63}) {
        std::vector<double> a(pos+t+1),b(a.size()); double ma=-INFINITY,mb=-INFINITY;
        for(size_t r=0;r<a.size();r++) {
            double da=0,db=0;
            for(int j=0;j<LATENT;j++) {
                const float v=q[((size_t)t*HEADS+h)*LATENT+j];
                da+=(double)__half2float(__float2half_rn(v))*__half2float(kv[r*LATENT+j]);
                db+=(double)v*__half2float(kv[r*LATENT+j]);
            }
            a[r]=da/sqrt((double)HEAD_DIM); b[r]=db/sqrt((double)HEAD_DIM);
            ma=fmax(ma,a[r]);mb=fmax(mb,b[r]);
        }
        double sa=0,sb=0; for(size_t r=0;r<a.size();r++) { a[r]=exp(a[r]-ma);sa+=a[r]; b[r]=exp(b[r]-mb);sb+=b[r]; }
        for(int j:{0,127,511}) {
            double ya=0,yb=0;
            for(size_t r=0;r<a.size();r++) {
                ya+=(double)__half2float(__float2half_rn((float)(a[r]/sa)))*__half2float(kv[r*LATENT+j]);
                yb+=b[r]/sb*__half2float(kv[r*LATENT+j]);
            }
            const float got=out[((size_t)t*HEADS+h)*LATENT+j];
            rounded=fmax(rounded,fabs(got-ya)); original=fmax(original,fabs(got-yb));
        }
    }}
    printf("rows=%d pos=%d rounded_abs=%.9g original_abs=%.9g guards=%d\n",rows,pos,rounded,original,GUARD);
#ifdef TEST_DENSE_GEMM
    CHECK(rounded<=ROUNDED_ABS);
    CHECK(cudaFree(scratch)==cudaSuccess); CHECK(cublasDestroy(handle)==CUBLAS_STATUS_SUCCESS);
#endif
    CHECK(original<=ORIGINAL_ABS);
    CHECK(cudaEventDestroy(start)==cudaSuccess);CHECK(cudaEventDestroy(stop)==cudaSuccess);
    CHECK(cudaFree(dq)==cudaSuccess);CHECK(cudaFree(dy)==cudaSuccess);CHECK(cudaFree(dk)==cudaSuccess);
}
int main(int argc,char **argv) {
    if(argc>1) { run(128,1920,true); return 0; }
    run(128,0,false);run(128,1920,false);run(2048,3,false);
}
