/* Production codec with host-memory or real CUDA transfer adapters.
 * Neither mode establishes full-model ownership or next-token parity. */
#define _POSIX_C_SOURCE 200809L
#include <unistd.h>
#include <assert.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <limits.h>
#ifdef MIMO2_TEST_CUDA
#include <cuda_runtime_api.h>
#endif
enum { MIMO2_LAYERS = 48, MIMO2_DRAFT_LAYERS = 3, DS4_N_VOCAB = 152576,
       DS4_SESSION_PAYLOAD_U32_FIELDS = 13, DS4_SESSION_IO_CHUNK = 65536,
       DS4_SESSION_PAYLOAD_MAGIC = 0x44533450, DS4_SESSION_PAYLOAD_VERSION = 3 };
typedef struct { uint64_t raw_bytes, scratch_bytes, total_bytes; unsigned raw_cap, prefill_cap; } ds4_context_memory;
#include "../ds4_mimo2_plan.h"
typedef struct { uint8_t *p; uint64_t size; } ds4_gpu_tensor;
typedef struct {
    ds4_gpu_tensor *kv[48], *logits;
    unsigned kv_cap[48], context, cap, position;
    unsigned draft_pos[MIMO2_DRAFT_LAYERS], draft_saved[MIMO2_DRAFT_LAYERS];
    unsigned trial_n, verify_rows, span_n;
    uint64_t media_tag, checkpoint_tag;
    bool failed, hidden_valid;
    unsigned *dflash_pos;
    unsigned dflash_saved_n;
} ds4_mimo2_graph;
static bool fail_sync, fail_write;
static int ds4_gpu_synchronize(void) {
#ifdef MIMO2_TEST_CUDA
    return !fail_sync && cudaDeviceSynchronize() == cudaSuccess;
#else
    return !fail_sync;
#endif
}
static void payload_set_err(char *err, size_t n, const char *msg) { if (n) { snprintf(err,n,"%s",msg); } }
static int ds4_gpu_tensor_write(ds4_gpu_tensor *t, uint64_t off, const void *p, uint64_t n) {
    if (fail_write || off > t->size || n > t->size-off) { return 0; }
    #ifdef MIMO2_TEST_CUDA
    return cudaMemcpy(t->p+off,p,n,cudaMemcpyHostToDevice) == cudaSuccess;
#else
    memcpy(t->p+off,p,n); return 1;
#endif
}
static int tensor_read(const ds4_gpu_tensor *t,uint64_t off,void *p,uint64_t n) {
    if (off > t->size || n > t->size-off) { return 0; }
#ifdef MIMO2_TEST_CUDA
    return cudaMemcpy(p,t->p+off,n,cudaMemcpyDeviceToHost) == cudaSuccess;
#else
    memcpy(p,t->p+off,n); return 1;
#endif
}
static int payload_write_bytes(FILE *f,const void *p,uint64_t n,char *e,size_t z) {
    (void)e;(void)z;return fwrite(p,1,n,f)!=n;
}
static int payload_read_bytes(FILE *f,void *p,uint64_t n,uint64_t *left,char *e,size_t z) {
    (void)e;(void)z;if (n>*left || fread(p,1,n,f)!=n) { return 1; } *left-=n;return 0;
}
static int payload_write_u32(FILE *f,uint32_t v,char *e,size_t z) { return payload_write_bytes(f,&v,4,e,z); }
static int payload_read_u32(FILE *f,uint32_t *v,uint64_t *l,char *e,size_t z) { return payload_read_bytes(f,v,4,l,e,z); }
static int payload_write_logits(FILE *f,const float *v,uint8_t *b,char *e,size_t z) { (void)b;return payload_write_bytes(f,v,DS4_N_VOCAB*4,e,z); }
static int payload_write_tensor_span(FILE *f,const ds4_gpu_tensor *t,uint64_t o,uint64_t n,uint8_t *b,size_t c,char *e,size_t z) {
    if(o>t->size || n>t->size-o) { return 1; }
    while(n) {
        const size_t span=n>c?c:(size_t)n;
        if(!tensor_read(t,o,b,span) || payload_write_bytes(f,b,span,e,z)) { return 1; }
        o+=span;n-=span;
    }
    return 0;
}
static int payload_read_tensor_span(FILE *f,ds4_gpu_tensor *t,uint64_t o,uint64_t n,uint8_t *b,size_t c,uint64_t *l,char *e,size_t z) {
    if(o>t->size || n>t->size-o) { return 1; }
    while(n) {
        const size_t span=n>c?c:(size_t)n;
        if(payload_read_bytes(f,b,span,l,e,z) || !ds4_gpu_tensor_write(t,o,b,span)) { return 1; }
        o+=span;n-=span;
    }
    return 0;
}
static unsigned allocation_failure;
static void *payload_alloc(size_t bytes) {
    if (allocation_failure && --allocation_failure == 0) { return NULL; }
    return malloc(bytes);
}
#define malloc payload_alloc
#include "../ds4_mimo2_payload.inc"
#undef malloc
static ds4_gpu_tensor *tensor(uint64_t n) {
    ds4_gpu_tensor *t=malloc(sizeof(*t));assert(t);t->size=n;
#ifdef MIMO2_TEST_CUDA
    assert(cudaMalloc((void **)&t->p,n)==cudaSuccess);
    assert(cudaMemset(t->p,0xa5,n)==cudaSuccess);
#else
    t->p=malloc(n);assert(t->p);memset(t->p,0xa5,n);
#endif
    return t;
}
static void init(ds4_mimo2_graph *g,unsigned ctx,unsigned cap,unsigned n) {
    memset(g,0,sizeof(*g));g->context=ctx;g->cap=cap;g->position=n;
    g->logits=tensor(DS4_N_VOCAB*4);
    for(unsigned il=0;il<48;il++) {
        g->kv_cap[il]=mimo2_kv_capacity(il,ctx,cap);
        g->kv[il]=tensor((uint64_t)g->kv_cap[il]*(mimo2_is_full(il)?4:8)*320*2);
    }
}
static void destroy(ds4_mimo2_graph *g) {
    for(unsigned i=0;i<49;i++) {
        ds4_gpu_tensor *t=i<48?g->kv[i]:g->logits;
#ifdef MIMO2_TEST_CUDA
        assert(cudaFree(t->p)==cudaSuccess);
#else
        free(t->p);
#endif
        free(t);
    }
}
static int restore(ds4_mimo2_graph *g,FILE *f,uint64_t size,int **tokens,float *logits) {
    rewind(f);uint32_t h[13];char err[128]={0};uint64_t left=size;
    for(unsigned i=0;i<13;i++) { if(payload_read_u32(f,&h[i],&left,err,sizeof(err))) { return 1; } }
    return mimo2_payload_restore(g,f,&left,h,tokens,logits,err,sizeof(err));
}
static void trial(unsigned n,unsigned source_cap,unsigned dest_cap) {
    ds4_mimo2_graph a,b;init(&a,600,source_cap,n);init(&b,700,dest_cap,0);
    int *tokens=malloc(n*sizeof(int)),*got=NULL;float *logits=malloc(DS4_N_VOCAB*4),*out=malloc(DS4_N_VOCAB*4);assert(tokens&&logits&&out);
    for(unsigned i=0;i<n;i++) { tokens[i]=(int)(i%DS4_N_VOCAB); }
    for(unsigned i=0;i<DS4_N_VOCAB;i++) { logits[i]=(float)i/37; }
    for(unsigned il=0;il<48;il++) {
        const unsigned row=(mimo2_is_full(il)?4:8)*320*2;
        uint8_t *buffer=malloc(a.kv[il]->size);assert(buffer);memset(buffer,0xa5,a.kv[il]->size);
        for(unsigned pos=0;pos<n;pos++) {
            uint8_t *p=buffer+(uint64_t)(pos%a.kv_cap[il])*row;
            for(unsigned x=0;x<row;x++) { p[x]=(uint8_t)((pos>>((x%4)*8))^(il*7+x)); }
        }
        assert(ds4_gpu_tensor_write(a.kv[il],0,buffer,a.kv[il]->size));free(buffer);
    }
    FILE *f=tmpfile();assert(f);char err[128];assert(!mimo2_payload_save(&a,tokens,n,logits,f,err,sizeof(err)));
    const uint64_t size=(uint64_t)ftell(f);assert(size==mimo2_payload_bytes(&a,n));
    b.hidden_valid=true; b.trial_n=3; b.verify_rows=3; b.span_n=2;
    b.media_tag=9; b.checkpoint_tag=9; b.draft_pos[0]=4; b.draft_saved[1]=1;
    unsigned *ring=malloc(1024*sizeof(unsigned));assert(ring);
    for(unsigned i=0;i<1024;i++) { ring[i]=7; }
    b.dflash_pos=ring; b.dflash_saved_n=3;
    assert(!restore(&b,f,size,&got,out));assert(!b.failed && b.position==n);
    assert(!b.hidden_valid && !b.trial_n && !b.verify_rows && !b.span_n);
    assert(!b.media_tag && !b.checkpoint_tag && !b.draft_pos[0] && !b.draft_saved[1]);
    assert(!b.dflash_saved_n);
    for(unsigned i=0;i<1024;i++) { assert(ring[i]==UINT_MAX); }
    assert(!memcmp(tokens,got,n*4) && !memcmp(logits,out,DS4_N_VOCAB*4));free(got);got=NULL;
    assert(tensor_read(b.logits,0,out,DS4_N_VOCAB*4));assert(!memcmp(logits,out,DS4_N_VOCAB*4));
    for(unsigned il=0;il<48;il++) {
        const unsigned row=(mimo2_is_full(il)?4:8)*320*2;
        const unsigned first=mimo2_is_full(il)||n<128?0:n-128;
        uint8_t *buffer=malloc(b.kv[il]->size);assert(buffer);assert(tensor_read(b.kv[il],0,buffer,b.kv[il]->size));
        for(unsigned pos=first;pos<n;pos++) {
            const uint8_t *p=buffer+(uint64_t)(pos%b.kv_cap[il])*row;
            for(unsigned x=0;x<row;x++) { assert(p[x]==(uint8_t)((pos>>((x%4)*8))^(il*7+x))); }
        }
        free(buffer);
    }
    assert(restore(&b,f,size-1,&got,out));assert(b.failed && b.position==0 && !got);
    fail_write=true;assert(restore(&b,f,size,&got,out));assert(b.failed && !got);fail_write=false;
    fail_sync=true;assert(restore(&b,f,size,&got,out));assert(b.failed && !got);fail_sync=false;
    for(unsigned k=1;k<=2;k++) {
        allocation_failure=k;assert(restore(&b,f,size,&got,out));assert(b.failed && !got);
    }
    allocation_failure=0;
    assert(!restore(&b,f,size,&got,out));free(got);got=NULL;
    /* Corrupt token, logits, family tag and flags separately. */
    const long offsets[]={52,52+(long)n*4,5*4,8*4};
    const uint32_t bad[]={DS4_N_VOCAB,0x7fc00000,0,1};
    for(unsigned k=0;k<4;k++) {
        uint32_t old;fseek(f,offsets[k],SEEK_SET);assert(fread(&old,4,1,f)==1);
        fseek(f,offsets[k],SEEK_SET);assert(fwrite(&bad[k],4,1,f)==1);
        assert(restore(&b,f,size,&got,out));assert(b.failed && !got);
        fseek(f,offsets[k],SEEK_SET);assert(fwrite(&old,4,1,f)==1);
    }
    assert(!restore(&b,f,size,&got,out));free(got);got=NULL;
    assert(!fflush(f));assert(!ftruncate(fileno(f),(off_t)size-1));
    assert(restore(&b,f,size,&got,out));assert(b.failed && !got);
    fclose(f);free(tokens);free(logits);free(out);free(ring);destroy(&a);destroy(&b);
}
int main(void) {
#ifdef MIMO2_TEST_CUDA
    puts("Adapter: actual CUDA allocations and H2D/D2H transfers");
#else
    puts("Adapter: host-memory device stubs");
#endif
    const unsigned n[]={1,127,128,129,257,599,600};
    for(unsigned i=0;i<7;i++) { trial(n[i],7,32);trial(n[i],32,1); }
    puts("14 checkpoint cases passed: logical full/SWA KV, different capacities, logits, malformed headers/tokens/NaNs/length, allocation/device failures, actual truncation and recovery");
    return 0;
}
