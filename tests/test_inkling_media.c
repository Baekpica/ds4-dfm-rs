/* Source BTHWC folding, exact-GELU and 80-codebook audio embedding gates. */
#include "ds4_gpu.h"
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum { OFFSET = 4096, BINS = 80, LEVELS = 16, HIDDEN = 4096, AUDIO_ROWS = 5,
       BF_SHIFT = 16, BF_HALF = 0x7fff, GELU_POINTS = 1025,
       MAP_BYTES = OFFSET + BINS * LEVELS * HIDDEN * sizeof(uint16_t) };
#define CHECK(expr) do { if (!(expr)) { \
    fprintf(stderr, "FAIL line %d: %s\n", __LINE__, #expr); exit(1); \
} } while (0)

static uint16_t bits(float value) {
    uint32_t u; memcpy(&u, &value, sizeof(u));
    return (u + BF_HALF + ((u >> BF_SHIFT) & 1)) >> BF_SHIFT;
}
static float bf(float value) {
    uint32_t u = (uint32_t)bits(value) << BF_SHIFT;
    memcpy(&value, &u, sizeof(value)); return value;
}
static ds4_gpu_tensor *upload(const void *data, size_t bytes) {
    ds4_gpu_tensor *out = ds4_gpu_tensor_alloc(bytes); CHECK(out);
    if (data) { CHECK(ds4_gpu_tensor_write(out, 0, data, bytes)); }
    return out;
}

static void fold_case(unsigned stage, unsigned patches) {
    /* Shapes are from pinned HMLP scale planning, independent of GPU indexing. */
    const unsigned shapes[][6] = {{2,40,40,3,1,5}, {2,8,8,128,1,2},
                                   {2,4,4,320,1,4}, {2,1,1,4800,2,1}};
    const unsigned *s = shapes[stage];
    unsigned t=s[0], h=s[1], w=s[2], c=s[3], tf=s[4], hf=s[5];
    size_t count=(size_t)patches*t*h*w*c, bytes=count*sizeof(float);
    float *x=malloc(bytes), *got=malloc(bytes); CHECK(x && got);
    for (size_t i=0; i<count; i++) { x[i]=((int)(i*37%1013)-506)/127.0f; }
    ds4_gpu_tensor *dx=upload(x,bytes), *out=upload(NULL,bytes);
    CHECK(ds4_gpu_inkling_fold(out,dx,patches,stage));
    CHECK(ds4_gpu_tensor_read(out,0,got,bytes));
    size_t dst=0;
    for (unsigned b=0;b<patches;b++) {
        for (unsigned nt=0;nt<t/tf;nt++) {
            for (unsigned nh=0;nh<h/hf;nh++) {
                for (unsigned nw=0;nw<w/hf;nw++) {
                    for (unsigned ft=0;ft<tf;ft++) {
                        for (unsigned fh=0;fh<hf;fh++) {
                            for (unsigned fw=0;fw<hf;fw++) {
                                for (unsigned ch=0;ch<c;ch++) {
                                    size_t src=(((((size_t)b*t+nt*tf+ft)*h+nh*hf+fh)*w+nw*hf+fw)*c+ch);
                                    CHECK(got[dst++]==bf(x[src]));
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    CHECK(dst==count);
    CHECK(!ds4_gpu_inkling_fold(dx,dx,patches,stage));
    CHECK(!ds4_gpu_inkling_fold(out,dx,patches+1,stage));
    CHECK(!ds4_gpu_inkling_fold(out,dx,patches,4));
    CHECK(!ds4_gpu_inkling_fold(out,dx,0,stage));
    ds4_gpu_tensor_free(dx); ds4_gpu_tensor_free(out); free(x); free(got);
    printf("fold stage=%u patches=%u source permutation exact\n",stage,patches);
}

static void gelu_case(void) {
    float x[GELU_POINTS],got[GELU_POINTS],again[GELU_POINTS];
    for (unsigned i=0;i<GELU_POINTS;i++) { x[i]=((int)i-GELU_POINTS/2)/64.0f+0.00001f; }
    ds4_gpu_tensor *dx=upload(x,sizeof(x)), *out=upload(NULL,sizeof(got));
    CHECK(ds4_gpu_inkling_gelu(out,dx,GELU_POINTS));
    CHECK(ds4_gpu_tensor_read(out,0,got,sizeof(got)));
    for (unsigned i=0;i<GELU_POINTS;i++) {
        double v=bf(x[i]), want=0.5*v*erfc(-v/sqrt(2.0));
        double ulp=ldexp(1.0,ilogb(fmax(fabs(want),0x1p-126))-7);
        CHECK(isfinite(got[i]) && got[i]==bf(got[i]));
        CHECK(fabs(got[i]-want)<=0.501*ulp+3e-7);
    }
    CHECK(ds4_gpu_inkling_gelu(dx,dx,GELU_POINTS));
    CHECK(ds4_gpu_tensor_read(dx,0,again,sizeof(again)));
    CHECK(memcmp(got,again,sizeof(got))==0);
    CHECK(!ds4_gpu_inkling_gelu(out,dx,GELU_POINTS+1));
    CHECK(!ds4_gpu_inkling_gelu(out,dx,UINT64_MAX));
    CHECK(!ds4_gpu_inkling_gelu(out,dx,0));
    ds4_gpu_tensor_free(dx); ds4_gpu_tensor_free(out);
    puts("exact-GELU BF16 boundaries and in-place output passed");
}

static float audio_weight(unsigned code, unsigned col) {
    return ((int)((code*37+col*19)%257)-128)/256.0f;
}
static void audio_case(const void *map) {
    int32_t ids[AUDIO_ROWS*BINS];
    float *got=malloc(AUDIO_ROWS*HIDDEN*sizeof(float)); CHECK(got);
    ds4_gpu_tensor *di=upload(NULL,sizeof(ids));
    ds4_gpu_tensor *out=upload(NULL,AUDIO_ROWS*HIDDEN*sizeof(float));
    struct ds4_layer_graph_key key={0}; key.n_tok=AUDIO_ROWS; key.cur_hc=di; key.q=out;
    unsigned captures=0,replays=0;
    for (unsigned step=0;step<7;step++) {
        for (unsigned i=0;i<AUDIO_ROWS*BINS;i++) { ids[i]=(i*7+step*3)%LEVELS; }
        if (step==5) { ids[2*BINS+3]=-1; }
        if (step==6) { ids[4*BINS+7]=LEVELS; }
        CHECK(ds4_gpu_tensor_write(di,0,ids,sizeof(ids)));
        int mode=ds4_cuda_layer_graph_begin_or_replay(0,&key);
        if (mode!=1) { CHECK(ds4_gpu_inkling_audio(out,di,map,MAP_BYTES,OFFSET,AUDIO_ROWS)); }
        if (mode==0) { ds4_cuda_layer_graph_end_or_commit(0); captures++; }
        replays+=mode==1;
        CHECK(ds4_gpu_tensor_read(out,0,got,AUDIO_ROWS*HIDDEN*sizeof(float)));
        for (unsigned row=0;row<AUDIO_ROWS;row++) {
            for (unsigned col=0;col<HIDDEN;col++) {
                double sum=0; int valid=1;
                for (unsigned bin=0;bin<BINS;bin++) {
                    int32_t code=ids[row*BINS+bin];
                    if (code<0 || code>=LEVELS) { valid=0; break; }
                    sum+=audio_weight(bin*LEVELS+(unsigned)code,col);
                }
                if (valid) { CHECK(got[row*HIDDEN+col]==bf((float)sum)); }
                else { CHECK(isnan(got[row*HIDDEN+col])); }
            }
        }
    }
    CHECK(captures==1 && replays==5);
    CHECK(!ds4_gpu_inkling_audio(out,di,map,OFFSET,OFFSET,AUDIO_ROWS));
    CHECK(!ds4_gpu_inkling_audio(out,di,map,MAP_BYTES,OFFSET+1,AUDIO_ROWS));
    CHECK(!ds4_gpu_inkling_audio(out,di,map,MAP_BYTES,OFFSET,AUDIO_ROWS+1));
    CHECK(!ds4_gpu_inkling_audio(out,di,map,MAP_BYTES,OFFSET,0));
    ds4_gpu_tensor_free(di); ds4_gpu_tensor_free(out); free(got);
    puts("80-codebook BF16 audio sum, invalid IDs and live capture inputs passed");
}

int main(void) {
    CHECK(ds4_gpu_init());
    void *map=NULL; CHECK(posix_memalign(&map,OFFSET,MAP_BYTES)==0);
    memset(map,0,MAP_BYTES);
    uint16_t *weights=(uint16_t *)((char *)map+OFFSET);
    for (unsigned code=0;code<BINS*LEVELS;code++) {
        for (unsigned col=0;col<HIDDEN;col++) { weights[code*HIDDEN+col]=bits(audio_weight(code,col)); }
    }
    CHECK(ds4_gpu_set_model_map(map,MAP_BYTES));
    for (unsigned stage=0;stage<4;stage++) { fold_case(stage,1); fold_case(stage,3); }
    gelu_case(); audio_case(map);
    ds4_gpu_cleanup(); free(map);
    puts("Inkling media primitive gates passed"); return 0;
}
