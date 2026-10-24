/* vq2q2k_probe.c — q2z 提速转档判决: VQ blob 值 → q2_K 再量化 roundtrip 误差探针。
 * 口径: relL2(q2k(vq), vq) —— 转档新增误差, 与量化期已知 relL2(vq, fp) 同尺对比。
 * 用法: ./vq2q2k_probe <dql_vq_LXX.bin> [n_experts=8]
 * 输出: 每专家 w1/w3/w2 的转档 relL2 + 汇总。CPU-only, 不碰 GPU。 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include "quants.h"

/* q2_K 解码(ds4_cuda.cu host_deq_q2k_block 同义, 84B/256el) */
static float f16f(uint16_t h){
    uint32_t s=(h>>15)&1,e=(h>>10)&31,m=h&1023,f;
    if(e==0) f=(s<<31)|0;
    else if(e==31) f=(s<<31)|0x7f800000|(m<<13);
    else f=(s<<31)|((e+112)<<23)|(m<<13);
    if(e==0&&m){ float v=m/1024.0f/16384.0f; return s? -v:v; }
    float out; memcpy(&out,&f,4); return out;
}
static void deq_q2k_block(const uint8_t *blk, float *o){
    const uint8_t *sc=blk,*qs=blk+16;
    uint16_t hd,hm; memcpy(&hd,blk+80,2); memcpy(&hm,blk+82,2);
    float d=f16f(hd),dm=f16f(hm);
    for(int j=0;j<16;j++){
        float dj=d*(sc[j]&0xF),mj=dm*(sc[j]>>4);
        for(int i=0;i<16;i++){
            int idx=j*16+i,qp=(idx/128)*32+(idx%32);
            int q=(qs[qp]>>((idx%128)/32*2))&3;
            o[idx]=dj*q-mj;
        }
    }
}

int main(int argc,char**argv){
    if(argc<2){ fprintf(stderr,"用法: %s <dql_vq_LXX.bin> [n_exp]\n",argv[0]); return 1; }
    int NE=argc>2?atoi(argv[2]):8;
    int fd=open(argv[1],O_RDONLY); if(fd<0){perror("open");return 1;}
    struct stat st; fstat(fd,&st);
    const uint8_t *map=mmap(NULL,st.st_size,PROT_READ,MAP_PRIVATE,fd,0);
    const uint64_t *vtab=(const uint64_t*)(map+16);
    ds4q_quantize_init(DS4Q_TYPE_Q2_K);
    double agg[3]={0,0,0}; int cnt[3]={0,0,0};
    const char *nm[3]={"w1","w3","w2"};
    for(int e=0;e<NE;e++){
        for(int w=0;w<3;w++){
            uint64_t off=vtab[(size_t)e*3+w];
            if(!off) continue;
            /* vq_unpack_dequant 逐字复刻(不 include vq_qc.h 免 static 依赖网): 头16B + 码本 + 行gain + 位流 */
            const uint8_t *pay=map+off;
            uint32_t mg; memcpy(&mg,pay,4);
            uint16_t d16,n16; memcpy(&d16,pay+4,2); memcpy(&n16,pay+6,2);
            uint32_t rows,cols; memcpy(&rows,pay+8,4); memcpy(&cols,pay+12,4);
            uint32_t nbit=0; while((1u<<nbit)<(uint32_t)n16) nbit++;
            const uint8_t *cb=pay+16,*gr=cb+(size_t)n16*d16*2,*ix=gr+(size_t)rows*2;
            float *W=malloc((size_t)rows*cols*4);
            for(uint32_t r=0;r<rows;r++){
                uint16_t gh; memcpy(&gh,gr+(size_t)r*2,2);
                float g=f16f(gh);
                size_t i0=(size_t)r*(cols/d16);
                for(uint32_t c=0;c<cols/d16;c++){
                    size_t bit=(i0+c)*nbit; uint32_t wq;
                    memcpy(&wq,ix+(bit>>3),4);
                    uint32_t v=(wq>>(bit&7))&((1u<<nbit)-1u);
                    const uint8_t *ce=cb+(size_t)v*d16*2;
                    for(uint32_t k=0;k<d16;k++){
                        uint16_t ch; memcpy(&ch,ce+(size_t)k*2,2);
                        W[(size_t)r*cols+c*d16+k]=g*f16f(ch);
                    }
                }
            }
            /* q2_K roundtrip */
            size_t rb=ds4q_row_size(DS4Q_TYPE_Q2_K,cols);
            uint8_t *enc=malloc((size_t)rows*rb);
            ds4q_quantize_chunk(DS4Q_TYPE_Q2_K,W,enc,0,rows,cols,NULL);
            double se=0,ss=0;
            float *tmp=malloc(cols*4);
            for(uint32_t r=0;r<rows;r++){
                for(uint32_t b=0;b<cols/256;b++){
                    deq_q2k_block(enc+(size_t)r*rb+(size_t)b*84,tmp);
                    for(int i=0;i<256;i++){
                        float x=W[(size_t)r*cols+b*256+i],y=tmp[i];
                        se+=(double)(x-y)*(x-y); ss+=(double)x*x;
                    }
                }
            }
            double rel=sqrt(se/(ss+1e-30));
            agg[w]+=rel; cnt[w]++;
            if(e<3) printf("e%03d %s [%ux%u] relL2(q2k(vq),vq)=%.4f\n",e,nm[w],rows,cols,rel);
            free(W); free(enc); free(tmp);
        }
    }
    for(int w=0;w<3;w++) if(cnt[w]) printf("★均值 %s: %.4f (%d 专家)\n",nm[w],agg[w]/cnt[w],cnt[w]);
    return 0;
}
