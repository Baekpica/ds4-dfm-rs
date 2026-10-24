/* go2b_parity.c — C go2b 编码器 vs Python go2b_encode.py 数值对齐探针(长跑前机制审计)。
 * 输入: W.f32(rows×cols 行主序) X.f32(nact×cols) — 由 go2b_parity.sh 的 python 端 dump。
 * 输出: ①自检 decode(blocks)==wq 逐值 ②输出级 relL2/cos(W@x vs 重建@x) — 与 py 端同口径对表。
 * 用法: ./go2b_parity W.f32 rows cols X.f32 nact [blocks_out]
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
/* BLAS 平台分支与 zlayer 同款: Darwin=Accelerate, Linux=cblas(scipy_openblas 前缀
 * 符号靠 Makefile 的 BLAS_RENAMES 改名接上; 少了 BLAS_CFLAGS 会在链接期爆未定义符号)。 */
#if defined(__APPLE__)
#include <Accelerate/Accelerate.h>
#else
#include <cblas.h>
#endif
#include "onebit_quant.h"   /* go1b_fp16 转换(与运行时/量化器同一实现) */
void dq_matmul(const float *X,const float *W,float *out,int S,int K,int M){
    cblas_sgemm(CblasRowMajor,CblasNoTrans,CblasTrans,S,M,K,1.0f,X,K,W,K,0.0f,out,M);
}
#include "go2b_qc.h"

static float *rdf(const char*p,size_t n){
    FILE*f=fopen(p,"rb"); if(!f){perror(p);exit(1);}
    float*b=malloc(n*4);
    if(fread(b,4,n,f)!=n){fprintf(stderr,"%s 尺寸不符(需 %zu f32)\n",p,n);exit(1);}
    fclose(f); return b;
}
int main(int argc,char**argv){
    if(argc<6){fprintf(stderr,"用法: %s W.f32 rows cols X.f32 nact [blocks_out]\n",argv[0]);return 2;}
    int rows=atoi(argv[2]),cols=atoi(argv[3]),nact=atoi(argv[5]);
    float *W=rdf(argv[1],(size_t)rows*cols);
    float *X=nact>0?rdf(argv[4],(size_t)nact*cols):NULL;
    size_t nb=(size_t)rows*go2b_row_bytes(cols);
    uint8_t*blk=malloc(nb);
    float *wq=malloc((size_t)rows*cols*4);
    dq_go2b_encode(W,rows,cols,X,nact,blk,wq);
    float *dec=malloc((size_t)rows*cols*4);
    dq_go2b_bytes_dequant(blk,rows,cols,dec);
    double mx=0; for(size_t i=0;i<(size_t)rows*cols;i++){ double d=fabs((double)dec[i]-wq[i]); if(d>mx)mx=d; }
    printf("往返自检 max|decode-wq|=%.2e (%s)\n",mx,mx<1e-6?"PASS":"FAIL");
    if(X&&nact>0){
        float *Ot=malloc((size_t)nact*rows*4),*Oq=malloc((size_t)nact*rows*4);
        dq_matmul(X,W,Ot,nact,cols,rows); dq_matmul(X,wq,Oq,nact,cols,rows);
        double e2=0,a2=0,nu=0,n1=0,n2=0;
        for(size_t i=0;i<(size_t)nact*rows;i++){ double d=(double)Ot[i]-Oq[i]; e2+=d*d; a2+=(double)Ot[i]*Ot[i];
            nu+=(double)Ot[i]*Oq[i]; n1+=(double)Ot[i]*Ot[i]; n2+=(double)Oq[i]*Oq[i]; }
        printf("C-go2b 输出级 relL2=%.4f cos=%.4f (nact=%d)\n",sqrt(e2/(a2+1e-30)),nu/(sqrt(n1)*sqrt(n2)+1e-30),nact);
    }
    { double e2=0,a2=0; for(size_t i=0;i<(size_t)rows*cols;i++){ double d=(double)W[i]-wq[i]; e2+=d*d; a2+=(double)W[i]*W[i]; }
      printf("C-go2b 权重级 relL2=%.4f\n",sqrt(e2/(a2+1e-30))); }
    if(argc>6){ FILE*f=fopen(argv[6],"wb"); fwrite(blk,1,nb,f); fclose(f); printf("blocks → %s (%zu B)\n",argv[6],nb); }
    return 0;
}
