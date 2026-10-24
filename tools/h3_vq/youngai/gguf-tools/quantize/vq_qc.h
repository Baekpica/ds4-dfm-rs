/* vq_qc.h — VQ 码本量化(v2.2 shipping 配置) 量化器侧: 编码/打包/解码/逐层门日志。
 * 配置(2026-07-25 L23 层级判决 −8.8% GO): 热(64/层) w1/w3/w2 → vq4×512 (9bit/4w=2.25bpw);
 * 冷(192/层) w1/w3 → vq8×256 (1B/8w=1.0bpw), w2 → signref 保持(码本对 1bit×w2 无效实测)。
 * 每矩阵布局: [u32 'DQVQ'][u16 dim][u16 nc][u32 rows][u32 cols]
 *             [码本 nc*dim f16][g_r rows f16][索引流: dim4→9bit×4w 8索引9字节; dim8→1B/8w]
 * 依赖: dq_matmul(fwd.c), g2_inv(go2b_qc.h), go1b_fp16 helpers(onebit_quant)。
 * 编码链与 vq_shim.c/g1c_vq_sweep.vq_gptq 同口径(质量对齐已验: C 0.5120 vs py 0.5121)。 */
#ifndef DS4QUANT_VQ_H
#define DS4QUANT_VQ_H
#include <unistd.h>   /* pwrite(Linux 上须在首个使用前有原型; 自含化 2026-08-17) */
#ifdef __APPLE__
#include <Accelerate/Accelerate.h>   /* vDSP argmin(vq_assign_q 热点, 2026-07-27) */
#else
/* Linux/Spark 移植(2026-08-17): vDSP 三件套的标量等价(热点在 sgemm, 这里量小) */
typedef size_t vDSP_Length;
static inline void vDSP_vsmul(const float *a, int sa, const float *s, float *o, int so, vDSP_Length n) {
    (void)sa; (void)so;
    for (vDSP_Length i = 0; i < n; i++) o[i] = a[i] * (*s);
}
static inline void vDSP_vadd(const float *a, int sa, const float *b, int sb, float *o, int so, vDSP_Length n) {
    (void)sa; (void)sb; (void)so;
    for (vDSP_Length i = 0; i < n; i++) o[i] = a[i] + b[i];
}
static inline void vDSP_minvi(const float *a, int sa, float *mv, vDSP_Length *mi, vDSP_Length n) {
    (void)sa;
    float m = a[0]; vDSP_Length k = 0;
    for (vDSP_Length i = 1; i < n; i++) if (a[i] < m) { m = a[i]; k = i; }
    *mv = m; *mi = k;
}
#endif

#define VQ_MAGIC 0x51565144u   /* 'DQVQ' LE */

static int vq_cold_dim(void); static int vq_cold_nc(void);
static int vq_w2_dim(void); static int vq_w2_nc(void);
/* 热档(2026-08-27 可配化): 原先三个文件八处写死 vq4x512, 于是"热数拉满"就等于全员 2.25bpw
 * = 平权, 86G 预算下分配器无处可花钱, 专家分档实验根本跑不起来。现由计划表 hotdim/hotnc
 * 决定, 缺省仍是 4/512 ⇒ 所有历史 rplan 文件行为逐字节不变。 */
static int vq_hot_dim(void); static int vq_hot_nc(void);
/* --vq 请求位(原 DS4_VQ env, 2026-08-31 禁 env 铁律清退): ds4quant_run 与
 * deepseek4-quantize 都是单 TU 且各自 include 本头 → 头内 static 即进程内共享,
 * 各自的 CLI 解析器在解析期调 setter 置位, 之后所有读点走 dq_vq_on()。 */
static int dq_vq_requested=0;
static inline void dq_vq_set(int on){ dq_vq_requested=on?1:0; }
static int dq_vq_on(void){ return dq_vq_requested; }

#include "vq_qc_bytes.h"   /* vq_idx_bytes / vq_payload_bytes: 与体积分配器共用同一份字节账 */

/* ---- 最近邻(分块 sgemm 语义, 此处用 dq_matmul 批量) ---- */
#ifdef DS4QUANT_CUDA
/* vq_gpu.cu 的 GPU 常驻核(08-18 用户令): assign 批+GPTQ 组整段 */
extern int vqg_ready(void);
extern void vqg_assign(const float*,int,const float*,int,int,int*);
extern int vqg_shm_ok(int nc,int dim);   /* GPU 可跑判定=共享内存真限额, 不是魔法数(见 vq_gpu.cu) */
extern void vqg_gptq_group(float*,const float*,const double*,int,int,int,int,int,int*,const float*);
extern void vqg_gptq_full(float*,const float*,const double*,int,int,int,int,int,int,int*,const float*);
#endif
static void vq_assign_q(const float *V,int nv,int dim,const float *C,int nc,int *idx){
#ifdef DS4QUANT_CUDA
    if(nv>=8192&&vqg_shm_ok(nc,dim)&&vqg_ready()){ vqg_assign(V,nv,C,nc,dim,idx); return; }
#endif
    const int BS=8192;
    float *c2=malloc((size_t)nc*4),*G=malloc((size_t)BS*nc*4),*vtmp=malloc((size_t)nc*4);
    for(int c=0;c<nc;c++){ double s=0; for(int d=0;d<dim;d++) s+=(double)C[c*dim+d]*C[c*dim+d]; c2[c]=(float)s; }
    for(int i0=0;i0<nv;i0+=BS){
        int b=nv-i0<BS?nv-i0:BS;
        dq_matmul(V+(size_t)i0*dim,C,G,b,dim,nc);    /* G[b,nc]=V·Cᵀ */
        /* vDSP argmin(2026-07-27 提速): v=c2-2g 批算+首现最小值索引, 与原标量扫描
         * (严格< 保首现)逐位同判 → 确定性不变。profile: 标量扫描曾 8768 采样登顶。 */
        for(int i=0;i<b;i++){ const float*g=G+(size_t)i*nc;
            float nm2=-2.0f;
            vDSP_vsmul(g,1,&nm2,vtmp,1,(vDSP_Length)nc);
            vDSP_vadd(vtmp,1,c2,1,vtmp,1,(vDSP_Length)nc);
            float mv; vDSP_Length mi;
            vDSP_minvi(vtmp,1,&mv,&mi,(vDSP_Length)nc);
            idx[i0+i]=(int)mi; }
    }
    free(c2);free(G);free(vtmp);
}

/* vqt_now: 单调时钟。名字是段计时时代的遗产, 但 ds4quant_run 的 BFLT 段账/层账全用它,
 * 是共享基元不是诊断(诊断累计 g_vqt/DS4_VQ_TIMING 已随禁 env 铁律删除)。 */
#include <time.h>
static double vqt_now(void){ struct timespec ts; clock_gettime(CLOCK_MONOTONIC,&ts); return ts.tv_sec+ts.tv_nsec*1e-9; }
/* ---- 编码: kmeans(步长子采样,8轮) + GPTQ 列组反馈 + 行乘子; 输出码本/索引/g_r + dequant ---- */
static void vq_encode_full(const float *W,int rows,int cols,int dim,int nc,
                           const float *X,int n_act,
                           float *Cb_out,int *idx_out,float *gr_out,float *Wq_opt){
    size_t N=(size_t)rows*cols; int nv=(int)(N/dim);
    int nsub=nv<200000?nv:200000, stride=nv/nsub;
    float *sub=malloc((size_t)nsub*dim*4);
    for(int i=0;i<nsub;i++) memcpy(sub+(size_t)i*dim,W+(size_t)i*stride*dim,(size_t)dim*4);
    float *C=Cb_out;
    int cst=nsub/nc; if(cst<1)cst=1;
    for(int c=0;c<nc;c++) memcpy(C+(size_t)c*dim,sub+(size_t)(c*cst)*dim,(size_t)dim*4);
    int *ai=malloc((size_t)nsub*sizeof(int));
    double *acc=malloc((size_t)nc*dim*8); long *cnt=malloc((size_t)nc*8);
    for(int it=0;it<8;it++){
        vq_assign_q(sub,nsub,dim,C,nc,ai);
        memset(acc,0,(size_t)nc*dim*8); memset(cnt,0,(size_t)nc*8);
        for(int i=0;i<nsub;i++){ int c=ai[i]; cnt[c]++;
            for(int d=0;d<dim;d++) acc[(size_t)c*dim+d]+=sub[(size_t)i*dim+d]; }
        for(int c=0;c<nc;c++) if(cnt[c]>0)
            for(int d=0;d<dim;d++) C[(size_t)c*dim+d]=(float)(acc[(size_t)c*dim+d]/cnt[c]);
    }
    free(sub);free(ai);free(acc);free(cnt);
    /* f16 往返码本(与存储一致再分配) */
    for(int i=0;i<nc*dim;i++) C[i]=go1b_fp16_to_fp32(go1b_fp32_to_fp16(C[i]));
    int grp=128;
    float *Wk=malloc((size_t)rows*grp*4),*seg=malloc((size_t)rows*dim*4);
    int *sidx=malloc((size_t)rows*sizeof(int));
    double *H=malloc((size_t)grp*grp*8),*Hi=malloc((size_t)grp*grp*8);
    float *XbT=(X&&n_act>0)?malloc((size_t)grp*n_act*4):NULL;
    /* ★通道加权选码字 v2: GPTQ 一致权 1/diag(H⁻¹)(2026-08-31 深夜)★
     * v1 用 s_j=E[x_j²] 实测退化(wt2 裸 KLD 0.46186→0.48138, Σmin 0.7855→0.7802;
     * top1 反 +0.49pp = 机制生效但方向错): H_jj 忽略"该列误差可否被后续列补偿",
     * 把可补偿的 massive 列过度保护, 牺牲不可补偿列。GPTQ 贪心的真列成本 =
     * 1/[H⁻¹]_jj(可补偿列自动降权), Hi 每组现成, 零新计算。无反馈组(求逆失败)平权。 */
    float *cw=NULL;
    if(XbT) cw=calloc((size_t)cols,4);   /* 逐组从 Hi 对角填, 0=该组平权 */
    float *Wq=Wq_opt?Wq_opt:malloc(N*4); int own=Wq_opt?0:1;
#ifdef DS4QUANT_CUDA
    /* v3 全矩阵批量(08-18): 组间独立 → 先算全部组 Hi(H cublas 逐组+CPU inv), 再一次
     * kernel 吃整矩阵(grid.y=组), per-组 sync 32→1 次/矩阵(7.8ms×24.6k=193s/层 的本体)。 */
    if(cols%grp==0&&grp%dim==0&&vqg_shm_ok(nc,dim)&&vqg_ready()){
        const int ngrp=cols/grp;
        double *Hi_all=NULL; int fb_all=0;
        if(XbT){
            Hi_all=malloc((size_t)ngrp*grp*grp*8); fb_all=1;
            for(int gi=0;gi<ngrp&&fb_all;gi++){
                const int j0=gi*grp;
                for(int j=0;j<grp;j++) for(int t=0;t<n_act;t++) XbT[(size_t)j*n_act+t]=X[(size_t)t*cols+j0+j];
                { float *Hf=malloc((size_t)grp*grp*4);
                  dq_matmul(XbT,XbT,Hf,grp,n_act,grp);
                  for(int i=0;i<grp*grp;i++) H[i]=Hf[i];
                  free(Hf); }
                double dm=0; for(int i=0;i<grp;i++) dm+=H[(size_t)i*grp+i]; dm/=grp;
                for(int i=0;i<grp;i++) H[(size_t)i*grp+i]+=0.02*(dm+1e-9);
                if(g2_inv(H,grp,Hi_all+(size_t)gi*grp*grp)!=0) fb_all=0;
                else if(cw){ const double *hi2=Hi_all+(size_t)gi*grp*grp;
                    for(int j=0;j<grp;j++){ double d=hi2[(size_t)j*grp+j];
                        cw[j0+j]=(float)(1.0/(d>1e-12?d:1e-12)); } }
            }
        }
        float *WkF=malloc((size_t)rows*cols*4);
        memcpy(WkF,W,(size_t)rows*cols*4);
        static __thread int *sidx_f=NULL; static __thread int sf_cap=0;
        const int nsegs=cols/dim;
        if(sf_cap<rows*nsegs){ free(sidx_f); sidx_f=malloc((size_t)rows*nsegs*sizeof(int)); sf_cap=rows*nsegs; }
        vqg_gptq_full(WkF,C,Hi_all,rows,cols,grp,dim,nc,fb_all,sidx_f,fb_all?cw:NULL);
        for(size_t i=0;i<(size_t)rows*nsegs;i++){
            const int c=sidx_f[i];
            idx_out[i]=c;
            memcpy(Wq+i*dim,C+(size_t)c*dim,(size_t)dim*4);
        }
        free(WkF); if(Hi_all) free(Hi_all);
        goto gptq_done;
    }
#endif
    for(int j0=0;j0<cols;j0+=grp){
        int g=cols-j0<grp?cols-j0:grp; int ok=0;
        if(XbT){
            for(int j=0;j<g;j++) for(int t=0;t<n_act;t++) XbT[(size_t)j*n_act+t]=X[(size_t)t*cols+j0+j];
            /* H=XbT·XbTᵀ 改走 dq_matmul(GPU 化 2026-08-18): 原手写标量三重循环
             * =128²/2×2906×768矩阵≈14.6T 标量op/层, 段计时实锤 GPTQ 段主导。
             * f32 GEMM 累计 vs double: H 本身带 2% 对角阻尼的近似 Hessian, held 对拍验。 */
            { float *Hf=malloc((size_t)g*g*4);
              dq_matmul(XbT,XbT,Hf,g,n_act,g);
              for(int i=0;i<g;i++) for(int j=0;j<g;j++) H[(size_t)i*g+j]=Hf[(size_t)i*g+j];
              free(Hf); }
            double dm=0; for(int i=0;i<g;i++) dm+=H[(size_t)i*g+i]; dm/=g;
            for(int i=0;i<g;i++) H[(size_t)i*g+i]+=0.02*(dm+1e-9);
            ok=g2_inv(H,g,Hi)==0;
            if(ok&&cw) for(int j=0;j<g;j++){ double d=Hi[(size_t)j*g+j];
                cw[j0+j]=(float)(1.0/(d>1e-12?d:1e-12)); }
        }
        for(int r=0;r<rows;r++) memcpy(Wk+(size_t)r*g,W+(size_t)r*cols+j0,(size_t)g*4);
#ifdef DS4QUANT_CUDA
        /* GPU 常驻组核(08-18): 段循环+误差反馈整组进 kernel(行间独立=每线程一行,
         * 行内与 CPU 逐项同序; Hi double→f32 唯一近似, held 对拍验)。 */
        if(g%dim==0&&vqg_shm_ok(nc,dim)&&vqg_ready()){
            const int nseg2=g/dim;
            static __thread int *sidx_g=NULL; static __thread int sg_cap=0;
            if(sg_cap<rows*nseg2){ free(sidx_g); sidx_g=malloc((size_t)rows*nseg2*sizeof(int)); sg_cap=rows*nseg2; }
            vqg_gptq_group(Wk,C,Hi,rows,g,dim,nc,ok,sidx_g,(ok&&cw)?cw+j0:NULL);
            for(int s2=0;s2<nseg2;s2++){ const int jj=s2*dim;
                for(int r=0;r<rows;r++){ const int c=sidx_g[(size_t)r*nseg2+s2];
                    idx_out[((size_t)r*cols+j0+jj)/dim]=c;
                    memcpy(Wq+(size_t)r*cols+j0+jj,C+(size_t)c*dim,(size_t)dim*4); } }
            continue;
        }
#endif
        for(int jj=0;jj+dim<=g;jj+=dim){
            for(int r=0;r<rows;r++) memcpy(seg+(size_t)r*dim,Wk+(size_t)r*g+jj,(size_t)dim*4);
            if(cw&&ok){ /* 加权选码字(与 GPU 核同式, 防两路静默分歧); 无 Hi 组/平权走 vq_assign_q 快路 */
                const float *wv=cw+j0+jj;
                for(int r=0;r<rows;r++){ const float *v=seg+(size_t)r*dim;
                    float best=3.4e38f; int bi=0;
                    for(int c=0;c<nc;c++){ const float *cb=C+(size_t)c*dim; float val=0.0f;
                        for(int d=0;d<dim;d++){ const float dd=v[d]-cb[d]; val+=wv[d]*dd*dd; }
                        if(val<best){best=val;bi=c;} }
                    sidx[r]=bi; }
            } else vq_assign_q(seg,rows,dim,C,nc,sidx);
            for(int r=0;r<rows;r++){
                const float *q=C+(size_t)sidx[r]*dim;
                idx_out[((size_t)r*cols+j0+jj)/dim]=sidx[r];
                memcpy(Wq+(size_t)r*cols+j0+jj,q,(size_t)dim*4);
                if(ok&&jj+dim<g){
                    for(int d=0;d<dim;d++){
                        double hjj=Hi[(size_t)(jj+d)*g+jj+d]; if(fabs(hjj)<1e-30)hjj=1e-30;
                        float er=(float)((seg[(size_t)r*dim+d]-q[d])/hjj);
                        if(er==0.0f) continue;
                        const double *hrow=Hi+(size_t)(jj+d)*g; float *wkr=Wk+(size_t)r*g;
                        for(int j2=jj+dim;j2<g;j2++) wkr[j2]-=er*(float)hrow[j2];
                    } } }
        }
    }
#ifdef DS4QUANT_CUDA
gptq_done:
#endif
    free(Wk);free(seg);free(sidx);free(H);free(Hi); if(XbT)free(XbT); if(cw)free(cw);
    /* 行乘子 g_r(闭式 ridge) + f16 往返 */
    if(X&&n_act>=8){
        float *P=malloc((size_t)n_act*rows*4),*Y=malloc((size_t)n_act*rows*4);
        dq_matmul(X,Wq,P,n_act,cols,rows); dq_matmul(X,W,Y,n_act,cols,rows);
        for(int r=0;r<rows;r++){
            double sp2=0,spy=0;
            for(int t=0;t<n_act;t++){ double p=P[(size_t)t*rows+r]; sp2+=p*p; spy+=p*(double)Y[(size_t)t*rows+r]; }
            double lam=sp2/(n_act+1.0),l0=1e-3*sp2+1e-9; if(lam<l0)lam=l0;
            double gr=(spy+lam)/(sp2+lam); if(gr<0.25)gr=0.25; else if(gr>4.0)gr=4.0;
            float gf=go1b_fp16_to_fp32(go1b_fp32_to_fp16((float)gr));
            gr_out[r]=gf;
            float *o=Wq+(size_t)r*cols; for(int j=0;j<cols;j++) o[j]*=gf;
        }
        free(P);free(Y);
    } else for(int r=0;r<rows;r++) gr_out[r]=1.0f;
    if(own) free(Wq);
}

/* ---- 打包/解包 ---- */
static size_t vq_pack(const float *Cb,const int *idx,const float *gr,
                      int rows,int cols,int dim,int nc,uint8_t *out){
    uint8_t *p=out;
    uint32_t mg=VQ_MAGIC; memcpy(p,&mg,4); p+=4;
    uint16_t d16=(uint16_t)dim,n16=(uint16_t)nc; memcpy(p,&d16,2);p+=2; memcpy(p,&n16,2);p+=2;
    uint32_t r32=(uint32_t)rows,c32=(uint32_t)cols; memcpy(p,&r32,4);p+=4; memcpy(p,&c32,4);p+=4;
    for(int i=0;i<nc*dim;i++){ uint16_t h=go1b_fp32_to_fp16(Cb[i]); memcpy(p,&h,2); p+=2; }
    for(int r=0;r<rows;r++){ uint16_t h=go1b_fp32_to_fp16(gr[r]); memcpy(p,&h,2); p+=2; }
    size_t nidx=(size_t)rows*cols/dim;
    int bits=1; while((1<<bits)<nc) bits++;   /* ★通用位宽(2026-08-09 重建): 256→8/512→9/1024→10;
                                                 旧 9bit 硬编码是 q2 时代遗留, nc=1024 高位截断=盘上乱码 */
    if(bits==8){ for(size_t i=0;i<nidx;i++) *p++=(uint8_t)idx[i]; }
    else {      /* 位流 LE, 与 9bit 老路径逐位兼容 */
        size_t nb=(nidx*(size_t)bits+7)/8+1; memset(p,0,nb);
        for(size_t i=0;i<nidx;i++){ size_t bit=i*(size_t)bits;
            uint32_t m=((uint32_t)idx[i]&((1u<<bits)-1))<<(bit&7);
            p[bit>>3]    |=(uint8_t)m;
            p[(bit>>3)+1]|=(uint8_t)(m>>8);
            if(m>>16) p[(bit>>3)+2]|=(uint8_t)(m>>16);
        }
        p+=nb;
    }
    return (size_t)(p-out);
}
#ifdef DS4QUANT_CUDA
extern int vqg_unpack_dequant(const uint8_t*,float*,int,int,int,int,int);
#endif
static int vq_unpack_dequant(const uint8_t *buf,size_t sz,float *Wout,int *rows_o,int *cols_o){
    const uint8_t *p=buf;
    uint32_t mg; memcpy(&mg,p,4); p+=4; if(mg!=VQ_MAGIC) return -1;
    uint16_t dim,nc; memcpy(&dim,p,2);p+=2; memcpy(&nc,p,2);p+=2;
    uint32_t rows,cols; memcpy(&rows,p,4);p+=4; memcpy(&cols,p,4);p+=4;
    if(rows_o)*rows_o=(int)rows; if(cols_o)*cols_o=(int)cols;
#ifdef DS4QUANT_CUDA
    /* B 回放主体=768 矩阵标量位流解码(18s/层); 大矩阵 GPU 整解(逐位同语义) */
    if((size_t)rows*cols>=1u<<20&&dim<=8&&nc<=1024){
        int bits2=1; while((1<<bits2)<nc) bits2++;
        if(vqg_unpack_dequant(buf,Wout,(int)rows,(int)cols,dim,nc,bits2)) return 0;
    }
#endif
    float *C=malloc((size_t)nc*dim*4);
    for(int i=0;i<nc*dim;i++){ uint16_t h; memcpy(&h,p,2); p+=2; C[i]=go1b_fp16_to_fp32(h); }
    float *gr=malloc((size_t)rows*4);
    for(uint32_t r=0;r<rows;r++){ uint16_t h; memcpy(&h,p,2); p+=2; gr[r]=go1b_fp16_to_fp32(h); }
    size_t nidx=(size_t)rows*cols/dim;
    int bits=1; while((1<<bits)<nc) bits++;
    for(size_t i=0;i<nidx;i++){
        uint32_t v;
        if(bits==8) v=p[i];
        else { size_t bit=i*(size_t)bits; v=(uint32_t)p[bit>>3]|((uint32_t)p[(bit>>3)+1]<<8)|((uint32_t)p[(bit>>3)+2]<<16); v=(v>>(bit&7))&((1u<<bits)-1); }
        size_t r=(i*dim)/cols;
        float g=gr[r]; const float *cb=C+(size_t)v*dim;
        float *o=Wout+i*dim;
        for(int d=0;d<dim;d++) o[d]=cb[d]*g;
    }
    (void)sz; free(C); free(gr); return 0;
}

/* ---- 层侧车(dql_vq_L%02d.bin): [hdr16][表 256×3 u64 off][载荷...] 偏移确定性 ---- */
#define VQSC_MAGIC 0x4C565144u  /* 'DQVL' */
static void vq_sidecar_path(const char*lf,int L,char*out,size_t n){
    strncpy(out,lf,n-1); out[n-1]=0;
    char*sl=strrchr(out,'/'); size_t dl=sl?(size_t)(sl-out+1):0;
    snprintf(out+dl,n-dl,"dql_vq_L%02d.bin",L);
}
static size_t vq_hdr_bytes(void){ return 16 + 256*3*8; }
/* which: 0=w1 1=w3 2=w2; 返回 0=该矩阵无 VQ 载荷 */
/* ★参数化槽位布局(2026-08-09 重建: 冷档从 rplan getter 取, 冷 w2>0 时带载荷 —
 * 公式已对 q2(1,751,665,168B)/q4(2,023,770,896B) 两个已知产物逐字节验证)★ */
static size_t vq_slot_off(int L,int e,int which,int moei,int dim_model){
    size_t off=vq_hdr_bytes();
    size_t hw13=vq_payload_bytes(moei,dim_model,vq_hot_dim(),vq_hot_nc()), hw2=vq_payload_bytes(dim_model,moei,vq_hot_dim(),vq_hot_nc());
    size_t cw13=vq_payload_bytes(moei,dim_model,vq_cold_dim(),vq_cold_nc());
    int w2d=vq_w2_dim();
    size_t cw2=w2d>0?vq_payload_bytes(dim_model,moei,w2d,vq_w2_nc()):0;
    for(int e2=0;e2<e;e2++){
        if(g2_hot_slot(L,e2)>=0) off+=2*hw13+hw2; else off+=2*cw13+cw2;
    }
    int hot=g2_hot_slot(L,e)>=0;
    if(!hot&&which==2&&cw2==0) return 0;
    if(which>=1) off+= hot?hw13:cw13;
    if(which>=2) off+= hot?hw13:cw13;
    return off;
}
static size_t vq_total_bytes(int L,int moei,int dim_model){
    size_t off=vq_hdr_bytes();
    size_t hw13=vq_payload_bytes(moei,dim_model,vq_hot_dim(),vq_hot_nc()), hw2=vq_payload_bytes(dim_model,moei,vq_hot_dim(),vq_hot_nc());
    size_t cw13=vq_payload_bytes(moei,dim_model,vq_cold_dim(),vq_cold_nc());
    int w2d=vq_w2_dim();
    size_t cw2=w2d>0?vq_payload_bytes(dim_model,moei,w2d,vq_w2_nc()):0;
    for(int e2=0;e2<256;e2++) off+= (g2_hot_slot(L,e2)>=0)?(2*hw13+hw2):(2*cw13+cw2);
    return off;
}

/* 编码+打包+落盘一条龙; 返回 dequant 浮点(合并态链), cos_out 可 NULL */
static float *vq_export_matrix(const float*W,int rows,int cols,const float*X,int n_act,
                               int dim,int nc,int vqfd,size_t off,double*cos_out){
    float *Wq=malloc((size_t)rows*cols*4);
    float *Cb=malloc((size_t)nc*dim*4),*gr=malloc((size_t)rows*4);
    int *idx=malloc(((size_t)rows*cols/dim)*sizeof(int));
    vq_encode_full(W,rows,cols,dim,nc,X,n_act,Cb,idx,gr,Wq);
    if(vqfd>0&&off>0){
        size_t pb=vq_payload_bytes(rows,cols,dim,nc);
        uint8_t *buf=malloc(pb);
        size_t got=vq_pack(Cb,idx,gr,rows,cols,dim,nc,buf);
        if(got!=pb) fprintf(stderr,"[vq] pack %zu != 预期 %zu (r%d c%d d%d nc%d)\n",got,pb,rows,cols,dim,nc);
        if(pwrite(vqfd,buf,got,(off_t)off)!=(ssize_t)got) perror("vq-pwrite");
        free(buf);
    }
    if(cos_out){ double num=0,na=0,nb=0; size_t N=(size_t)rows*cols;
        for(size_t i=0;i<N;i++){ num+=(double)W[i]*Wq[i]; na+=(double)W[i]*W[i]; nb+=(double)Wq[i]*Wq[i]; }
        *cos_out=num/(sqrt(na)*sqrt(nb)+1e-30); }
    free(Cb);free(idx);free(gr);
    return Wq;
}

/* signref 同形 API: dequant 浮点(合并态链用) */
static float *dq_quant_expert_vq(const float *W,int nrows,int ncols,const float *X,int n_act,int dim,int nc){
    float *Wq=malloc((size_t)nrows*ncols*4);
    float *Cb=malloc((size_t)nc*dim*4); float *gr=malloc((size_t)nrows*4);
    int *idx=malloc(((size_t)nrows*ncols/dim)*sizeof(int));
    vq_encode_full(W,nrows,ncols,dim,nc,X,n_act,Cb,idx,gr,Wq);
    free(Cb);free(idx);free(gr);
    return Wq;
}

/* ── 战役期 API 重建(2026-08-09: git checkout 误覆盖未提交现版, 按主文件调用面重建;
 *    验收闸 = r30_campaign.sh probe1 复现 cos=0.9699 / held=0.0668 / vq md5=4843132e…) ── */
extern int g_vq_dim, g_vq_nc;   /* rplan 逐层档位全局(主文件 vq_rplan 写入) */
/* 冷档 8 维/256 码 = 2.25bpw 的既有定数(改它 = 换盘上格式, 侧车偏移表全体挪位);
 * 原 DS4_VQ_COLD_DIM/NC 两个 env 从无活脚本设过, 2026-08-31 随禁 env 铁律写死。 */
static int vq_cold_dim(void){ return g_vq_dim?g_vq_dim:8; }
static int vq_cold_nc(void){ return g_vq_nc?g_vq_nc:256; }
/* 顺序补偿变体: 调用侧已传量化态 hidden(hc), 每行输出匹配闭式缩放本就烘在
 * vq_encode_full 的 g_r 段 → 与非 seq 同实现, flag 仅语义标注。 */
static float *vq_export_matrix_seq(const float*W,int rows,int cols,const float*X,int n_act,
                               int dim,int nc,int vqfd,size_t off,double*cos_out,int seq){
    (void)seq;
    return vq_export_matrix(W,rows,cols,X,n_act,dim,nc,vqfd,off,cos_out);
}
static float *dq_quant_expert_vq_seq(const float *W,int nrows,int ncols,const float *X,int n_act,int dim,int nc,int seq){
    (void)seq;
    return dq_quant_expert_vq(W,nrows,ncols,X,n_act,dim,nc);
}
#endif
