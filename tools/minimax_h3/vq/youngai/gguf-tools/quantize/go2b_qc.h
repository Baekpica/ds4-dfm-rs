/* go2b_qc.h — GO2B(type41) C 编码/解码 + 热表 (go2b_encode.py 数值口径逐行移植)。
 * 块 68B/256el: [f16 d1][f16 d2][s1 32B][s2 32B]; 值=(s1?+d1:-d1)+(s2?+d2:-d2), 位 g→字节 g/8 位 g%8。
 * 编码链(= act-scale(原 DS4_GO2B_ACT_SCALE env, 已随 env 大扫除拔除) 验证赢家口径, 全43层 +27.2%):
 *   ① _fit_d1d2 nf: 行分位点(75/25)初值 + 3轮 Lloyd(level3/2 条件均值, 空集回 hi/lo)
 *   ② GPTQ 误差反馈分配(校准行≥8; 列组128 Hessian=XbᵀXb+0.02·mean(diag), err/Hinv[j,j] 反馈后列)
 *   ③ 2轮联合迭代: 固定码解每行 2×2 输出最优(d1,d2) → f16 往返重建 level → 重分配
 * 与 py 的口径差(有依据): w2 的激活喂真 hc(量化 q1/q3 顺序补偿链输出), 非层输入前 2048 维近似
 * —— 部署真实分布, 严格更准; gate/up 仍喂层输入行(同 py)。校准行内部截 256(py [:256] 同口径)。
 */
#ifndef DS4QUANT_GO2B_H
#define DS4QUANT_GO2B_H

#define GO2B_BLK_QK    256
#define GO2B_BLK_BYTES 68
static inline size_t go2b_row_bytes(int ncols){ return (size_t)(ncols/GO2B_BLK_QK)*GO2B_BLK_BYTES; }

/* 运行时同款 dequant(kernel_mul_mm_id_go2b 口径) */
static void dq_go2b_bytes_dequant(const uint8_t*b,int nrows,int ncols,float*out){
    int nblk=ncols/GO2B_BLK_QK;
    for(int r=0;r<nrows;r++){
        const uint8_t*rb=b+(size_t)r*nblk*GO2B_BLK_BYTES;
        float*orow=out+(size_t)r*ncols;
        for(int bl=0;bl<nblk;bl++){
            const uint8_t*bd=rb+(size_t)bl*GO2B_BLK_BYTES;
            uint16_t h1,h2; memcpy(&h1,bd,2); memcpy(&h2,bd+2,2);
            float d1=go1b_fp16_to_fp32(h1), d2=go1b_fp16_to_fp32(h2);
            const uint8_t*s1=bd+4,*s2=bd+36;
            for(int k=0;k<GO2B_BLK_QK;k++){
                int j=bl*GO2B_BLK_QK+k; if(j>=ncols)break;
                orow[j]=(((s1[k>>3]>>(k&7))&1)?d1:-d1)+(((s2[k>>3]>>(k&7))&1)?d2:-d2);
            }
        }
    }
}

static int g2_cmp_f(const void*a,const void*b){ float x=*(const float*)a,y=*(const float*)b; return (x>y)-(x<y); }
static float g2_quantile_sorted(const float*a,int n,double q){   /* np.quantile 线性插值 */
    double pos=q*(double)(n-1); int lo=(int)pos; double fr=pos-(double)lo;
    if(lo>=n-1) return a[n-1];
    return (float)((double)a[lo]*(1.0-fr)+(double)a[lo+1]*fr);
}
/* _fit_d1d2 nf 口径(单行): 分位点初值 + 3 轮 Lloyd; scratch=ncols float */
static void g2_fit_d1d2_row(const float*w,int n,float*scratch,float*d1o,float*d2o){
    for(int j=0;j<n;j++) scratch[j]=fabsf(w[j]);
    qsort(scratch,(size_t)n,sizeof(float),g2_cmp_f);
    double hi=g2_quantile_sorted(scratch,n,0.75), lo=g2_quantile_sorted(scratch,n,0.25);
    double d1=(hi+lo)/2.0, d2=(hi-lo)/2.0;
    for(int it=0;it<3;it++){
        double l2=d1-d2,l3=d1+d2;
        double s3=0,s2v=0; long n3=0,n2=0;
        for(int j=0;j<n;j++){
            double v=w[j];
            double c0=fabs(v+l3),c1=fabs(v+l2),c2=fabs(v-l2),c3=fabs(v-l3);
            int k=0; double m=c0;                      /* np.argmin: 取第一个最小 */
            if(c1<m){m=c1;k=1;} if(c2<m){m=c2;k=2;} if(c3<m){m=c3;k=3;}
            if(k==3){ s3+=v; n3++; } else if(k==2){ s2v+=v; n2++; }
        }
        double v3=n3? s3/(double)n3 : hi;
        double v2=n2? s2v/(double)n2 : lo;
        d1=(v3+v2)/2.0; d2=fabs((v3-v2)/2.0)+1e-6;
    }
    *d1o=(float)d1; *d2o=(float)d2;
}
static inline int g2_nearest(float v,const float*l4){
    int k=0; float m=fabsf(v-l4[0]);
    for(int i=1;i<4;i++){ float d=fabsf(v-l4[i]); if(d<m){m=d;k=i;} }
    return k;
}
/* 就地 Gauss-Jordan 求逆(double, 列主元); 0=成功 */
static int g2_inv(const double*A,int n,double*Ainv){
    int stride=2*n;
    double*M=malloc((size_t)n*stride*sizeof(double)); if(!M) return -1;
    for(int i=0;i<n;i++){ for(int j=0;j<n;j++){ M[(size_t)i*stride+j]=A[(size_t)i*n+j]; M[(size_t)i*stride+n+j]=(i==j)?1.0:0.0; } }
    for(int c=0;c<n;c++){
        int p=c; double mx=fabs(M[(size_t)c*stride+c]);
        for(int r=c+1;r<n;r++){ double v=fabs(M[(size_t)r*stride+c]); if(v>mx){mx=v;p=r;} }
        if(mx<1e-30){ free(M); return -1; }
        if(p!=c) for(int j=0;j<stride;j++){ double t=M[(size_t)c*stride+j]; M[(size_t)c*stride+j]=M[(size_t)p*stride+j]; M[(size_t)p*stride+j]=t; }
        double d=M[(size_t)c*stride+c];
        for(int j=0;j<stride;j++) M[(size_t)c*stride+j]/=d;
        for(int r=0;r<n;r++){ if(r==c) continue; double f2=M[(size_t)r*stride+c]; if(f2==0.0) continue;
            for(int j=0;j<stride;j++) M[(size_t)r*stride+j]-=f2*M[(size_t)c*stride+j]; }
    }
    for(int i=0;i<n;i++) memcpy(Ainv+(size_t)i*n,M+(size_t)i*stride+n,(size_t)n*sizeof(double));
    free(M); return 0;
}
/* _gptq_assign 口径: X 有效(≥8行)走误差反馈; 否则纯最近邻。
 * GPTQ 恒开(原 DS4_GO2B_NO_GPTQ 诊断关断已随禁 env 铁律删除)。 */
static void g2_assign(const float*W,int rows,int cols,const float*L4,
                      const float*X,int n,int grp,float ridge,
                      float*Wq,int8_t*sidx){
    if(!X||n<8){
        for(int r=0;r<rows;r++){ const float*wr=W+(size_t)r*cols; const float*l4=L4+(size_t)r*4;
            float*qr=Wq+(size_t)r*cols; int8_t*sr=sidx+(size_t)r*cols;
            for(int j=0;j<cols;j++){ int k=g2_nearest(wr[j],l4); sr[j]=(int8_t)k; qr[j]=l4[k]; } }
        return;
    }
    double *H=malloc((size_t)grp*grp*sizeof(double)),*Hinv=malloc((size_t)grp*grp*sizeof(double));
    float *XbT=malloc((size_t)grp*n*sizeof(float));
    float *Wk=malloc((size_t)rows*grp*sizeof(float)),*err=malloc((size_t)rows*sizeof(float));
    for(int j0=0;j0<cols;j0+=grp){
        int g=(cols-j0<grp)?(cols-j0):grp;
        for(int j=0;j<g;j++) for(int t=0;t<n;t++) XbT[(size_t)j*n+t]=X[(size_t)t*cols+j0+j];
        for(int i=0;i<g;i++) for(int j=i;j<g;j++){ double s=0;
            const float*xi=XbT+(size_t)i*n,*xj=XbT+(size_t)j*n;
            for(int t=0;t<n;t++) s+=(double)xi[t]*xj[t];
            H[(size_t)i*g+j]=H[(size_t)j*g+i]=s; }
        double dm=0; for(int i=0;i<g;i++) dm+=H[(size_t)i*g+i]; dm/=(double)g;
        for(int i=0;i<g;i++) H[(size_t)i*g+i]+=ridge*(dm+1e-9);
        int inv_ok = g2_inv(H,g,Hinv)==0;
        for(int r=0;r<rows;r++) memcpy(Wk+(size_t)r*g,W+(size_t)r*cols+j0,(size_t)g*sizeof(float));
        for(int j=0;j<g;j++){
            double hjj=inv_ok?Hinv[(size_t)j*g+j]:1.0; if(fabs(hjj)<1e-30)hjj=1e-30;
            for(int r=0;r<rows;r++){
                const float*l4=L4+(size_t)r*4;
                float col=Wk[(size_t)r*g+j];
                int k=g2_nearest(col,l4); float q=l4[k];
                Wq[(size_t)r*cols+j0+j]=q; sidx[(size_t)r*cols+j0+j]=(int8_t)k;
                err[r]=(float)(((double)col-q)/hjj);
            }
            if(inv_ok&&j+1<g){
                const double*hrow=Hinv+(size_t)j*g;
                for(int r=0;r<rows;r++){ float er=err[r]; if(er==0.0f) continue;
                    float*wkr=Wk+(size_t)r*g;
                    for(int jj=j+1;jj<g;jj++) wkr[jj]-=er*(float)hrow[jj]; }
            }
        }
    }
    free(H);free(Hinv);free(XbT);free(Wk);free(err);
}
static const int g2_s1of[4]={0,0,1,1}, g2_s2of[4]={0,1,0,1};   /* level idx → (s1,s2) 位 */
/* _act_d1d2: 固定码 ±1, 每行 2×2 输出最优 (|d1|,|d2|); P/T 走 Accelerate sgemm。
 * Yadj(可NULL)[t*rows+r]: 目标平移(α·W·(x̃−x̂), GPTAQ 非对称目标, 2026-07-25 G1b)。 */
static void g2_act_d1d2(const float*W,int rows,int cols,const int8_t*sidx,
                        const float*X,int n,float*d1,float*d2,const float*Yadj){
    size_t N=(size_t)rows*cols;
    float *B1=malloc(N*sizeof(float)),*B2=malloc(N*sizeof(float));
    for(size_t i=0;i<N;i++){ int k=sidx[i]; B1[i]=g2_s1of[k]?1.0f:-1.0f; B2[i]=g2_s2of[k]?1.0f:-1.0f; }
    float *P1=malloc((size_t)n*rows*4),*P2=malloc((size_t)n*rows*4),*T=malloc((size_t)n*rows*4);
    dq_matmul(X,B1,P1,n,cols,rows); dq_matmul(X,B2,P2,n,cols,rows); dq_matmul(X,W,T,n,cols,rows);
    if(Yadj) for(size_t i=0;i<(size_t)n*rows;i++) T[i]+=Yadj[i];
    for(int r=0;r<rows;r++){
        double a=0,b=0,c=0,r1=0,r2=0;
        for(int t=0;t<n;t++){ double p1=P1[(size_t)t*rows+r],p2=P2[(size_t)t*rows+r],tv=T[(size_t)t*rows+r];
            a+=p1*p1; b+=p1*p2; c+=p2*p2; r1+=tv*p1; r2+=tv*p2; }
        double det=a*c-b*b; if(fabs(det)<1e-12) det=1e-12;
        d1[r]=(float)fabs((c*r1-b*r2)/det); d2[r]=(float)fabs((a*r2-b*r1)/det);
    }
    free(B1);free(B2);free(P1);free(P2);free(T);
}
/* 编码整矩阵: out(可NULL=只要浮点) 68B 块字节, wq_opt(可NULL) dequant 浮点(== decode(out))。
 * X[n_act×ncols]=真实激活; gate/up 喂层输入, w2 喂顺序补偿 hc。 */
static void dq_go2b_encode_adj(const float*W,int nrows,int ncols,const float*X,int n_act,
                           uint8_t*out,float*wq_opt,const float*Yadj);
static void dq_go2b_encode(const float*W,int nrows,int ncols,const float*X,int n_act,
                           uint8_t*out,float*wq_opt){ dq_go2b_encode_adj(W,nrows,ncols,X,n_act,out,wq_opt,NULL); }
static void dq_go2b_encode_adj(const float*W,int nrows,int ncols,const float*X,int n_act,
                           uint8_t*out,float*wq_opt,const float*Yadj){
    if(n_act>256) n_act=256;
    int nblk=ncols/GO2B_BLK_QK;
    float *d1=malloc((size_t)nrows*4),*d2=malloc((size_t)nrows*4),*L4=malloc((size_t)nrows*16);
    float *scratch=malloc((size_t)ncols*4);
    for(int r=0;r<nrows;r++) g2_fit_d1d2_row(W+(size_t)r*ncols,ncols,scratch,&d1[r],&d2[r]);
    free(scratch);
    for(int r=0;r<nrows;r++){
        float a=go1b_fp16_to_fp32(go1b_fp32_to_fp16(d1[r])), b=go1b_fp16_to_fp32(go1b_fp32_to_fp16(d2[r]));
        L4[(size_t)r*4+0]=-(a+b); L4[(size_t)r*4+1]=-(a-b); L4[(size_t)r*4+2]=(a-b); L4[(size_t)r*4+3]=(a+b);
    }
    int own_wq=0; float*Wq=wq_opt;
    if(!Wq){ Wq=malloc((size_t)nrows*ncols*sizeof(float)); own_wq=1; }
    int8_t *sidx=malloc((size_t)nrows*ncols);
    const int grp=128;       /* GPTQ 组宽: 原 DS4_GO2B_GRP 从无活脚本改过, 写死 */
    float ridge=0.02f;
    g2_assign(W,nrows,ncols,L4,X,n_act,grp,ridge,Wq,sidx);
    int actrounds=(!X||n_act<8)?0:2;   /* 激活重拟合轮数: 原 DS4_GO2B_ACT 关断诊断已删 */
    for(int round=0;round<actrounds;round++){
        g2_act_d1d2(W,nrows,ncols,sidx,X,n_act,d1,d2,Yadj);
        for(int r=0;r<nrows;r++){
            float a=go1b_fp16_to_fp32(go1b_fp32_to_fp16(d1[r])), b=go1b_fp16_to_fp32(go1b_fp32_to_fp16(d2[r]));
            L4[(size_t)r*4+0]=-(a+b); L4[(size_t)r*4+1]=-(a-b); L4[(size_t)r*4+2]=(a-b); L4[(size_t)r*4+3]=(a+b);
        }
        g2_assign(W,nrows,ncols,L4,X,n_act,grp,ridge,Wq,sidx);
    }
    if(out){
        size_t rb=go2b_row_bytes(ncols);
        for(int r=0;r<nrows;r++){
            uint16_t h1=go1b_fp32_to_fp16(d1[r]),h2=go1b_fp32_to_fp16(d2[r]);
            uint8_t*rd=out+(size_t)r*rb;
            for(int bl=0;bl<nblk;bl++){
                uint8_t*bd=rd+(size_t)bl*GO2B_BLK_BYTES;
                memset(bd,0,GO2B_BLK_BYTES);
                bd[0]=(uint8_t)(h1&0xFF); bd[1]=(uint8_t)(h1>>8);
                bd[2]=(uint8_t)(h2&0xFF); bd[3]=(uint8_t)(h2>>8);
            }
            const int8_t*sr=sidx+(size_t)r*ncols;
            for(int j=0;j<ncols;j++){ int k=sr[j]; int bl=j>>8,k2=j&255;
                uint8_t*bd=rd+(size_t)bl*GO2B_BLK_BYTES;
                if(g2_s1of[k]) bd[4+(k2>>3)] |=(uint8_t)(1u<<(k2&7));
                if(g2_s2of[k]) bd[36+(k2>>3)]|=(uint8_t)(1u<<(k2&7));
            }
        }
    }
    free(d1);free(d2);free(L4);free(sidx); if(own_wq)free(Wq);
}
/* signref 同形 API: 量化-重构浮点(合并态前向/反修用) */
static float *dq_quant_expert_go2b_adj(const float*W,int nrows,int ncols,const float*X,int n_act,const float*Yadj){
    float *wq=malloc((size_t)nrows*ncols*sizeof(float));
    dq_go2b_encode_adj(W,nrows,ncols,X,n_act,NULL,wq,Yadj);
    return wq;
}
static float *dq_quant_expert_go2b(const float*W,int nrows,int ncols,const float*X,int n_act){
    return dq_quant_expert_go2b_adj(W,nrows,ncols,X,n_act,NULL);
}

/* ===== 热表(prog_active_top64): L<n>: id id ... — 热专家走 go2b, 冷保 go1b ===== */
static int GO2B_HOT=0;
static int16_t G2_SLOT[64][256];
static uint16_t G2_IDS[64][256]; static int G2_K[64];
static int go2b_hot_load(const char*path,int nlayers){
    FILE*f=fopen(path,"r"); if(!f) return -1;
    for(int L=0;L<64;L++){ G2_K[L]=0; for(int e=0;e<256;e++) G2_SLOT[L][e]=-1; }
    char ln[8192];
    while(fgets(ln,sizeof(ln),f)){
        if(ln[0]!='L') continue;
        char*c=strchr(ln,':'); if(!c) continue;
        int L=atoi(ln+1); if(L<0||L>=nlayers||L>=64) continue;
        char*p=c+1; int k=0;
        while(*p&&k<256){
            while(*p==' '||*p=='\t')p++;
            if(*p<'0'||*p>'9')break;
            int e=(int)strtol(p,&p,10);
            if(e>=0&&e<256&&G2_SLOT[L][e]<0){ G2_SLOT[L][e]=(int16_t)k; G2_IDS[L][k]=(uint16_t)e; k++; }
        }
        G2_K[L]=k;
    }
    fclose(f); return 0;
}
static inline int g2_hot_slot(int L,int e){ return (GO2B_HOT&&L>=0&&L<64&&G2_K[L]>0)?(int)G2_SLOT[L][e]:-1; }   /* G2_K[L]>0: 未 arm 层静态零初值防误判热(多层回放雷, 2026-07-29) */
/* 回放恒走 go2b(原 DS4_GO2B_REPLAY=0 强制 go1b 的 A/B 诊断已随禁 env 铁律删除)。 */
static inline int g2_replay_en(void){ return 1; }
#endif
