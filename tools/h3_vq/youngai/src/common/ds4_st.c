/* ds4_st.c — 最小 DeepSeek V4 HF 读器 (C): FP8 E4M3 + 128×128 块 F32 scale safetensors。
 * 全仓唯一 safetensors 实现(2026-08-25 重构阶段2 自 quant/st_read.c 升格;
 * 旧路径留转发 stub 服务 7 个源 include 式消费方, 链接式转换见阶段7)。
 * 对应 ds4reader.py。忠实移植: e4m3 LUT + 块 scale dequant + safetensors JSON 头(最小解析)。
 * 验证: --selftest 读一个真实 HF 权重前几值, 与 numpy R.read_weight 对拍。
 * 编译: cc -O3 -DST_READ_SELFTEST -lm st_read.c -o st_selftest
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include <pthread.h>
#include <unistd.h>
#include <fcntl.h>

/* ★页缓存旁路(2026-08-22)★ FP 锚一趟要读 ~138GB 权重, 而每块权重在一趟里**只用一次**,
 * 零复用。默认走页缓存 = 138GB 灌进 121GB 的 cache, 逼内核全程回收, 线程集体卡在
 * folio_wait_bit_common, 读吞吐被压到 2.1 GB/s(盘 20 并发实测 8 GB/s)。
 * 读完立刻 DONTNEED 丢掉, 让缓存只服务真正会复用的东西。 */
static ssize_t st_pread(int fd, void *buf, size_t n, off_t off) {
    ssize_t r = pread(fd, buf, n, off);
    if (r > 0) posix_fadvise(fd, off, (off_t)r, POSIX_FADV_DONTNEED);
    return r;
}

#include "ds4_fp8.h"

static float ST_LUT[256];
/* LUT 改由 src/common/ds4_fp8.h 的唯一 E4M3 解码填充(重构阶段2 收敛 6 份副本)。
 * 数值逐位等价原 powf 式: 全部是 2 的幂精确缩放, NaN/±0 槽位同。 */
static void st_lut_init(void) {
    for (int b = 0; b < 256; b++) ST_LUT[b] = ds4_e4m3fn_to_f32((uint8_t)b);
}

/* 在 JSON 文本里找 "name":{...} 段, 取 dtype/shape[2]/data_offsets[2]. 返回段起始或 NULL. */
static const char *st_find(const char *hdr, const char *name, char *dtype, long *shape, long *off) {
    char key[512]; snprintf(key,sizeof(key),"\"%s\"",name);
    const char *p=strstr(hdr,key); if(!p) return NULL;
    const char *d=strstr(p,"\"dtype\":\""); if(!d) return NULL; d+=9;
    int i=0; while(*d && *d!='"' && i<15) dtype[i++]=*d++; dtype[i]=0;
    const char *sh=strstr(p,"\"shape\":["); if(!sh) return NULL; sh+=9;
    shape[0]=strtol(sh,(char**)&sh,10); shape[1]=1;
    if(*sh==',') { sh++; shape[1]=strtol(sh,(char**)&sh,10); }
    const char *of=strstr(p,"\"data_offsets\":["); if(!of) return NULL; of+=16;
    off[0]=strtol(of,(char**)&of,10); if(*of==',') of++; off[1]=strtol(of,(char**)&of,10);
    return p;
}

/* ★I/O 层重做(2026-08-22)★
 * 原路每读一个权重: strstr 扫 5.5MB index JSON → fopen+读整个 168KB shard 头 → 再 fopen
 * 读数据; scale 再走一遍。一趟 FP 锚 3.3 万权重 ⇒ 6.6 万次 fopen、11GB 小随机头读、
 * 180GB 索引扫描。实测磁盘只跑出 118 MB/s(盘能力 3.6 GB/s), 线程全卡在页缓存等待。
 * 改成: ①index 一次性解析成按名排序的数组, bsearch 取 shard ②每个 shard 的 fd 与头
 * 只开/解一次并缓存(48 个) ③数据读走 pread(无 fseek 竞态, 天然线程安全)。 */
typedef struct { const char *key, *val; } st_kv;
typedef struct { char name[256]; char *hdr; long data_start; int fd; } st_shd;
typedef struct {
    char hf[1024]; char *idx_json;
    st_kv *kv; long nkv;                 /* weight_map: name→shard, 按 name 排序 */
    st_shd sh[128]; int nsh;             /* shard 句柄+头缓存 */
    pthread_mutex_t lock;
} st_ctx;

/* FP4 行并行 dequant worker(文件级; 嵌套函数在 noexecstack 下会静默串行) */
typedef struct { const uint8_t*buf,*sb; float*w; long Rr,Cin,nblk; const float*T; } st_fp4_ctx;
typedef struct { st_fp4_ctx*c; long r0,r1; } st_fp4_arg;
static void *st_fp4_worker(void*va){
    st_fp4_arg*p=(st_fp4_arg*)va; st_fp4_ctx*c=p->c;
    for(long r=p->r0;r<p->r1;r++) for(long b=0;b<c->nblk;b++){
        uint8_t e=c->sb[(size_t)r*c->nblk+b]; uint32_t u = e==0 ? 0x00400000u : ((uint32_t)e<<23);
        float sc; memcpy(&sc,&u,4);
        const uint8_t *src=c->buf+((size_t)r*c->nblk+b)*16;
        float *dst=c->w+(size_t)r*c->Cin+(size_t)b*32;
        for(int j=0;j<16;j++){ dst[2*j]=c->T[src[j]&0x0f]*sc; dst[2*j+1]=c->T[(src[j]>>4)&0x0f]*sc; }
    }
    return (void*)0;
}

static int st_kvcmp(const void *a, const void *b) {
    return strcmp(((const st_kv *)a)->key, ((const st_kv *)b)->key);
}

static char *st_slurp(const char *path, long *len) {
    FILE *f=fopen(path,"rb"); if(!f) return NULL;
    fseek(f,0,SEEK_END); long n=ftell(f); fseek(f,0,SEEK_SET);
    char *b=malloc(n+1); if(fread(b,1,n,f)!=(size_t)n){free(b);fclose(f);return NULL;} b[n]=0; fclose(f); if(len)*len=n; return b;
}

static void st_open(st_ctx *c, const char *hf) {
    st_lut_init(); snprintf(c->hf,sizeof(c->hf),"%s",hf);
    char p[1200]; snprintf(p,sizeof(p),"%s/model.safetensors.index.json",hf);
    c->idx_json=st_slurp(p,NULL);
    if(!c->idx_json){ fprintf(stderr,"st_read: no index %s\n",p); exit(1); }
    c->nsh=0; pthread_mutex_init(&c->lock,NULL);
    /* weight_map 就地解析: "name": "shard" 逐对, 把引号改 NUL 让 key/val 直接指进 idx_json */
    char *w=strstr(c->idx_json,"\"weight_map\"");
    if(!w){ fprintf(stderr,"st_read: index 无 weight_map\n"); exit(1); }
    w=strchr(w,'{'); if(!w){ fprintf(stderr,"st_read: weight_map 格式异常\n"); exit(1); }
    long cap=4096; c->kv=malloc((size_t)cap*sizeof(st_kv)); c->nkv=0;
    char *q=w+1;
    while(*q){
        while(*q==' '||*q=='\n'||*q=='\t'||*q==',') q++;
        if(*q=='}'||!*q) break;
        if(*q!='"') break;
        char *ks=++q; while(*q&&*q!='"') q++; if(!*q) break; *q++=0;
        while(*q==' '||*q==':') q++;
        if(*q!='"') break;
        char *vs=++q; while(*q&&*q!='"') q++; if(!*q) break; *q++=0;
        if(c->nkv==cap){ cap*=2; c->kv=realloc(c->kv,(size_t)cap*sizeof(st_kv)); }
        c->kv[c->nkv].key=ks; c->kv[c->nkv].val=vs; c->nkv++;
    }
    qsort(c->kv,(size_t)c->nkv,sizeof(st_kv),st_kvcmp);
}

/* 找 name 所在 shard 文件名 (index weight_map 里 "name": "shard", 冒号后可能有空格). */
static int st_shard(st_ctx *c, const char *name, char *shard) {
    st_kv k; k.key=name; k.val=NULL;
    const st_kv *r=bsearch(&k,c->kv,(size_t)c->nkv,sizeof(st_kv),st_kvcmp);
    if(!r) return 0;
    snprintf(shard,256,"%s",r->val); return 1;
}

/* shard 句柄+头缓存: 一个 shard 只 open/解头一次, 之后所有读走 pread(线程安全) */
static st_shd *st_shard_get(st_ctx *c, const char *shard) {
    for(int i=0;i<c->nsh;i++) if(!strcmp(c->sh[i].name,shard)) return &c->sh[i];
    pthread_mutex_lock(&c->lock);
    for(int i=0;i<c->nsh;i++) if(!strcmp(c->sh[i].name,shard)){ pthread_mutex_unlock(&c->lock); return &c->sh[i]; }
    if(c->nsh>=128){ pthread_mutex_unlock(&c->lock); return NULL; }
    char p[1300]; snprintf(p,sizeof(p),"%s/%s",c->hf,shard);
    int fd=open(p,O_RDONLY); if(fd<0){ pthread_mutex_unlock(&c->lock); return NULL; }
    uint64_t n; if(pread(fd,&n,8,0)!=8){ close(fd); pthread_mutex_unlock(&c->lock); return NULL; }
    char *hdr=malloc(n+1);
    if(pread(fd,hdr,n,8)!=(ssize_t)n){ free(hdr); close(fd); pthread_mutex_unlock(&c->lock); return NULL; }
    hdr[n]=0;
    st_shd *e=&c->sh[c->nsh];
    snprintf(e->name,sizeof(e->name),"%s",shard); e->hdr=hdr; e->data_start=8+(long)n; e->fd=fd;
    c->nsh++;                       /* 先填后自增: 读者看到的条目一定是完整的 */
    pthread_mutex_unlock(&c->lock);
    return e;
}

/* 读 shard 头: 返回 header json (malloc) + data_start. */
static char *st_shard_hdr(st_ctx *c, const char *shard, long *data_start) {
    char p[1300]; snprintf(p,sizeof(p),"%s/%s",c->hf,shard);
    FILE *f=fopen(p,"rb"); if(!f) return NULL;
    uint64_t n; if(fread(&n,8,1,f)!=1){fclose(f);return NULL;}
    char *hdr=malloc(n+1); if(fread(hdr,1,n,f)!=n){free(hdr);fclose(f);return NULL;} hdr[n]=0;
    *data_start=8+(long)n; fclose(f); return hdr;
}

/* read_weight(name) → dequant f32 [R*C] (malloc). fp8 E4M3 * 128×128 块scale. R/C out. */
float *st_read_weight(st_ctx *c, const char *name, long *R_out, long *C_out) {
    char shard[256]; if(!st_shard(c,name,shard)){fprintf(stderr,"st: no %s\n",name);return NULL;}
    st_shd *SH=st_shard_get(c,shard); if(!SH) return NULL;
    long ds=SH->data_start; char *hdr=SH->hdr; int fdw=SH->fd;   /* 头与 fd 都是缓存的, 不再 fopen/读头 */
    char dt[16]; long shape[2],off[2];
    if(!st_find(hdr,name,dt,shape,off)) return NULL;
    long Rr=shape[0],Cc=shape[1];
    float *w=malloc((size_t)Rr*Cc*sizeof(float));
    if(strcmp(dt,"F8_E4M3")==0){
        uint8_t *buf=malloc((size_t)Rr*Cc);
        if(st_pread(fdw,buf,(size_t)Rr*Cc,ds+off[0])!=(ssize_t)((size_t)Rr*Cc)){free(buf);free(w);return NULL;}
        for(size_t i=0;i<(size_t)Rr*Cc;i++) w[i]=ST_LUT[buf[i]]; free(buf);
        /* 块scale name→.scale [R/128,C/128]。dtype 两世代: 老 Base 落盘 F32;
         * 0731 起是真 F8_E8M0(1 字节纯指数, 值=2^(e-127); e=0 按 0x00400000 位型,
         * 与 deepseek4-quantize.c e8m0_to_f32 逐位一致)。按头里的 dtype 分派, 硬拒其他。 */
        char sname[512]; snprintf(sname,sizeof(sname),"%s",name);
        char *ww=strstr(sname,".weight"); if(ww) strcpy(ww,".scale");
        char shard2[256]; st_shard(c,sname,shard2);
        st_shd *SH2=st_shard_get(c,shard2); long ds2=SH2?SH2->data_start:0; char *hdr2=SH2?SH2->hdr:NULL;
        char dt2[16]; long ssh[2],soff[2];
        if(hdr2 && st_find(hdr2,sname,dt2,ssh,soff)){
            long sbr=ssh[0],sbc=ssh[1]; int fd2=SH2->fd;
            float *sc=malloc((size_t)sbr*sbc*sizeof(float));
            if(strcmp(dt2,"F8_E8M0")==0){
                uint8_t *sb=malloc((size_t)sbr*sbc);
                if(st_pread(fd2,sb,(size_t)sbr*sbc,ds2+soff[0])!=(ssize_t)((size_t)sbr*sbc)){fprintf(stderr,"st: scale 读不满 %s\n",sname);exit(1);}
                for(size_t i=0;i<(size_t)sbr*sbc;i++){
                    uint32_t u = sb[i]==0 ? 0x00400000u : ((uint32_t)sb[i]<<23);
                    memcpy(&sc[i],&u,4);
                }
                free(sb);
            } else if(strcmp(dt2,"F32")==0){
                if(st_pread(fd2,sc,(size_t)sbr*sbc*4,ds2+soff[0])!=(ssize_t)((size_t)sbr*sbc*4)){fprintf(stderr,"st: scale 读不满 %s\n",sname);exit(1);}
            } else {
                fprintf(stderr,"st: 未知 scale dtype %s (%s) — 拒跑\n",dt2,sname); exit(1);
            }
            for(long r=0;r<Rr;r++) for(long cc=0;cc<Cc;cc++)
                w[(size_t)r*Cc+cc]*=sc[(size_t)(r/128)*sbc+(cc/128)];
            free(sc);
        }
    } else if(strcmp(dt,"I8")==0){
        /* ★0731 routed 专家 = MXFP4★: I8 容器 [R, C/2](每字节 2 个 E2M1 nibble, 低位先)
         * + F8_E8M0 scale [R, C/32](1×32 微块)。返回口径与其余分支一致: 解包后 f32 [R, C真]。
         * 几何/查表与 deepseek4-quantize.c dequant_fp4_weight 逐位一致(那边已验过 packed release)。 */
        /* FP4 值表来自 src/common/ds4_fp8.h 唯一实现(重构阶段2 收敛) */
        static float FP4T[16]; static int fp4t_init=0;
        if(!fp4t_init){ for(int i=0;i<16;i++) FP4T[i]=ds4_fp4_nibble_to_f32((uint8_t)i); fp4t_init=1; }
        long Cin=Cc*2, nblk=Cin/32;
        if(Cin%32){ fprintf(stderr,"st: FP4 %s C=%ld 不整除32\n",name,Cin); exit(1); }
        char sname[512]; snprintf(sname,sizeof(sname),"%s",name);
        char *ww=strstr(sname,".weight"); if(ww) strcpy(ww,".scale");
        char shard2[256]; long ds2=0; char *hdr2=NULL; char dt2[16]; long ssh[2],soff[2];
        st_shd *SH2=NULL;
        if(!st_shard(c,sname,shard2) || !(SH2=st_shard_get(c,shard2)) ||
           !(hdr2=SH2->hdr) || ((ds2=SH2->data_start),0) ||
           !st_find(hdr2,sname,dt2,ssh,soff) || strcmp(dt2,"F8_E8M0")!=0 ||
           ssh[0]!=Rr || ssh[1]!=nblk){
            fprintf(stderr,"st: FP4 %s 缺配套 E8M0 scale [R,C/32] — 拒跑\n",name); exit(1);
        }
        w=realloc(w,(size_t)Rr*Cin*sizeof(float));   /* 真列数是 2×容器列 */
        uint8_t *buf=malloc((size_t)Rr*Cc);
        if(st_pread(fdw,buf,(size_t)Rr*Cc,ds+off[0])!=(ssize_t)((size_t)Rr*Cc)){fprintf(stderr,"st: FP4 读不满 %s\n",name);exit(1);}
        uint8_t *sb=malloc((size_t)Rr*nblk);
        if(st_pread(SH2->fd,sb,(size_t)Rr*nblk,ds2+soff[0])!=(ssize_t)((size_t)Rr*nblk)){fprintf(stderr,"st: FP4 scale 读不满 %s\n",sname);exit(1);}
        /* ★行并行 dequant(2026-08-23 v2: 嵌套函数 trampoline 在 noexecstack 下
         * pthread_create 静默失败→串行 fallback, 换文件级 worker)。行独立数值逐位不变。 */
        {
            st_fp4_ctx fc={buf,sb,w,Rr,Cin,nblk,FP4T};
            int nth=16; if(nth>Rr)nth=(int)Rr;
            pthread_t th[16]; st_fp4_arg pa[16];
            long per=(Rr+nth-1)/nth; int cnt=0;
            for(int t=0;t<nth;t++){ long r0=t*per,r1=r0+per>Rr?Rr:r0+per; if(r0>=r1)break;
                pa[cnt].c=&fc; pa[cnt].r0=r0; pa[cnt].r1=r1;
                if(pthread_create(&th[cnt],NULL,st_fp4_worker,&pa[cnt])){ st_fp4_worker(&pa[cnt]); continue; }
                cnt++; }
            for(int t=0;t<cnt;t++) pthread_join(th[t],NULL);
        }
        free(buf); free(sb);
        Cc=Cin;   /* 下游按真实列数走 */
    } else if(strcmp(dt,"BF16")==0){
        uint16_t *buf=malloc((size_t)Rr*Cc*2);
        if(st_pread(fdw,buf,(size_t)Rr*Cc*2,ds+off[0])!=(ssize_t)((size_t)Rr*Cc*2)){fprintf(stderr,"st: BF16 读不满 %s\n",name);exit(1);}
        for(size_t i=0;i<(size_t)Rr*Cc;i++){ uint32_t u=(uint32_t)buf[i]<<16; memcpy(&w[i],&u,4);} free(buf);
    } else if(strcmp(dt,"I64")==0){ /* int64 → float (eid≤255精确) */
        int64_t *buf=malloc((size_t)Rr*Cc*8);
        if(st_pread(fdw,buf,(size_t)Rr*Cc*8,ds+off[0])!=(ssize_t)((size_t)Rr*Cc*8)){fprintf(stderr,"st: I64 读不满 %s\n",name);exit(1);}
        for(size_t i=0;i<(size_t)Rr*Cc;i++) w[i]=(float)buf[i]; free(buf);
    } else { /* F32 */
        if(st_pread(fdw,w,(size_t)Rr*Cc*4,ds+off[0])!=(ssize_t)((size_t)Rr*Cc*4)){fprintf(stderr,"st: F32 读不满 %s\n",name);exit(1);}
    }
    if(R_out)*R_out=Rr; if(C_out)*C_out=Cc; return w;   /* fd/头由 st_ctx 缓存持有, 不在这里关 */
}

#ifdef ST_READ_SELFTEST
int main(int argc,char**argv){
    if(argc<2){fprintf(stderr,"usage: st_selftest <hf-dir> [tensor-name]\n");return 2;}
    const char *hf=argv[1];
    st_ctx c; st_open(&c,hf);
    const char *name=argc>2?argv[2]:"layers.0.attn_norm.weight";
    long R,C; float *w=st_read_weight(&c,name,&R,&C);
    if(!w){printf("FAIL read %s\n",name);return 1;}
    printf("%s R=%ld C=%ld first6:",name,R,C);
    for(int i=0;i<6 && i<R*C;i++) printf(" %.6f",w[i]);
    printf("\n"); free(w); return 0;
}
#endif
