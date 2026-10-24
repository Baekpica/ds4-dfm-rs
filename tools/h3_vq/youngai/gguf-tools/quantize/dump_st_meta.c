/* dump_st_meta.c — safetensors 张量清单器(2026-09-10, V4.1-Flash 底座体积账)。
 *
 * 【为什么要它】量化前必须知道"哪些张量、多少参数、原始多少字节"。V4.1-Flash 的 HF
 * 没有 model.safetensors.index.json(48 个分片各自带头), 而 src/common/ds4_st.c 是 V4
 * 专用读器(FP8 E4M3 + 128×128 块 scale), 对 V4.1 的 32×32 ue8m0 + 专家 FP4 不适用 ——
 * 但**清单**不需要 dequant, 只要读每个分片的 JSON 头。所以单独一个只读头的工具。
 *
 * 不这么做会怎样: 徒手按 config.json 推参数量, 漏掉 engram/vision/mtp/scale_inv 这些
 * 结构外张量, 体积账直接差几十 GB(历史上徒手换算连错三次, 见 fable5 bpw 口径注记)。
 *
 * 输出一行一张量: <name> <dtype> <bytes> <numel> <shape>
 * 汇总交给外层 awk —— 分类规则会随战役变, 不写死进 C。
 *
 * 编译: cc -O2 -o dump_st_meta dump_st_meta.c
 * 用法: dump_st_meta <hf-dir>
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <dirent.h>

/* safetensors 头: [8B u64 LE header_len][header_len 字节 JSON][张量数据]。
 * JSON 形如 {"name":{"dtype":"F8_E4M3","shape":[a,b],"data_offsets":[s,e]}, ..., "__metadata__":{...}}
 * 这里只做扫描式解析(不建树): 逐个找 '"' 开头的 key, 再在它的 {...} 内取三个字段。 */

static const char *skip_ws(const char *p){ while(*p==' '||*p=='\n'||*p=='\t'||*p=='\r') p++; return p; }

/* 取 "key":"值" 里的字符串值, 写进 out。找不到返回 0。 */
static int get_str(const char *seg, const char *key, char *out, size_t n){
    char pat[64]; snprintf(pat,sizeof pat,"\"%s\":",key);
    const char *p=strstr(seg,pat); if(!p) return 0;
    p=skip_ws(p+strlen(pat)); if(*p!='"') return 0; p++;
    size_t i=0; while(*p && *p!='"' && i+1<n) out[i++]=*p++;
    out[i]=0; return 1;
}

/* 取 "key":[a,b,...] 的整数数组, 最多 nmax 个, 返回实际个数。 */
static int get_ints(const char *seg, const char *key, long long *v, int nmax){
    char pat[64]; snprintf(pat,sizeof pat,"\"%s\":",key);
    const char *p=strstr(seg,pat); if(!p) return 0;
    p=skip_ws(p+strlen(pat)); if(*p!='[') return 0; p++;
    int n=0;
    for(;;){
        p=skip_ws(p);
        if(*p==']'||!*p) break;
        char *end; long long x=strtoll(p,&end,10);
        if(end==p) break;
        if(n<nmax) v[n]=x;
        n++; p=skip_ws(end);
        if(*p==',') p++;
    }
    return n;
}

static void scan_shard(const char *path){
    FILE *f=fopen(path,"rb");
    if(!f){ fprintf(stderr,"打不开 %s\n",path); exit(1); }
    uint64_t hlen=0;
    if(fread(&hlen,8,1,f)!=1){ fprintf(stderr,"%s 头长读不到\n",path); exit(1); }
    /* 头长的合理性: 太大说明不是 safetensors(或字节序/截断), 与其读进 GB 级垃圾不如硬拒 */
    if(hlen<2 || hlen>(uint64_t)1<<30){ fprintf(stderr,"%s header_len=%llu 不合理\n",path,(unsigned long long)hlen); exit(1); }
    char *hdr=malloc(hlen+1);
    if(!hdr){ fprintf(stderr,"内存不足 %llu\n",(unsigned long long)hlen); exit(1); }
    if(fread(hdr,1,hlen,f)!=hlen){ fprintf(stderr,"%s 头读不满\n",path); exit(1); }
    hdr[hlen]=0; fclose(f);

    const char *p=hdr;
    p=skip_ws(p); if(*p=='{') p++;
    while(*p){
        p=skip_ws(p);
        if(*p!='"') { if(!*p||*p=='}') break; p++; continue; }
        p++;
        char name[512]; size_t i=0;
        while(*p && *p!='"' && i+1<sizeof name) name[i++]=*p++;
        name[i]=0; if(*p=='"') p++;
        p=skip_ws(p); if(*p!=':'){ continue; } p++;
        p=skip_ws(p);
        if(*p!='{'){ /* 非对象值(不该出现), 跳到下一个逗号 */
            while(*p && *p!=',') p++; if(*p==',') p++; continue;
        }
        /* 取这一段 {...}(safetensors 的张量条目内部无嵌套对象, 数深度即可) */
        const char *seg=p; int depth=0;
        while(*p){ if(*p=='{') depth++; else if(*p=='}'){ depth--; if(!depth){ p++; break; } } p++; }
        size_t seglen=(size_t)(p-seg);
        char *sbuf=malloc(seglen+1); memcpy(sbuf,seg,seglen); sbuf[seglen]=0;

        if(strcmp(name,"__metadata__")!=0){
            char dt[64]="?"; long long shp[8]={0}, off[2]={0,0};
            get_str(sbuf,"dtype",dt,sizeof dt);
            int nd=get_ints(sbuf,"shape",shp,8);
            int no=get_ints(sbuf,"data_offsets",off,2);
            long long bytes = (no==2)? off[1]-off[0] : -1;
            long long numel = 1;
            for(int k=0;k<nd && k<8;k++) numel*=shp[k];
            if(nd==0) numel=0;
            printf("%s\t%s\t%lld\t%lld\t",name,dt,bytes,numel);
            for(int k=0;k<nd && k<8;k++) printf("%s%lld",k?"x":"",shp[k]);
            if(nd==0) printf("scalar");
            printf("\n");
        }
        free(sbuf);
        p=skip_ws(p); if(*p==',') p++;
    }
    free(hdr);
}

int main(int argc,char**argv){
    if(argc<2){ fprintf(stderr,"usage: dump_st_meta <hf-dir>\n"); return 2; }
    DIR *d=opendir(argv[1]);
    if(!d){ fprintf(stderr,"目录打不开 %s\n",argv[1]); return 2; }
    /* 分片名排序无所谓(汇总在外层), 但要全扫到 —— 漏一个分片= 体积账少几 GB 且看不出来 */
    struct dirent *e; int n=0;
    char path[4096];
    while((e=readdir(d))){
        const char *s=strstr(e->d_name,".safetensors");
        if(!s || s[12]) continue;             /* 必须以 .safetensors 结尾 */
        snprintf(path,sizeof path,"%s/%s",argv[1],e->d_name);
        scan_shard(path); n++;
    }
    closedir(d);
    fprintf(stderr,"扫了 %d 个分片\n",n);
    return n? 0 : 1;
}
