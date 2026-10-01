/* Dump production CUDA stages for bounded real-weight prefix comparison.
 * Including the native implementation exposes static graph helpers only
 * to this probe; ordinary applications still use the Rust host boundary. */
#define DS4_USE_CUDA 1
#include "../ds4.c"

static const char *prefix_manifest(void) {
    const char *manifest = getenv("DS4_CUDA_WEIGHT_IPC_MANIFEST");
    const char *scope = getenv("DS4_CUDA_WEIGHT_IPC_SCOPE");
    const char *copy = getenv("DS4_CUDA_COPY_MODEL");
    if (!manifest || !manifest[0] || (copy && copy[0]) ||
        (scope && scope[0] && strcmp(scope,"base") && strcmp(scope,"both"))) {
        fprintf(stderr,"prefix probe requires a canonical base weight owner; "
                       "set DS4_CUDA_WEIGHT_IPC_MANIFEST and unset DS4_CUDA_COPY_MODEL\n");
        return NULL;
    }
    return manifest;
}

static bool prefix_weights(ds4_model *m, const char *path, const char *manifest) {
    /* Import the serving owner's canonical ranges before matrix execution.
     * Terminal mapped fallback cannot allocate a second full weight copy. */
    if (!ds4_gpu_iquest_policy() || !ds4_gpu_init()) { return false; }
    const int fd = m->split_count > 1u ? -1 : m->fd;
    if (!ds4_gpu_set_model_fd(fd) ||
        ds4_gpu_model_source_bind(m->map,m->size,DS4_MSRC_ROLE_PRIMARY,fd,
            DS4_RESIDENCY_HOST_MAPPED,"base",path) < 0 ||
        !ds4_gpu_set_model_map(m->map,m->size) ||
        !ds4_gpu_import_model_ipc_manifest(m->map,m->size,manifest,"base")) {
        return false;
    }
    model_release_mapping_cache(m);
    return true;
}

static bool dump_stage(const char *dir, const char *name, const ds4_gpu_tensor *t, uint64_t bytes) {
    char path[1024]; snprintf(path, sizeof(path), "%s/%s", dir, name);
    void *values = malloc(bytes); FILE *fp = fopen(path, "wb");
    bool ok = values && fp && ds4_gpu_tensor_read(t, 0, values, bytes);
    if (ok) { ok = fwrite(values, 1, bytes, fp) == bytes; }
    if (fp) { ok = fclose(fp) == 0 && ok; }
    free(values); return ok;
}

int main(int argc, char **argv) {
    if (argc != 3) { fprintf(stderr, "usage: test_iquest_prefix MODEL.gguf OUTPUT_DIR\n"); return 2; }
    const char *manifest = prefix_manifest();
    if (!manifest) { return 2; }
    const unsigned n=3; const int tokens[]={1,2,3}; const unsigned positions[]={0,1,2};
    ds4_model m={0}; ds4_weights w={0}; ds4_iquest_graph g={0};
    g_ds4_shape=DS4_SHAPE_IQUEST_Q1;
    model_open(&m,argv[1],false,false); iquest_bind(&w,&m);
    if (!prefix_weights(&m,argv[1],manifest) || !iquest_graph_alloc(&g,8,4,IQ_MTP_OFF)) {
        ds4_gpu_cleanup(); model_close(&m); return 1;
    }
    bool ok=ds4_gpu_tensor_write(g.tokens,0,tokens,sizeof(tokens)) &&
        ds4_gpu_tensor_write(g.positions,0,positions,sizeof(positions)) &&
        ds4_gpu_embed_tokens_quant_tensor(g.ws.b_cur,g.tokens,m.map,m.size,
            w.token_embd->abs_offset,w.token_embd->type,IQ_VOCAB,n,IQ_EMBED) &&
        iquest_round(g.ws.b_cur,(uint64_t)n*IQ_EMBED) &&
        dump_stage(argv[2],"embedding.f32",g.ws.b_cur,(uint64_t)n*IQ_EMBED*4);
    for (unsigned il=0;ok&&il<2;il++) {
        ok=iquest_layer(&g,&m,&w.layer[il],g.kv[il],g.kv_cap[il],n,0,il==0?IQ_DENSE_LAYER:IQ_MOE_LAYER);
        char name[64];
#define DUMP(label,tensor,count) do { snprintf(name,sizeof(name),"layer%u-%s",il,label); \
    ok=ok&&dump_stage(argv[2],name,(tensor),(uint64_t)(count)*4); } while(0)
        DUMP("hidden.f32",g.ws.b_cur,n*IQ_EMBED);
        DUMP("q.f32",g.ws.b_q,n*IQ_HEADS*IQ_HEAD);
        DUMP("k.f32",g.ws.b_k,n*IQ_KV_HEADS*IQ_HEAD);
        DUMP("v.f32",g.ws.b_v,n*IQ_KV_HEADS*IQ_HEAD);
        DUMP("heads.f32",g.ws.b_heads,n*IQ_HEADS*IQ_HEAD);
        DUMP("attn_out.f32",g.ws.b_attn_out,n*IQ_EMBED);
        DUMP("ffn_out.f32",g.ws.b_ffn_out,n*IQ_EMBED);
        snprintf(name,sizeof(name),"layer%u-kv.q8_0",il);
        ok=ok&&dump_stage(argv[2],name,g.kv[il],(uint64_t)n*IQ_Q8_ROW_BLOCKS*sizeof(iquest_q8));
        if (il) {
            DUMP("router.f32",g.ws.b_router_logits,n*IQ_EXPERTS);
            DUMP("ids.u32",g.ws.b_router_selected,n*IQ_USED);
            DUMP("weights.f32",g.ws.b_router_weights,n*IQ_USED);
            DUMP("gate.f32",g.ws.b_routed_gate,n*IQ_USED*IQ_FF);
            DUMP("up.f32",g.ws.b_routed_up,n*IQ_USED*IQ_FF);
            DUMP("mid.f32",g.ws.b_routed_mid,n*IQ_USED*IQ_FF);
            DUMP("down.f32",g.ws.b_routed_down,n*IQ_USED*IQ_EMBED);
        }
#undef DUMP
    }
    uint64_t live[DS4_MEMC__COUNT]={0};
    for (int cls=DS4_MEMC_WEIGHT_ARENA; cls<=DS4_MEMC_WEIGHT_HOST_PIN; cls++) {
        ds4_mem_cell cell={0};
        const int domain=cls==DS4_MEMC_WEIGHT_HOST_PIN?DS4_MEMD_PINNED_HOST:DS4_MEMD_UNIFIED_DEVICE;
        ok=ds4_gpu_mem_census_read(cls,domain,&cell)==0 && ok;
        live[cls]=ds4_mem_cell_live(&cell);
    }
    const uint64_t owned=live[DS4_MEMC_WEIGHT_ARENA]+live[DS4_MEMC_WEIGHT_SPAN]+live[DS4_MEMC_WEIGHT_WHOLE];
    const uint64_t derived=live[DS4_MEMC_WEIGHT_DERIVED]+live[DS4_MEMC_WEIGHT_ARTIFACT];
    const uint64_t imported=live[DS4_MEMC_WEIGHT_IMPORT], faults=ds4_gpu_mem_census_faults();
    ok=ok && !owned && !derived && imported && !faults;
    printf("{\"scope\":\"published-real-weight-two-layer-prefix\",\"layers\":2,\"tokens\":[1,2,3],\"positions\":[0,1,2],\"ok\":%s,\"requested_residency\":\"mapped\",\"weight_owner_imported\":%s,\"canonical_owned_device_bytes\":%llu,\"derived_device_bytes\":%llu,\"imported_device_bytes\":%llu,\"registered_host_bytes\":%llu,\"graph_device_bytes\":%llu,\"census_faults\":%llu,\"whole_model_quality_qualified\":false}\n",
        ok?"true":"false",imported?"true":"false",(unsigned long long)owned,
        (unsigned long long)derived,(unsigned long long)imported,
        (unsigned long long)live[DS4_MEMC_WEIGHT_HOST_PIN],(unsigned long long)g.bytes,
        (unsigned long long)faults);
    iquest_graph_free(&g); ds4_gpu_cleanup(); model_close(&m);
    return ok?0:1;
}
