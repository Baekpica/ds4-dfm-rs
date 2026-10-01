/* Naive is CUDA-only; keep Metal linkage explicit and fail before execution. */
int ds4_gpu_naive_round(ds4_gpu_tensor *x, uint64_t count) {
    (void)x; (void)count; return 0;
}
int ds4_gpu_naive_rope(ds4_gpu_tensor *x, const ds4_gpu_tensor *table,
        uint32_t heads, uint32_t width, uint32_t rows) {
    (void)x; (void)table; (void)heads; (void)width; (void)rows; return 0;
}
int ds4_gpu_naive_rms(ds4_gpu_tensor *out, const ds4_gpu_tensor *x,
        const void *map, uint64_t size, uint64_t offset, uint32_t rows) {
    (void)out; (void)x; (void)map; (void)size; (void)offset; (void)rows; return 0;
}
int ds4_gpu_naive_key_norm(ds4_gpu_tensor *x, const void *map, uint64_t size,
        uint64_t weight, uint64_t bias, uint32_t rows) {
    (void)x; (void)map; (void)size; (void)weight; (void)bias; (void)rows; return 0;
}
int ds4_gpu_naive_kv_store(ds4_gpu_tensor *cache, const ds4_gpu_tensor *k,
        const ds4_gpu_tensor *v, const ds4_gpu_tensor *positions,
        uint32_t heads, uint32_t rows, uint32_t capacity) {
    (void)cache; (void)k; (void)v; (void)positions; (void)heads; (void)rows; (void)capacity; return 0;
}
int ds4_gpu_naive_index_store(ds4_gpu_tensor *codes, ds4_gpu_tensor *scales,
        ds4_gpu_tensor *query, const ds4_gpu_tensor *key, const ds4_gpu_tensor *positions,
        uint32_t rows, uint32_t capacity) {
    (void)codes; (void)scales; (void)query; (void)key; (void)positions; (void)rows; (void)capacity; return 0;
}
int ds4_gpu_naive_select(ds4_gpu_tensor *ids, ds4_gpu_tensor *scores,
        ds4_gpu_tensor *a, ds4_gpu_tensor *b, const ds4_gpu_tensor *query,
        const ds4_gpu_tensor *codes, const ds4_gpu_tensor *scales,
        const ds4_gpu_tensor *weights, const ds4_gpu_tensor *positions,
        uint32_t history, uint32_t rows) {
    (void)ids; (void)scores; (void)a; (void)b; (void)query; (void)codes; (void)scales;
    (void)weights; (void)positions; (void)history; (void)rows; return 0;
}
int ds4_gpu_naive_attention(ds4_gpu_tensor *out, const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *cache, const ds4_gpu_tensor *positions, const ds4_gpu_tensor *ids,
        const void *map, uint64_t size, uint64_t sink, uint32_t heads,
        uint32_t capacity, uint32_t rows, uint32_t window) {
    (void)out; (void)q; (void)cache; (void)positions; (void)ids; (void)map; (void)size;
    (void)sink; (void)heads; (void)capacity; (void)rows; (void)window; return 0;
}
int ds4_gpu_naive_router(ds4_gpu_tensor *ids, ds4_gpu_tensor *weights,
        const ds4_gpu_tensor *logits, const void *map, uint64_t size,
        uint64_t offset, uint32_t rows) {
    (void)ids; (void)weights; (void)logits; (void)map; (void)size; (void)offset; (void)rows; return 0;
}
int ds4_gpu_naive_swiglu(ds4_gpu_tensor *out, const ds4_gpu_tensor *gate,
        const ds4_gpu_tensor *up, uint64_t count) {
    (void)out; (void)gate; (void)up; (void)count; return 0;
}
int ds4_gpu_naive_down(ds4_gpu_tensor *out, const ds4_gpu_tensor *gate,
        const ds4_gpu_tensor *up, const ds4_gpu_tensor *ids,
        const void *map, uint64_t size, uint64_t offset, uint64_t bytes,
        uint32_t tokens) {
    (void)out; (void)gate; (void)up; (void)ids; (void)map;
    (void)size; (void)offset; (void)bytes; (void)tokens; return -1;
}
int ds4_gpu_naive_sum(ds4_gpu_tensor *out, const ds4_gpu_tensor *down,
        const ds4_gpu_tensor *weights, uint32_t rows) {
    (void)out; (void)down; (void)weights; (void)rows; return 0;
}
int ds4_gpu_naive_sum_add(ds4_gpu_tensor *cur, const ds4_gpu_tensor *down,
        const ds4_gpu_tensor *weights, uint32_t rows) {
    (void)cur; (void)down; (void)weights; (void)rows; return -1;
}
int ds4_gpu_naive_add(ds4_gpu_tensor *cur, const ds4_gpu_tensor *other, uint64_t count) {
    (void)cur; (void)other; (void)count; return 0;
}
int ds4_gpu_naive_df_tap(ds4_gpu_tensor *out, const ds4_gpu_tensor *hidden,
        uint32_t first, uint32_t rows, uint32_t tap) {
    (void)out; (void)hidden; (void)first; (void)rows; (void)tap; return 0;
}
int ds4_gpu_naive_df_mask(ds4_gpu_tensor *hidden, const void *map,
        uint64_t size, uint64_t offset, uint32_t rows) {
    (void)hidden; (void)map; (void)size; (void)offset; (void)rows; return 0;
}
int ds4_gpu_naive_df_norm(ds4_gpu_tensor *x, const void *map, uint64_t size,
        uint64_t offset, uint32_t heads, uint32_t rows) {
    (void)x; (void)map; (void)size; (void)offset; (void)heads; (void)rows; return 0;
}
int ds4_gpu_naive_df_rope(ds4_gpu_tensor *x, const ds4_gpu_tensor *table,
        uint32_t heads, uint32_t rows) {
    (void)x; (void)table; (void)heads; (void)rows; return 0;
}
int ds4_gpu_naive_df_store(ds4_gpu_tensor *cache, const ds4_gpu_tensor *k,
        const ds4_gpu_tensor *v, const ds4_gpu_tensor *positions, uint32_t rows) {
    (void)cache; (void)k; (void)v; (void)positions; (void)rows; return 0;
}
int ds4_gpu_naive_df_attn(ds4_gpu_tensor *out, const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *cache, const ds4_gpu_tensor *k, const ds4_gpu_tensor *v,
        uint32_t first, uint32_t start, uint32_t rows) {
    (void)out; (void)q; (void)cache; (void)k; (void)v; (void)first; (void)start; (void)rows; return 0;
}
int ds4_gpu_naive_df_conf(ds4_gpu_tensor *out, const ds4_gpu_tensor *hidden,
        const ds4_gpu_tensor *markov, const void *map, uint64_t size, uint64_t weight, uint64_t bias) {
    (void)out; (void)hidden; (void)markov; (void)map; (void)size; (void)weight; (void)bias; return 0;
}
int ds4_gpu_naive_df_top2(ds4_gpu_tensor *out, const ds4_gpu_tensor *logits) {
    (void)out; (void)logits; return 0;
}
