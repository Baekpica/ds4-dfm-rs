/* IQuest is CUDA-only; preserve Metal linkage and reject execution. */
int ds4_gpu_iquest_policy(void) {
    return 0;
}

int ds4_gpu_iquest_rms(ds4_gpu_tensor *out, const ds4_gpu_tensor *x,
        const void *map, uint64_t size, uint64_t offset,
        uint32_t width, uint32_t rows) {
    (void)out; (void)x; (void)map; (void)size; (void)offset;
    (void)width; (void)rows; return 0;
}

int ds4_gpu_iquest_rope(ds4_gpu_tensor *x, const ds4_gpu_tensor *positions,
        uint32_t heads, uint32_t rows, float theta) {
    (void)x; (void)positions; (void)heads; (void)rows; (void)theta; return 0;
}

int ds4_gpu_iquest_router(ds4_gpu_tensor *ids, ds4_gpu_tensor *weights,
        const ds4_gpu_tensor *logits, uint32_t rows) {
    (void)ids; (void)weights; (void)logits; (void)rows; return 0;
}

int ds4_gpu_iquest_kv(ds4_gpu_tensor *cache, const ds4_gpu_tensor *key,
        const ds4_gpu_tensor *value, const ds4_gpu_tensor *positions,
        uint32_t rows, uint32_t capacity) {
    (void)cache; (void)key; (void)value; (void)positions;
    (void)rows; (void)capacity; return 0;
}

int ds4_gpu_iquest_attn(ds4_gpu_tensor *out, const ds4_gpu_tensor *query,
        const ds4_gpu_tensor *cache, const ds4_gpu_tensor *positions,
        const void *map, uint64_t size, uint64_t offset,
        uint32_t rows, uint32_t capacity, uint32_t window) {
    (void)out; (void)query; (void)cache; (void)positions;
    (void)map; (void)size; (void)offset; (void)rows; (void)capacity;
    (void)window; return 0;
}

int ds4_gpu_iquest_add(ds4_gpu_tensor *out, const ds4_gpu_tensor *residual,
        const ds4_gpu_tensor *branch, uint32_t rows, float scale) {
    (void)out; (void)residual; (void)branch; (void)rows; (void)scale; return 0;
}

int ds4_gpu_iquest_sum(ds4_gpu_tensor *out, const ds4_gpu_tensor *experts,
        const ds4_gpu_tensor *weights, uint32_t rows) {
    (void)out; (void)experts; (void)weights; (void)rows; return 0;
}

int ds4_gpu_iquest_swiglu(ds4_gpu_tensor *out, const ds4_gpu_tensor *gate,
        const ds4_gpu_tensor *up, uint64_t count) {
    (void)out; (void)gate; (void)up; (void)count; return 0;
}

int ds4_gpu_iquest_expert(ds4_gpu_tensor *out, const ds4_gpu_tensor *x,
        const ds4_gpu_tensor *ids, const void *map, uint64_t size, uint64_t offset,
        uint32_t type, uint32_t in, uint32_t width, uint32_t rows,
        uint32_t used, uint32_t input_used) {
    (void)out; (void)x; (void)ids; (void)map; (void)size; (void)offset;
    (void)type; (void)in; (void)width; (void)rows; (void)used;
    (void)input_used; return 0;
}
