/* Exercise the production graph dispatch with mocked GPU projections.
 * This specifically detects asymmetric quantization promotions: reading an
 * IQ2_XXS up matrix with the gate's IQ2_XS type corrupts block addressing. */
#define DS4_USE_CUDA 1
#include "../ds4.c"

static unsigned expected_gate, expected_up, calls, wrong;
static int projection(uint64_t offset, unsigned type) {
    const unsigned expected = offset == 11 ? expected_gate : offset == 22 ? expected_up : DS4_TENSOR_Q4_K;
    calls++;
    if (type != expected) { wrong++; return 0; }
    return 1;
}

int ds4_gpu_iquest_expert(ds4_gpu_tensor *out, const ds4_gpu_tensor *x,
    const ds4_gpu_tensor *ids, const void *map, uint64_t size, uint64_t offset,
    uint32_t type, uint32_t in, uint32_t width, uint32_t rows, uint32_t used, uint32_t input_used) {
    (void)out;(void)x;(void)ids;(void)map;(void)size;(void)in;(void)width;(void)rows;(void)used;(void)input_used;
    return projection(offset, type);
}
int ds4_gpu_routed_matmul_guarded_tensor(ds4_gpu_tensor *out, const ds4_gpu_tensor *x,
    const ds4_gpu_tensor *ids, const void *map, uint64_t size, uint64_t offset,
    uint64_t bytes, uint32_t type, uint32_t in, uint32_t width, uint32_t experts,
    uint32_t rows, uint32_t used, uint32_t maximum) {
    (void)out;(void)x;(void)ids;(void)map;(void)size;(void)bytes;(void)in;(void)width;
    (void)experts;(void)rows;(void)used;(void)maximum;
    return projection(offset, type);
}
int ds4_gpu_routed_gate_up_tensor(ds4_gpu_tensor *gate, ds4_gpu_tensor *up,
    const ds4_gpu_tensor *x, const ds4_gpu_tensor *ids, const void *map, uint64_t size,
    uint64_t go, uint64_t gb, uint64_t uo, uint64_t ub, uint32_t type,
    uint32_t in, uint32_t width, uint32_t experts, uint32_t rows, uint32_t used) {
    (void)gate;(void)up;(void)x;(void)ids;(void)map;(void)size;(void)gb;(void)ub;
    (void)in;(void)width;(void)experts;(void)rows;(void)used;
    return projection(go, type) && projection(uo, type);
}
int ds4_gpu_naive_swiglu(ds4_gpu_tensor *out, const ds4_gpu_tensor *gate,
    const ds4_gpu_tensor *up, uint64_t count) { (void)out;(void)gate;(void)up;(void)count; return 1; }
int ds4_gpu_iquest_swiglu(ds4_gpu_tensor *out, const ds4_gpu_tensor *gate,
    const ds4_gpu_tensor *up, uint64_t count) { (void)out;(void)gate;(void)up;(void)count; return 1; }
int ds4_gpu_iquest_sum(ds4_gpu_tensor *out, const ds4_gpu_tensor *experts,
    const ds4_gpu_tensor *weights, uint32_t rows) { (void)out;(void)experts;(void)weights;(void)rows; return 1; }
int ds4_gpu_motif3_round_bf16_tensor(ds4_gpu_tensor *out, const ds4_gpu_tensor *x,
    uint64_t count) { (void)out;(void)x;(void)count; return 1; }

int main(void) {
    const unsigned pairs[][2] = {{DS4_TENSOR_IQ2_XS,DS4_TENSOR_IQ2_XXS},
        {DS4_TENSOR_IQ2_XXS,DS4_TENSOR_IQ2_XS},{DS4_TENSOR_BF16,DS4_TENSOR_F16},
        {DS4_TENSOR_IQ2_XXS,DS4_TENSOR_IQ2_XXS}};
    unsigned failed = 0;
    for (unsigned i = 0; i < sizeof(pairs)/sizeof(pairs[0]); i++) {
        ds4_exaone_batch_ws b = {0}; ds4_model m = {0}; ds4_layer_weights l = {0};
        ds4_tensor gate = {0}, up = {0}, down = {0};
        gate.type=expected_gate=pairs[i][0]; gate.abs_offset=11;
        up.type=expected_up=pairs[i][1]; up.abs_offset=22;
        down.type=DS4_TENSOR_Q4_K; down.abs_offset=33;
        l.ffn_gate_exps=&gate; l.ffn_up_exps=&up; l.ffn_down_exps=&down;
        calls=wrong=0;
        const bool ok=iquest_experts(&b,&m,&l,1);
        printf("{\"gate_type\":%u,\"up_type\":%u,\"ok\":%s,\"wrong_dispatches\":%u,\"projection_calls\":%u}\n",
            gate.type,up.type,ok?"true":"false",wrong,calls);
        failed += !ok || wrong || calls != 3;
    }
    return failed ? 1 : 0;
}
