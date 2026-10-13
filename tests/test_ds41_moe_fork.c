/* The runner includes the exact native v41_moe function. Launchers record
 * ownership and inject failure; no numerical kernels or model are simulated. */
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>

#define DS4_N_EMBD 256u
#define DS4_N_FF_EXP 256u
#define DS4_N_EXPERT 4u
#define DS4_N_EXPERT_USED 2u
#define DS4_N_LAYER 2u
#define DS4_V41_GEMV_MAX_TOK 8u
#define DS4_TENSOR_F32 0u
#define DS4_TENSOR_BF16 30u
#define DS4_SWIGLU_CLAMP_EXP 10.f
#define DS4_EXPERT_WEIGHT_SCALE 1.f

typedef struct { int unused; } ds4_gpu_tensor;
typedef struct { uint32_t type; uint64_t abs_offset, bytes; } ds4_tensor;
typedef struct { const void *map; uint64_t size; } ds4_model;
typedef struct {
    const ds4_tensor *ffn_gate_shexp, *ffn_up_shexp, *ffn_down_shexp;
    const ds4_tensor *ffn_gate_inp, *ffn_exp_probs_b, *ffn_exps_vq;
} ds4_layer_weights;
typedef struct {
    uint32_t n;
    int draft;
    /* Vision seam v41_moe reads (ds4_ds41_forward.inc:288-291). */
    ds4_gpu_tensor *vrows;
    int vrows_on;
    ds4_gpu_tensor *sg, *su, *sh, *so, *xn, *glog, *sel, *rw, *routed, *y;
    const ds4_tensor *tower_exps_vq[4];
    const uint64_t *tower_exp_off[4];
} ds4_v41_state;

static struct { uint32_t mtp_experts, mtp_used; } g_ds4_v41 = {4u, 2u};
static ds4_gpu_tensor *g_v41_bias_vl[DS4_N_LAYER];  /* ds4_ds41_vision.inc:22 */
static int g_steps, g_fail_step, g_fork_enabled, g_active, g_joins, g_begins;

static int step_ok(void) { return ++g_steps != g_fail_step; }
static int ds4_gpu_side_mark(void) { return g_fork_enabled; }
static int ds4_gpu_side_begin(void) { g_active = 1; g_begins++; return 1; }
static int ds4_gpu_side_main(void) { return 1; }
static int ds4_gpu_side_join(void) { g_active = 0; g_joins++; return 1; }

static bool v41_tproj(const ds4_model *m, ds4_gpu_tensor *out, const ds4_tensor *w,
                      uint64_t in_dim, uint64_t out_dim, const ds4_gpu_tensor *x,
                      uint32_t n, int round_out) { return step_ok(); }
static int ds4_gpu_v41_swiglu_tensor(ds4_gpu_tensor *out, ...) { return step_ok(); }
static int ds4_gpu_v41_matmul_f32_tensor(ds4_gpu_tensor *out, ...) { return step_ok(); }
static int ds4_gpu_v41_matmul_bf16_tensor(ds4_gpu_tensor *out, ...) { return step_ok(); }
static int ds4_gpu_v41_router_tensor(ds4_gpu_tensor *out, ...) { return step_ok(); }
static int ds4_gpu_v41_routed_moe_tensor(ds4_gpu_tensor *out, ...) { return step_ok(); }
static int ds4_gpu_v41_mtp_moe_tensor(ds4_gpu_tensor *out, ...) { return step_ok(); }
static int ds4_gpu_v41_moe_tail_tensor(ds4_gpu_tensor *out, ...) { return step_ok(); }
static int ds4_gpu_tensor_copy(ds4_gpu_tensor *out, ...) { return step_ok(); }
static int ds4_gpu_v41_add_tensor(ds4_gpu_tensor *out, ...) { return step_ok(); }
static int ds4_gpu_v41_round_bf16_tensor(ds4_gpu_tensor *out, ...) { return step_ok(); }

#include "ds41_moe_under_test.inc"

static int run_case(uint32_t gate_type, int fork_enabled, int fail_step) {
    const ds4_tensor weight = {DS4_TENSOR_BF16, 0u, 64u}, gate = {gate_type, 0u, 64u};
    const ds4_layer_weights layer = {&weight, &weight, &weight, &gate, &weight, &weight};
    const ds4_model model = {NULL, 64u};
    ds4_v41_state state = {0};
    state.n = 1u;
    g_steps = 0; g_fail_step = fail_step; g_fork_enabled = fork_enabled;
    g_active = 0; g_joins = 0; g_begins = 0;
    const bool ok = v41_moe(&model, &layer, &state, 0u);
    const int expected_joins = g_begins ? 1 : 0;
    if (ok != (fail_step == 0) || g_active || g_begins != fork_enabled || g_joins != expected_joins) {
        printf("FAIL gate=%u fork=%d fail_step=%d result=%d active=%d begins=%d joins=%d/%d\n",
               gate_type, fork_enabled, fail_step, ok, g_active, g_begins, g_joins, expected_joins);
        return 1;
    }
    return 0;
}

int main(void) {
    const uint32_t gate_types[] = {DS4_TENSOR_BF16, DS4_TENSOR_F32};
    int failures = 0, cases = 0;
    for (unsigned gate = 0; gate < sizeof gate_types / sizeof gate_types[0]; gate++) {
        for (int fork_enabled = 0; fork_enabled <= 1; fork_enabled++) {
            failures += run_case(gate_types[gate], fork_enabled, 0);
            const int steps = g_steps;
            cases++;
            for (int step = 1; step <= steps; step++) {
                failures += run_case(gate_types[gate], fork_enabled, step);
                cases++;
            }
        }
    }
    printf("ds41 MoE fork gate: %s (%d cases, %d failures)\n", failures ? "FAIL" : "PASS", cases, failures);
    return failures ? 1 : 0;
}
