/* Execute the exact native setter against a recording backend. NULL is the
 * backend's clear operation, so admission must preserve that command. */
#include <stdint.h>
#include <stdio.h>

#define DS4_N_LAYER 2u
#define DS4_VARIANT_V41 41u
#define DS4_MODEL_VARIANT g_variant

static uint32_t g_variant;
static int g_calls, g_result;
static uint32_t g_layer, g_experts, g_embd;
static const float *g_data;

static int ds4_gpu_v41_set_gr_override(uint32_t il, const float *data,
                                      uint32_t n_expert, uint32_t n_embd) {
    g_calls++;
    g_layer = il; g_data = data; g_experts = n_expert; g_embd = n_embd;
    return g_result;
}

#include "ds41_gr_under_test.inc"

static int run_case(const char *name, uint32_t variant, uint32_t layer,
                     const float *data, uint32_t experts, uint32_t embd,
                     int backend_result, int expected_calls, int expected_result) {
    g_variant = variant; g_calls = 0; g_result = backend_result;
    const int result = ds4_v41_zchain_gr_set(NULL, layer, data, experts, embd);
    const int forwarded = !expected_calls ||
        (g_layer == layer && g_data == data && g_experts == experts && g_embd == embd);
    if (result != expected_result || g_calls != expected_calls || !forwarded) {
        printf("FAIL %s: result=%d/%d calls=%d/%d forwarded=%d\n",
               name, result, expected_result, g_calls, expected_calls, forwarded);
        return 1;
    }
    return 0;
}

int main(void) {
    const float data[] = {1.0f};
    int failures = 0;
    failures += run_case("upload", DS4_VARIANT_V41, 0u, data, 4u, 8u, 1, 1, 1);
    failures += run_case("clear", DS4_VARIANT_V41, 0u, NULL, 0u, 0u, 1, 1, 1);
    failures += run_case("clear shape", DS4_VARIANT_V41, 1u, NULL, 4u, 8u, 1, 1, 1);
    failures += run_case("upload failure", DS4_VARIANT_V41, 0u, data, 4u, 8u, 0, 1, 0);
    failures += run_case("clear failure", DS4_VARIANT_V41, 0u, NULL, 0u, 0u, 0, 1, 0);
    failures += run_case("layer overflow", DS4_VARIANT_V41, DS4_N_LAYER, data, 4u, 8u, 1, 0, 0);
    failures += run_case("clear overflow", DS4_VARIANT_V41, DS4_N_LAYER, NULL, 0u, 0u, 1, 0, 0);
    failures += run_case("wrong variant", 0u, 0u, data, 4u, 8u, 1, 0, 0);
    failures += run_case("wrong variant clear", 0u, 0u, NULL, 0u, 0u, 1, 0, 0);
    printf("ds41 GR forwarding gate: %s (%d failures)\n", failures ? "FAIL" : "PASS", failures);
    return failures ? 1 : 0;
}
