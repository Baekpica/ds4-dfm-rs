/* Aligned mixed models use raw-span promotion beside replacement artifacts.
 * Already allocated spans must not remain unfunded in the next quote. */
#include "../ds4.c"

int main(void) {
    enum { SPAN = 1048576, TOTAL = 3 * SPAN, SETTLED = 16 * SPAN };
    unsigned char *source = xcalloc(TOTAL, 1);
    ds4_gov_modes_init();
    if (!ds4_gpu_init()) { return 1; }
    for (unsigned i = 0; i < 2; i++) {
        if (ds4_gpu_cache_model_range(source, TOTAL, i * SPAN, SPAN, "lease-test") != 1) {
            ds4_die("span was not promoted");
        }
        const ds4_gov_lease *lease = &g_gov_ledger.lease[DS4_GOVC_ENGINE_BOOT];
        if (lease->intent != (uint64_t)(i + 1) * SPAN || lease->resident != lease->intent) {
            ds4_die("populated raw spans remain unfunded in the boot lease");
        }
        if (ds4_gpu_cache_model_range(source, TOTAL, i * SPAN, SPAN, "dedup-test") != 2 ||
            lease->resident != (uint64_t)(i + 1) * SPAN) {
            ds4_die("deduplicated span changed committed bytes");
        }
    }
    ds4_gov_publish_use(DS4_GOVC_ENGINE_BOOT, SETTLED, SETTLED);
    ds4_gpu_model_plan_freeze();
    if (ds4_gpu_cache_model_range(source, TOTAL, 2 * SPAN, SPAN, "late-test") != 1 ||
        g_gov_ledger.lease[DS4_GOVC_ENGINE_BOOT].resident != SETTLED) {
        ds4_die("late span promotion replaced the settled boot census");
    }
    ds4_gpu_cleanup();
    free(source);
    puts("CUDA raw span lease: incremental commit and dedup PASS");
    return 0;
}
