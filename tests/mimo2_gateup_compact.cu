// SPDX-License-Identifier: MIT
// Exercise compact-input dispatch through the production IQ2 SoA entry.
// Build as mimo2_gateup_schedule.cu, adding -ldl. EXPECT_COMPACT is 0 or 1;
// compare PROBE_DUMP files from independent OFF/ON processes for parity.
#define main schedule_main
#include "mimo2_gateup_schedule.cu"
#undef main

#include <dlfcn.h>

namespace {
constexpr const char *COMPACT = "DS4_MIMO2_INPUT_Q8_COMPACT";
int compact_calls = 0, gathered_calls = 0;
#ifdef DS4_TEST_REFUSE_D2R
int refusals = 0;
#endif
}

extern "C" cudaError_t __cudaLaunchKernel(cudaKernel_t kernel, dim3 grid,
        dim3 block, void **args, size_t shared, cudaStream_t stream) {
    using Launch = cudaError_t (*)(cudaKernel_t, dim3, dim3, void **, size_t, cudaStream_t);
    static Launch launch = reinterpret_cast<Launch>(dlsym(RTLD_NEXT, "__cudaLaunchKernel"));
    if (!launch) { std::fprintf(stderr, "CUDA launch symbol missing\n"); std::exit(1); }
    const char *name = nullptr;
    check(cudaFuncGetName(&name, reinterpret_cast<const void *>(kernel)));
    if (name && std::strstr(name, "quantize_mmq_q8_1")) {
        const int32_t *ids = nullptr;
        std::memcpy(&ids, args[1], sizeof(ids));
        if (ids) { gathered_calls++; } else { compact_calls++; }
    }
    return launch(kernel, grid, block, args, shared, stream);
}

#ifdef DS4_TEST_REFUSE_D2R
// Link with --wrap=<pair-launch symbol from nm ds4_mmq_d2r.o>. Refusing both
// modes compares compact->sorted recovery against the same generic MMQ path.
extern "C" int refuse_pair(const void *, const void *, int64_t, const void *,
        const int32_t *, const int32_t *, float *, float *, int, int, int64_t,
        int, int, void *, size_t, cudaStream_t, const int32_t *, int)
    asm("__wrap__Z35ds4_mmq_iq2_xxs_moe_d2r_pair_launchPKvS0_lS0_PKiS2_PfS3_iiliiPvmP11CUstream_stS2_i");
extern "C" int refuse_pair(const void *, const void *, int64_t, const void *,
        const int32_t *, const int32_t *, float *, float *, int, int, int64_t,
        int, int, void *, size_t, cudaStream_t, const int32_t *, int) {
    refusals++;
    return 1;
}
#endif

int main(int argc, char **argv) {
    const char *expected = std::getenv("EXPECT_COMPACT");
    if (!expected || (std::strcmp(expected, "0") && std::strcmp(expected, "1"))) {
        std::fprintf(stderr, "set EXPECT_COMPACT=0 or 1\n");
        return 1;
    }
    const int rc = schedule_main(argc, argv);
    if (rc) { return rc; }
    const int reps = argc > 1 ? argument(argv[1], MAX_REPS) : DEFAULT_REPS;
    const int calls = WARMUPS + reps;
    const int want_compact = !std::strcmp(expected, "1") ? calls : 0;
#ifdef DS4_TEST_REFUSE_D2R
    const int want_gathered = calls;
    if (refusals != calls) { std::fprintf(stderr, "refusal coverage failed\n"); return 1; }
#else
    const int want_gathered = calls - want_compact;
#endif
    if (compact_calls != want_compact || gathered_calls != want_gathered) {
        std::fprintf(stderr, "%s: compact=%d gathered=%d expected compact=%d/%d\n",
                     COMPACT, compact_calls, gathered_calls, want_compact, calls);
        return 1;
    }
    std::printf("compact_dispatch_exact=true compact=%d gathered=%d\n", compact_calls, gathered_calls);
    return 0;
}
