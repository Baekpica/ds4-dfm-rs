/* Model-free admission gate: the GLM optimization cannot promote generic Q2. */
#include "../cuda/mmq/ds4_glm_q2.h"
#include <cstdio>
#include <cstdlib>
#include <cstdint>

#define CHECK(x) do { if (!(x)) { \
    std::fprintf(stderr, "GLM Q2 policy FAIL %d: %s\n", __LINE__, #x); std::exit(1); \
} } while (0)

static MoePolicy select(int cc = 1210, uint32_t type = 10, int m = 4096,
        int k = 2048, int rows = 1024, int experts = 3389, int used = 1,
        uint64_t stride = 2753688, MoeLayout layout = MoeLayout::Raw) {
    return glm_q2_policy(cc, type, m, k, rows, experts, used, stride, layout);
}

int main() {
    CHECK(select() == MoePolicy::GlmQ2Down);
    CHECK(select(1210, 10, 4096, 2048, 256, 32, 1, 2752512) == MoePolicy::GlmQ2Down);
    CHECK(select(1200) == MoePolicy::Generic);
    CHECK(select(900) == MoePolicy::Generic);
    CHECK(select(1210, 17) == MoePolicy::Generic);
    CHECK(select(1210, 10, 2048) == MoePolicy::Generic);
    CHECK(select(1210, 10, 4096, 4096) == MoePolicy::Generic);
    CHECK(select(1210, 10, 4096, 2048, 255) == MoePolicy::Generic);
    CHECK(select(1210, 10, 4096, 2048, 1024, 31) == MoePolicy::Generic);
    CHECK(select(1210, 10, 4096, 2048, 1024, 3389, 8) == MoePolicy::Generic);
    CHECK(select(1210, 10, 4096, 2048, 1024, 3389, 1, 2753688, MoeLayout::SoA) == MoePolicy::Generic);
    CHECK(select(1210, 10, 4096, 2048, 1024, 3389, 1, 0) == MoePolicy::Generic);
    CHECK(select(1210, 10, 4096, 2048, 1024, 3389, 1, 2752512 - 84) == MoePolicy::Generic);
    CHECK(select(1210, 10, 4096, 2048, 1024, 3389, 1, 2753689) == MoePolicy::Generic);
    CHECK(select(1210, 10, 4096, 2048, 524288) == MoePolicy::Generic);
    CHECK(select(1210, 10, 4096, 2048, 1024, 65535, 1, 2752512) == MoePolicy::GlmQ2Down);
    CHECK(select(1210, 10, 4096, 2048, 1024, 65536, 1, 2752512) == MoePolicy::Generic);
    CHECK(select(1210, 10, 4096, 2048, 1024, 100000) == MoePolicy::Generic);
    CHECK(select(1210, 10, 4096, 2048, 1024, 3389, 1, UINT64_MAX) == MoePolicy::Generic);
    std::puts("GLM Q2 policy PASS (Spark/raw/type/shape/width/stride/complete spans)");
}
