// Fixed-topology attention residuals against the runtime-indexed fallback.
#include "cuda/inkling2_res.cuh"
#include <cassert>
#include <cmath>
#include <cstdio>
#include <vector>

static void check(cudaError_t e) {
    if (e != cudaSuccess) { fprintf(stderr, "%s\n", cudaGetErrorString(e)); abort(); }
}

template <typename T> static T *upload(const std::vector<T> &v) {
    T *p = nullptr;
    check(cudaMalloc(&p, v.size() * sizeof(T)));
    check(cudaMemcpy(p, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice));
    return p;
}

static void run(uint32_t rows) {
    const uint32_t cap = rows + 3;
    const size_t count = (size_t)rows * INK2_HIDDEN;
    std::vector<float> x(count), q(3 * INK2_HIDDEN);
    std::vector<uint16_t> blocks((size_t)DS4_INK2_BLOCKS * cap * INK2_HIDDEN), norms(q.size());
    for (size_t i = 0; i < x.size(); i++) { x[i] = __bfloat162float(__float2bfloat16(sinf(i % 101 * .07f))); }
    for (size_t i = 0; i < q.size(); i++) {
        q[i] = sinf(i % 151 * .09f);
        norms[i] = __bfloat16_as_ushort(__float2bfloat16(1.f + i % 17 * .01f));
    }
    for (size_t i = 0; i < blocks.size(); i++) {
        blocks[i] = __bfloat16_as_ushort(__float2bfloat16(cosf(i % 137 * .03f)));
    }
    float *dx = upload(x), *dq = upload(q), *ref = nullptr, *got = nullptr;
    uint16_t *db = upload(blocks), *dn = upload(norms);
    check(cudaMalloc(&ref, 3 * count * sizeof(float)));
    check(cudaMalloc(&got, 3 * count * sizeof(float)));
    for (uint32_t nb = 0; nb <= DS4_INK2_BLOCKS; nb++) {
        for (uint32_t nq : {0u, 1u, 3u}) {
            const float *query = nq ? dq : nullptr;
            const size_t size = count * (nq ? nq : 1u);
            ink2_attn_res_kernel<><<<rows, INK2_THREADS>>>(ref, ref + count, ref + 2 * count,
                db, nb, cap, dx, query, nq, dn, dn + INK2_HIDDEN, dn + 2 * INK2_HIDDEN);
            ink2_res_launch(got, got + count, got + 2 * count, db, nb, cap, dx, query, nq ? nq : 1u,
                dn, dn + INK2_HIDDEN, dn + 2 * INK2_HIDDEN, rows, nullptr);
            std::vector<float> a(size), b(size);
            check(cudaMemcpy(a.data(), ref, size * sizeof(float), cudaMemcpyDeviceToHost));
            check(cudaMemcpy(b.data(), got, size * sizeof(float), cudaMemcpyDeviceToHost));
            size_t different = 0;
            double error = 0, square = 0;
            for (size_t i = 0; i < size; i++) {
                assert(std::isfinite(a[i]) && std::isfinite(b[i]));
                different += a[i] != b[i];
                error += pow((double)a[i] - b[i], 2); square += (double)a[i] * a[i];
            }
            if (different) { printf("rows=%u blocks=%u queries=%u different=%zu rel_rms=%g\n", rows, nb, nq,
                different, sqrt(error / square)); }
            assert(different == 0);
        }
    }
    cudaFree(dx); cudaFree(dq); cudaFree(db); cudaFree(dn); cudaFree(ref); cudaFree(got);
    printf("attention residual rows=%u: all block/query widths equal\n", rows);
}

int main() {
    for (uint32_t rows : {1u, 2u, 4u, 5u, 17u, 2048u}) { run(rows); }
}
