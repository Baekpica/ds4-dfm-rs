// Fixed-topology attention residuals against the runtime-indexed fallback,
// and the decode rows' multi-CTA split against it within one BF16 step (its
// F32 sums run in another order).
#include "cuda/inkling2_res.cuh"
#include <algorithm>
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

static void run_split(uint32_t rows) {
    enum { PER_MILLION = 2000 };
    constexpr float CANCEL = 1e-6f;
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
        blocks[i] = __bfloat16_as_ushort(__float2bfloat16(cosf(i % 137 * .03f) + .2f * sinf(i % 4099 * .011f)));
    }
    float *dx = upload(x), *dq = upload(q), *ref = nullptr, *got = nullptr, *scratch = nullptr;
    uint16_t *db = upload(blocks), *dn = upload(norms);
    check(cudaMalloc(&ref, 3 * count * sizeof(float)));
    check(cudaMalloc(&got, 3 * count * sizeof(float)));
    check(cudaMalloc(&scratch, INK2_RES_SCRATCH * sizeof(float)));
    size_t moved = 0, total = 0;
    for (uint32_t nb = 0; nb <= DS4_INK2_BLOCKS; nb++) {
        for (uint32_t nq : {1u, 3u}) {
            const size_t size = count * nq;
            ink2_attn_res_kernel<><<<rows, INK2_THREADS>>>(ref, ref + count, ref + 2 * count,
                db, nb, cap, dx, dq, nq, dn, dn + INK2_HIDDEN, dn + 2 * INK2_HIDDEN);
            const bool taken = ink2_res_split(scratch, got, got + count, got + 2 * count, db, nb, cap, dx, dq, nq,
                dn, dn + INK2_HIDDEN, dn + 2 * INK2_HIDDEN, rows, nullptr);
            assert(taken);
            std::vector<float> a(size), b(size);
            check(cudaMemcpy(a.data(), ref, size * sizeof(float), cudaMemcpyDeviceToHost));
            check(cudaMemcpy(b.data(), got, size * sizeof(float), cudaMemcpyDeviceToHost));
            float top = 0.0f;
            for (float v : a) { top = std::max(top, std::fabs(v)); }
            for (size_t i = 0; i < size; i++) {
                assert(std::isfinite(a[i]) && std::isfinite(b[i]));
                if (a[i] == b[i]) { continue; }
                moved++;
                // One step of the larger value's BF16 grid; a weighted sum
                // that cancels to near zero keeps F32's absolute error.
                const float mag = std::max(std::fabs(a[i]), std::fabs(b[i]));
                const float bound = std::max(std::ldexp(1.0f, std::ilogb(mag) - 7), CANCEL * top);
                if (std::fabs(a[i] - b[i]) > bound) {
                    printf("rows=%u blocks=%u queries=%u at %zu: %.9g vs %.9g\n", rows, nb, nq, i, a[i], b[i]);
                }
                assert(std::fabs(a[i] - b[i]) <= bound);
            }
            total += size;
        }
    }
    assert(moved * 1000000 <= (size_t)PER_MILLION * total);
    cudaFree(dx); cudaFree(dq); cudaFree(db); cudaFree(dn); cudaFree(ref); cudaFree(got); cudaFree(scratch);
    printf("attention residual split rows=%u: %zu of %zu moved one BF16 step\n", rows, moved, total);
}

int main() {
    setvbuf(stdout, nullptr, _IOLBF, 0);   // a failed assertion keeps the lines before it
    for (uint32_t rows : {1u, 2u, 4u, 5u, 17u, 2048u}) { run(rows); }
    for (uint32_t rows : {1u, 3u, 8u}) { run_split(rows); }
}
