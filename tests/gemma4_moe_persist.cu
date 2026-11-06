/* Persistent layer-book MoE decode gate (cuda/gemma4_moe_persist.cuh).
 *
 * One synthetic R13+8 layer at the artifact's exact geometry: one 8192-word
 * E4M3 main book, the layer's 256-word gate/up residual book, its 256-word
 * down residual book, and eight routed experts of 704 x 2816 gate/up and
 * 2816 x 704 down records. The persistent gate/up and down kernels run
 * beside the kernels the per-expert loop uses today (g4_vq_gemv_glu and
 * g4_vq_gemv), so the numbers are the parity of the new pass against the
 * path it replaces, not against a reimplementation:
 *
 *   gu    persistent warp-per-row against g4_vq_gemv_glu's block-per-row:
 *         the row sum's FP32 order differs (Class B), the epilogue does not;
 *   down  the same input row fed to both: expected bit-identical, because
 *         the per-expert dot keeps the warp<8> lane walk and reduction;
 *   full  persistent gu -> down against the reference gu -> down -> the
 *         scatter's one rounded add per expert in ascending expert order.
 *
 * The route ids are deliberately scrambled and the weights distinct, so a
 * kernel that summed in slot order instead of ascending expert order would
 * show up in the exact count.
 *
 * build: make tests/gemma4_moe_persist
 * run:   ./tests/gemma4_moe_persist
 */
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <algorithm>
#include "../ds4_gemma4_plan.h"
#include "../cuda/gemma4_moe_persist.cuh"
#include "../cuda/gemma4_moe_fused.cuh"

#define CUDA_OK(call) do { cudaError_t e_=(call); if(e_!=cudaSuccess){ \
    std::fprintf(stderr,"%s: %s\n",#call,cudaGetErrorString(e_)); std::exit(2); } } while(0)
static unsigned failures = 0;

static float bf16r(float x) { return __bfloat162float(__float2bfloat16_rn(x)); }

struct rng {
    uint32_t s;
    uint32_t next() { s ^= s << 13; s ^= s >> 17; s ^= s << 5; return s; }
};

static uint8_t *upload(const void *p, size_t n) {
    uint8_t *d = nullptr;
    CUDA_OK(cudaMalloc(&d, n));
    CUDA_OK(cudaMemcpy(d, p, n, cudaMemcpyHostToDevice));
    return d;
}

/* One synthetic GVQ1 matrix record. The 12-bit index stream is packed the
 * way g4_vq_index reads it (cuda/gemma_vq_decode.cuh:58-66): two indices per
 * three bytes, least significant bit first, with the plane byte carrying the
 * 13th bit of eight vectors. */
struct gmat {
    std::vector<uint8_t> main, plane, res;
    std::vector<__half> gain;
    uint8_t *d_main = nullptr, *d_plane = nullptr, *d_res = nullptr;
    __half *d_gain = nullptr;
    uint32_t rows = 0, cols = 0;
};

static gmat make_mat(uint32_t rows, uint32_t cols, rng &r) {
    gmat m;
    m.rows = rows;
    m.cols = cols;
    const uint64_t vectors = (uint64_t)rows * cols / G4_VQ_VECTOR;
    if (vectors % 2u) { std::fprintf(stderr, "odd vector count\n"); std::exit(2); }
    m.main.resize(vectors * 12 / 8);
    m.plane.resize(vectors / 8);
    m.res.resize(vectors);
    m.gain.resize(rows);
    for (uint64_t v = 0; v < vectors; v += 2) {
        const uint32_t i0 = r.next() & 0xFFFu, i1 = r.next() & 0xFFFu;
        const uint64_t at = v / 2 * 3;
        m.main[at] = (uint8_t)i0;
        m.main[at + 1] = (uint8_t)(((i0 >> 8) & 15u) | ((i1 & 15u) << 4));
        m.main[at + 2] = (uint8_t)(i1 >> 4);
    }
    for (auto &b : m.plane) { b = (uint8_t)r.next(); }
    for (auto &b : m.res) { b = (uint8_t)r.next(); }
    for (auto &h : m.gain) { h = __float2half_rn(0.02f + (float)(r.next() % 1000u) * 1e-5f); }
    m.d_main = upload(m.main.data(), m.main.size());
    m.d_plane = upload(m.plane.data(), m.plane.size());
    m.d_res = upload(m.res.data(), m.res.size());
    m.d_gain = (__half *)upload(m.gain.data(), m.gain.size() * sizeof(__half));
    return m;
}

static glmvq_matrix desc(const gmat &m, const uint8_t *book, const uint8_t *res_book) {
    glmvq_matrix q{};
    q.main = m.d_main;
    q.plane = m.d_plane;
    q.book = book;
    q.gain = m.d_gain;
    q.rows = m.rows;
    q.cols = m.cols;
    q.precision = GLMVQ_E4M3;
    q.res = m.d_res;
    q.res_book = res_book;
    return q;
}

static std::vector<uint8_t> make_book(uint32_t words, rng &r) {
    std::vector<uint8_t> b((size_t)words * G4_VQ_VECTOR);
    for (auto &x : b) {
        uint32_t v = r.next() & 0x7Fu;
        if (v == 0x7Fu) { v = 0x7Eu; }   /* E4M3 FN: 0x7F/0xFF are NaN */
        x = (uint8_t)(v | ((r.next() >> 8) & 0x80u));
    }
    return b;
}

static void report(const char *name, const float *got, const float *want, uint32_t count,
        double rel_limit) {
    double num = 0.0, den = 0.0, mx = 0.0;
    uint32_t same = 0;
    for (uint32_t i = 0; i < count; i++) {
        const double d = (double)got[i] - (double)want[i];
        num += d * d;
        den += (double)want[i] * want[i];
        if (std::fabs(d) > mx) { mx = std::fabs(d); }
        same += got[i] == want[i];
    }
    const double rel = std::sqrt(num / (den > 0.0 ? den : 1.0));
    std::printf("{\"test\":\"%s\",\"relL2\":%.9g,\"max_error\":%.9g,\"exact\":%u,"
        "\"count\":%u,\"limit\":%.3g}\n", name, rel, mx, same, count, rel_limit);
    if (!(rel <= rel_limit)) {
        std::fprintf(stderr, "%s: relL2 %.3e above %.3e\n", name, rel, rel_limit);
        failures++;
    }
}

static float *read_f32(const void *dev, uint32_t count) {
    float *h = (float *)std::malloc((size_t)count * sizeof(float));
    CUDA_OK(cudaMemcpy(h, dev, (size_t)count * sizeof(float), cudaMemcpyDeviceToHost));
    return h;
}

int main(void) {
    rng r{0x9e3779b9u};
    const uint32_t used = G4_USED, ff = G4_FF, embed = G4_EMBED;
    const uint32_t selected[G4_USED] = {5, 120, 17, 99, 3, 42, 60, 8};
    const float weights[G4_USED] = {0.31f, 0.77f, 0.12f, 0.55f, 0.93f, 0.24f, 0.68f, 0.41f};

    std::vector<uint8_t> book = make_book(G4_WORDS, r);
    std::vector<uint8_t> res_gu = make_book(256, r), res_down = make_book(256, r);
    uint8_t *d_book = upload(book.data(), book.size());
    uint8_t *d_res_gu = upload(res_gu.data(), res_gu.size());
    uint8_t *d_res_down = upload(res_down.data(), res_down.size());

    std::vector<gmat> gate(used), up(used), down(used);
    for (uint32_t e = 0; e < used; e++) {
        gate[e] = make_mat(ff, embed, r);
        up[e] = make_mat(ff, embed, r);
        down[e] = make_mat(embed, ff, r);
    }
    std::vector<glmvq_matrix> table(3u * G4_EXPERTS);
    for (uint32_t i = 0; i < G4_EXPERTS; i++) {
        table[i] = desc(gate[0], d_book, d_res_gu);
        table[G4_EXPERTS + i] = desc(up[0], d_book, d_res_gu);
        table[2u * G4_EXPERTS + i] = desc(down[0], d_book, d_res_down);
    }
    for (uint32_t e = 0; e < used; e++) {
        table[selected[e]] = desc(gate[e], d_book, d_res_gu);
        table[G4_EXPERTS + selected[e]] = desc(up[e], d_book, d_res_gu);
        table[2u * G4_EXPERTS + selected[e]] = desc(down[e], d_book, d_res_down);
    }
    glmvq_matrix *d_table = (glmvq_matrix *)upload(table.data(),
        table.size() * sizeof(glmvq_matrix));
    uint32_t *d_selected = (uint32_t *)upload(selected, sizeof(selected));
    float *d_weights = (float *)upload(weights, sizeof(weights));
    __nv_bfloat16 *d_pmid = nullptr;
    CUDA_OK(cudaMalloc(&d_pmid, (size_t)used * ff * sizeof(__nv_bfloat16)));
    float *d_out = nullptr;
    CUDA_OK(cudaMalloc(&d_out, (size_t)embed * sizeof(float)));

    std::vector<float> input(embed);
    for (uint32_t i = 0; i < embed; i++) {
        input[i] = std::sin((i + 1) * 0.117f) + 0.3f * std::cos((i + 3) * 0.031f);
    }
    float *d_input = (float *)upload(input.data(), input.size() * sizeof(float));

    /* The reference: the per-expert loop's two GEMVs, slot order. */
    std::vector<float *> d_mid_ref(used), d_out_ref(used);
    for (uint32_t e = 0; e < used; e++) {
        CUDA_OK(cudaMalloc(&d_mid_ref[e], (size_t)ff * sizeof(float)));
        CUDA_OK(cudaMalloc(&d_out_ref[e], (size_t)embed * sizeof(float)));
        if (!g4_vq_gemv_glu(table[selected[e]], table[G4_EXPERTS + selected[e]], d_input,
                d_mid_ref[e], G4_VQ_AUTO, 256, 0) ||
            !g4_vq_gemv(table[2u * G4_EXPERTS + selected[e]], d_mid_ref[e], d_out_ref[e],
                G4_VQ_AUTO, 256, 0)) {
            std::fprintf(stderr, "reference kernels failed\n");
            return 2;
        }
    }
    CUDA_OK(cudaDeviceSynchronize());

    /* Ascending expert order, ties by slot: the scatter's order. */
    std::vector<uint32_t> sorted(used);
    for (uint32_t i = 0; i < used; i++) { sorted[i] = i; }
    std::stable_sort(sorted.begin(), sorted.end(), [&](uint32_t a, uint32_t b) {
        return selected[a] < selected[b];
    });

    int sms = 48;
    CUDA_OK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0));
    const uint32_t ctas = (uint32_t)sms;
    CUDA_OK(cudaFuncSetAttribute(g4_moe_gu_kernel<G4_MOE_WARPS>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, G4_MOE_GU_STAGE));
    CUDA_OK(cudaFuncSetAttribute(g4_moe_down_kernel<G4_MOE_WARPS>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, G4_MOE_DOWN_STAGE));

    /* A: the gate/up pass alone. The persistent row e is the e-th expert in
     * ascending order; the reference row is that expert's slot. */
    g4_moe_gu_kernel<G4_MOE_WARPS><<<ctas, G4_MOE_THREADS, G4_MOE_GU_STAGE>>>(d_table, 0,
        d_selected, d_input, d_pmid);
    CUDA_OK(cudaGetLastError());
    CUDA_OK(cudaDeviceSynchronize());
    {
        std::vector<__nv_bfloat16> got(used * ff);
        CUDA_OK(cudaMemcpy(got.data(), d_pmid, got.size() * sizeof(__nv_bfloat16),
            cudaMemcpyDeviceToHost));
        std::vector<float> a(used * ff), b(used * ff);
        for (uint32_t e = 0; e < used; e++) {
            const float *ref = read_f32(d_mid_ref[sorted[e]], ff);
            for (uint32_t row = 0; row < ff; row++) {
                a[e * ff + row] = __bfloat162float(got[e * ff + row]);
                b[e * ff + row] = ref[row];
            }
            std::free((void *)ref);
        }
        report("gu", a.data(), b.data(), used * ff, 1e-5);
    }

    /* B: the down pass on the reference's own mid rows, one expert weighted
     * 1 and the rest 0: expected bit-identical to the reference down + the
     * scatter's rounded add. */
    {
        std::vector<__nv_bfloat16> pack(used * ff);
        for (uint32_t e = 0; e < used; e++) {
            const float *ref = read_f32(d_mid_ref[sorted[e]], ff);
            for (uint32_t row = 0; row < ff; row++) {
                pack[e * ff + row] = __float2bfloat16_rn(ref[row]);
            }
            std::free((void *)ref);
        }
        __nv_bfloat16 *d_pack = (__nv_bfloat16 *)upload(pack.data(),
            pack.size() * sizeof(__nv_bfloat16));
        float one[G4_USED] = {0.0f};
        one[sorted[0]] = 1.0f;
        float *d_one = (float *)upload(one, sizeof(one));
        g4_moe_down_kernel<G4_MOE_WARPS><<<ctas, G4_MOE_THREADS, G4_MOE_DOWN_STAGE>>>(d_table, 0,
            d_selected, d_pack, d_one, d_out);
        CUDA_OK(cudaGetLastError());
        CUDA_OK(cudaDeviceSynchronize());
        const float *ref = read_f32(d_out_ref[sorted[0]], embed);
        std::vector<float> got(embed), want(embed);
        CUDA_OK(cudaMemcpy(got.data(), d_out, (size_t)embed * sizeof(float),
            cudaMemcpyDeviceToHost));
        for (uint32_t col = 0; col < embed; col++) { want[col] = bf16r(ref[col]); }
        report("down", got.data(), want.data(), embed, 0.0);
        std::free((void *)ref);
        CUDA_OK(cudaFree(d_pack));
        CUDA_OK(cudaFree(d_one));
    }

    /* C: the whole pass against the reference loop and the scatter. */
    g4_moe_gu_kernel<G4_MOE_WARPS><<<ctas, G4_MOE_THREADS, G4_MOE_GU_STAGE>>>(d_table, 0,
        d_selected, d_input, d_pmid);
    g4_moe_down_kernel<G4_MOE_WARPS><<<ctas, G4_MOE_THREADS, G4_MOE_DOWN_STAGE>>>(d_table, 0,
        d_selected, d_pmid, d_weights, d_out);
    CUDA_OK(cudaGetLastError());
    CUDA_OK(cudaDeviceSynchronize());
    {
        std::vector<float> want(embed, 0.0f), got(embed);
        for (uint32_t e = 0; e < used; e++) {
            const uint32_t slot = sorted[e];
            const float *ref = read_f32(d_out_ref[slot], embed);
            for (uint32_t col = 0; col < embed; col++) {
                want[col] = bf16r(want[col] + ref[col] * weights[slot]);
            }
            std::free((void *)ref);
        }
        CUDA_OK(cudaMemcpy(got.data(), d_out, (size_t)embed * sizeof(float),
            cudaMemcpyDeviceToHost));
        report("full", got.data(), want.data(), embed, 1e-5);
    }

    /* Indicative timing on this device: the reference launches 16 kernels
     * per layer, the persistent pass 2. */
    {
        const int iters = 20;
        cudaEvent_t a, b;
        CUDA_OK(cudaEventCreate(&a));
        CUDA_OK(cudaEventCreate(&b));
        CUDA_OK(cudaEventRecord(a));
        for (int it = 0; it < iters; it++) {
            for (uint32_t e = 0; e < used; e++) {
                g4_vq_gemv_glu(table[selected[e]], table[G4_EXPERTS + selected[e]], d_input,
                    d_mid_ref[e], G4_VQ_AUTO, 256, 0);
                g4_vq_gemv(table[2u * G4_EXPERTS + selected[e]], d_mid_ref[e], d_out_ref[e],
                    G4_VQ_AUTO, 256, 0);
            }
        }
        CUDA_OK(cudaEventRecord(b));
        CUDA_OK(cudaEventSynchronize(b));
        float ref_ms = 0.0f;
        CUDA_OK(cudaEventElapsedTime(&ref_ms, a, b));
        CUDA_OK(cudaEventRecord(a));
        for (int it = 0; it < iters; it++) {
            g4_moe_gu_kernel<G4_MOE_WARPS><<<ctas, G4_MOE_THREADS, G4_MOE_GU_STAGE>>>(d_table, 0,
                d_selected, d_input, d_pmid);
            g4_moe_down_kernel<G4_MOE_WARPS><<<ctas, G4_MOE_THREADS, G4_MOE_DOWN_STAGE>>>(d_table, 0,
                d_selected, d_pmid, d_weights, d_out);
        }
        CUDA_OK(cudaEventRecord(b));
        CUDA_OK(cudaEventSynchronize(b));
        float persist_ms = 0.0f;
        CUDA_OK(cudaEventElapsedTime(&persist_ms, a, b));
        std::printf("{\"test\":\"timing\",\"reference_ms\":%.4f,\"persistent_ms\":%.4f,"
            "\"speedup\":%.3f,\"sms\":%d}\n", ref_ms / iters, persist_ms / iters,
            ref_ms / persist_ms, sms);
        CUDA_OK(cudaEventDestroy(a));
        CUDA_OK(cudaEventDestroy(b));
    }

    std::printf("{\"failures\":%u}\n", failures);
    return failures ? 1 : 0;
}
