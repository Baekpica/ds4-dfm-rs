// Scalar oracles for the MiMo media kernels. Not a weight load.
#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include "../cuda/mimo2_media.cuh"

static void check(cudaError_t code) {
    if (code != cudaSuccess) {
        fprintf(stderr, "%s\n", cudaGetErrorString(code));
        exit(1);
    }
}

static int close_enough(const std::vector<float> &got, const std::vector<float> &want, float tol, const char *name) {
    float max_error = 0.f;
    if (got.size() != want.size()) { return 2; }
    for (size_t i = 0; i < got.size(); i++) {
        if (!std::isfinite(got[i]) || !std::isfinite(want[i])) { return 3; }
        max_error = fmaxf(max_error, fabsf(got[i] - want[i]));
    }
    printf("%s max_abs_error=%.9g\n", name, max_error);
    return max_error > tol ? 4 : 0;
}

static int test_patch() {
    const int n = 1, oc = 3, ic = 1, kt = 2, p = 2, width = ic * kt * p * p;
    std::vector<float> in(width), w0(oc * ic * p * p), w1(w0.size()), got(oc), want(oc);
    for (int i = 0; i < width; i++) { in[i] = 0.1f * (i + 1); }
    for (size_t i = 0; i < w0.size(); i++) {
        w0[i] = 0.01f * (int)i - 0.05f;
        w1[i] = -0.02f * (int)i;
    }
    for (int col = 0; col < oc; col++) {
        float acc = 0.f;
        for (int c = 0; c < ic; c++) {
            for (int t = 0; t < kt; t++) {
                const float *w = t ? w1.data() : w0.data();
                for (int kh = 0; kh < p; kh++) {
                    for (int kw = 0; kw < p; kw++) {
                        acc += in[((c * kt + t) * p + kh) * p + kw] *
                               w[((col * ic + c) * p + kh) * p + kw];
                    }
                }
            }
        }
        want[col] = acc;
    }
    float *d_in, *d_w0, *d_w1, *d_out;
    check(cudaMalloc(&d_in, in.size() * 4));
    check(cudaMalloc(&d_w0, w0.size() * 4));
    check(cudaMalloc(&d_w1, w1.size() * 4));
    check(cudaMalloc(&d_out, got.size() * 4));
    check(cudaMemcpy(d_in, in.data(), in.size() * 4, cudaMemcpyHostToDevice));
    check(cudaMemcpy(d_w0, w0.data(), w0.size() * 4, cudaMemcpyHostToDevice));
    check(cudaMemcpy(d_w1, w1.data(), w1.size() * 4, cudaMemcpyHostToDevice));
    mimo2_patch<<<1, 32>>>(d_out, d_in, d_w0, d_w1, n, oc, ic, kt, p);
    check(cudaDeviceSynchronize());
    check(cudaMemcpy(got.data(), d_out, got.size() * 4, cudaMemcpyDeviceToHost));
    cudaFree(d_in); cudaFree(d_w0); cudaFree(d_w1); cudaFree(d_out);
    return close_enough(got, want, 1e-5f, "patch");
}

static int test_rope() {
    const int n = 2, heads = 1, hd = 8, stride = hd;
    std::vector<float> x(n * hd), cosv(n * hd), sinv(n * hd), want(n * hd);
    for (int i = 0; i < n * hd; i++) {
        x[i] = 0.25f * (i - 3);
        cosv[i] = cosf(0.1f * i);
        sinv[i] = sinf(0.1f * i);
    }
    for (int row = 0; row < n; row++) {
        for (int d = 0; d < hd; d++) {
            const float rot = d < hd / 2 ? -x[row * hd + d + hd / 2] : x[row * hd + d - hd / 2];
            want[row * hd + d] = x[row * hd + d] * cosv[row * hd + d] + rot * sinv[row * hd + d];
        }
    }
    float *d_x, *d_c, *d_s;
    check(cudaMalloc(&d_x, x.size() * 4));
    check(cudaMalloc(&d_c, cosv.size() * 4));
    check(cudaMalloc(&d_s, sinv.size() * 4));
    check(cudaMemcpy(d_x, x.data(), x.size() * 4, cudaMemcpyHostToDevice));
    check(cudaMemcpy(d_c, cosv.data(), cosv.size() * 4, cudaMemcpyHostToDevice));
    check(cudaMemcpy(d_s, sinv.data(), sinv.size() * 4, cudaMemcpyHostToDevice));
    mimo2_rope<<<1, 32>>>(d_x, d_c, d_s, n, heads, hd, stride, 0);
    check(cudaDeviceSynchronize());
    check(cudaMemcpy(x.data(), d_x, x.size() * 4, cudaMemcpyDeviceToHost));
    cudaFree(d_x); cudaFree(d_c); cudaFree(d_s);
    return close_enough(x, want, 1e-5f, "rope");
}

static int test_attn() {
    const int n = 4, qh = 2, kv = 1, hd = 4, window = 1;
    const int qw = qh * hd, kw = kv * hd;
    std::vector<float> q(n * qw), k(n * kw), v(n * kw), sinks(qh), got(n * qw), want(n * qw);
    for (int i = 0; i < (int)q.size(); i++) { q[i] = 0.05f * ((i % 7) - 3); }
    for (int i = 0; i < (int)k.size(); i++) { k[i] = 0.07f * ((i % 5) - 2); v[i] = 0.03f * (i % 4); }
    sinks[0] = 0.2f; sinks[1] = -0.4f;
    const float scale = 1.f / sqrtf((float)hd);
    for (int row = 0; row < n; row++) {
        for (int head = 0; head < qh; head++) {
            float maxv = sinks[head];
            for (int key = 0; key < n; key++) {
                if (abs(row - key) > window) { continue; }
                float dot = 0.f;
                for (int d = 0; d < hd; d++) {
                    dot += q[(row * qh + head) * hd + d] * k[key * hd + d];
                }
                maxv = fmaxf(maxv, dot * scale);
            }
            float sum = expf(sinks[head] - maxv);
            for (int key = 0; key < n; key++) {
                if (abs(row - key) > window) { continue; }
                float dot = 0.f;
                for (int d = 0; d < hd; d++) {
                    dot += q[(row * qh + head) * hd + d] * k[key * hd + d];
                }
                sum += expf(dot * scale - maxv);
            }
            for (int d = 0; d < hd; d++) { want[(row * qh + head) * hd + d] = 0.f; }
            for (int key = 0; key < n; key++) {
                if (abs(row - key) > window) { continue; }
                float dot = 0.f;
                for (int d = 0; d < hd; d++) {
                    dot += q[(row * qh + head) * hd + d] * k[key * hd + d];
                }
                const float w = expf(dot * scale - maxv) / sum;
                for (int d = 0; d < hd; d++) {
                    want[(row * qh + head) * hd + d] += w * v[key * hd + d];
                }
            }
        }
    }
    float *d_q, *d_k, *d_v, *d_s, *d_o;
    check(cudaMalloc(&d_q, q.size() * 4));
    check(cudaMalloc(&d_k, k.size() * 4));
    check(cudaMalloc(&d_v, v.size() * 4));
    check(cudaMalloc(&d_s, sinks.size() * 4));
    check(cudaMalloc(&d_o, got.size() * 4));
    check(cudaMemcpy(d_q, q.data(), q.size() * 4, cudaMemcpyHostToDevice));
    check(cudaMemcpy(d_k, k.data(), k.size() * 4, cudaMemcpyHostToDevice));
    check(cudaMemcpy(d_v, v.data(), v.size() * 4, cudaMemcpyHostToDevice));
    check(cudaMemcpy(d_s, sinks.data(), sinks.size() * 4, cudaMemcpyHostToDevice));
    mimo2_attn<<<1, 32>>>(d_o, d_q, d_k, d_v, d_s, n, qh, kv, hd, qw, kw, kw, 0, 0, 0, window, 0, 0);
    check(cudaDeviceSynchronize());
    check(cudaMemcpy(got.data(), d_o, got.size() * 4, cudaMemcpyDeviceToHost));
    cudaFree(d_q); cudaFree(d_k); cudaFree(d_v); cudaFree(d_s); cudaFree(d_o);
    return close_enough(got, want, 1e-5f, "attn");
}

static int test_group() {
    const int n = 8, heads = 1, hd = 2, group = 4;
    std::vector<float> q(n * hd), k(n * hd), v(n * hd), got(n * hd), want(n * hd);
    for (int i = 0; i < n * hd; i++) { q[i] = 0.1f * i; k[i] = 0.2f; v[i] = (float)(i / hd); }
    const float scale = 1.f / sqrtf((float)hd);
    for (int row = 0; row < n; row++) {
        float maxv = -INFINITY;
        for (int key = 0; key < n; key++) {
            if (row / group != key / group) { continue; }
            float dot = 0.f;
            for (int d = 0; d < hd; d++) { dot += q[row * hd + d] * k[key * hd + d]; }
            maxv = fmaxf(maxv, dot * scale);
        }
        float sum = 0.f;
        for (int key = 0; key < n; key++) {
            if (row / group != key / group) { continue; }
            float dot = 0.f;
            for (int d = 0; d < hd; d++) { dot += q[row * hd + d] * k[key * hd + d]; }
            sum += expf(dot * scale - maxv);
        }
        for (int d = 0; d < hd; d++) { want[row * hd + d] = 0.f; }
        for (int key = 0; key < n; key++) {
            if (row / group != key / group) { continue; }
            float dot = 0.f;
            for (int d = 0; d < hd; d++) { dot += q[row * hd + d] * k[key * hd + d]; }
            const float w = expf(dot * scale - maxv) / sum;
            for (int d = 0; d < hd; d++) { want[row * hd + d] += w * v[key * hd + d]; }
        }
    }
    float *d_q, *d_k, *d_v, *d_o;
    check(cudaMalloc(&d_q, q.size() * 4));
    check(cudaMalloc(&d_k, k.size() * 4));
    check(cudaMalloc(&d_v, v.size() * 4));
    check(cudaMalloc(&d_o, got.size() * 4));
    check(cudaMemcpy(d_q, q.data(), q.size() * 4, cudaMemcpyHostToDevice));
    check(cudaMemcpy(d_k, k.data(), k.size() * 4, cudaMemcpyHostToDevice));
    check(cudaMemcpy(d_v, v.data(), v.size() * 4, cudaMemcpyHostToDevice));
    mimo2_attn<<<1, 32>>>(d_o, d_q, d_k, d_v, nullptr, n, heads, heads, hd, hd, hd, hd, 0, 0, 0, -1, 0, group);
    check(cudaDeviceSynchronize());
    check(cudaMemcpy(got.data(), d_o, got.size() * 4, cudaMemcpyDeviceToHost));
    cudaFree(d_q); cudaFree(d_k); cudaFree(d_v); cudaFree(d_o);
    return close_enough(got, want, 1e-5f, "group");
}

static int test_norm_swiglu_gather() {
    const int n = 2, dim = 4, units = 3, width = 4;
    std::vector<float> x(n * dim), w(dim), b(dim), got(n * dim), want(n * dim);
    std::vector<float> gate(n * dim), up(n * dim), sw(n * dim), sw_want(n * dim);
    std::vector<float> src(units * width), dst(units * width), gather_want(units * width);
    std::vector<int> index = {2, 0, 1};
    for (int i = 0; i < n * dim; i++) {
        x[i] = 0.5f * i - 1.f;
        gate[i] = 0.3f * i - 0.4f;
        up[i] = 0.2f * i;
    }
    for (int i = 0; i < dim; i++) { w[i] = 1.f + 0.1f * i; b[i] = 0.05f * i; }
    for (int row = 0; row < n; row++) {
        float mean = 0.f;
        for (int i = 0; i < dim; i++) { mean += x[row * dim + i]; }
        mean /= dim;
        float var = 0.f;
        for (int i = 0; i < dim; i++) {
            const float d = x[row * dim + i] - mean;
            var += d * d;
        }
        var /= dim;
        const float inv = 1.f / sqrtf(var + 1e-5f);
        for (int i = 0; i < dim; i++) {
            want[row * dim + i] = (x[row * dim + i] - mean) * inv * w[i] + b[i];
        }
    }
    for (int i = 0; i < n * dim; i++) {
        const float g = gate[i] + b[i % dim];
        sw_want[i] = (g / (1.f + expf(-g))) * (up[i] + w[i % dim]);
    }
    for (int i = 0; i < units * width; i++) { src[i] = 0.1f * i; }
    for (int u = 0; u < units; u++) {
        for (int c = 0; c < width; c++) { gather_want[u * width + c] = src[index[u] * width + c]; }
    }
    float *d_x, *d_y, *d_w, *d_b, *d_g, *d_u, *d_s, *d_src, *d_dst;
    int *d_i;
    check(cudaMalloc(&d_x, x.size() * 4));
    check(cudaMalloc(&d_y, x.size() * 4));
    check(cudaMalloc(&d_w, w.size() * 4));
    check(cudaMalloc(&d_b, b.size() * 4));
    check(cudaMalloc(&d_g, gate.size() * 4));
    check(cudaMalloc(&d_u, up.size() * 4));
    check(cudaMalloc(&d_s, sw.size() * 4));
    check(cudaMalloc(&d_src, src.size() * 4));
    check(cudaMalloc(&d_dst, dst.size() * 4));
    check(cudaMalloc(&d_i, index.size() * 4));
    check(cudaMemcpy(d_x, x.data(), x.size() * 4, cudaMemcpyHostToDevice));
    check(cudaMemcpy(d_w, w.data(), w.size() * 4, cudaMemcpyHostToDevice));
    check(cudaMemcpy(d_b, b.data(), b.size() * 4, cudaMemcpyHostToDevice));
    check(cudaMemcpy(d_g, gate.data(), gate.size() * 4, cudaMemcpyHostToDevice));
    check(cudaMemcpy(d_u, up.data(), up.size() * 4, cudaMemcpyHostToDevice));
    check(cudaMemcpy(d_src, src.data(), src.size() * 4, cudaMemcpyHostToDevice));
    check(cudaMemcpy(d_i, index.data(), index.size() * 4, cudaMemcpyHostToDevice));
    mimo2_layernorm<<<n, 32, 32 * sizeof(float)>>>(d_y, d_x, d_w, d_b, dim, 1e-5f);
    mimo2_swiglu<<<1, 32>>>(d_s, d_g, d_u, d_b, d_w, n, dim);
    mimo2_gather<<<1, 32>>>(d_dst, d_src, d_i, units, width);
    check(cudaDeviceSynchronize());
    check(cudaMemcpy(got.data(), d_y, got.size() * 4, cudaMemcpyDeviceToHost));
    check(cudaMemcpy(sw.data(), d_s, sw.size() * 4, cudaMemcpyDeviceToHost));
    check(cudaMemcpy(dst.data(), d_dst, dst.size() * 4, cudaMemcpyDeviceToHost));
    cudaFree(d_x); cudaFree(d_y); cudaFree(d_w); cudaFree(d_b); cudaFree(d_g); cudaFree(d_u);
    cudaFree(d_s); cudaFree(d_src); cudaFree(d_dst); cudaFree(d_i);
    int rc = close_enough(got, want, 1e-5f, "layernorm");
    if (rc) { return rc; }
    rc = close_enough(sw, sw_want, 1e-5f, "swiglu");
    if (rc) { return rc; }
    return close_enough(dst, gather_want, 0.f, "gather");
}

static int test_conv_rvq() {
    const int n_in = 5, cin = 2, cout = 3, k = 3, stride = 2, pad = 1;
    const int n_out = (n_in + 2 * pad - k) / stride + 1;
    std::vector<float> in(cin * n_in), weight(cout * cin * k), bias(cout), got(n_out * cout), want(n_out * cout);
    for (int i = 0; i < (int)in.size(); i++) { in[i] = 0.1f * (i + 1); }
    for (int i = 0; i < (int)weight.size(); i++) { weight[i] = 0.05f * ((i % 9) - 4); }
    for (int i = 0; i < cout; i++) { bias[i] = 0.01f * i; }
    for (int t = 0; t < n_out; t++) {
        for (int oc = 0; oc < cout; oc++) {
            float acc = bias[oc];
            for (int ic = 0; ic < cin; ic++) {
                for (int kk = 0; kk < k; kk++) {
                    const int src = t * stride + kk - pad;
                    if (src < 0 || src >= n_in) { continue; }
                    acc += in[src + n_in * ic] * weight[(oc * cin + ic) * k + kk];
                }
            }
            want[t * cout + oc] = acc;
        }
    }
    const int rows = 2, dim = 4, bins = 3;
    std::vector<float> residual = {1, 0, 0, 0, 0, 1, 0, 0};
    std::vector<float> book(bins * dim);
    for (int b = 0; b < bins; b++) {
        for (int d = 0; d < dim; d++) { book[b * dim + d] = (d == b % dim) ? 1.f : 0.f; }
    }
    std::vector<int> ids(rows), id_want(rows);
    std::vector<float> left(residual), left_want(residual);
    for (int row = 0; row < rows; row++) {
        float best = -INFINITY;
        int best_i = 0;
        for (int b = 0; b < bins; b++) {
            float dot = 0.f, norm = 0.f;
            for (int d = 0; d < dim; d++) {
                dot += left_want[row * dim + d] * book[b * dim + d];
                norm += book[b * dim + d] * book[b * dim + d];
            }
            const float score = 2.f * dot - norm;
            if (score > best || (score == best && b < best_i)) { best = score; best_i = b; }
        }
        id_want[row] = best_i;
        for (int d = 0; d < dim; d++) { left_want[row * dim + d] -= book[best_i * dim + d]; }
    }
    float *d_in, *d_w, *d_b, *d_o, *d_r, *d_book;
    int *d_ids;
    check(cudaMalloc(&d_in, in.size() * 4));
    check(cudaMalloc(&d_w, weight.size() * 4));
    check(cudaMalloc(&d_b, bias.size() * 4));
    check(cudaMalloc(&d_o, got.size() * 4));
    check(cudaMalloc(&d_r, residual.size() * 4));
    check(cudaMalloc(&d_book, book.size() * 4));
    check(cudaMalloc(&d_ids, ids.size() * 4));
    check(cudaMemcpy(d_in, in.data(), in.size() * 4, cudaMemcpyHostToDevice));
    check(cudaMemcpy(d_w, weight.data(), weight.size() * 4, cudaMemcpyHostToDevice));
    check(cudaMemcpy(d_b, bias.data(), bias.size() * 4, cudaMemcpyHostToDevice));
    check(cudaMemcpy(d_r, residual.data(), residual.size() * 4, cudaMemcpyHostToDevice));
    check(cudaMemcpy(d_book, book.data(), book.size() * 4, cudaMemcpyHostToDevice));
    mimo2_conv1d<<<1, 32>>>(d_o, d_in, d_w, d_b, n_in, n_out, cin, cout, k, stride, pad);
    mimo2_rvq<<<rows, 256>>>(d_ids, d_r, d_book, dim, bins);
    check(cudaDeviceSynchronize());
    check(cudaMemcpy(got.data(), d_o, got.size() * 4, cudaMemcpyDeviceToHost));
    check(cudaMemcpy(ids.data(), d_ids, ids.size() * 4, cudaMemcpyDeviceToHost));
    check(cudaMemcpy(left.data(), d_r, left.size() * 4, cudaMemcpyDeviceToHost));
    cudaFree(d_in); cudaFree(d_w); cudaFree(d_b); cudaFree(d_o);
    cudaFree(d_r); cudaFree(d_book); cudaFree(d_ids);
    int rc = close_enough(got, want, 1e-5f, "conv1d");
    if (rc) { return rc; }
    for (int i = 0; i < rows; i++) {
        if (ids[i] != id_want[i]) {
            printf("rvq id %d got %d want %d\n", i, ids[i], id_want[i]);
            return 5;
        }
    }
    printf("rvq ids exact\n");
    return close_enough(left, left_want, 1e-5f, "rvq");
}

int main() {
    int rc = test_patch();
    if (rc) { return rc; }
    rc = test_rope();
    if (rc) { return rc; }
    rc = test_attn();
    if (rc) { return rc; }
    rc = test_group();
    if (rc) { return rc; }
    rc = test_norm_swiglu_gather();
    if (rc) { return rc; }
    rc = test_conv_rvq();
    return rc;
}
