// Scalar softmax oracle includes causal/SWA masks, GQA head mapping and sinks.
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include "../cuda/mimo2_primitives.cuh"
#include "../cuda/mimo2_prefill.cuh"

static void check(cudaError_t status) {
    if (status != cudaSuccess) {
        fprintf(stderr, "%s\n", cudaGetErrorString(status));
        exit(1);
    }
}

static int test_swa_hmma() {
    enum { HEADS = 64, KEY = 192, VALUE = 128, ROWS = 65 };
    for (unsigned kv : {8u}) {
        for (unsigned start : {0u, 5u, 127u, 129u, 4095u, 262144u}) {
            for (unsigned use_sink : {0u, 1u}) {
                const unsigned window = 128;
                // Leave space for the captured transition at short contexts.
                const unsigned capacity = std::min(256u, start + ROWS + 8);
                const unsigned stride = kv * (KEY + VALUE);
                std::vector<__half> cache(capacity * stride, __float2half(0));
                std::vector<float> q(ROWS * HEADS * KEY), out(ROWS * HEADS * VALUE);
                std::vector<float> sinks(HEADS);
                std::vector<unsigned> positions(ROWS);
                for (unsigned row = 0; row < ROWS; row++) { positions[row] = start + row; }
                for (unsigned i = 0; i < q.size(); i++) { q[i] = sinf(i * 0.013f); }
                for (unsigned h = 0; h < HEADS; h++) { sinks[h] = h % 3 == 0 ? 12 : h * 0.03f; }
                const unsigned retained = start + ROWS > capacity ? start + ROWS - capacity : 0;
                for (unsigned p = retained; p < start + ROWS; p++) {
                    for (unsigned d = 0; d < stride; d++) {
                        cache[(p % capacity) * stride + d] = __float2half_rn(sinf(p * 0.047f + d * 0.017f));
                    }
                }
                float *dq, *ds, *dout;
                __half *dc;
                unsigned *dp;
                check(cudaMalloc(&dq, q.size() * sizeof(float)));
                check(cudaMalloc(&ds, sinks.size() * sizeof(float)));
                check(cudaMalloc(&dout, out.size() * sizeof(float)));
                check(cudaMalloc(&dc, cache.size() * sizeof(__half)));
                check(cudaMalloc(&dp, positions.size() * sizeof(unsigned)));
                check(cudaMemcpy(dq, q.data(), q.size() * sizeof(float), cudaMemcpyHostToDevice));
                check(cudaMemcpy(ds, sinks.data(), sinks.size() * sizeof(float), cudaMemcpyHostToDevice));
                check(cudaMemcpy(dc, cache.data(), cache.size() * sizeof(__half), cudaMemcpyHostToDevice));
                check(cudaMemcpy(dp, positions.data(), positions.size() * sizeof(unsigned), cudaMemcpyHostToDevice));
                mimo2_hmma::prefill<mimo2_hmma::Async, 128><<<dim3((ROWS+63)/64, HEADS), 128>>>(
                    dout, dq, dc, use_sink ? ds : nullptr, dp, ROWS, kv, capacity);
                check(cudaGetLastError());
                check(cudaMemcpy(out.data(), dout, out.size() * sizeof(float), cudaMemcpyDeviceToHost));
                auto oracle_error = [&]() {
                    double error = 0;
                    for (unsigned row = 0; row < ROWS; row++) {
                        const unsigned pos = positions[row];
                        const unsigned first = window && pos + 1 > window ? pos + 1 - window : 0;
                        for (unsigned h = 0; h < HEADS; h++) {
                            const unsigned kh = h / (HEADS / kv);
                            std::vector<double> score(pos - first + 1);
                            double maximum = use_sink ? sinks[h] : -INFINITY;
                            for (unsigned p = first; p <= pos; p++) {
                                double dot = 0;
                                for (unsigned d = 0; d < KEY; d++) {
                                    dot += (double)q[(row * HEADS + h) * KEY + d] *
                                        __half2float(cache[(p % capacity) * stride + kh * KEY + d]);
                                }
                                score[p - first] = dot / sqrt((double)KEY);
                                maximum = std::max(maximum, score[p - first]);
                            }
                            double denominator = use_sink ? exp(sinks[h] - maximum) : 0;
                            for (double &s : score) { s = exp(s - maximum); denominator += s; }
                            for (unsigned d = 0; d < VALUE; d++) {
                                double value = 0;
                                for (unsigned p = first; p <= pos; p++) {
                                    value += score[p - first] * __half2float(
                                        cache[(p % capacity) * stride + kv * KEY + kh * VALUE + d]);
                                }
                                const float actual = out[(row * HEADS + h) * VALUE + d];
                                if (!std::isfinite(actual)) { return (double)INFINITY; }
                                error = std::max(error, fabs(actual - value / denominator));
                            }
                        }
                    }
                    return error;
                };
                const double error = oracle_error();
                printf("kv=%u start=%u sink=%u max_abs_error=%.9g\n", kv, start, use_sink, error);
                if (error > 2e-5) { return 3; }
                cudaStream_t stream;
                cudaGraph_t graph;
                cudaGraphExec_t replay;
                check(cudaStreamCreate(&stream));
                check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
                mimo2_hmma::prefill<mimo2_hmma::Async, 128><<<dim3((ROWS+63)/64, HEADS), 128, 0, stream>>>(
                    dout, dq, dc, use_sink ? ds : nullptr, dp, ROWS, kv, capacity);
                check(cudaStreamEndCapture(stream, &graph));
                check(cudaGraphInstantiate(&replay, graph, nullptr, nullptr, 0));
                for (unsigned &p : positions) { p++; }
                const unsigned fresh = positions.back();
                for (unsigned d = 0; d < stride; d++) {
                    cache[(fresh % capacity) * stride + d] =
                        __float2half_rn(sinf(fresh * 0.047f + d * 0.017f));
                }
                check(cudaMemcpy(dc, cache.data(), cache.size() * sizeof(__half), cudaMemcpyHostToDevice));
                check(cudaMemcpy(dp, positions.data(), positions.size() * sizeof(unsigned), cudaMemcpyHostToDevice));
                mimo2_hmma::prefill<mimo2_hmma::Async, 128><<<dim3((ROWS+63)/64, HEADS), 128>>>(
                    dout, dq, dc, use_sink ? ds : nullptr, dp, ROWS, kv, capacity);
                check(cudaGetLastError());
                check(cudaMemcpy(out.data(), dout, out.size() * sizeof(float), cudaMemcpyDeviceToHost));
                std::vector<float> replayed(out.size());
                check(cudaGraphLaunch(replay, stream));
                check(cudaStreamSynchronize(stream));
                check(cudaMemcpy(replayed.data(), dout, out.size() * sizeof(float), cudaMemcpyDeviceToHost));
                if (out != replayed) { return 4; }
                const double replay_error = oracle_error();
                printf("swa_updated_kv_replay_fp64_error=%.9g\n", replay_error);
                if (replay_error > 2e-5) { return 13; }
                check(cudaGraphExecDestroy(replay)); check(cudaGraphDestroy(graph));
                check(cudaStreamDestroy(stream));
                puts("changed_position_capture=eager_exact");
                check(cudaFree(dq)); check(cudaFree(ds)); check(cudaFree(dout));
                check(cudaFree(dc)); check(cudaFree(dp));
            }
        }
    }
    return 0;
}

int main() {
    // Same predicate the host launch uses. 2048-row steps tile from 10240.
    // 4096-row prefill chunks tile from 8192. Decode and SWA stay on the walk.
    if (m2_use_tile(0, 4, 2048, 8192) != 0) { return 20; }
    if (m2_use_tile(0, 4, 2048, 10240) != 1) { return 20; }
    if (m2_use_tile(0, 4, 4096, 6144) != 0) { return 20; }
    if (m2_use_tile(0, 4, 4096, 8192) != 1) { return 20; }
    if (m2_use_tile(0, 4, 1, 65536) != 0) { return 20; }
    if (m2_use_tile(128, 8, 4096, 65536) != 0) { return 20; }
    puts("tile_gate=2048@10240,4096@8192");
    enum { HEADS = 64, KEY = 192, VALUE = 128, ROWS = 3 };
    for (unsigned kv : {4u, 8u}) {
        for (unsigned start : {5u, 129u, 4095u, 262144u}) {
            for (unsigned use_sink : {0u, 1u}) {
                const unsigned window = kv == 8 ? 128 : 0;
                // Full attention's scalar oracle is linear in position.
                // The high index is the SWA-128 ring, which stays window-sized.
                if (window == 0 && start > 4095u) { continue; }
                const unsigned capacity = window ? std::min(256u, start + ROWS) : start + ROWS;
                const unsigned stride = kv * (KEY + VALUE);
                std::vector<__half> cache(capacity * stride);
                std::vector<float> q(ROWS * HEADS * KEY), out(ROWS * HEADS * VALUE);
                std::vector<float> sinks(HEADS);
                std::vector<unsigned> positions(ROWS);
                for (unsigned row = 0; row < ROWS; row++) { positions[row] = start + row; }
                for (unsigned i = 0; i < q.size(); i++) { q[i] = sinf(i * 0.013f); }
                for (unsigned h = 0; h < HEADS; h++) { sinks[h] = h % 3 == 0 ? 12 : h * 0.03f; }
                for (unsigned p = 0; p < start + ROWS; p++) {
                    for (unsigned d = 0; d < stride; d++) {
                        cache[(p % capacity) * stride + d] = __float2half_rn(sinf(p * 0.047f + d * 0.017f));
                    }
                }
                float *dq, *ds, *dout;
                __half *dc;
                unsigned *dp;
                check(cudaMalloc(&dq, q.size() * sizeof(float)));
                check(cudaMalloc(&ds, sinks.size() * sizeof(float)));
                check(cudaMalloc(&dout, out.size() * sizeof(float)));
                check(cudaMalloc(&dc, cache.size() * sizeof(__half)));
                check(cudaMalloc(&dp, positions.size() * sizeof(unsigned)));
                check(cudaMemcpy(dq, q.data(), q.size() * sizeof(float), cudaMemcpyHostToDevice));
                check(cudaMemcpy(ds, sinks.data(), sinks.size() * sizeof(float), cudaMemcpyHostToDevice));
                check(cudaMemcpy(dc, cache.data(), cache.size() * sizeof(__half), cudaMemcpyHostToDevice));
                check(cudaMemcpy(dp, positions.data(), positions.size() * sizeof(unsigned), cudaMemcpyHostToDevice));
                mimo2_attention<<<dim3(HEADS / 4, ROWS), 128>>>(
                    dout, dq, dc, use_sink ? ds : nullptr, dp, kv, capacity, window);
                check(cudaGetLastError());
                check(cudaMemcpy(out.data(), dout, out.size() * sizeof(float), cudaMemcpyDeviceToHost));
                double error = 0;
                for (unsigned row = 0; row < ROWS; row++) {
                    const unsigned pos = positions[row];
                    const unsigned first = window && pos + 1 > window ? pos + 1 - window : 0;
                    for (unsigned h = 0; h < HEADS; h++) {
                        const unsigned kh = h / (HEADS / kv);
                        std::vector<double> score(pos - first + 1);
                        double maximum = use_sink ? sinks[h] : -INFINITY;
                        for (unsigned p = first; p <= pos; p++) {
                            double dot = 0;
                            for (unsigned d = 0; d < KEY; d++) {
                                dot += (double)q[(row * HEADS + h) * KEY + d] *
                                    __half2float(cache[(p % capacity) * stride + kh * KEY + d]);
                            }
                            score[p - first] = dot / sqrt((double)KEY);
                            maximum = std::max(maximum, score[p - first]);
                        }
                        double denominator = use_sink ? exp(sinks[h] - maximum) : 0;
                        for (double &s : score) { s = exp(s - maximum); denominator += s; }
                        for (unsigned d = 0; d < VALUE; d++) {
                            double value = 0;
                            for (unsigned p = first; p <= pos; p++) {
                                value += score[p - first] * __half2float(
                                    cache[(p % capacity) * stride + kv * KEY + kh * VALUE + d]);
                            }
                            const float actual = out[(row * HEADS + h) * VALUE + d];
                            if (!std::isfinite(actual)) { return 2; }
                            error = std::max(error, fabs(actual - value / denominator));
                        }
                    }
                }
                printf("kv=%u start=%u sink=%u max_abs_error=%.9g\n", kv, start, use_sink, error);
                if (error > 2e-5) { return 3; }
                cudaStream_t stream;
                cudaGraph_t graph;
                cudaGraphExec_t replay;
                check(cudaStreamCreate(&stream));
                check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
                mimo2_attention<<<dim3(HEADS / 4, ROWS), 128, 0, stream>>>(
                    dout, dq, dc, use_sink ? ds : nullptr, dp, kv, capacity, window);
                check(cudaStreamEndCapture(stream, &graph));
                check(cudaGraphInstantiate(&replay, graph, nullptr, nullptr, 0));
                for (unsigned &p : positions) { p--; }
                check(cudaMemcpy(dp, positions.data(), positions.size() * sizeof(unsigned), cudaMemcpyHostToDevice));
                mimo2_attention<<<dim3(HEADS / 4, ROWS), 128>>>(
                    dout, dq, dc, use_sink ? ds : nullptr, dp, kv, capacity, window);
                check(cudaGetLastError());
                check(cudaMemcpy(out.data(), dout, out.size() * sizeof(float), cudaMemcpyDeviceToHost));
                std::vector<float> replayed(out.size());
                check(cudaGraphLaunch(replay, stream));
                check(cudaStreamSynchronize(stream));
                check(cudaMemcpy(replayed.data(), dout, out.size() * sizeof(float), cudaMemcpyDeviceToHost));
                if (out != replayed) { return 4; }
                check(cudaGraphExecDestroy(replay)); check(cudaGraphDestroy(graph));
                check(cudaStreamDestroy(stream));
                puts("changed_position_capture=eager_exact");
                check(cudaFree(dq)); check(cudaFree(ds)); check(cudaFree(dout));
                check(cudaFree(dc)); check(cudaFree(dp));
            }
        }
    }
    // Wide full-attention rows. The tile kernel must match the walking kernel
    // and the scalar oracle. DS4_MIMO2_FATTN=0 keeps the walking launch.
    for (unsigned start : {0u, 1000u, 4096u, 32768u}) {
        for (unsigned use_sink : {0u, 1u}) {
            enum { ROWS_T = 65, KV = 4 };
            const unsigned window = 0;
            const unsigned capacity = start + ROWS_T + 8;
            const unsigned stride = KV * (KEY + VALUE);
            std::vector<__half> cache(capacity * stride, __float2half(0));
            std::vector<float> q(ROWS_T * HEADS * KEY), walk(ROWS_T * HEADS * VALUE);
            std::vector<float> tiled(ROWS_T * HEADS * VALUE), sinks(HEADS);
            std::vector<unsigned> positions(ROWS_T);
            for (unsigned row = 0; row < ROWS_T; row++) { positions[row] = start + row; }
            for (unsigned i = 0; i < q.size(); i++) { q[i] = sinf(i * 0.013f); }
            for (unsigned h = 0; h < HEADS; h++) { sinks[h] = h % 3 == 0 ? 12 : h * 0.03f; }
            for (unsigned p = 0; p < start + ROWS_T; p++) {
                for (unsigned d = 0; d < stride; d++) {
                    cache[p * stride + d] = __float2half_rn(sinf(p * 0.047f + d * 0.017f));
                }
            }
            float *dq, *ds, *dout;
            __half *dc;
            unsigned *dp;
            check(cudaMalloc(&dq, q.size() * sizeof(float)));
            check(cudaMalloc(&ds, sinks.size() * sizeof(float)));
            check(cudaMalloc(&dout, tiled.size() * sizeof(float)));
            check(cudaMalloc(&dc, cache.size() * sizeof(__half)));
            check(cudaMalloc(&dp, positions.size() * sizeof(unsigned)));
            check(cudaMemcpy(dq, q.data(), q.size() * sizeof(float), cudaMemcpyHostToDevice));
            check(cudaMemcpy(ds, sinks.data(), sinks.size() * sizeof(float), cudaMemcpyHostToDevice));
            check(cudaMemcpy(dc, cache.data(), cache.size() * sizeof(__half), cudaMemcpyHostToDevice));
            check(cudaMemcpy(dp, positions.data(), positions.size() * sizeof(unsigned), cudaMemcpyHostToDevice));
            mimo2_attention<<<dim3(HEADS / 4, ROWS_T), 128>>>(
                dout, dq, dc, use_sink ? ds : nullptr, dp, KV, capacity, window);
            check(cudaGetLastError());
            check(cudaMemcpy(walk.data(), dout, walk.size() * sizeof(float), cudaMemcpyDeviceToHost));
            mimo2_attn_tile<<<dim3(ROWS_T, KV), 512>>>(
                dout, dq, dc, use_sink ? ds : nullptr, dp, KV, capacity);
            check(cudaGetLastError());
            check(cudaMemcpy(tiled.data(), dout, tiled.size() * sizeof(float), cudaMemcpyDeviceToHost));
            double walk_error = 0;
            for (unsigned i = 0; i < tiled.size(); i++) {
                if (!std::isfinite(tiled[i])) { return 5; }
                if (tiled[i] != walk[i]) {
                    walk_error = std::max(walk_error, (double)fabsf(tiled[i] - walk[i]));
                }
            }
            std::vector<float> hmma(walk.size());
            mimo2_hmma::prefill<<<dim3((ROWS_T + 63) / 64, HEADS), 128>>>(
                dout, dq, dc, use_sink ? ds : nullptr, dp, ROWS_T, KV, capacity);
            check(cudaGetLastError());
            check(cudaMemcpy(hmma.data(), dout, hmma.size() * sizeof(float), cudaMemcpyDeviceToHost));
            double hmma_error = 0;
            for (unsigned i = 0; i < hmma.size(); i++) {
                if (!std::isfinite(hmma[i])) { return 10; }
                hmma_error = std::max(hmma_error, (double)fabsf(hmma[i] - tiled[i]));
            }
            printf("hmma start=%u sink=%u walk_error=%.9g\n", start, use_sink, hmma_error);
            mimo2_hmma::prefill<mimo2_hmma::Async><<<dim3((ROWS_T + 63) / 64, HEADS), 128>>>(
                dout, dq, dc, use_sink ? ds : nullptr, dp, ROWS_T, KV, capacity);
            check(cudaGetLastError());
            std::vector<float> copied(hmma.size());
            check(cudaMemcpy(copied.data(), dout, copied.size() * sizeof(float), cudaMemcpyDeviceToHost));
            if (copied != hmma) { return 12; }
            puts("hmma_async_copy=scalar_exact");
            // Independent FP64 oracle, including long walks. Compare the
            // tensor-core error to the existing path's accumulation error.
            double oracle_error = 0, hmma_oracle = 0;
            auto compare_oracle = [&]() {
                oracle_error = hmma_oracle = 0;
                // Sample both MMA row halves, warp edges and the ragged CTA.
                for (unsigned row : {0u, 7u, 8u, 15u, 16u, 31u, 32u, 63u, 64u}) {
                    const unsigned pos = positions[row];
                    for (unsigned h : {0u, 15u, 16u, 31u, 48u, 63u}) {
                        const unsigned kh = h / (HEADS / KV);
                        std::vector<double> score(pos + 1);
                        double maximum = use_sink ? sinks[h] : -INFINITY;
                        for (unsigned p = 0; p <= pos; p++) {
                            double dot = 0;
                            for (unsigned d = 0; d < KEY; d++) {
                                dot += (double)q[(row * HEADS + h) * KEY + d] *
                                    __half2float(cache[(p % capacity) * stride + kh * KEY + d]);
                            }
                            score[p] = dot / sqrt((double)KEY);
                            maximum = std::max(maximum, score[p]);
                        }
                        double denominator = use_sink ? exp(sinks[h] - maximum) : 0;
                        for (double &s : score) { s = exp(s - maximum); denominator += s; }
                        for (unsigned d = 0; d < VALUE; d++) {
                            double value = 0;
                            for (unsigned p = 0; p <= pos; p++) {
                                value += score[p] * __half2float(
                                    cache[(p % capacity) * stride + KV * KEY + kh * VALUE + d]);
                            }
                            oracle_error = std::max(oracle_error, fabs(
                                (double)tiled[(row * HEADS + h) * VALUE + d] - value / denominator));
                            hmma_oracle = std::max(hmma_oracle, fabs(
                                (double)hmma[(row * HEADS + h) * VALUE + d] - value / denominator));
                        }
                    }
                }
            };
            compare_oracle();
            printf("tile start=%u sink=%u cap=%u walk_error=%.9g oracle_error=%.9g\n",
                start, use_sink, capacity, walk_error, oracle_error);
            printf("hmma_fp64_error=%.9g walk_fp64_error=%.9g\n", hmma_oracle, oracle_error);
            const double hmma_limit = std::max(2e-5, oracle_error * 1.25);
            if (walk_error != 0 || (start <= 4096 && oracle_error > 2e-5) || hmma_oracle > hmma_limit) { return 6; }
            // 32-byte L2 loads must reproduce the scalar tile bit for bit.
            // DS4_MIMO2_FATTN_L2=0 keeps that scalar launch.
            mimo2_attn_l2<<<dim3(ROWS_T, KV), 512>>>(
                dout, dq, dc, use_sink ? ds : nullptr, dp, KV, capacity);
            check(cudaGetLastError());
            check(cudaMemcpy(walk.data(), dout, walk.size() * sizeof(float), cudaMemcpyDeviceToHost));
            if (walk != tiled) { return 8; }
            auto advance = [&](unsigned count) {
                const unsigned old_last = positions.back();
                for (unsigned &p : positions) { p += count; }
                for (unsigned p = old_last + 1; p <= positions.back(); p++) {
                    for (unsigned d = 0; d < stride; d++) {
                        cache[p * stride + d] = __float2half_rn(sinf(p * 0.047f + d * 0.017f));
                    }
                }
                check(cudaMemcpy(dc, cache.data(), cache.size() * sizeof(__half), cudaMemcpyHostToDevice));
                check(cudaMemcpy(dp, positions.data(), positions.size() * sizeof(unsigned), cudaMemcpyHostToDevice));
            };
            cudaStream_t stream;
            cudaGraph_t graph;
            cudaGraphExec_t replay;
            check(cudaStreamCreate(&stream));
            check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
            mimo2_attn_tile<<<dim3(ROWS_T, KV), 512, 0, stream>>>(
                dout, dq, dc, use_sink ? ds : nullptr, dp, KV, capacity);
            check(cudaStreamEndCapture(stream, &graph));
            check(cudaGraphInstantiate(&replay, graph, nullptr, nullptr, 0));
            advance(3);
            mimo2_attn_tile<<<dim3(ROWS_T, KV), 512>>>(
                dout, dq, dc, use_sink ? ds : nullptr, dp, KV, capacity);
            check(cudaGetLastError());
            check(cudaMemcpy(walk.data(), dout, walk.size() * sizeof(float), cudaMemcpyDeviceToHost));
            std::vector<float> replayed(walk.size());
            check(cudaGraphLaunch(replay, stream));
            check(cudaStreamSynchronize(stream));
            check(cudaMemcpy(replayed.data(), dout, replayed.size() * sizeof(float), cudaMemcpyDeviceToHost));
            if (walk != replayed) { return 7; }
            check(cudaGraphExecDestroy(replay)); check(cudaGraphDestroy(graph));
            check(cudaStreamDestroy(stream));
            puts("tile_changed_position_capture=eager_exact");
            check(cudaStreamCreate(&stream));
            check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
            mimo2_hmma::prefill<mimo2_hmma::Async><<<dim3((ROWS_T + 63) / 64, HEADS), 128, 0, stream>>>(
                dout, dq, dc, use_sink ? ds : nullptr, dp, ROWS_T, KV, capacity);
            check(cudaStreamEndCapture(stream, &graph));
            check(cudaGraphInstantiate(&replay, graph, nullptr, nullptr, 0));
            advance(5);
            mimo2_hmma::prefill<mimo2_hmma::Async><<<dim3((ROWS_T + 63) / 64, HEADS), 128>>>(
                dout, dq, dc, use_sink ? ds : nullptr, dp, ROWS_T, KV, capacity);
            check(cudaGetLastError());
            check(cudaMemcpy(hmma.data(), dout, hmma.size() * sizeof(float), cudaMemcpyDeviceToHost));
            check(cudaGraphLaunch(replay, stream));
            check(cudaStreamSynchronize(stream));
            check(cudaMemcpy(replayed.data(), dout, replayed.size() * sizeof(float), cudaMemcpyDeviceToHost));
            if (hmma != replayed) { return 11; }
            mimo2_attention<<<dim3(HEADS / 4, ROWS_T), 128>>>(
                dout, dq, dc, use_sink ? ds : nullptr, dp, KV, capacity, window);
            check(cudaGetLastError());
            check(cudaMemcpy(tiled.data(), dout, tiled.size() * sizeof(float), cudaMemcpyDeviceToHost));
            compare_oracle();
            printf("full_updated_kv_replay_fp64_error=%.9g walk_error=%.9g\n", hmma_oracle, oracle_error);
            if (hmma_oracle > std::max(2e-5, oracle_error * 1.25)) { return 14; }
            check(cudaGraphExecDestroy(replay)); check(cudaGraphDestroy(graph));
            check(cudaStreamDestroy(stream));
            puts("hmma_changed_position_capture=eager_exact");
            check(cudaFree(dq)); check(cudaFree(ds)); check(cudaFree(dout));
            check(cudaFree(dc)); check(cudaFree(dp));
        }
    }
    // One-row full attention. 32 key slices must match the walk. The sink
    // lives on slice 0. DS4_MIMO2_ATTN_SPLIT=0 keeps the walking launch.
    for (unsigned start : {0u, 17u, 128u, 1000u, 4095u}) {
        for (unsigned use_sink : {0u, 1u}) {
            enum { KV = 4 };
            const unsigned capacity = start + 1;
            const unsigned stride = KV * (KEY + VALUE);
            std::vector<__half> cache(capacity * stride);
            std::vector<float> q(HEADS * KEY), sinks(HEADS), walk(HEADS * VALUE), split(HEADS * VALUE);
            std::vector<unsigned> positions(1, start);
            for (unsigned i = 0; i < q.size(); i++) { q[i] = sinf(i * 0.013f); }
            for (unsigned h = 0; h < HEADS; h++) { sinks[h] = h % 5 == 0 ? 2.0f : h * 0.01f; }
            for (unsigned p = 0; p < capacity; p++) {
                for (unsigned d = 0; d < stride; d++) {
                    cache[p * stride + d] = __float2half_rn(sinf(p * 0.02f + d * 0.01f));
                }
            }
            float *dq, *ds, *dout, *pmax, *pden, *pacc;
            __half *dc;
            unsigned *dp;
            const size_t npartial = (size_t)M2_DECODE_SPLITS * HEADS;
            check(cudaMalloc(&dq, q.size() * sizeof(float)));
            check(cudaMalloc(&ds, sinks.size() * sizeof(float)));
            check(cudaMalloc(&dout, walk.size() * sizeof(float)));
            check(cudaMalloc(&dc, cache.size() * sizeof(__half)));
            check(cudaMalloc(&dp, sizeof(unsigned)));
            check(cudaMalloc(&pmax, npartial * sizeof(float)));
            check(cudaMalloc(&pden, npartial * sizeof(float)));
            check(cudaMalloc(&pacc, npartial * VALUE * sizeof(float)));
            check(cudaMemcpy(dq, q.data(), q.size() * sizeof(float), cudaMemcpyHostToDevice));
            check(cudaMemcpy(ds, sinks.data(), sinks.size() * sizeof(float), cudaMemcpyHostToDevice));
            check(cudaMemcpy(dc, cache.data(), cache.size() * sizeof(__half), cudaMemcpyHostToDevice));
            check(cudaMemcpy(dp, positions.data(), sizeof(unsigned), cudaMemcpyHostToDevice));
            mimo2_attention<<<dim3(HEADS / 4, 1), 128>>>(
                dout, dq, dc, use_sink ? ds : nullptr, dp, KV, capacity, 0);
            check(cudaGetLastError());
            check(cudaMemcpy(walk.data(), dout, walk.size() * sizeof(float), cudaMemcpyDeviceToHost));
            mimo2_attn_split<<<dim3(M2_DECODE_SPLITS, KV), 512>>>(
                pmax, pden, pacc, dq, dc, use_sink ? ds : nullptr, dp, KV, capacity,
                M2_DECODE_SPLITS, 0);
            mimo2_attn_merge<<<HEADS, 32>>>(dout, pmax, pden, pacc, M2_DECODE_SPLITS);
            check(cudaGetLastError());
            check(cudaMemcpy(split.data(), dout, split.size() * sizeof(float), cudaMemcpyDeviceToHost));
            double err = 0;
            for (unsigned i = 0; i < split.size(); i++) {
                if (!std::isfinite(split[i])) { return 9; }
                err = std::max(err, (double)fabsf(split[i] - walk[i]));
            }
            printf("split start=%u sink=%u err=%.9g\n", start, use_sink, err);
            // Slice merge reorders the online softmax. A wrong slice is ~1.
            if (err > 1e-4) { return 9; }
            check(cudaFree(dq)); check(cudaFree(ds)); check(cudaFree(dout));
            check(cudaFree(dc)); check(cudaFree(dp));
            check(cudaFree(pmax)); check(cudaFree(pden)); check(cudaFree(pacc));
        }
    }
    return test_swa_hmma();
}
