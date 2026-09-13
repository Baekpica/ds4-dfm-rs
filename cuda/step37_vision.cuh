/* Step F16 projector: CHW pixels, row-major ViT, interleaved 2D RoPE.
 * Equations follow the pinned vision_encoder.py; no model ownership here. */
#pragma once

// Keep the documented libdevice implementation under --use_fast_math.
// Its automatic __sincosf substitution loses phase accuracy even at x=40.
extern "C" __device__ void __nv_sincosf(float, float *, float *);

__global__ static void s37v_im2col(float *out, const float *in,
        unsigned edge, unsigned channels, unsigned kernel) {
    const unsigned stride = kernel == 14 ? 14 : 2, pad = kernel == 14 ? 0 : 1;
    const unsigned output = (edge + 2 * pad - kernel) / stride + 1;
    const unsigned inner = channels * kernel * kernel;
    const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= output * output * inner) { return; }
    const unsigned row = i / inner, k = i % inner;
    const unsigned c = k / (kernel * kernel);
    const int x = (row % output) * stride + k % kernel - pad;
    const int y = (row / output) * stride + (k / kernel) % kernel - pad;
    float v = 0;
    if (x >= 0 && y >= 0 && x < (int)edge && y < (int)edge) {
        const unsigned at = kernel == 14 ? (c * edge + y) * edge + x :
            (y * edge + x) * channels + c;
        v = in[at];
    }
    out[i] = v;
}

__global__ static void s37v_position(float *hidden, const float *position, unsigned edge) {
    enum { DIM = 1536, SOURCE = 52 };
    const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= edge * edge * DIM) { return; }
    const unsigned row = i / DIM, channel = i % DIM;
    // Learned position interpolation has no antialias filter. The processor's
    // antialiased RGB resize is a separate operation with a different contract.
    const float sx = fmaxf(0, (row % edge + 0.5f) * ((float)SOURCE / edge) - 0.5f);
    const float sy = fmaxf(0, (row / edge + 0.5f) * ((float)SOURCE / edge) - 0.5f);
    const unsigned x0 = (unsigned)sx, y0 = (unsigned)sy;
    const unsigned x1 = min(x0 + 1, (unsigned)SOURCE - 1), y1 = min(y0 + 1, (unsigned)SOURCE - 1);
    const float wx = sx - x0, wy = sy - y0;
    const float top = position[(y0 * SOURCE + x0) * DIM + channel] * (1 - wx) +
                      position[(y0 * SOURCE + x1) * DIM + channel] * wx;
    const float bottom = position[(y1 * SOURCE + x0) * DIM + channel] * (1 - wx) +
                         position[(y1 * SOURCE + x1) * DIM + channel] * wx;
    hidden[i] += top * (1 - wy) + bottom * wy;
}

// Exact F32 cache from EncoderRope2D: 1 / (10000 ** (arange(0,48,2)/48)).
// Fixed model geometry avoids fast-math powf error and repeated exponentiation.
__device__ __constant__ static float s37v_inv_freq[24] = {
    0x1p+0f, 0x1.5cd25p-1f, 0x1.db4c78p-2f, 0x1.43d136p-2f,
    0x1.b93a6ap-3f, 0x1.2c9af4p-3f, 0x1.99999ap-4f, 0x1.170ea8p-4f,
    0x1.7c3d2ap-5f, 0x1.030dc6p-5f, 0x1.60fb8ep-6f, 0x1.e0f7ecp-7f,
    0x1.47ae14p-7f, 0x1.be7dd2p-8f, 0x1.3030f4p-8f, 0x1.9e7c7p-9f,
    0x1.1a62d2p-9f, 0x1.80c65cp-10f, 0x1.0624dep-10f, 0x1.653174p-11f,
    0x1.e6b4b8p-12f, 0x1.4b96cp-12f, 0x1.c3d14ep-13f, 0x1.33d1e4p-13f
};

__global__ static void s37v_qkv(float *qkv, const float *bias, unsigned edge) {
    enum { HEADS = 16, HEAD = 96, HALF = HEAD / 2, QUARTER = HEAD / 4 };
    const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= edge * edge * 3 * HEADS * HALF) { return; }
    const unsigned pair = i % HALF, head = (i / HALF) % (3 * HEADS);
    const unsigned row = i / (HALF * 3 * HEADS), offset = head * HEAD + 2 * pair;
    const float a = qkv[2 * i] + bias[offset], b = qkv[2 * i + 1] + bias[offset + 1];
    if (head >= 2 * HEADS) {
        qkv[2 * i] = a; qkv[2 * i + 1] = b; return;
    }
    // x occupies the first 48 dimensions, y the last 48. Each adjacent pair
    // rotates together. Crops use their actual coordinates, not a scaled grid.
    const unsigned position = pair < QUARTER ? row % edge : row / edge;
    const float angle = position * s37v_inv_freq[pair % QUARTER];
    float sine, cosine; __nv_sincosf(angle, &sine, &cosine);
    qkv[2 * i] = a * cosine - b * sine;
    qkv[2 * i + 1] = b * cosine + a * sine;
}

__global__ static void s37v_quick_gelu(float *x, const float *bias, unsigned count, unsigned dim) {
    const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) { return; }
    const float v = x[i] + bias[i % dim];
    x[i] = v / (1 + expf(-1.702f * v));
}

__global__ static void s37v_residual(float *residual, const float *x,
        const float *bias, const float *scale, unsigned count) {
    const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) { residual[i] += (x[i] + bias[i % 1536]) * scale[i % 1536]; }
}
