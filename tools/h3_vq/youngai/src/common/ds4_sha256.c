/* ds4_sha256.c — 见 ds4_sha256.h。 */
#include "ds4_sha256.h"
#include <string.h>

static const uint32_t K[64] = {
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
};

#define ROR(x, r) (((x) >> (r)) | ((x) << (32 - (r))))

static void block(uint32_t h[8], const uint8_t *p) {
    uint32_t w[64];
    for (int i = 0; i < 16; i++) w[i] = (uint32_t)p[4 * i] << 24 | (uint32_t)p[4 * i + 1] << 16 | (uint32_t)p[4 * i + 2] << 8 | p[4 * i + 3];
    for (int i = 16; i < 64; i++) {
        const uint32_t s0 = ROR(w[i - 15], 7) ^ ROR(w[i - 15], 18) ^ (w[i - 15] >> 3);
        const uint32_t s1 = ROR(w[i - 2], 17) ^ ROR(w[i - 2], 19) ^ (w[i - 2] >> 10);
        w[i] = w[i - 16] + s0 + w[i - 7] + s1;
    }
    uint32_t a = h[0], b = h[1], c = h[2], d = h[3], e = h[4], f = h[5], g = h[6], k = h[7];
    for (int i = 0; i < 64; i++) {
        const uint32_t t1 = k + (ROR(e, 6) ^ ROR(e, 11) ^ ROR(e, 25)) + ((e & f) ^ (~e & g)) + K[i] + w[i];
        const uint32_t t2 = (ROR(a, 2) ^ ROR(a, 13) ^ ROR(a, 22)) + ((a & b) ^ (a & c) ^ (b & c));
        k = g; g = f; f = e; e = d + t1; d = c; c = b; b = a; a = t1 + t2;
    }
    h[0] += a; h[1] += b; h[2] += c; h[3] += d; h[4] += e; h[5] += f; h[6] += g; h[7] += k;
}

void ds4_sha256_init(ds4_sha256_ctx *c) {
    static const uint32_t iv[8] = { 0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19 };
    memcpy(c->h, iv, sizeof iv); c->len = 0; c->n = 0;
}

void ds4_sha256_update(ds4_sha256_ctx *c, const void *data, size_t n) {
    const uint8_t *p = (const uint8_t *)data;
    c->len += n;
    while (n) {
        if (!c->n && n >= 64) { block(c->h, p); p += 64; n -= 64; continue; }
        size_t k = 64 - c->n; if (k > n) k = n;
        memcpy(c->buf + c->n, p, k); c->n += k; p += k; n -= k;
        if (c->n == 64) { block(c->h, c->buf); c->n = 0; }
    }
}

void ds4_sha256_final(ds4_sha256_ctx *c, uint8_t out[32]) {
    const uint64_t bits = c->len * 8;
    const uint8_t one = 0x80, zero = 0;
    ds4_sha256_update(c, &one, 1);
    while (c->n != 56) ds4_sha256_update(c, &zero, 1);
    uint8_t L[8];
    for (int i = 0; i < 8; i++) L[i] = (uint8_t)(bits >> (56 - 8 * i));
    ds4_sha256_update(c, L, 8);
    for (int i = 0; i < 8; i++) { out[4 * i] = (uint8_t)(c->h[i] >> 24); out[4 * i + 1] = (uint8_t)(c->h[i] >> 16); out[4 * i + 2] = (uint8_t)(c->h[i] >> 8); out[4 * i + 3] = (uint8_t)c->h[i]; }
}

void ds4_sha256(const void *p, size_t n, uint8_t out[32]) {
    ds4_sha256_ctx c; ds4_sha256_init(&c); ds4_sha256_update(&c, p, n); ds4_sha256_final(&c, out);
}

/* 内外两层的"键 ⊕ pad"只吃一个块, 预先算好状态: PBKDF2 每轮就只剩两次 32 字节的收尾, 不用每轮重做键块 */
typedef struct { ds4_sha256_ctx in, out; } hmac_key;

static void hmac_prep(hmac_key *hk, const void *key, size_t kn) {
    uint8_t kb[64] = {0}, pad[64];
    if (kn > 64) ds4_sha256(key, kn, kb); else memcpy(kb, key, kn);
    for (int i = 0; i < 64; i++) pad[i] = kb[i] ^ 0x36;
    ds4_sha256_init(&hk->in); ds4_sha256_update(&hk->in, pad, 64);
    for (int i = 0; i < 64; i++) pad[i] = kb[i] ^ 0x5c;
    ds4_sha256_init(&hk->out); ds4_sha256_update(&hk->out, pad, 64);
}

static void hmac_run(const hmac_key *hk, const void *msg, size_t mn, uint8_t out[32]) {
    ds4_sha256_ctx c = hk->in; uint8_t ih[32];
    ds4_sha256_update(&c, msg, mn); ds4_sha256_final(&c, ih);
    c = hk->out; ds4_sha256_update(&c, ih, 32); ds4_sha256_final(&c, out);
}

void ds4_hmac_sha256(const void *key, size_t kn, const void *msg, size_t mn, uint8_t out[32]) {
    hmac_key hk; hmac_prep(&hk, key, kn); hmac_run(&hk, msg, mn, out);
}

void ds4_pbkdf2_sha256(const void *pw, size_t pn, const void *salt, size_t sn, uint32_t iter, uint8_t *out, size_t outn) {
    hmac_key hk; hmac_prep(&hk, pw, pn);
    for (uint32_t blk = 1; outn; blk++) {
        const uint8_t be[4] = { (uint8_t)(blk >> 24), (uint8_t)(blk >> 16), (uint8_t)(blk >> 8), (uint8_t)blk };
        ds4_sha256_ctx c = hk.in; uint8_t ih[32], u[32], t[32];
        ds4_sha256_update(&c, salt, sn); ds4_sha256_update(&c, be, 4); ds4_sha256_final(&c, ih);
        c = hk.out; ds4_sha256_update(&c, ih, 32); ds4_sha256_final(&c, u);
        memcpy(t, u, 32);
        for (uint32_t i = 1; i < iter; i++) { hmac_run(&hk, u, 32, u); for (int j = 0; j < 32; j++) t[j] ^= u[j]; }
        const size_t k = outn < 32 ? outn : 32;
        memcpy(out, t, k); out += k; outn -= k;
    }
}
