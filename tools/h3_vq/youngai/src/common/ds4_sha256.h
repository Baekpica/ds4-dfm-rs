/* ds4_sha256.h — SHA-256 / HMAC-SHA256 / PBKDF2-HMAC-SHA256(FIPS 180-4, RFC 2104, RFC 8018)。
 * 为什么自己写: 工作台登录(src/studio/studio_auth.c)要存口令哈希、签会话票, 而全仓不链任何加密库
 * (Linux 上 OpenSSL 不保证在, Mac 的 CommonCrypto 又只在一边), 两百行纯 C 两边一样。
 * 只做哈希与派生, 不做加解密。金标向量在 tests/unit/t_sha256.c(NIST / RFC 4231 / RFC 7914)。 */
#ifndef DS4_SHA256_H
#define DS4_SHA256_H
#include <stddef.h>
#include <stdint.h>

typedef struct { uint32_t h[8]; uint64_t len; uint8_t buf[64]; size_t n; } ds4_sha256_ctx;

void ds4_sha256_init(ds4_sha256_ctx *c);
void ds4_sha256_update(ds4_sha256_ctx *c, const void *p, size_t n);
void ds4_sha256_final(ds4_sha256_ctx *c, uint8_t out[32]);
void ds4_sha256(const void *p, size_t n, uint8_t out[32]);
void ds4_hmac_sha256(const void *key, size_t kn, const void *msg, size_t mn, uint8_t out[32]);
/* 派生 outn 字节; iter 轮。耗时与 iter 成正比(本机 Mac 60 万轮实测 0.27 秒, 量法见 t_sha256.c) */
void ds4_pbkdf2_sha256(const void *pw, size_t pn, const void *salt, size_t sn, uint32_t iter, uint8_t *out, size_t outn);

#endif
