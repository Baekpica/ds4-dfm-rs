/* ds4_gguf.h — GGUF v3 容器只读解析的全仓唯一实现。
 *
 * 此前 GGUF 头解析在仓里有 9 份人肉副本(ds4.c / zsolve / deepseek4-quantize /
 * dump_gguf_meta / vq_merge_v4 / emit_z / legacy 四件 / zlayer), 格式一变就是
 * 9 处改动面。本模块收敛谱系最清楚的一支: vq_merge_v4.c parse_header(有"与
 * .py 逐字节 md5 相同"金标) → zlayer.c gg_open(2026-08-25 过闸) → 这里。
 *
 * 范围: mmap 只读 + 张量目录。KV 段按类型跳过(与 gguf-py 同一 13 类型表);
 * 需要读 KV 值的消费方(dump_gguf_meta)在阶段7迁移时补 typed getter。
 * 错误口径: 返回 -1 + err 缓冲, 不打印不退出 —— die 还是继续由调用方定。 */
#ifndef DS4_COMMON_GGUF_H
#define DS4_COMMON_GGUF_H

#include <stdint.h>
#include <stddef.h>

typedef struct {
    char *name;
    uint32_t nd, type;
    uint64_t ne[4];       /* 未用维=1 */
    uint64_t off;         /* 相对数据区起点 */
} ds4_gguf_tensor;

typedef struct {
    const uint8_t *map;   /* 整文件 mmap(只读) */
    size_t msz;
    ds4_gguf_tensor *t;
    int nt;
    uint64_t data0;       /* 数据区起点(alignment=32 对齐后) */
    uint64_t kv0, nkv;    /* KV 段起点(=24)与条数: ds4_gguf_kv 按需重走, 打开时不解值 */
} ds4_gguf;

/* 按键找 KV 值: 返回值字节起点(类型码之后), *type = GGUF 类型码(8 字串: u64 长+字节; 9 数组: u32 元素类型+u64 个数+逐项);
 * 没有返回 NULL。调用方按类型自己读 —— 2026-10-13 旁白小模型读音素表首用。 */
const uint8_t *ds4_gguf_kv(const ds4_gguf *g, const char *key, uint32_t *type);

/* 打开并解析目录。成功 0; 失败 -1 且 err(可 NULL)带一句人话。 */
int ds4_gguf_open(ds4_gguf *g, const char *path, char *err, size_t errlen);
void ds4_gguf_close(ds4_gguf *g);

/* 按名找张量; 无则 NULL。 */
const ds4_gguf_tensor *ds4_gguf_find(const ds4_gguf *g, const char *name);

/* 张量数据指针与字节数(按类型几何算; 类型不认识返回 NULL)。 */
const uint8_t *ds4_gguf_tensor_data(const ds4_gguf *g, const ds4_gguf_tensor *t,
                                    uint64_t *nbytes_out);

#endif /* DS4_COMMON_GGUF_H */
