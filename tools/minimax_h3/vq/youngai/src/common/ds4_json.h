/* ds4_json.h — 全仓唯一的 JSON 词法基元 + 可增长字节缓冲。
 * 2026-10-10 从 src/server/server_buf.c 挪来: 训练器(src/core/core_ptrain_data.c)读 jsonl 料也要解 JSON 字符串,
 * 不许第二份实现(一件事只写一处)。服务端的 buf 类型就是这里的 ds4_buf(server_types.h typedef 一下), 调用方一行不改。
 * 内存不够直接 abort: 这些缓冲只装请求/料的文本, 分不到就是机器已经没救, 不装"返回 false 让上层处理"的样子。 */
#ifndef DS4_JSON_H
#define DS4_JSON_H
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

typedef struct {
    char *ptr;
    size_t len;
    size_t cap;
} ds4_buf;

void ds4_buf_append(ds4_buf *b, const void *p, size_t n);
void ds4_buf_putc(ds4_buf *b, char c);
void ds4_buf_puts(ds4_buf *b, const char *s);
void ds4_buf_printf(ds4_buf *b, const char *fmt, ...);
char *ds4_buf_take(ds4_buf *b);   /* 交出 malloc 的字符串(空缓冲给 ""), 缓冲清零 */
void ds4_buf_free(ds4_buf *b);

/* 解析只认用得上的字段、其余跳过。跳过是递归的(JSON 值会嵌套), 所以封一个顶:
 * 不封的话一个没用的 {"x":[[[...]]]} 就能在拒绝请求之前把整个 C 栈吃光。 */
#define JSON_MAX_NESTING 256

/* 词法: *p 指向当前位置, 成功就前移; 失败 *p 位置未定义(调用方整条放弃) */
void json_ws(const char **p);
bool json_lit(const char **p, const char *lit);
bool json_string(const char **p, char **out);   /* 带转义与 \uXXXX(含代理对)还原, *out 为 malloc 的 UTF-8 */
bool json_number(const char **p, double *out);
bool json_int(const char **p, int *out);        /* 负数钳到 0, 超 INT_MAX 钳到 INT_MAX */
bool json_bool(const char **p, bool *out);
bool json_skip_value(const char **p);
bool json_raw_value(const char **p, char **out);   /* 一个值的原文(malloc) */
void ds4_json_escape(ds4_buf *b, const char *s);   /* 带双引号的 JSON 字符串字面量追加进 b */

#endif
