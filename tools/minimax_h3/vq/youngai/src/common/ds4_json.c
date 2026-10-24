/* ds4_json.c — JSON 词法基元 + 字节缓冲(接口见 ds4_json.h)。从 server_buf.c 机械挪来, 行为不变; 只把 die()/xmalloc 换成本地的 abort 路。 */
#include "ds4_json.h"
#include <ctype.h>
#include <limits.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void *jmalloc(size_t n) {
    void *p = malloc(n ? n : 1);
    if (!p) { fprintf(stderr, "ds4_json: out of memory\n"); abort(); }
    return p;
}

static void ds4_buf_reserve(ds4_buf *b, size_t add) {
    if (add > SIZE_MAX - b->len - 1) { fprintf(stderr, "ds4_json: buffer overflow\n"); abort(); }
    size_t need = b->len + add + 1;
    if (need <= b->cap) return;
    size_t cap = b->cap ? b->cap * 2 : 256;
    while (cap < need) cap *= 2;
    char *p = realloc(b->ptr, cap);
    if (!p) { fprintf(stderr, "ds4_json: out of memory\n"); abort(); }
    b->ptr = p;
    b->cap = cap;
}

void ds4_buf_append(ds4_buf *b, const void *p, size_t n) {
    ds4_buf_reserve(b, n);
    memcpy(b->ptr + b->len, p, n);
    b->len += n;
    b->ptr[b->len] = '\0';
}

void ds4_buf_putc(ds4_buf *b, char c) { ds4_buf_append(b, &c, 1); }
void ds4_buf_puts(ds4_buf *b, const char *s) { ds4_buf_append(b, s, strlen(s)); }

void ds4_buf_printf(ds4_buf *b, const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    va_list ap2;
    va_copy(ap2, ap);
    int n = vsnprintf(NULL, 0, fmt, ap);
    va_end(ap);
    if (n < 0) { fprintf(stderr, "ds4_json: vsnprintf failed\n"); abort(); }
    ds4_buf_reserve(b, (size_t)n);
    vsnprintf(b->ptr + b->len, b->cap - b->len, fmt, ap2);
    va_end(ap2);
    b->len += (size_t)n;
}

char *ds4_buf_take(ds4_buf *b) {
    if (!b->ptr) { char *e = jmalloc(1); e[0] = '\0'; return e; }
    char *p = b->ptr;
    memset(b, 0, sizeof(*b));
    return p;
}

void ds4_buf_free(ds4_buf *b) {
    free(b->ptr);
    memset(b, 0, sizeof(*b));
}

void json_ws(const char **p) {
    while (**p && isspace((unsigned char)**p)) (*p)++;
}

bool json_lit(const char **p, const char *lit) {
    size_t n = strlen(lit);
    if (strncmp(*p, lit, n) != 0) return false;
    *p += n;
    return true;
}

static int json_hex(char c) {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return 10 + c - 'a';
    if (c >= 'A' && c <= 'F') return 10 + c - 'A';
    return -1;
}

static void utf8_put(ds4_buf *b, uint32_t cp) {
    if (cp <= 0x7f) {
        ds4_buf_putc(b, (char)cp);
    } else if (cp <= 0x7ff) {
        ds4_buf_putc(b, (char)(0xc0 | (cp >> 6)));
        ds4_buf_putc(b, (char)(0x80 | (cp & 0x3f)));
    } else if (cp <= 0xffff) {
        ds4_buf_putc(b, (char)(0xe0 | (cp >> 12)));
        ds4_buf_putc(b, (char)(0x80 | ((cp >> 6) & 0x3f)));
        ds4_buf_putc(b, (char)(0x80 | (cp & 0x3f)));
    } else {
        ds4_buf_putc(b, (char)(0xf0 | (cp >> 18)));
        ds4_buf_putc(b, (char)(0x80 | ((cp >> 12) & 0x3f)));
        ds4_buf_putc(b, (char)(0x80 | ((cp >> 6) & 0x3f)));
        ds4_buf_putc(b, (char)(0x80 | (cp & 0x3f)));
    }
}

static bool json_u16(const char **p, uint32_t *out) {
    if ((*p)[0] != '\\' || (*p)[1] != 'u') return false;
    uint32_t cp = 0;
    for (int i = 0; i < 4; i++) {
        int h = json_hex((*p)[2 + i]);
        if (h < 0) return false;
        cp = (cp << 4) | (uint32_t)h;
    }
    *p += 6;
    *out = cp;
    return true;
}

bool json_string(const char **p, char **out) {
    json_ws(p);
    if (**p != '"') return false;
    (*p)++;
    ds4_buf b = {0};
    while (**p && **p != '"') {
        unsigned char c = (unsigned char)*(*p)++;
        if (c != '\\') {
            ds4_buf_putc(&b, (char)c);
            continue;
        }
        c = (unsigned char)*(*p)++;
        switch (c) {
        case '"': ds4_buf_putc(&b, '"'); break;
        case '\\': ds4_buf_putc(&b, '\\'); break;
        case '/': ds4_buf_putc(&b, '/'); break;
        case 'b': ds4_buf_putc(&b, '\b'); break;
        case 'f': ds4_buf_putc(&b, '\f'); break;
        case 'n': ds4_buf_putc(&b, '\n'); break;
        case 'r': ds4_buf_putc(&b, '\r'); break;
        case 't': ds4_buf_putc(&b, '\t'); break;
        case 'u': {
            *p -= 2;
            uint32_t cp = 0, lo = 0;
            if (!json_u16(p, &cp)) goto fail;
            if (cp >= 0xd800 && cp <= 0xdbff && json_u16(p, &lo) && lo >= 0xdc00 && lo <= 0xdfff) {
                cp = 0x10000u + ((cp - 0xd800u) << 10) + (lo - 0xdc00u);
            }
            utf8_put(&b, cp);
            break;
        }
        default:
            goto fail;
        }
    }
    if (**p != '"') goto fail;
    (*p)++;
    *out = ds4_buf_take(&b);
    return true;
fail:
    ds4_buf_free(&b);
    return false;
}

bool json_number(const char **p, double *out) {
    json_ws(p);
    char *end = NULL;
    double v = strtod(*p, &end);
    if (end == *p) return false;
    *p = end;
    *out = v;
    return true;
}

bool json_int(const char **p, int *out) {
    double v = 0.0;
    if (!json_number(p, &v)) return false;
    if (v < 0) v = 0;
    if (v > INT_MAX) v = INT_MAX;
    *out = (int)v;
    return true;
}

bool json_bool(const char **p, bool *out) {
    json_ws(p);
    if (json_lit(p, "true")) { *out = true; return true; }
    if (json_lit(p, "false")) { *out = false; return true; }
    return false;
}

static bool json_skip_value_depth(const char **p, int depth);

static bool json_skip_array_depth(const char **p, int depth) {
    if (depth >= JSON_MAX_NESTING) return false;
    json_ws(p);
    if (**p != '[') return false;
    (*p)++;
    json_ws(p);
    if (**p == ']') { (*p)++; return true; }
    for (;;) {
        if (!json_skip_value_depth(p, depth + 1)) return false;
        json_ws(p);
        if (**p == ']') { (*p)++; return true; }
        if (**p != ',') return false;
        (*p)++;
    }
}

static bool json_skip_object_depth(const char **p, int depth) {
    if (depth >= JSON_MAX_NESTING) return false;
    json_ws(p);
    if (**p != '{') return false;
    (*p)++;
    json_ws(p);
    if (**p == '}') { (*p)++; return true; }
    for (;;) {
        char *key = NULL;
        if (!json_string(p, &key)) return false;
        free(key);
        json_ws(p);
        if (**p != ':') return false;
        (*p)++;
        if (!json_skip_value_depth(p, depth + 1)) return false;
        json_ws(p);
        if (**p == '}') { (*p)++; return true; }
        if (**p != ',') return false;
        (*p)++;
    }
}

static bool json_skip_value_depth(const char **p, int depth) {
    json_ws(p);
    if (**p == '"') {
        char *s = NULL;
        bool ok = json_string(p, &s);
        free(s);
        return ok;
    }
    if (**p == '{') return json_skip_object_depth(p, depth);
    if (**p == '[') return json_skip_array_depth(p, depth);
    if (json_lit(p, "true") || json_lit(p, "false") || json_lit(p, "null")) return true;
    double v = 0.0;
    return json_number(p, &v);
}

bool json_skip_value(const char **p) { return json_skip_value_depth(p, 0); }

bool json_raw_value(const char **p, char **out) {
    json_ws(p);
    const char *start = *p;
    if (!json_skip_value(p)) return false;
    size_t n = (size_t)(*p - start);
    char *s = jmalloc(n + 1);
    memcpy(s, start, n);
    s[n] = '\0';
    *out = s;
    return true;
}

void ds4_json_escape(ds4_buf *b, const char *s) {
    ds4_buf_putc(b, '"');
    for (; *s; s++) {
        unsigned char c = (unsigned char)*s;
        if (c == '"' || c == '\\') { ds4_buf_putc(b, '\\'); ds4_buf_putc(b, (char)c); }
        else if (c == '\n') ds4_buf_puts(b, "\\n");
        else if (c == '\r') ds4_buf_puts(b, "\\r");
        else if (c == '\t') ds4_buf_puts(b, "\\t");
        else if (c < 0x20) ds4_buf_printf(b, "\\u%04x", (unsigned)c);
        else ds4_buf_putc(b, (char)c);
    }
    ds4_buf_putc(b, '"');
}
