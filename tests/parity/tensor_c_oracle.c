/* GGUF tensor directory + split sibling + nbytes oracle from ds4.c at v0.6.3-dfm.
 * Standalone: do not include ds4.c. */

#include <fcntl.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#define DS4_GGUF_MAGIC 0x46554747u
#define DS4_MAX_DIMS 8

enum {
    GGUF_VALUE_UINT8 = 0, GGUF_VALUE_INT8 = 1, GGUF_VALUE_UINT16 = 2,
    GGUF_VALUE_INT16 = 3, GGUF_VALUE_UINT32 = 4, GGUF_VALUE_INT32 = 5,
    GGUF_VALUE_FLOAT32 = 6, GGUF_VALUE_BOOL = 7, GGUF_VALUE_STRING = 8,
    GGUF_VALUE_ARRAY = 9, GGUF_VALUE_UINT64 = 10, GGUF_VALUE_INT64 = 11,
    GGUF_VALUE_FLOAT64 = 12,
};

typedef struct { const char *name; uint32_t block_elems; uint32_t block_bytes; } gguf_type_info;
static const gguf_type_info gguf_types[] = {
    [0]  = {"f32", 1, 4}, [1] = {"f16", 1, 2}, [2] = {"q4_0", 32, 18},
    [3]  = {"q4_1", 32, 20}, [6] = {"q5_0", 32, 22}, [7] = {"q5_1", 32, 24},
    [8]  = {"q8_0", 32, 34}, [9] = {"q8_1", 32, 40}, [10] = {"q2_k", 256, 84},
    [11] = {"q3_k", 256, 110}, [12] = {"q4_k", 256, 144}, [13] = {"q5_k", 256, 176},
    [14] = {"q6_k", 256, 210}, [15] = {"q8_k", 256, 292}, [16] = {"iq2_xxs", 256, 66},
    [17] = {"iq2_xs", 256, 74}, [18] = {"iq3_xxs", 256, 98}, [19] = {"iq1_s", 256, 110},
    [20] = {"iq4_nl", 256, 50}, [21] = {"iq3_s", 256, 110}, [22] = {"iq2_s", 256, 82},
    [23] = {"iq4_xs", 256, 136}, [24] = {"i8", 1, 1}, [25] = {"i16", 1, 2},
    [26] = {"i32", 1, 4}, [27] = {"i64", 1, 8}, [28] = {"f64", 1, 8},
    [29] = {"iq1_m", 256, 56}, [30] = {"bf16", 1, 2},
};

static uint64_t align_up(uint64_t value, uint64_t alignment) {
    uint64_t rem = value % alignment;
    return rem == 0 ? value : value + alignment - rem;
}

static const char *tensor_type_name(uint32_t type) {
    uint32_t n = sizeof(gguf_types) / sizeof(gguf_types[0]);
    if (type >= n || gguf_types[type].name == NULL) return "unknown";
    return gguf_types[type].name;
}

static bool tensor_nbytes(uint32_t type, uint64_t elements, uint64_t *bytes) {
    uint32_t n = sizeof(gguf_types) / sizeof(gguf_types[0]);
    if (type >= n || gguf_types[type].name == NULL || gguf_types[type].block_elems == 0)
        return false;
    uint64_t blocks = (elements + gguf_types[type].block_elems - 1) / gguf_types[type].block_elems;
    if (blocks > UINT64_MAX / gguf_types[type].block_bytes) return false;
    *bytes = blocks * gguf_types[type].block_bytes;
    return true;
}

static bool model_split_sibling_path(const char *path, uint32_t index,
                                     uint32_t count, char *out, size_t out_len) {
    const char *dash = strrchr(path, '-');
    if (!dash) return false;
    unsigned parsed_count = 0;
    if (sscanf(dash, "-%05u.gguf", &parsed_count) != 1 || parsed_count != count)
        return false;
    if (dash - path < 9) return false;
    const char *of = dash - 3;
    if (strncmp(of, "-of", 3) != 0) return false;
    const char *num = of - 5;
    if (num - path < 1 || num[-1] != '-') return false;
    for (int i = 0; i < 5; i++) {
        if (num[i] < '0' || num[i] > '9') return false;
    }
    const size_t prefix_len = (size_t)(num - path);
    const int n = snprintf(out, out_len, "%.*s%05u-of-%05u.gguf",
                           (int)prefix_len, path, index + 1u, count);
    return n > 0 && (size_t)n < out_len;
}

static void die(const char *m) { fprintf(stderr, "tensor_c_oracle: %s\n", m); exit(2); }

static void dump_nbytes(void) {
    static const uint64_t elems[] = {1, 31, 32, 256, 257};
    for (uint32_t typ = 0; typ <= 30; typ++) {
        for (size_t i = 0; i < sizeof(elems) / sizeof(elems[0]); i++) {
            uint64_t b = 0;
            if (tensor_nbytes(typ, elems[i], &b))
                printf("NBYTES type=%u name=%s elems=%llu bytes=%llu\n",
                       typ, tensor_type_name(typ),
                       (unsigned long long)elems[i], (unsigned long long)b);
            else
                printf("NBYTES type=%u name=%s elems=%llu FAIL\n",
                       typ, tensor_type_name(typ), (unsigned long long)elems[i]);
        }
    }
}

static uint32_t ru32(const uint8_t *p) {
    uint32_t v; memcpy(&v, p, 4); return v;
}
static uint64_t ru64(const uint8_t *p) {
    uint64_t v; memcpy(&v, p, 8); return v;
}

static bool skip_value(const uint8_t *base, uint64_t size, uint64_t *pos, uint32_t typ, int depth);

static bool skip_value(const uint8_t *base, uint64_t size, uint64_t *pos, uint32_t typ, int depth) {
    static const uint64_t scalar[] = {
        1, 1, 2, 2, 4, 4, 4, 1, 0, 0, 8, 8, 8
    };
    if (depth > 8) return false;
    if (typ <= 12 && scalar[typ]) {
        if (*pos + scalar[typ] > size) return false;
        *pos += scalar[typ];
        return true;
    }
    if (typ == GGUF_VALUE_STRING) {
        if (*pos + 8 > size) return false;
        uint64_t n = ru64(base + *pos); *pos += 8;
        if (*pos + n > size) return false;
        *pos += n;
        return true;
    }
    if (typ == GGUF_VALUE_ARRAY) {
        if (*pos + 4 + 8 > size) return false;
        uint32_t it = ru32(base + *pos); *pos += 4;
        uint64_t len = ru64(base + *pos); *pos += 8;
        for (uint64_t i = 0; i < len; i++)
            if (!skip_value(base, size, pos, it, depth + 1)) return false;
        return true;
    }
    return false;
}

static int dump_parse(const char *path) {
    int fd = open(path, O_RDONLY);
    if (fd < 0) { printf("ERROR io\n"); return 0; }
    struct stat st;
    if (fstat(fd, &st) || st.st_size < 32) { close(fd); printf("ERROR too-small\n"); return 0; }
    uint64_t size = (uint64_t)st.st_size;
    void *map = mmap(NULL, (size_t)size, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    if (map == MAP_FAILED) { printf("ERROR io\n"); return 0; }
    const uint8_t *base = map;
    if (size < 24 || ru32(base) != DS4_GGUF_MAGIC) {
        munmap(map, (size_t)size); printf("ERROR not-gguf\n"); return 0;
    }
    uint32_t version = ru32(base + 4);
    uint64_t n_tensors = ru64(base + 8);
    uint64_t n_kv = ru64(base + 16);
    if (version != 3) { munmap(map, (size_t)size); printf("ERROR unsupported-version\n"); return 0; }
    uint64_t pos = 24;
    uint64_t alignment = 32;
    for (uint64_t i = 0; i < n_kv; i++) {
        if (pos + 8 > size) { munmap(map, (size_t)size); printf("ERROR truncated\n"); return 0; }
        uint64_t klen = ru64(base + pos); pos += 8;
        if (pos + klen + 4 > size) { munmap(map, (size_t)size); printf("ERROR truncated\n"); return 0; }
        const uint8_t *key = base + pos; pos += klen;
        uint32_t typ = ru32(base + pos); pos += 4;
        if (klen == 17 && memcmp(key, "general.alignment", 17) == 0 && typ == GGUF_VALUE_UINT32
            && pos + 4 <= size) {
            uint32_t a = ru32(base + pos);
            if (a) alignment = a;
        }
        if (!skip_value(base, size, &pos, typ, 0)) {
            munmap(map, (size_t)size); printf("ERROR truncated\n"); return 0;
        }
    }
    uint64_t dir = pos;
    typedef struct {
        char name[256]; uint32_t ndim; uint64_t dim[8]; uint32_t type;
        uint64_t rel, elems, bytes, abs;
    } tens;
    tens *tv = calloc((size_t)n_tensors, sizeof(*tv));
    if (!tv && n_tensors) die("oom");
    for (uint64_t i = 0; i < n_tensors; i++) {
        if (pos + 8 > size) { printf("ERROR truncated\n"); goto out; }
        uint64_t nlen = ru64(base + pos); pos += 8;
        if (pos + nlen + 4 > size) { printf("ERROR truncated\n"); goto out; }
        size_t copy = nlen < 255 ? (size_t)nlen : 255;
        memcpy(tv[i].name, base + pos, copy); tv[i].name[copy] = 0;
        pos += nlen;
        tv[i].ndim = ru32(base + pos); pos += 4;
        if (tv[i].ndim == 0 || tv[i].ndim > DS4_MAX_DIMS) { printf("ERROR bad-dims\n"); goto out; }
        tv[i].elems = 1;
        for (uint32_t d = 0; d < tv[i].ndim; d++) {
            if (pos + 8 > size) { printf("ERROR truncated\n"); goto out; }
            tv[i].dim[d] = ru64(base + pos); pos += 8;
            if (tv[i].dim[d] != 0 && tv[i].elems > UINT64_MAX / tv[i].dim[d]) {
                printf("ERROR overflow\n"); goto out;
            }
            tv[i].elems *= tv[i].dim[d];
        }
        if (pos + 12 > size) { printf("ERROR truncated\n"); goto out; }
        tv[i].type = ru32(base + pos); pos += 4;
        tv[i].rel = ru64(base + pos); pos += 8;
        if (!tensor_nbytes(tv[i].type, tv[i].elems, &tv[i].bytes)) tv[i].bytes = 0;
    }
    uint64_t data_pos = align_up(pos, alignment);
    for (uint64_t i = 0; i < n_tensors; i++) {
        if (tv[i].rel > UINT64_MAX - data_pos) { printf("ERROR overflow\n"); goto out; }
        tv[i].abs = data_pos + tv[i].rel;
        if (tv[i].bytes != 0 && (tv[i].abs > size || tv[i].bytes > size - tv[i].abs)) {
            printf("ERROR outside-file\n"); goto out;
        }
    }
    {
        long page_l = sysconf(_SC_PAGESIZE);
        uint64_t page = page_l > 0 ? (uint64_t)page_l : 4096u;
        printf("DATA_POS %llu ALIGN %llu PAGE %llu SHARDS 1 N %llu\n",
               (unsigned long long)data_pos, (unsigned long long)alignment,
               (unsigned long long)page, (unsigned long long)n_tensors);
        printf("SHARD %s size=%llu base=0\n", path, (unsigned long long)size);
        for (uint64_t i = 0; i < n_tensors; i++) {
            printf("T %s ndim=%u dims=", tv[i].name, tv[i].ndim);
            for (uint32_t d = 0; d < tv[i].ndim; d++) {
                if (d) putchar(',');
                printf("%llu", (unsigned long long)tv[i].dim[d]);
            }
            printf(" type=%u(%s) elems=%llu bytes=%llu rel=%llu abs=%llu shard=0\n",
                   tv[i].type, tensor_type_name(tv[i].type),
                   (unsigned long long)tv[i].elems, (unsigned long long)tv[i].bytes,
                   (unsigned long long)tv[i].rel, (unsigned long long)tv[i].abs);
        }
        (void)dir;
    }
out:
    free(tv);
    munmap(map, (size_t)size);
    return 0;
}

int main(int argc, char **argv) {
    const char *cmd = argc > 1 ? argv[1] : "";
    if (!strcmp(cmd, "nbytes")) {
        dump_nbytes();
        return 0;
    }
    if (!strcmp(cmd, "sibling") && argc >= 5) {
        char out[4096];
        unsigned index = (unsigned)strtoul(argv[3], NULL, 10);
        unsigned count = (unsigned)strtoul(argv[4], NULL, 10);
        if (model_split_sibling_path(argv[2], index, count, out, sizeof(out)))
            printf("SIBLING %s\n", out);
        else
            printf("SIBLING FAIL\n");
        return 0;
    }
    if (!strcmp(cmd, "parse") && argc >= 3) {
        return dump_parse(argv[2]);
    }
    fprintf(stderr, "usage: tensor_c_oracle nbytes|sibling PATH INDEX COUNT|parse FILE\n");
    return 2;
}
