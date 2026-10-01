// PQ2_0 / GGUF dequant oracle: dequantizes tensor rows with the PrismML fork's
// own ggml reference code and prints a stable checksum, so a from-scratch
// implementation (ds4) can be verified bit-exactly.
//
// Build (from the fork workspace):
//   cc -O2 pq2_ref.c -o pq2_ref \
//     -I<fork>/ggml/include -L<fork>/build/bin \
//     -lggml-base -Wl,-rpath,<fork>/build/bin -lm
//
// Usage: pq2_ref <model.gguf> [max_rows_per_tensor] [tensor_name_substring]

#include "ggml.h"
#include "gguf.h"

#include <fcntl.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

static uint64_t fnv1a(uint64_t h, const void * p, size_t n) {
    const uint8_t * b = (const uint8_t *) p;
    for (size_t i = 0; i < n; i++) {
        h ^= b[i];
        h *= 0x100000001b3ull;
    }
    return h;
}

int main(int argc, char ** argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s model.gguf [max_rows] [substring]\n", argv[0]);
        return 2;
    }
    const char * path = argv[1];
    long max_rows = argc > 2 ? strtol(argv[2], NULL, 10) : 64;
    const char * filt = argc > 3 ? argv[3] : "";

    int fd = open(path, O_RDONLY);
    if (fd < 0) { perror("open"); return 1; }
    struct stat st;
    if (fstat(fd, &st) != 0) { perror("fstat"); return 1; }
    const uint8_t * base = mmap(NULL, (size_t) st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (base == MAP_FAILED) { perror("mmap"); return 1; }

    struct ggml_context * data_ctx = NULL;
    struct gguf_init_params params = { .no_alloc = true, .ctx = &data_ctx };
    struct gguf_context * ctx = gguf_init_from_file(path, params);
    if (ctx == NULL) { fprintf(stderr, "gguf_init_from_file failed\n"); return 1; }

    const uint64_t data_off = gguf_get_data_offset(ctx);
    const int64_t  n_tensor = gguf_get_n_tensors(ctx);

    for (int64_t i = 0; i < n_tensor; i++) {
        const char * name = gguf_get_tensor_name(ctx, i);
        if (filt[0] && strstr(name, filt) == NULL) continue;

        struct ggml_tensor * t = ggml_get_tensor(data_ctx, name);
        if (t == NULL) { fprintf(stderr, "missing tensor %s\n", name); return 1; }

        const size_t row_bytes = ggml_row_size(t->type, t->ne[0]);
        const int64_t n_rows = (int64_t) ggml_nrows(t);
        const int64_t rows = max_rows > 0 && n_rows > max_rows ? max_rows : n_rows;

        const uint8_t * rows_base = base + data_off + gguf_get_tensor_offset(ctx, i);

        float * buf = (float *) malloc((size_t) t->ne[0] * sizeof(float));
        if (buf == NULL) { fprintf(stderr, "oom\n"); return 1; }

        const struct ggml_type_traits * tr = ggml_get_type_traits(t->type);
        uint64_t h = 0xcbf29ce484222325ull;
        double sum = 0.0;
        for (int64_t r = 0; r < rows; r++) {
            const void * src = rows_base + (size_t) r * row_bytes;
            if (tr->to_float != NULL) {
                tr->to_float(src, buf, t->ne[0]);
            } else if (t->type == GGML_TYPE_F32) {
                memcpy(buf, src, (size_t) t->ne[0] * sizeof(float));
            } else if (t->type == GGML_TYPE_F16) {
                ggml_fp16_to_fp32_row((const ggml_fp16_t *) src, buf, t->ne[0]);
            } else if (t->type == GGML_TYPE_BF16) {
                ggml_bf16_to_fp32_row((const ggml_bf16_t *) src, buf, t->ne[0]);
            } else {
                fprintf(stderr, "no dequant path for %s type %d\n", name, (int) t->type);
                return 1;
            }
            h = fnv1a(h, buf, (size_t) t->ne[0] * sizeof(float));
            for (int64_t k = 0; k < t->ne[0]; k++) sum += buf[k];
        }

        printf("%s type=%d ne0=%" PRId64 " rows=%" PRId64 "/%" PRId64
               " row_bytes=%zu checksum=%016" PRIx64 " sum=%.6f head=[%.7g %.7g %.7g %.7g]\n",
               name, (int) t->type, t->ne[0], rows, n_rows, row_bytes, h,
               sum, buf[0], buf[1 % t->ne[0]], buf[2 % t->ne[0]], buf[3 % t->ne[0]]);
        free(buf);
    }

    gguf_free(ctx);
    if (data_ctx) ggml_free(data_ctx);
    munmap((void *) base, (size_t) st.st_size);
    close(fd);
    return 0;
}
