/*
 * h3_st_probe — inventory a MiniMax-H3 safetensors directory with the
 * reference engine's own reader.
 *
 * P3's two-sided diff runs this against `ds4-core`'s
 * `minimax_h3::safetensors` on the same bytes. The parser compiled here is
 * `/data/h3.c` at eae5d54 (`h3_safetensors.c`, `h3_weights.c`), unmodified;
 * only this driver is ours.
 *
 * Usage: h3_st_probe <dir>
 * Output: one TSV row per tensor, then a TOTAL line.
 */

#include "h3_safetensors.h"

#include <dirent.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int has_suffix(const char *name) {
    static const char suffix[] = ".safetensors";
    size_t length = strlen(name);
    size_t suffix_length = sizeof(suffix) - 1;
    return length > suffix_length &&
           strcmp(name + length - suffix_length, suffix) == 0;
}

static int compare(const void *left, const void *right) {
    const char *const *a = left;
    const char *const *b = right;
    return strcmp(*a, *b);
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: h3_st_probe <dir>\n");
        return 2;
    }

    DIR *stream = opendir(argv[1]);
    if (!stream) {
        perror(argv[1]);
        return 1;
    }

    char **paths = NULL;
    size_t count = 0;
    size_t capacity = 0;
    struct dirent *entry;

    while ((entry = readdir(stream)) != NULL) {
        if (!has_suffix(entry->d_name)) {
            continue;
        }
        if (count == capacity) {
            size_t next = capacity ? capacity * 2 : 8;
            char **grown = realloc(paths, next * sizeof(*grown));
            if (!grown) {
                fprintf(stderr, "out of memory\n");
                return 1;
            }
            paths = grown;
            capacity = next;
        }
        size_t length = strlen(argv[1]) + strlen(entry->d_name) + 2;
        paths[count] = malloc(length);
        if (!paths[count]) {
            fprintf(stderr, "out of memory\n");
            return 1;
        }
        snprintf(paths[count], length, "%s/%s", argv[1], entry->d_name);
        count++;
    }
    closedir(stream);

    if (!count) {
        fprintf(stderr, "%s: no safetensors files\n", argv[1]);
        return 1;
    }

    qsort(paths, count, sizeof(*paths), compare);

    unsigned long long tensors = 0;
    unsigned long long tensor_bytes = 0;
    unsigned long long file_bytes = 0;

    for (size_t index = 0; index < count; index++) {
        h3_st_header header;
        char error[384];

        if (!h3_st_read_header(paths[index], &header, error, sizeof(error))) {
            fprintf(stderr, "%s\n", error);
            return 1;
        }

        const char *file = strrchr(paths[index], '/');
        file = file ? file + 1 : paths[index];

        for (size_t item = 0; item < header.tensor_count; item++) {
            const h3_st_tensor *tensor = &header.tensors[item];
            unsigned long long bytes = tensor->data_end - tensor->data_begin;

            printf("%s\t%s\t%s\t", file, tensor->name,
                   h3_dtype_name(tensor->dtype));

            for (int dimension = 0; dimension < tensor->ndim; dimension++) {
                printf("%s%llu", dimension ? "x" : "",
                       (unsigned long long)tensor->shape[dimension]);
            }

            printf("\t%llu\t%llu\n", bytes,
                   (unsigned long long)tensor->file_offset);

            tensors++;
            tensor_bytes += bytes;
        }

        file_bytes += header.file_size;
        h3_st_free_header(&header);
    }

    printf("TOTAL\tshards=%zu\ttensors=%llu\ttensor_bytes=%llu\tfile_bytes=%llu\n",
           count, tensors, tensor_bytes, file_bytes);
    return 0;
}
