/*
 * h3_tok_probe — encode stdin lines with the reference engine's own tokenizer.
 *
 * The tokenizer gate runs this against `ds4-core`'s
 * `minimax_h3::tokenizer` on the same lines. The tokenizer compiled here is
 * `/data/h3.c` at eae5d54 (`h3_tokenizer.c`), unmodified; only this driver is
 * ours. `pad_empty` is 1, as the engine's prompt path passes it
 * (`h3.c:1419`).
 *
 * Usage: h3_tok_probe <tokenizer.json> < lines
 * Output: one line of space-separated ids per input line.
 */

#include "h3_tokenizer.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: h3_tok_probe <tokenizer.json>\n");
        return 2;
    }

    char error[512];
    h3_tokenizer *tokenizer = h3_tokenizer_load(argv[1], error, sizeof(error));
    if (!tokenizer) {
        fprintf(stderr, "%s\n", error);
        return 1;
    }

    char *line = NULL;
    size_t capacity = 0;
    ssize_t length;

    while ((length = getline(&line, &capacity, stdin)) >= 0) {
        if (length > 0 && line[length - 1] == '\n') {
            line[--length] = '\0';
        }

        uint32_t *ids = NULL;
        size_t count = 0;

        if (!h3_tokenizer_encode(tokenizer, line, 1, &ids, &count,
                                 error, sizeof(error))) {
            fprintf(stderr, "%s\n", error);
            free(line);
            h3_tokenizer_free(tokenizer);
            return 1;
        }

        for (size_t index = 0; index < count; index++) {
            printf("%s%u", index ? " " : "", ids[index]);
        }
        putchar('\n');
        h3_tokenizer_ids_free(ids);
    }

    free(line);
    h3_tokenizer_free(tokenizer);
    return 0;
}
