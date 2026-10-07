/* Metadata-only native tokenizer oracle. No session or inference is created. */
#define DS4_NO_GPU
#include "../ds4.c"

enum { MAX_TEXT_BYTES = 16 * 1024 * 1024 };

int main(int argc, char **argv) {
    if (argc != 4) {
        fprintf(stderr, "usage: %s MODEL TEXT TOKENS.i32\n", argv[0]);
        return 2;
    }
    FILE *fp = fopen(argv[2], "rb");
    if (!fp) { return 1; }
    if (fseeko(fp, 0, SEEK_END)) { fclose(fp); return 1; }
    const off_t size = ftello(fp);
    if (size <= 0 || size > MAX_TEXT_BYTES || fseeko(fp, 0, SEEK_SET)) {
        fclose(fp);
        return 1;
    }
    char *text = xmalloc((size_t)size + 1u);
    const size_t got = fread(text, 1u, (size_t)size, fp);
    fclose(fp);
    if (got != (size_t)size) { free(text); return 1; }
    text[size] = 0;

    ds4_model model;
    model_open(&model, argv[1], false, false);
    config_validate_model(&model);
    ds4_vocab vocab;
    vocab_load(&vocab, &model);
    token_vec tokens = {0};
    tokenize_rendered_chat_vocab(&vocab, text, &tokens);
    fp = fopen(argv[3], "wb");
    int ok = fp && fwrite(tokens.v, sizeof(*tokens.v), (size_t)tokens.len, fp) == (size_t)tokens.len;
    if (fp && fclose(fp)) { ok = 0; }
    printf("{\"tokens\":%d}\n", tokens.len);
    token_vec_free(&tokens);
    vocab_free(&vocab);
    model_close(&model);
    free(text);
    return ok ? 0 : 1;
}
