/* C tokenizer oracle. Includes ds4.c (DS4_NO_GPU) so encode/decode/stop
 * match v0.6.5-dfm without linking CUDA. Synthetic metadata-only GGUFs
 * are opened with model_open; g_ds4_shape is assigned from the family
 * argument so vocab_load does not need config_validate_model. */

#include "../../ds4.c"

#include <ctype.h>

static void die_usage(void)
{
    fprintf(stderr,
            "usage: tokenizer_c_oracle FAMILY GGUF CMD [ARG]\n"
            "  FAMILY: deepseek4|motif3|solar-open2|exaone-moe|dots3-note|qwen4exp|glm5-next\n"
            "  CMD: specials | encode HEX | render HEX | decode ID | stop ID | chat MODE\n");
    exit(2);
}

static void set_family(const char *fam)
{
    if (strcmp(fam, "deepseek4") == 0) g_ds4_shape = DS4_SHAPE_FLASH;
    else if (strcmp(fam, "motif3") == 0) g_ds4_shape = DS4_SHAPE_MOTIF3;
    else if (strcmp(fam, "solar-open2") == 0) g_ds4_shape = DS4_SHAPE_SOLAR_OPEN2_250B;
    else if (strcmp(fam, "exaone-moe") == 0) g_ds4_shape = DS4_SHAPE_KEXAONE_236B;
    else if (strcmp(fam, "dots3-note") == 0) g_ds4_shape = DS4_SHAPE_DOTS3_NOTE_PREV;
    else if (strcmp(fam, "qwen4exp") == 0) g_ds4_shape = DS4_SHAPE_QWEN38_FLASH_NEXT;
    else if (strcmp(fam, "glm5-next") == 0) g_ds4_shape = DS4_SHAPE_GLM53_FLASH;
    else {
        fprintf(stderr, "unknown family: %s\n", fam);
        exit(2);
    }
}

static int hex_nibble(int c)
{
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
    return -1;
}

static char *unhex(const char *s, size_t *out_len)
{
    size_t n = strlen(s);
    if (n % 2 != 0) {
        fprintf(stderr, "hex length must be even\n");
        exit(2);
    }
    char *buf = xmalloc(n / 2 + 1);
    size_t w = 0;
    for (size_t i = 0; i < n; i += 2) {
        int hi = hex_nibble((unsigned char)s[i]);
        int lo = hex_nibble((unsigned char)s[i + 1]);
        if (hi < 0 || lo < 0) {
            fprintf(stderr, "invalid hex\n");
            exit(2);
        }
        buf[w++] = (char)((hi << 4) | lo);
    }
    buf[w] = '\0';
    if (out_len) *out_len = w;
    return buf;
}

static void print_tokens(const token_vec *tv)
{
    printf("TOKENS");
    for (int i = 0; i < tv->len; i++) printf(" %d", tv->v[i]);
    printf("\n");
}

static void print_hex(const char *p, size_t n)
{
    printf("TEXT ");
    for (size_t i = 0; i < n; i++) printf("%02x", (unsigned char)p[i]);
    printf("\n");
}

static int eos_of(const ds4_vocab *v)
{
    if (DS4_MODEL_FAMILY == DS4_MODEL_FAMILY_SOLAR_OPEN2 && v->eot_id >= 0)
        return v->eot_id;
    return v->eos_id;
}

int main(int argc, char **argv)
{
    ds4_model model;
    ds4_vocab vocab;
    token_vec tv = {0};

    if (argc < 4) die_usage();
    set_family(argv[1]);
    model_open(&model, argv[2], false, false);
    vocab_load(&vocab, &model);

    if (strcmp(argv[3], "specials") == 0) {
        printf("SPECIALS family=%u bos=%d eos=%d eot=%d user=%d assistant=%d "
               "think_start=%d think_end=%d tool_call_start=%d "
               "end_of_turn=%d dsml=%d dots3_eotext=%d n_vocab=%d "
               "engine_eos=%d\n",
               (unsigned)DS4_MODEL_FAMILY, vocab.bos_id, vocab.eos_id,
               vocab.eot_id, vocab.user_id, vocab.assistant_id,
               vocab.think_start_id, vocab.think_end_id,
               vocab.tool_call_start_id, vocab.end_of_turn_id, vocab.dsml_id,
               vocab.dots3_endoftext_id, vocab.n_vocab, eos_of(&vocab));
    } else if ((strcmp(argv[3], "encode") == 0 || strcmp(argv[3], "render") == 0) &&
               argc == 5) {
        size_t n = 0;
        char *text = unhex(argv[4], &n);
        (void)n;
        if (argv[3][0] == 'r') tokenize_rendered_chat_vocab(&vocab, text, &tv);
        else bpe_tokenize_text(&vocab, text, &tv);
        print_tokens(&tv);
        free(text);
    } else if (strcmp(argv[3], "decode") == 0 && argc == 5) {
        int id = atoi(argv[4]);
        size_t n = 0;
        char *s = vocab_token_text(&vocab, id, &n);
        print_hex(s, n);
        free(s);
    } else if (strcmp(argv[3], "stop") == 0 && argc == 5) {
        int id = atoi(argv[4]);
        printf("STOP %d\n", vocab_token_is_generation_stop(&vocab, id) ? 1 : 0);
    } else if (strcmp(argv[3], "chat") == 0 && argc == 5) {
        ds4_engine engine = {0};
        engine.vocab = vocab;
        ds4_think_mode mode = (ds4_think_mode)atoi(argv[4]);
        ds4_chat_begin(&engine, &tv);
        ds4_chat_append_effort_prefix(&engine, &tv, mode);
        ds4_chat_append_message(&engine, &tv, "system",
                                "Policy <think>system</think>.");
        ds4_chat_append_message(&engine, &tv, "developer", "Developer policy.");
        ds4_chat_append_message(&engine, &tv, "user", "hello");
        ds4_chat_append_message(
            &engine, &tv, "assistant",
            DS4_MODEL_FAMILY == DS4_MODEL_FAMILY_SOLAR_OPEN2
                ? "<|think:start|>trace<|think:end|>answer"
                : "<think>trace</think>answer");
        if (DS4_MODEL_FAMILY == DS4_MODEL_FAMILY_SOLAR_OPEN2)
            token_vec_push(&tv, vocab.im_end_id);
        ds4_chat_append_message(
            &engine, &tv, "tool",
            "A </tool_response> B </dots_function_response> C "
            "<|tool_response:end|> D");
        ds4_chat_append_message(
            &engine, &tv, "function",
            "raw:\xff A </tool_response> B </dots_function_response> C "
            "<|tool_response:end|> D");
        if (DS4_MODEL_FAMILY == DS4_MODEL_FAMILY_SOLAR_OPEN2)
            token_vec_push(&tv, vocab.im_end_id);
        ds4_chat_append_assistant_prefix(&engine, &tv, mode);
        print_tokens(&tv);
    } else {
        die_usage();
    }

    token_vec_free(&tv);
    vocab_free(&vocab);
    model_close(&model);
    return 0;
}
