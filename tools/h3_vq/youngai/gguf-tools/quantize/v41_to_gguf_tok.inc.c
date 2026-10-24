/* v41_to_gguf_tok.inc.c — tokenizer 常量(v41_tokenizer_consts.py 产物 TOKC)→ tokenizer.ggml.tokens / merges 两个字串数组 +
 * model/pre/bos/eos(v41_to_gguf.c 的分片, 单 TU; 2026-10-13 为守 500 行拆出, 内容一字未动)。
 * 引擎 vocab_load 只吃这几个键; pre="joyai-llm" 是 core_bpe.c 写死的切分规则名。 */
static void emit_tokenizer(const char *path) {
    FILE *f = fopen(path, "rb"); if (!f) die("打不开 tokenizer_consts.bin");
    uint32_t h[4]; if (fread(h, 4, 4, f) != 4 || h[0] != 0x434B4F54u || h[1] != 1) die("tokenizer_consts.bin 魔数/版本不对(TOKC v1)");
    uint32_t nt = h[2], nm = h[3];
    if ((int)nt != C.vocab_size) { fprintf(stderr, "★tokenizer token 数 %u != vocab_size %d★\n", nt, C.vocab_size); exit(1); }
    char **tok = (char **)malloc(sizeof(char *) * nt), **mg = (char **)malloc(sizeof(char *) * nm);
    for (uint32_t i = 0; i < nt + nm; i++) {
        uint32_t L; if (fread(&L, 4, 1, f) != 1 || L > 4096) die("tokenizer_consts.bin 读坏");
        char *s = (char *)malloc(L + 1); if (fread(s, 1, L, f) != L) die("tokenizer_consts.bin 读不满"); s[L] = 0;
        if (i < nt) tok[i] = s; else mg[i - nt] = s;
    }
    fclose(f);
    ds4gw_kv_str(&W, "tokenizer.ggml.model", "gpt2");
    ds4gw_kv_str(&W, "tokenizer.ggml.pre", "joyai-llm");
    ds4gw_kv_arr_str(&W, "tokenizer.ggml.tokens", (const char *const *)tok, nt);
    ds4gw_kv_arr_str(&W, "tokenizer.ggml.merges", (const char *const *)mg, nm);
    ds4gw_kv_u32(&W, "tokenizer.ggml.bos_token_id", 0);
    ds4gw_kv_u32(&W, "tokenizer.ggml.eos_token_id", 1);
    fprintf(stderr, "[tokenizer] %u token / %u merges 进 KV\n", nt, nm);
    for (uint32_t i = 0; i < nt; i++) free(tok[i]); for (uint32_t i = 0; i < nm; i++) free(mg[i]); free(tok); free(mg);
}
