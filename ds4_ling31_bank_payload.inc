/* Bank disk KV: the serial session's snapshot format over one bank graph. */
static uint64_t ling31_bank_payload_bytes(ds4_batch_ctx *ctx, uint32_t bank) {
    if (!ctx || !ctx->ling31 || bank >= ctx->max_seq || !ctx->bank_hist_valid[bank] ||
        !ctx->bank_hist_len[bank] || !ctx->ling31->bank_logits_valid[bank]) { return 0; }
    return ling31_payload_bytes(&ctx->ling31->graph[bank], ctx->bank_hist_len[bank]);
}

static int ling31_bank_save_payload(ds4_batch_ctx *ctx, uint32_t bank, FILE *fp, char *err, size_t errlen) {
    if (!ling31_bank_payload_bytes(ctx, bank)) {
        payload_set_err(err, errlen, "Ling-3.1 bank has no snapshot-ready frontier");
        return 1;
    }
    return ling31_payload_save(&ctx->ling31->graph[bank], ctx->bank_hist + (size_t)bank * ctx->seq_cap,
        ctx->bank_hist_len[bank], ctx->ling31->bank_logits + (size_t)bank * DS4_N_VOCAB, fp, err, errlen);
}

static int ling31_bank_load_payload(ds4_batch_ctx *ctx, uint32_t bank, FILE *fp, uint64_t bytes,
        char *err, size_t errlen) {
    ds4_ling31_batch_runtime *rt = ctx->ling31;
    ctx->bank_gen[bank]++;
    ctx->bank_hist_valid[bank] = 0;
    ctx->bank_hist_len[bank] = 0;
    ling31_bank_reset(rt, bank);
    uint32_t h[DS4_SESSION_PAYLOAD_U32_FIELDS];
    for (uint32_t i = 0; i < DS4_SESSION_PAYLOAD_U32_FIELDS; i++) {
        if (payload_read_u32(fp, &h[i], &bytes, err, errlen)) { return 1; }
    }
    if (!h[7] || h[7] > ctx->seq_cap) {
        payload_set_err(err, errlen, "Ling-3.1 bank payload exceeds token bound");
        return 1;
    }
    int *tokens = NULL;
    float *logits = rt->bank_logits + (size_t)bank * DS4_N_VOCAB;
    if (ling31_payload_restore(&rt->graph[bank], fp, &bytes, h, &tokens, logits, err, errlen)) {
        free(tokens);
        return 1;
    }
    memcpy(ctx->bank_hist + (size_t)bank * ctx->seq_cap, tokens, (size_t)h[7] * sizeof(int));
    free(tokens);
    ctx->bank_hist_len[bank] = h[7];
    ctx->bank_hist_valid[bank] = 1;
    rt->bank_logits_valid[bank] = 1;
    return 0;
}
