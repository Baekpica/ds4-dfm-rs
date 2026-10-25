/* Bank disk KV: the serial session's snapshot format over one bank graph. */
static uint64_t minimax_bank_payload_bytes(ds4_batch_ctx *ctx, uint32_t bank) {
    if (!ctx || !ctx->minimax || bank >= ctx->max_seq || !ctx->bank_hist_valid[bank] ||
        !ctx->bank_hist_len[bank] || !ctx->minimax->bank_logits_valid[bank]) { return 0; }
    return minimax_payload_bytes(&ctx->minimax->graph[bank], ctx->bank_hist_len[bank]);
}

static int minimax_bank_save_payload(ds4_batch_ctx *ctx, uint32_t bank, FILE *fp, char *err, size_t errlen) {
    if (!minimax_bank_payload_bytes(ctx, bank)) {
        payload_set_err(err, errlen, "MiniMax bank has no snapshot-ready frontier");
        return 1;
    }
    return minimax_payload_save(&ctx->minimax->graph[bank], ctx->bank_hist + (size_t)bank * ctx->seq_cap,
        ctx->bank_hist_len[bank], ctx->minimax->bank_logits + (size_t)bank * DS4_N_VOCAB, fp, err, errlen);
}

static int minimax_bank_load_payload(ds4_batch_ctx *ctx, uint32_t bank, FILE *fp, uint64_t bytes,
        char *err, size_t errlen) {
    ds4_minimax_batch_runtime *rt = ctx->minimax;
    ctx->bank_gen[bank]++;
    ctx->bank_hist_valid[bank] = 0;
    ctx->bank_hist_len[bank] = 0;
    minimax_bank_reset(rt, bank);
    uint32_t h[DS4_SESSION_PAYLOAD_U32_FIELDS];
    for (uint32_t i = 0; i < DS4_SESSION_PAYLOAD_U32_FIELDS; i++) {
        if (payload_read_u32(fp, &h[i], &bytes, err, errlen)) { return 1; }
    }
    if (!h[7] || h[7] > ctx->seq_cap) {
        payload_set_err(err, errlen, "MiniMax bank payload exceeds token bound");
        return 1;
    }
    int *tokens = NULL;
    float *logits = rt->bank_logits + (size_t)bank * DS4_N_VOCAB;
    if (minimax_payload_restore(&rt->graph[bank], fp, &bytes, h, &tokens, logits, err, errlen)) {
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
