/* Ling image spans: placeholder coverage and the M-RoPE position plan. */
#include "../ds4.c"
#define CHECK(x) do { if (!(x)) { \
    fprintf(stderr, "Ling media FAIL %d: %s\n", __LINE__, #x); exit(1); \
} } while (0)

/* One image of merged grid gh x gw occupies gh*gw rows. */
static ds4_ling3vl_media span_media(unsigned position, unsigned gh, unsigned gw) {
    ds4_ling3vl_media m = {0};
    m.count = 1;
    m.rows = gh * gw;
    m.spans[0] = (ds4_ling3vl_image_span){
        .position = position, .rows = gh * gw, .feature = 0,
        .grid_h = 2u * gh, .grid_w = 2u * gw};
    return m;
}

static void text_is_one_dimensional(void) {
    int tokens[8];
    int32_t out[8 * 3];
    for (unsigned i = 0; i < 8; i++) { tokens[i] = 7; }
    CHECK(ling3vl_media_positions(NULL, tokens, 8, 5, out));
    for (unsigned i = 0; i < 8; i++) {
        CHECK(out[3 * i] == (int32_t)(5 + i));
        CHECK(out[3 * i + 1] == out[3 * i]);
        CHECK(out[3 * i + 2] == out[3 * i]);
    }
}

/* Rows 0..1 text, then a 2x3 merged span, then text. The span costs
 * max(2, 3) = 3 positions, not its six rows. */
static void image_span_lays_out_a_grid(void) {
    enum { GH = 2u, GW = 3u, START = 2u, ROWS = GH * GW, TOTAL = START + ROWS + 2u };
    int tokens[TOTAL];
    int32_t out[TOTAL * 3];
    for (unsigned i = 0; i < TOTAL; i++) {
        tokens[i] = i >= START && i < START + ROWS ? (int)LING3VL_IMAGE_TOKEN : 7;
    }
    ds4_ling3vl_media media = span_media(START, GH, GW);
    CHECK(ling3vl_media_positions(&media, tokens, TOTAL, 0, out));
    for (unsigned i = 0; i < START; i++) {
        CHECK(out[3 * i] == (int32_t)i && out[3 * i + 1] == (int32_t)i);
    }
    for (unsigned y = 0; y < GH; y++) {
        for (unsigned x = 0; x < GW; x++) {
            const unsigned row = START + y * GW + x;
            CHECK(out[3 * row] == (int32_t)START);
            CHECK(out[3 * row + 1] == (int32_t)(START + y));
            CHECK(out[3 * row + 2] == (int32_t)(START + x));
        }
    }
    const unsigned after = START + ROWS;
    CHECK(out[3 * after] == (int32_t)(START + GW));
    CHECK(out[3 * (after + 1)] == (int32_t)(START + GW + 1u));

    /* A later prefill chunk replays the same plan, so its rows keep the
     * cursor the first chunk established. */
    int32_t tail[2 * 3];
    CHECK(ling3vl_media_positions(&media, tokens + after, 2, after, tail));
    CHECK(tail[0] == (int32_t)(START + GW));
    CHECK(tail[3] == (int32_t)(START + GW + 1u));
}

static void coverage_is_checked_exactly(void) {
    enum { START = 3u, ROWS = 4u, TOTAL = 12u };
    int tokens[TOTAL];
    for (unsigned i = 0; i < TOTAL; i++) {
        tokens[i] = i >= START && i < START + ROWS ? (int)LING3VL_IMAGE_TOKEN : 7;
    }
    ds4_tokens prompt = {.v = tokens, .len = TOTAL, .cap = TOTAL};
    const uint8_t payload[] = {1, 2, 3};
    ds4_ling3vl_image_input images[] = {
        {.data = payload, .data_len = sizeof(payload), .token_offset = START},
    };
    /* 4 merged rows come from a 4x4 patch grid. */
    ds4_ling3vl_image_info info[] = {{.grid_h = 4, .grid_w = 4, .token_count = ROWS}};
    ds4_ling3vl_media media = {0};
    /* A plane too small for the span has nowhere to put its rows. */
    CHECK(!ling3vl_media_check(&prompt, images, 1, info, ROWS - 1u, &media));
    CHECK(ling3vl_media_check(&prompt, images, 1, info, TOTAL, &media));
    CHECK(media.count == 1 && media.rows == ROWS && media.spans[0].grid_w == 4);

    /* The token count has to agree with the grid. */
    info[0].token_count = ROWS + 1u;
    CHECK(!ling3vl_media_check(&prompt, images, 1, info, TOTAL, &media));
    info[0].token_count = ROWS;

    /* A placeholder outside the declared span is unbound. */
    tokens[0] = (int)LING3VL_IMAGE_TOKEN;
    CHECK(!ling3vl_media_check(&prompt, images, 1, info, TOTAL, &media));
    tokens[0] = 7;

    /* A text token inside the span would receive image features. */
    tokens[START + 1u] = 7;
    CHECK(!ling3vl_media_check(&prompt, images, 1, info, TOTAL, &media));
    tokens[START + 1u] = (int)LING3VL_IMAGE_TOKEN;

    images[0].token_offset = TOTAL;
    CHECK(!ling3vl_media_check(&prompt, images, 1, info, TOTAL, &media));
    images[0].token_offset = START;
    CHECK(ling3vl_media_check(&prompt, images, 1, info, TOTAL, &media));
}

int main(void) {
    g_ds4_shape = DS4_SHAPE_LING30_FLASH_VL;
    text_is_one_dimensional();
    image_span_lays_out_a_grid();
    coverage_is_checked_exactly();
    printf("Ling media OK\n");
    return 0;
}
