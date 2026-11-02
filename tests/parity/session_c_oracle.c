/* Session host-policy oracle. Includes ds4.c (DS4_NO_GPU) so rewrite /
 * common_prefix / rewind / invalidate match v0.6.5-dfm. Sync plans dump
 * the session_sync reuse/rebuild/generation rules without a GPU graph. */

#include "../../ds4.c"

#include <ctype.h>

static void die_usage(void)
{
    fprintf(stderr,
            "usage: session_c_oracle CMD ...\n"
            "  rewrite LIVE CANONICAL COMMON\n"
            "  prefix LIVE_IDS PROMPT_IDS\n"
            "  rewrite-from CTX LIVE_IDS PROMPT_IDS COMMON\n"
            "  plan FAMILY BACKEND CTX PREFILL VALID SOLAR LIVE;PROMPT\n"
            "  rewind CTX LIVE_LEN POS SOLAR\n"
            "  invalidate\n"
            "  create CTX\n");
    exit(2);
}

static void parse_ids(const char *s, token_vec *tv)
{
    tv->len = 0;
    if (!s || !s[0] || strcmp(s, "-") == 0) return;
    if (strncmp(s, "n:", 2) == 0) {
        long n = strtol(s + 2, NULL, 10);
        for (long i = 0; i < n; i++) token_vec_push(tv, 1);
        return;
    }
    const char *p = s;
    while (*p) {
        if (*p == ',') { p++; continue; }
        char *end = NULL;
        long v = strtol(p, &end, 10);
        if (end == p) break;
        token_vec_push(tv, (int)v);
        p = end;
    }
}

static ds4_session *dummy_live(int ctx, const token_vec *live, int valid)
{
    ds4_session *s = xcalloc(1, sizeof(*s));
    s->ctx_size = ctx;
    s->generation = 1u;
    s->checkpoint_valid = valid != 0;
    for (int i = 0; i < live->len; i++) token_vec_push(&s->checkpoint, live->v[i]);
    return s;
}

static void dummy_free(ds4_session *s)
{
    if (!s) return;
    free(s->checkpoint.v);
    free(s);
}

static void set_family(const char *fam)
{
    if (strcmp(fam, "deepseek4") == 0) g_ds4_shape = DS4_SHAPE_FLASH;
    else if (strcmp(fam, "motif3") == 0) g_ds4_shape = DS4_SHAPE_MOTIF3;
    else if (strcmp(fam, "solar-open2") == 0) g_ds4_shape = DS4_SHAPE_SOLAR_OPEN2_250B;
    else if (strcmp(fam, "exaone-moe") == 0) g_ds4_shape = DS4_SHAPE_KEXAONE_236B;
    else if (strcmp(fam, "dots3-note") == 0) g_ds4_shape = DS4_SHAPE_DOTS3_NOTE_PREV;
}

static void cmd_rewrite(int argc, char **argv)
{
    if (argc != 5) die_usage();
    int live = atoi(argv[2]);
    int canon = atoi(argv[3]);
    int common = atoi(argv[4]);
    printf("REBUILD %d\n", ds4_session_rewrite_requires_rebuild(live, canon, common) ? 1 : 0);
}

static void cmd_prefix(int argc, char **argv)
{
    token_vec live = {0}, prompt = {0};
    ds4_session *s;
    if (argc != 4) die_usage();
    parse_ids(argv[2], &live);
    parse_ids(argv[3], &prompt);
    s = dummy_live(4096, &live, 1);
    printf("COMMON %d\n", ds4_session_common_prefix(s, &prompt));
    dummy_free(s);
    free(live.v);
    free(prompt.v);
}

static void cmd_rewrite_from(int argc, char **argv)
{
    token_vec live = {0}, prompt = {0};
    ds4_session *s;
    int common, ctx;
    char err[256];
    if (argc != 6) die_usage();
    ctx = atoi(argv[2]);
    parse_ids(argv[3], &live);
    parse_ids(argv[4], &prompt);
    common = atoi(argv[5]);
    s = dummy_live(ctx, &live, 1);
    /* Same guards as ds4_session_rewrite_from_common, without calling
     * ds4_session_sync (that pulls the distributed GPU path). */
    (void)err;
    if (prompt.len <= 0 || prompt.len > ctx || !s->checkpoint_valid ||
        common < 0 || common > s->checkpoint.len || common > prompt.len) {
        printf("REWRITE error\n");
    } else {
        int match = 1;
        for (int i = 0; i < common; i++) {
            if (s->checkpoint.v[i] != prompt.v[i]) { match = 0; break; }
        }
        if (!match) printf("REWRITE error\n");
        else if (common == s->checkpoint.len) printf("REWRITE extend\n");
        else if (ds4_session_rewrite_requires_rebuild(s->checkpoint.len, prompt.len, common))
            printf("REWRITE rebuild\n");
        else printf("REWRITE error\n");
    }
    dummy_free(s);
    free(live.v);
    free(prompt.v);
}

static void cmd_plan(int argc, char **argv)
{
    token_vec live = {0}, prompt = {0};
    const char *fam, *pair;
    int ctx, valid, solar, plen;
    uint32_t prefill;
    int cuda;
    int err = 0, start = 0, rebuild = 0, bump = 0, fence = 0, bounds = 0;
    if (argc != 9) { die_usage(); }
    fam = argv[2];
    set_family(fam);
    cuda = strcmp(argv[3], "cpu") != 0;
    ctx = atoi(argv[4]);
    prefill = (uint32_t)atoi(argv[5]);
    if (prefill == 0) prefill = 1;
    valid = atoi(argv[6]) != 0;
    solar = atoi(argv[7]) != 0;
    pair = argv[8];
    {
        const char *semi = strchr(pair, ';');
        char live_s[4096], prompt_s[4096];
        live_s[0] = prompt_s[0] = 0;
        if (semi) {
            size_t n = (size_t)(semi - pair);
            if (n >= sizeof(live_s)) n = sizeof(live_s) - 1;
            memcpy(live_s, pair, n);
            live_s[n] = 0;
            snprintf(prompt_s, sizeof(prompt_s), "%s", semi + 1);
        } else {
            snprintf(live_s, sizeof(live_s), "%s", pair);
        }
        parse_ids(live_s, &live);
        parse_ids(prompt_s, &prompt);
    }
    plen = prompt.len;
    if (plen <= 0 || plen > ctx) {
        err = 1;
        bounds = 1;
        printf("PLAN err=%d start=%d rebuild=%d bump=%d fence=%d bounds=%d\n",
               err, start, rebuild, bump, fence, bounds);
        free(live.v);
        free(prompt.v);
        return;
    }

    {
        int can_extend = 0;
        if (valid && prompt.len >= live.len) {
            ds4_tokens pref = live;
            if (ds4_tokens_starts_with(&prompt, &pref)) {
                if (strcmp(fam, "solar-open2") == 0 && !solar) can_extend = 0;
                else can_extend = 1;
            }
        }
        if (can_extend) {
            start = live.len;
            if (strcmp(fam, "dots3-note") == 0 && start > 0 && prefill > 0) {
                uint32_t tail = (uint32_t)start % prefill;
                if (tail) start -= (int)tail;
            }
            printf("PLAN err=0 start=%d rebuild=0 bump=0 fence=0 bounds=0\n", start);
            free(live.v);
            free(prompt.v);
            return;
        }
    }

    rebuild = 1;
    if (strcmp(fam, "motif3") != 0 && strcmp(fam, "dots3-note") != 0) bump = 1;
    if (strcmp(fam, "deepseek4") == 0 && cuda) {
        uint32_t f = ds4_prefill_fence_rows();
        uint32_t width = prefill;
        if ((uint32_t)plen < width) width = (uint32_t)plen;
        if (f != 0u && width > f) {
            printf("PLAN err=1 start=0 rebuild=1 bump=0 fence=1 bounds=0\n");
            free(live.v);
            free(prompt.v);
            return;
        }
    }
    printf("PLAN err=0 start=0 rebuild=%d bump=%d fence=0 bounds=0\n", rebuild, bump);
    (void)cuda;
    free(live.v);
    free(prompt.v);
}

static void cmd_rewind(int argc, char **argv)
{
    ds4_session *s;
    token_vec live = {0};
    int ctx, live_len, pos, solar, old, bump;
    if (argc != 6) die_usage();
    ctx = atoi(argv[2]);
    live_len = atoi(argv[3]);
    pos = atoi(argv[4]);
    solar = atoi(argv[5]) != 0;
    for (int i = 0; i < live_len; i++) token_vec_push(&live, i);
    s = dummy_live(ctx, &live, 1);
    old = s->checkpoint.len;
    ds4_session_rewind(s, pos);
    bump = s->generation > 1u;
    {
        int solar_invalid = solar && (s->checkpoint.len != old);
        int valid = 1;
        if (solar_invalid) valid = 0;
        printf("REWIND pos=%d bump=%d solar_invalid=%d valid=%d gen=%llu\n",
               s->checkpoint.len, bump ? 1 : 0, solar_invalid ? 1 : 0, valid,
               (unsigned long long)s->generation);
    }
    dummy_free(s);
    free(live.v);
}

static void cmd_invalidate(void)
{
    token_vec live = {0};
    ds4_session *s;
    token_vec_push(&live, 1);
    token_vec_push(&live, 2);
    token_vec_push(&live, 3);
    s = dummy_live(16, &live, 1);
    ds4_session_invalidate(s);
    printf("INVALID gen=%llu valid=%d len=%d\n",
           (unsigned long long)s->generation,
           s->checkpoint_valid ? 1 : 0,
           s->checkpoint.len);
    dummy_free(s);
    free(live.v);
}

static void cmd_create(int argc, char **argv)
{
    int ctx;
    if (argc != 3) die_usage();
    ctx = atoi(argv[2]);
    printf("CREATE gen=1 pos=0 valid=0 ctx=%d\n", ctx);
}

int main(int argc, char **argv)
{
    if (argc < 2) die_usage();
    if (strcmp(argv[1], "rewrite") == 0) cmd_rewrite(argc, argv);
    else if (strcmp(argv[1], "prefix") == 0) cmd_prefix(argc, argv);
    else if (strcmp(argv[1], "rewrite-from") == 0) cmd_rewrite_from(argc, argv);
    else if (strcmp(argv[1], "plan") == 0) cmd_plan(argc, argv);
    else if (strcmp(argv[1], "rewind") == 0) cmd_rewind(argc, argv);
    else if (strcmp(argv[1], "invalidate") == 0) cmd_invalidate();
    else if (strcmp(argv[1], "create") == 0) cmd_create(argc, argv);
    else die_usage();
    return 0;
}
