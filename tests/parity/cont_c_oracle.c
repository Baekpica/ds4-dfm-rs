/* Continuation-registry oracle. Policy copied from ds4_server.c Inc 5a/5b/5c
 * at v0.6.3-dfm (publish / resolve / hold / pin / TTL / bank claim). */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

enum { API_OPENAI = 0, API_ANTHROPIC = 1, API_RESPONSES = 2 };
enum { LIVE = 0, REPLAY = 1 };
enum { OWNER_SERIAL = 0, OWNER_BANK = 1 };

typedef struct rec {
    int state, owner, proto, owner_id, frontier, hard_refs;
    uint64_t gen;
    double publish, pin_expiry;
    char ids[8][96];
    int nids;
} rec;

typedef struct {
    rec v[32];
    int n, max_records, serial_live;
    double grace, ttl, pin_dead, shed;
} reg;

static void r_init(reg *r)
{
    memset(r, 0, sizeof(*r));
    r->max_records = 64;
    r->grace = 60;
    r->ttl = 300;
    r->pin_dead = 60;
    r->shed = 5;
    r->serial_live = -1;
}

static int set_eq(const rec *a, char **ids, int n)
{
    if (a->nids != n) return 0;
    for (int i = 0; i < n; i++) {
        int ok = 0;
        for (int j = 0; j < a->nids; j++)
            if (strcmp(a->ids[j], ids[i]) == 0) ok = 1;
        if (!ok) return 0;
    }
    return 1;
}

static int find(reg *r, int proto, const char *id)
{
    if (!id || !id[0]) return -1;
    for (int i = 0; i < r->n; i++) {
        if ((int)r->v[i].proto != proto) continue;
        for (int j = 0; j < r->v[i].nids; j++)
            if (strcmp(r->v[i].ids[j], id) == 0) return i;
    }
    return -1;
}

static void demote(reg *r, int i)
{
    if (i < 0 || i >= r->n || r->v[i].state != LIVE) return;
    r->v[i].state = REPLAY;
    if (r->serial_live == i) r->serial_live = -1;
}

static void remove_i(reg *r, int i)
{
    demote(r, i);
    for (int k = i + 1; k < r->n; k++) r->v[k - 1] = r->v[k];
    r->n--;
    if (r->serial_live == i) r->serial_live = -1;
    else if (r->serial_live > i) r->serial_live--;
}

static void prune(reg *r)
{
    while (r->n > r->max_records) {
        int victim = -1;
        for (int i = r->n - 1; i >= 0; i--) {
            if (r->v[i].state == REPLAY && r->v[i].hard_refs <= 0) {
                victim = i;
                break;
            }
        }
        if (victim < 0) break;
        remove_i(r, victim);
    }
}

static void expire(reg *r, double now)
{
    if (r->ttl <= 0) return;
    for (int i = 0; i < r->n; i++) {
        if (r->v[i].state == LIVE && now - r->v[i].publish > r->ttl)
            demote(r, i);
    }
}

static int n_live(const reg *r)
{
    int n = 0;
    for (int i = 0; i < r->n; i++) if (r->v[i].state == LIVE) n++;
    return n;
}

static void publish(reg *r, int proto, char **ids, int nids, int owner,
                    int owner_id, uint64_t gen, int frontier, double now)
{
    if (nids <= 0 || gen == 0 || frontier <= 0) return;
    if (owner == OWNER_SERIAL) {
        if (r->serial_live >= 0) demote(r, r->serial_live);
    } else {
        for (int i = 0; i < r->n; i++) {
            if (r->v[i].state == LIVE && r->v[i].owner == OWNER_BANK &&
                r->v[i].owner_id == owner_id) {
                demote(r, i);
                break;
            }
        }
    }
    memmove(&r->v[1], &r->v[0], (size_t)r->n * sizeof(rec));
    memset(&r->v[0], 0, sizeof(rec));
    r->v[0].state = LIVE;
    r->v[0].owner = owner;
    r->v[0].proto = proto;
    r->v[0].owner_id = owner == OWNER_BANK ? owner_id : 0;
    r->v[0].gen = gen;
    r->v[0].frontier = frontier;
    r->v[0].publish = now;
    r->v[0].nids = nids > 8 ? 8 : nids;
    for (int i = 0; i < r->v[0].nids; i++)
        snprintf(r->v[0].ids[i], 96, "%s", ids[i]);
    r->n++;
    if (r->serial_live >= 0) r->serial_live++;
    if (owner == OWNER_SERIAL) r->serial_live = 0;
    prune(r);
}

static int live_has(reg *r, int proto, const char *id, double now)
{
    expire(r, now);
    int i = find(r, proto, id);
    return i >= 0 && r->v[i].state == LIVE;
}

static int known(reg *r, const char *id)
{
    return find(r, API_OPENAI, id) >= 0 || find(r, API_ANTHROPIC, id) >= 0 ||
           find(r, API_RESPONSES, id) >= 0;
}

static int resolve(reg *r, int proto, char **ids, int n, uint64_t gen,
                   int pos, double now)
{
    if (n <= 0 || gen == 0) return 0;
    expire(r, now);
    int i = find(r, proto, ids[0]);
    if (i < 0) return 0;
    rec *x = &r->v[i];
    return x->state == LIVE && x->owner == OWNER_SERIAL && x->proto == proto &&
           set_eq(x, ids, n) && x->gen == gen && x->frontier == pos;
}

static int claim(reg *r, int proto, char **ids, int n, double now,
                 int *bank, uint64_t *gen, int *front)
{
    if (n <= 0) return 0;
    expire(r, now);
    int i = find(r, proto, ids[0]);
    if (i < 0) return 0;
    rec *x = &r->v[i];
    if (!(x->state == LIVE && x->owner == OWNER_BANK && x->proto == proto &&
          set_eq(x, ids, n)))
        return 0;
    if (bank) *bank = x->owner_id;
    if (gen) *gen = x->gen;
    if (front) *front = x->frontier;
    return 1;
}

static int hold(reg *r, int proto, char **ids, int n, double now, int *retry)
{
    expire(r, now);
    if (r->serial_live < 0) return 0;
    rec *x = &r->v[r->serial_live];
    if (n > 0 && x->proto == proto && set_eq(x, ids, n)) return 0;
    double shed_w = r->shed < r->grace ? r->shed : r->grace;
    double shed_left = shed_w > 0 ? shed_w - (now - x->publish) : 0;
    int pinned = x->hard_refs > 0 && r->pin_dead > 0 && now < x->pin_expiry;
    if (shed_left <= 0 && !pinned) return 0;
    double left = shed_left;
    if (pinned && x->pin_expiry - now > left) left = x->pin_expiry - now;
    int ra = (int)(left + 0.999);
    if (ra < 1) ra = 1;
    if (retry) *retry = ra;
    return 1;
}

static void hold_print(int h, int retry)
{
    if (h) printf("HOLD 1 retry=%d\n", retry);
    else printf("HOLD 0\n");
}

static int bank_ref_matches(const rec *x, uint64_t gen, int frontier)
{
    return x && x->state == LIVE && x->owner == OWNER_BANK && gen != 0 &&
           frontier > 0 && x->gen == gen && x->frontier == frontier;
}

static int bank_protects(const reg *r, const rec *x, double now,
                         int query_ok, uint64_t gen, int frontier)
{
    if (!x || x->state != LIVE || x->owner != OWNER_BANK) return 0;
    if (query_ok && !bank_ref_matches(x, gen, frontier)) return 0;
    int grace = r->grace > 0 && now - x->publish < r->grace;
    int pinned = x->hard_refs > 0 && r->pin_dead > 0 && now < x->pin_expiry;
    return grace || pinned;
}

static int bank_retry(const reg *r, int bank, double now, int query_ok,
                      uint64_t gen, int frontier, int *retry)
{
    double left_min = -1;
    for (int i = 0; i < r->n; i++) {
        const rec *x = &r->v[i];
        if (x->owner_id != bank ||
            !bank_protects(r, x, now, query_ok, gen, frontier))
            continue;
        double grace_left = r->grace > 0 ? r->grace - (now - x->publish) : 0;
        double pin_left = x->hard_refs > 0 && r->pin_dead > 0 && now < x->pin_expiry
                              ? x->pin_expiry - now
                              : 0;
        double left = grace_left > pin_left ? grace_left : pin_left;
        if (left > 0 && (left_min < 0 || left < left_min)) left_min = left;
    }
    if (left_min <= 0) return 0;
    int ra = (int)(left_min + 0.999);
    if (ra < 1) ra = 1;
    if (retry) *retry = ra;
    return 1;
}

static void bank_retry_print(const char *name, int protected, int retry)
{
    printf("%s=", name);
    hold_print(protected, retry);
}

static void script_publish(void)
{
    reg r;
    char a[] = "toolu_regA", b[] = "toolu_regB", c[] = "toolu_regC";
    char *ab[2], *sub[1], *sup[3];
    double now = 1000;
    r_init(&r);
    ab[0] = a;
    ab[1] = b;
    publish(&r, API_ANTHROPIC, ab, 2, OWNER_SERIAL, 0, 7, 100, now);
    printf("live_anth_a=%d\n", live_has(&r, API_ANTHROPIC, "toolu_regA", now));
    printf("live_anth_b=%d\n", live_has(&r, API_ANTHROPIC, "toolu_regB", now));
    printf("live_resp_a=%d\n", live_has(&r, API_RESPONSES, "toolu_regA", now));
    printf("resolve_ok=%d\n", resolve(&r, API_ANTHROPIC, ab, 2, 7, 100, now));
    printf("resolve_gen=%d\n", resolve(&r, API_ANTHROPIC, ab, 2, 8, 100, now));
    printf("resolve_pos=%d\n", resolve(&r, API_ANTHROPIC, ab, 2, 7, 101, now));
    printf("resolve_proto=%d\n", resolve(&r, API_RESPONSES, ab, 2, 7, 100, now));
    sub[0] = a;
    printf("resolve_sub=%d\n", resolve(&r, API_ANTHROPIC, sub, 1, 7, 100, now));
    sup[0] = a;
    sup[1] = b;
    sup[2] = c;
    printf("resolve_sup=%d\n", resolve(&r, API_ANTHROPIC, sup, 3, 7, 100, now));
    demote(&r, r.serial_live);
    printf("live_after_demote=%d\n", live_has(&r, API_ANTHROPIC, "toolu_regA", now));
    printf("resolve_after_demote=%d\n", resolve(&r, API_ANTHROPIC, ab, 2, 7, 100, now));
    printf("known_after_demote=%d\n", known(&r, "toolu_regA"));
}

static void script_cap(void)
{
    reg r;
    char id[32], *p[1];
    double now = 1000;
    r_init(&r);
    r.max_records = 4;
    for (int t = 1; t <= 2; t++) {
        snprintf(id, sizeof(id), "toolu_turn%d", t);
        p[0] = id;
        publish(&r, API_ANTHROPIC, p, 1, OWNER_SERIAL, 0, 3, 50 * t, now);
    }
    printf("live1=%d live2=%d n_live=%d n_rec=%d\n",
           live_has(&r, API_ANTHROPIC, "toolu_turn1", now),
           live_has(&r, API_ANTHROPIC, "toolu_turn2", now),
           n_live(&r), r.n);
    for (int t = 3; t <= 8; t++) {
        snprintf(id, sizeof(id), "toolu_turn%d", t);
        p[0] = id;
        publish(&r, API_ANTHROPIC, p, 1, OWNER_SERIAL, 0, 3, 50 * t, now);
    }
    printf("n_rec=%d known1=%d known2=%d live8=%d n_live=%d\n",
           r.n, known(&r, "toolu_turn1"), known(&r, "toolu_turn2"),
           live_has(&r, API_ANTHROPIC, "toolu_turn8", now), n_live(&r));
}

static void script_hold(void)
{
    reg r;
    char *id[1];
    int h, retry = 0;
    r_init(&r);
    char hold_id[] = "toolu_hold";
    id[0] = hold_id;
    publish(&r, API_ANTHROPIC, id, 1, OWNER_SERIAL, 0, 4, 70, 1000);
    h = hold(&r, API_OPENAI, NULL, 0, 1001, &retry);
    hold_print(h, retry);
    h = hold(&r, API_ANTHROPIC, id, 1, 1001, &retry);
    hold_print(h, retry);
    h = hold(&r, API_OPENAI, NULL, 0, 1011, &retry);
    hold_print(h, retry);
    printf("still_live=%d\n",
           r.serial_live >= 0 && r.v[r.serial_live].state == LIVE);
    h = hold(&r, API_OPENAI, NULL, 0, 1131, &retry);
    hold_print(h, retry);
    {
        expire(&r, 1131);
        int i = find(&r, API_ANTHROPIC, "toolu_hold");
        if (i >= 0 && r.v[i].state == LIVE) {
            r.v[i].hard_refs++;
            r.v[i].pin_expiry = 1131 + r.pin_dead;
        }
    }
    h = hold(&r, API_OPENAI, NULL, 0, 1131, &retry);
    hold_print(h, retry);
    if (r.serial_live >= 0) r.v[r.serial_live].pin_expiry = 1130;
    h = hold(&r, API_OPENAI, NULL, 0, 1131, &retry);
    hold_print(h, retry);
    if (r.serial_live >= 0 && r.v[r.serial_live].hard_refs > 0)
        r.v[r.serial_live].hard_refs--;
    printf("hard_refs=%d\n",
           r.serial_live >= 0 ? r.v[r.serial_live].hard_refs : 0);
}

static void script_ttl(void)
{
    reg r;
    char *id[1];
    r_init(&r);
    char ttl_id[] = "toolu_ttl";
    id[0] = ttl_id;
    publish(&r, API_ANTHROPIC, id, 1, OWNER_SERIAL, 0, 4, 70, 1000);
    printf("live_before=%d\n", live_has(&r, API_ANTHROPIC, "toolu_ttl", 1000));
    printf("live_after=%d\n", live_has(&r, API_ANTHROPIC, "toolu_ttl", 1301));
    printf("n_live=%d\n", n_live(&r));
    printf("resolve=%d\n", resolve(&r, API_ANTHROPIC, id, 1, 4, 70, 1301));
    printf("known=%d\n", known(&r, "toolu_ttl"));
}

static void script_bank(void)
{
    reg r;
    char *id[1];
    int bank = -1, front = 0, ok;
    uint64_t gen = 0;
    double now = 1000;
    r_init(&r);
    char bk1[] = "toolu_bk1", ser1[] = "toolu_ser1", bk2[] = "toolu_bk2";
    char bk3[] = "toolu_bk3", dead[] = "toolu_bk_dead";
    id[0] = bk1;
    publish(&r, API_ANTHROPIC, id, 1, OWNER_BANK, 2, 7, 100, now);
    printf("live=%d resp=%d serial_live=%d\n",
           live_has(&r, API_ANTHROPIC, "toolu_bk1", now),
           live_has(&r, API_RESPONSES, "toolu_bk1", now),
           r.serial_live >= 0);
    ok = claim(&r, API_ANTHROPIC, id, 1, now, &bank, &gen, &front);
    if (ok) printf("claim=%d,%llu,%d\n", bank, (unsigned long long)gen, front);
    else printf("claim=-\n");
    printf("claim_resp=%d\n", claim(&r, API_RESPONSES, id, 1, now, NULL, NULL, NULL));
    printf("resolve_serial=%d\n", resolve(&r, API_ANTHROPIC, id, 1, 7, 100, now));
    id[0] = ser1;
    publish(&r, API_ANTHROPIC, id, 1, OWNER_SERIAL, 0, 9, 40, now);
    printf("n_live=%d\n", n_live(&r));
    demote(&r, r.serial_live);
    printf("n_live_after=%d live_bk1=%d\n",
           n_live(&r), live_has(&r, API_ANTHROPIC, "toolu_bk1", now));
    id[0] = bk2;
    publish(&r, API_ANTHROPIC, id, 1, OWNER_BANK, 2, 8, 120, now);
    printf("live_bk1=%d live_bk2=%d\n",
           live_has(&r, API_ANTHROPIC, "toolu_bk1", now),
           live_has(&r, API_ANTHROPIC, "toolu_bk2", now));
    id[0] = bk3;
    publish(&r, API_RESPONSES, id, 1, OWNER_BANK, 3, 2, 80, now);
    printf("n_live=%d\n", n_live(&r));
    id[0] = dead;
    publish(&r, API_ANTHROPIC, id, 1, OWNER_BANK, 4, 0, 100, now);
    publish(&r, API_ANTHROPIC, id, 1, OWNER_BANK, 4, 5, 0, now);
    printf("known_dead=%d\n", known(&r, "toolu_bk_dead"));
    for (int i = 0; i < r.n; i++)
        if (r.v[i].state == LIVE) r.v[i].publish -= 301;
    int ttl_live = live_has(&r, API_ANTHROPIC, "toolu_bk2", now);
    int live = n_live(&r);
    int known_bk2 = known(&r, "toolu_bk2");
    printf("ttl_live=%d n_live=%d known_bk2=%d\n",
           ttl_live, live, known_bk2);
}

static void script_bank_protection(void)
{
    reg r;
    char id_text[] = "toolu_protected", *id[1] = {id_text};
    int retry = 0, protected = 0;
    double now = 1001;
    r_init(&r);
    r.pin_dead = 20;
    publish(&r, API_ANTHROPIC, id, 1, OWNER_BANK, 5, 3, 200, 1000);
    protected = bank_retry(&r, 5, now, 1, 3, 200, &retry);
    bank_retry_print("current", protected, retry);
    protected = bank_retry(&r, 5, now, 1, 4, 200, &retry);
    bank_retry_print("stale", protected, retry);
    protected = bank_retry(&r, 5, now, 0, 0, 0, &retry);
    bank_retry_print("unknown", protected, retry);
    r.v[0].publish -= 100;
    protected = bank_retry(&r, 5, now, 1, 3, 200, &retry);
    bank_retry_print("lapsed", protected, retry);
    r.v[0].hard_refs++;
    r.v[0].pin_expiry = now + r.pin_dead;
    protected = bank_retry(&r, 5, now, 1, 3, 200, &retry);
    bank_retry_print("pinned", protected, retry);
    r.v[0].hard_refs--;
    protected = bank_retry(&r, 5, now, 1, 3, 200, &retry);
    bank_retry_print("unpinned", protected, retry);
}

int main(int argc, char **argv)
{
    if (argc < 2) {
        fprintf(stderr, "usage: cont_c_oracle SCRIPT\n");
        return 2;
    }
    if (strcmp(argv[1], "publish-resolve-demote") == 0) script_publish();
    else if (strcmp(argv[1], "supersede-cap") == 0) script_cap();
    else if (strcmp(argv[1], "grace-hold") == 0) script_hold();
    else if (strcmp(argv[1], "ttl") == 0) script_ttl();
    else if (strcmp(argv[1], "bank-claim") == 0) script_bank();
    else if (strcmp(argv[1], "bank-protection") == 0) script_bank_protection();
    else {
        fprintf(stderr, "unknown script\n");
        return 2;
    }
    return 0;
}
