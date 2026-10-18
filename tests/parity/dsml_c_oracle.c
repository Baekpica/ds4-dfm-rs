/* DSML decode-state + sampling-override oracle from ds4_server.c at v0.6.3-dfm. */

#include <ctype.h>
#include <stdarg.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define DS4_DSML "｜DSML｜"
#define DS4_TOOL_CALLS_START "<" DS4_DSML "tool_calls>"
#define DS4_TOOL_CALLS_END "</" DS4_DSML "tool_calls>"
#define DS4_INVOKE_START "<" DS4_DSML "invoke"
#define DS4_INVOKE_END "</" DS4_DSML "invoke>"
#define DS4_PARAM_START "<" DS4_DSML "parameter"
#define DS4_PARAM_END "</" DS4_DSML "parameter>"
#define DS4_TOOL_CALLS_START_SHORT "<DSML｜tool_calls>"
#define DS4_TOOL_CALLS_END_SHORT "</DSML｜tool_calls>"
#define DS4_INVOKE_START_SHORT "<DSML｜invoke"
#define DS4_INVOKE_END_SHORT "</DSML｜invoke>"
#define DS4_PARAM_START_SHORT "<DSML｜parameter"
#define DS4_PARAM_END_SHORT "</DSML｜parameter>"

/* /data/ds4/ds4_server.c:7376 glm_tool_syntax, :7595-7611 dialect select. */
#define GLM_TOOL_CALLS_START "<tool_call>"
#define GLM_TOOL_CALLS_END "</tool_call>"
#define GLM_ARG_KEY_START "<arg_key>"
#define GLM_ARG_KEY_END "</arg_key>"
#define GLM_ARG_VALUE_START "<arg_value>"
#define GLM_ARG_VALUE_END "</arg_value>"

#define SERVER_MODEL_SYNTAX_DEEPSEEK 0
#define SERVER_MODEL_SYNTAX_GLM 3

#define DS4_SAMPLE_OVERRIDE_NONE 0
#define DS4_SAMPLE_OVERRIDE_GREEDY 1
#define DS4_SAMPLE_OVERRIDE_TOKEN(t) ((t) + 2)

#define TOOL_CHOICE_AUTO 0
#define TOOL_CHOICE_NONE 1
#define TOOL_CHOICE_REQUIRED 2
#define DS4_THINK_NONE 0
#define DS4_THINK_LOW 1
#define DS4_THINK_HIGH 2
#define DS4_THINK_MAX 3

typedef enum {
    DSML_DECODE_OUTSIDE = 0,
    DSML_DECODE_STRUCTURAL,
    DSML_DECODE_STRING_BODY,
    DSML_DECODE_JSON_STRUCTURAL,
    DSML_DECODE_JSON_STRING,
} dsml_decode_state;

typedef enum {
    DSML_TRACK_SEARCH,
    DSML_TRACK_STRUCTURAL,
    DSML_TRACK_STRING_BODY,
    DSML_TRACK_JSON_PARAM,
    DSML_TRACK_DONE,
} dsml_track_mode;

typedef struct {
    const char *tool_calls_start, *tool_calls_end, *invoke_start, *invoke_end, *param_start, *param_end;
} dsml_syntax;

static const dsml_syntax glm_tool_syntax = {
    GLM_TOOL_CALLS_START, GLM_TOOL_CALLS_END,
    GLM_ARG_KEY_START, GLM_ARG_KEY_END, GLM_ARG_VALUE_START, GLM_ARG_VALUE_END,
};

static const dsml_syntax dsml_syntaxes[] = {
    { DS4_TOOL_CALLS_START, DS4_TOOL_CALLS_END, DS4_INVOKE_START, DS4_INVOKE_END, DS4_PARAM_START, DS4_PARAM_END },
    { DS4_TOOL_CALLS_START_SHORT, DS4_TOOL_CALLS_END_SHORT, DS4_INVOKE_START_SHORT, DS4_INVOKE_END_SHORT, DS4_PARAM_START_SHORT, DS4_PARAM_END_SHORT },
    { "<tool_calls>", "</tool_calls>", "<invoke", "</invoke>", "<parameter", "</parameter>" },
};

typedef struct {
    dsml_track_mode mode;
    dsml_decode_state decode;
    const dsml_syntax *syn;
    size_t pos;
    bool json_in_string, json_escaped;
    int model_syntax;
} dsml_decode_tracker;

/* C raw_partial_lit: any proper prefix of the literal. */
static bool raw_partial_any(const char *raw, size_t raw_len, size_t pos, const char *lit) {
    size_t n = strlen(lit);
    if (pos > raw_len || raw_len - pos >= n) return false;
    return !memcmp(raw + pos, lit, raw_len - pos);
}

/* ds4_server.c:7755 observe_tool_markers reads the tracker, not the text. */
static const char *find_any_tool_end(const char *s) {
    const char *best = NULL;
    const char *cands[] = {
        DS4_TOOL_CALLS_END, DS4_TOOL_CALLS_END_SHORT,
        "</tool_calls>", "</tool_call>",
    };
    for (size_t i = 0; i < sizeof(cands) / sizeof(cands[0]); i++) {
        const char *p = strstr(s, cands[i]);
        if (p && (!best || p < best)) best = p;
    }
    return best;
}
static void observe_tool_markers(const dsml_decode_tracker *tracker, const char *scan,
                                 bool *saw_start, bool *saw_end, bool *orphan_end) {
    if (tracker->syn) *saw_start = true;
    if (tracker->mode == DSML_TRACK_DONE) *saw_end = true;
    else if (!*saw_start && scan && find_any_tool_end(scan) && orphan_end) *orphan_end = true;
}

static void die(const char *m) { fprintf(stderr, "dsml_c_oracle: %s\n", m); exit(2); }
static char *xstrndup(const char *s, size_t n) {
    char *p = malloc(n + 1); if (!p) die("oom"); memcpy(p, s ? s : "", n); p[n] = 0; return p;
}
static char *dsml_unescape(const char *s) {
    size_t n = strlen(s); char *out = malloc(n + 1); size_t o = 0;
    for (size_t i = 0; i < n; ) {
        if (s[i] != '&') { out[o++] = s[i++]; continue; }
        if (!strncmp(s + i, "&amp;", 5)) { out[o++] = '&'; i += 5; }
        else if (!strncmp(s + i, "&lt;", 4)) { out[o++] = '<'; i += 4; }
        else if (!strncmp(s + i, "&gt;", 4)) { out[o++] = '>'; i += 4; }
        else if (!strncmp(s + i, "&quot;", 6)) { out[o++] = '"'; i += 6; }
        else if (!strncmp(s + i, "&apos;", 6)) { out[o++] = '\''; i += 6; }
        else out[o++] = s[i++];
    }
    out[o] = 0; return out;
}
static char *dsml_attr(const char *tag, const char *name) {
    char pat[64]; snprintf(pat, sizeof(pat), "%s=\"", name);
    const char *p = strstr(tag, pat); if (!p) return NULL;
    p += strlen(pat); const char *e = strchr(p, '"'); if (!e) return NULL;
    char *raw = xstrndup(p, (size_t)(e - p)); char *u = dsml_unescape(raw); free(raw); return u;
}
static bool raw_full_lit(const char *raw, size_t raw_len, size_t pos, const char *lit) {
    size_t n = strlen(lit);
    return pos <= raw_len && raw_len - pos >= n && !memcmp(raw + pos, lit, n);
}
static bool raw_partial_lit(const char *raw, size_t raw_len, size_t pos, const char *lit) {
    size_t n = strlen(lit);
    if (pos > raw_len || raw_len - pos >= n) return false;
    return !memcmp(raw + pos, lit, raw_len - pos);
}
static bool raw_partial_lit_min(const char *raw, size_t raw_len, size_t pos, const char *lit, size_t min_len) {
    size_t lit_len = strlen(lit);
    if (!raw || pos > raw_len || raw_len - pos >= lit_len) return false;
    size_t avail = raw_len - pos;
    return avail >= min_len && !memcmp(raw + pos, lit, avail);
}
static const char *find_lit_bounded(const char *s, size_t n, const char *lit) {
    size_t m = strlen(lit);
    if (m == 0) return s;
    if (n < m) return NULL;
    for (size_t i = 0; i <= n - m; i++) if (!memcmp(s + i, lit, m)) return s + i;
    return NULL;
}
static bool dsml_attr_is_string_true(const char *raw, size_t raw_len, size_t tag_start, size_t tag_end) {
    if (tag_end <= tag_start || tag_end > raw_len) return false;
    char *tag = xstrndup(raw + tag_start, tag_end - tag_start);
    char *is = dsml_attr(tag, "string");
    bool r = is && !strcmp(is, "true");
    free(is); free(tag); return r;
}
static size_t dsml_max_tool_start_len(void) {
    size_t max = 0;
    for (size_t i = 0; i < sizeof(dsml_syntaxes) / sizeof(dsml_syntaxes[0]); i++) {
        size_t n = strlen(dsml_syntaxes[i].tool_calls_start);
        if (n > max) max = n;
    }
    return max;
}
static bool dsml_find_tool_start(const char *raw, size_t raw_len, size_t *pos_out, const dsml_syntax **syn_out) {
    const char *best = NULL; const dsml_syntax *best_syn = NULL;
    for (size_t i = 0; i < sizeof(dsml_syntaxes) / sizeof(dsml_syntaxes[0]); i++) {
        const char *p = find_lit_bounded(raw, raw_len, dsml_syntaxes[i].tool_calls_start);
        if (p && (!best || p < best)) { best = p; best_syn = &dsml_syntaxes[i]; }
    }
    if (!best) return false;
    *pos_out = (size_t)(best - raw) + strlen(best_syn->tool_calls_start);
    *syn_out = best_syn; return true;
}
static bool dsml_find_tool_start_from(const char *raw, size_t raw_len, size_t start, size_t *pos_out, const dsml_syntax **syn_out) {
    if (start > raw_len) return false;
    size_t rel = 0;
    if (!dsml_find_tool_start(raw + start, raw_len - start, &rel, syn_out)) return false;
    *pos_out = start + rel; return true;
}
static bool raw_suffix_partial_lit(const char *raw, size_t raw_len, const char *lit, size_t min_len) {
    size_t lit_len = strlen(lit);
    if (!raw || raw_len == 0 || lit_len == 0) return false;
    size_t max = raw_len < lit_len ? raw_len : lit_len - 1;
    for (size_t n = min_len; n <= max; n++) if (!memcmp(raw + raw_len - n, lit, n)) return true;
    return false;
}
static dsml_decode_state dsml_decode_scan_json_param(const char *raw, size_t raw_len, size_t pos, const dsml_syntax *syn) {
    bool in_string = false, escaped = false;
    while (pos < raw_len) {
        if (!in_string && raw_full_lit(raw, raw_len, pos, syn->param_end)) return DSML_DECODE_STRUCTURAL;
        unsigned char c = (unsigned char)raw[pos++];
        if (in_string) {
            if (escaped) escaped = false;
            else if (c == '\\') escaped = true;
            else if (c == '"') in_string = false;
        } else if (c == '"') in_string = true;
    }
    if (!in_string && raw_suffix_partial_lit(raw, raw_len, syn->param_end, 2)) return DSML_DECODE_STRUCTURAL;
    return in_string ? DSML_DECODE_JSON_STRING : DSML_DECODE_JSON_STRUCTURAL;
}
static dsml_decode_state dsml_decode_state_for_text(const char *raw, size_t raw_len) {
    if (!raw || raw_len == 0) return DSML_DECODE_OUTSIDE;
    size_t pos = 0; const dsml_syntax *syn = NULL;
    if (!dsml_find_tool_start(raw, raw_len, &pos, &syn)) return DSML_DECODE_OUTSIDE;
    for (;;) {
        while (pos < raw_len && isspace((unsigned char)raw[pos])) pos++;
        if (pos >= raw_len) return DSML_DECODE_STRUCTURAL;
        if (raw_full_lit(raw, raw_len, pos, syn->tool_calls_end)) return DSML_DECODE_OUTSIDE;
        if (raw_full_lit(raw, raw_len, pos, syn->invoke_end)) { pos += strlen(syn->invoke_end); continue; }
        if (raw_full_lit(raw, raw_len, pos, syn->invoke_start)) {
            const char *tag_end = memchr(raw + pos, '>', raw_len - pos);
            if (!tag_end) return DSML_DECODE_STRUCTURAL;
            pos = (size_t)(tag_end - raw) + 1; continue;
        }
        if (raw_full_lit(raw, raw_len, pos, syn->param_start)) {
            size_t tag_start = pos;
            const char *tag_end_ptr = memchr(raw + pos, '>', raw_len - pos);
            if (!tag_end_ptr) return DSML_DECODE_STRUCTURAL;
            size_t tag_end = (size_t)(tag_end_ptr - raw) + 1;
            bool string_value = dsml_attr_is_string_true(raw, raw_len, tag_start, tag_end);
            pos = tag_end;
            if (string_value) {
                const char *end = find_lit_bounded(raw + pos, raw_len - pos, syn->param_end);
                if (!end) {
                    if (raw_suffix_partial_lit(raw, raw_len, syn->param_end, 2)) return DSML_DECODE_STRUCTURAL;
                    return DSML_DECODE_STRING_BODY;
                }
                pos = (size_t)(end - raw) + strlen(syn->param_end); continue;
            }
            dsml_decode_state json_state = dsml_decode_scan_json_param(raw, raw_len, pos, syn);
            if (json_state == DSML_DECODE_STRUCTURAL) {
                const char *end = find_lit_bounded(raw + pos, raw_len - pos, syn->param_end);
                if (!end) return DSML_DECODE_STRUCTURAL;
                pos = (size_t)(end - raw) + strlen(syn->param_end); continue;
            }
            return json_state;
        }
        for (size_t i = 0; i < sizeof(dsml_syntaxes) / sizeof(dsml_syntaxes[0]); i++) {
            if (raw_partial_lit(raw, raw_len, pos, dsml_syntaxes[i].tool_calls_end) ||
                raw_partial_lit(raw, raw_len, pos, dsml_syntaxes[i].invoke_start) ||
                raw_partial_lit(raw, raw_len, pos, dsml_syntaxes[i].invoke_end) ||
                raw_partial_lit(raw, raw_len, pos, dsml_syntaxes[i].param_start) ||
                raw_partial_lit(raw, raw_len, pos, dsml_syntaxes[i].param_end))
                return DSML_DECODE_STRUCTURAL;
        }
        return DSML_DECODE_STRUCTURAL;
    }
}
static void dsml_decode_tracker_init(dsml_decode_tracker *dt) {
    memset(dt, 0, sizeof(*dt)); dt->mode = DSML_TRACK_SEARCH; dt->decode = DSML_DECODE_OUTSIDE;
}
static void dsml_decode_tracker_update(dsml_decode_tracker *dt, const char *raw, size_t raw_len) {
    if (!dt || !raw) return;
    for (;;) {
        if (dt->mode == DSML_TRACK_DONE) { dt->decode = DSML_DECODE_OUTSIDE; return; }
        if (dt->mode == DSML_TRACK_SEARCH) {
            size_t pos = 0; const dsml_syntax *syn = NULL;
            bool found;
            if (dt->model_syntax == SERVER_MODEL_SYNTAX_GLM) {
                syn = &glm_tool_syntax;
                const char *start = find_lit_bounded(raw + dt->pos, raw_len - dt->pos,
                                                     syn->tool_calls_start);
                found = start != NULL;
                if (found) pos = (size_t)(start - raw) + strlen(syn->tool_calls_start);
            } else {
                found = dsml_find_tool_start_from(raw, raw_len, dt->pos, &pos, &syn);
            }
            if (!found) {
                size_t hold = dsml_max_tool_start_len();
                dt->pos = raw_len > hold ? raw_len - hold : 0;
                dt->decode = DSML_DECODE_OUTSIDE; return;
            }
            dt->syn = syn; dt->pos = pos; dt->mode = DSML_TRACK_STRUCTURAL; dt->decode = DSML_DECODE_STRUCTURAL;
        }
        if (dt->mode == DSML_TRACK_STRING_BODY) {
            while (dt->pos < raw_len) {
                if (raw_full_lit(raw, raw_len, dt->pos, dt->syn->param_end)) {
                    dt->pos += strlen(dt->syn->param_end);
                    dt->mode = DSML_TRACK_STRUCTURAL; dt->decode = DSML_DECODE_STRUCTURAL; goto structural;
                }
                if (raw_partial_any(raw, raw_len, dt->pos, dt->syn->param_end)) {
                    dt->decode = raw_len - dt->pos >= 2 ? DSML_DECODE_STRUCTURAL : DSML_DECODE_STRING_BODY;
                    return;
                }
                dt->pos++;
            }
            dt->decode = DSML_DECODE_STRING_BODY; return;
        }
        if (dt->mode == DSML_TRACK_JSON_PARAM) {
            while (dt->pos < raw_len) {
                if (!dt->json_in_string) {
                    if (raw_full_lit(raw, raw_len, dt->pos, dt->syn->param_end)) {
                        dt->pos += strlen(dt->syn->param_end);
                        dt->mode = DSML_TRACK_STRUCTURAL; dt->decode = DSML_DECODE_STRUCTURAL; goto structural;
                    }
                    if (raw_partial_lit_min(raw, raw_len, dt->pos, dt->syn->param_end, 2)) { dt->decode = DSML_DECODE_STRUCTURAL; return; }
                }
                unsigned char c = (unsigned char)raw[dt->pos++];
                if (dt->json_in_string) {
                    if (dt->json_escaped) dt->json_escaped = false;
                    else if (c == '\\') dt->json_escaped = true;
                    else if (c == '"') dt->json_in_string = false;
                } else if (c == '"') dt->json_in_string = true;
            }
            dt->decode = dt->json_in_string ? DSML_DECODE_JSON_STRING : DSML_DECODE_JSON_STRUCTURAL;
            return;
        }
structural:
        while (dt->mode == DSML_TRACK_STRUCTURAL) {
            while (dt->pos < raw_len && isspace((unsigned char)raw[dt->pos])) dt->pos++;
            if (dt->pos >= raw_len) { dt->decode = DSML_DECODE_STRUCTURAL; return; }
            if (raw_full_lit(raw, raw_len, dt->pos, dt->syn->tool_calls_end)) {
                dt->mode = DSML_TRACK_DONE; dt->pos += strlen(dt->syn->tool_calls_end);
                dt->decode = DSML_DECODE_OUTSIDE; return;
            }
            if (raw_full_lit(raw, raw_len, dt->pos, dt->syn->invoke_end)) { dt->pos += strlen(dt->syn->invoke_end); continue; }
            if (raw_full_lit(raw, raw_len, dt->pos, dt->syn->invoke_start)) {
                const char *tag_end = memchr(raw + dt->pos, '>', raw_len - dt->pos);
                if (!tag_end) { dt->decode = DSML_DECODE_STRUCTURAL; return; }
                dt->pos = (size_t)(tag_end - raw) + 1; continue;
            }
            if (raw_full_lit(raw, raw_len, dt->pos, dt->syn->param_start)) {
                size_t tag_start = dt->pos;
                const char *tag_end = memchr(raw + dt->pos, '>', raw_len - dt->pos);
                if (!tag_end) { dt->decode = DSML_DECODE_STRUCTURAL; return; }
                size_t tag_after = (size_t)(tag_end - raw) + 1;
                bool string_value = dt->model_syntax == SERVER_MODEL_SYNTAX_GLM ||
                    dsml_attr_is_string_true(raw, raw_len, tag_start, tag_after);
                dt->pos = tag_after;
                if (string_value) { dt->mode = DSML_TRACK_STRING_BODY; dt->decode = DSML_DECODE_STRING_BODY; }
                else { dt->mode = DSML_TRACK_JSON_PARAM; dt->json_in_string = false; dt->json_escaped = false; dt->decode = DSML_DECODE_JSON_STRUCTURAL; }
                break;
            }
            if (raw_partial_lit(raw, raw_len, dt->pos, dt->syn->tool_calls_end) ||
                raw_partial_lit(raw, raw_len, dt->pos, dt->syn->invoke_start) ||
                raw_partial_lit(raw, raw_len, dt->pos, dt->syn->invoke_end) ||
                raw_partial_lit(raw, raw_len, dt->pos, dt->syn->param_start) ||
                raw_partial_lit(raw, raw_len, dt->pos, dt->syn->param_end)) {
                dt->decode = DSML_DECODE_STRUCTURAL; return;
            }
            /* GLM puts the function and argument names between tags
             * (/data/ds4/ds4_server.c:7746-7752). */
            if (dt->model_syntax == SERVER_MODEL_SYNTAX_GLM) { dt->pos++; continue; }
            dt->decode = DSML_DECODE_STRUCTURAL; return;
        }
    }
}

static const char *state_name(dsml_decode_state s) {
    switch (s) {
    case DSML_DECODE_OUTSIDE: return "outside";
    case DSML_DECODE_STRUCTURAL: return "structural";
    case DSML_DECODE_STRING_BODY: return "string-body";
    case DSML_DECODE_JSON_STRUCTURAL: return "json-structural";
    case DSML_DECODE_JSON_STRING: return "json-string";
    }
    return "?";
}
static void dump_state(const char *raw) {
    size_t n = strlen(raw);
    dsml_decode_state ref = dsml_decode_state_for_text(raw, n);
    dsml_decode_tracker tr; dsml_decode_tracker_init(&tr);
    dsml_decode_tracker_update(&tr, raw, n);
    printf("ref=%s tracker=%s\n", state_name(ref), state_name(tr.decode));
}

/* The reference's GLM vectors (ds4_server.c:20571, :20608): the accumulated
 * text is fed fragment by fragment against the GLM dialect, and the state after
 * each fragment is printed.  `bytewise` feeds one byte at a time, which is what
 * the reference's own loop does in its second pass. */
static const char *const glm_parts[] = {
    "prose <tool_call>bash<arg_key>command</arg_key>",
    "<arg_value>printf '<html>'",
    "<",
    "/arg_",
    "value>",
    "<arg_key>description</arg_key><arg_value>test",
    "</arg_value></tool_call>",
};

static void dump_glm_boundaries(bool bytewise) {
    dsml_decode_tracker tr; dsml_decode_tracker_init(&tr);
    tr.model_syntax = SERVER_MODEL_SYNTAX_GLM;
    char raw[512]; size_t used = 0; raw[0] = 0;
    printf("states=");
    for (size_t i = 0; i < sizeof(glm_parts) / sizeof(glm_parts[0]); i++) {
        const char *part = glm_parts[i];
        if (bytewise) {
            for (const char *c = part; *c; c++) {
                raw[used++] = *c; raw[used] = 0;
                dsml_decode_tracker_update(&tr, raw, used);
            }
        } else {
            size_t n = strlen(part);
            memcpy(raw + used, part, n); used += n; raw[used] = 0;
            dsml_decode_tracker_update(&tr, raw, used);
        }
        printf("%s%s", i ? "," : "", state_name(tr.decode));
    }
    printf("\n");
}

static void dump_glm_control_text(void) {
    char raw[512];
    snprintf(raw, sizeof(raw),
        "<think>reason</think><tool_call>write<arg_key>content</arg_key><arg_value>"
        "literal </tool_call> " DS4_TOOL_CALLS_END " </think> <think>");
    dsml_decode_tracker tr; dsml_decode_tracker_init(&tr);
    tr.model_syntax = SERVER_MODEL_SYNTAX_GLM;
    size_t n = strlen(raw);
    for (size_t i = 1; i <= n; i++) dsml_decode_tracker_update(&tr, raw, i);
    bool start = false, end = false, orphan = false;
    observe_tool_markers(&tr, raw, &start, &end, &orphan);
    printf("tracker=%s start=%d end=%d\n", state_name(tr.decode), start ? 1 : 0, end ? 1 : 0);
}

static int agent_turn_reasoning_cap(int think_mode, int max_tokens) {
    const int reserve = 64;
    int budget_cap = max_tokens > reserve ? max_tokens - reserve : 0;
    int effort_cap = 64;
    if (think_mode == DS4_THINK_HIGH) effort_cap = 256;
    else if (think_mode == DS4_THINK_MAX) effort_cap = 512;
    return budget_cap < effort_cap ? budget_cap : effort_cap;
}

static int sampling_override(bool track_tools, bool saw_tool_start, bool thinking_inside,
                             int completion, int chat_dsml, dsml_decode_state dsml,
                             int *tool_pos, int *think_pos,
                             int tool_choice, bool has_tool_results, int think_mode, int max_tokens,
                             const int *tool_pref, int tool_len, const int *think_pref, int think_len) {
    bool required_pending = tool_choice == TOOL_CHOICE_REQUIRED && track_tools && !saw_tool_start;
    bool reserve_post_thinking = required_pending || has_tool_results;
    if (reserve_post_thinking && thinking_inside) {
        if (completion >= agent_turn_reasoning_cap(think_mode, max_tokens) &&
            *think_pos < think_len) {
            return DS4_SAMPLE_OVERRIDE_TOKEN(think_pref[(*think_pos)++]);
        }
        return DS4_SAMPLE_OVERRIDE_NONE;
    }
    if (required_pending && *tool_pos < tool_len)
        return DS4_SAMPLE_OVERRIDE_TOKEN(tool_pref[(*tool_pos)++]);
    dsml_decode_state state = (track_tools && chat_dsml) ? dsml : DSML_DECODE_OUTSIDE;
    if (state != DSML_DECODE_OUTSIDE &&
        state != DSML_DECODE_STRING_BODY && state != DSML_DECODE_JSON_STRING)
        return DS4_SAMPLE_OVERRIDE_GREEDY;
    return DS4_SAMPLE_OVERRIDE_NONE;
}

int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "usage: dsml_c_oracle SCRIPT\n"); return 2; }
    const char *name = argv[1];
    if (!strcmp(name, "state-prefix")) {
        char raw[256]; snprintf(raw, sizeof(raw), "%s\n%s name=\"edit\">\n", DS4_TOOL_CALLS_START, DS4_INVOKE_START);
        dump_state(raw);
    } else if (!strcmp(name, "state-path-param")) {
        char raw[320]; snprintf(raw, sizeof(raw), "%s\n%s name=\"edit\">\n%s name=\"path\" string=\"true\">/tmp/a.py",
            DS4_TOOL_CALLS_START, DS4_INVOKE_START, DS4_PARAM_START);
        dump_state(raw);
    } else if (!strcmp(name, "state-path-closing")) {
        char raw[320]; snprintf(raw, sizeof(raw), "%s\n%s name=\"edit\">\n%s name=\"path\" string=\"true\">/tmp/a.py</",
            DS4_TOOL_CALLS_START, DS4_INVOKE_START, DS4_PARAM_START);
        dump_state(raw);
    } else if (!strcmp(name, "state-json-struct")) {
        char raw[320]; snprintf(raw, sizeof(raw), "%s\n%s name=\"edit\">\n%s name=\"edits\" string=\"false\">[{",
            DS4_TOOL_CALLS_START, DS4_INVOKE_START, DS4_PARAM_START);
        dump_state(raw);
    } else if (!strcmp(name, "state-json-string")) {
        char raw[384]; snprintf(raw, sizeof(raw),
            "%s\n%s name=\"edit\">\n%s name=\"edits\" string=\"false\">[{\"newText\":\"for i in",
            DS4_TOOL_CALLS_START, DS4_INVOKE_START, DS4_PARAM_START);
        dump_state(raw);
    } else if (!strcmp(name, "state-done")) {
        char raw[384]; snprintf(raw, sizeof(raw),
            "%s\n%s name=\"edit\">\n%s name=\"edits\" string=\"false\">[]%s\n%s\n%s",
            DS4_TOOL_CALLS_START, DS4_INVOKE_START, DS4_PARAM_START, DS4_PARAM_END, DS4_INVOKE_END, DS4_TOOL_CALLS_END);
        dump_state(raw);
    } else if (!strcmp(name, "glm-boundaries-token")) {
        dump_glm_boundaries(false);
    } else if (!strcmp(name, "glm-boundaries-byte")) {
        dump_glm_boundaries(true);
    } else if (!strcmp(name, "glm-control-text")) {
        dump_glm_control_text();
    } else if (!strcmp(name, "override-required")) {
        int tool[] = {101, 202}; int tp = 0, hp = 0;
        int a = sampling_override(true, false, false, 0, 1, DSML_DECODE_OUTSIDE, &tp, &hp,
            TOOL_CHOICE_REQUIRED, false, DS4_THINK_NONE, 128, tool, 2, NULL, 0);
        int b = sampling_override(true, false, false, 0, 1, DSML_DECODE_OUTSIDE, &tp, &hp,
            TOOL_CHOICE_REQUIRED, false, DS4_THINK_NONE, 128, tool, 2, NULL, 0);
        printf("%d %d\n", a, b);
    } else if (!strcmp(name, "override-think-cap")) {
        int think[] = {303}; int tp = 0, hp = 0;
        int a = sampling_override(true, false, true, 0, 1, DSML_DECODE_OUTSIDE, &tp, &hp,
            TOOL_CHOICE_REQUIRED, false, DS4_THINK_LOW, 128, NULL, 0, think, 1);
        int b = sampling_override(true, false, true, 64, 1, DSML_DECODE_OUTSIDE, &tp, &hp,
            TOOL_CHOICE_REQUIRED, false, DS4_THINK_LOW, 128, NULL, 0, think, 1);
        printf("%d %d\n", a, b);
    } else if (!strcmp(name, "override-tool-result")) {
        int think[] = {303}; int tp = 0, hp = 0;
        int a = sampling_override(false, false, true, 63, 1, DSML_DECODE_OUTSIDE, &tp, &hp,
            TOOL_CHOICE_AUTO, true, DS4_THINK_LOW, 128, NULL, 0, think, 1);
        int b = sampling_override(false, false, true, 64, 1, DSML_DECODE_OUTSIDE, &tp, &hp,
            TOOL_CHOICE_AUTO, true, DS4_THINK_LOW, 128, NULL, 0, think, 1);
        printf("%d %d\n", a, b);
    } else if (!strcmp(name, "cap-low-128")) {
        printf("%d\n", agent_turn_reasoning_cap(DS4_THINK_LOW, 128));
    } else if (!strcmp(name, "cap-high-128")) {
        printf("%d\n", agent_turn_reasoning_cap(DS4_THINK_HIGH, 128));
    } else if (!strcmp(name, "cap-max-600")) {
        printf("%d\n", agent_turn_reasoning_cap(DS4_THINK_MAX, 600));
    } else { fprintf(stderr, "unknown script\n"); return 2; }
    return 0;
}
