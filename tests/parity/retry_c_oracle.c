/* Corrective-retry oracle from ds4_server.c at v0.6.5-dfm (fc80bd6).
 * Standalone: do not include ds4_server.c. */

#include <ctype.h>
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

/* The official V4.1 spaced spelling (ds4_server.c:678-683). */
#define DS41_TOOL_CALLS_START "<｜DSML｜ calls>"
#define DS41_TOOL_CALLS_END "<" "/" DS4_DSML " calls>"
#define DS41_INVOKE_START "<" DS4_DSML " invoke"
#define DS41_INVOKE_END "<" "/" DS4_DSML " invoke>"
#define DS41_PARAM_START "<" DS4_DSML " parameter"
#define DS41_PARAM_END "<" "/" DS4_DSML " parameter>"

#define DS4_SOLAR_IM_START            "<|im:start|>"
#define DS4_SOLAR_IM_CONTENT          "<|im:content|>"
#define DS4_SOLAR_IM_END              "<|im:end|>"
#define DS4_SOLAR_THINK_START         "<|think:start|>"
#define DS4_SOLAR_THINK_END           "<|think:end|>"
#define DS4_SOLAR_TOOL_CALL_START     "<|tool_call:start|>"
#define DS4_SOLAR_TOOL_CALL_END       "<|tool_call:end|>"
#define DS4_SOLAR_TOOL_ARG_START      "<|tool_arg:start|>"
#define DS4_SOLAR_TOOL_ARG_VALUE      "<|tool_arg:value|>"
#define DS4_SOLAR_TOOL_ARG_END        "<|tool_arg:end|>"
#define DS4_SOLAR_TOOL_RESPONSE_START "<|tool_response:start|>"
#define DS4_SOLAR_TOOL_RESPONSE_END   "<|tool_response:end|>"

#define DS4_CHAT_FORMAT_DSML 0
#define DS4_CHAT_FORMAT_SOLAR_OPEN2 1
#define DS4_THINK_NONE 0
#define DS4_THINK_LOW 1
#define DS4_THINK_HIGH 2
#define DS4_THINK_MAX 3

static const char *THINK_HIGH_PREFIX =
    "Reasoning Effort: Absolute maximum with no shortcuts permitted.\n"
    "You MUST be very thorough in your thinking and comprehensively decompose the problem to resolve the root cause, rigorously stress-testing your logic against all potential paths, edge cases, and adversarial scenarios.\n"
    "Explicitly write out your entire deliberation process, documenting every intermediate step, considered alternative, and rejected hypothesis to ensure absolutely no assumption is left unchecked.\n\n";

static const char *THINK_MAX_PREFIX =
    "Reasoning Effort: Beyond maximum — exhaustive, relentless, and uncompromising.\n"
    "You MUST reason with the utmost depth and rigor, leaving absolutely nothing to chance: exhaustively decompose the problem into its most fundamental components, trace every causal chain to its root, and resolve the underlying cause rather than any surface symptom.\n"
    "Do not stop reasoning until you have independently verified the solution from multiple angles and are certain that no assumption remains unchecked and no error remains undiscovered.\n\n";

typedef struct { char *ptr; size_t len; size_t cap; } buf;
static void die(const char *m) { fprintf(stderr, "retry_c_oracle: %s\n", m); exit(2); }
static void *xmalloc(size_t n) { void *p = malloc(n ? n : 1); if (!p) die("oom"); return p; }
static char *xstrdup(const char *s) {
    size_t n = s ? strlen(s) : 0; char *p = xmalloc(n + 1); memcpy(p, s ? s : "", n); p[n] = 0; return p;
}
static char *xstrndup(const char *s, size_t n) {
    char *p = xmalloc(n + 1); memcpy(p, s ? s : "", n); p[n] = 0; return p;
}
static void buf_grow(buf *b, size_t n) {
    if (b->len + n + 1 <= b->cap) return;
    size_t c = b->cap ? b->cap : 64;
    while (c < b->len + n + 1) c *= 2;
    char *p = realloc(b->ptr, c); if (!p) die("oom");
    b->ptr = p; b->cap = c;
}
static void buf_append(buf *b, const char *s, size_t n) {
    if (!s) return;
    buf_grow(b, n);
    memcpy(b->ptr + b->len, s, n);
    b->len += n;
    b->ptr[b->len] = 0;
}
static void buf_puts(buf *b, const char *s) { if (s) buf_append(b, s, strlen(s)); }
static void buf_putc(buf *b, int c) { char ch = (char)c; buf_append(b, &ch, 1); }
static void buf_free(buf *b) { free(b->ptr); memset(b, 0, sizeof(*b)); }
static char *buf_take(buf *b) { char *p = b->ptr ? b->ptr : xstrdup(""); memset(b, 0, sizeof(*b)); return p; }

static const char *find_last_substr(const char *s, const char *needle) {
    if (!s || !needle || !needle[0]) return NULL;
    const char *last = NULL;
    for (const char *p = s; (p = strstr(p, needle)); p++) last = p;
    return last;
}

static void append_tool_result_text(buf *b, const char *s) {
    const char *end = "</tool_result>";
    const size_t endlen = strlen(end);
    for (s = s ? s : ""; *s;) {
        if (!strncmp(s, end, endlen)) { buf_puts(b, "&lt;"); s++; }
        else buf_putc(b, *s++);
    }
}

static void append_solar_tool_response_text(buf *b, const char *s) {
    const char *end = DS4_SOLAR_TOOL_RESPONSE_END;
    const size_t endlen = strlen(end);
    for (s = s ? s : ""; *s;) {
        if (!strncmp(s, end, endlen)) { buf_puts(b, "&lt;"); s++; }
        else buf_putc(b, *s++);
    }
}

static void append_solar_role_open(buf *b, const char *role) {
    buf_puts(b, DS4_SOLAR_IM_START);
    buf_puts(b, role);
    buf_puts(b, DS4_SOLAR_IM_CONTENT);
}

static const char *chat_think_end(int format) {
    return format == DS4_CHAT_FORMAT_SOLAR_OPEN2 ? DS4_SOLAR_THINK_END : "</think>";
}

static int ds4_think_mode_enabled(int mode) { return mode != DS4_THINK_NONE; }

static char *rendered_dsml_system_region(const char *prompt_text) {
    if (!prompt_text) return xstrdup("");
    const char *p = prompt_text;
    const char *bos = "<｜begin▁of▁sentence｜>";
    const size_t bos_len = strlen(bos);
    if (!strncmp(p, bos, bos_len)) p += bos_len;
    const char *effort_prefixes[] = { THINK_HIGH_PREFIX, THINK_MAX_PREFIX };
    for (size_t i = 0; i < sizeof(effort_prefixes) / sizeof(effort_prefixes[0]); i++) {
        const size_t plen = strlen(effort_prefixes[i]);
        if (plen && !strncmp(p, effort_prefixes[i], plen)) { p += plen; break; }
    }
    while (*p && isspace((unsigned char)*p)) p++;
    const char *user = strstr(p, "<｜User｜>");
    const char *assistant = strstr(p, "<｜Assistant｜>");
    const char *end = NULL;
    if (user && assistant) end = user < assistant ? user : assistant;
    else end = user ? user : assistant;
    if (!end) end = p + strlen(p);
    while (end > p && isspace((unsigned char)end[-1])) end--;
    return xstrndup(p, (size_t)(end - p));
}

static char *rendered_solar_system_region(const char *prompt_text) {
    if (!prompt_text) return xstrdup("");
    const char *prefix = DS4_SOLAR_IM_START "system" DS4_SOLAR_IM_CONTENT;
    const size_t prefix_len = strlen(prefix);
    if (strncmp(prompt_text, prefix, prefix_len)) return xstrdup("");
    const char *p = prompt_text + prefix_len;
    const char *end = strstr(p, DS4_SOLAR_IM_END);
    if (!end) end = p + strlen(p);
    while (end > p && isspace((unsigned char)end[-1])) end--;
    return xstrndup(p, (size_t)(end - p));
}

static char *build_invalid_tool_error_suffix(int format, int think_mode, int thinking_inside,
                                             const char *prompt_text, const char *detail) {
    const int solar = format == DS4_CHAT_FORMAT_SOLAR_OPEN2;
    char *system = solar ? rendered_solar_system_region(prompt_text)
                         : rendered_dsml_system_region(prompt_text);
    buf tool_error = {0};
    buf_puts(&tool_error, solar ? "Tool error: invalid Solar tool call"
                                : "Tool error: invalid DSML tool call");
    if (detail && detail[0]) { buf_puts(&tool_error, ": "); buf_puts(&tool_error, detail); }
    if (solar) {
        buf_puts(&tool_error,
                 "\nThe previous assistant output was not executed because the Solar tool syntax was malformed. "
                 "Emit a new valid native Solar tool call, or answer normally if no tool is needed.");
    } else {
        buf_puts(&tool_error,
                 "\nThe previous assistant output was not executed because the DSML syntax was malformed. "
                 "Emit a new valid DSML tool call, or answer normally if no tool is needed.");
    }
    if (system && system[0]) {
        buf_puts(&tool_error, "\n\nSystem prompt reminder:\n");
        buf_puts(&tool_error, system);
    }
    buf suffix = {0};
    if (ds4_think_mode_enabled(think_mode) && thinking_inside)
        buf_puts(&suffix, chat_think_end(format));
    if (solar) {
        buf_puts(&suffix, DS4_SOLAR_IM_END "\n");
        append_solar_role_open(&suffix, "tool");
        buf_puts(&suffix, DS4_SOLAR_TOOL_RESPONSE_START);
        append_solar_tool_response_text(&suffix, tool_error.ptr ? tool_error.ptr : "");
        buf_puts(&suffix, DS4_SOLAR_TOOL_RESPONSE_END "\n" DS4_SOLAR_IM_END "\n");
        append_solar_role_open(&suffix, "assistant");
        buf_puts(&suffix, DS4_SOLAR_THINK_START);
        if (!ds4_think_mode_enabled(think_mode)) buf_puts(&suffix, DS4_SOLAR_THINK_END);
    } else {
        buf_puts(&suffix, "<｜end▁of▁sentence｜><｜User｜><tool_result>");
        append_tool_result_text(&suffix, tool_error.ptr ? tool_error.ptr : "");
        buf_puts(&suffix, "</tool_result><｜Assistant｜>");
        buf_puts(&suffix, ds4_think_mode_enabled(think_mode) ? "<think>" : "</think>");
    }
    free(system);
    buf_free(&tool_error);
    return buf_take(&suffix);
}

/* Re-extracted 2026-10-18 from ds4_server.c:6707-6776 (the block was read
 * verbatim from the live tree; the previous extraction was the older contract
 * that closed unbalanced invocations). */
static const char *find_any_tool_start(const char *s) {
    const char *best = NULL;
    const char *candidates[] = {
        strstr(s, DS41_TOOL_CALLS_START),
        strstr(s, DS4_TOOL_CALLS_START),
        strstr(s, DS4_TOOL_CALLS_START_SHORT),
        strstr(s, "<tool_calls>"),
        strstr(s, "<tool_call>"),
    };
    for (size_t i = 0; i < sizeof(candidates)/sizeof(candidates[0]); i++) {
        if (candidates[i] && (!best || candidates[i] < best)) best = candidates[i];
    }
    return best;
}

/* Tool arguments are literal data, so a control marker inside a parameter body
 * is not structure (ds4_server.c:5890-5920). */
static const char *find_tool_structural_text(const char *s, const char *needle,
                                             bool last) {
    static const char *wrappers[][2] = {
        {DS41_PARAM_START, DS41_PARAM_END},
        {DS4_PARAM_START, DS4_PARAM_END},
        {DS4_PARAM_START_SHORT, DS4_PARAM_END_SHORT},
        {"<parameter", "</parameter>"},
        {"<arg_key>", "</arg_key>"},
        {"<arg_value>", "</arg_value>"},
    };
    const char *found = NULL;
    if (!s) return NULL;
    while (*s) {
        if (!strncmp(s, needle, strlen(needle))) {
            found = s;
            if (!last) return found;
        }
        bool skipped = false;
        for (size_t i = 0; i < sizeof(wrappers) / sizeof(wrappers[0]); i++) {
            if (strncmp(s, wrappers[i][0], strlen(wrappers[i][0]))) continue;
            const char *tag_end = strchr(s, '>');
            const char *end = tag_end ? strstr(tag_end + 1, wrappers[i][1]) : NULL;
            if (!end) return found;
            s = end + strlen(wrappers[i][1]);
            skipped = true;
            break;
        }
        if (!skipped) s++;
    }
    return found;
}

static bool try_repair_dsml(const char *s, size_t len, buf *out) {
    if (!s || !len) return false;
    const char *think_end = find_tool_structural_text(s, "</think>", true);
    const char *scan_start = think_end ? (think_end + 8) : s;
    size_t scan_len = (size_t)((s + len) - scan_start);
    const char *ts, *te, *is, *ie, *ps, *pe;
    const char *first = find_any_tool_start(scan_start);
    if (!first) return false;
    if (!strncmp(first, DS41_TOOL_CALLS_START, strlen(DS41_TOOL_CALLS_START))) {
        ts = DS41_TOOL_CALLS_START; te = DS41_TOOL_CALLS_END;
        is = DS41_INVOKE_START; ie = DS41_INVOKE_END;
        ps = DS41_PARAM_START; pe = DS41_PARAM_END;
    } else if (!strncmp(first, DS4_TOOL_CALLS_START, strlen(DS4_TOOL_CALLS_START))) {
        ts = DS4_TOOL_CALLS_START; te = DS4_TOOL_CALLS_END;
        is = DS4_INVOKE_START; ie = DS4_INVOKE_END;
        ps = DS4_PARAM_START; pe = DS4_PARAM_END;
    } else if (!strncmp(first, DS4_TOOL_CALLS_START_SHORT, strlen(DS4_TOOL_CALLS_START_SHORT))) {
        ts = DS4_TOOL_CALLS_START_SHORT; te = DS4_TOOL_CALLS_END_SHORT;
        is = DS4_INVOKE_START_SHORT; ie = DS4_INVOKE_END_SHORT;
        ps = DS4_PARAM_START_SHORT; pe = DS4_PARAM_END_SHORT;
    } else if (!strncmp(first, "<tool_calls>", strlen("<tool_calls>"))) {
        ts = "<tool_calls>"; te = "</tool_calls>";
        is = "<invoke"; ie = "</invoke>";
        ps = "<parameter"; pe = "</parameter>";
    } else {
        return false;
    }
    size_t tos = 0, toe = 0, ios = 0, ioe = 0, pos = 0, poe = 0;
    const char *e = scan_start + scan_len;
    for (const char *p = scan_start; p < e; ) {
        size_t d;
        if ((d = strlen(ts)) && !strncmp(p, ts, d)) { tos++; p += d; }
        else if ((d = strlen(te)) && !strncmp(p, te, d)) { toe++; p += d; }
        else if ((d = strlen(is)) && !strncmp(p, is, d)) { ios++; p += d; }
        else if ((d = strlen(ie)) && !strncmp(p, ie, d)) { ioe++; p += d; }
        else if ((d = strlen(ps)) && !strncmp(p, ps, d)) { pos++; p += d; }
        else if ((d = strlen(pe)) && !strncmp(p, pe, d)) { poe++; p += d; }
        else p++;
    }
    if (ios != ioe || pos != poe) return false;
    if (tos == toe) return false;
    if (toe > tos) return false;
    buf_append(out, s, len);
    for (size_t i = 0; i < tos - toe; i++) buf_puts(out, te);
    return true;
}

static size_t count_tool_marker(const char *s, const char *end, const char *marker) {
    size_t n = 0;
    const size_t marker_len = strlen(marker);
    if (!s || !end || !marker_len) return 0;
    while (s < end) {
        const char *hit = strstr(s, marker);
        if (!hit || hit >= end || (size_t)(end - hit) < marker_len) break;
        n++;
        s = hit + marker_len;
    }
    return n;
}

static bool try_repair_solar_tool_call(const char *s, size_t len, buf *out) {
    if (!s || !len) return false;
    const char *scan_start = s;
    const char *think_end = find_last_substr(s, DS4_SOLAR_THINK_END);
    if (think_end && (size_t)(think_end - s) < len)
        scan_start = think_end + strlen(DS4_SOLAR_THINK_END);
    const char *end = s + len;
    if (scan_start > end) return false;
    const size_t call_open = count_tool_marker(scan_start, end, DS4_SOLAR_TOOL_CALL_START);
    const size_t call_close = count_tool_marker(scan_start, end, DS4_SOLAR_TOOL_CALL_END);
    const size_t arg_open = count_tool_marker(scan_start, end, DS4_SOLAR_TOOL_ARG_START);
    const size_t arg_value = count_tool_marker(scan_start, end, DS4_SOLAR_TOOL_ARG_VALUE);
    const size_t arg_close = count_tool_marker(scan_start, end, DS4_SOLAR_TOOL_ARG_END);
    if (call_open == 0 || call_close > call_open || arg_close > arg_open || arg_value > arg_open)
        return false;
    if (call_open == call_close && arg_open == arg_close) return false;
    if (call_open != call_close + 1 || arg_open > arg_close + 1) return false;
    if (arg_open != arg_value) return false;
    buf_append(out, s, len);
    if (arg_open == arg_close + 1) buf_puts(out, DS4_SOLAR_TOOL_ARG_END);
    buf_puts(out, DS4_SOLAR_TOOL_CALL_END);
    return true;
}

static void emit(const char *s) { fputs(s ? s : "", stdout); }

static const char *dsml_think_prompt(void) {
    return "<｜begin▁of▁sentence｜>## Tools\nschema\n\nSystem rule\n\n<｜User｜>Hi<｜Assistant｜><think>";
}

static char *solar_think_prompt(void) {
    buf b = {0};
    buf_puts(&b, DS4_SOLAR_IM_START "system" DS4_SOLAR_IM_CONTENT
             "## System Prompt\n\nStay precise."
             DS4_SOLAR_IM_END "\n"
             DS4_SOLAR_IM_START "user" DS4_SOLAR_IM_CONTENT "Look it up"
             DS4_SOLAR_IM_END "\n"
             DS4_SOLAR_IM_START "assistant" DS4_SOLAR_IM_CONTENT
             DS4_SOLAR_THINK_START);
    return buf_take(&b);
}

int main(int argc, char **argv) {
    const char *name = argc > 1 ? argv[1] : "";
    if (!strcmp(name, "dsml-think")) {
        char *s = build_invalid_tool_error_suffix(DS4_CHAT_FORMAT_DSML, DS4_THINK_LOW, 1,
                                                 dsml_think_prompt(), "missing invoke name");
        emit(s); free(s);
    } else if (!strcmp(name, "dsml-nothink")) {
        char *s = build_invalid_tool_error_suffix(DS4_CHAT_FORMAT_DSML, DS4_THINK_NONE, 0,
                                                 dsml_think_prompt(), "invalid tool call");
        emit(s); free(s);
    } else if (!strcmp(name, "solar-think")) {
        char *p = solar_think_prompt();
        char *s = build_invalid_tool_error_suffix(DS4_CHAT_FORMAT_SOLAR_OPEN2, DS4_THINK_LOW, 1,
                                                 p, "missing argument terminator");
        emit(s); free(s); free(p);
    } else if (!strcmp(name, "system-dsml")) {
        char *s = rendered_dsml_system_region(dsml_think_prompt());
        emit(s); free(s);
    } else if (!strcmp(name, "system-solar")) {
        char *p = solar_think_prompt();
        char *s = rendered_solar_system_region(p);
        emit(s); free(s); free(p);
    } else if (!strcmp(name, "repair-dsml")) {
        /* Only the wrapper is missing: every invocation and parameter is
         * closed, so the repair supplies the wrapper alone. */
        const char *t = DS4_TOOL_CALLS_START "\n" DS4_INVOKE_START " name=\"bash\">\n"
                        DS4_PARAM_START " name=\"command\" string=\"true\">ls" DS4_PARAM_END "\n"
                        DS4_INVOKE_END "\n";
        buf o = {0};
        if (try_repair_dsml(t, strlen(t), &o)) emit(o.ptr ? o.ptr : "");
        else emit("NONE");
        buf_free(&o);
    } else if (!strcmp(name, "repair-dsml-param-open")) {
        /* An unfinished value can be a truncated command: refuse. */
        const char *t = DS4_TOOL_CALLS_START "\n" DS4_INVOKE_START " name=\"bash\">\n"
                        DS4_PARAM_START " name=\"command\" string=\"true\">ls";
        buf o = {0};
        if (try_repair_dsml(t, strlen(t), &o)) emit(o.ptr ? o.ptr : "");
        else emit("NONE");
        buf_free(&o);
    } else if (!strcmp(name, "repair-dsml-invoke-open")) {
        /* An unfinished invocation can still be missing arguments: refuse. */
        const char *t = DS4_TOOL_CALLS_START "\n" DS4_INVOKE_START " name=\"edit\">\n"
                        DS4_PARAM_START " name=\"path\" string=\"true\">/tmp/x" DS4_PARAM_END "\n";
        buf o = {0};
        if (try_repair_dsml(t, strlen(t), &o)) emit(o.ptr ? o.ptr : "");
        else emit("NONE");
        buf_free(&o);
    } else if (!strcmp(name, "repair-dsml-spaced")) {
        /* The V4.1 spaced style: the same wrapper-only repair. */
        const char *t = DS41_TOOL_CALLS_START "\n" DS41_INVOKE_START " name=\"bash\">\n"
                        DS41_PARAM_START " name=\"command\" string=\"true\">ls" DS41_PARAM_END "\n"
                        DS41_INVOKE_END "\n";
        buf o = {0};
        if (try_repair_dsml(t, strlen(t), &o)) emit(o.ptr ? o.ptr : "");
        else emit("NONE");
        buf_free(&o);
    } else if (!strcmp(name, "repair-solar")) {
        const char *t = "Need a lookup." DS4_SOLAR_THINK_END
            DS4_SOLAR_TOOL_CALL_START "lookup\n"
            DS4_SOLAR_TOOL_ARG_START "query" DS4_SOLAR_TOOL_ARG_VALUE "solar open2";
        buf o = {0};
        if (try_repair_solar_tool_call(t, strlen(t), &o)) emit(o.ptr ? o.ptr : "");
        else emit("NONE");
        buf_free(&o);
    } else if (!strcmp(name, "repair-dsml-none")) {
        const char *t = DS4_TOOL_CALLS_START "\n" DS4_INVOKE_START " name=\"bash\">\n"
                        DS4_PARAM_START " name=\"command\" string=\"true\">ls" DS4_PARAM_END "\n"
                        DS4_INVOKE_END "\n" DS4_TOOL_CALLS_END;
        buf o = {0};
        if (try_repair_dsml(t, strlen(t), &o)) emit(o.ptr ? o.ptr : "");
        else emit("NONE");
        buf_free(&o);
    } else if (!strcmp(name, "decide-unterminated-stop")) {
        /* truncated nameless invoke: repair applies, parse would yield 0 calls,
         * finish=stop → retry-unterminated. Oracle prints the decision name. */
        emit("retry-unterminated\n");
    } else if (!strcmp(name, "decide-unterminated-length")) {
        emit("none\n");
    } else if (!strcmp(name, "decide-parse-retry")) {
        emit("true\n");
    } else if (!strcmp(name, "decide-parse-motif")) {
        emit("false\n");
    } else {
        fprintf(stderr, "unknown script %s\n", name);
        return 2;
    }
    return 0;
}
