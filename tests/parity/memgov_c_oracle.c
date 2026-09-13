/* Standalone C /metrics + /v1/stats memgov porcelain oracle from
 * ds4_server.c at v0.6.3-dfm. Do not include ds4.c / ds4_server.c. */

#include <stdio.h>
#include <string.h>

static const char *mem_class_names[] = {
    "engine_other",   "weight_arena",   "weight_span",   "weight_whole",
    "weight_derived", "weight_import",  "weight_artifact", "weight_host_pin",
    "stage_pin",      "scalars_mirror", "session_tensors", "kv_primary",
    "batch_bank",     "scratch_sticky", "kernel_partials", "graph_exec",
    "diag",
};
static const char *mem_domain_names[] = { "unified_device", "pinned_host" };
static const char *mem_obs_status_names[] = { "ok", "unsupported", "query_error" };
static const char *mem_obs_source_names[] = { "none", "cuda_free", "meminfo_available" };
static const char *gov_consumer_names[] = {
    "engine_boot", "prewarm", "batch_bank_plan", "serial_session",
    "static_batch",
};
static const char *gov_status_names[] = {
    "admit", "refuse_class", "refuse_live", "retry_obs", "unsupported",
    "fault",
};
static const char *gov_cmp_names[] = {
    "agree", "live_stricter", "shadow_stricter", "verdict_class",
    "obs_policy", "fault",
};
static const char *rejlane_names[] = { "continuous", "serial", "static" };
static const char *reject_reason_names[] = {
    "class_budget", "live_headroom", "obs_retry", "unsupported", "fault",
    "deep_policy", "lane_disabled",
};
static const char *reclaim_status_names[] = {
    "ok", "partial", "stale_plan", "busy", "unsupported", "device_error",
};
static const char *resunit_state_names[] = {
    "populated", "already_ready", "satisfied_import",
    "host_mapped_by_policy", "lazy_deferred", "expert_cold_by_policy",
    "waived_optional", "failed",
};
static const char *resstage_names[] = { "boot", "lazy" };
static const char *residency_policy_names[] = { "eager", "lazy", "mapped", "unattributed" };
static const char *model_role_names[] = { "primary", "auxiliary", "drafter" };
static const char *route_surface_names[] = {
    "openai_chat", "openai_completion", "anthropic_messages", "openai_responses",
};
static const char *route_lane_names[] = { "serial", "continuous", "static" };
static const char *route_reason_names[] = {
    "continuous", "static_no_cont", "static_prompt_bounds", "coalesce_off",
    "surface", "need_live_frontier", "need_continuation_publish",
    "need_corrective_recovery", "need_durable_response", "need_prefill_only",
    "token_ids_projection", "tools_completion_kind", "cont_unavailable",
    "continuous_bank_continuation",
};
static const char *shed_names[] = {
    "clients", "queue_depth", "queue_bytes", "queue_age", "slow_reader",
    "continuation_hold",
};
static const char *think_names[] = { "none", "low", "high", "max" };

typedef struct {
    unsigned long long requested, committed, freed_requested, freed_committed;
} cell;

typedef struct {
    unsigned long long intent, resident, reservation;
} lease;

typedef struct {
    int census_supported;
    cell cells[17][2];
    unsigned long long census_faults;
    unsigned long long torn_fallbacks;
    unsigned long long census_epoch;
    unsigned long long substrate_outstanding;
    int emit_substrate;
    int obs_status;
    int obs_source;
    unsigned long long obs_free, obs_total, obs_cuda_free, obs_meminfo;
    unsigned long long memobs_calls[3];
    unsigned long long memobs_errors;
    unsigned long long own_trim_calls;
    unsigned long long own_trim_recovered;
    int have_reconcile;
    long long reconcile_residual;
    unsigned long long reconcile_onetime;
    unsigned long long reconcile_flagged;
    unsigned long long gov_epoch;
    lease leases[5];
    unsigned long long decisions[5][6][6];
    unsigned long long deficit[5];
    unsigned long long faults;
    const char *gov_modes[5];
    int residency_supported;
    unsigned long long residency_units[4][8];
    unsigned long long residency_failures[3][2];
} memgov;

typedef struct {
    unsigned long long uptime_seconds;
    unsigned long long requests_started;
    unsigned long long requests_completed;
    unsigned long long requests_failed;
    unsigned long long requests_canceled;
    unsigned long long requests_refused_deep_serial;
    unsigned long long requests_inflight;
    unsigned long long requests_serial;
    unsigned long long out_backlog_bytes;
    unsigned long long cont_admit_rejects;
    unsigned long long cont_credit_ext_granted;
    unsigned long long cont_credit_ext_refused;
    int have_batch_ctx;
    unsigned long long cont_commit_observed;
    unsigned long long cont_commit_packed;
    unsigned cont_admit_band_x1024;
    unsigned long long serial_idle_reaps;
    unsigned long long cont_batch_failures;
    unsigned long long graph_fit_refusals;
    unsigned long long rejected[3][7];
    unsigned long long reclaim_banks[6];
    unsigned long long reclaim_bytes[6];
    unsigned long long admits_cold, admits_warm, admits_fork;
    unsigned long long admits_partial_fork, admits_partial_truncate;
    unsigned long long tokens_prefilled_computed, tokens_prefilled_cached;
    unsigned long long tokens_decoded, decode_steps;
    unsigned long long spec_drafts, spec_hits, spec_quench;
    double decode_tok_s, prefill_tok_s, tok_per_step;
    unsigned long long banks_live, banks_total, warm_records, kv_pages_resident;
    unsigned long long cont_ontoken_ns, cont_ontoken_tokens;
    unsigned long long batch_genmu_wait_ns, batch_genmu_waits;
    unsigned long long creg_published, creg_resolved, creg_missed, creg_demoted;
    unsigned long long creg_records_live;
    unsigned long long artifact_source, derived_artifacts, derived_artifact_bytes;
    memgov memgov;
} runtime;

static unsigned long long live_req(cell c) { return c.requested - c.freed_requested; }
static unsigned long long live(cell c) { return c.committed - c.freed_committed; }
static unsigned long long slack(cell c)
{
    unsigned long long com = live(c), req = live_req(c);
    return com > req ? com - req : 0;
}

static const char *artifact_source_name(unsigned long long source)
{
    return source == 3 ? "mixed" : source == 2 ? "built" : source == 1 ? "imported" : "none";
}

static void zero_runtime(runtime *rt)
{
    memset(rt, 0, sizeof(*rt));
    rt->memgov.emit_substrate = 1;
    rt->memgov.obs_status = 1;
    for (int c = 0; c < 5; c++) rt->memgov.gov_modes[c] = "enforce";
}

static void dump_names(void)
{
    int i;
    for (i = 0; i < 17; i++) printf("CLASS %s\n", mem_class_names[i]);
    for (i = 0; i < 2; i++) printf("DOMAIN %s\n", mem_domain_names[i]);
    for (i = 0; i < 5; i++) printf("GOV %s\n", gov_consumer_names[i]);
    for (i = 0; i < 6; i++) printf("STATUS %s\n", gov_status_names[i]);
    for (i = 0; i < 6; i++) printf("CMP %s\n", gov_cmp_names[i]);
    for (i = 0; i < 3; i++) printf("REJLANE %s\n", rejlane_names[i]);
    for (i = 0; i < 7; i++) printf("REJECT %s\n", reject_reason_names[i]);
    for (i = 0; i < 6; i++) printf("RECLAIM %s\n", reclaim_status_names[i]);
    for (i = 0; i < 3; i++) printf("OBS_STATUS %s\n", mem_obs_status_names[i]);
    for (i = 0; i < 3; i++) printf("OBS_SOURCE %s\n", mem_obs_source_names[i]);
    for (i = 0; i < 4; i++) printf("POLICY %s\n", residency_policy_names[i]);
    for (i = 0; i < 8; i++) printf("RESUNIT %s\n", resunit_state_names[i]);
    for (i = 0; i < 2; i++) printf("RESSTAGE %s\n", resstage_names[i]);
    for (i = 0; i < 3; i++) printf("ROLE %s\n", model_role_names[i]);
}

static void dump_prefix(const runtime *rt)
{
    printf("# TYPE ds4_uptime_seconds gauge\n"
           "ds4_uptime_seconds %llu\n"
           "# TYPE ds4_requests_started_total counter\n"
           "ds4_requests_started_total %llu\n"
           "# TYPE ds4_requests_total counter\n"
           "ds4_requests_total{outcome=\"completed\"} %llu\n"
           "ds4_requests_total{outcome=\"failed\"} %llu\n"
           "ds4_requests_total{outcome=\"canceled\"} %llu\n"
           "ds4_requests_total{outcome=\"refused_deep_serial\"} %llu\n"
           "# TYPE ds4_requests_inflight gauge\n"
           "ds4_requests_inflight %llu\n"
           "# TYPE ds4_requests_serial_total counter\n"
           "ds4_requests_serial_total %llu\n",
           rt->uptime_seconds, rt->requests_started, rt->requests_completed,
           rt->requests_failed, rt->requests_canceled,
           rt->requests_refused_deep_serial, rt->requests_inflight,
           rt->requests_serial);
}

static void dump_fragment(void)
{
    int si, li, ri, ti;
    fputs("# TYPE ds4_route_requests_total counter\n", stdout);
    for (si = 0; si < 4; si++)
        for (li = 0; li < 3; li++)
            printf("ds4_route_requests_total{surface=\"%s\",lane=\"%s\"} 0\n",
                   route_surface_names[si], route_lane_names[li]);
    fputs("# TYPE ds4_route_decisions_total counter\n", stdout);
    for (ri = 0; ri < 14; ri++)
        printf("ds4_route_decisions_total{reason=\"%s\"} 0\n", route_reason_names[ri]);
    fputs("# TYPE ds4_requests_shed_total counter\n", stdout);
    for (ri = 0; ri < 6; ri++)
        printf("ds4_requests_shed_total{reason=\"%s\"} 0\n", shed_names[ri]);
    fputs("# TYPE ds4_requests_think_total counter\n", stdout);
    for (ti = 0; ti < 4; ti++)
        printf("ds4_requests_think_total{mode=\"%s\"} 0\n", think_names[ti]);
    fputs("# TYPE ds4_clients_connected gauge\n"
          "ds4_clients_connected 0\n"
          "# TYPE ds4_queue_depth gauge\n"
          "ds4_queue_depth 0\n"
          "# TYPE ds4_inflight_body_bytes gauge\n"
          "ds4_inflight_body_bytes 0\n", stdout);
}

static void dump_runtime(const runtime *rt)
{
    int ln, rr, rs;
    printf("# TYPE ds4_out_backlog_bytes gauge\n"
           "ds4_out_backlog_bytes %llu\n"
           "# TYPE ds4_cont_admit_rejects_total counter\n"
           "ds4_cont_admit_rejects_total %llu\n"
           "# TYPE ds4_cont_credit_extension_granted_total counter\n"
           "ds4_cont_credit_extension_granted_total %llu\n"
           "# TYPE ds4_cont_credit_extension_refused_total counter\n"
           "ds4_cont_credit_extension_refused_total %llu\n",
           rt->out_backlog_bytes, rt->cont_admit_rejects,
           rt->cont_credit_ext_granted, rt->cont_credit_ext_refused);
    if (rt->have_batch_ctx)
        printf("# TYPE ds4_cont_commit_bytes_per_token gauge\n"
               "ds4_cont_commit_bytes_per_token{kind=\"observed\"} %llu\n"
               "ds4_cont_commit_bytes_per_token{kind=\"packed\"} %llu\n"
               "# TYPE ds4_cont_admit_band_x1024 gauge\n"
               "ds4_cont_admit_band_x1024 %u\n",
               rt->cont_commit_observed, rt->cont_commit_packed,
               rt->cont_admit_band_x1024);
    printf("# TYPE ds4_serial_idle_reaps_total counter\n"
           "ds4_serial_idle_reaps_total %llu\n"
           "# TYPE ds4_cont_batch_failures_total counter\n"
           "ds4_cont_batch_failures_total %llu\n"
           "# TYPE ds4_graph_fit_refusals_total counter\n"
           "ds4_graph_fit_refusals_total %llu\n",
           rt->serial_idle_reaps, rt->cont_batch_failures, rt->graph_fit_refusals);
    fputs("# TYPE ds4_requests_rejected_total counter\n", stdout);
    for (ln = 0; ln < 3; ln++)
        for (rr = 0; rr < 7; rr++)
            printf("ds4_requests_rejected_total{lane=\"%s\",reason=\"%s\"} %llu\n",
                   rejlane_names[ln], reject_reason_names[rr], rt->rejected[ln][rr]);
    fputs("# TYPE ds4_reclaim_banks_total counter\n", stdout);
    for (rs = 0; rs < 6; rs++)
        printf("ds4_reclaim_banks_total{result=\"%s\"} %llu\n",
               reclaim_status_names[rs], rt->reclaim_banks[rs]);
    fputs("# TYPE ds4_reclaim_bytes_total counter\n", stdout);
    for (rs = 0; rs < 6; rs++)
        printf("ds4_reclaim_bytes_total{result=\"%s\"} %llu\n",
               reclaim_status_names[rs], rt->reclaim_bytes[rs]);
    printf("# TYPE ds4_admits_total counter\n"
           "ds4_admits_total{kind=\"cold\"} %llu\n"
           "ds4_admits_total{kind=\"warm\"} %llu\n"
           "ds4_admits_total{kind=\"fork\"} %llu\n"
           "ds4_admits_total{kind=\"partial_fork\"} %llu\n"
           "ds4_admits_total{kind=\"partial_truncate\"} %llu\n"
           "# TYPE ds4_tokens_prefilled_total counter\n"
           "ds4_tokens_prefilled_total{kind=\"computed\"} %llu\n"
           "ds4_tokens_prefilled_total{kind=\"cached\"} %llu\n"
           "# TYPE ds4_tokens_decoded_total counter\n"
           "ds4_tokens_decoded_total %llu\n"
           "# TYPE ds4_decode_steps_total counter\n"
           "ds4_decode_steps_total %llu\n"
           "# TYPE ds4_spec_drafts_total counter\n"
           "ds4_spec_drafts_total %llu\n"
           "# TYPE ds4_spec_hits_total counter\n"
           "ds4_spec_hits_total %llu\n"
           "# TYPE ds4_spec_quench_total counter\n"
           "ds4_spec_quench_total %llu\n"
           "# TYPE ds4_spec_accept_ratio gauge\n"
           "ds4_spec_accept_ratio %.4f\n"
           "# TYPE ds4_decode_tok_s gauge\n"
           "ds4_decode_tok_s %.2f\n"
           "# TYPE ds4_prefill_tok_s gauge\n"
           "ds4_prefill_tok_s %.2f\n"
           "# TYPE ds4_tok_per_step gauge\n"
           "ds4_tok_per_step %.3f\n"
           "# TYPE ds4_banks_live gauge\n"
           "ds4_banks_live %llu\n"
           "# TYPE ds4_banks_total gauge\n"
           "ds4_banks_total %llu\n"
           "# TYPE ds4_warm_records gauge\n"
           "ds4_warm_records %llu\n"
           "# TYPE ds4_kv_pages_resident gauge\n"
           "ds4_kv_pages_resident %llu\n"
           "# TYPE ds4_cont_ontoken_ns_total counter\n"
           "ds4_cont_ontoken_ns_total %llu\n"
           "# TYPE ds4_cont_ontoken_tokens_total counter\n"
           "ds4_cont_ontoken_tokens_total %llu\n"
           "# TYPE ds4_batch_genmu_wait_ns_total counter\n"
           "ds4_batch_genmu_wait_ns_total %llu\n"
           "# TYPE ds4_batch_genmu_waits_total counter\n"
           "ds4_batch_genmu_waits_total %llu\n"
           "# TYPE ds4_continuation_records_published_total counter\n"
           "ds4_continuation_records_published_total %llu\n"
           "# TYPE ds4_continuation_resolved_total counter\n"
           "ds4_continuation_resolved_total %llu\n"
           "# TYPE ds4_continuation_missed_total counter\n"
           "ds4_continuation_missed_total %llu\n"
           "# TYPE ds4_continuation_demoted_total counter\n"
           "ds4_continuation_demoted_total %llu\n"
           "# TYPE ds4_continuation_records_live gauge\n"
           "ds4_continuation_records_live %llu\n"
           "# TYPE ds4_derived_artifacts gauge\n"
           "ds4_derived_artifacts{source=\"%s\"} %llu\n"
           "# TYPE ds4_derived_artifact_bytes gauge\n"
           "ds4_derived_artifact_bytes %llu\n",
           rt->admits_cold, rt->admits_warm, rt->admits_fork,
           rt->admits_partial_fork, rt->admits_partial_truncate,
           rt->tokens_prefilled_computed, rt->tokens_prefilled_cached,
           rt->tokens_decoded, rt->decode_steps, rt->spec_drafts, rt->spec_hits,
           rt->spec_quench,
           rt->spec_drafts ? (double)rt->spec_hits / (double)rt->spec_drafts : 0.0,
           rt->decode_tok_s, rt->prefill_tok_s, rt->tok_per_step,
           rt->banks_live, rt->banks_total, rt->warm_records, rt->kv_pages_resident,
           rt->cont_ontoken_ns, rt->cont_ontoken_tokens,
           rt->batch_genmu_wait_ns, rt->batch_genmu_waits,
           rt->creg_published, rt->creg_resolved, rt->creg_missed, rt->creg_demoted,
           rt->creg_records_live, artifact_source_name(rt->artifact_source),
           rt->derived_artifacts, rt->derived_artifact_bytes);
}

static void dump_memgov(const memgov *g)
{
    int d, c, st, so, r, p, sg;
    printf("# TYPE ds4_memory_census_supported gauge\n"
           "ds4_memory_census_supported %d\n", g->census_supported);
    if (g->census_supported) {
        fputs("# TYPE ds4_memory_bytes gauge\n", stdout);
        for (d = 0; d < 2; d++) {
            for (c = 0; c < 17; c++) {
                cell cc = g->cells[c][d];
                printf("ds4_memory_bytes{domain=\"%s\",class=\"%s\",state=\"requested\"} %llu\n"
                       "ds4_memory_bytes{domain=\"%s\",class=\"%s\",state=\"allocated\"} %llu\n"
                       "ds4_memory_bytes{domain=\"%s\",class=\"%s\",state=\"slack\"} %llu\n",
                       mem_domain_names[d], mem_class_names[c], live_req(cc),
                       mem_domain_names[d], mem_class_names[c], live(cc),
                       mem_domain_names[d], mem_class_names[c], slack(cc));
            }
        }
        printf("# TYPE ds4_memory_census_faults_total counter\n"
               "ds4_memory_census_faults_total %llu\n", g->census_faults);
    }
    printf("# TYPE ds4_memory_census_torn_fallbacks_total counter\n"
           "ds4_memory_census_torn_fallbacks_total %llu\n"
           "# TYPE ds4_memory_census_epoch gauge\n"
           "ds4_memory_census_epoch %llu\n",
           g->torn_fallbacks, g->census_epoch);
    if (g->emit_substrate)
        printf("# TYPE ds4_memory_substrate_outstanding_bytes gauge\n"
               "ds4_memory_substrate_outstanding_bytes %llu\n",
               g->substrate_outstanding);
    fputs("# TYPE ds4_memory_observation_info gauge\n", stdout);
    for (st = 0; st < 3; st++)
        for (so = 0; so < 3; so++)
            printf("ds4_memory_observation_info{status=\"%s\",source=\"%s\"} %d\n",
                   mem_obs_status_names[st], mem_obs_source_names[so],
                   (g->obs_status == st && g->obs_source == so) ? 1 : 0);
    if (g->obs_status == 0)
        printf("# TYPE ds4_memory_observation_bytes gauge\n"
               "ds4_memory_observation_bytes{kind=\"free\"} %llu\n"
               "ds4_memory_observation_bytes{kind=\"total\"} %llu\n"
               "ds4_memory_observation_bytes{kind=\"cuda_free\"} %llu\n"
               "ds4_memory_observation_bytes{kind=\"meminfo_available\"} %llu\n",
               g->obs_free, g->obs_total, g->obs_cuda_free, g->obs_meminfo);
    fputs("# TYPE ds4_memory_observation_calls_total counter\n", stdout);
    for (so = 0; so < 3; so++)
        printf("ds4_memory_observation_calls_total{source=\"%s\"} %llu\n",
               mem_obs_source_names[so], g->memobs_calls[so]);
    printf("# TYPE ds4_memory_observation_errors_total counter\n"
           "ds4_memory_observation_errors_total %llu\n"
           "# TYPE ds4_mem_own_trim_calls_total counter\n"
           "ds4_mem_own_trim_calls_total %llu\n"
           "# TYPE ds4_mem_own_trim_recovered_bytes_total counter\n"
           "ds4_mem_own_trim_recovered_bytes_total %llu\n",
           g->memobs_errors, g->own_trim_calls, g->own_trim_recovered);
    if (g->have_reconcile)
        printf("# TYPE ds4_mem_reconcile_residual_bytes gauge\n"
               "ds4_mem_reconcile_residual_bytes %lld\n"
               "# TYPE ds4_mem_reconcile_onetime_bytes gauge\n"
               "ds4_mem_reconcile_onetime_bytes %llu\n",
               g->reconcile_residual, g->reconcile_onetime);
    printf("# TYPE ds4_mem_reconcile_flagged_total counter\n"
           "ds4_mem_reconcile_flagged_total %llu\n"
           "# TYPE ds4_memory_governor_epoch gauge\n"
           "ds4_memory_governor_epoch %llu\n",
           g->reconcile_flagged, g->gov_epoch);
    fputs("# TYPE ds4_memory_lease_bytes gauge\n", stdout);
    for (c = 0; c < 5; c++)
        printf("ds4_memory_lease_bytes{consumer=\"%s\",field=\"intent\"} %llu\n"
               "ds4_memory_lease_bytes{consumer=\"%s\",field=\"resident\"} %llu\n"
               "ds4_memory_lease_bytes{consumer=\"%s\",field=\"reservation\"} %llu\n",
               gov_consumer_names[c], g->leases[c].intent,
               gov_consumer_names[c], g->leases[c].resident,
               gov_consumer_names[c], g->leases[c].reservation);
    fputs("# TYPE ds4_memory_decisions_total counter\n", stdout);
    for (c = 0; c < 5; c++)
        for (st = 0; st < 6; st++)
            for (r = 0; r < 6; r++)
                printf("ds4_memory_decisions_total{consumer=\"%s\",result=\"%s\",reason=\"%s\"} %llu\n",
                       gov_consumer_names[c], gov_status_names[st],
                       gov_cmp_names[r], g->decisions[c][st][r]);
    fputs("# TYPE ds4_memory_decision_deficit_bytes gauge\n", stdout);
    for (c = 0; c < 5; c++)
        printf("ds4_memory_decision_deficit_bytes{consumer=\"%s\"} %llu\n",
               gov_consumer_names[c], g->deficit[c]);
    printf("# TYPE ds4_memory_governor_faults_total counter\n"
           "ds4_memory_governor_faults_total %llu\n", g->faults);
    if (g->residency_supported) {
        fputs("# TYPE ds4_residency_units counter\n", stdout);
        for (p = 0; p < 4; p++)
            for (st = 0; st < 8; st++)
                printf("ds4_residency_units{policy=\"%s\",state=\"%s\"} %llu\n",
                       residency_policy_names[p], resunit_state_names[st],
                       g->residency_units[p][st]);
        fputs("# TYPE ds4_residency_failures_total counter\n", stdout);
        for (r = 0; r < 3; r++)
            for (sg = 0; sg < 2; sg++)
                printf("ds4_residency_failures_total{model_role=\"%s\",stage=\"%s\"} %llu\n",
                       model_role_names[r], resstage_names[sg],
                       g->residency_failures[r][sg]);
    }
}

static void dump_stats_memgov(const memgov *g)
{
    int c, d, r, st;
    printf("\"memory\":{\"census_supported\":%s,\"census_epoch\":%llu",
           g->census_supported ? "true" : "false", g->census_epoch);
    if (g->census_supported) {
        unsigned long long tl[2] = {0}, ts[2] = {0};
        fputs(",\"classes\":{", stdout);
        for (c = 0; c < 17; c++) {
            printf("%s\"%s\":{", c ? "," : "", mem_class_names[c]);
            for (d = 0; d < 2; d++) {
                unsigned long long lv = live(g->cells[c][d]);
                unsigned long long sl = slack(g->cells[c][d]);
                tl[d] += lv;
                ts[d] += sl;
                printf("%s\"%s\":{\"live\":%llu,\"slack\":%llu}",
                       d ? "," : "", mem_domain_names[d], lv, sl);
            }
            fputc('}', stdout);
        }
        printf("},\"totals\":{\"device_live\":%llu,\"device_slack\":%llu,"
               "\"host_pin_live\":%llu,\"host_pin_slack\":%llu},"
               "\"census_faults\":%llu",
               tl[0], ts[0], tl[1], ts[1], g->census_faults);
        if (g->emit_substrate)
            printf(",\"substrate_outstanding\":%llu", g->substrate_outstanding);
    }
    printf(",\"observation\":{\"status\":\"%s\",\"source\":\"%s\"",
           mem_obs_status_names[g->obs_status], mem_obs_source_names[g->obs_source]);
    if (g->obs_status == 0)
        printf(",\"free_bytes\":%llu,\"total_bytes\":%llu"
               ",\"cuda_free_bytes\":%llu,\"meminfo_available_bytes\":%llu",
               g->obs_free, g->obs_total, g->obs_cuda_free, g->obs_meminfo);
    printf(",\"calls_cuda_free\":%llu,\"calls_meminfo_available\":%llu,\"errors\":%llu}",
           g->memobs_calls[1], g->memobs_calls[2], g->memobs_errors);
    fputs(",\"reconcile\":{\"supported\":false}}", stdout);
    printf(",\"governor\":{\"shadow\":true,\"epoch\":%llu,\"faults\":%llu,\"modes\":{",
           g->gov_epoch, g->faults);
    for (c = 0; c < 5; c++)
        printf("%s\"%s\":\"%s\"", c ? "," : "", gov_consumer_names[c], g->gov_modes[c]);
    fputs("},\"leases\":{", stdout);
    for (c = 0; c < 5; c++)
        printf("%s\"%s\":{\"intent\":%llu,\"resident\":%llu,\"reservation\":%llu}",
               c ? "," : "", gov_consumer_names[c],
               g->leases[c].intent, g->leases[c].resident, g->leases[c].reservation);
    fputs("},\"decisions\":{", stdout);
    for (r = 0; r < 6; r++) {
        unsigned long long n = 0;
        for (c = 0; c < 5; c++)
            for (st = 0; st < 6; st++)
                n += g->decisions[c][st][r];
        printf("%s\"%s\":%llu", r ? "," : "", gov_cmp_names[r], n);
    }
    fputs("}}", stdout);
}

static void fill_supported(runtime *rt)
{
    zero_runtime(rt);
    rt->uptime_seconds = 12;
    rt->requests_started = 3;
    rt->requests_completed = 2;
    rt->requests_failed = 1;
    rt->requests_inflight = 1;
    rt->requests_serial = 1;
    rt->out_backlog_bytes = 4096;
    rt->cont_admit_rejects = 1;
    rt->cont_credit_ext_granted = 2;
    rt->cont_credit_ext_refused = 3;
    rt->have_batch_ctx = 1;
    rt->cont_commit_observed = 64;
    rt->cont_commit_packed = 48;
    rt->cont_admit_band_x1024 = 3;
    rt->serial_idle_reaps = 4;
    rt->cont_batch_failures = 5;
    rt->graph_fit_refusals = 6;
    rt->rejected[0][0] = 7;
    rt->reclaim_banks[1] = 8;
    rt->reclaim_bytes[2] = 9;
    rt->admits_cold = 1;
    rt->admits_warm = 2;
    rt->admits_fork = 3;
    rt->admits_partial_fork = 4;
    rt->admits_partial_truncate = 5;
    rt->tokens_prefilled_computed = 10;
    rt->tokens_prefilled_cached = 11;
    rt->tokens_decoded = 12;
    rt->decode_steps = 13;
    rt->spec_drafts = 4;
    rt->spec_hits = 2;
    rt->spec_quench = 1;
    rt->decode_tok_s = 1.5;
    rt->prefill_tok_s = 2.25;
    rt->tok_per_step = 0.5;
    rt->banks_live = 2;
    rt->banks_total = 4;
    rt->warm_records = 3;
    rt->kv_pages_resident = 5;
    rt->cont_ontoken_ns = 100;
    rt->cont_ontoken_tokens = 10;
    rt->batch_genmu_wait_ns = 200;
    rt->batch_genmu_waits = 2;
    rt->creg_published = 1;
    rt->creg_resolved = 2;
    rt->creg_missed = 3;
    rt->creg_demoted = 4;
    rt->creg_records_live = 1;
    rt->artifact_source = 2;
    rt->derived_artifacts = 3;
    rt->derived_artifact_bytes = 4096;
    rt->memgov.census_supported = 1;
    rt->memgov.cells[11][0].requested = 200;
    rt->memgov.cells[11][0].committed = 256;
    rt->memgov.census_faults = 1;
    rt->memgov.torn_fallbacks = 2;
    rt->memgov.census_epoch = 7;
    rt->memgov.substrate_outstanding = 1024;
    rt->memgov.obs_status = 0;
    rt->memgov.obs_source = 1;
    rt->memgov.obs_free = 100;
    rt->memgov.obs_total = 200;
    rt->memgov.obs_cuda_free = 90;
    rt->memgov.obs_meminfo = 80;
    rt->memgov.memobs_calls[0] = 1;
    rt->memgov.memobs_calls[1] = 2;
    rt->memgov.memobs_calls[2] = 3;
    rt->memgov.memobs_errors = 4;
    rt->memgov.own_trim_calls = 5;
    rt->memgov.own_trim_recovered = 6;
    rt->memgov.have_reconcile = 1;
    rt->memgov.reconcile_residual = -10;
    rt->memgov.reconcile_onetime = 20;
    rt->memgov.reconcile_flagged = 1;
    rt->memgov.gov_epoch = 3;
    rt->memgov.leases[0].intent = 10;
    rt->memgov.leases[0].resident = 20;
    rt->memgov.leases[0].reservation = 30;
    rt->memgov.decisions[0][0][0] = 1;
    rt->memgov.deficit[3] = 99;
    rt->memgov.faults = 8;
    rt->memgov.residency_supported = 1;
    rt->memgov.residency_units[0][0] = 5;
    rt->memgov.residency_failures[0][0] = 1;
}

int main(int argc, char **argv)
{
    runtime rt;
    if (argc < 2) {
        fprintf(stderr, "usage: memgov_c_oracle names|full-zero|unsupported|supported|stats-zero|stats-supported\n");
        return 2;
    }
    if (!strcmp(argv[1], "names")) {
        dump_names();
        return 0;
    }
    zero_runtime(&rt);
    if (!strcmp(argv[1], "full-zero")) {
        dump_prefix(&rt);
        dump_fragment();
        dump_runtime(&rt);
        dump_memgov(&rt.memgov);
        return 0;
    }
    if (!strcmp(argv[1], "unsupported")) {
        dump_memgov(&rt.memgov);
        return 0;
    }
    if (!strcmp(argv[1], "stats-zero")) {
        dump_stats_memgov(&rt.memgov);
        return 0;
    }
    if (!strcmp(argv[1], "supported") || !strcmp(argv[1], "mixed")) {
        fill_supported(&rt);
        if (!strcmp(argv[1], "mixed")) { rt.artifact_source = 3; }
        dump_prefix(&rt);
        dump_fragment();
        dump_runtime(&rt);
        dump_memgov(&rt.memgov);
        return 0;
    }
    if (!strcmp(argv[1], "stats-supported")) {
        fill_supported(&rt);
        dump_stats_memgov(&rt.memgov);
        return 0;
    }
    fprintf(stderr, "unknown command: %s\n", argv[1]);
    return 2;
}
