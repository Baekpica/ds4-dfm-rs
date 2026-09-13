//! C↔Rust `/metrics` + `/v1/stats` memgov porcelain.

use ds4_server::{
    dump_memgov_names, render_memgov_metrics, render_metrics, render_stats_memgov_json, AdmitState,
    MemgovSnap, ReconcileSnap, RouteMetrics, RuntimeMetrics,
};

use std::path::PathBuf;
use std::process::Command;

fn oracle() -> PathBuf {
    if let Ok(p) = std::env::var("DS4_MEMGOV_C_ORACLE") {
        return PathBuf::from(p);
    }
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/parity/memgov_c_oracle")
}

fn require_oracle() -> PathBuf {
    let p = oracle();
    assert!(
        p.exists(),
        "build the C oracle first: make tests/parity/memgov_c_oracle (missing {})",
        p.display()
    );
    p
}

fn c_str(args: &[&str]) -> String {
    let out = Command::new(require_oracle())
        .args(args)
        .output()
        .expect("run memgov_c_oracle");
    assert!(
        out.status.success(),
        "memgov_c_oracle {:?} failed: {}",
        args,
        String::from_utf8_lossy(&out.stderr)
    );
    String::from_utf8(out.stdout).expect("oracle utf8")
}

fn supported_runtime() -> RuntimeMetrics {
    let mut rt = RuntimeMetrics::default();
    rt.uptime_seconds = 12;
    rt.requests_started = 3;
    rt.requests_completed = 2;
    rt.requests_failed = 1;
    rt.requests_inflight = 1;
    rt.requests_serial = 1;
    rt.out_backlog_bytes = 4096;
    rt.cont_admit_rejects = 1;
    rt.cont_credit_ext_granted = 2;
    rt.cont_credit_ext_refused = 3;
    rt.have_batch_ctx = true;
    rt.cont_commit_observed = 64;
    rt.cont_commit_packed = 48;
    rt.cont_admit_band_x1024 = 3;
    rt.serial_idle_reaps = 4;
    rt.cont_batch_failures = 5;
    rt.graph_fit_refusals = 6;
    rt.rejected[0][0] = 7;
    rt.reclaim_banks[1] = 8;
    rt.reclaim_bytes[2] = 9;
    rt.admits_cold = 1;
    rt.admits_warm = 2;
    rt.admits_fork = 3;
    rt.admits_partial_fork = 4;
    rt.admits_partial_truncate = 5;
    rt.tokens_prefilled_computed = 10;
    rt.tokens_prefilled_cached = 11;
    rt.tokens_decoded = 12;
    rt.decode_steps = 13;
    rt.spec_drafts = 4;
    rt.spec_hits = 2;
    rt.spec_quench = 1;
    rt.decode_tok_s = 1.5;
    rt.prefill_tok_s = 2.25;
    rt.tok_per_step = 0.5;
    rt.banks_live = 2;
    rt.banks_total = 4;
    rt.warm_records = 3;
    rt.kv_pages_resident = 5;
    rt.cont_ontoken_ns = 100;
    rt.cont_ontoken_tokens = 10;
    rt.batch_genmu_wait_ns = 200;
    rt.batch_genmu_waits = 2;
    rt.creg_published = 1;
    rt.creg_resolved = 2;
    rt.creg_missed = 3;
    rt.creg_demoted = 4;
    rt.creg_records_live = 1;
    rt.artifact_source = 2;
    rt.derived_artifacts = 3;
    rt.derived_artifact_bytes = 4096;
    rt.memgov.census_supported = true;
    rt.memgov.cells[11][0].requested = 200;
    rt.memgov.cells[11][0].committed = 256;
    rt.memgov.census_faults = 1;
    rt.memgov.torn_fallbacks = 2;
    rt.memgov.census_epoch = 7;
    rt.memgov.substrate_outstanding = 1024;
    rt.memgov.obs_status = 0;
    rt.memgov.obs_source = 1;
    rt.memgov.obs_free = 100;
    rt.memgov.obs_total = 200;
    rt.memgov.obs_cuda_free = 90;
    rt.memgov.obs_meminfo = 80;
    rt.memgov.memobs_calls = [1, 2, 3];
    rt.memgov.memobs_errors = 4;
    rt.memgov.own_trim_calls = 5;
    rt.memgov.own_trim_recovered = 6;
    rt.memgov.reconcile = Some(ReconcileSnap {
        residual: -10,
        onetime: 20,
    });
    rt.memgov.reconcile_flagged = 1;
    rt.memgov.gov_epoch = 3;
    rt.memgov.leases[0].intent = 10;
    rt.memgov.leases[0].resident = 20;
    rt.memgov.leases[0].reservation = 30;
    rt.memgov.decisions[0][0][0] = 1;
    rt.memgov.deficit[3] = 99;
    rt.memgov.faults = 8;
    rt.memgov.residency_supported = true;
    rt.memgov.residency_units[0][0] = 5;
    rt.memgov.residency_failures[0][0] = 1;
    rt
}

#[test]
fn memgov_names_match_c() {
    assert_eq!(dump_memgov_names(), c_str(&["names"]));
}

#[test]
fn metrics_full_zero_matches_c() {
    let rust = render_metrics(
        &RouteMetrics::default(),
        &AdmitState::default(),
        &RuntimeMetrics::default(),
    );
    assert_eq!(rust, c_str(&["full-zero"]));
}

#[test]
fn memgov_unsupported_matches_c() {
    assert_eq!(
        render_memgov_metrics(&MemgovSnap::default()),
        c_str(&["unsupported"])
    );
}

#[test]
fn metrics_supported_matches_c() {
    let rust = render_metrics(
        &RouteMetrics::default(),
        &AdmitState::default(),
        &supported_runtime(),
    );
    assert_eq!(rust, c_str(&["supported"]));
}

#[test]
fn mixed_artifact_source_matches_c() {
    let mut rt = supported_runtime();
    rt.artifact_source = 3;
    let rust = render_metrics(&RouteMetrics::default(), &AdmitState::default(), &rt);
    assert!(rust.contains("source=\"mixed\""));
    assert_eq!(rust, c_str(&["mixed"]));
}

#[test]
fn stats_memgov_zero_matches_c() {
    assert_eq!(
        render_stats_memgov_json(&MemgovSnap::default()),
        c_str(&["stats-zero"])
    );
}

#[test]
fn stats_memgov_supported_matches_c() {
    assert_eq!(
        render_stats_memgov_json(&supported_runtime().memgov),
        c_str(&["stats-supported"])
    );
}
