use super::harness::{drive_static_chat, drive_static_chat_with, pair_jobs, SpyStatic};
use super::*;
use crate::generate::GenerateError;
use crate::route::{WireSurface, LANE_SERIAL, LANE_STATIC, NEED_STREAMING};

fn peer(footprint: i64, peer_ok: bool) -> CoalescePeer {
    CoalescePeer { footprint, peer_ok }
}

#[test]
fn static_width_error_matches_c_bridge_string_when_n_lt_2() {
    // Given: C generate_batch_jobs is n>=2
    // When: owner sees n=1
    // Then: C bridge width string
    assert_eq!(static_width_error(0), Some(STATIC_WIDTH_ERR));
    assert_eq!(static_width_error(1), Some(STATIC_WIDTH_ERR));
    assert_eq!(
        STATIC_WIDTH_ERR,
        "static batch request count is out of range"
    );
}

#[test]
fn static_width_admits_n_eq_2() {
    // Given: a coalesced pair
    // When: owner checks width
    // Then: admitted
    assert_eq!(static_width_error(2), None);
    assert_eq!(static_width_error(3), None);
}

#[test]
fn run_static_returns_c_error_string_when_n_eq_1() {
    // Given: one fixture row
    let tokens = [10, 20];
    let jobs = [StaticJob {
        tokens: &tokens,
        max_new_tokens: 8,
        eos: 99,
    }];
    let mut spy = SpyStatic::default();

    // When
    let err = run_static(&mut spy, &jobs).unwrap_err();

    // Then: C string, generate_static not called
    match err {
        GenerateError::Engine(msg) => assert_eq!(msg, STATIC_WIDTH_ERR),
        other => panic!("expected Engine, got {other:?}"),
    }
    assert_eq!(spy.calls, 0);
}

#[test]
fn run_static_invokes_generate_static_when_n_eq_2() {
    // Given: two fixture rows (no GGUF)
    let first = [10, 20];
    let second = [30];
    let jobs = pair_jobs(&first, &second);
    let mut spy = SpyStatic::default();

    // When
    let rows = run_static(&mut spy, &jobs).unwrap();

    // Then: per-row lengths stay ragged; fallback stays cold
    assert_eq!(spy.calls, 1);
    assert_eq!(spy.fallbacks, 0);
    assert_eq!(rows.len(), 2);
    assert_eq!(rows[0].tokens, first);
    assert_eq!(rows[1].tokens, second);
    assert_eq!([rows[0].tokens.len(), rows[1].tokens.len()], [2, 1]);
}

#[test]
fn static_routed_n1_collapses_to_serial() {
    // Given: route_decide picked LANE_STATIC (seq_cap=0, needs=0)
    // When: a single static-routed request is served (n=1, no coalesce)
    let (response, inner) = drive_static_chat();
    let text = String::from_utf8_lossy(&response);

    // Then: C worker_main n==1 → run_job_single. generate_static stays
    // n>=2 (see run_static_returns_c_error_string_when_n_eq_1). Decision
    // reason stays static; the entry lane is serial.
    assert!(text.starts_with("HTTP/1.1 200 OK"), "{text}");
    assert!(
        !text.contains(STATIC_WIDTH_ERR),
        "HTTP n=1 must not refuse width: {text}"
    );
    assert!(
        text.contains("serial-fallback"),
        "serial decode must run: {text}"
    );
    assert_eq!(inner.runtime.requests_serial, 1);
    assert_eq!(
        inner.metrics.route_requests[WireSurface::OpenaiChat as usize][LANE_SERIAL as usize],
        1
    );
    assert_eq!(
        inner.metrics.route_requests[WireSurface::OpenaiChat as usize][LANE_STATIC as usize],
        0
    );
}

#[test]
fn static_routed_request_invokes_generate_static_when_n_eq_2() {
    // Given: route_decide picked LANE_STATIC and one sibling is waiting
    let mut spy = SpyStatic {
        siblings: vec![OwnedStaticJob {
            tokens: vec![30],
            max_new_tokens: 1,
            eos: -1,
        }],
        ..Default::default()
    };

    // When: the HTTP request is served through handle_client_inner
    let (response, inner) = drive_static_chat_with(&mut spy);
    let text = String::from_utf8_lossy(&response);

    // Then: generate_static ran; serial session path stayed cold
    assert_eq!(spy.calls, 1);
    assert!(text.starts_with("HTTP/1.1 200 OK"), "{text}");
    assert!(
        !text.contains(STATIC_WIDTH_ERR),
        "width refuse must not fire at n=2: {text}"
    );
    assert!(
        !text.contains("serial-fallback"),
        "serial decode must not run: {text}"
    );
    assert_eq!(inner.runtime.requests_serial, 0);
    assert_eq!(
        inner.metrics.route_requests[WireSurface::OpenaiChat as usize][LANE_STATIC as usize],
        1
    );
}

#[test]
fn settle_static_finish_maps_c_eos_and_budget() {
    // Given: C r->finish ? "stop" : "length"
    // When: EOS / budget / over-budget
    // Then: wire strings stay stop|length; over-budget clamps
    assert_eq!(settle_static_finish(StaticFinish::Stop, 3, 8), ("stop", 3));
    assert_eq!(
        settle_static_finish(StaticFinish::Length, 3, 8),
        ("length", 3)
    );
    assert_eq!(
        settle_static_finish(StaticFinish::Stop, 9, 8),
        ("length", 8)
    );
}

#[test]
fn static_row_cut_at_a_stop_token_finishes_stop() {
    // Given: native decoded past a host stop token to the budget
    let row = StaticRow {
        tokens: vec![5, 6, 99, 7, 8],
        finish: StaticFinish::Length,
    };
    // When: the host cuts the text at its stop set
    let (text, finish, emitted) = settle::detok_static_row(&row, 5, |t| t == 99, |t| vec![t as u8]);
    // Then: the request stopped; the text ends before the stop token
    assert_eq!((text, finish, emitted), (vec![5, 6], "stop", 2));
}

#[test]
fn static_routed_request_settles_c_chat_terminal_when_n_eq_2() {
    // Given: LANE_STATIC n=2 success (todos 11-12 wrote "{}")
    let mut spy = SpyStatic {
        siblings: vec![OwnedStaticJob {
            tokens: vec![30],
            max_new_tokens: 1,
            eos: -1,
        }],
        ..Default::default()
    };

    // When
    let (response, inner) = drive_static_chat_with(&mut spy);
    let text = String::from_utf8_lossy(&response);

    // Then: C write_batch_completion shape; no invented timings/IDs
    assert_eq!(spy.calls, 1);
    assert!(text.starts_with("HTTP/1.1 200 OK"), "{text}");
    assert!(
        text.contains("\"object\":\"chat.completion\""),
        "missing C object: {text}"
    );
    assert!(
        text.contains("\"finish_reason\":\"stop\""),
        "missing C finish_reason: {text}"
    );
    assert!(
        !text.contains("\"timings\""),
        "C static omits timings: {text}"
    );
    assert!(
        !text.contains("serial-fallback"),
        "serial decode must not run: {text}"
    );
    assert_eq!(inner.runtime.requests_serial, 0);
}

#[test]
fn coalesce_take_joins_two_queued_static_jobs_when_c_would() {
    // Given: C coalesce_gather head + one static-peer-ok queued job under cap/budget
    let queued = [peer(8, true)];
    let limits = CoalesceLimits {
        cap: 8,
        max_tok_total: 4096,
    };

    // When
    let take = coalesce_take(8, &queued, limits);

    // Then: n=2 group (head kept, one extra taken)
    assert_eq!(take, 1);
}

#[test]
fn coalesce_take_stops_at_non_peer_to_preserve_fifo() {
    // Given: next queued job is not static-peer-ok, a later one is
    let queued = [peer(8, false), peer(8, true)];

    // When
    let take = coalesce_take(8, &queued, CoalesceLimits::UNBOUNDED);

    // Then: C breaks on the non-batchable head; later peer stays queued
    assert_eq!(take, 0);
}

#[test]
fn coalesce_take_leaves_token_overflow_peer_queued() {
    // Given: adding the next footprint would exceed max_tok_total
    let queued = [peer(100, true)];
    let limits = CoalesceLimits {
        cap: 8,
        max_tok_total: 150,
    };

    // When
    let take = coalesce_take(100, &queued, limits);

    // Then: C splits into another batch; head stays alone
    assert_eq!(take, 0);
}

#[test]
fn run_static_uses_c_fallback_when_ctx_overflows() {
    // Given: n=2 fits gather but not StaticBatchContext width
    let first = [10, 20];
    let second = [30];
    let jobs = pair_jobs(&first, &second);
    let mut spy = SpyStatic {
        ctx_max_seq: 1,
        ..Default::default()
    };

    // When
    let err = run_static(&mut spy, &jobs).unwrap_err();

    // Then: C fallback text, generate_static skipped, never serial
    assert_eq!(spy.calls, 0);
    assert_eq!(spy.fallbacks, 1);
    match err {
        GenerateError::Engine(msg) => assert_eq!(msg, STATIC_FALLBACK_ERR),
        other => panic!("expected Engine, got {other:?}"),
    }
}

#[test]
fn run_static_routed_coalesces_queued_sibling_when_c_would() {
    // Given: one queued sibling C would join (peer-ok, under budget)
    let current_tokens = [10, 20];
    let current = StaticJob {
        tokens: &current_tokens,
        max_new_tokens: 8,
        eos: -1,
    };
    let mut spy = SpyStatic {
        siblings: vec![OwnedStaticJob {
            tokens: vec![30],
            max_new_tokens: 1,
            eos: -1,
        }],
        ..Default::default()
    };

    // When
    let rows = run_static_routed(&mut spy, current).unwrap();

    // Then: one generate_static call at n=2
    assert_eq!(spy.calls, 1);
    assert_eq!(spy.ns, vec![2]);
    assert_eq!(spy.fallbacks, 0);
    assert_eq!(rows.len(), 2);
}

#[test]
fn static_peer_ok_matches_c_needs_free_openai() {
    // Given: C job_static_peer_ok / needs==0 on OpenAI chat
    let ok = StaticPeerSpec {
        needs: 0,
        surface: WireSurface::OpenaiChat,
        cont_anthropic: false,
        cont_responses: false,
    };
    let streaming = StaticPeerSpec {
        needs: NEED_STREAMING,
        surface: WireSurface::OpenaiChat,
        cont_anthropic: false,
        cont_responses: false,
    };

    // When / Then
    assert!(static_peer_ok(ok));
    assert!(!static_peer_ok(streaming));
}
