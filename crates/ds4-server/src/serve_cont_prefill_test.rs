use std::ffi::OsStr;

use crate::serve_cont_prefill::{
    owner_tick_call_count, owner_tick_pair, reset_owner_tick_call_count, tick_roll_prefill,
    PrefillChunkPolicy, RollPhase, TickOp, DEFAULT_PREFILL_CHUNK, DEFAULT_PREFILL_CHUNK_LIVE,
    ENV_PREFILL_CHUNK, ENV_PREFILL_CHUNK_LIVE,
};
use crate::serve_cont_roll::ContRoll;

#[test]
fn default_chunk_env_names_match_c() {
    assert_eq!(ENV_PREFILL_CHUNK, "DS4_CONT_PREFILL_CHUNK");
    assert_eq!(ENV_PREFILL_CHUNK_LIVE, "DS4_CONT_PREFILL_CHUNK_LIVE");
    let policy = PrefillChunkPolicy::from_os(None, None, false);
    assert_eq!(policy.boot, DEFAULT_PREFILL_CHUNK);
    assert_eq!(policy.live, DEFAULT_PREFILL_CHUNK_LIVE);
    assert!(policy.interleave());
    if std::env::var_os(ENV_PREFILL_CHUNK).is_none()
        && std::env::var_os(ENV_PREFILL_CHUNK_LIVE).is_none()
    {
        assert_eq!(PrefillChunkPolicy::from_env(), policy);
    }
}

#[test]
fn live_decode_is_stepped_while_another_job_prefills_a_chunk() {
    // Given: one live decode job and another still prefilling
    let mut roll = ContRoll::new();
    roll.enqueue(1);
    roll.enqueue(2);
    let decode = roll.admit().expect("live decode admits");
    let prefill = roll
        .admit()
        .expect("prefill admits while decode is generating");
    let policy = PrefillChunkPolicy::from_raw(4096, 512, false);
    let mut jobs = [
        (decode, RollPhase::Decode { remaining: 4 }),
        (prefill, RollPhase::Prefill { remaining: 3000 }),
    ];

    // When: one C-loop pass
    let ops = tick_roll_prefill(policy, &mut jobs);

    // Then: prefill advances one LIVE chunk and the live job is decode-stepped
    assert_eq!(
        ops,
        vec![
            TickOp::Prefill {
                user: prefill,
                tokens: 512
            },
            TickOp::Decode { user: decode },
        ]
    );
    assert_eq!(jobs[1].1, RollPhase::Prefill { remaining: 2488 });
    assert_eq!(jobs[0].1, RollPhase::Decode { remaining: 3 });
}

#[test]
fn live_chunk_env_zero_disables_live_shrink_as_c() {
    // Given: LIVE=0 (C: interleave off; drain at boot width)
    let policy =
        PrefillChunkPolicy::from_os(Some(OsStr::new("4096")), Some(OsStr::new("0")), false);
    assert!(!policy.interleave());
    assert_eq!(policy.chunk_width(true), 4096);
    let mut jobs = [
        (1, RollPhase::Decode { remaining: 4 }),
        (2, RollPhase::Prefill { remaining: 5000 }),
    ];

    // When: one C-loop pass
    let ops = tick_roll_prefill(policy, &mut jobs);

    // Then: two boot-width chunks drain before any decode step
    assert_eq!(
        ops,
        vec![
            TickOp::Prefill {
                user: 2,
                tokens: 4096
            },
            TickOp::Prefill {
                user: 2,
                tokens: 904
            },
            TickOp::Decode { user: 1 },
        ]
    );
    assert_eq!(jobs[1].1, RollPhase::Done);
    assert_eq!(jobs[0].1, RollPhase::Decode { remaining: 3 });
}

#[test]
fn idle_prefill_keeps_boot_width_when_nobody_is_decoding() {
    // Given: interleave on, but no live decode
    let policy = PrefillChunkPolicy::from_raw(4096, 512, false);
    assert_eq!(policy.chunk_width(false), 4096);
    let mut jobs = [(1, RollPhase::Prefill { remaining: 5000 })];

    // When: one C-loop pass
    let ops = tick_roll_prefill(policy, &mut jobs);

    // Then: idle prefill keeps the boot chunk, not LIVE shrink
    assert_eq!(
        ops,
        vec![TickOp::Prefill {
            user: 1,
            tokens: 4096
        }]
    );
    assert_eq!(jobs[0].1, RollPhase::Prefill { remaining: 904 });
}

#[test]
fn an_impossible_mix_does_not_bind() {
    let mut req = ds4_core::ServingRequest::default();
    req.sched_chunk = Some(8192);
    req.native_chunk = Some(1024);
    let plan = ds4_core::resolve_plan(
        &req,
        Some(ds4_core::serving_caps(
            ds4_core::ModelFamily::Qwen4Exp,
            ds4_core::Variant::Qwen38FlashNext,
        )),
        &ds4_core::EngineFacts::default(),
    );
    assert!(!plan.may_listen());
    let cfg = crate::ServerConfig::default();
    assert!(crate::listen_if_allowed(&cfg, &plan)
        .expect("bind skipped")
        .is_none());
}

#[test]
fn live_decode_uses_a_verified_native_capped_chunk() {
    let mut req = ds4_core::ServingRequest::default();
    req.sched_chunk = Some(3000);
    req.sched_chunk_live = Some(700);
    req.native_chunk = Some(2048);
    let plan = ds4_core::resolve_plan(
        &req,
        Some(ds4_core::serving_caps(
            ds4_core::ModelFamily::Qwen4Exp,
            ds4_core::Variant::Qwen38FlashNext,
        )),
        &ds4_core::EngineFacts::default(),
    );
    let native = plan.effective.native_chunk.expect("native");
    assert!(ds4_core::VERIFIED_PREFILL_CHUNKS.contains(&plan.effective.sched_chunk));
    assert!(ds4_core::VERIFIED_PREFILL_CHUNKS.contains(&plan.effective.sched_chunk_live));
    assert!(plan.effective.sched_chunk <= native);
    assert!(plan.effective.sched_chunk_live <= native);
    assert_ne!(plan.effective.sched_chunk, 3000);

    let policy = PrefillChunkPolicy::from_raw(
        plan.effective.sched_chunk as i32,
        plan.effective.sched_chunk_live as i32,
        false,
    );
    assert!(policy.boot <= native);
    assert!(policy.live <= native);
    assert!(ds4_core::VERIFIED_PREFILL_CHUNKS.contains(&policy.boot));
    assert!(ds4_core::VERIFIED_PREFILL_CHUNKS.contains(&policy.live));

    let mut jobs = [
        (1, RollPhase::Decode { remaining: 4 }),
        (2, RollPhase::Prefill { remaining: 3000 }),
    ];
    let ops = tick_roll_prefill(policy, &mut jobs);
    assert_eq!(
        ops,
        vec![
            TickOp::Prefill {
                user: 2,
                tokens: policy.live
            },
            TickOp::Decode { user: 1 },
        ]
    );
}

#[test]
fn owner_tick_pair_steps_live_decode_while_peer_prefills_live_chunk() {
    reset_owner_tick_call_count();
    let policy = PrefillChunkPolicy::from_raw(4096, 512, false);

    let ops = owner_tick_pair(policy, 4, 3000);

    assert_eq!(owner_tick_call_count(), 1);
    assert_eq!(
        ops,
        vec![
            TickOp::Prefill {
                user: 2,
                tokens: 512
            },
            TickOp::Decode { user: 1 },
        ]
    );
}
