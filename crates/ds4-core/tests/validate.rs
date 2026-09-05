//! Host config_validate vs C validate_c_oracle. Synthetic metadata only.

use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

fn oracle() -> PathBuf {
    if let Ok(p) = std::env::var("DS4_VALIDATE_C_ORACLE") {
        return PathBuf::from(p);
    }
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/parity/validate_c_oracle")
}

fn require_oracle() -> PathBuf {
    let p = oracle();
    assert!(
        p.exists(),
        "build the C oracle first: make tests/parity/validate_c_oracle (missing {})",
        p.display()
    );
    p
}

fn c_parse(path: &Path) -> String {
    let out = Command::new(require_oracle())
        .arg(path)
        .output()
        .expect("run validate_c_oracle");
    assert!(
        out.status.success(),
        "oracle failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    String::from_utf8(out.stdout).expect("oracle utf8")
}

enum Val<'a> {
    U32(u32),
    U64(u64),
    F32(f32),
    Bool(bool),
    Str(&'a str),
    ArrayU32(&'a [u32]),
    ArrayF32(&'a [f32]),
    ArrayBool(&'a [bool]),
}

fn put_u32(buf: &mut Vec<u8>, v: u32) {
    buf.extend_from_slice(&v.to_le_bytes());
}
fn put_u64(buf: &mut Vec<u8>, v: u64) {
    buf.extend_from_slice(&v.to_le_bytes());
}
fn put_str(buf: &mut Vec<u8>, s: &str) {
    put_u64(buf, s.len() as u64);
    buf.extend_from_slice(s.as_bytes());
}

fn write_gguf(path: &Path, kvs: &[(&str, Val<'_>)]) {
    let mut buf = Vec::new();
    put_u32(&mut buf, 0x4655_4747);
    put_u32(&mut buf, 3);
    put_u64(&mut buf, 0);
    put_u64(&mut buf, kvs.len() as u64);
    for (key, val) in kvs {
        put_str(&mut buf, key);
        match val {
            Val::U32(v) => {
                put_u32(&mut buf, 4);
                buf.extend_from_slice(&v.to_le_bytes());
            }
            Val::U64(v) => {
                put_u32(&mut buf, 10);
                buf.extend_from_slice(&v.to_le_bytes());
            }
            Val::F32(v) => {
                put_u32(&mut buf, 6);
                buf.extend_from_slice(&v.to_le_bytes());
            }
            Val::Bool(v) => {
                put_u32(&mut buf, 7);
                buf.push(u8::from(*v));
            }
            Val::Str(s) => {
                put_u32(&mut buf, 8);
                put_str(&mut buf, s);
            }
            Val::ArrayU32(items) => {
                put_u32(&mut buf, 9);
                put_u32(&mut buf, 4);
                put_u64(&mut buf, items.len() as u64);
                for x in *items {
                    buf.extend_from_slice(&x.to_le_bytes());
                }
            }
            Val::ArrayF32(items) => {
                put_u32(&mut buf, 9);
                put_u32(&mut buf, 6);
                put_u64(&mut buf, items.len() as u64);
                for x in *items {
                    buf.extend_from_slice(&x.to_le_bytes());
                }
            }
            Val::ArrayBool(items) => {
                put_u32(&mut buf, 9);
                put_u32(&mut buf, 7);
                put_u64(&mut buf, items.len() as u64);
                for x in *items {
                    buf.push(u8::from(*x));
                }
            }
        }
    }
    while buf.len() < 32 {
        buf.push(0);
    }
    fs::write(path, buf).unwrap();
}

fn tmp(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join("ds4-validate-parity");
    fs::create_dir_all(&dir).unwrap();
    dir.join(name)
}

fn assert_same(path: &Path) {
    let c = c_parse(path);
    let rust = ds4_core::dump_validate(path);
    assert_eq!(rust, c, "mismatch for {}", path.display());
}

const FLASH_DIMS: &[(&str, u32)] = &[
    ("deepseek4.block_count", 43),
    ("deepseek4.embedding_length", 4096),
    ("deepseek4.vocab_size", 129280),
    ("deepseek4.attention.head_count", 64),
    ("deepseek4.attention.head_count_kv", 1),
    ("deepseek4.attention.key_length", 512),
    ("deepseek4.attention.value_length", 512),
    ("deepseek4.rope.dimension_count", 64),
    ("deepseek4.attention.q_lora_rank", 1024),
    ("deepseek4.attention.output_lora_rank", 1024),
    ("deepseek4.attention.output_group_count", 8),
    ("deepseek4.expert_count", 256),
    ("deepseek4.expert_used_count", 6),
    ("deepseek4.expert_feed_forward_length", 2048),
    ("deepseek4.expert_shared_count", 1),
    ("deepseek4.hash_layer_count", 3),
    ("deepseek4.attention.sliding_window", 128),
    ("deepseek4.attention.indexer.head_count", 64),
    ("deepseek4.attention.indexer.key_length", 128),
    ("deepseek4.attention.indexer.top_k", 512),
    ("deepseek4.hyper_connection.count", 4),
    ("deepseek4.hyper_connection.sinkhorn_iterations", 20),
];

fn flash_compress(il: u32) -> u32 {
    if il < 2 {
        0
    } else if (il & 1) == 0 {
        4
    } else {
        128
    }
}

fn pro_compress(il: u32) -> u32 {
    if il < 2 {
        128
    } else if (il & 1) == 0 {
        4
    } else {
        128
    }
}

fn flash_ok<'a>(
    compress: &'a [u32],
    swiglu: &'a [f32],
    arch: Option<&'a str>,
) -> Vec<(&'a str, Val<'a>)> {
    let mut v = Vec::new();
    if let Some(a) = arch {
        v.push(("general.architecture", Val::Str(a)));
    }
    for (k, n) in FLASH_DIMS {
        v.push((*k, Val::U32(*n)));
    }
    v.push((
        "deepseek4.attention.compress_ratios",
        Val::ArrayU32(compress),
    ));
    v.push(("deepseek4.swiglu_clamp_exp", Val::ArrayF32(swiglu)));
    v.push(("deepseek4.rope.freq_base", Val::F32(10000.0)));
    v.push((
        "deepseek4.attention.compress_rope_freq_base",
        Val::F32(160000.0),
    ));
    v.push(("deepseek4.expert_weights_scale", Val::F32(1.5)));
    v.push((
        "deepseek4.attention.layer_norm_rms_epsilon",
        Val::F32(1.0e-6),
    ));
    v.push(("deepseek4.hyper_connection.epsilon", Val::F32(1.0e-6)));
    v.push(("deepseek4.expert_weights_norm", Val::Bool(true)));
    v
}

fn pro_ok<'a>(compress: &'a [u32], swiglu: &'a [f32]) -> Vec<(&'a str, Val<'a>)> {
    let mut v = vec![("general.architecture", Val::Str("deepseek4"))];
    for (k, n) in FLASH_DIMS {
        let n = match *k {
            "deepseek4.block_count" => 61,
            "deepseek4.embedding_length" => 7168,
            "deepseek4.attention.head_count" => 128,
            "deepseek4.attention.q_lora_rank" => 1536,
            "deepseek4.attention.output_group_count" => 16,
            "deepseek4.expert_count" => 384,
            "deepseek4.expert_feed_forward_length" => 3072,
            "deepseek4.attention.indexer.top_k" => 1024,
            _ => *n,
        };
        v.push((*k, Val::U32(n)));
    }
    v.push((
        "deepseek4.attention.compress_ratios",
        Val::ArrayU32(compress),
    ));
    v.push(("deepseek4.swiglu_clamp_exp", Val::ArrayF32(swiglu)));
    v.push(("deepseek4.rope.freq_base", Val::F32(10000.0)));
    v.push((
        "deepseek4.attention.compress_rope_freq_base",
        Val::F32(160000.0),
    ));
    v.push(("deepseek4.expert_weights_scale", Val::F32(2.5)));
    v.push((
        "deepseek4.attention.layer_norm_rms_epsilon",
        Val::F32(1.0e-6),
    ));
    v.push(("deepseek4.hyper_connection.epsilon", Val::F32(1.0e-6)));
    v.push(("deepseek4.expert_weights_norm", Val::Bool(true)));
    v
}

fn motif_ok() -> Vec<(&'static str, Val<'static>)> {
    vec![
        ("general.architecture", Val::Str("motif3")),
        ("motif3.block_count", Val::U32(53)),
        ("motif3.context_length", Val::U64(262144)),
        ("motif3.embedding_length", Val::U32(4096)),
        ("motif3.vocab_size", Val::U32(220160)),
        ("motif3.feed_forward_length", Val::U32(12288)),
        ("motif3.leading_dense_block_count", Val::U32(2)),
        ("motif3.expert_count", Val::U32(384)),
        ("motif3.expert_used_count", Val::U32(8)),
        ("motif3.expert_feed_forward_length", Val::U32(1280)),
        ("motif3.expert_shared_count", Val::U32(1)),
        ("motif3.expert_gating_func", Val::U32(1)),
        ("motif3.attention.head_count", Val::U32(80)),
        ("motif3.attention.head_count_kv", Val::U32(16)),
        ("motif3.attention.noise_head_count", Val::U32(16)),
        ("motif3.attention.key_length", Val::U32(192)),
        ("motif3.attention.value_length", Val::U32(128)),
        ("motif3.attention.q_lora_rank", Val::U32(1024)),
        ("motif3.attention.kv_lora_rank", Val::U32(512)),
        ("motif3.attention.rope_dimension_count", Val::U32(64)),
        ("motif3.attention.sliding_window", Val::U32(128)),
        ("motif3.attention.sliding_window_period", Val::U32(4)),
        ("motif3.mhc.expansion_rate", Val::U32(4)),
        ("motif3.mhc.sinkhorn_iterations", Val::U32(20)),
        ("motif3.mtp.block_count", Val::U32(1)),
        ("motif3.expert_weights_norm", Val::Bool(true)),
        ("motif3.attention.elementwise_output_gate", Val::Bool(true)),
        ("motif3.mhc.enabled", Val::Bool(true)),
        ("motif3.polynorm.sigmoid_weight", Val::Bool(true)),
        ("motif3.rope.scaling.apply_mscale", Val::Bool(false)),
        ("motif3.expert_weights_scale", Val::F32(2.0)),
        ("motif3.expert_score_correction", Val::F32(1.0e-4)),
        ("motif3.attention.layer_norm_rms_epsilon", Val::F32(1.0e-5)),
        ("motif3.rope.freq_base", Val::F32(10000.0)),
        ("motif3.rope.freq_base_swa", Val::F32(10000.0)),
        ("motif3.rope.scaling.factor", Val::F32(64.0)),
        ("motif3.rope.scaling.beta_fast", Val::F32(32.0)),
        ("motif3.rope.scaling.beta_slow", Val::F32(1.0)),
        ("motif3.rope.scaling.mscale", Val::F32(1.0)),
        ("motif3.mhc.h_post_coefficient", Val::F32(1.0)),
        ("motif3.polynorm.output_scale", Val::F32(0.5)),
        ("motif3.polynorm.bias_clamp", Val::F32(0.5)),
        ("motif3.hidden_clamp", Val::F32(1_000_000.0)),
        (
            "motif3.attention.sliding_window_pattern",
            Val::Str("interleave"),
        ),
        ("motif3.rope.scaling.type", Val::Str("yarn")),
        ("motif3.activation", Val::Str("poly_norm")),
        (
            "motif3.source.config_sha256",
            Val::Str("30f14b635d3258a18c3ff7e69829f8fbfa775e87477ffabb59a79115bba820a5"),
        ),
    ]
}

fn dots3_ok() -> Vec<(&'static str, Val<'static>)> {
    vec![
        ("general.architecture", Val::Str("dots3-note")),
        ("dots3-note.block_count", Val::U32(47)),
        ("dots3-note.context_length", Val::U64(524288)),
        ("dots3-note.embedding_length", Val::U32(5120)),
        ("dots3-note.vocab_size", Val::U32(152064)),
        ("dots3-note.feed_forward_length", Val::U32(13824)),
        ("dots3-note.leading_dense_block_count", Val::U32(1)),
        ("dots3-note.expert_count", Val::U32(256)),
        ("dots3-note.expert_used_count", Val::U32(8)),
        ("dots3-note.expert_feed_forward_length", Val::U32(1536)),
        ("dots3-note.expert_shared_count", Val::U32(1)),
        ("dots3-note.attention.head_count", Val::U32(128)),
        ("dots3-note.attention.head_count_kv", Val::U32(128)),
        ("dots3-note.attention.key_length", Val::U32(192)),
        ("dots3-note.attention.value_length", Val::U32(128)),
        ("dots3-note.sliding_window", Val::U32(513)),
        ("dots3-note.index_topk", Val::U32(2048)),
        ("dots3-note.q_lora_rank", Val::U32(1024)),
        ("dots3-note.kv_lora_rank", Val::U32(512)),
        ("dots3-note.swa_kv_lora_rank", Val::U32(1024)),
        ("dots3-note.full_attention_count", Val::U32(13)),
        ("dots3-note.language_only", Val::Bool(true)),
        ("dots3-note.mtp.present", Val::Bool(true)),
        ("dots3-note.rope.freq_base", Val::F32(80_000_000.0)),
        ("dots3-note.rope.freq_base_swa", Val::F32(50_000.0)),
        (
            "dots3-note.attention.layer_norm_rms_epsilon",
            Val::F32(1.0e-5),
        ),
        (
            "dots3-note.source.config_sha256",
            Val::Str("99b7de680dd456111c36efb8749f8ae7177328e97b65a3e39a6700cbc1173833"),
        ),
    ]
}

fn solar_ok(sched: &[u32]) -> Vec<(&str, Val<'_>)> {
    vec![
        ("general.architecture", Val::Str("solar-open2")),
        ("solar-open2.block_count", Val::U32(48)),
        ("solar-open2.context_length", Val::U64(1_048_576)),
        ("solar-open2.embedding_length", Val::U32(4096)),
        ("solar-open2.vocab_size", Val::U32(196608)),
        ("solar-open2.feed_forward_length", Val::U32(10240)),
        ("solar-open2.attention.head_count", Val::U32(64)),
        ("solar-open2.attention.key_length", Val::U32(128)),
        ("solar-open2.attention.value_length", Val::U32(128)),
        ("solar-open2.expert_count", Val::U32(320)),
        ("solar-open2.expert_used_count", Val::U32(8)),
        ("solar-open2.expert_feed_forward_length", Val::U32(1280)),
        ("solar-open2.expert_shared_count", Val::U32(1)),
        ("solar-open2.leading_dense_block_count", Val::U32(0)),
        ("solar-open2.ssm.conv_kernel", Val::U32(4)),
        ("solar-open2.kda.head_dim", Val::U32(128)),
        ("solar-open2.expert_gating_func", Val::U32(2)),
        (
            "solar-open2.attention.layer_norm_rms_epsilon",
            Val::F32(1.0e-5),
        ),
        ("solar-open2.expert_weights_scale", Val::F32(1.0)),
        ("solar-open2.expert_weights_norm", Val::Bool(true)),
        ("solar-open2.rope.freq_base", Val::F32(10000.0)),
        ("solar-open2.attention.head_count_kv", Val::ArrayU32(sched)),
    ]
}

fn exaone_ok(pattern: &[bool]) -> Vec<(&str, Val<'_>)> {
    vec![
        ("general.architecture", Val::Str("exaone-moe")),
        ("exaone-moe.block_count", Val::U32(49)),
        ("exaone-moe.context_length", Val::U64(262144)),
        ("exaone-moe.embedding_length", Val::U32(6144)),
        ("exaone-moe.vocab_size", Val::U32(153600)),
        ("exaone-moe.feed_forward_length", Val::U32(18432)),
        ("exaone-moe.attention.head_count", Val::U32(64)),
        ("exaone-moe.attention.head_count_kv", Val::U32(8)),
        ("exaone-moe.attention.key_length", Val::U32(128)),
        ("exaone-moe.attention.value_length", Val::U32(128)),
        ("exaone-moe.expert_count", Val::U32(128)),
        ("exaone-moe.expert_used_count", Val::U32(8)),
        ("exaone-moe.expert_feed_forward_length", Val::U32(2048)),
        (
            "exaone-moe.expert_shared_feed_forward_length",
            Val::U32(2048),
        ),
        ("exaone-moe.expert_shared_count", Val::U32(1)),
        ("exaone-moe.expert_group_count", Val::U32(1)),
        ("exaone-moe.expert_group_used_count", Val::U32(1)),
        ("exaone-moe.expert_gating_func", Val::U32(2)),
        ("exaone-moe.leading_dense_block_count", Val::U32(1)),
        ("exaone-moe.nextn_predict_layers", Val::U32(1)),
        ("exaone-moe.attention.sliding_window", Val::U32(128)),
        ("exaone-moe.rope.freq_base", Val::F32(1_000_000.0)),
        (
            "exaone-moe.attention.layer_norm_rms_epsilon",
            Val::F32(1.0e-5),
        ),
        ("exaone-moe.expert_weights_scale", Val::F32(2.5)),
        ("exaone-moe.expert_weights_norm", Val::Bool(true)),
        (
            "exaone-moe.attention.sliding_window_pattern",
            Val::ArrayBool(pattern),
        ),
    ]
}

fn solar_sched() -> Vec<u32> {
    (0..48).map(|il| if il % 4 == 0 { 8 } else { 0 }).collect()
}

fn exaone_pattern() -> Vec<bool> {
    (0..49).map(|il| (il % 4) != 3).collect()
}

#[test]
fn families_ok_match_c() {
    let compress: Vec<u32> = (0..43).map(flash_compress).collect();
    let swiglu = vec![10.0f32; 43];
    let flash = tmp("flash-ok.gguf");
    write_gguf(&flash, &flash_ok(&compress, &swiglu, None));
    assert_same(&flash);
    assert_eq!(ds4_core::dump_validate(&flash), "ok\n");

    let pro_c: Vec<u32> = (0..61).map(pro_compress).collect();
    let pro_s = vec![10.0f32; 61];
    let pro = tmp("pro-ok.gguf");
    write_gguf(&pro, &pro_ok(&pro_c, &pro_s));
    assert_same(&pro);

    let motif = tmp("motif-ok.gguf");
    write_gguf(&motif, &motif_ok());
    assert_same(&motif);

    let dots = tmp("dots3-ok.gguf");
    write_gguf(&dots, &dots3_ok());
    assert_same(&dots);

    let sched = solar_sched();
    let solar = tmp("solar-ok.gguf");
    write_gguf(&solar, &solar_ok(&sched));
    assert_same(&solar);

    let pat = exaone_pattern();
    let exa = tmp("exaone-ok.gguf");
    write_gguf(&exa, &exaone_ok(&pat));
    assert_same(&exa);
}

#[test]
fn error_tapes_match_c() {
    let miss = tmp("motif-miss.gguf");
    write_gguf(&miss, &[("general.architecture", Val::Str("motif3"))]);
    assert_same(&miss);
    assert_eq!(
        ds4_core::dump_validate(&miss),
        "missing-key motif3.block_count\n"
    );

    let glm = tmp("glm.gguf");
    write_gguf(&glm, &[("general.architecture", Val::Str("glm-dsa"))]);
    assert_same(&glm);
    assert_eq!(ds4_core::dump_validate(&glm), "unsupported-arch glm-dsa\n");

    let mut bad_select = Vec::new();
    for (k, n) in FLASH_DIMS {
        let n = if *k == "deepseek4.block_count" { 1 } else { *n };
        bad_select.push((*k, Val::U32(n)));
    }
    let unsupported = tmp("ds-unsupported.gguf");
    write_gguf(&unsupported, &bad_select);
    assert_same(&unsupported);
    assert_eq!(ds4_core::dump_validate(&unsupported), "unsupported\n");

    let compress: Vec<u32> = (0..43).map(flash_compress).collect();
    let swiglu = vec![10.0f32; 43];
    let mut kvs = flash_ok(&compress, &swiglu, Some("deepseek4"));
    if let Some((_, Val::F32(v))) = kvs
        .iter_mut()
        .find(|(k, _)| *k == "deepseek4.expert_weights_scale")
    {
        *v = 9.0;
    }
    let mismatch = tmp("flash-mismatch-f32.gguf");
    write_gguf(&mismatch, &kvs);
    assert_same(&mismatch);
    assert_eq!(
        ds4_core::dump_validate(&mismatch),
        "mismatch-f32 expert_weights_scale\n"
    );

    let mut groups = flash_ok(&compress, &swiglu, Some("deepseek4"));
    groups.push(("deepseek4.expert_group_count", Val::U32(1)));
    let mismatch_u32 = tmp("flash-mismatch.gguf");
    write_gguf(&mismatch_u32, &groups);
    assert_same(&mismatch_u32);
    assert_eq!(
        ds4_core::dump_validate(&mismatch_u32),
        "mismatch expert_group_count\n"
    );

    let mut no_freq = flash_ok(&compress, &swiglu, Some("deepseek4"));
    no_freq.retain(|(k, _)| *k != "deepseek4.rope.freq_base");
    let missing_f = tmp("flash-missing-f32.gguf");
    write_gguf(&missing_f, &no_freq);
    assert_same(&missing_f);
    assert_eq!(
        ds4_core::dump_validate(&missing_f),
        "missing-key deepseek4.rope.freq_base\n"
    );

    let mut bad_c = compress.clone();
    bad_c[2] = 99;
    let ratio = tmp("flash-compress.gguf");
    write_gguf(&ratio, &flash_ok(&bad_c, &swiglu, Some("deepseek4")));
    assert_same(&ratio);
    assert_eq!(ds4_core::dump_validate(&ratio), "compress-ratio 2\n");

    let short = tmp("flash-array-short.gguf");
    write_gguf(&short, &flash_ok(&[0, 0], &swiglu, Some("deepseek4")));
    assert_same(&short);
    assert_eq!(
        ds4_core::dump_validate(&short),
        "array-short deepseek4.attention.compress_ratios\n"
    );

    let mut type_kvs = flash_ok(&compress, &swiglu, Some("deepseek4"));
    if let Some((_, v)) = type_kvs
        .iter_mut()
        .find(|(k, _)| *k == "deepseek4.attention.compress_ratios")
    {
        *v = Val::ArrayF32(&swiglu);
    }
    let atype = tmp("flash-array-type.gguf");
    write_gguf(&atype, &type_kvs);
    assert_same(&atype);
    assert_eq!(
        ds4_core::dump_validate(&atype),
        "array-type deepseek4.attention.compress_ratios\n"
    );

    let mut sched = solar_sched();
    sched[0] = 0;
    let solar_bad = tmp("solar-schedule.gguf");
    write_gguf(&solar_bad, &solar_ok(&sched));
    assert_same(&solar_bad);
    assert_eq!(ds4_core::dump_validate(&solar_bad), "schedule 0\n");

    let mut pat = exaone_pattern();
    pat[3] = true;
    let swa = tmp("exaone-swa.gguf");
    write_gguf(&swa, &exaone_ok(&pat));
    assert_same(&swa);
    assert_eq!(ds4_core::dump_validate(&swa), "swa-pattern 3\n");

    let mut motif = motif_ok();
    if let Some((_, Val::Str(s))) = motif
        .iter_mut()
        .find(|(k, _)| *k == "motif3.source.config_sha256")
    {
        *s = "deadbeef";
    }
    let sha = tmp("motif-sha.gguf");
    write_gguf(&sha, &motif);
    assert_same(&sha);
    assert_eq!(
        ds4_core::dump_validate(&sha),
        "mismatch-string motif3.source.config_sha256\n"
    );

    let tiny = tmp("tiny.gguf");
    fs::write(&tiny, [0u8; 16]).unwrap();
    assert_same(&tiny);

    let bad_magic = tmp("bad-magic.gguf");
    fs::write(&bad_magic, [0u8; 32]).unwrap();
    assert_same(&bad_magic);
}

#[test]
fn host_compress_ratios_match_formula() {
    let flash = ds4_core::host_compress_ratios(&ds4_core::SHAPE_FLASH);
    assert_eq!(flash.len(), 43);
    for (il, got) in flash.iter().enumerate() {
        assert_eq!(*got, flash_compress(il as u32));
    }
    let motif = ds4_core::host_compress_ratios(&ds4_core::SHAPE_MOTIF3);
    assert!(motif.is_empty());
}
