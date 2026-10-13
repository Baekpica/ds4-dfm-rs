//! V4.1 engram gates, model-free.
//!
//! The hash is compared against the engine's own `v41_engram_hash`
//! (fixtures/engram/gen_engram_ref.c, compiled and run to produce
//! engram_ref.txt), which is the same function the engine ran to capture the
//! golden set's `erows_L01/L14.txt`; the same rows were also checked against
//! that capture first-hand on 2026-10-09 (port plan §6.6).
//!
//! The shard reader is checked against a synthetic two-plane file: row r is
//! `head_dim` weight bytes at `weight_off + r*head_dim` and `head_dim/32`
//! scale bytes at `scale_off + r*(head_dim/32)` (core_v41_engram.c:236-242).

use std::fs;
use std::path::PathBuf;

use ds4_core::{EngramHash, EngramShard};

fn fixture_text() -> &'static str {
    include_str!("fixtures/engram/engram_ref.txt")
}

fn load_hash() -> EngramHash {
    let mut h = EngramHash {
        token_map: Vec::new(),
        multipliers: Vec::new(),
        primes: Vec::new(),
        offsets: Vec::new(),
        max_ngram: 0,
        heads: 0,
        pad: 0,
        n_vocab: 0,
    };
    for line in fixture_text().lines() {
        let mut it = line.split_whitespace();
        match it.next().unwrap() {
            "NGRAM" => h.max_ngram = it.next().unwrap().parse().unwrap(),
            "HEADS" => h.heads = it.next().unwrap().parse().unwrap(),
            "PAD" => h.pad = it.next().unwrap().parse().unwrap(),
            "NVOCAB" => h.n_vocab = it.next().unwrap().parse().unwrap(),
            "TMAPI32" => {
                let n: usize = it.next().unwrap().parse().unwrap();
                h.token_map = it.take(n).map(|v| v.parse().unwrap()).collect();
            }
            "MULT" => {
                let _n: usize = it.next().unwrap().parse().unwrap();
                h.multipliers = it.map(|v| v.parse().unwrap()).collect();
            }
            "PRIM" => {
                let _n: usize = it.next().unwrap().parse().unwrap();
                h.primes = it.map(|v| v.parse().unwrap()).collect();
            }
            "OFFS" => {
                let _n: usize = it.next().unwrap().parse().unwrap();
                h.offsets = it.map(|v| v.parse().unwrap()).collect();
            }
            "HIST" | "ROW" => {}
            other => panic!("unknown fixture line {other}"),
        }
    }
    h
}

fn history() -> Vec<i32> {
    for line in fixture_text().lines() {
        if let Some(rest) = line.strip_prefix("HIST ") {
            let mut it = rest.split_whitespace();
            let _n: usize = it.next().unwrap().parse().unwrap();
            return it.map(|v| v.parse().unwrap()).collect();
        }
    }
    panic!("no HIST line");
}

#[test]
fn hash_matches_the_engine_on_every_position_and_layer() {
    let h = load_hash();
    let hist = history();
    let mut checked = 0;
    for line in fixture_text().lines() {
        let Some(rest) = line.strip_prefix("ROW ") else {
            continue;
        };
        let mut it = rest.split_whitespace();
        let ei: u32 = it.next().unwrap().parse().unwrap();
        let p: i64 = it.next().unwrap().parse().unwrap();
        let want: Vec<i64> = it.map(|v| v.parse().unwrap()).collect();
        let got = h.rows(&hist, p, ei);
        assert_eq!(got, want, "row mismatch at ei={ei} p={p}");
        checked += 1;
    }
    assert_eq!(checked, 24);
}

/// The hash on the artifact's real constants must reproduce the engine's own
/// captured rows: `p1.erows_L01/L14.txt` are what the engine printed for
/// prompt p1 (`golden/` on the Spark), one line per position, 24 row ids
/// each. This is what pins the flat layer-major indexing of the multipliers
/// and primes: reading them in their GGUF dims order agrees with nothing.
#[test]
fn hash_reproduces_the_golden_prompt_rows() {
    let h = EngramHash {
        token_map: include_bytes!("fixtures/engram/token_map.i32.bin")
            .chunks_exact(4)
            .map(|b| i32::from_le_bytes(b.try_into().unwrap()))
            .collect(),
        multipliers: include_bytes!("fixtures/engram/multipliers.i64.bin")
            .chunks_exact(8)
            .map(|b| i64::from_le_bytes(b.try_into().unwrap()))
            .collect(),
        primes: include_bytes!("fixtures/engram/primes.i64.bin")
            .chunks_exact(8)
            .map(|b| i64::from_le_bytes(b.try_into().unwrap()))
            .collect(),
        offsets: include_bytes!("fixtures/engram/offsets.i64.bin")
            .chunks_exact(8)
            .map(|b| i64::from_le_bytes(b.try_into().unwrap()))
            .collect(),
        max_ngram: 4,
        heads: 8,
        pad: 2,
        n_vocab: 129_280,
    };
    let ids: Vec<i32> = include_str!("fixtures/engram/p1.ids.txt")
        .split_whitespace()
        .map(|v| v.parse().unwrap())
        .collect();
    assert_eq!(ids.len(), 8);
    for (ei, golden) in [
        (0u32, include_str!("fixtures/engram/p1.erows_L01.txt")),
        (1u32, include_str!("fixtures/engram/p1.erows_L14.txt")),
    ] {
        let lines: Vec<&str> = golden.lines().collect();
        assert_eq!(lines.len(), ids.len(), "one row line per position");
        for (p, line) in lines.iter().enumerate() {
            let want: Vec<i64> = line
                .split_whitespace()
                .map(|v| v.parse().unwrap())
                .collect();
            assert_eq!(
                h.rows(&ids, p as i64, ei),
                want,
                "golden rows ei={ei} p={p}"
            );
        }
    }
}

fn tmp(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join("ds4-engram");
    fs::create_dir_all(&dir).unwrap();
    dir.join(name)
}

#[test]
fn shard_reads_the_two_row_planes() {
    // 4 rows, head_dim 256: weight plane at 4096, scale plane right after,
    // plus one alignment block of slack - an O_DIRECT read of the last row
    // rounds up to the block and needs those bytes to exist, which the real
    // shards always have (the plane ends before the file does; the engine's
    // own `v41_edio_pread` demands the full aligned span the same way).
    const ROWS: u64 = 4;
    const HD: u32 = 256;
    const NSC: u64 = 8;
    const W_OFF: u64 = 4096;
    let s_off = W_OFF + ROWS * HD as u64;
    let mut bytes = vec![0u8; (s_off + ROWS * NSC) as usize + 4096];
    for r in 0..ROWS {
        let w = W_OFF + r * HD as u64;
        for i in 0..HD as u64 {
            bytes[(w + i) as usize] = (r * 37 + i) as u8;
        }
        let s = s_off + r * NSC;
        for i in 0..NSC {
            bytes[(s + i) as usize] = 0x80 | (r as u8 + i as u8);
        }
    }
    let path = tmp("shard.bin");
    fs::write(&path, &bytes).unwrap();
    let shard = EngramShard::open(&path).expect("open shard");
    shard
        .check_span(W_OFF, s_off, ROWS, HD as u64)
        .expect("span");
    let row = shard.row(W_OFF, s_off, 2, HD).expect("row 2");
    assert_eq!(row.weights.len(), HD as usize);
    assert_eq!(row.scale.len(), NSC as usize);
    assert_eq!(row.weights[0], (2 * 37) as u8);
    assert_eq!(row.weights[255], (2 * 37 + 255) as u8);
    assert_eq!(
        row.scale,
        vec![
            0x80 | 2,
            0x80 | 3,
            0x80 | 4,
            0x80 | 5,
            0x80 | 6,
            0x80 | 7,
            0x80 | 8,
            0x80 | 9
        ]
    );
    // Well past the end is refused, exactly like the engine's span check.
    assert!(shard
        .check_span(W_OFF, s_off, ROWS + 32, HD as u64)
        .is_err());
}
