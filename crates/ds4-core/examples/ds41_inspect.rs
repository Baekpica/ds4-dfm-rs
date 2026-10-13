//! Metadata-only inventory of a DeepSeek V4.1 artifact; never loads weights.
//!
//!   cargo run -p ds4-core --example ds41_inspect -- <model.gguf> [--tensors]
//!
//! Prints the header's identifying keys, the per-type tensor census, and the
//! V4.1-specific keys the engine reads (core_validate_v41.c:46-154). This is
//! the P2/P3 check of docs/deepseek41-port-plan.md: the Rust host must see the
//! same artifact the C engine sees before any forward work starts.
use ds4_core::{tensor_type_name, GgufFile, TensorInventory};
use std::collections::BTreeMap;
use std::path::Path;

/// The keys the engine reads for V4.1 (core_validate_v41.c), plus the plain
/// identity keys and the dims `select_shape_from_metadata` matches on. A
/// missing one here is a loader gap, not a curiosity.
const V41_KEYS: &[&str] = &[
    "general.architecture",
    "general.name",
    // dims: what ds4_select_shape_from_metadata compares (core_shape_select.c:166)
    "deepseek4.block_count",
    "deepseek4.embedding_length",
    "deepseek4.vocab_size",
    "deepseek4.attention.head_count",
    "deepseek4.attention.head_count_kv",
    "deepseek4.attention.key_length",
    "deepseek4.attention.value_length",
    "deepseek4.rope.dimension_count",
    "deepseek4.attention.q_lora_rank",
    "deepseek4.attention.output_lora_rank",
    "deepseek4.attention.output_group_count",
    "deepseek4.expert_count",
    "deepseek4.expert_used_count",
    "deepseek4.expert_feed_forward_length",
    "deepseek4.expert_shared_count",
    "deepseek4.hash_layer_count",
    "deepseek4.attention.sliding_window",
    "deepseek4.attention.indexer.head_count",
    "deepseek4.attention.indexer.key_length",
    "deepseek4.attention.indexer.top_k",
    "deepseek4.hyper_connection.count",
    "deepseek4.hyper_connection.sinkhorn_iterations",
    // V4.1 additions (core_validate_v41.c:46-154)
    "deepseek4.context_length",
    "deepseek4.attention.kv_source_layers",
    "deepseek4.attention.index_source_layers",
    "deepseek4.attention.candidate.source_layer",
    "deepseek4.attention.candidate.topk_blocks",
    "deepseek4.attention.candidate.block_size",
    "deepseek4.engram.layer_ids",
    "deepseek4.engram.num_embeddings",
    "deepseek4.engram.max_ngram_size",
    "deepseek4.mtp.tower_count",
    "deepseek4.mtp.expert_count",
    "deepseek4.mtp.block_size",
    "deepseek4.mtp.expert_used_count",
    "deepseek4.mtp.noise_token_id",
    "deepseek4.mtp.markov_rank",
    "deepseek4.mtp.target_layers",
];

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let path = args
        .next()
        .ok_or("usage: ds41_inspect <model.gguf> [--tensors] [--engram-dir <dir>] [--ids <file>] [--erows-out <prefix>]")?;
    let mut list_tensors = false;
    let mut engram_dir: Option<String> = None;
    let mut ids_path: Option<String> = None;
    let mut erows_out: Option<String> = None;
    let mut zchain: Option<String> = None;
    let mut posttrain: Option<String> = None;
    let mut zchain_scale = 1.0f32;
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--tensors" => list_tensors = true,
            "--engram-dir" => engram_dir = args.next(),
            "--ids" => ids_path = args.next(),
            "--erows-out" => erows_out = args.next(),
            "--zchain" => zchain = args.next(),
            "--posttrain" => posttrain = args.next(),
            "--zchain-scale" => {
                zchain_scale = args.next().ok_or("--zchain-scale needs a value")?.parse()?
            }
            other => return Err(format!("unknown argument {other}").into()),
        }
    }
    let p = Path::new(&path);

    let g = GgufFile::open(p)?;
    let inv = TensorInventory::open(p)?;

    println!(
        "file: {} kv={} tensors={} data_pos={} alignment={}",
        path,
        g.kv_entries().len(),
        inv.tensors.len(),
        inv.data_pos,
        inv.alignment
    );

    // Per-type census: the types above 30 are this family's, so a census
    // shows at a glance whether 40-44 are parsed.
    let mut census: BTreeMap<u32, (usize, u64)> = BTreeMap::new();
    for t in &inv.tensors {
        let e = census.entry(t.typ).or_insert((0, 0));
        e.0 += 1;
        e.1 += t.bytes;
    }
    println!("types:");
    for (typ, (count, bytes)) in &census {
        println!(
            "  {typ:>3} {:<10} tensors={count} bytes={bytes}",
            tensor_type_name(*typ)
        );
    }

    println!("keys:");
    for key in V41_KEYS {
        let seen = match g.get_string(key) {
            Some(v) => format!("str {:?}", String::from_utf8_lossy(v)),
            None => match g.get_u32(key) {
                Some(v) => format!("u32 {v}"),
                None => match g.get_u64_compat(key) {
                    Some(v) => format!("u64 {v}"),
                    None => match g.get_array(key) {
                        Some(arr) => format!("array len={}", arr.len),
                        None => "ABSENT".to_string(),
                    },
                },
            },
        };
        println!("  {key} = {seen}");
    }

    if list_tensors {
        println!("tensors:");
        for t in &inv.tensors {
            println!("  {}", t.dump_line());
        }
    }

    // Identification and validation: the two steps that decide whether this
    // host can accept the artifact at all.
    match ds4_core::identify_file(&g) {
        Ok(id) => {
            println!("identify: {}", id.identify_line());
            match ds4_core::validate_file(&g, &id.shape) {
                Ok(()) => println!("validate: ok"),
                Err(e) => println!("validate: {}", e.token()),
            }
            // Bind: every required name must find a tensor, and the plan
            // should consume the published inventory rather than leave it.
            // V4.1 resolves through the metadata wire (source layers, engram
            // layers, tower form), not from the shape alone; its layout is
            // checked from the same wire.
            let mut wire: Option<ds4_core::V41Wire> = None;
            let plan = if id.shape.variant == ds4_core::Variant::DeepSeek41Flash {
                match ds4_core::V41Wire::load(&g, &id.shape) {
                    Ok(mut w) => {
                        if let Some(dir) = engram_dir.as_deref() {
                            w.apply_engram_dir(dir);
                        }
                        println!(
                            "v41 wire: kv-sources={:?} index-sources={:?} engram={:?} \
                             towers={} experts={} targets={:?} markov-rank={:?}",
                            w.is_kv_source
                                .iter()
                                .enumerate()
                                .filter(|(_, &b)| b)
                                .map(|(i, _)| i)
                                .collect::<Vec<_>>(),
                            w.is_index_source
                                .iter()
                                .enumerate()
                                .filter(|(_, &b)| b)
                                .map(|(i, _)| i)
                                .collect::<Vec<_>>(),
                            w.engram_layers,
                            w.mtp_towers,
                            w.mtp_experts,
                            w.mtp_targets,
                            w.mtp_markov_rank,
                        );
                        println!("v41 compress-ratios: {:?}", w.compress_ratios);
                        let plan = ds4_core::BindPlan::resolve_v41(id.shape, &w, &inv);
                        match ds4_core::validate_layouts_v41(&plan, &w, &inv) {
                            Ok(()) => println!("layout: ok"),
                            Err(e) => println!("layout: {}", e.token()),
                        }
                        wire = Some(w);
                        plan
                    }
                    Err(e) => {
                        println!("v41 wire failed: {}", e.token());
                        ds4_core::BindPlan::resolve_catalog(None, id.shape, &inv)
                    }
                }
            } else {
                ds4_core::BindPlan::resolve_catalog(None, id.shape, &inv)
            };
            let missing: Vec<&str> = plan
                .slots
                .iter()
                .filter(|s| s.need == ds4_core::BindNeed::Required && s.tensor.is_none())
                .map(|s| s.name.as_str())
                .collect();
            let bound = plan.slots.iter().filter(|s| s.tensor.is_some()).count();
            println!(
                "bind: slots={} bound={} required-missing={}",
                plan.slots.len(),
                bound,
                missing.len()
            );
            for name in missing.iter().take(12) {
                println!("  missing: {name}");
            }

            // Engram: open the shards in place (--engram-dir), check the row
            // span and read row 0; with --ids, recompute the hash rows for
            // every position so they can be diffed against the engine's own
            // golden capture (--erows-out <prefix> writes
            // <prefix>.erows_L<nn>.txt, the golden file naming).
            if let Some(w) = &wire {
                for (ei, path) in w.engram_table_path.iter().enumerate() {
                    let shard = match ds4_core::EngramShard::open(Path::new(path)) {
                        Ok(s) => s,
                        Err(e) => {
                            println!("engram shard {ei}: {}", e.token());
                            continue;
                        }
                    };
                    let head_dim = u64::from(w.engram_head_dim);
                    if let Err(e) = shard.check_span(
                        w.engram_weight_off[ei],
                        w.engram_scale_off[ei],
                        w.engram_rows[ei],
                        head_dim,
                    ) {
                        println!("engram shard {ei}: span {}", e.token());
                        continue;
                    }
                    match shard.row(
                        w.engram_weight_off[ei],
                        w.engram_scale_off[ei],
                        0,
                        w.engram_head_dim,
                    ) {
                        Ok(row) => println!(
                            "engram shard {ei}: size={} direct={} rows={} row0 w[0..8]={:02x?} scale={:02x?}",
                            shard.size(),
                            shard.is_direct(),
                            w.engram_rows[ei],
                            &row.weights[..8],
                            row.scale
                        ),
                        Err(e) => println!("engram shard {ei}: row0 {}", e.token()),
                    }
                }
                if let (Some(ids_path), Some(prefix)) = (ids_path.as_deref(), erows_out.as_deref())
                {
                    let ids: Vec<i32> = std::fs::read_to_string(ids_path)?
                        .split_whitespace()
                        .map(|v| v.parse().unwrap())
                        .collect();
                    match ds4_core::EngramHash::load(&g, &inv, w, id.shape.n_vocab) {
                        Ok(Some(h)) => {
                            for (ei, layer) in w.engram_layers.iter().enumerate() {
                                let file = format!("{prefix}.erows_L{layer:02}.txt");
                                let mut out = String::new();
                                for p in 0..ids.len() {
                                    let rows = h.rows(&ids, p as i64, ei as u32);
                                    let line: Vec<String> =
                                        rows.iter().map(|r| r.to_string()).collect();
                                    out.push_str(&line.join(" "));
                                    out.push('\n');
                                }
                                std::fs::write(&file, out)?;
                                println!("erows: wrote {file}");
                            }
                        }
                        Ok(None) => println!("erows: no engram layers"),
                        Err(e) => println!("erows: {}", e.token()),
                    }
                }
            }

            // Sidecars: the ②/③ directories are merged the way the engine
            // merges them at load (V41Zchain::load: base.fnv gate, rb from ②
            // only, gains multiplied, amp ranks concatenated), then the ②
            // directory's fingerprint is recomputed for the base.fnv
            // evidence. Both are admission checks, not weights.
            let (n_expert, n_embd, n_layer) =
                (id.shape.n_expert, id.shape.n_embd, id.shape.n_layer);
            if zchain.is_some() || posttrain.is_some() {
                let geom = ds4_core::ZchainGeom {
                    n_layer,
                    n_expert,
                    n_embd,
                };
                match ds4_core::V41Zchain::load(
                    zchain.as_deref().map(Path::new),
                    posttrain.as_deref().map(Path::new),
                    geom,
                    zchain_scale,
                ) {
                    Ok(Some(z)) => {
                        let base = match &z.base {
                            ds4_core::BaseFingerprint::Absent => "absent".to_string(),
                            ds4_core::BaseFingerprint::Checked { hash, files } => {
                                format!("ok {hash:016x}/{files}")
                            }
                        };
                        let k = match z.k_range() {
                            Some((lo, hi)) => format!("{lo}..{hi}"),
                            None => "none".to_string(),
                        };
                        println!(
                            "zchain-merge: gr={} rb={} amp={} k={k} base={base} beta={}",
                            z.n_gr_layers(),
                            z.n_rb_layers(),
                            z.n_amp_layers(),
                            z.beta
                        );
                        // The layer both directories carry (the product), with
                        // factor bits for a bit-exact external comparison.
                        if let Some((il, g)) = z.gr.iter().enumerate().find_map(|(il, g)| {
                            g.as_ref().filter(|g| g.from2 && g.from3).map(|g| (il, g))
                        }) {
                            let f = &g.factor;
                            let nz = f.iter().take(4096).filter(|&&v| v != 1.0).count();
                            println!(
                                "zchain-merge: merged L{il:02} f[0]={:#010x} f[100]={:#010x} f[1000]={:#010x} f[4095]={:#010x} nonzero={nz}/4096",
                                f[0].to_bits(),
                                f[100].to_bits(),
                                f[1000].to_bits(),
                                f[4095].to_bits()
                            );
                        }
                        if let Some((il, g)) =
                            z.gr.iter()
                                .enumerate()
                                .find_map(|(il, g)| g.as_ref().map(|g| (il, g)))
                        {
                            println!(
                                "zchain-merge: first gr L{il:02} from2={} from3={} f[0]={:.9}",
                                g.from2, g.from3, g.factor[0]
                            );
                        }
                        if let Some((il, a)) = z
                            .amp
                            .iter()
                            .enumerate()
                            .find_map(|(il, a)| a.as_ref().map(|a| (il, a)))
                        {
                            println!(
                                "zchain-merge: first amp L{il:02} k2={} k3={} typ={}/{}",
                                a.k2, a.k3, a.typ2, a.typ3
                            );
                        }
                    }
                    Ok(None) => println!("zchain-merge: no directories"),
                    Err(e) => println!("zchain-merge: {}", e.token()),
                }
            }
            if let Some(zdir) = zchain.as_deref() {
                let (hash, files) = ds4_core::gr_dir_fnv(Path::new(zdir), n_layer);
                println!("zchain: fnv={hash:016x}/{files}");
            }
            if let Some(pdir) = posttrain.as_deref() {
                let dir = Path::new(pdir);
                match ds4_core::check_base_fnv(
                    dir,
                    zchain.as_deref().map(Path::new),
                    n_layer,
                ) {
                    Ok(ds4_core::BaseFingerprint::Absent) => {
                        println!("posttrain: base.fnv ABSENT (warn and pass)")
                    }
                    Ok(ds4_core::BaseFingerprint::Checked { hash, files }) => {
                        println!("posttrain: base.fnv ok {hash:016x}/{files}")
                    }
                    Err(ds4_core::SidecarError::BaseMismatch {
                        want,
                        want_files,
                        have,
                        have_files,
                    }) => println!(
                        "posttrain: base.fnv MISMATCH want {want:016x}/{want_files} have {have:016x}/{have_files}"
                    ),
                    Err(e) => println!("posttrain: {}", e.token()),
                }
                match ds4_core::GrSidecar::read(&dir.join("gr_L39.bin"), n_expert, n_embd) {
                    Ok(Some(gr)) => println!(
                        "posttrain: gr_L39 type={} factor[0]={:.6}",
                        gr.typ, gr.factor[0]
                    ),
                    Ok(None) => println!("posttrain: no gr_L39.bin"),
                    Err(e) => println!("posttrain: gr_L39 {}", e.token()),
                }
            }
        }
        Err(e) => println!("identify failed: {e}"),
    }
    Ok(())
}
