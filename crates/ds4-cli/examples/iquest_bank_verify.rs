//! Exact bank-state gates and separate prefill-width diagnostics on real weights.
use ds4_core::{
    parse_prefix, snapshot_mem, Backend, BatchCtx, ContAdmit, ContDone, ContDriver, Model,
    ModelFamily, ModelOpenOption, HEADER_BYTES,
};
use serde_json::{json, Value};
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

type GateResult<T> = Result<T, Box<dyn std::error::Error>>;
const IO_CHUNK: usize = 1024 * 1024;
const DEFAULT_CTX: i32 = 8192;
const DEFAULT_CHUNK: usize = 128;
const MAIN_WINDOW: usize = 4096;
const HELP: &str = "usage: iquest_bank_verify MODEL TOKENS.i32 OUT --cut N [--serial-from N] [--ctx N] [--chunk N]\n\
TOKENS is an explicit little-endian i32 prompt. --cut must precede its end.\n\
Source prefills to --serial-from (default --cut), teacher-forces individual\n\
rows to --cut, then prefills the suffix. Two banks share an existing VMM owner.\n\
Copy/rewind/disk/matched-schedule payloads must be byte-identical. A separate\n\
cold prefill reports width-dependent logits; it is not a quality acceptance.\n";

struct Options {
    model: String,
    tokens: PathBuf,
    out: PathBuf,
    cut: usize,
    serial_from: usize,
    ctx: i32,
    chunk: usize,
}

fn options(args: &[String]) -> GateResult<Options> {
    if args.len() < 5 || (args.len() - 3) % 2 != 0 {
        return Err(HELP.into());
    }
    let mut cut = None;
    let mut serial = None;
    let mut ctx = DEFAULT_CTX;
    let mut chunk = DEFAULT_CHUNK;
    for pair in args[3..].chunks_exact(2) {
        match pair[0].as_str() {
            "--cut" => cut = Some(pair[1].parse()?),
            "--serial-from" => serial = Some(pair[1].parse()?),
            "--ctx" => ctx = pair[1].parse()?,
            "--chunk" => chunk = pair[1].parse()?,
            other => return Err(format!("unknown argument {other}").into()),
        }
    }
    let cut = cut.ok_or("--cut is required")?;
    let serial_from = serial.unwrap_or(cut);
    if !(2..=524_288).contains(&ctx)
        || cut == 0
        || cut >= ctx as usize
        || serial_from == 0
        || serial_from > cut
        || chunk == 0
        || chunk > 8192
    {
        return Err("invalid context, cut, serial boundary or chunk".into());
    }
    Ok(Options {
        model: args[0].clone(),
        tokens: args[1].clone().into(),
        out: args[2].clone().into(),
        cut,
        serial_from,
        ctx,
        chunk,
    })
}

#[derive(Default)]
struct Request {
    pending: Option<ContAdmit>,
    admitted: Option<(i32, i32, i32)>,
    output: Option<Vec<i32>>,
}

impl ContDriver for Request {
    fn admit(&mut self) -> Option<ContAdmit> {
        self.pending.take()
    }
    fn on_token(&mut self, _: usize, _: i32) -> bool {
        true
    }
    fn on_admitted(&mut self, _: usize, cached: i32, computed: i32, bank: i32) -> bool {
        self.admitted = Some((cached, computed, bank));
        true
    }
    fn on_done(&mut self, _: usize, tokens: &[i32], _: i32, _: ContDone) {
        self.output = Some(tokens.to_vec());
    }
}

// A one-token budget samples frontier logits without evaluating that token.
// Every evaluated row is therefore supplied explicitly by this diagnostic.
fn prefill(
    batch: &BatchCtx<'_>,
    tokens: &[i32],
    bank: i32,
    source: Option<i32>,
    cached: usize,
) -> GateResult<i32> {
    let mut admit = ContAdmit::cold(0, tokens.to_vec(), 1);
    admit.place_bank = bank + 1;
    admit.fork_bank = source.map_or(0, |id| id + 1);
    admit.n_cached = cached.try_into()?;
    let mut request = Request {
        pending: Some(admit),
        ..Default::default()
    };
    batch.continuous_generate(&mut request)?;
    if request.admitted != Some((cached as i32, (tokens.len() - cached) as i32, bank)) {
        return Err(format!("wrong admission: {:?}", request.admitted).into());
    }
    if batch.bank_snapshot(bank)?.tokens != tokens {
        return Err("prefill changed the requested token frontier".into());
    }
    match request.output.as_deref() {
        Some([token]) => Ok(*token),
        other => Err(format!("expected one sampled token, got {other:?}").into()),
    }
}

fn compare_files(a: &Path, b: &Path) -> GateResult<u64> {
    let mut a = File::open(a)?;
    let mut b = File::open(b)?;
    let length = a.metadata()?.len();
    if b.metadata()?.len() != length {
        return Err("payload lengths differ".into());
    }
    let mut left = vec![0; IO_CHUNK];
    let mut right = vec![0; IO_CHUNK];
    for offset in (0..length).step_by(IO_CHUNK) {
        let count = (length - offset).min(IO_CHUNK as u64) as usize;
        a.read_exact(&mut left[..count])?;
        b.read_exact(&mut right[..count])?;
        if let Some(index) = left[..count]
            .iter()
            .zip(&right[..count])
            .position(|(a, b)| a != b)
        {
            return Err(format!("payload differs at byte {}", offset + index as u64).into());
        }
    }
    Ok(length)
}

fn save_equal(batch: &BatchCtx<'_>, bank: i32, actual: &Path, expected: &Path) -> GateResult<u64> {
    batch.save_bank_payload(bank, actual)?;
    compare_files(actual, expected)
}

fn logits(path: &Path, tokens: &[i32], chunk: usize) -> GateResult<Vec<f32>> {
    let mut file = File::open(path)?;
    let mut header = vec![0; HEADER_BYTES + tokens.len() * 4];
    file.read_exact(&mut header)?;
    let prefix = parse_prefix(&header)?;
    if prefix.fields[3] as usize != chunk
        || prefix.fields[8] != 1
        || prefix.fields[7] as usize != tokens.len()
        || !prefix
            .tokens
            .iter()
            .zip(tokens)
            .all(|(a, b)| *a == *b as u32)
    {
        return Err("unexpected native payload shape or MTP mode".into());
    }
    let vocab = prefix.fields[11] as usize;
    file.seek(SeekFrom::Start(prefix.prefix_len() as u64))?;
    let mut raw = vec![0; vocab * 4];
    file.read_exact(&mut raw)?;
    let result: Vec<_> = raw
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect();
    if result.is_empty()
        || result.iter().any(|x| !x.is_finite())
        || result.iter().all(|x| *x == 0.0)
    {
        return Err("empty, nonfinite or identically zero logits".into());
    }
    Ok(result)
}

fn logit_delta(a: &[f32], b: &[f32], eos: usize) -> GateResult<Value> {
    if a.len() != b.len() || eos >= a.len() {
        return Err("incompatible logits".into());
    }
    let top = |values: &[f32]| {
        let mut ids: Vec<_> = (0..values.len()).collect();
        ids.sort_unstable_by(|a, b| values[*b].total_cmp(&values[*a]).then(a.cmp(b)));
        ids[..ids.len().min(5)]
            .iter()
            .map(|id| json!({"id":id,"logit":values[*id]}))
            .collect::<Vec<_>>()
    };
    let mut max = 0.0f64;
    let mut square = 0.0f64;
    let mut changed = 0;
    for (a, b) in a.iter().zip(b) {
        let delta = f64::from(*a) - f64::from(*b);
        max = max.max(delta.abs());
        square += delta * delta;
        changed += usize::from(a.to_bits() != b.to_bits());
    }
    Ok(json!({"changed":changed,"vocab":a.len(),"max_abs":max,
        "rms":(square / a.len() as f64).sqrt(),"source_top":top(a),"cold_top":top(b),
        "source_eos":a[eos],"cold_eos":b[eos]}))
}

fn run(o: Options) -> GateResult<()> {
    // Refuse an independent model residency. The owner remains alive externally.
    let manifest = std::env::var("DS4_CUDA_WEIGHT_IPC_MANIFEST")?;
    if manifest.is_empty() || std::env::var_os("DS4_CUDA_COPY_MODEL").is_some() {
        return Err("requires an existing VMM owner and no DS4_CUDA_COPY_MODEL".into());
    }
    let scope = std::env::var("DS4_CUDA_WEIGHT_IPC_SCOPE").unwrap_or_default();
    if !matches!(scope.as_str(), "" | "base" | "both") {
        return Err("IPC scope must include base weights".into());
    }
    let raw = std::fs::read(&o.tokens)?;
    if raw.len() % 4 != 0 {
        return Err("unaligned token file".into());
    }
    let tokens: Vec<_> = raw
        .chunks_exact(4)
        .map(|b| i32::from_le_bytes(b.try_into().unwrap()))
        .collect();
    if tokens.len() <= o.cut || tokens.len() + 2 >= o.ctx as usize {
        return Err("cut must precede prompt end with room for continuation".into());
    }
    std::fs::create_dir_all(&o.out)?;
    let initial_memory = snapshot_mem();
    let model = Model::open_with_support_options(
        &o.model,
        Backend::Cuda,
        0,
        true,
        None,
        None,
        &[
            ModelOpenOption::MtpDraftTokens(3),
            ModelOpenOption::MtpMargin(0.0),
        ],
    )?;
    if model.family() != ModelFamily::IQuestQ1
        || tokens
            .iter()
            .any(|t| *t < 0 || *t >= model.vocab().n_vocab())
    {
        return Err("requires real IQuest-Q1 weights and valid tokens".into());
    }
    let batch = model.batch_ctx_fit(o.ctx, 2, o.ctx)?;
    if batch.max_seq() != 2 || !batch.supports_partial_reuse() {
        return Err("requires exactly two fitted banks with partial reuse".into());
    }
    let prefix = o.out.join("prefix.kv");
    let source = o.out.join("source.kv");
    let compare = o.out.join("compare.kv");
    let cold = o.out.join("cold.kv");
    prefill(&batch, &tokens[..o.serial_from], 0, None, 0)?;
    for end in o.serial_from + 1..=o.cut {
        prefill(&batch, &tokens[..end], 0, None, end - 1)?;
    }
    batch.save_bank_payload(0, &prefix)?;
    let first = prefill(&batch, &tokens, 0, None, o.cut)?;
    batch.save_bank_payload(0, &source)?;
    let source_snapshot = batch.bank_snapshot(0)?;
    let source_logits = logits(&source, &tokens, o.chunk)?;

    // Full copy and wrapped partial copy must preserve source bytes and lineage.
    prefill(&batch, &tokens, 1, Some(0), tokens.len())?;
    let payload_bytes = save_equal(&batch, 1, &compare, &source)?;
    save_equal(&batch, 0, &compare, &source)?;
    prefill(&batch, &tokens[..o.cut], 1, Some(0), o.cut)?;
    let prefix_bytes = save_equal(&batch, 1, &compare, &prefix)?;
    save_equal(&batch, 0, &compare, &source)?;
    if batch.bank_snapshot(0)? != source_snapshot {
        return Err("fork changed source lineage".into());
    }

    // Replaying the same suffix width isolates restore correctness from MMQ order.
    let replay = prefill(&batch, &tokens, 1, None, o.cut)?;
    save_equal(&batch, 1, &compare, &source)?;
    if first != replay {
        return Err("matched-schedule sampled token differs".into());
    }
    prefill(&batch, &tokens[..o.cut], 1, Some(1), o.cut)?;
    save_equal(&batch, 1, &compare, &prefix)?;
    prefill(&batch, &tokens, 1, None, o.cut)?;
    save_equal(&batch, 1, &compare, &source)?;

    let restored = batch.load_bank_payload_range(1, &prefix, 0, prefix_bytes)?;
    if restored.tokens != tokens[..o.cut] {
        return Err("disk restore frontier differs".into());
    }
    save_equal(&batch, 1, &compare, &prefix)?;
    prefill(&batch, &tokens, 1, None, o.cut)?;
    save_equal(&batch, 1, &compare, &source)?;
    eprintln!(
        "exact bank gates passed: cut={} end={} bytes={payload_bytes}",
        o.cut,
        tokens.len()
    );

    // Cold chunk boundaries intentionally differ. Record their effect without
    // calling a changed output acceptable or turning a numeric tolerance into a gate.
    let cold_first = prefill(&batch, &tokens, 1, None, 0)?;
    batch.save_bank_payload(1, &cold)?;
    let cold_logits = logits(&cold, &tokens, o.chunk)?;
    let eos = usize::try_from(model.token_eos())?;
    let at_prompt = logit_delta(&source_logits, &cold_logits, eos)?;
    let mut next_tokens = tokens.clone();
    next_tokens.push(first);
    let source_next = prefill(&batch, &next_tokens, 0, None, tokens.len())?;
    batch.save_bank_payload(0, &compare)?;
    let source_next_logits = logits(&compare, &next_tokens, o.chunk)?;
    let cold_next = prefill(&batch, &next_tokens, 1, None, tokens.len())?;
    batch.save_bank_payload(1, &compare)?;
    let cold_next_logits = logits(&compare, &next_tokens, o.chunk)?;
    let final_memory = snapshot_mem();
    if final_memory.census.faults != initial_memory.census.faults {
        return Err("memory census faults increased".into());
    }
    let report = json!({"model":o.model,"tokens_file":o.tokens,"manifest":manifest,
        "ctx":o.ctx,"chunk":o.chunk,"banks":batch.max_seq(),"cut":o.cut,
        "serial_from":o.serial_from,"prompt_tokens":tokens.len(),"mtp":true,
        "target_ring":(MAIN_WINDOW + o.chunk - 1).min(o.ctx as usize),
        "checkpoint_after_physical_wrap":o.cut > (MAIN_WINDOW + o.chunk - 1).min(o.ctx as usize),
        "payload_bytes":payload_bytes,"prefix_bytes":prefix_bytes,
        "exact":{"source_preserved":true,"full_fork":true,"partial_fork":true,
        "matched_schedule":true,"in_place_rewind":true,"disk_restore":true},
        "cross_width":{"source_first":first,"cold_first":cold_first,"at_prompt":at_prompt,
        "forced_first":first,"source_next":source_next,"cold_next":cold_next,
        "after_first":logit_delta(&source_next_logits,&cold_next_logits,eos)?},
        "memory_faults_unchanged":true,"cross_width_quality_qualified":false});
    std::fs::write(
        o.out.join("report.json"),
        serde_json::to_vec_pretty(&report)?,
    )?;
    println!("{}", serde_json::to_string_pretty(&report)?);
    Ok(())
}

fn main() -> GateResult<()> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    if args
        .iter()
        .any(|arg| matches!(arg.as_str(), "--help" | "-h"))
    {
        print!("{HELP}");
        return Ok(());
    }
    run(options(&args)?)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validates_schedule() {
        let args = ["m", "t", "o", "--cut", "4240", "--serial-from", "4216"].map(String::from);
        let parsed = options(&args).unwrap();
        assert_eq!((parsed.cut, parsed.serial_from), (4240, 4216));
        let bad = ["m", "t", "o", "--cut", "4240", "--serial-from", "4241"].map(String::from);
        assert!(options(&bad).is_err());
    }

    #[test]
    fn logs_eos_flip_without_tolerance() {
        let delta = logit_delta(&[1.0, 1.125], &[1.125, 1.0], 1).unwrap();
        assert_eq!(delta["changed"], 2);
        assert_eq!(delta["source_top"][0]["id"], 1);
        assert_eq!(delta["cold_top"][0]["id"], 0);
        assert_eq!(delta["max_abs"], 0.125);
    }

    #[test]
    fn compares_entire_payload() {
        let dir = std::env::temp_dir().join(format!("iquest-bank-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let a = dir.join("a");
        let b = dir.join("b");
        let mut bytes = vec![7; IO_CHUNK + 17];
        std::fs::write(&a, &bytes).unwrap();
        std::fs::write(&b, &bytes).unwrap();
        assert_eq!(compare_files(&a, &b).unwrap(), bytes.len() as u64);
        bytes[IO_CHUNK + 16] = 8;
        std::fs::write(&b, &bytes).unwrap();
        assert!(compare_files(&a, &b)
            .unwrap_err()
            .to_string()
            .contains("1048592"));
        std::fs::remove_dir_all(dir).unwrap();
    }
}
