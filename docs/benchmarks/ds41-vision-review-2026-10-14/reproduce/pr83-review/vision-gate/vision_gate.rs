//! One model open; serial, fresh native request states. Correctness timings
//! include logits readback and are not performance qualification.
use ds4_core::v41_media::{V41Images, V41VisionMeta};
use ds4_core::{Backend, Model, ModelOpenOption, V41RunOptions, V41Sampling};
use std::error::Error;
use std::fs::{self, File};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Instant;

type Result<T> = std::result::Result<T, Box<dyn Error>>;
const OUTPUTS: usize = 32;
const LONG_OUTPUTS: usize = 1024;
const VERIFY_K: i32 = 5;
const MEM_BUDGET_MB: i32 = 110000;
const IO_BYTES: usize = 1024 * 1024;
const LONG_PROMPT_MIN: usize = 2048;

#[derive(Clone, Copy, PartialEq)]
enum Overlay { Bare, Gr }
#[derive(Clone, Copy, PartialEq)]
enum Mode { Pure, Fixed5 }
#[derive(Clone, Copy, PartialEq)]
enum Graph { Eager, Captured }
#[derive(Clone, Copy, PartialEq)]
enum Cache { Disabled, Enabled }
#[derive(Clone, Copy, PartialEq)]
enum Reencode { Normal, Force }

struct Input { raw: Vec<i32>, images: Option<V41Images>, image_count: usize }
impl Input {
    fn tokens(&self) -> &[i32] {
        self.images.as_ref().map_or(&self.raw, |images| &images.tokens)
    }
}

struct Case<'a> {
    name: &'a str,
    outputs: usize,
    overlay: Overlay,
    mode: Mode,
    graph: Graph,
    cache: Cache,
    reencode: Reencode,
    reused: usize,
}
impl<'a> Case<'a> {
    fn medium(name: &'a str, outputs: usize, reused: usize) -> Self {
        Self { name, outputs, reused, overlay: Overlay::Gr, mode: Mode::Pure,
            graph: Graph::Captured, cache: Cache::Enabled, reencode: Reencode::Normal }
    }

    fn extended(name: &'a str, outputs: usize, overlay: Overlay, graph: Graph) -> Self {
        Self { name, graph, overlay, outputs, reused: 0, mode: Mode::Pure,
            cache: Cache::Disabled, reencode: Reencode::Normal }
    }
}

struct Outcome { name: String, ids: Vec<i32>, logits: PathBuf }
struct Gate<'a> {
    model: &'a Model,
    model_path: &'a Path,
    engram: &'a str,
    gr: &'a str,
    out: PathBuf,
    stats: File,
    comparisons: File,
    hashes: File,
}

fn sha(path: &Path) -> Result<String> {
    let output = Command::new("sha256sum").arg(path).output()?;
    if !output.status.success() { return Err(format!("sha256sum failed: {}", path.display()).into()); }
    let text = String::from_utf8(output.stdout)?;
    let digest = text.split_whitespace().next().ok_or("missing SHA256")?;
    if digest.len() != 64 || !digest.bytes().all(|c| c.is_ascii_hexdigit()) {
        return Err("invalid SHA256 receipt".into());
    }
    Ok(digest.to_owned())
}

fn write_ids(path: &Path, ids: &[i32]) -> Result<()> {
    let mut file = File::create(path)?;
    for id in ids { writeln!(file, "{id}")?; }
    Ok(())
}

fn inspect_logits(path: &Path, vocab: usize) -> Result<(u64, u64)> {
    let mut file = File::open(path)?;
    let bytes = file.metadata()?.len();
    let row_bytes = vocab as u64 * 4;
    if bytes == 0 || bytes % row_bytes != 0 { return Err("invalid logits extent".into()); }
    let mut buffer = vec![0u8; IO_BYTES];
    let mut nonfinite = 0;
    let mut remaining = bytes;
    while remaining > 0 {
        let count = remaining.min(buffer.len() as u64) as usize;
        file.read_exact(&mut buffer[..count])?;
        for value in buffer[..count].chunks_exact(4) {
            nonfinite += u64::from(!f32::from_le_bytes(value.try_into()?).is_finite());
        }
        remaining -= count as u64;
    }
    Ok((bytes / row_bytes, nonfinite))
}

fn same_logits(left: &Path, right: &Path, skip: u64) -> Result<()> {
    let mut a = File::open(left)?;
    let mut b = File::open(right)?;
    let bytes = a.metadata()?.len();
    if skip > bytes || bytes - skip != b.metadata()?.len() {
        return Err("logits extents differ".into());
    }
    a.seek(SeekFrom::Start(skip))?;
    let mut aa = vec![0u8; IO_BYTES];
    let mut bb = vec![0u8; IO_BYTES];
    let mut at = 0;
    while at < bytes - skip {
        let count = (bytes - skip - at).min(IO_BYTES as u64) as usize;
        a.read_exact(&mut aa[..count])?;
        b.read_exact(&mut bb[..count])?;
        if aa[..count] != bb[..count] {
            let index = aa[..count].iter().zip(&bb[..count]).position(|(x, y)| x != y).unwrap();
            return Err(format!("logits differ at compared byte {}", at + index as u64).into());
        }
        at += count as u64;
    }
    Ok(())
}

impl Gate<'_> {
    fn hash(&mut self, path: &Path) -> Result<()> {
        writeln!(self.hashes, "{}  {}", sha(path)?, path.display())?;
        self.hashes.flush()?;
        Ok(())
    }

    fn case(&mut self, input: &Input, case: Case<'_>) -> Result<Outcome> {
        let prompt = input.tokens();
        let vocab = self.model.vocab().n_vocab() as usize;
        let ctx = self.model.v41_ctx().ok_or("not ds41")? as usize;
        if prompt.is_empty() || prompt.len() + case.outputs + VERIFY_K as usize + 2 > ctx ||
            prompt.iter().any(|&t| t < 0 || t as usize >= vocab) {
            return Err(format!("{}: prompt/context precondition", case.name).into());
        }
        match case.reencode {
            Reencode::Normal => std::env::remove_var("DS41_VISION_REENCODE"),
            Reencode::Force => std::env::set_var("DS41_VISION_REENCODE", "1"),
        }
        let mount = if case.overlay == Overlay::Gr { "gr" } else { "bare" };
        let mode = if case.mode == Mode::Pure { "pure" } else { "fixed5" };
        let graph = if case.graph == Graph::Captured { "captured" } else { "eager" };
        let reencode = if case.reencode == Reencode::Force { "force" } else { "normal" };
        let mut options = V41RunOptions {
            engram_dir: Some(self.engram), no_engram: false,
            zchain: if case.overlay == Overlay::Gr { Some(self.gr) } else { None },
            dspark: Some(if case.mode == Mode::Pure { 0 } else { 2 }),
            verify_k: if case.mode == Mode::Fixed5 { VERIFY_K } else { 0 },
            graph: Some(case.graph == Graph::Captured),
            kv_reuse: Some(case.cache == Cache::Enabled),
            sampling: Some(V41Sampling::default()), images: input.images.as_ref(),
            emit_trace: case.mode == Mode::Fixed5, ..V41RunOptions::default()
        };
        let mut ids = Vec::new();
        let mut emit = |token| {
            ids.push(token);
            token >= 0 && (token as usize) < vocab && token != self.model.token_eos() && ids.len() < case.outputs
        };
        let mut progress = |_: &str, _: i32, _: i32| true;
        eprintln!("@@ BEGIN {} prompt={} requested={} overlay={mount} mode={mode} graph={graph} reencode={reencode}",
            case.name, prompt.len(), case.outputs);
        let begin = Instant::now();
        let result = self.model.v41_generate(self.model_path, prompt, case.outputs as i32,
            &mut options, &mut emit, &mut progress);
        let wall = begin.elapsed().as_secs_f64();
        let reused = ds4_core::v41_kvreuse_last_reused();
        let (rounds, offered, accepted) = ds4_core::v41_last_spec_stats();
        let mut problems = Vec::new();
        if let Err(error) = result { problems.push(error.to_string()); }
        if ids.len() != case.outputs || ids.iter().any(|&t| t < 0 || t as usize >= vocab) {
            problems.push(format!("incomplete/invalid emissions {}/{} (EOS={})", ids.len(), case.outputs, ids.contains(&self.model.token_eos())));
        }
        if reused != case.reused as i32 { problems.push(format!("reuse {reused}/{}", case.reused)); }
        if accepted < 0 || accepted > offered { problems.push("invalid acceptance counters".into()); }
        if case.mode == Mode::Pure && (rounds != 0 || offered != 0 || accepted != 0) {
            problems.push("pure run used speculation".into());
        }
        if case.mode == Mode::Fixed5 && (rounds <= 0 || i64::from(offered) != i64::from(rounds) * i64::from(VERIFY_K)) {
            problems.push(format!("fixed5 inactive or wrong widths: {rounds}/{offered}"));
        }
        let logits = self.out.join(format!("{}.f32", case.name));
        let current = self.out.join("current.f32");
        let mut rows = 0;
        let mut nonfinite = 0;
        if current.is_file() {
            fs::rename(&current, &logits)?;
            match inspect_logits(&logits, vocab) {
                Ok((n, bad)) => { rows = n; nonfinite = bad; }
                Err(error) => problems.push(error.to_string()),
            }
            self.hash(&logits)?;
        } else { problems.push("missing logits dump".into()); }
        if nonfinite != 0 { problems.push(format!("{nonfinite} nonfinite logits")); }
        if case.mode == Mode::Pure && rows != case.outputs as u64 {
            problems.push(format!("incomplete full-logit rows {rows}/{}", case.outputs));
        }
        for (suffix, tokens) in [("raw.ids", input.raw.as_slice()), ("prompt.ids", prompt), ("tokens.ids", ids.as_slice())] {
            let path = self.out.join(format!("{}.{}", case.name, suffix));
            write_ids(&path, tokens)?;
            self.hash(&path)?;
        }
        let text_path = self.out.join(format!("{}.text", case.name));
        let mut text = File::create(&text_path)?;
        for &token in &ids {
            if token >= 0 && (token as usize) < vocab && token != self.model.token_eos() {
                text.write_all(&self.model.token_text(token)?)?;
            }
        }
        drop(text);
        self.hash(&text_path)?;
        let (images, image_rows) = input.images.as_ref().map_or((0, 0), |i| (input.image_count, i.n_rows));
        let scope = if case.mode == Mode::Pure { "full" } else { "partial_single_decisions" };
        let status = if problems.is_empty() { "PASS" } else { "FAIL" };
        writeln!(self.stats, "{},{mount},{mode},{graph},{},{reencode},{},{images},{image_rows},{},{},{reused},{},{vocab},{rows},{scope},{nonfinite},{rounds},{offered},{accepted},{wall:.6},{status}",
            case.name, if case.cache == Cache::Enabled { "on" } else { "off" }, prompt.len(), case.outputs, ids.len(), case.reused)?;
        self.stats.flush()?;
        eprintln!("@@ END {} status={status} emitted={} reused={reused} logits_rows={rows} wall_s={wall:.6}", case.name, ids.len());
        if !problems.is_empty() {
            fs::write(self.out.join(format!("{}.error.txt", case.name)), problems.join("\n"))?;
            return Err(format!("{}: {}", case.name, problems.join("; ")).into());
        }
        Ok(Outcome { name: case.name.to_owned(), ids, logits })
    }

    fn compare(&mut self, left: &Outcome, right: &Outcome, skip: usize, mode: Mode) -> Result<()> {
        let mut result = if left.ids.get(skip..) == Some(right.ids.as_slice()) { Ok(()) }
            else { Err("token streams differ".into()) };
        if result.is_ok() && mode == Mode::Pure {
            result = same_logits(&left.logits, &right.logits, skip as u64 * self.model.vocab().n_vocab() as u64 * 4);
        }
        let status = if result.is_ok() { "PASS" } else { "FAIL" };
        writeln!(self.comparisons, "{},{},{skip},{},{status}", left.name, right.name,
            if mode == Mode::Pure { "tokens_and_all_logits" } else { "tokens_only" })?;
        self.comparisons.flush()?;
        eprintln!("@@ COMPARE {} {} {status}", left.name, right.name);
        result
    }

    fn seed(&mut self, input: &Input, prefix: &str) -> Result<()> {
        // A one-entry checkpoint has no public clear API. A different leading
        // raw token replaces it before the real prompt is saved cold.
        let vocab = self.model.vocab().n_vocab();
        let mut token = (input.tokens()[0] + 1) % vocab;
        if token == self.model.token_eos() { token = (token + 1) % vocab; }
        let flush = Input { raw: vec![token], images: None, image_count: 0 };
        self.case(&flush, Case::medium(&format!("{prefix}-flush"), 1, 0))?;
        self.case(input, Case::medium(&format!("{prefix}-seed"), 1, 0))?;
        Ok(())
    }
}

fn tokenized(model: &Model, text: &str) -> Result<Vec<i32>> {
    Ok(model.tokenize_text(text)?.as_slice().to_vec())
}

fn image_input(raw: Vec<i32>, bytes: &[&[u8]], meta: &V41VisionMeta, context: usize) -> Result<Input> {
    let images = V41Images::expand(&raw, bytes, meta, context)?;
    Ok(Input { raw, images: Some(images), image_count: bytes.len() })
}

fn medium(gate: &mut Gate<'_>, meta: &V41VisionMeta, a: &[u8], b: &[u8]) -> Result<Input> {
    let ctx = gate.model.v41_ctx().ok_or("not ds41")? as usize;
    let mut raw = tokenized(gate.model, "Study this image carefully: ")?;
    raw.push(meta.image_token_id);
    raw.extend(tokenized(gate.model, "\nWrite a detailed 2000-word essay about its colors, geometry, composition, and visual interpretation. Start the essay now and continue without summarizing.\n")?);
    let base = image_input(raw.clone(), &[a], meta, ctx)?;
    let changed = image_input(raw.clone(), &[b], meta, ctx)?;
    if base.tokens() != changed.tokens() {
        return Err("identity fixtures must have equal token layouts and geometry".into());
    }
    let p = base.tokens().len();
    let cold = gate.case(&base, Case::medium("cold-p", OUTPUTS, 0))?;
    let full = gate.case(&base, Case::medium("full-hit-p", OUTPUTS, p))?;
    gate.compare(&cold, &full, 0, Mode::Pure)?;
    let mut suffix_raw = raw.clone();
    suffix_raw.push(cold.ids[0]);
    let suffix = image_input(suffix_raw, &[a], meta, ctx)?;
    let suffix_run = gate.case(&suffix, Case::medium("warm-suffix-one", OUTPUTS - 1, p))?;
    gate.compare(&cold, &suffix_run, 1, Mode::Pure)?;

    gate.seed(&base, "changed")?;
    let miss = gate.case(&changed, Case::medium("changed-image-miss", OUTPUTS, 0))?;
    let mut reference = Case::medium("changed-image-cold", OUTPUTS, 0);
    reference.cache = Cache::Disabled;
    let reference = gate.case(&changed, reference)?;
    gate.compare(&miss, &reference, 0, Mode::Pure)?;

    let mut follow_raw = raw.clone();
    follow_raw.extend_from_slice(&cold.ids);
    follow_raw.extend(tokenized(gate.model, "\nContinue the visual analysis with more detail.\n")?);
    let follow = image_input(follow_raw.clone(), &[a], meta, ctx)?;
    follow_raw.extend(tokenized(gate.model, "\nNow also compare this second image: ")?);
    follow_raw.push(meta.image_token_id);
    follow_raw.extend(tokenized(gate.model, "\nContinue the long essay comparing both images.\n")?);
    let appended = image_input(follow_raw, &[a, b], meta, ctx)?;
    for (label, input) in [("follow-text", &follow), ("append-image", &appended)] {
        if !input.tokens().starts_with(base.tokens()) { return Err("extension lost the base prefix".into()); }
        gate.seed(&base, &format!("{label}-forced"))?;
        let forced_name = format!("{label}-reencode");
        let mut forced = Case::medium(&forced_name, OUTPUTS, p);
        forced.reencode = Reencode::Force;
        let off = gate.case(input, forced)?;
        gate.seed(&base, &format!("{label}-normal"))?;
        let on = gate.case(input, Case::medium(&format!("{label}-reuse"), OUTPUTS, p))?;
        gate.compare(&off, &on, 0, Mode::Pure)?;
    }

    let mut pair_raw = tokenized(gate.model, "Compare these images in a detailed 2000-word essay. First: ")?;
    pair_raw.push(meta.image_token_id);
    pair_raw.extend(tokenized(gate.model, " Second: ")?);
    pair_raw.push(meta.image_token_id);
    pair_raw.extend(tokenized(gate.model, "\nBegin the essay and continue at length.\n")?);
    let ab = image_input(pair_raw.clone(), &[a, b], meta, ctx)?;
    let ba = image_input(pair_raw, &[b, a], meta, ctx)?;
    if ab.tokens() != ba.tokens() { return Err("reorder case changed token geometry".into()); }
    gate.seed(&ab, "reorder")?;
    let reordered = gate.case(&ba, Case::medium("reordered-images-miss", OUTPUTS, 0))?;
    let mut ref_case = Case::medium("reordered-images-cold", OUTPUTS, 0);
    ref_case.cache = Cache::Disabled;
    let reference = gate.case(&ba, ref_case)?;
    gate.compare(&reordered, &reference, 0, Mode::Pure)?;
    Ok(base)
}

fn extended(gate: &mut Gate<'_>, input: &Input, outputs: usize, mtp: Mode) -> Result<()> {
    for overlay in [Overlay::Bare, Overlay::Gr] {
        let mount = if overlay == Overlay::Bare { "bare" } else { "gr" };
        let eager_name = format!("extended-{mount}-pure-eager");
        let captured_name = format!("extended-{mount}-pure-captured");
        let eager = gate.case(input, Case::extended(&eager_name, outputs, overlay, Graph::Eager))?;
        let captured = gate.case(input, Case::extended(&captured_name, outputs, overlay, Graph::Captured))?;
        gate.compare(&eager, &captured, 0, Mode::Pure)?;
        if mtp == Mode::Fixed5 {
            for (graph, reference) in [(Graph::Eager, &eager), (Graph::Captured, &captured)] {
                let capture = if graph == Graph::Eager { "eager" } else { "captured" };
                let name = format!("extended-{mount}-fixed5-{capture}");
                let mut case = Case::extended(&name, outputs, overlay, graph);
                case.mode = Mode::Fixed5;
                let spec = gate.case(input, case)?;
                gate.compare(reference, &spec, 0, Mode::Fixed5)?;
            }
        }
    }
    Ok(())
}

fn run() -> Result<()> {
    if !cfg!(target_endian = "little") { return Err("the dump contract requires little-endian f32".into()); }
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.len() < 7 {
        return Err("usage: vision_gate MODEL VISION ENGRAM GR GOLDENS INPUTS OUT [--long ESSAY_IDS] [--mtp]".into());
    }
    let mut long = None;
    let mut mtp = Mode::Pure;
    let mut index = 7;
    while index < args.len() {
        match args[index].as_str() {
            "--mtp" => { mtp = Mode::Fixed5; index += 1; }
            "--long" if index + 1 < args.len() => { long = Some(PathBuf::from(&args[index + 1])); index += 2; }
            other => return Err(format!("unknown/incomplete option {other}").into()),
        }
    }
    for key in ["DS41_FWD_TRACE", "DS41_DUMP_HEAD", "DS41_FEED_DIGEST", "DS41_VISION_REENCODE", "DS41_NO_ENGRAM"] {
        if std::env::var_os(key).is_some() { return Err(format!("unset conflicting diagnostic {key}").into()); }
    }
    let out = PathBuf::from(&args[6]);
    if out.exists() && fs::read_dir(&out)?.next().is_some() { return Err("OUT must be fresh or empty".into()); }
    fs::create_dir_all(&out)?;
    let out = fs::canonicalize(out)?;
    let golden_dir = fs::canonicalize(&args[4])?;
    let input_dir = fs::canonicalize(&args[5])?;
    let a = fs::read(input_dir.join("image-a.png"))?;
    let b = fs::read(input_dir.join("image-b.png"))?;
    if a == b { return Err("identity images have identical bytes".into()); }
    let meta = V41VisionMeta::read(&args[1])?;
    std::env::set_var("DS41_GEN_LOGITS", out.join("current.f32"));
    std::env::set_var("DS41_ROLLBACK_CHECK", "1");
    ds4_core::set_mem_budget_mb(MEM_BUDGET_MB);
    fs::write(out.join("config.tsv"), format!("model\t{}\nvision\t{}\nengram\t{}\ngr\t{}\nmem_budget_mb\t{MEM_BUDGET_MB}\nmodel_opens\t1\nsampling\tpure-temperature-0\nengram_enabled\ttrue\nclock_policy_mhz\t300-2200; unchanged\nlong\t{}\nmtp\t{}\nfixed5_logits\tpartial single-decision rows; not full verify logits\n",
        args[0], args[1], args[2], args[3], long.as_ref().map_or(String::from("off"), |p| p.display().to_string()),
        if mtp == Mode::Fixed5 { "fixed5" } else { "off" }))?;
    let model = Model::open_configured(&args[0], Backend::Cuda, 0, true, None,
        &[ModelOpenOption::Vision(args[1].clone())])?;
    let mut gate = Gate { model: &model, model_path: Path::new(&args[0]), engram: &args[2], gr: &args[3],
        out: out.clone(), stats: File::create(out.join("stats.csv"))?,
        comparisons: File::create(out.join("comparisons.csv"))?, hashes: File::create(out.join("SHA256SUMS"))? };
    writeln!(gate.stats, "name,overlay,mode,graph,cache,reencode,prompt,images,image_rows,requested,emitted,reused,expected_reused,vocab,logit_rows,logit_scope,nonfinite,rounds,offered,accepted,wall_s,status")?;
    writeln!(gate.comparisons, "left,right,skip_left_rows,scope,status")?;
    let mut goldens = File::create(out.join("goldens.csv"))?;
    writeln!(goldens, "name,status,wall_s")?;
    for (name, image, golden) in [("panel", "review-market-panel.png", "golden_panel.bin"),
        ("bug", "bug.jpeg", "golden_bug.bin"), ("homer", "homer.png", "golden_homer.bin")] {
        let image_path = golden_dir.join(image);
        let golden_path = golden_dir.join(golden);
        gate.hash(&image_path)?;
        gate.hash(&golden_path)?;
        eprintln!("@@ GOLDEN BEGIN {name}");
        let begin = Instant::now();
        let passed = model.v41_vision_golden(&meta, &image_path, &golden_path)?;
        writeln!(goldens, "{name},{},{:.6}", if passed { "PASS" } else { "FAIL" }, begin.elapsed().as_secs_f64())?;
        goldens.flush()?;
        eprintln!("@@ GOLDEN END {name} status={}", if passed { "PASS" } else { "FAIL" });
        if !passed { return Err(format!("{name} golden failed").into()); }
    }
    gate.hash(&input_dir.join("image-a.png"))?;
    gate.hash(&input_dir.join("image-b.png"))?;
    let base = medium(&mut gate, &meta, &a, &b)?;
    if let Some(path) = long {
        let ids = fs::read_to_string(&path)?.split_whitespace().map(str::parse::<i32>).collect::<std::result::Result<Vec<_>, _>>()?;
        if ids.len() < LONG_PROMPT_MIN { return Err("--long requires an essay prompt of at least 2048 raw tokens".into()); }
        let mut raw = vec![meta.image_token_id];
        raw.extend(ids);
        let input = image_input(raw, &[&a], &meta, model.v41_ctx().ok_or("not ds41")? as usize)?;
        gate.hash(&path)?;
        extended(&mut gate, &input, LONG_OUTPUTS, mtp)?;
    } else if mtp == Mode::Fixed5 { extended(&mut gate, &base, OUTPUTS, mtp)?; }
    drop(goldens);
    for name in ["config.tsv", "stats.csv", "comparisons.csv", "goldens.csv"] { gate.hash(&out.join(name))?; }
    gate.hash(&std::env::current_exe()?)?;
    std::env::remove_var("DS41_VISION_REENCODE");
    fs::write(out.join("COMPLETE"), "Driver checks passed. Run audit.py against the native log for graph/rollback evidence.\n")?;
    Ok(())
}

fn main() {
    if let Err(error) = run() { eprintln!("vision_gate: FAIL: {error}"); std::process::exit(1); }
}
