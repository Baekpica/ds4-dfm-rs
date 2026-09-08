use crate::{
    cli,
    experiment::{self, Scout},
    fit,
};
use ds4_perf::artifact::{self, Artifact, Payload, Reference};
use serde::{Deserialize, Serialize};
use std::{collections::BTreeMap, fs, path::Path};

#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Delta {
    pub metric: String,
    pub ctx: u64,
    pub unit: String,
    pub baseline: Vec<f64>,
    pub candidate: Vec<f64>,
    pub baseline_median: f64,
    pub candidate_median: f64,
    pub median_percent: f64,
    pub lower_percent: f64,
    pub upper_percent: f64,
}

#[derive(Debug, Default, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LogitError {
    pub max_abs: f64,
    pub max_scaled: f64,
    pub bad: u64,
    pub checked: u64,
}

#[derive(Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Verdict {
    Pass,
    Improved,
    Regressed,
    Inconclusive,
    Incorrect,
    Incomparable,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Comparison {
    pub verdict: Verdict,
    pub metrics: Vec<Delta>,
    pub correctness: LogitError,
    /// Number of proof comparisons with a greedy sequence or argmax mismatch.
    pub token_mismatches: u64,
    pub max_slowdown_percent: f64,
    pub logit_atol: f64,
    pub logit_rtol: f64,
    pub changes: BTreeMap<String, String>,
    pub reasons: Vec<String>,
    pub method: String,
}

impl Payload for Comparison {
    const KIND: &'static str = "compare";
    fn validate(&self) -> Result<(), String> {
        if self.method != "sample-extrema-envelope-v1" {
            return Err("unsupported comparison method".into());
        }
        if [
            self.max_slowdown_percent,
            self.logit_atol,
            self.logit_rtol,
            self.correctness.max_abs,
            self.correctness.max_scaled,
        ]
        .into_iter()
        .any(|v| !v.is_finite() || v < 0.0)
        {
            return Err("invalid comparison policy or result".into());
        }
        if self.verdict == Verdict::Incomparable {
            if self.reasons.is_empty() || !self.metrics.is_empty() {
                return Err("incomparable result lacks a reason".into());
            }
            return Ok(());
        }
        if self.metrics.is_empty()
            || self.correctness.checked == 0
            || self.correctness.bad > self.correctness.checked
        {
            return Err("conclusive comparison lacks timing/correctness evidence".into());
        }
        let expected = verdict(
            &self.metrics,
            &self.correctness,
            self.token_mismatches,
            self.max_slowdown_percent,
        );
        if self.verdict != expected {
            return Err("comparison verdict disagrees with evidence".into());
        }
        for d in &self.metrics {
            if [
                d.baseline_median,
                d.candidate_median,
                d.median_percent,
                d.lower_percent,
                d.upper_percent,
            ]
            .into_iter()
            .any(|v| !v.is_finite())
                || d.baseline
                    .iter()
                    .chain(&d.candidate)
                    .any(|v| !v.is_finite() || *v <= 0.0)
            {
                return Err("invalid comparison samples".into());
            }
            let computed = delta(&d.metric, d.ctx, &d.baseline, &d.candidate)?;
            if d.baseline.len() < 3
                || d.baseline.len() != d.candidate.len()
                || d.unit != computed.unit
                || [
                    d.baseline_median,
                    d.candidate_median,
                    d.median_percent,
                    d.lower_percent,
                    d.upper_percent,
                ] != [
                    computed.baseline_median,
                    computed.candidate_median,
                    computed.median_percent,
                    computed.lower_percent,
                    computed.upper_percent,
                ]
            {
                return Err("comparison summary disagrees with raw samples".into());
            }
        }
        Ok(())
    }
}

fn delta(metric: &str, ctx: u64, a: &[f64], b: &[f64]) -> Result<Delta, String> {
    if a.is_empty() || b.is_empty() || a.iter().chain(b).any(|v| !v.is_finite() || *v <= 0.0) {
        return Err("nonpositive or missing timing samples".into());
    }
    let baseline_median = fit::median(a);
    let candidate_median = fit::median(b);
    let extrema = |v: &[f64]| {
        (
            v.iter().copied().fold(f64::INFINITY, f64::min),
            v.iter().copied().fold(0.0, f64::max),
        )
    };
    let (amin, amax) = extrema(a);
    let (bmin, bmax) = extrema(b);
    Ok(Delta {
        metric: metric.into(),
        ctx,
        unit: if metric == "first_token" {
            "seconds"
        } else {
            "seconds/token"
        }
        .into(),
        baseline: a.into(),
        candidate: b.into(),
        baseline_median,
        candidate_median,
        median_percent: 100.0 * (candidate_median / baseline_median - 1.0),
        lower_percent: 100.0 * (bmin / amax - 1.0),
        upper_percent: 100.0 * (bmax / amin - 1.0),
    })
}

fn logit_error(a: &[f64], b: &[f64], atol: f64, rtol: f64) -> Result<LogitError, String> {
    if a.is_empty() || a.len() != b.len() || a.iter().chain(b).any(|v| !v.is_finite()) {
        return Err("invalid full-vocabulary logits".into());
    }
    let mut error = LogitError::default();
    for (a, b) in a.iter().zip(b) {
        let difference = (a - b).abs();
        let tolerance = atol + rtol * a.abs();
        error.max_abs = error.max_abs.max(difference);
        // A zero tolerance is valid for an exact parity contract.
        error.max_scaled = error
            .max_scaled
            .max((difference / tolerance.max(f64::MIN_POSITIVE)).min(f64::MAX));
        error.bad += u64::from(difference > tolerance);
        error.checked += 1;
    }
    Ok(error)
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Frontier {
    source: String,
    model: String,
    backend: String,
    quality: bool,
    quant_bits: u64,
    prompt_tokens: u64,
    frontier_tokens: u64,
    prefill_tokens: u64,
    ctx: u64,
    vocab: usize,
    argmax_id: usize,
    argmax_logit: f64,
    logits: Vec<f64>,
}

fn proof_file<T: serde::de::DeserializeOwned>(
    root: &Path,
    reference: &Reference,
) -> Result<T, String> {
    let path = root.join(&reference.path);
    if artifact::hash(&path)? != reference.sha256 {
        return Err("correctness evidence changed".into());
    }
    let file = fs::File::open(&path).map_err(|e| e.to_string())?;
    if file.metadata().map_err(|e| e.to_string())?.len() > 64 * 1024 * 1024 {
        return Err("proof file exceeds 64 MiB".into());
    }
    serde_json::from_reader(file).map_err(|e| format!("{}: {e}", path.display()))
}

type FrontierProofs = BTreeMap<u64, Vec<(Frontier, Vec<i32>)>>;

fn proofs(root: &Path, scout: &Scout) -> Result<FrontierProofs, String> {
    let mut proofs: BTreeMap<_, Vec<_>> = BTreeMap::new();
    for sample in &scout.samples {
        for row in &sample.rows {
            let name = format!("frontier_{:06}.logits.json", row.ctx);
            let frontier: Frontier = proof_file(
                root,
                sample
                    .proof_files
                    .get(&name)
                    .ok_or("missing full-vocabulary frontier; use scout --proof")?,
            )?;
            let name = format!("tokens-{}.json", row.ctx);
            let tokens: Vec<i32> = proof_file(
                root,
                sample
                    .proof_files
                    .get(&name)
                    .ok_or("missing greedy-token evidence; use scout --proof")?,
            )?;
            if frontier.source != "ds4-bench"
                || frontier.backend != "cuda"
                || frontier.model.is_empty()
                || frontier.vocab == 0
                || frontier.logits.len() != frontier.vocab
                || frontier.argmax_id >= frontier.vocab
                || frontier.frontier_tokens != row.ctx
                || frontier.prompt_tokens != row.ctx
                || frontier.prefill_tokens != row.prefill_tokens
                || frontier.ctx <= row.ctx
                || tokens.len() as u64 != row.gen_tokens
                || tokens
                    .iter()
                    .any(|t| *t < 0 || *t as usize >= frontier.vocab)
                || frontier.logits.iter().any(|v| !v.is_finite())
                || !frontier.argmax_logit.is_finite()
                || frontier.logits[frontier.argmax_id] != frontier.argmax_logit
                || frontier.logits.iter().any(|v| *v > frontier.argmax_logit)
            {
                return Err("frontier/token proof disagrees with benchmark row".into());
            }
            if scout.workload.is_some()
                && scout.model_argument.as_deref() != Some(Path::new(&frontier.model))
            {
                return Err(
                    "proof model differs from the captured and hashed workload argument".into(),
                );
            }
            proofs.entry(row.ctx).or_default().push((frontier, tokens));
        }
    }
    Ok(proofs)
}

pub fn validate_proofs(root: &Path, scout: &Scout) -> Result<(), String> {
    if scout.samples.len() != scout.requested_repeats as usize
        || scout.samples.iter().any(|s| s.rows.is_empty())
    {
        return Err("missing samples for requested proof".into());
    }
    proofs(root, scout).map(|_| ())
}

fn identity(a: &Scout, b: &Scout) -> Result<BTreeMap<String, String>, String> {
    if !a.process_scope.verified
        || !b.process_scope.verified
        || a.process_scope.expected_owners != b.process_scope.expected_owners
    {
        return Err("GPU process snapshots do not match the intended owners for both runs".into());
    }
    if !a.unreviewed_environment.is_empty() || !b.unreviewed_environment.is_empty() {
        return Err("unreviewed DS4/CUDA/LD environment controls prevent matched comparison; see scout.json".into());
    }
    let (Some(ga), Some(gb)) = (&a.gpu, &b.gpu) else {
        return Err("comparison requires verified CUDA device identity".into());
    };
    if !ga.same_device(gb) {
        return Err("device or driver differs".into());
    }
    if a.command[1..] != b.command[1..] || a.cwd != b.cwd {
        return Err("benchmark arguments or working directory differ".into());
    }
    if a.cache_policy != "warmup-then-fresh" || a.cache_policy != b.cache_policy {
        return Err("comparison requires matching explicit --cache-policy".into());
    }
    let (Some(wa), Some(wb)) = (&a.workload, &b.workload) else {
        return Err(
            "comparison requires --workload with hashed model/prompt and cache contract".into(),
        );
    };
    let hashes = |w: &experiment::Workload| {
        w.files
            .iter()
            .map(|(k, v)| (k.clone(), v.sha256.clone()))
            .collect::<BTreeMap<_, _>>()
    };
    if wa.name != wb.name
        || wa.family != wb.family
        || hashes(wa) != hashes(wb)
        || wa.shape != wb.shape
        || wa.cache_state != wb.cache_state
    {
        return Err("workload model/prompt/shape/cache identity differs".into());
    }
    let mut changes = BTreeMap::new();
    if a.executable_sha256 != b.executable_sha256 {
        changes.insert(
            "executable_sha256".into(),
            format!("{} -> {}", a.executable_sha256, b.executable_sha256),
        );
    }
    let keys: std::collections::BTreeSet<_> =
        a.environment.keys().chain(b.environment.keys()).collect();
    for key in keys {
        if a.environment.get(key) == b.environment.get(key) {
            continue;
        }
        if !crate::knobs::tunable(key) {
            return Err(format!("non-tuning runtime/memory policy changed: {key}"));
        }
        for value in [a.environment.get(key), b.environment.get(key)]
            .into_iter()
            .flatten()
        {
            crate::knobs::validate(key, value, &wa.family)?;
        }
        changes.insert(
            key.clone(),
            format!(
                "{:?} -> {:?}",
                a.environment.get(key),
                b.environment.get(key)
            ),
        );
    }
    Ok(changes)
}

fn analyze(
    a: &Scout,
    ar: &Path,
    b: &Scout,
    br: &Path,
    args: &cli::Compare,
) -> Result<Comparison, String> {
    let changes = identity(a, b)?;
    if a.requested_repeats != b.requested_repeats
        || a.samples.len() != a.requested_repeats as usize
        || b.samples.len() != b.requested_repeats as usize
    {
        return Err("comparison requires equal complete requested sample counts".into());
    }
    if a.samples.len() < 3 || b.samples.len() < 3 {
        return Err("comparison requires at least three fresh unprofiled samples per side".into());
    }
    type TimingPair = (Vec<f64>, Vec<f64>);
    let mut timing: BTreeMap<(String, u64), TimingPair> = BTreeMap::new();
    let shape = |s: &experiment::Sample| {
        s.rows
            .iter()
            .map(|r| (r.ctx, r.prefill_tokens, r.gen_tokens, r.kvcache_bytes))
            .collect::<Vec<_>>()
    };
    let expected = shape(&a.samples[0]);
    if expected.is_empty()
        || expected
            .iter()
            .map(|r| r.0)
            .collect::<std::collections::BTreeSet<_>>()
            .len()
            != expected.len()
    {
        return Err("empty or duplicate benchmark frontiers".into());
    }
    for (candidate, scout) in [(false, a), (true, b)] {
        for sample in &scout.samples {
            if shape(sample) != expected {
                return Err("token counts/frontiers/KV allocation differ across samples".into());
            }
            for row in &sample.rows {
                for (name, value) in [
                    ("prefill", 1.0 / row.prefill_tps),
                    ("decode", 1.0 / row.gen_tps),
                    ("first_token", row.first_token_sec),
                ] {
                    let entry = timing.entry((name.into(), row.ctx)).or_default();
                    if candidate {
                        entry.1.push(value);
                    } else {
                        entry.0.push(value);
                    }
                }
            }
        }
    }
    let baseline = proofs(ar, a)?;
    let candidate = proofs(br, b)?;
    let mut correctness = LogitError::default();
    let mut token_mismatches = 0;
    for (ctx, base) in &baseline {
        let (reference, tokens) = &base[0];
        let candidates = candidate.get(ctx).ok_or("missing candidate frontier")?;
        for (frontier, ids) in base.iter().chain(candidates) {
            if frontier.source != reference.source
                || frontier.backend != reference.backend
                || frontier.model != reference.model
                || frontier.quality != reference.quality
                || frontier.quant_bits != reference.quant_bits
                || frontier.ctx != reference.ctx
            {
                return Err("quality/quantization/context proof metadata differs".into());
            }
            token_mismatches +=
                u64::from(ids != tokens || frontier.argmax_id != reference.argmax_id);
            let error = logit_error(
                &reference.logits,
                &frontier.logits,
                args.logit_atol,
                args.logit_rtol,
            )?;
            correctness.max_abs = correctness.max_abs.max(error.max_abs);
            correctness.max_scaled = correctness.max_scaled.max(error.max_scaled);
            correctness.bad += error.bad;
            correctness.checked += error.checked;
        }
    }
    let metrics = timing
        .into_iter()
        .map(|((name, ctx), (a, b))| delta(&name, ctx, &a, &b))
        .collect::<Result<Vec<_>, _>>()?;
    let verdict = verdict(
        &metrics,
        &correctness,
        token_mismatches,
        args.max_slowdown_percent,
    );
    Ok(Comparison {
        verdict,
        metrics,
        correctness,
        token_mismatches,
        max_slowdown_percent: args.max_slowdown_percent,
        logit_atol: args.logit_atol,
        logit_rtol: args.logit_rtol,
        changes,
        reasons: Vec::new(),
        method: "sample-extrema-envelope-v1".into(),
    })
}

fn verdict(
    metrics: &[Delta],
    correctness: &LogitError,
    mismatches: u64,
    tolerance: f64,
) -> Verdict {
    if correctness.bad > 0 || mismatches > 0 {
        return Verdict::Incorrect;
    }
    if metrics.iter().any(|d| d.lower_percent > tolerance) {
        return Verdict::Regressed;
    }
    if metrics.iter().any(|d| d.upper_percent > tolerance) {
        return Verdict::Inconclusive;
    }
    if metrics
        .iter()
        .any(|d| d.metric != "first_token" && d.upper_percent < -1.0)
    {
        return Verdict::Improved;
    }
    Verdict::Pass
}

pub fn run(args: &cli::Compare) -> Result<(), String> {
    if [args.max_slowdown_percent, args.logit_atol, args.logit_rtol]
        .into_iter()
        .any(|v| !v.is_finite() || v < 0.0)
    {
        return Err("comparison tolerances must be finite and nonnegative".into());
    }
    artifact::directory(&args.out)?;
    let result = (|| {
        let a = experiment::load(&args.baseline)?;
        let b = experiment::load(&args.candidate)?;
        analyze(
            a.require()?,
            args.baseline.parent().unwrap_or(Path::new(".")),
            b.require()?,
            args.candidate.parent().unwrap_or(Path::new(".")),
            args,
        )
    })();
    let data = match result {
        Ok(data) => data,
        Err(error) => Comparison {
            verdict: Verdict::Incomparable,
            metrics: Vec::new(),
            correctness: LogitError::default(),
            token_mismatches: 0,
            max_slowdown_percent: args.max_slowdown_percent,
            logit_atol: args.logit_atol,
            logit_rtol: args.logit_rtol,
            changes: BTreeMap::new(),
            reasons: vec![error],
            method: "sample-extrema-envelope-v1".into(),
        },
    };
    let passed = matches!(data.verdict, Verdict::Pass | Verdict::Improved);
    let mut artifact = Artifact::new(data);
    artifact.complete = artifact
        .data
        .as_ref()
        .is_some_and(|d| d.verdict != Verdict::Incomparable);
    for path in [&args.baseline, &args.candidate] {
        if path.is_file() {
            artifact.inputs.push(artifact::reference(path, &args.out)?);
        }
    }
    artifact::save(&args.out.join("compare.json"), &artifact)?;
    println!(
        "compare: {:?}; {}",
        artifact.data.as_ref().unwrap().verdict,
        args.out.join("compare.json").display()
    );
    if args.regression && !passed {
        return Err("regression gate did not pass; see compare.json".into());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn preserves_live_roundtrip() {
        // These measured Qwen ratios exposed a lossy JSON parse on reload.
        let recorded: Comparison =
            serde_json::from_str(include_str!("../tests/fixtures/compare-live.json")).unwrap();
        recorded.validate().unwrap();
        let encoded = serde_json::to_vec(&recorded).unwrap();
        let mut loaded: Comparison = serde_json::from_slice(&encoded).unwrap();
        loaded.validate().unwrap();
        loaded.metrics[0].median_percent += 0.01;
        assert!(loaded.validate().is_err());
    }

    #[test]
    fn separates_regression_noise() {
        let stable = delta("prefill", 2048, &[1.0, 1.01, 0.99], &[1.1, 1.11, 1.09]).unwrap();
        assert!(stable.lower_percent > 3.0);
        let noise = delta("prefill", 2048, &[0.9, 1.0, 1.1], &[0.9, 1.03, 1.2]).unwrap();
        assert!(noise.lower_percent < 3.0 && noise.upper_percent > 3.0);
        assert!(delta("prefill", 2048, &[1.0], &[0.0]).is_err());
        assert!(delta("prefill", 2048, &[1.0], &[f64::NAN]).is_err());
    }
    #[test]
    fn checks_every_logit() {
        let a = vec![0.0, 1.0, -2.0, 0.001];
        assert!(logit_error(&a, &a, 1e-4, 1e-4).unwrap().bad == 0);
        let mut b = a.clone();
        b[3] = 0.01;
        assert_eq!(logit_error(&a, &b, 1e-4, 1e-4).unwrap().bad, 1);
        assert!(logit_error(&a, &[f64::NAN], 1e-4, 1e-4).is_err());
    }
}
