use super::*;
use std::collections::BTreeMap;

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Candidate {
    arguments: BTreeMap<String, u64>,
    qualified: bool,
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Controls {
    family: String,
    capabilities: Value,
    candidates: Vec<Candidate>,
}

impl Payload for Controls {
    const KIND: &'static str = "serving-controls";
    fn validate(&self) -> Result<(), String> {
        if self.family.is_empty()
            || self
                .candidates
                .iter()
                .any(|c| c.qualified || c.arguments.len() != 1)
        {
            return Err("invalid unqualified serving controls".into());
        }
        Ok(())
    }
}

pub(super) fn run(args: &cli::ServingControls) -> Result<(), String> {
    let value: Value = serde_json::from_slice(&fs::read(&args.plan).map_err(|e| e.to_string())?)
        .map_err(|e| e.to_string())?;
    let plan = value.get("serving").unwrap_or(&value);
    let family = plan["family"].as_str().ok_or("missing plan family")?;
    let caps = plan
        .get("controls")
        .ok_or("plan lacks capability-derived controls")?;
    let chunks = caps["scheduler_chunks"]
        .as_array()
        .ok_or("missing scheduler chunk controls")?;
    let cap = plan["effective"]["native_chunk"].as_u64().unwrap_or(0);
    let mut candidates = Vec::new();
    for (flag, current) in [
        ("--prefill-chunk", "sched_chunk"),
        ("--prefill-chunk-live", "sched_chunk_live"),
    ] {
        let mut seen = BTreeSet::new();
        for chunk in chunks {
            let width = chunk
                .as_u64()
                .filter(|v| *v > 0 && *v <= cap)
                .ok_or("scheduler chunk outside native capability")?;
            if !seen.insert(width) {
                return Err("duplicate scheduler chunk control".into());
            }
            if plan["effective"][current].as_u64() == Some(width) {
                continue;
            }
            candidates.push(Candidate {
                arguments: BTreeMap::from([(flag.into(), width)]),
                qualified: false,
            });
        }
    }
    let mut result = Artifact::new(Controls {
        family: family.into(),
        capabilities: caps.clone(),
        candidates,
    });
    fs::create_dir(&args.out).map_err(|e| e.to_string())?;
    result
        .inputs
        .push(artifact::reference(&args.plan, &args.out)?);
    result.warnings.push("Proposals only; every candidate requires its complete serving workload and fresh qualification.".into());
    artifact::save(&args.out.join("controls.json"), &result)
}
