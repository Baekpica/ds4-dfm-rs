//! Link an operator-controlled stop/start to the first disk-restore request.
use super::*;

fn process_pair(previous: &Value, current: &Value) -> Result<(), String> {
    let old = &previous["server"];
    let new = &current["server"];
    let pid = old["pid"]
        .as_u64()
        .ok_or("restart seed needs --server-pid")?;
    let old_start = old["start_ticks"]
        .as_u64()
        .ok_or("seed start time missing")?;
    let new_start = new["start_ticks"]
        .as_u64()
        .ok_or("restore needs --server-pid")?;
    if old["boot_id"] == new["boot_id"] && (new_start <= old_start || old["pid"] == new["pid"]) {
        return Err("restart restore requires a newer server process".into());
    }
    for key in [
        "executable",
        "executable_sha256",
        "argv_sha256",
        "cwd",
        "environment",
        "source",
        "unreviewed_environment",
    ] {
        if old.get(key).is_none() || old[key] != new[key] {
            return Err(format!("restart runtime changed: {key}"));
        }
    }
    if previous["inputs"] != current["inputs"] {
        return Err("restart input identities changed".into());
    }
    if let Ok(stat) = fs::read_to_string(format!("/proc/{pid}/stat")) {
        let fields = stat
            .rsplit_once(')')
            .ok_or("invalid seed process stat")?
            .1
            .split_whitespace()
            .collect::<Vec<_>>();
        if fields.get(19).and_then(|v| v.parse::<u64>().ok()) == Some(old_start)
            && fields.first() != Some(&"Z")
        {
            return Err("seed server is still running; stop it before restart collection".into());
        }
    }
    Ok(())
}

fn seed_verified(
    workload: &Workload,
    identity: &context::Identity,
    checked: &mut artifact::Verification,
) -> Result<review::Run, String> {
    let path = workload
        .restart_from
        .as_ref()
        .ok_or("missing restart seed")?;
    let seed = review::Run::load_verified(path, checked)?;
    let data = seed.data()?;
    if !data.passed
        || data.family != workload.family
        || data.serving_plan["effective"]["disk"] != true
    {
        return Err("restart seed must pass on the same family with disk enabled".into());
    }
    process_pair(
        &serde_json::to_value(&data.identity).map_err(|e| e.to_string())?,
        &serde_json::to_value(identity).map_err(|e| e.to_string())?,
    )?;
    let previous: Workload = seed.json(Path::new("workload.json"))?;
    let restored = workload
        .cases
        .first()
        .and_then(|case| case.request["messages"].as_array())
        .ok_or("restore request missing")?;
    if !previous.cases.iter().any(|case| {
        case.request["messages"]
            .as_array()
            .is_some_and(|messages| !messages.is_empty() && restored.starts_with(messages))
    }) {
        return Err("restore conversation does not extend a seeded request".into());
    }
    Ok(seed)
}

pub(super) fn prepare(
    workload: &Workload,
    identity: &context::Identity,
    out: &Path,
    repeats: u32,
    checked: &mut artifact::Verification,
) -> Result<Option<artifact::Reference>, String> {
    let Some(path) = &workload.restart_from else {
        return Ok(None);
    };
    if repeats != 1 {
        return Err(
            "restart restore needs a fresh process per capture; --repeats must be 1".into(),
        );
    }
    seed_verified(workload, identity, checked)?;
    Ok(Some(artifact::reference(path, out)?))
}

pub(super) fn first_request(
    workload: &Workload,
    current: &Value,
    before: &Value,
    index: usize,
    repeat: u32,
    checked: &mut artifact::Verification,
) -> Result<(), String> {
    first_verified(workload, current, before, index, repeat, checked)?;
    checked.finish()
}

fn first_verified(
    workload: &Workload,
    current: &Value,
    before: &Value,
    index: usize,
    repeat: u32,
    checked: &mut artifact::Verification,
) -> Result<(), String> {
    if index != 0
        || repeat != 0
        || route_count(before)? != 0
        || current["effective"]["disk"] != true
    {
        return Err(
            "restart restore must be the new server's first request with disk enabled".into(),
        );
    }
    let seed: Artifact<Evidence> = artifact::load_verified(
        workload
            .restart_from
            .as_ref()
            .ok_or("restart seed missing")?,
        checked,
    )?;
    let previous = &seed.require()?.serving_plan;
    if post_open_quote(previous) != post_open_quote(current) {
        return Err("restart post-open memory quote changed".into());
    }
    for key in ["family", "requested", "effective", "qualified", "controls"] {
        if previous[key] != current[key] {
            return Err(format!("restart serving plan changed: {key}"));
        }
    }
    Ok(())
}

pub(super) fn review(
    run: &review::Run,
    workload: &Workload,
    checked: &mut artifact::Verification,
) -> Result<(), String> {
    let data = run.data()?;
    if workload.restart_from.is_none() {
        if data.restart_from.is_some() {
            return Err("unexpected restart linkage".into());
        }
        return Ok(());
    }
    let reference = data
        .restart_from
        .as_ref()
        .ok_or("restart evidence lacks seed reference")?;
    let bytes = run.bytes(Path::new(&reference.path))?;
    let expected = workload
        .restart_from
        .as_ref()
        .ok_or("restart seed missing")?;
    if artifact::hash_bytes(&bytes) != reference.sha256
        || artifact::hash(expected)? != reference.sha256
        || data.requested_repeats != 1
    {
        return Err("restart seed reference changed".into());
    }
    seed_verified(workload, &data.identity, checked)?;
    let before: Value = run.json(Path::new("case-000/before.json"))?;
    first_verified(workload, &data.serving_plan, &before, 0, 0, checked)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    #[test]
    fn requires_new_process_and_same_runtime() {
        let old = json!({"server":{"pid":4294967000u64,"start_ticks":1,"boot_id":"fixture","executable":"server","executable_sha256":"binary","argv_sha256":"args","cwd":"/tmp","environment":{},"source":{},"unreviewed_environment":[]},"inputs":{}});
        assert!(process_pair(&old, &old).is_err_and(|e| e.contains("newer")));
        let mut new = old.clone();
        new["server"]["pid"] = json!(4294967001u64);
        new["server"]["start_ticks"] = json!(2);
        assert!(process_pair(&old, &new).is_ok());
        new["server"]["executable_sha256"] = json!("changed");
        assert!(process_pair(&old, &new).is_err_and(|e| e.contains("runtime changed")));
    }
}
