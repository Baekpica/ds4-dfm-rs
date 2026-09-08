use crate::experiment::Workload;
use std::{
    collections::{BTreeMap, BTreeSet},
    ffi::OsString,
    path::{Path, PathBuf},
};

fn shards(path: &Path) -> Result<Vec<PathBuf>, String> {
    let Some(name) = path
        .file_name()
        .and_then(|n| n.to_str())
        .and_then(|n| n.strip_suffix(".gguf"))
    else {
        return Ok(vec![path.into()]);
    };
    let Some((prefix, count)) = name.rsplit_once("-of-") else {
        return Ok(vec![path.into()]);
    };
    let Some((base, index)) = prefix.rsplit_once('-') else {
        return Err("invalid GGUF shard name".into());
    };
    let count = count
        .parse::<u32>()
        .map_err(|_| "invalid GGUF shard count")?;
    let index = index
        .parse::<u32>()
        .map_err(|_| "invalid GGUF shard index")?;
    if count == 0 || count > 10000 || index == 0 || index > count {
        return Err("invalid GGUF shard set".into());
    }
    Ok((1..=count)
        .map(|i| path.with_file_name(format!("{base}-{i:05}-of-{count:05}.gguf")))
        .collect())
}

fn arguments(command: &[OsString]) -> Result<BTreeMap<String, PathBuf>, String> {
    let mut paths = BTreeMap::new();
    let mut args = command.iter().skip(1);
    while let Some(arg) = args.next() {
        let arg = arg
            .to_str()
            .ok_or("workload protocol requires UTF-8 option names")?;
        match arg {
            "-m" | "--model" | "--prompt-file" | "--chat-prompt-file" | "--mtp" => {
                let path = args.next().ok_or("benchmark path argument missing")?;
                let key = match arg {
                    "-m" | "--model" => "model",
                    "--mtp" => "mtp",
                    _ => "prompt",
                };
                paths.insert(key.into(), PathBuf::from(path));
            }
            "--cuda" | "--quality" | "--warm-weights" => {}
            "-sys" | "--system" | "--backend" | "-t" | "--threads" | "--ctx-start"
            | "--ctx-max" | "--ctx-alloc" | "--step-incr" | "--step-mul" | "--gen-tokens"
            | "--tokens" | "-n" | "--mtp-draft" | "--mtp-margin" => {
                args.next().ok_or("benchmark option value missing")?;
            }
            _ => {
                return Err(format!(
                    "workload protocol ds4-bench-v1 does not cover option {arg}"
                ))
            }
        }
    }
    if !paths.contains_key("model") || !paths.contains_key("prompt") {
        return Err(
            "workload protocol needs explicit -m and --prompt-file/--chat-prompt-file".into(),
        );
    }
    Ok(paths)
}

pub fn model_argument(command: &[OsString]) -> Result<PathBuf, String> {
    arguments(command)?
        .remove("model")
        .ok_or("benchmark model argument missing".into())
}

impl Workload {
    pub fn verify_scope(
        &self,
        command: &[OsString],
        env: &BTreeMap<OsString, OsString>,
    ) -> Result<(), String> {
        if self.protocol != "ds4-bench-v1" {
            return Err("unsupported workload protocol; use ds4-bench-v1".into());
        }
        let declared: BTreeSet<_> = self.files.values().map(|v| v.path.clone()).collect();
        let require = |path: &Path| -> Result<(), String> {
            let path = path
                .canonicalize()
                .map_err(|e| format!("{}: {e}", path.display()))?;
            if !declared.contains(&path) {
                return Err(format!("workload lacks consumed input: {}", path.display()));
            }
            Ok(())
        };
        let arguments = arguments(command)?;
        for (key, path) in &arguments {
            let path = path.canonicalize().map_err(|e| e.to_string())?;
            if let Some(file) = self.files.get(key) {
                if file.path != path {
                    return Err(format!("workload {key} differs from benchmark argument"));
                }
            } else {
                return Err(format!("workload lacks {key} input"));
            }
            for shard in shards(&path)? {
                require(&shard)?;
            }
            if key != "model" {
                continue;
            }
            // PLE is a sidecar read by the Qwen host even when weights are IPC.
            let parent = path.parent().ok_or("model has no parent directory")?;
            let ple = parent.join("ple/ple-manifest.json");
            if ple.exists() {
                require(&ple)?;
                let manifest: serde_json::Value =
                    serde_json::from_slice(&std::fs::read(&ple).map_err(|e| e.to_string())?)
                        .map_err(|e| e.to_string())?;
                let parts = manifest
                    .get("logical_parts")
                    .and_then(|v| v.as_array())
                    .ok_or("unrecognized PLE manifest")?;
                for part in parts {
                    let file = part
                        .get("physical_file")
                        .and_then(|v| v.as_str())
                        .ok_or("PLE physical file missing")?;
                    require(&parent.join(file))?;
                }
            }
        }
        for key in ["DS4_CUDA_WEIGHT_IPC_MANIFEST", "DS4_WEIGHT_SERVER"] {
            if let Some(path) = env.get(std::ffi::OsStr::new(key)).filter(|v| !v.is_empty()) {
                require(Path::new(path))?;
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn expands_split_models() {
        let result = shards(Path::new("/models/a-00001-of-00003.gguf")).unwrap();
        assert_eq!(result.len(), 3);
        assert_eq!(result[2], Path::new("/models/a-00003-of-00003.gguf"));
        assert!(shards(Path::new("/models/a-00004-of-00003.gguf")).is_err());
    }
    #[test]
    fn parses_values_as_values() {
        let args = [
            "bench",
            "--system",
            "--mtp",
            "-m",
            "main.gguf",
            "--prompt-file",
            "prompt.txt",
        ]
        .map(Into::into);
        let paths = arguments(&args).unwrap();
        assert!(!paths.contains_key("mtp"));
        assert_eq!(paths["model"], Path::new("main.gguf"));
    }
}
