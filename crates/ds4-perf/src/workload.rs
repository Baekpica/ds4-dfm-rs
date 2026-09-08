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
            let selected = env
                .get(std::ffi::OsStr::new("DS4_QWEN_PLE_DIR"))
                .filter(|v| !v.is_empty())
                .map(PathBuf::from);
            let ple = selected
                .as_ref()
                .map(|p| p.join("ple-manifest.json"))
                .unwrap_or_else(|| parent.join("ple/ple-manifest.json"));
            if selected.is_some() || ple.exists() {
                require(&ple)?;
                let manifest: serde_json::Value =
                    serde_json::from_slice(&std::fs::read(&ple).map_err(|e| e.to_string())?)
                        .map_err(|e| e.to_string())?;
                let fp8 = manifest.get("format_version").and_then(|v| v.as_u64()) == Some(2);
                let root = if fp8 {
                    ple.parent().ok_or("PLE manifest has no parent")?
                } else {
                    selected.as_deref().unwrap_or(parent)
                };
                let parts = manifest
                    .get("logical_parts")
                    .and_then(|v| v.as_array())
                    .ok_or("unrecognized PLE manifest")?;
                for part in parts {
                    let file = part
                        .get("physical_file")
                        .and_then(|v| v.as_str())
                        .ok_or("PLE physical file missing")?;
                    require(&root.join(file))?;
                }
                if fp8 {
                    let scale = manifest
                        .pointer("/quantization/scale/path")
                        .and_then(|v| v.as_str())
                        .ok_or("PLE FP8 scale missing")?;
                    require(&root.join(scale))?;
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
    fn sidecar_override_is_pinned() {
        use crate::experiment::InputFile;
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let root = std::env::temp_dir().join(format!("ds4-fp8-{}-{nonce}", std::process::id()));
        let sidecar = root.join("fp8");
        std::fs::create_dir_all(&sidecar).unwrap();
        for name in ["model", "prompt", "fp8/part.bin", "fp8/scale.bin"] {
            std::fs::write(root.join(name), b"fixture").unwrap();
        }
        std::fs::write(
            sidecar.join("ple-manifest.json"),
            br#"{
            "format_version":2,"logical_parts":[{"physical_file":"part.bin"}],
            "quantization":{"scale":{"path":"scale.bin"}}
        }"#,
        )
        .unwrap();
        let mut workload = Workload {
            protocol: "ds4-bench-v1".into(),
            name: "fixture".into(),
            family: "qwen4exp".into(),
            files: BTreeMap::new(),
            shape: BTreeMap::new(),
            cache_state: "fresh".into(),
        };
        for name in ["model", "prompt"] {
            workload.files.insert(
                name.into(),
                InputFile {
                    path: root.join(name).canonicalize().unwrap(),
                    sha256: String::new(),
                },
            );
        }
        let command = vec![
            "bench".into(),
            "-m".into(),
            root.join("model").into(),
            "--prompt-file".into(),
            root.join("prompt").into(),
        ];
        let env = BTreeMap::from([("DS4_QWEN_PLE_DIR".into(), sidecar.clone().into())]);
        let missing_manifest = workload.verify_scope(&command, &env).is_err();
        for name in ["ple-manifest.json", "part.bin"] {
            workload.files.insert(
                name.into(),
                InputFile {
                    path: sidecar.join(name).canonicalize().unwrap(),
                    sha256: String::new(),
                },
            );
        }
        let missing_scale = workload.verify_scope(&command, &env).is_err();
        workload.files.insert(
            "scale".into(),
            InputFile {
                path: sidecar.join("scale.bin").canonicalize().unwrap(),
                sha256: String::new(),
            },
        );
        let complete = workload.verify_scope(&command, &env);
        std::fs::remove_dir_all(root).unwrap();
        assert!(missing_manifest, "an override must pin its manifest");
        assert!(missing_scale, "FP8 scale bytes are a consumed input");
        complete.unwrap();
    }

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
