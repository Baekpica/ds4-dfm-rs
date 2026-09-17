//! Local file identity for disk KV, not a full weight-content attestation.

use ds4_core::{identify_gguf, TensorInventory};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::io::Read;
#[cfg(unix)]
use std::os::unix::fs::MetadataExt;
use std::path::{Path, PathBuf};

const SMALL_MAX: u64 = 16 * 1024 * 1024;

#[derive(Clone)]
struct Input {
    path: PathBuf,
    contents: bool,
    optional: bool,
}

impl Input {
    fn weights(path: PathBuf) -> Self {
        Self {
            path,
            contents: false,
            optional: false,
        }
    }
    fn optional(path: PathBuf) -> Self {
        Self {
            path,
            contents: true,
            optional: true,
        }
    }
    fn small(path: PathBuf) -> Self {
        Self {
            path,
            contents: true,
            optional: false,
        }
    }
}

/// Source metadata captured before native open. Rechecked before enabling disk KV.
pub struct CacheIdentity {
    inputs: Vec<Input>,
    before: Value,
    settings: String,
}

impl CacheIdentity {
    pub fn capture(model: &Path, sidecars: &[&Path], settings: &str) -> Result<Self, String> {
        let roots: Vec<_> = std::iter::once(model)
            .chain(sidecars.iter().copied())
            .map(|path| Input::weights(path.to_owned()))
            .collect();
        let roots_before = snapshot(&roots)?;
        let mut inputs = Vec::new();
        for path in std::iter::once(model).chain(sidecars.iter().copied()) {
            let inventory = TensorInventory::open(path).map_err(|e| e.to_string())?;
            inputs.extend(
                inventory
                    .shards
                    .into_iter()
                    .map(|shard| Input::weights(shard.path)),
            );
            let parent = path.parent().unwrap_or_else(|| Path::new("."));
            inputs.push(Input::optional(parent.join("chat_template.jinja")));
            inputs.push(Input::optional(parent.join("tokenizer_config.json")));
        }
        let shape = identify_gguf(model).map_err(|e| e.to_string())?.shape;
        let ple = if shape.family == ds4_core::ModelFamily::Qwen4Exp {
            Some(add_ple(model, &mut inputs)?)
        } else {
            None
        };
        inputs.sort_by(|a, b| a.path.cmp(&b.path));
        inputs.dedup_by(|a, b| a.path == b.path);
        let settings = format!("{settings}\nshape={shape:?}\nenv={:?}", state_env());
        let before = snapshot(&inputs)?;
        if snapshot(&roots)? != roots_before
            || ple.is_some_and(|(input, expected)| snapshot(&[input]).ok() != Some(expected))
        {
            return Err("artifact metadata changed while enumerating identity inputs".into());
        }
        Ok(Self {
            before,
            inputs,
            settings,
        })
    }

    pub fn finish(self, effective: &str) -> Result<[u8; 32], String> {
        if snapshot(&self.inputs)? != self.before {
            return Err("model/tokenizer/template inputs changed while opening the model".into());
        }
        let runtime = snapshot(&runtime_inputs()?)?;
        let settings = format!(
            "{}\neffective={effective}\nenv={:?}",
            self.settings,
            state_env()
        );
        Ok(digest(
            &json!({"sources": self.before, "runtime": runtime}),
            &settings,
        ))
    }
}

fn digest(files: &Value, settings: &str) -> [u8; 32] {
    Sha256::digest(
        serde_json::to_vec(&json!({
            "method": "local-file-stat-v1", "files": files, "settings": settings,
        }))
        .expect("identity JSON is serializable"),
    )
    .into()
}

fn stamp(metadata: &fs::Metadata) -> Result<Value, String> {
    #[cfg(unix)]
    return Ok(json!({"device": metadata.dev(), "inode": metadata.ino(),
        "bytes": metadata.len(), "mtime": [metadata.mtime(), metadata.mtime_nsec()],
        "ctime": [metadata.ctime(), metadata.ctime_nsec()]}));
    #[cfg(not(unix))]
    {
        let _ = metadata;
        Err("local-file-stat-v1 requires Unix file identity".into())
    }
}

fn snapshot(inputs: &[Input]) -> Result<Value, String> {
    let mut out = Vec::new();
    for input in inputs {
        let mut file = match fs::File::open(&input.path) {
            Ok(file) => file,
            Err(e) if input.optional && e.kind() == std::io::ErrorKind::NotFound => {
                out.push(json!({"path": input.path, "missing": true}));
                continue;
            }
            Err(e) => return Err(format!("{}: {e}", input.path.display())),
        };
        let before = stamp(&file.metadata().map_err(|e| e.to_string())?)?;
        let path = input.path.canonicalize().map_err(|e| e.to_string())?;
        let hash = if input.contents {
            let mut bytes = Vec::new();
            (&mut file)
                .take(SMALL_MAX + 1)
                .read_to_end(&mut bytes)
                .map_err(|e| e.to_string())?;
            if bytes.len() as u64 > SMALL_MAX {
                return Err(format!(
                    "identity sidecar exceeds 16 MiB: {}",
                    path.display()
                ));
            }
            Some(
                Sha256::digest(&bytes)
                    .iter()
                    .map(|b| format!("{b:02x}"))
                    .collect::<String>(),
            )
        } else {
            None
        };
        let after = stamp(&file.metadata().map_err(|e| e.to_string())?)?;
        if before != after
            || before != stamp(&fs::metadata(&input.path).map_err(|e| e.to_string())?)?
        {
            return Err(format!(
                "identity input changed while reading: {}",
                path.display()
            ));
        }
        out.push(json!({"path": path, "stat": before, "contents_sha256": hash}));
    }
    Ok(Value::Array(out))
}

fn state_env() -> BTreeMap<String, Vec<u8>> {
    std::env::vars_os().filter_map(|(key, value)| {
        Some((key.into_string().ok()?, value.as_encoded_bytes().to_vec()))
    }).filter(|(key, _)| {
        (key.starts_with("DS4_") || key == "CUDA_VISIBLE_DEVICES" || key == "NVIDIA_TF32_OVERRIDE")
            // Keep native compute/KV/RoPE controls, including DS4_CONT_MTP_*
            // and DS4_CONT_DSPARK. Only these two continuous scheduling keys
            // are redundant with the normalized effective schedule in finish.
            && !key.starts_with("DS4_CUDA_WEIGHT_IPC")
            && !key.starts_with("DS4_SERVER_")
            && !key.starts_with("DS4_MEM")
            && !matches!(key.as_str(), "DS4_CONT_PREFILL_CHUNK" | "DS4_CONT_PREFILL_CHUNK_LIVE"
                | "DS4_SESSION_LAZY_GRAPH" | "DS4_SESSION_GRAPH_FIT"
                | "DS4_NO_BOOT_PREWARM" | "DS4_BOOT_PREWARM" | "DS4_GOV_TRACE")
    }).collect()
}

fn runtime_inputs() -> Result<Vec<Input>, String> {
    let mut paths = BTreeSet::new();
    #[cfg(target_os = "linux")]
    {
        paths.insert(PathBuf::from("/proc/self/exe"));
        // Native inference is linked into the executable. Include mapped
        // shared libraries as well, including dynamically opened CUDA libraries.
        let maps = fs::read_to_string("/proc/self/maps").map_err(|e| e.to_string())?;
        for line in maps.lines() {
            let name = line
                .split_whitespace()
                .skip(5)
                .collect::<Vec<_>>()
                .join(" ");
            if name.starts_with('/') && name.contains(".so") {
                paths.insert(PathBuf::from(name));
            }
        }
    }
    #[cfg(not(target_os = "linux"))]
    {
        paths.insert(std::env::current_exe().map_err(|e| e.to_string())?);
    }
    Ok(paths.into_iter().map(Input::weights).collect())
}

fn add_ple(model: &Path, inputs: &mut Vec<Input>) -> Result<(Input, Value), String> {
    let parent = model.parent().ok_or("model has no parent")?;
    let selected = std::env::var_os("DS4_QWEN_PLE_DIR")
        .filter(|v| !v.is_empty())
        .map(PathBuf::from);
    let path = selected
        .as_ref()
        .map(|p| p.join("ple-manifest.json"))
        .unwrap_or_else(|| parent.join("ple/ple-manifest.json"));
    ple_inputs(path, parent, selected.as_deref(), inputs)
}

fn ple_inputs(
    path: PathBuf,
    parent: &Path,
    selected: Option<&Path>,
    inputs: &mut Vec<Input>,
) -> Result<(Input, Value), String> {
    let manifest_input = Input::small(path.clone());
    let before = snapshot(std::slice::from_ref(&manifest_input))?;
    let mut raw = Vec::new();
    fs::File::open(&path)
        .map_err(|e| e.to_string())?
        .take(SMALL_MAX + 1)
        .read_to_end(&mut raw)
        .map_err(|e| e.to_string())?;
    if raw.len() as u64 > SMALL_MAX {
        return Err("PLE manifest exceeds 16 MiB".into());
    }
    let manifest: Value = serde_json::from_slice(&raw).map_err(|e| e.to_string())?;
    let fp8 = manifest["format_version"].as_u64() == Some(2);
    let root = if fp8 {
        path.parent().ok_or("PLE manifest has no parent")?
    } else {
        selected.unwrap_or(parent)
    };
    for part in manifest["logical_parts"]
        .as_array()
        .ok_or("PLE logical parts missing")?
    {
        let file = part["physical_file"]
            .as_str()
            .ok_or("PLE physical file missing")?;
        inputs.push(Input::weights(root.join(file)));
    }
    if fp8 {
        let scale = manifest
            .pointer("/quantization/scale/path")
            .and_then(Value::as_str)
            .ok_or("PLE FP8 scale missing")?;
        inputs.push(Input::small(root.join(scale)));
    }
    inputs.push(manifest_input.clone());
    Ok((manifest_input, before))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn continuous_inference_controls_change_identity() {
        for (key, before, after) in [
            ("DS4_CONT_MTP_DEPTH", Some("1"), Some("2")),
            ("DS4_CONT_MTP_MODE", Some("0"), Some("2")),
            ("DS4_CONT_MTP_BATCH_DRAFT", None, Some("1")),
            ("DS4_CONT_DSPARK", Some("0"), Some("1")),
            ("DS4_CONT_MTP_DRAFT_PROBE", None, Some("1")),
        ] {
            let old = std::env::var_os(key);
            let fingerprint = |value: Option<&str>| {
                match value {
                    Some(value) => std::env::set_var(key, value),
                    None => std::env::remove_var(key),
                }
                digest(&Value::Null, &format!("env={:?}", state_env()))
            };
            let before = fingerprint(before);
            let after = fingerprint(after);
            match old {
                Some(value) => std::env::set_var(key, value),
                None => std::env::remove_var(key),
            }
            assert_ne!(before, after, "native compute control was omitted: {key}");
        }
    }

    #[test]
    #[cfg(unix)]
    fn non_utf8_env_is_bound() {
        use std::os::unix::ffi::OsStringExt;
        let key = "DS4_CACHE_TEST";
        let old = std::env::var_os(key);
        std::env::set_var(key, std::ffi::OsString::from_vec(vec![0xff, 7]));
        let result = std::panic::catch_unwind(state_env);
        match old {
            Some(value) => std::env::set_var(key, value),
            None => std::env::remove_var(key),
        }
        assert!(
            result.is_ok(),
            "identity must preserve non-UTF8 environment bytes"
        );
        assert_eq!(result.unwrap()[key], [0xff, 7]);
    }

    #[test]
    fn changed_shard_invalidates() {
        let dir = std::env::temp_dir().join(format!("ds4-cache-identity-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let a = dir.join("one.gguf");
        let b = dir.join("two.gguf");
        std::fs::write(&a, b"weights-one").unwrap();
        std::fs::write(&b, b"weights-two").unwrap();
        let inputs = vec![Input::weights(a), Input::weights(b.clone())];
        let before = snapshot(&inputs).unwrap();
        assert_eq!(before, snapshot(&inputs).unwrap());
        let replacement = dir.join("replacement");
        std::fs::write(&replacement, b"weights-new").unwrap();
        std::fs::rename(replacement, &b).unwrap();
        assert_ne!(before, snapshot(&inputs).unwrap());
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn template_and_state_are_bound() {
        let dir = std::env::temp_dir().join(format!("ds4-cache-template-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let template = dir.join("chat_template.jinja");
        let inputs = vec![Input::optional(template.clone())];
        let missing = snapshot(&inputs).unwrap();
        std::fs::write(&template, "{{ messages }}").unwrap();
        let old = snapshot(&inputs).unwrap();
        assert_ne!(missing, old);
        std::fs::write(&template, "{{ tools }}").unwrap();
        let new = snapshot(&inputs).unwrap();
        assert_ne!(old, new);
        assert_ne!(digest(&old, "rope=1"), digest(&old, "rope=2"));
        assert_ne!(digest(&old, "runtime=1"), digest(&old, "runtime=2"));
        assert_ne!(digest(&old, "runtime=1"), digest(&new, "runtime=1"));
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn opening_input_change_fails() {
        let dir = std::env::temp_dir().join(format!("ds4-cache-open-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        let path = dir.join("weights");
        fs::write(&path, b"before").unwrap();
        let inputs = vec![Input::weights(path.clone())];
        let identity = CacheIdentity {
            before: snapshot(&inputs).unwrap(),
            inputs,
            settings: "test".into(),
        };
        fs::write(path, b"changed").unwrap();
        assert!(identity
            .finish("ctx=2048")
            .unwrap_err()
            .contains("changed while opening"));
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn bounded_sidecar_and_ple() {
        let dir = std::env::temp_dir().join(format!("ds4-cache-ple-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        let ple = dir.join("ple");
        fs::create_dir_all(&ple).unwrap();
        let part = ple.join("part.bin");
        let scale = ple.join("scale.bin");
        fs::write(&part, b"part").unwrap();
        fs::write(&scale, b"scale").unwrap();
        let path = ple.join("ple-manifest.json");
        fs::write(
            &path,
            serde_json::to_vec(&json!({"format_version":2,
            "logical_parts":[{"physical_file":"part.bin"}],
            "quantization":{"scale":{"path":"scale.bin"}}}))
            .unwrap(),
        )
        .unwrap();
        let mut inputs = Vec::new();
        let (manifest, baseline) = ple_inputs(path, &dir, None, &mut inputs).unwrap();
        assert_eq!(snapshot(&[manifest]).unwrap(), baseline);
        assert_eq!(inputs.len(), 3);
        let old = snapshot(&inputs).unwrap();
        fs::write(&part, b"PART").unwrap();
        let changed = snapshot(&inputs).unwrap();
        assert_ne!(old, changed);
        fs::write(&scale, b"SCALE").unwrap();
        assert_ne!(changed, snapshot(&inputs).unwrap());
        fs::File::create(&scale)
            .unwrap()
            .set_len(SMALL_MAX + 1)
            .unwrap();
        assert!(snapshot(&inputs).unwrap_err().contains("exceeds 16 MiB"));
        fs::remove_file(part).unwrap();
        assert!(snapshot(&inputs).is_err());
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    #[ignore = "metadata-only: set DS4_IDENTITY_MODEL and optional DS4_IDENTITY_MTP"]
    fn artifact_restart_identity() {
        let model = std::env::var("DS4_IDENTITY_MODEL").unwrap();
        let mtp = std::env::var("DS4_IDENTITY_MTP").ok();
        let sidecars: Vec<_> = mtp.as_deref().map(Path::new).into_iter().collect();
        let first = CacheIdentity::capture(Path::new(&model), &sidecars, "cuda").unwrap();
        let expected: usize = std::env::var("DS4_IDENTITY_SHARDS")
            .unwrap()
            .parse()
            .unwrap();
        assert_eq!(
            first
                .inputs
                .iter()
                .filter(|v| v.path.extension().is_some_and(|ext| ext == "gguf"))
                .count(),
            expected
        );
        let first = first.finish("ctx=2048;mtp=on").unwrap();
        let second = CacheIdentity::capture(Path::new(&model), &sidecars, "cuda")
            .unwrap()
            .finish("ctx=2048;mtp=on")
            .unwrap();
        assert_eq!(first, second);
    }
}
