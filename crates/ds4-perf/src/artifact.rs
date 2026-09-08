use serde::{de::DeserializeOwned, Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    fs,
    io::{Read, Write},
    path::Path,
    time::SystemTime,
};

/// Each payload evolves independently. Unknown versions fail closed.
pub trait Payload {
    const KIND: &'static str;
    const SCHEMA: u32 = 1;
    fn validate(&self) -> Result<(), String>;
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Reference {
    pub path: String,
    pub sha256: String,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Artifact<T> {
    pub schema_version: u32,
    pub kind: String,
    pub producer_version: String,
    pub created_unix: u64,
    pub complete: bool,
    pub inputs: Vec<Reference>,
    pub warnings: Vec<String>,
    pub data: Option<T>,
}

impl<T: Payload> Artifact<T> {
    pub fn new(data: T) -> Self {
        Self {
            schema_version: T::SCHEMA,
            kind: T::KIND.into(),
            producer_version: env!("CARGO_PKG_VERSION").into(),
            created_unix: SystemTime::now()
                .duration_since(SystemTime::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
            complete: true,
            inputs: Vec::new(),
            warnings: Vec::new(),
            data: Some(data),
        }
    }
    pub fn failed(message: String) -> Self {
        Self {
            schema_version: T::SCHEMA,
            kind: T::KIND.into(),
            producer_version: env!("CARGO_PKG_VERSION").into(),
            created_unix: SystemTime::now()
                .duration_since(SystemTime::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
            complete: false,
            inputs: Vec::new(),
            warnings: vec![message],
            data: None,
        }
    }
    pub fn require(&self) -> Result<&T, String> {
        if !self.complete {
            return Err(format!("incomplete {} evidence", T::KIND));
        }
        self.data
            .as_ref()
            .ok_or_else(|| format!("missing {} payload", T::KIND))
    }
}

fn hex(bytes: impl IntoIterator<Item = u8>) -> String {
    bytes
        .into_iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

pub fn hash_bytes(bytes: &[u8]) -> String {
    hex(Sha256::digest(bytes))
}

pub fn hash(path: &Path) -> Result<String, String> {
    let mut file = fs::File::open(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let mut state = Sha256::new();
    let mut buffer = [0u8; 64 * 1024];
    loop {
        let count = file.read(&mut buffer).map_err(|e| e.to_string())?;
        if count == 0 {
            break;
        }
        state.update(&buffer[..count]);
    }
    Ok(hex(state.finalize()))
}

pub fn reference(path: &Path, directory: &Path) -> Result<Reference, String> {
    let canonical = path.canonicalize().map_err(|e| e.to_string())?;
    let base = directory.canonicalize().map_err(|e| e.to_string())?;
    let relative = canonical.strip_prefix(base).unwrap_or(&canonical);
    Ok(Reference {
        path: relative.to_string_lossy().into(),
        sha256: hash(&canonical)?,
    })
}

pub fn save<T: Serialize + Payload>(path: &Path, value: &Artifact<T>) -> Result<(), String> {
    if value.kind != T::KIND || value.schema_version != T::SCHEMA {
        return Err("artifact type/schema mismatch".into());
    }
    if let Some(data) = &value.data {
        data.validate()?;
    }
    if value.complete && value.data.is_none() {
        return Err("complete artifact has no data".into());
    }
    let mut bytes = serde_json::to_vec_pretty(value).map_err(|e| e.to_string())?;
    bytes.push(b'\n');
    let nonce = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    let temporary = path.with_extension(format!("tmp-{}-{nonce}", std::process::id()));
    let result = (|| {
        let mut file = fs::OpenOptions::new()
            .create_new(true)
            .write(true)
            .open(&temporary)?;
        file.write_all(&bytes)?;
        file.sync_all()?;
        // Atomic publication with no overwrite, even if another writer wins.
        fs::hard_link(&temporary, path)
    })();
    let _ = fs::remove_file(temporary);
    result.map_err(|e| format!("{}: {e}", path.display()))
}

pub fn load<T: DeserializeOwned + Payload>(path: &Path) -> Result<Artifact<T>, String> {
    let file = fs::File::open(path).map_err(|e| format!("{}: {e}", path.display()))?;
    const MAX_JSON: u64 = 64 * 1024 * 1024;
    if file.metadata().map_err(|e| e.to_string())?.len() > MAX_JSON {
        return Err("artifact exceeds 64 MiB".into());
    }
    let value: Artifact<T> = serde_json::from_reader(file.take(MAX_JSON))
        .map_err(|e| format!("{}: {e}", path.display()))?;
    if value.schema_version != T::SCHEMA || value.kind != T::KIND {
        return Err(format!(
            "{}: expected {} schema {}",
            path.display(),
            T::KIND,
            T::SCHEMA
        ));
    }
    if let Some(data) = &value.data {
        data.validate()?;
    }
    if value.complete {
        value.require()?;
    }
    verify_refs(
        path,
        &value.inputs,
        &mut std::collections::BTreeMap::new(),
        0,
    )?;
    Ok(value)
}

fn verify_refs(
    path: &Path,
    refs: &[Reference],
    checked: &mut std::collections::BTreeMap<std::path::PathBuf, String>,
    depth: usize,
) -> Result<(), String> {
    if depth > 32 {
        return Err("evidence reference depth exceeds 32".into());
    }
    for input in refs {
        let target = path
            .parent()
            .unwrap_or(Path::new("."))
            .join(&input.path)
            .canonicalize()
            .map_err(|e| e.to_string())?;
        if let Some(digest) = checked.get(&target) {
            if *digest != input.sha256 {
                return Err(format!("evidence changed: {}", input.path));
            }
            continue;
        }
        let digest = hash(&target)?;
        if digest != input.sha256 {
            return Err(format!("evidence changed: {}", input.path));
        }
        checked.insert(target.clone(), digest);
        if target.extension().is_none_or(|e| e != "json") {
            continue;
        }
        let file = fs::File::open(&target).map_err(|e| e.to_string())?;
        if file.metadata().map_err(|e| e.to_string())?.len() > 64 * 1024 * 1024 {
            return Err("JSON evidence exceeds 64 MiB".into());
        }
        let value: serde_json::Value = serde_json::from_reader(file).map_err(|e| e.to_string())?;
        if value.get("kind").is_none() || value.get("schema_version").is_none() {
            continue;
        }
        let refs: Vec<Reference> = serde_json::from_value(
            value
                .get("inputs")
                .ok_or("nested artifact inputs missing")?
                .clone(),
        )
        .map_err(|e| e.to_string())?;
        verify_refs(&target, &refs, checked, depth + 1)?;
    }
    Ok(())
}

pub fn directory(path: &Path) -> Result<(), String> {
    if let Some(parent) = path.parent().filter(|p| !p.as_os_str().is_empty()) {
        fs::create_dir_all(parent).map_err(|e| e.to_string())?;
    }
    fs::create_dir(path).map_err(|e| {
        format!(
            "cannot create new evidence directory {}: {e}",
            path.display()
        )
    })
}

/// Import an inspected machine/calibration bundle with its relative raw inputs.
/// External references require re-inspection into a self-contained directory.
pub fn bundle<T: DeserializeOwned + Payload>(
    path: &Path,
    out: &Path,
) -> Result<std::path::PathBuf, String> {
    load::<T>(path)?;
    let source = path.canonicalize().map_err(|e| e.to_string())?;
    let root = source.parent().ok_or("bundle has no parent")?;
    directory(out)?;
    let mut visited = std::collections::BTreeSet::new();
    let mut pending = vec![source.clone()];
    let mut total = 0u64;
    while let Some(path) = pending.pop() {
        let path = path.canonicalize().map_err(|e| e.to_string())?;
        if !visited.insert(path.clone()) {
            continue;
        }
        let relative = path
            .strip_prefix(root)
            .map_err(|_| "bundle has external raw inputs; create a self-contained inspection")?;
        total = total
            .checked_add(fs::metadata(&path).map_err(|e| e.to_string())?.len())
            .ok_or("bundle size overflow")?;
        if total > 256 * 1024 * 1024 {
            return Err("machine/calibration bundle exceeds 256 MiB".into());
        }
        let destination = out.join(relative);
        if let Some(parent) = destination.parent() {
            fs::create_dir_all(parent).map_err(|e| e.to_string())?;
        }
        let mut source_file = fs::File::open(&path).map_err(|e| e.to_string())?;
        let mut copy = fs::OpenOptions::new()
            .create_new(true)
            .write(true)
            .open(&destination)
            .map_err(|e| e.to_string())?;
        std::io::copy(&mut source_file, &mut copy).map_err(|e| e.to_string())?;
        if path.extension().is_none_or(|e| e != "json") {
            continue;
        }
        let value: serde_json::Value =
            serde_json::from_reader(fs::File::open(&path).map_err(|e| e.to_string())?)
                .map_err(|e| e.to_string())?;
        if value.get("kind").is_none() || value.get("schema_version").is_none() {
            continue;
        }
        let inputs: Vec<Reference> =
            serde_json::from_value(value.get("inputs").ok_or("bundle inputs missing")?.clone())
                .map_err(|e| e.to_string())?;
        pending.extend(
            inputs
                .into_iter()
                .map(|r| path.parent().unwrap().join(r.path)),
        );
    }
    let imported = out.join(source.file_name().unwrap());
    load::<T>(&imported)?;
    Ok(imported)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[derive(Debug, Serialize, Deserialize)]
    struct Example {
        value: u32,
    }
    impl Payload for Example {
        const KIND: &'static str = "example";
        fn validate(&self) -> Result<(), String> {
            Ok(())
        }
    }
    #[test]
    fn transitive_hashes() {
        let root = std::env::temp_dir().join(format!("ds4-tree-{}", std::process::id()));
        directory(&root).unwrap();
        fs::write(root.join("raw"), "original").unwrap();
        let mut first = Artifact::new(Example { value: 1 });
        first
            .inputs
            .push(reference(&root.join("raw"), &root).unwrap());
        save(&root.join("first.json"), &first).unwrap();
        let mut second = Artifact::new(Example { value: 2 });
        second
            .inputs
            .push(reference(&root.join("first.json"), &root).unwrap());
        save(&root.join("second.json"), &second).unwrap();
        fs::write(root.join("raw"), "altered").unwrap();
        let rejected = load::<Example>(&root.join("second.json")).is_err();
        fs::remove_dir_all(root).unwrap();
        assert!(rejected, "nested evidence tampering must fail");
    }

    #[test]
    fn moved_evidence_and_tampering() {
        let root = std::env::temp_dir().join(format!("ds4-evidence-{}", std::process::id()));
        let source = root.join("source");
        directory(&source).unwrap();
        fs::write(source.join("raw"), "original").unwrap();
        let mut artifact = Artifact::new(Example { value: 7 });
        artifact
            .inputs
            .push(reference(&source.join("raw"), &source).unwrap());
        save(&source.join("example.json"), &artifact).unwrap();
        assert!(save(
            &source.join("example.json"),
            &Artifact::new(Example { value: 99 })
        )
        .is_err());
        let moved = root.join("moved");
        fs::rename(&source, &moved).unwrap();
        assert_eq!(
            load::<Example>(&moved.join("example.json"))
                .unwrap()
                .require()
                .unwrap()
                .value,
            7
        );
        fs::write(moved.join("raw"), "changed").unwrap();
        assert!(load::<Example>(&moved.join("example.json"))
            .unwrap_err()
            .contains("evidence changed"));
        fs::remove_dir_all(root).unwrap();
    }
}
