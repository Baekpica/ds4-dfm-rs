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
    hash_file(&mut file)
}

fn hash_file(file: &mut fs::File) -> Result<String, String> {
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

#[derive(Clone, PartialEq, Eq)]
struct FileStamp {
    size: u64,
    modified: SystemTime,
    #[cfg(unix)]
    identity: [u64; 6],
}

impl FileStamp {
    fn read(metadata: &fs::Metadata) -> Result<Self, String> {
        #[cfg(unix)]
        use std::os::unix::fs::MetadataExt;
        if !metadata.is_file() {
            return Err("evidence input must be a regular file".into());
        }
        Ok(Self {
            size: metadata.len(),
            modified: metadata.modified().map_err(|e| e.to_string())?,
            #[cfg(unix)]
            identity: [
                metadata.dev(),
                metadata.ino(),
                metadata.mtime() as u64,
                metadata.mtime_nsec() as u64,
                metadata.ctime() as u64,
                metadata.ctime_nsec() as u64,
            ],
        })
    }
}

/// Share full-byte verification only within one operation. Reuse never trusts
/// an expected hash alone: replacement or metadata drift invalidates the read.
#[derive(Default)]
pub struct Verification {
    files: std::collections::BTreeMap<std::path::PathBuf, (FileStamp, String)>,
    aliases: std::collections::BTreeMap<std::path::PathBuf, std::path::PathBuf>,
    #[cfg(test)]
    full_reads: usize,
}

impl Verification {
    pub fn verify(&mut self, path: &Path, expected: &str) -> Result<(), String> {
        let alias = std::env::current_dir()
            .map_err(|e| e.to_string())?
            .join(path);
        let canonical = path.canonicalize().map_err(|e| e.to_string())?;
        if self
            .aliases
            .get(&alias)
            .is_some_and(|previous| previous != &canonical)
        {
            return Err(format!("evidence path changed: {}", path.display()));
        }
        let mut file = fs::File::open(&canonical).map_err(|e| e.to_string())?;
        let before = FileStamp::read(&file.metadata().map_err(|e| e.to_string())?)?;
        let digest = match self.files.get(&canonical) {
            Some((stamp, digest)) => {
                if *stamp != before {
                    return Err(format!(
                        "evidence changed during verification: {}",
                        path.display()
                    ));
                }
                digest.clone()
            }
            None => {
                #[cfg(test)]
                {
                    self.full_reads += 1;
                }
                hash_file(&mut file)?
            }
        };
        if digest != expected {
            return Err(format!("evidence changed: {}", path.display()));
        }
        let after = FileStamp::read(&file.metadata().map_err(|e| e.to_string())?)?;
        let current = FileStamp::read(&fs::metadata(&canonical).map_err(|e| e.to_string())?)?;
        if before != after
            || before != current
            || path.canonicalize().ok().as_ref() != Some(&canonical)
        {
            return Err(format!(
                "evidence changed during verification: {}",
                path.display()
            ));
        }
        self.aliases.insert(alias, canonical.clone());
        self.files.insert(canonical, (before, digest));
        Ok(())
    }

    pub fn finish(&self) -> Result<(), String> {
        for (alias, target) in &self.aliases {
            if alias.canonicalize().ok().as_ref() != Some(target) {
                return Err(format!("evidence path changed: {}", alias.display()));
            }
        }
        for (path, (before, _)) in &self.files {
            let current = FileStamp::read(&fs::metadata(path).map_err(|e| e.to_string())?)?;
            if *before != current {
                return Err(format!(
                    "evidence changed during verification: {}",
                    path.display()
                ));
            }
        }
        Ok(())
    }
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
    let mut checked = Verification::default();
    let value = load_verified(path, &mut checked)?;
    checked.finish()?;
    Ok(value)
}

pub fn load_verified<T: DeserializeOwned + Payload>(
    path: &Path,
    checked: &mut Verification,
) -> Result<Artifact<T>, String> {
    let file = fs::File::open(path).map_err(|e| format!("{}: {e}", path.display()))?;
    const MAX_JSON: u64 = 64 * 1024 * 1024;
    if file.metadata().map_err(|e| e.to_string())?.len() > MAX_JSON {
        return Err("artifact exceeds 64 MiB".into());
    }
    let mut bytes = Vec::new();
    file.take(MAX_JSON + 1)
        .read_to_end(&mut bytes)
        .map_err(|e| e.to_string())?;
    if bytes.len() as u64 > MAX_JSON {
        return Err("artifact exceeds 64 MiB".into());
    }
    checked.verify(path, &hash_bytes(&bytes))?;
    let value: Artifact<T> =
        serde_json::from_slice(&bytes).map_err(|e| format!("{}: {e}", path.display()))?;
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
        checked,
        &mut std::collections::BTreeSet::new(),
        0,
    )?;
    Ok(value)
}

fn verify_refs(
    path: &Path,
    refs: &[Reference],
    checked: &mut Verification,
    traversed: &mut std::collections::BTreeSet<std::path::PathBuf>,
    depth: usize,
) -> Result<(), String> {
    if depth > 32 {
        return Err("evidence reference depth exceeds 32".into());
    }
    for input in refs {
        let alias = path.parent().unwrap_or(Path::new(".")).join(&input.path);
        let target = alias.canonicalize().map_err(|e| e.to_string())?;
        checked.verify(&alias, &input.sha256)?;
        if !traversed.insert(target.clone()) {
            continue;
        }
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
        verify_refs(&target, &refs, checked, traversed, depth + 1)?;
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
    fn verification_rejects_drift_and_conflicting_digest() {
        let root = std::env::temp_dir().join(format!("ds4-verified-{}", std::process::id()));
        directory(&root).unwrap();
        let path = root.join("input");
        fs::write(&path, b"first").unwrap();
        let expected = hash(&path).unwrap();
        let mut checked = Verification::default();
        checked.verify(&path, &expected).unwrap();
        checked.verify(&path, &expected).unwrap();
        assert_eq!(checked.full_reads, 1);
        assert!(checked.verify(&path, &hash_bytes(b"other")).is_err());
        fs::write(&path, b"other").unwrap();
        assert!(checked.verify(&path, &expected).is_err());
        assert!(checked.finish().is_err());

        let mut replaced = Verification::default();
        replaced.verify(&path, &hash_bytes(b"other")).unwrap();
        let replacement = root.join("replacement");
        fs::write(&replacement, b"other").unwrap();
        fs::rename(replacement, &path).unwrap();
        assert!(
            replaced.finish().is_err(),
            "same bytes on another inode must fail"
        );
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn operation_reuses_shared_inputs_across_artifacts() {
        let root = std::env::temp_dir().join(format!("ds4-shared-{}", std::process::id()));
        directory(&root).unwrap();
        let input = root.join("weights");
        fs::write(&input, b"same input").unwrap();
        for index in 0..2 {
            let mut evidence = Artifact::new(Example { value: index });
            evidence.inputs.push(reference(&input, &root).unwrap());
            save(&root.join(format!("{index}.json")), &evidence).unwrap();
        }
        let mut checked = Verification::default();
        for index in 0..2 {
            load_verified::<Example>(&root.join(format!("{index}.json")), &mut checked).unwrap();
        }
        assert_eq!(checked.full_reads, 3, "two envelopes and one shared input");
        checked.finish().unwrap();
        let envelope = root.join("0.json");
        let bytes = fs::read(&envelope).unwrap();
        fs::write(&envelope, [bytes.as_slice(), b"\n"].concat()).unwrap();
        assert!(
            checked.finish().is_err(),
            "root envelope must remain pinned"
        );
        fs::write(&input, b"new input!").unwrap();
        assert!(load_verified::<Example>(&root.join("1.json"), &mut checked).is_err());
        assert!(checked.finish().is_err());
        fs::remove_dir_all(root).unwrap();
    }

    #[cfg(unix)]
    #[test]
    fn verification_rejects_retargeted_alias() {
        let root = std::env::temp_dir().join(format!("ds4-alias-{}", std::process::id()));
        directory(&root).unwrap();
        fs::write(root.join("a"), b"same").unwrap();
        fs::write(root.join("b"), b"same").unwrap();
        let alias = root.join("alias");
        std::os::unix::fs::symlink(root.join("a"), &alias).unwrap();
        let mut checked = Verification::default();
        checked.verify(&alias, &hash_bytes(b"same")).unwrap();
        fs::remove_file(&alias).unwrap();
        std::os::unix::fs::symlink(root.join("b"), &alias).unwrap();
        assert!(checked.finish().is_err());
        assert!(checked.verify(&alias, &hash_bytes(b"same")).is_err());
        fs::remove_dir_all(root).unwrap();
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
