//! The artifact contract this family opens: `vq-release.json` plus the
//! containers it names.
//!
//! The release is written by `vq-family release` in the VQ tools repository
//! (`schema vq-release/1`) and lists one entry per tensor the builder wrote:
//! its part, role, shape, container geometry, byte count and sha256, plus the
//! totals and a fingerprint over the indexed set. This module reads that file
//! and checks it against the bytes on open, without hashing the whole
//! artifact: each file's exact size and every container header are verified,
//! and [`Release::verify`] re-reads the digests when a gate asks for them.
//!
//! Container layout, read first-hand from the writer's reader in
//! `/data/VQ_tools/crates/vq-format/src/lib.rs:1-30,483-560`:
//!
//!     RQVQ: [u8 'RQVQ'][u16 dim][u16 nc][u16 stages][u16 groups][u32 rows]
//!           [u32 cols][u32 codebook format][u64 payload bytes]
//!           payloads, stage-major then group-minor
//!     DQVQ: [u8 'DQVQ'][u16 dim][u16 nc][u32 rows][u32 cols]
//!           [codebook nc*dim][row gains rows*2][indices]
//!     DQV3: the same fields, a 24-byte header, a 12-bit main stream and an
//!           optional 1-bit plane above 12 bits
//!     RQVF: [u8 'RQVF'][u32 rows][u32 cols][rows*cols f32]
//!
//! The payload size arithmetic below is the writer's own
//! (`vq-format/src/lib.rs:520-560`): a container whose declared payload does
//! not account for every byte is refused rather than truncated or padded.

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};

use serde_json::Value;

use super::Error;

use sha2::{Digest, Sha256};

/// The release file's name inside the artifact directory.
pub const FILE: &str = "vq-release.json";
/// The only schema this reader accepts.
pub const SCHEMA: &str = "vq-release/1";

const RQVQ: &[u8; 4] = b"RQVQ";
const DQVQ: &[u8; 4] = b"DQVQ";
const DQV3: &[u8; 4] = b"DQV3";
const RQVF: &[u8; 4] = b"RQVF";
const CB_F16: u32 = 0;
const CB_E4M3: u32 = 0x334d_3445;

/// Which residual payload layout a container carries.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Payload {
    V2,
    V3,
}

/// Which codebook element a container stores.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Codebook {
    F16,
    E4M3,
}

/// One tensor of the release.
#[derive(Clone, Debug)]
pub struct Entry {
    pub name: String,
    pub part: String,
    pub role: String,
    pub encoded: bool,
    pub rows: usize,
    pub cols: usize,
    /// Container geometry; `None` for a copied f32 sidecar.
    pub dim: Option<usize>,
    pub nc: Option<usize>,
    pub stages: Option<usize>,
    pub groups: Option<usize>,
    pub codebook: Option<Codebook>,
    pub payload: Option<Payload>,
    pub bytes: u64,
    pub sha256: String,
}

impl Entry {
    /// Is this tensor a matrix the runtime must dequantise?
    pub fn is_matrix(&self) -> bool {
        self.encoded
    }
}

/// The artifact totals the release states.
#[derive(Clone, Debug)]
pub struct Totals {
    pub entries: u64,
    pub encoded: u64,
    pub copied: u64,
    pub bytes: u64,
    pub encoded_bytes: u64,
    pub weights: u64,
    pub parameters: u64,
    pub fingerprint: String,
    pub manifest_sha256: String,
}

/// A parsed and byte-checked release.
#[derive(Clone, Debug)]
pub struct Release {
    pub dir: PathBuf,
    pub totals: Totals,
    entries: Vec<Entry>,
    index: HashMap<String, usize>,
}

fn text<'a>(value: &'a Value, key: &str, what: &str) -> Result<&'a str, Error> {
    value
        .get(key)
        .and_then(Value::as_str)
        .ok_or_else(|| Error::new(format!("{what}: {key} is not a string")))
}

fn number(value: &Value, key: &str, what: &str) -> Result<u64, Error> {
    value
        .get(key)
        .and_then(Value::as_u64)
        .ok_or_else(|| Error::new(format!("{what}: {key} is not an integer")))
}

fn optional_number(value: &Value, key: &str, what: &str) -> Result<Option<u64>, Error> {
    match value.get(key) {
        None | Some(Value::Null) => Ok(None),
        Some(v) => v
            .as_u64()
            .map(Some)
            .ok_or_else(|| Error::new(format!("{what}: {key} is not an integer"))),
    }
}

fn u16le(b: &[u8], at: usize) -> usize {
    u16::from_le_bytes([b[at], b[at + 1]]) as usize
}

fn u32le(b: &[u8], at: usize) -> u32 {
    u32::from_le_bytes([b[at], b[at + 1], b[at + 2], b[at + 3]])
}

fn u64le(b: &[u8], at: usize) -> u64 {
    u64::from_le_bytes([
        b[at],
        b[at + 1],
        b[at + 2],
        b[at + 3],
        b[at + 4],
        b[at + 5],
        b[at + 6],
        b[at + 7],
    ])
}

fn codebook_bytes(format: u32, what: &str) -> Result<usize, Error> {
    match format {
        CB_F16 => Ok(2),
        CB_E4M3 => Ok(1),
        other => Err(Error::new(format!(
            "{what}: unknown codebook format {other:#010x}"
        ))),
    }
}

fn index_bits(nc: usize) -> usize {
    let mut nbit = 1usize;
    while (1usize << nbit) < nc {
        nbit += 1;
    }
    nbit
}

/// The per-payload byte count the writer's reader computes, so a container
/// that would be cut short or overrun is refused.
fn payload_bytes(
    dim: usize,
    nc: usize,
    rows: usize,
    cols: usize,
    groups: usize,
    cb: usize,
    v3: bool,
) -> Result<usize, Error> {
    if dim == 0 || cols % dim != 0 || groups == 0 || rows % groups != 0 {
        return Err(Error::new(format!(
            "container shape {rows}x{cols} dim {dim} groups {groups} is inconsistent"
        )));
    }
    let group_rows = rows / groups;
    let nidx_row = cols / dim;
    let nbit = index_bits(nc);
    let per = if v3 {
        if nbit > 13 {
            return Err(Error::new(format!(
                "DQV3 carries 13 index bits; nc {nc} needs {nbit}"
            )));
        }
        let main = nbit.min(12);
        let mrow = (nidx_row * main + 7) / 8;
        let prow = if nbit > 12 { (nidx_row + 7) / 8 } else { 0 };
        24 + nc * dim * cb + group_rows * 2 + group_rows * (mrow + prow)
    } else {
        let nidx = group_rows * nidx_row;
        let idx = if nbit == 8 {
            nidx
        } else {
            (nidx * nbit + 7) / 8 + 1
        };
        16 + nc * dim * cb + group_rows * 2 + idx
    };
    Ok(per)
}

fn hex64(text: &str) -> bool {
    text.len() == 64 && text.bytes().all(|c| c.is_ascii_hexdigit())
}

/// Read and byte-check `<dir>/vq-release.json`.
pub fn open(dir: &Path) -> Result<Release, Error> {
    let path = dir.join(FILE);
    let bytes = std::fs::read(&path).map_err(|e| Error::new(format!("{}: {e}", path.display())))?;
    let root: Value = serde_json::from_slice(&bytes)
        .map_err(|e| Error::new(format!("{}: {e}", path.display())))?;
    if text(&root, "schema", "release")? != SCHEMA {
        return Err(Error::new(format!(
            "{}: schema is not {SCHEMA}",
            path.display()
        )));
    }
    let artifact = root
        .get("artifact")
        .ok_or_else(|| Error::new("release: no artifact block"))?;
    let totals = Totals {
        entries: number(artifact, "entries", "release.artifact")?,
        encoded: number(artifact, "encoded", "release.artifact")?,
        copied: number(artifact, "copied", "release.artifact")?,
        bytes: number(artifact, "bytes", "release.artifact")?,
        encoded_bytes: number(artifact, "encoded_bytes", "release.artifact")?,
        weights: number(artifact, "weights", "release.artifact")?,
        parameters: number(artifact, "parameters", "release.artifact")?,
        fingerprint: text(artifact, "fingerprint", "release.artifact")?.to_string(),
        manifest_sha256: text(
            artifact
                .get("manifest")
                .ok_or_else(|| Error::new("release.artifact: no manifest block"))?,
            "sha256",
            "release.artifact.manifest",
        )?
        .to_string(),
    };
    if !hex64(&totals.fingerprint) || !hex64(&totals.manifest_sha256) {
        return Err(Error::new(
            "release: fingerprint and manifest sha256 must be 64 hex digits",
        ));
    }

    let listed = root
        .get("entries")
        .and_then(Value::as_array)
        .ok_or_else(|| Error::new("release: no entries array"))?;
    if listed.len() as u64 != totals.entries {
        return Err(Error::new(format!(
            "release: artifact says {} entries, the array has {}",
            totals.entries,
            listed.len()
        )));
    }

    let mut entries = Vec::with_capacity(listed.len());
    let mut seen = HashSet::with_capacity(listed.len());
    for (at, value) in listed.iter().enumerate() {
        let what = format!("release.entries[{at}]");
        let name = text(value, "name", &what)?.to_string();
        if name.is_empty() || name.contains(['/', '\\']) || name.contains("..") {
            return Err(Error::new(format!(
                "{what}: {:?} is not one file name",
                name
            )));
        }
        if !seen.insert(name.clone()) {
            return Err(Error::new(format!("release: duplicate entry {name}")));
        }
        let kind = text(value, "kind", &what)?;
        let encoded = match kind {
            "encoded" => true,
            "copied" => false,
            other => {
                return Err(Error::new(format!(
                    "{what}: kind {other:?} is neither encoded nor copied"
                )))
            }
        };
        let entry = Entry {
            name,
            part: text(value, "part", &what)?.to_string(),
            role: text(value, "role", &what)?.to_string(),
            encoded,
            rows: number(value, "rows", &what)? as usize,
            cols: number(value, "cols", &what)? as usize,
            dim: optional_number(value, "dim", &what)?.map(|v| v as usize),
            nc: optional_number(value, "nc", &what)?.map(|v| v as usize),
            stages: optional_number(value, "stages", &what)?.map(|v| v as usize),
            groups: optional_number(value, "groups", &what)?.map(|v| v as usize),
            codebook: match value.get("codebook").and_then(Value::as_str) {
                None => None,
                Some("f16") => Some(Codebook::F16),
                Some("e4m3") => Some(Codebook::E4M3),
                Some(other) => {
                    return Err(Error::new(format!(
                        "{what}: codebook {other:?} is neither f16 nor e4m3"
                    )))
                }
            },
            payload: None,
            bytes: number(value, "bytes", &what)?,
            sha256: text(value, "sha256", &what)?.to_string(),
        };
        if !hex64(&entry.sha256) {
            return Err(Error::new(format!(
                "{what} ({}): sha256 is not 64 hex digits",
                entry.name
            )));
        }
        entries.push(entry);
    }

    let mut release = Release {
        dir: dir.to_path_buf(),
        totals,
        entries,
        index: HashMap::new(),
    };
    release.check_totals()?;
    for at in 0..release.entries.len() {
        let path = dir.join(format!("{}.bin", release.entries[at].name));
        let size = std::fs::metadata(&path)
            .map_err(|e| Error::new(format!("{}: {e}", path.display())))?
            .len();
        let entry = &mut release.entries[at];
        if size != entry.bytes {
            return Err(Error::new(format!(
                "{}: file is {size} bytes, release says {}",
                path.display(),
                entry.bytes
            )));
        }
        if entry.encoded {
            check_container(&path, entry)?;
        } else {
            check_sidecar(&path, entry)?;
        }
    }
    for (at, entry) in release.entries.iter().enumerate() {
        release.index.insert(entry.name.clone(), at);
    }
    Ok(release)
}

fn check_container(path: &Path, entry: &mut Entry) -> Result<(), Error> {
    let mut head = [0u8; 36];
    let read = read_at(path, &mut head)?;
    if read < head.len() {
        return Err(Error::new(format!(
            "{}: shorter than an RQVQ header and one payload",
            path.display()
        )));
    }
    if &head[0..4] != RQVQ {
        return Err(Error::new(format!(
            "{}: not an RQVQ container",
            path.display()
        )));
    }
    let dim = u16le(&head, 4);
    let nc = u16le(&head, 6);
    let stages = u16le(&head, 8);
    let groups = match u16le(&head, 10) {
        0 | 1 => 1,
        g => g,
    };
    let rows = u32le(&head, 12) as usize;
    let cols = u32le(&head, 16) as usize;
    let format = u32le(&head, 20);
    let payload = u64le(&head, 24);
    let name = &entry.name;
    if rows != entry.rows || cols != entry.cols {
        return Err(Error::new(format!(
            "{name}: header {rows}x{cols}, release says {}x{}",
            entry.rows, entry.cols
        )));
    }
    for (field, header, recorded) in [
        ("dim", dim, entry.dim),
        ("nc", nc, entry.nc),
        ("stages", stages, entry.stages),
        ("groups", groups, entry.groups),
    ] {
        if recorded != Some(header) {
            return Err(Error::new(format!(
                "{name}: header {field} is {header}, release says {recorded:?}"
            )));
        }
    }
    let cb = codebook_bytes(format, name)?;
    let recorded = entry
        .codebook
        .ok_or_else(|| Error::new(format!("{name}: release records no codebook")))?;
    let expected = match cb {
        2 => Codebook::F16,
        _ => Codebook::E4M3,
    };
    if recorded != expected {
        return Err(Error::new(format!(
            "{name}: header codebook {expected:?}, release says {recorded:?}"
        )));
    }
    if dim == 0 || nc == 0 || stages == 0 || rows == 0 || cols == 0 {
        return Err(Error::new(format!("{name}: header states a zero field")));
    }
    let v3 = &head[32..36] == DQV3;
    if !v3 && &head[32..36] != DQVQ {
        return Err(Error::new(format!(
            "{name}: first payload is neither DQVQ nor DQV3"
        )));
    }
    let per = payload_bytes(dim, nc, rows, cols, groups, cb, v3)?;
    let expected = (stages * groups * per) as u64;
    if payload != expected {
        return Err(Error::new(format!(
            "{name}: header payload {payload} bytes, layout implies {expected}"
        )));
    }
    let want = 32 + payload;
    let size = std::fs::metadata(path)
        .map_err(|e| Error::new(format!("{}: {e}", path.display())))?
        .len();
    if size != want {
        return Err(Error::new(format!(
            "{name}: file is {size} bytes, header and payload are {want}"
        )));
    }
    entry.payload = Some(if v3 { Payload::V3 } else { Payload::V2 });
    Ok(())
}

fn check_sidecar(path: &Path, entry: &mut Entry) -> Result<(), Error> {
    let mut head = [0u8; 12];
    let read = read_at(path, &mut head)?;
    if read < head.len() || &head[0..4] != RQVF {
        return Err(Error::new(format!(
            "{}: not an RQVF sidecar",
            path.display()
        )));
    }
    let rows = u32le(&head, 4) as usize;
    let cols = u32le(&head, 8) as usize;
    if rows != entry.rows || cols != entry.cols {
        return Err(Error::new(format!(
            "{}: sidecar {rows}x{cols}, release says {}x{}",
            entry.name, entry.rows, entry.cols
        )));
    }
    if entry.dim.is_some()
        || entry.nc.is_some()
        || entry.stages.is_some()
        || entry.groups.is_some()
        || entry.codebook.is_some()
    {
        return Err(Error::new(format!(
            "{}: a copied tensor records a container geometry",
            entry.name
        )));
    }
    let want = 12 + rows * cols * 4;
    let size = std::fs::metadata(path)
        .map_err(|e| Error::new(format!("{}: {e}", path.display())))?
        .len() as usize;
    if size != want {
        return Err(Error::new(format!(
            "{}: sidecar is {size} bytes, shape needs {want}",
            entry.name
        )));
    }
    Ok(())
}

fn read_at(path: &Path, buf: &mut [u8]) -> Result<usize, Error> {
    use std::io::Read;
    let mut file =
        std::fs::File::open(path).map_err(|e| Error::new(format!("{}: {e}", path.display())))?;
    let mut done = 0usize;
    while done < buf.len() {
        let n = file
            .read(&mut buf[done..])
            .map_err(|e| Error::new(format!("{}: {e}", path.display())))?;
        if n == 0 {
            break;
        }
        done += n;
    }
    Ok(done)
}

impl Release {
    pub fn entries(&self) -> &[Entry] {
        &self.entries
    }

    pub fn get(&self, name: &str) -> Option<&Entry> {
        self.index.get(name).map(|at| &self.entries[*at])
    }

    fn check_totals(&self) -> Result<(), Error> {
        let encoded = self.entries.iter().filter(|e| e.encoded).count() as u64;
        let copied = self.entries.len() as u64 - encoded;
        let bytes: u64 = self.entries.iter().map(|e| e.bytes).sum();
        let encoded_bytes: u64 = self
            .entries
            .iter()
            .filter(|e| e.encoded)
            .map(|e| e.bytes)
            .sum();
        let weights: u64 = self
            .entries
            .iter()
            .filter(|e| e.encoded)
            .map(|e| e.rows as u64 * e.cols as u64)
            .sum();
        let parameters: u64 = self
            .entries
            .iter()
            .map(|e| e.rows as u64 * e.cols as u64)
            .sum();
        for (field, stated, derived) in [
            ("encoded", self.totals.encoded, encoded),
            ("copied", self.totals.copied, copied),
            ("bytes", self.totals.bytes, bytes),
            ("encoded_bytes", self.totals.encoded_bytes, encoded_bytes),
            ("weights", self.totals.weights, weights),
            ("parameters", self.totals.parameters, parameters),
        ] {
            if stated != derived {
                return Err(Error::new(format!(
                    "release.artifact.{field} says {stated}, the entries sum to {derived}"
                )));
            }
        }
        Ok(())
    }

    /// Re-read every file and compare its sha256 with the release. This is the
    /// gate path, not the open path: opening a 9.5 GB artifact must not hash
    /// 9.5 GB, but a release is only trusted after a verify.
    pub fn verify(&self) -> Result<(), Error> {
        for entry in &self.entries {
            let path = self.dir.join(format!("{}.bin", entry.name));
            let mut file = std::fs::File::open(&path)
                .map_err(|e| Error::new(format!("{}: {e}", path.display())))?;
            let mut hasher = Sha256::new();
            let mut buf = vec![0u8; 1 << 20];
            loop {
                use std::io::Read;
                let n = file
                    .read(&mut buf)
                    .map_err(|e| Error::new(format!("{}: {e}", path.display())))?;
                if n == 0 {
                    break;
                }
                hasher.update(&buf[..n]);
            }
            let digest = hasher.finalize();
            let mut hex = String::with_capacity(64);
            for byte in digest {
                hex.push_str(&format!("{byte:02x}"));
            }
            if hex != entry.sha256 {
                return Err(Error::new(format!(
                    "{}: sha256 is {hex}, release says {}",
                    entry.name, entry.sha256
                )));
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn container() -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(b"RQVQ");
        out.extend_from_slice(&2u16.to_le_bytes());
        out.extend_from_slice(&2u16.to_le_bytes());
        out.extend_from_slice(&1u16.to_le_bytes());
        out.extend_from_slice(&1u16.to_le_bytes());
        out.extend_from_slice(&2u32.to_le_bytes());
        out.extend_from_slice(&4u32.to_le_bytes());
        out.extend_from_slice(&0u32.to_le_bytes());
        out.extend_from_slice(&30u64.to_le_bytes());
        out.extend_from_slice(b"DQVQ");
        out.extend_from_slice(&2u16.to_le_bytes());
        out.extend_from_slice(&2u16.to_le_bytes());
        out.extend_from_slice(&2u32.to_le_bytes());
        out.extend_from_slice(&4u32.to_le_bytes());
        out.resize(out.len() + 8 + 4 + 2, 0);
        out
    }

    fn sidecar() -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(b"RQVF");
        out.extend_from_slice(&2u32.to_le_bytes());
        out.extend_from_slice(&3u32.to_le_bytes());
        out.extend_from_slice(&vec![0u8; 2 * 3 * 4]);
        out
    }

    fn release_value(
        encoded: u64,
        copied: u64,
        bytes: u64,
        weights: u64,
        parameters: u64,
    ) -> Value {
        serde_json::json!({
            "schema": SCHEMA,
            "artifact": {
                "entries": 2,
                "encoded": encoded,
                "copied": copied,
                "bytes": bytes,
                "encoded_bytes": 62,
                "weights": weights,
                "parameters": parameters,
                "fingerprint": "00".repeat(32),
                "manifest": {"sha256": "11".repeat(32)},
            },
            "entries": [
                {
                    "name": "a.encoded",
                    "part": "text",
                    "role": "q_proj",
                    "kind": "encoded",
                    "rows": 2,
                    "cols": 4,
                    "dim": 2,
                    "nc": 2,
                    "stages": 1,
                    "groups": 1,
                    "codebook": "f16",
                    "bytes": 62,
                    "sha256": "00".repeat(32),
                },
                {
                    "name": "b.copied",
                    "part": "text",
                    "role": "input_norm",
                    "kind": "copied",
                    "rows": 2,
                    "cols": 3,
                    "dim": null,
                    "nc": null,
                    "stages": null,
                    "groups": null,
                    "codebook": null,
                    "bytes": 36,
                    "sha256": "00".repeat(32),
                }
            ]
        })
    }

    fn artifact(tag: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "ds4-gemma4-{tag}-{}-{:?}",
            std::process::id(),
            std::thread::current().id()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("a.encoded.bin"), container()).unwrap();
        std::fs::write(dir.join("b.copied.bin"), sidecar()).unwrap();
        std::fs::write(
            dir.join(FILE),
            serde_json::to_vec(&release_value(1, 1, 98, 8, 14)).unwrap(),
        )
        .unwrap();
        dir
    }

    #[test]
    fn a_release_opens_and_binds_both_kinds() {
        let dir = artifact("open");
        let release = open(&dir).unwrap();
        assert_eq!(release.totals.entries, 2);
        let encoded = release.get("a.encoded").unwrap();
        assert_eq!(encoded.payload, Some(Payload::V2));
        assert_eq!(encoded.codebook, Some(Codebook::F16));
        let copied = release.get("b.copied").unwrap();
        assert!(!copied.is_matrix());
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn a_container_shape_disagreement_is_named() {
        let dir = artifact("shape");
        /* Keep the totals consistent with the mutated shape so the failure
         * under test is the header comparison, not the sum check. */
        let mut value = release_value(1, 1, 98, 16, 22);
        value["artifact"]["weights"] = serde_json::json!(16);
        value["artifact"]["parameters"] = serde_json::json!(22);
        value["entries"][0]["rows"] = serde_json::json!(4);
        std::fs::write(dir.join(FILE), serde_json::to_vec(&value).unwrap()).unwrap();
        let error = open(&dir).unwrap_err().to_string();
        assert!(error.contains("a.encoded"), "{error}");
        assert!(error.contains("release says 4x4"), "{error}");
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn a_wrong_totals_sum_is_refused() {
        let dir = artifact("totals");
        let mut value = release_value(1, 1, 98, 9, 14);
        value["artifact"]["weights"] = serde_json::json!(9);
        std::fs::write(dir.join(FILE), serde_json::to_vec(&value).unwrap()).unwrap();
        let error = open(&dir).unwrap_err().to_string();
        assert!(error.contains("weights says 9"), "{error}");
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn a_truncated_sidecar_is_refused() {
        let dir = artifact("trunc");
        std::fs::write(dir.join("b.copied.bin"), sidecar()[..20].to_vec()).unwrap();
        let error = open(&dir).unwrap_err().to_string();
        assert!(error.contains("b.copied"), "{error}");
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn verify_catches_a_digest_the_open_path_does_not_read() {
        let dir = artifact("digest");
        let release = open(&dir).unwrap();
        let error = release.verify().unwrap_err().to_string();
        assert!(error.contains("a.encoded"), "{error}");
        assert!(error.contains("sha256"), "{error}");
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
