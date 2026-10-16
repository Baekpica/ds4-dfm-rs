use super::*;
use serde_json::{json, Value};

const REVISION: &str = "91b69215804f2d2036a8867f6ce44848e278dcda";

struct Release {
    root: PathBuf,
    files: Value,
    manifest: Value,
}

fn sha(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

impl Release {
    fn new() -> Self {
        let root = Temp::new().0.with_extension("release");
        fs::create_dir(&root).unwrap();
        let mut files = Vec::new();
        let mut entries = Vec::new();
        for (part, name) in [("native-a.gguf", "norm_a"), ("native-b.gguf", "norm_b")] {
            let tensor = Tensor {
                name,
                dims: vec![2],
                typ: 30,
                bytes: vec![0x80, 0x3f, 0x00, 0x40],
            };
            let (bytes, _) = container(&[("glm_vq.source_revision", REVISION)], &[tensor]);
            fs::write(root.join(part), &bytes).unwrap();
            files.push(json!({"path":part,"bytes":bytes.len(),"sha256":sha(&bytes)}));
            entries.push(json!({"part":part,"name":name,"shape":[2],"stored_shape":[2],"type_id":30,"payload_bytes":4,"payload_sha256":sha(&[0x80,0x3f,0x00,0x40]),"finite":true}));
        }
        let mut this = Self {
            root,
            files: json!({"files":files}),
            manifest: json!({"format_version":1,"source_revision":REVISION,"output_tensor_count":entries.len(),"tensors":entries}),
        };
        this.write();
        this
    }
    fn write(&mut self) {
        let bytes = serde_json::to_vec(&self.manifest).unwrap();
        fs::write(self.root.join("tensor-manifest.json"), &bytes).unwrap();
        let files = self.files["files"].as_array_mut().unwrap();
        files.retain(|e| e["path"] != "tensor-manifest.json");
        files.push(json!({"path":"tensor-manifest.json","bytes":bytes.len(),"sha256":sha(&bytes)}));
        fs::write(
            self.root.join("files.json"),
            serde_json::to_vec(&self.files).unwrap(),
        )
        .unwrap();
    }
}

impl Drop for Release {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.root);
    }
}

#[test]
fn named_parts_keep_mmap_offsets() {
    let release = Release::new();
    let inv = TensorInventory::open(&release.root).unwrap();
    assert_eq!(inv.shards.len(), 2);
    assert_eq!(inv.tensors.len(), 2);
    assert_eq!(inv.shards[0].base, 0);
    assert_eq!(inv.shards[1].base % inv.page, 0);
    for tensor in &inv.tensors {
        let shard = &inv.shards[tensor.shard as usize];
        let bytes = fs::read(&shard.path).unwrap();
        let off = (tensor.abs_offset - shard.base) as usize;
        assert_eq!(&bytes[off..off + 4], &[0x80, 0x3f, 0x00, 0x40]);
    }
}

#[test]
fn reverify_rewritten_release() {
    // Aged entries let the first verified open be reused by the second.
    let release = Release::new();
    let old = std::time::SystemTime::now() - std::time::Duration::from_secs(60);
    for entry in fs::read_dir(&release.root).unwrap() {
        let file = fs::File::options()
            .write(true)
            .open(entry.unwrap().path())
            .unwrap();
        file.set_modified(old).unwrap();
    }
    let first = TensorInventory::open(&release.root).unwrap();
    let again = TensorInventory::open(&release.root).unwrap();
    assert_eq!(format!("{first:?}"), format!("{again:?}"));
    // A rewrite after caching gets a new mtime and is verified again.
    let part = release.root.join("native-a.gguf");
    let mut bytes = fs::read(&part).unwrap();
    *bytes.last_mut().unwrap() ^= 1;
    fs::write(&part, bytes).unwrap();
    assert!(TensorInventory::open(&release.root).is_err());
}

#[test]
fn reject_missing_extra_parts() {
    let release = Release::new();
    let part = release.root.join("native-b.gguf");
    let bytes = fs::read(&part).unwrap();
    fs::remove_file(&part).unwrap();
    assert!(TensorInventory::open(&release.root).is_err());
    fs::write(&part, &bytes).unwrap();
    fs::write(release.root.join("unexpected.gguf"), &bytes).unwrap();
    assert!(TensorInventory::open(&release.root).is_err());
}

#[test]
fn reject_inconsistent_inventory() {
    let mut release = Release::new();
    let original = release.manifest.clone();
    for (field, value) in [
        ("name", json!("norm_a")),
        ("name", json!("unknown")),
        ("part", json!("../outside.gguf")),
        ("stored_shape", json!([1, 2])),
        ("shape", json!([1, 2])),
        ("payload_bytes", json!(2)),
        ("type_id", json!(1)),
        ("payload_sha256", json!("0".repeat(64))),
    ] {
        release.manifest = original.clone();
        release.manifest["tensors"][1][field] = value;
        release.write();
        assert!(
            TensorInventory::open(&release.root).is_err(),
            "accepted {field}"
        );
    }
    release.manifest = original;
    release.manifest["source_revision"] = json!("0".repeat(40));
    release.write();
    assert!(TensorInventory::open(&release.root).is_err());
}

#[test]
fn reject_changed_file_bytes() {
    let release = Release::new();
    let part = release.root.join("native-a.gguf");
    let mut bytes = fs::read(&part).unwrap();
    *bytes.last_mut().unwrap() ^= 1;
    fs::write(part, bytes).unwrap();
    assert!(TensorInventory::open(&release.root).is_err());
    let release = Release::new();
    let path = release.root.join("tensor-manifest.json");
    let mut bytes = fs::read(&path).unwrap();
    bytes.push(b' ');
    fs::write(path, bytes).unwrap();
    assert!(TensorInventory::open(&release.root).is_err());
}

#[test]
#[ignore = "requires the pinned 56-part local artifact; no model download"]
fn full_release_inventory() {
    let root = PathBuf::from(std::env::var_os("DS4_GLM_VQ_MODEL").expect("DS4_GLM_VQ_MODEL"));
    let inv = TensorInventory::open(&root).unwrap();
    assert_eq!(inv.shards.len(), 56);
    assert_eq!(inv.tensors.len(), 39862);
    let count = |typ| inv.tensors.iter().filter(|t| t.typ == typ).count();
    assert_eq!(count(VQ), 36288);
    assert_eq!(count(BOOK), 42);
    assert_eq!(count(FP8), 1050);
    println!(
        "GVQ1_RELEASE_PASS parts={} tensors={} file_bytes={} payload_bytes={}",
        inv.shards.len(),
        inv.tensors.len(),
        inv.shards.iter().map(|s| s.size).sum::<u64>(),
        inv.tensors.iter().map(|t| t.bytes).sum::<u64>()
    );
}
