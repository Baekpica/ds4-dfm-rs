//! Emit a family's layout table from a runtime pack: the `runtime/layout.json`
//! the converter writes beside the parts (`vq-gguf runtime`). The table is the
//! binding contract -- one entry per stored tensor with the artifact's own
//! name, type word and dimensions -- so it is generated from the artifact, not
//! transcribed from any other family's layout.
//!
//! usage: layout_from_pack <runtime/layout.json> > src/<family>/layout.rs
//! then `cargo fmt -p ds4-core` (the emitted one-line entries are wrapped by
//! the repo's rustfmt settings).
//!
//! The pack's dimensions are already in GGUF order (innermost first), which is
//! the order `ds4_core::LayoutSpec` compares against the tensor directory, so
//! they are emitted verbatim. Nothing here is family-specific: the module name
//! and the entries all come from the pack.
use std::io::Write;

use serde_json::Value;

type Fallible<T> = std::result::Result<T, Box<dyn std::error::Error>>;

/// The stored type words a VQ release may carry, in the order the generated
/// `mod ty` lists them.
const TYPES: &[(u32, &str)] = &[
    (0, "F32"),
    (8, "Q8_0"),
    (12, "Q4_K"),
    (30, "BF16"),
    (0x4756_4342, "GVBC"),
    (0x4756_5138, "GVQ8"),
    (0x4756_5238, "GVR8"),
];

fn error(message: impl Into<String>) -> Box<dyn std::error::Error> {
    message.into().into()
}

fn text<'a>(value: &'a Value, key: &str) -> Fallible<&'a str> {
    value
        .get(key)
        .and_then(Value::as_str)
        .ok_or_else(|| format!("missing string `{key}`").into())
}

fn number(value: &Value, key: &str) -> Fallible<u64> {
    value
        .get(key)
        .and_then(Value::as_u64)
        .ok_or_else(|| format!("missing integer `{key}`").into())
}

fn type_name(typ: u32) -> Fallible<&'static str> {
    TYPES
        .iter()
        .find(|(word, _)| *word == typ)
        .map(|(_, name)| *name)
        .ok_or_else(|| format!("unnamed stored type word 0x{typ:08x}: add it to TYPES").into())
}

/// One tensor of the table: its name, type word name and GGUF-order dims.
struct Entry {
    name: String,
    typ: &'static str,
    dims: Vec<u64>,
}

fn main() -> Fallible<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let [pack] = args.as_slice() else {
        return Err(error("usage: layout_from_pack <runtime/layout.json>"));
    };
    let pack: Value = serde_json::from_slice(&std::fs::read(pack)?)?;
    if text(&pack, "schema")? != "vq-layout/1" {
        return Err(error("unknown layout schema"));
    }
    let parts = pack
        .get("parts")
        .and_then(Value::as_array)
        .ok_or_else(|| error("missing `parts`"))?;

    // The embedding part holds the one table the head ties to; the trunk part
    // holds everything the device copies. Each part's entries keep the pack's
    // order, and a comment marks the part boundary so a diff reads part by
    // part.
    let mut sections: Vec<(String, Vec<Entry>)> = Vec::new();
    let mut embedding = None;
    let mut total = 0u64;
    for part in parts {
        let role = text(part, "role")?;
        let path = text(part, "path")?;
        let tensors = part
            .get("tensors")
            .and_then(Value::as_array)
            .ok_or_else(|| error("missing `tensors`"))?;
        let mut entries = Vec::with_capacity(tensors.len());
        for tensor in tensors {
            let name = text(tensor, "name")?;
            let typ = type_name(u32::try_from(number(tensor, "type")?)?)?;
            let dims = tensor
                .get("dims")
                .and_then(Value::as_array)
                .ok_or_else(|| error("missing `dims`"))?
                .iter()
                .map(|d| d.as_u64().ok_or_else(|| error("non-integer dim")))
                .collect::<Fallible<Vec<u64>>>()?;
            if dims.is_empty() || dims.len() > 8 {
                return Err(error(format!("{name}: unsupported rank {}", dims.len())));
            }
            if role == "embedding"
                && (tensors.len() != 1 || embedding.replace(name.to_owned()).is_some())
            {
                return Err(error("the embedding part must hold exactly one tensor"));
            }
            total += 1;
            entries.push(Entry {
                name: name.to_owned(),
                typ,
                dims,
            });
        }
        sections.push((format!("{role} part: {path}"), entries));
    }
    let embedding = embedding.ok_or_else(|| error("no embedding part in the pack"))?;
    // The pack carries the release constants too; the tensor total is a cheap
    // cross-check that the table is complete.
    if let Some(want) = pack
        .get("release")
        .and_then(|r| r.get("tensors"))
        .and_then(Value::as_u64)
    {
        if want != total {
            return Err(error(format!(
                "pack names {want} tensors but lists {total}"
            )));
        }
    }

    let mut out = std::io::stdout().lock();
    writeln!(
        out,
        "//! Generated from the release's runtime pack (`vq-gguf runtime`): every\n\
         //! stored tensor with the artifact's own name, type word and dimensions.\n\
         //! Do not edit by hand; regenerate with\n\
         //!   cargo run -q -p ds4-core --example layout_from_pack -- <runtime/layout.json>\n\
         //! The dimensions are GGUF order (innermost first), verbatim from the pack.\n\
         use crate::{{LayoutSpec, TypeClass}};\n\n\
         /// The stored type words this table uses.\n\
         mod ty {{"
    )?;
    for (word, name) in TYPES {
        if sections
            .iter()
            .flat_map(|(_, entries)| entries)
            .any(|e| e.typ == *name)
        {
            if *word <= u32::from(u8::MAX) {
                writeln!(out, "    pub(super) const {name}: u32 = {word};")?;
            } else {
                // The VQ container words read as ASCII tags in hex.
                writeln!(
                    out,
                    "    pub(super) const {name}: u32 = 0x{:04x}_{:04x};",
                    word >> 16,
                    word & 0xffff
                )?;
            }
        }
    }
    writeln!(
        out,
        "}}\n\n\
         /// The token table the head ties to: the artifact carries no `lm_head`.\n\
         pub const EMBEDDING: &str = \"{embedding}\";\n\n\
         fn spec(out: &mut Vec<LayoutSpec>, name: &str, typ: u32, dims: &[u64]) {{\n\
         \x20   let mut dim = [0u64; 8];\n\
         \x20   dim[..dims.len()].copy_from_slice(dims);\n\
         \x20   out.push(LayoutSpec {{\n\
         \x20       name: name.to_string(),\n\
         \x20       class: TypeClass::Exact(typ),\n\
         \x20       ndim: dims.len() as u32,\n\
         \x20       dim,\n\
         \x20   }});\n\
         }}\n\n\
         /// One entry per stored tensor, in pack order: the embedding part\n\
         /// first, the resident trunk second.\n\
         pub(super) fn layouts() -> Vec<LayoutSpec> {{\n\
         \x20   let mut out = Vec::with_capacity({total});"
    )?;
    for (header, entries) in &sections {
        writeln!(out, "    // {header}")?;
        for entry in entries {
            let dims: Vec<String> = entry.dims.iter().map(u64::to_string).collect();
            writeln!(
                out,
                "    spec(&mut out, \"{}\", ty::{}, &[{}]);",
                entry.name,
                entry.typ,
                dims.join(", ")
            )?;
        }
    }
    writeln!(out, "    out\n}}")?;
    Ok(())
}
