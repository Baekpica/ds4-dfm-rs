//! Metadata-only prompt inspection; no model/session or GPU initialization.
//! Usage: cargo run -p ds4-core --example chat_template -- MODEL.gguf CONTEXT.json

use ds4_core::{chat_template::Template, identify_gguf, Vocab};
use std::path::Path;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let model = args.next().ok_or("expected MODEL.gguf CONTEXT.json")?;
    let context = args.next().ok_or("expected CONTEXT.json")?;
    let path = Path::new(&model);
    let template = Template::from_model(path)?
        .ok_or("no official Jinja; encoder exception or missing artifact")?;
    let input: serde_json::Value = serde_json::from_slice(&std::fs::read(context)?)?;
    let prompt = template.render(&input)?;
    let identified = identify_gguf(path)?;
    let vocab = Vocab::load_path(path, identified.shape.family)?;
    let tokens = vocab.encode_rendered_chat(&prompt);
    println!(
        "{}",
        serde_json::json!({"source": template.source(), "prompt": prompt, "token_ids": tokens})
    );
    Ok(())
}
