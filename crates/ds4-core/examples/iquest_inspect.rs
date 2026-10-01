//! Model-free artifact gate. It mmaps the GGUF; it never allocates a GPU model.
use ds4_core::{
    identify_gguf, validate_file, validate_layouts, BindPlan, GgufFile, IQuestCache, IQuestPlan,
    ModelFamily, TensorInventory, Vocab,
};
use std::path::Path;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let file = std::env::args()
        .nth(1)
        .ok_or("usage: iquest_inspect MODEL.gguf [prefill_chunk]")?;
    let chunk: u32 = std::env::args()
        .nth(2)
        .unwrap_or_else(|| "1024".into())
        .parse()?;
    let path = Path::new(&file);
    let identified = identify_gguf(path)?;
    if identified.shape.family != ModelFamily::IQuestQ1 {
        return Err("expected iquest_q1 architecture".into());
    }
    let gguf = GgufFile::open(path)?;
    validate_file(&gguf, &identified.shape)?;
    let inventory = TensorInventory::open(path)?;
    let binding = BindPlan::resolve(identified.shape, &inventory);
    validate_layouts(&binding)?;
    let vocab = Vocab::load(&gguf, identified.shape.family)?;
    let text = "Hello, IQuest! 中文かなカナ123456\n\t_indented code => value";
    let tokens = vocab.encode_text(text);
    let eos = vocab.eos_id == 0;
    println!(
        "{}",
        serde_json::json!({
            "architecture": "iquest_q1", "tensor_count": inventory.tensors.len(),
            "contract_valid": true, "full_layers": IQuestPlan::full_layers(),
            "q8_0_kv_bytes_512k": IQuestPlan::kv_bytes(524288, chunk, IQuestCache::Target),
            "q8_0_kv_bytes_512k_with_mtp": IQuestPlan::kv_bytes(524288, chunk, IQuestCache::WithMtp),
            "prefill_chunk": chunk, "tokenizer_sample": text, "tokenizer_ids": tokens,
            "eos_zero": eos, "serving_qualified": false,
        })
    );
    Ok(())
}
