use ds4_core::{Backend, Model, ModelFamily, ModelOpenOption, TokenBuffer};
use serde_json::{json, Value};
use std::time::Instant;
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let a: Vec<_> = std::env::args().skip(1).collect();
    assert_eq!(a.len(), 5, "MODEL FIXTURE mtp2|plain LIMIT OUTPUT");
    let mut fixture: Value = serde_json::from_slice(&std::fs::read(&a[1])?)?;
    let mode = a[2].as_str();
    assert!(matches!(mode, "mtp2" | "plain"));
    let drafts = if mode == "mtp2" { 2 } else { 0 };
    let limit: usize = a[3].parse()?;
    assert!(limit > 0 && limit <= 128);
    let model = if mode == "mtp2" {
        Model::open_with_support_options(&a[0], Backend::Cuda, 8, true, None, None,
            &[ModelOpenOption::MtpDraftTokens(drafts)])?
    } else { Model::open(&a[0], Backend::Cuda, 8, false)? };
    assert_eq!(model.family(), ModelFamily::Ling31Vq);
    let vocab = model.vocab().n_vocab() as usize;
    let eos = model.token_eos();
    let selected = ["held_ko", "held_en", "held_zh", "held_code", "held_math", "held_json",
        "held_ko_summary", "held_ko_free", "held_tool_call", "held_tool_followup"];
    let corpus = model.tokenize_text(&std::fs::read_to_string("speed-bench/promessi_sposi.txt")?)?;
    assert!(corpus.len() >= 8192);
    fixture["cases"].as_array_mut().unwrap().insert(0,
        json!({"id":"corpus_8k", "prompt_ids": &corpus.as_slice()[..8192]}));
    let mut results = Vec::new();
    let mut warmed = false;
    for case in fixture["cases"].as_array().ok_or("cases missing")? {
        let id = case["id"].as_str().ok_or("id missing")?;
        let force = id == "corpus_8k";
        if !force && !selected.contains(&id) { continue; }
        let context = if force { 8321 } else { 4096 };
        let ids: Vec<i32> = case["prompt_ids"].as_array().ok_or("prompt missing")?
            .iter().map(|v| i32::try_from(v.as_i64().unwrap()).unwrap()).collect();
        assert!(ids.iter().all(|&v| v >= 0 && (v as usize) < vocab));
        let prompt = TokenBuffer::from_tokens(ids.clone());
        if !warmed {
            let mut warm = model.session(context)?;
            warm.sync(&prompt)?;
            for _ in 0..8 {
                let token = warm.argmax();
                if mode == "mtp2" { warm.eval_speculative_argmax(token, 3, eos)?; }
                else { warm.eval(token)?; }
            }
            drop(warm);
            warmed = true;
        }
        let mut session = model.session(context)?;
        let prefill = Instant::now();
        session.sync(&prompt)?;
        let prefill_seconds = prefill.elapsed().as_secs_f64();
        let frontier = session.copy_logits(vocab)?;
        let mut generated = Vec::new();
        let mut cycles = Vec::new();
        let start = Instant::now();
        while generated.len() < limit {
            let token = if force { session.argmax_excluding(eos) } else { session.argmax() };
            if !force && model.token_is_stop(token) { break; }
            let accepted = if mode == "mtp2" {
                session.eval_speculative_argmax(token, (limit - generated.len()) as i32, eos)?
            } else { session.eval(token)?; vec![token] };
            assert!(!accepted.is_empty());
            cycles.push((accepted.len(), accepted.iter().filter(|&&t| t != eos).count()));
            generated.extend(accepted.into_iter().filter(|&t| !force || t != eos));
        }
        let seconds = start.elapsed().as_secs_f64();
        let end = session.copy_logits(vocab)?;
        assert!(frontier.iter().chain(&end).all(|x| x.is_finite()));
        let r = json!({"id":id,"prompt_ids":ids,"tokens":generated,"cycles":cycles,
            "frontier_logits":frontier,"end_logits":end,"seconds":seconds,"prefill_seconds":prefill_seconds});
        println!("{}",json!({"id":id,"tokens":generated.len(),"seconds":seconds,"calls":cycles.len()}));
        results.push(r);
    }
    assert_eq!(results.len(), selected.len()+1);
    let tokens: usize = results.iter().filter(|v| v["id"] != "corpus_8k").map(|v| v["tokens"].as_array().unwrap().len()).sum();
    let seconds: f64 = results.iter().filter(|v| v["id"] != "corpus_8k").map(|v| v["seconds"].as_f64().unwrap()).sum();
    let result = json!({"mode":mode,"drafts":drafts,"limit":limit,"tokens":tokens,"seconds":seconds,
        "decode_tps":tokens as f64/seconds,"cases":results});
    std::fs::write(&a[4], serde_json::to_vec(&result)?)?;
    println!("{}",json!({"tokens":tokens,"seconds":seconds,"decode_tps":tokens as f64/seconds}));
    Ok(())
}
