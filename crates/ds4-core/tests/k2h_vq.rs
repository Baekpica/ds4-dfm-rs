//! K2-Horizon VQ release contract. Model-backed checks read only JSON assets
//! and GGUF parts (no GPU); point DS4_K2H_MODEL at the release directory.
use ds4_core::chat_template::{ChatOptions, RenderClock, Template};
use ds4_core::{identify_gguf, ChatThinkMode, ModelFamily, TensorInventory, Variant, Vocab};
use serde_json::{json, Value};
use std::path::PathBuf;

// release.json: 58 expert parts, the body and the embedding part.
const PARTS: usize = 60;
const TENSORS: usize = 34_134;
const K2_TEMPLATE: &str =
    include_str!("../../../tests/fixtures/chat-template/models/k2/chat_template.jinja");

fn root() -> PathBuf {
    PathBuf::from(std::env::var_os("DS4_K2H_MODEL").expect("DS4_K2H_MODEL"))
}

fn conversation() -> Vec<Value> {
    vec![
        json!({"role":"system","content":"Be concise. 한국어로 답해."}),
        json!({"role":"user","content":"  서울의 장소를 알려 줘.\n"}),
        json!({"role":"assistant","content":"서울역이 있어요.","reasoning_content":"장소를 고른다."}),
        json!({"role":"user","content":"하나 더."}),
    ]
}

fn tool_turns() -> (Vec<Value>, Vec<Value>) {
    let tool = json!({"type":"function","function":{"name":"lookup","description":"Find places.",
        "parameters":{"type":"object","properties":{"city":{"type":"string"},"limit":{"type":"integer"}},
        "required":["city"]}}});
    let messages = vec![
        json!({"role":"user","content":"Find a café in Seoul."}),
        json!({"role":"assistant","content":"","reasoning_content":"Use the tool.",
            "tool_calls":[{"id":"call_01","type":"function",
                "function":{"name":"lookup","arguments":{"city":"서울","limit":2}}}]}),
        json!({"role":"tool","tool_call_id":"call_01","content":"{\"items\":[\"café\"]}"}),
    ];
    (messages, vec![tool])
}

fn context(messages: &[Value], tools: &[Value], effort: &str, prompt: bool) -> Value {
    json!({"bos_token":"<|ifm|begin_of_text|>","messages":messages,"tools":tools,
        "add_generation_prompt":prompt,"reasoning_effort":effort})
}

#[test]
fn family_is_its_own_contract() {
    let shape = ds4_core::shape_for_variant(Variant::K2HorizonVq);
    let k2 = ds4_core::shape_for_variant(Variant::K2Horizon375B);
    assert_eq!(shape.family, ModelFamily::K2hVq);
    assert_eq!(shape.name, "K2-Horizon-375B-A23B-hVQ");
    assert_eq!(
        ModelFamily::from_oracle_name("k2_horizon_vq"),
        Some(ModelFamily::K2hVq)
    );
    // The same graph as the GGUF variant; only the family and contract differ.
    let mut same = shape;
    same.name = k2.name;
    same.family = k2.family;
    same.variant = k2.variant;
    assert_eq!(same, k2);
    let caps = ds4_core::caps_from_shape(shape);
    assert_eq!(caps.ctx_max, Some(524_288));
    assert_eq!(ds4_core::native_ctx_default(caps), Some(524_288));
}

#[test]
fn chat_options_keep_k2s_think_channel() {
    // The K2 output parser reads only <ifm|think>: every mode renders the
    // template's high effort, and off closes the span (the GGUF variant's rule).
    let template = Template::compile(K2_TEMPLATE, RenderClock::Fixed(0)).unwrap();
    let messages = [json!({"role":"user","content":"Hello"})];
    let render = |mode| {
        template
            .render_chat(
                &messages,
                &[],
                ChatOptions::new(Variant::K2HorizonVq as i32, mode),
            )
            .unwrap()
    };
    for mode in [ChatThinkMode::Low, ChatThinkMode::High, ChatThinkMode::Max] {
        assert!(render(mode).ends_with("<|ifm|im_start|>assistant\n<ifm|think>\n"));
    }
    assert!(render(ChatThinkMode::None).ends_with("<ifm|think>\n</ifm|think>\n"));
}

#[test]
#[ignore = "requires the release in DS4_K2H_MODEL; hashes every part, no GPU"]
fn release_opens_every_part() {
    let root = root();
    let id = identify_gguf(&root).unwrap();
    assert_eq!(id.shape.family, ModelFamily::K2hVq);
    assert_eq!(id.split_count as usize, PARTS);
    let inventory = TensorInventory::open(&root).unwrap();
    assert_eq!(inventory.shards.len(), PARTS);
    assert_eq!(inventory.tensors.len(), TENSORS);
    // Expert layers 3-60, the body, then the embedding part, whose table
    // native finds as the inventory's last tensor.
    assert!(inventory.shards[0].path.ends_with("layer-03-experts.gguf"));
    assert!(inventory.shards[PARTS - 2]
        .path
        .ends_with("body-nonexpert.gguf"));
    let last = inventory.tensors.last().unwrap();
    assert_eq!(last.name, "model.embed_tokens.weight");
    assert_eq!(last.shard as usize, PARTS - 1);
    ds4_core::validate_gguf(&root).unwrap();
    println!("K2H_RELEASE_PASS parts={PARTS} tensors={TENSORS}");
}

#[test]
#[ignore = "requires the release JSON assets in DS4_K2H_MODEL; no GPU"]
fn source_tokenizer_matches_the_reference_ids() {
    let vocab = Vocab::load_path(&root(), ModelFamily::K2hVq).unwrap();
    // generation_config.json eos_token_id [1, 250019]: <|ifm|endoftext|> and
    // <|ifm|im_end|>; the next turn's opener and think spans never stop.
    assert_eq!((vocab.bos_id, vocab.eos_id, vocab.eot_id), (0, 250_019, 1));
    let stops: Vec<i32> = (0..vocab.n_vocab()).filter(|&t| vocab.is_stop(t)).collect();
    assert_eq!(stops, [1, 250_019]);
    // tokenizers 0.22.2 on the pinned tokenizer.json, add_special_tokens=False.
    // NFC applies; marks and ZWJ continue punctuation runs; \p{No} are numbers.
    let cases: &[(&str, &[i32])] = &[
        ("hello", &[33785]),
        ("12345", &[4018, 927]),
        ("안녕하세요", &[76943, 47589, 245, 187709]),
        ("it's I'LL", &[286, 559, 373, 8, 932]),
        ("a\u{200d}b", &[66, 16727, 67]),
        ("Cafe\u{301} noir", &[147990, 497, 70603]),
        ("<{~-<\u{200d}|a~", &[29, 92, 67300, 29, 16727, 93, 66, 95]),
        (
            "\u{bd} \u{2153} \u{2460}",
            &[41189, 222, 72520, 243, 222, 89537],
        ),
        ("                x", &[692, 1460]),
        ("a\n\n\nb", &[66, 2275, 67]),
        (
            "def f(x):\n    return x\n",
            &[1559, 296, 4046, 1025, 315, 1026, 1460, 200],
        ),
        (
            "<|ifm|im_start|>user\n안녕<|ifm|im_end|>",
            &[250018, 2672, 200, 76943, 47589, 245, 250019],
        ),
        ("reserved_special_token_566", &[250616]),
        (
            "<ifm|think>\nplan</ifm|think>",
            &[250029, 200, 15137, 250030],
        ),
    ];
    for (text, expected) in cases {
        assert_eq!(vocab.encode_text(text), *expected, "{text:?}");
    }
}

#[test]
#[ignore = "requires the release JSON assets in DS4_K2H_MODEL; no GPU"]
fn release_template_keeps_the_pinned_grammar() {
    // The release's template is IFM's later official one: string content
    // renders as the d33e3ae4 fixture does; typed text parts join as text.
    let release = Template::from_model(&root()).unwrap().unwrap();
    let pinned = Template::compile(K2_TEMPLATE, RenderClock::Fixed(0)).unwrap();
    let (tool_messages, tools) = tool_turns();
    for effort in ["high", "medium", "low"] {
        for (messages, tools) in [
            (conversation(), vec![]),
            (tool_messages.clone(), tools.clone()),
        ] {
            for prompt in [true, false] {
                let context = context(&messages, &tools, effort, prompt);
                assert_eq!(
                    release.render(&context).unwrap(),
                    pinned.render(&context).unwrap()
                );
            }
        }
    }
    for (effort, opener) in [
        ("high", "<ifm|think>\n"),
        ("medium", "<ifm|think_fast>\n"),
        ("low", "<ifm|think_faster>\n"),
    ] {
        let text = release
            .render(&context(&conversation(), &[], effort, true))
            .unwrap();
        assert!(text.ends_with(&format!("<|ifm|im_start|>assistant\n{opener}")));
    }
    let parts = [json!({"role":"user","content":[
        {"type":"text","text":"Hello "},{"type":"text","text":"world"}]})];
    let text = [json!({"role":"user","content":"Hello world"})];
    assert_eq!(
        release.render(&context(&parts, &[], "high", true)).unwrap(),
        release.render(&context(&text, &[], "high", true)).unwrap()
    );
}
