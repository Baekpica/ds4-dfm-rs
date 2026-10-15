use ds4_core::{
    chat_template::{ChatOptions, Template},
    ChatThinkMode, ModelFamily, Variant, Vocab,
};
use serde_json::Value;
use std::path::PathBuf;

#[test]
fn family_stays_unqualified() {
    let shape = ds4_core::shape_for_variant(Variant::Glm53FlashVq);
    assert_eq!(shape.family, ModelFamily::Glm53Vq);
    assert_ne!(shape.family, ModelFamily::Glm53);
    assert_eq!(shape.name, "GLM-5.3-Flash-VQ");
    let caps = ds4_core::caps_from_shape(shape);
    assert_eq!(caps.qualified_ctx, None);
    assert_eq!(caps.qualified_banks, None);
    assert_eq!(caps.qualified_prompt, None);
}

#[test]
#[ignore = "requires original JSON assets in DS4_GLM_VQ_MODEL; no GPU"]
fn original_input_contract() {
    let root = PathBuf::from(std::env::var_os("DS4_GLM_VQ_MODEL").expect("DS4_GLM_VQ_MODEL"));
    let fixture: Value =
        serde_json::from_str(include_str!("../../../../tests/fixtures/glm_vq_input.json")).unwrap();
    let vocab = Vocab::load_path(&root, ModelFamily::Glm53Vq).unwrap();
    let template = Template::from_model(&root).unwrap().unwrap();
    let ids = |v: &Value| {
        v.as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_i64().unwrap() as i32)
            .collect::<Vec<_>>()
    };
    for case in fixture["raw"].as_array().unwrap() {
        let text = case["text"].as_str().unwrap();
        assert_eq!(vocab.encode_text(text), ids(&case["ids"]), "raw {text:?}");
    }
    for case in fixture["chats"].as_array().unwrap() {
        let context = &case["context"];
        let text = template.render(context).unwrap();
        assert_eq!(text, case["text"].as_str().unwrap(), "{context}");
        assert_eq!(vocab.encode_rendered_chat(&text), ids(&case["ids"]));
        let mode = match context["reasoning_effort"].as_str().unwrap() {
            "low" => ChatThinkMode::Low,
            "high" => ChatThinkMode::High,
            "max" => ChatThinkMode::Max,
            _ => unreachable!(),
        };
        let tools = context
            .get("tools")
            .and_then(Value::as_array)
            .map(Vec::as_slice)
            .unwrap_or(&[]);
        let rendered = template
            .render_chat(
                context["messages"].as_array().unwrap(),
                tools,
                ChatOptions::new(Variant::Glm53FlashVq as i32, mode),
            )
            .unwrap();
        assert_eq!(rendered, text);
    }
    for id in [154820, 154827, 154829] {
        assert!(vocab.is_stop(id));
    }
    for id in [3925, 154828, 154841, 154842] {
        assert!(!vocab.is_stop(id));
    }
    assert!(vocab.token_text(154856).is_empty());
    println!(
        "GVQ1_INPUT_PASS raw={} chats={}",
        fixture["raw"].as_array().unwrap().len(),
        fixture["chats"].as_array().unwrap().len()
    );
}
