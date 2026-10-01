use hf_chat_template::ChatTemplate;
use serde_json::{json, Value};

#[test]
fn numeric_context_stays_numeric() {
    let template = ChatTemplate::from_str(
        "{{ count + 1 }}|{{ count is number }}|{{ data | tojson }}|{{ ratio | string }}",
    )
    .unwrap();
    let context: Value =
        serde_json::from_str(r#"{"count":2,"data":{"z":[1,-2,1.5,1e-5],"a":true},"ratio":1e-5}"#)
            .unwrap();
    assert_eq!(
        template.render_context(&context).unwrap(),
        "3|True|{\"z\": [1, -2, 1.5, 1e-05], \"a\": true}|1e-05"
    );
}

#[test]
fn integer_lexemes_survive_json() {
    let template = ChatTemplate::from_str(
        "{{ data.n }}|{{ data.n | string }}|{{ data | tojson }}|{{ text | fromjson | tojson }}",
    )
    .unwrap();
    for number in [
        "18446744073709551617",
        "-18446744073709551617",
        "340282366920938463463374607431768211456",
        "-340282366920938463463374607431768211456",
    ] {
        let text = format!("{{\"n\":{number}}}");
        let data: Value = serde_json::from_str(&text).unwrap();
        assert_eq!(
            template
                .render_context(&json!({"data":data,"text":text}))
                .unwrap(),
            format!("{number}|{number}|{{\"n\": {number}}}|{{\"n\": {number}}}")
        );
    }
}

#[test]
fn wide_integer_math_fails() {
    let template = ChatTemplate::from_str("{{ n + 1 }}").unwrap();
    let context: Value =
        serde_json::from_str(r#"{"n":340282366920938463463374607431768211456}"#).unwrap();
    assert!(template.render_context(&context).is_err());
}

#[test]
fn unsigned_128_math_fails() {
    for number in [
        "170141183460469231731687303715884105728",
        "340282366920938463463374607431768211455",
    ] {
        let context: Value = serde_json::from_str(&format!(r#"{{"n":{number}}}"#)).unwrap();
        for expression in ["{{ n + n }}", "{{ n * n }}", "{{ n + 1 }}"] {
            let template = ChatTemplate::from_str(expression).unwrap();
            assert!(
                template.render_context(&context).is_err(),
                "{number}: {expression} must not coerce u128 to i128"
            );
        }
        let template = ChatTemplate::from_str("{{ n }}|{{ n | tojson }}").unwrap();
        assert_eq!(
            template.render_context(&context).unwrap(),
            format!("{number}|{number}")
        );
    }
}

#[test]
fn wide_integer_type_and_order() {
    let template = ChatTemplate::from_str(
        "{{ a is integer }}|{{ a is number }}|{{ a < b }}|{{ a == b }}|{{ a == c }}",
    )
    .unwrap();
    for (a, b) in [
        (
            "999999999999999999999999999999999999999",
            "1000000000000000000000000000000000000000",
        ),
        (
            "-1000000000000000000000000000000000000000",
            "-999999999999999999999999999999999999999",
        ),
        (
            "-999999999999999999999999999999999999999",
            "999999999999999999999999999999999999999",
        ),
    ] {
        let context: Value =
            serde_json::from_str(&format!(r#"{{"a":{a},"b":{b},"c":{a}}}"#)).unwrap();
        assert_eq!(
            template.render_context(&context).unwrap(),
            "True|True|True|False|True"
        );
    }
}

#[test]
fn native_numeric_tests_unchanged() {
    let template = ChatTemplate::from_str("{{ n is integer }}|{{ n is number }}").unwrap();
    for (number, expected) in [
        ("0", "True|True"),
        ("-1", "True|True"),
        ("170141183460469231731687303715884105727", "True|True"),
        ("-170141183460469231731687303715884105728", "True|True"),
        ("1.0", "False|True"),
        ("true", "False|False"),
        ("null", "False|False"),
        (r#""42""#, "False|False"),
    ] {
        let context: Value = serde_json::from_str(&format!(r#"{{"n":{number}}}"#)).unwrap();
        assert_eq!(template.render_context(&context).unwrap(), expected);
    }
}
