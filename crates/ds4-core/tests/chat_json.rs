use ds4_core::chat_template::{RenderClock, Template};
use serde_json::Value;

fn check_vector(name: &str) {
    let fixtures: Value = serde_json::from_str(include_str!(
        "../../../tests/fixtures/chat-template/json-vectors.json"
    ))
    .unwrap();
    let case = fixtures["vectors"]
        .as_array()
        .unwrap()
        .iter()
        .find(|case| case["name"] == name)
        .unwrap_or_else(|| panic!("missing Python fixture: {name}"));
    let template = Template::compile(case["template"].as_str().unwrap(), RenderClock::Fixed(0))
        .unwrap_or_else(|error| panic!("{name}: valid Jinja did not compile: {error}"));
    let result = template.render(&case["context"]);

    if let Some(expected) = case["expected"].as_str() {
        let actual = result.unwrap_or_else(|error| panic!("{name}: render failed: {error}"));
        assert_eq!(actual, expected, "{name}: Python JSON bytes differ");
        return;
    }

    assert!(
        result.is_err(),
        "{name}: Python raises {} ({}), renderer returned {result:?}",
        case["error_type"],
        case["error_message"]
    );
}

macro_rules! json_case {
    ($name:ident) => {
        #[test]
        fn $name() {
            check_vector(stringify!($name));
        }
    };
}

json_case!(nested_default);
json_case!(source_map_order);
json_case!(sort_keys_recursive);
json_case!(preserve_input_order);
json_case!(compact_separators);
json_case!(custom_separators);
json_case!(unicode_default);
json_case!(ascii_escape);
json_case!(string_escape);
json_case!(indent_two);
json_case!(indent_tab);
json_case!(indent_zero);
json_case!(indent_negative);
json_case!(indent_compact);
json_case!(empty_containers);
json_case!(integral_floats);
json_case!(signed_zero);
json_case!(small_exponents);
json_case!(lower_threshold);
json_case!(upper_threshold);
json_case!(large_exponents);
json_case!(unknown_keyword);
json_case!(invalid_indent);
json_case!(short_separators);
json_case!(invalid_separator);
