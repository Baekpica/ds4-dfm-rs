//! Family chat render from `ds4_server.c` at v0.6.5-dfm, including
//! tool-schema prompts and invoke reconstruct. Live tool stream
//! machines are host-owned (`tool_stream`).

use crate::json::{
    json_args_parse, json_escape_bytes, json_minify_raw_value, json_raw_value, Json,
};
use crate::parse::{ChatMsg, ChatPart, ToolCall, ToolChoice, ToolSchemaOrder};
use crate::route::{think_mode_enabled, Api, ThinkMode};

/// Copied from `ds4.c` `DS4_REASONING_EFFORT_HIGH_PREFIX`.
pub const THINK_HIGH_PREFIX: &str = concat!(
    "Reasoning Effort: Absolute maximum with no shortcuts permitted.\n",
    "You MUST be very thorough in your thinking and comprehensively decompose the problem to resolve the root cause, rigorously stress-testing your logic against all potential paths, edge cases, and adversarial scenarios.\n",
    "Explicitly write out your entire deliberation process, documenting every intermediate step, considered alternative, and rejected hypothesis to ensure absolutely no assumption is left unchecked.\n\n",
);

/// Copied from `ds4.c` `DS4_REASONING_EFFORT_MAX_PREFIX`.
pub const THINK_MAX_PREFIX: &str = concat!(
    "Reasoning Effort: Beyond maximum — exhaustive, relentless, and uncompromising.\n",
    "You MUST reason with the utmost depth and rigor, leaving absolutely nothing to chance: exhaustively decompose the problem into its most fundamental components, trace every causal chain to its root, and resolve the underlying cause rather than any surface symptom.\n",
    "Do not stop reasoning until you have independently verified the solution from multiple angles and are certain that no assumption remains unchecked and no error remains undiscovered.\n\n",
);

pub const DSML_BOS: &str = "<｜begin▁of▁sentence｜>";
pub const DSML_USER: &str = "<｜User｜>";
pub const DSML_ASSISTANT: &str = "<｜Assistant｜>";
pub const DSML_EOS: &str = "<｜end▁of▁sentence｜>";
pub const DSML_TOOL_CALLS: &str = "<｜DSML｜tool_calls>";
pub const MOTIF_TOOL_CALLS: &str = "<tool_call>";
pub const EXAONE_TOOL_CALLS: &str = "<tool_call>";
pub const DOTS3_TOOL_CALLS: &str = "<dots_function_call>";
pub const SOLAR_TOOL_CALLS: &str = "<|tool_call:start|>";
pub const SOLAR_TOOL_START: &str = "<|tool:start|>";
pub const SOLAR_TOOL_END: &str = "<|tool:end|>";
pub const SOLAR_TOOL_CALL_END: &str = "<|tool_call:end|>";
pub const SOLAR_TOOL_ARG_START: &str = "<|tool_arg:start|>";
pub const SOLAR_TOOL_ARG_VALUE: &str = "<|tool_arg:value|>";
pub const SOLAR_TOOL_ARG_END: &str = "<|tool_arg:end|>";
pub const SOLAR_IM_START: &str = "<|im:start|>";
pub const SOLAR_IM_CONTENT: &str = "<|im:content|>";
pub const SOLAR_IM_END: &str = "<|im:end|>";
pub const SOLAR_THINK_START: &str = "<|think:start|>";
pub const SOLAR_THINK_END: &str = "<|think:end|>";
pub const SOLAR_TOOL_RESPONSE_START: &str = "<|tool_response:start|>";
pub const SOLAR_TOOL_RESPONSE_END: &str = "<|tool_response:end|>";
pub const QWEN_IM_START: &str = "<|im_start|>";
pub const QWEN_IM_END: &str = "<|im_end|>";
pub const QWEN_TOOL_CALL_START: &str = "<tool_call>";
pub const QWEN_TOOL_CALL_END: &str = "</tool_call>";
pub const QWEN_TOOL_RESPONSE_START: &str = "<tool_response>";
pub const QWEN_TOOL_RESPONSE_END: &str = "</tool_response>";
pub const QWEN_VISION_START: &str = "<|vision_start|>";
pub const QWEN_IMAGE_PAD: &str = "<|image_pad|>";
pub const QWEN_VISION_END: &str = "<|vision_end|>";
pub const GLM_BOS: &str = "[gMASK]<sop>";
pub const GLM_TOOL_CALL_START: &str = "<tool_call>";
pub const GLM_TOOL_CALL_END: &str = "</tool_call>";
pub const GLM_VISION_START: &str = "<|begin_of_image|>";
pub const GLM_IMAGE: &str = "<|image|>";
pub const GLM_VISION_END: &str = "<|end_of_image|>";
pub const K2_BOS: &str = "<|ifm|begin_of_text|>";
pub const K2_IM_START: &str = "<|ifm|im_start|>";
pub const K2_IM_END: &str = "<|ifm|im_end|>";
pub const K2_THINK_START: &str = "<ifm|think>";
pub const K2_THINK_END: &str = "</ifm|think>";
pub const K2_TOOL_CALLS_START: &str = "<ifm|tool_calls>";
pub const K2_TOOL_CALLS_END: &str = "</ifm|tool_calls>";
pub const K2_TOOL_CALL_START: &str = "<ifm|tool_call>";
pub const K2_TOOL_CALL_END: &str = "</ifm|tool_call>";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelSyntax {
    DeepSeek = 0,
    SolarOpen2 = 2,
    Motif3 = 3,
    Exaone = 4,
    Dots3 = 5,
    Qwen4Exp = 6,
    Glm53 = 7,
    K2Horizon = 8,
}

/// C `server_model_syntax_for_engine`.
pub fn syntax_for_model_id(model_id: i32) -> ModelSyntax {
    match model_id {
        2 => ModelSyntax::SolarOpen2,
        3 => ModelSyntax::Motif3,
        4 => ModelSyntax::Exaone,
        5 => ModelSyntax::Dots3,
        6 => ModelSyntax::Qwen4Exp,
        7 => ModelSyntax::Glm53,
        8 => ModelSyntax::K2Horizon,
        _ => ModelSyntax::DeepSeek,
    }
}

pub fn tool_start_marker(syntax: ModelSyntax) -> &'static str {
    match syntax {
        ModelSyntax::SolarOpen2 => SOLAR_TOOL_CALLS,
        ModelSyntax::Motif3 | ModelSyntax::Exaone => MOTIF_TOOL_CALLS,
        ModelSyntax::Dots3 => DOTS3_TOOL_CALLS,
        ModelSyntax::Qwen4Exp => QWEN_TOOL_CALL_START,
        ModelSyntax::Glm53 => GLM_TOOL_CALL_START,
        ModelSyntax::K2Horizon => K2_TOOL_CALLS_START,
        ModelSyntax::DeepSeek => DSML_TOOL_CALLS,
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RenderError(pub &'static str);

impl std::fmt::Display for RenderError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.0)
    }
}

pub fn think_effort_prefix(mode: ThinkMode) -> &'static str {
    match mode {
        ThinkMode::High => THINK_HIGH_PREFIX,
        ThinkMode::Max => THINK_MAX_PREFIX,
        ThinkMode::None | ThinkMode::Low => "",
    }
}

pub fn role_is_system(role: &str) -> bool {
    role == "system" || role == "developer"
}

pub fn role_is_user_like(role: &str) -> bool {
    role == "user" || role == "tool" || role == "function"
}

fn chat_history_uses_tool_context(msgs: &[ChatMsg], tool_schemas: &str) -> bool {
    if !tool_schemas.is_empty() {
        return true;
    }
    msgs.iter().any(|m| {
        (m.role == "assistant" && !m.calls.is_empty()) || m.role == "tool" || m.role == "function"
    })
}

fn put(out: &mut Vec<u8>, s: &str) {
    out.extend_from_slice(s.as_bytes());
}

fn tool_schema_orders_find<'a>(
    orders: &'a [ToolSchemaOrder],
    name: &str,
) -> Option<&'a ToolSchemaOrder> {
    orders.iter().find(|o| o.name == name)
}

fn json_args_find_unused(args: &[crate::json::JsonArg], key: &str) -> Option<usize> {
    args.iter().position(|a| !a.used && a.key == key)
}

fn append_sentinel_escape(out: &mut Vec<u8>, s: &[u8], end: &[u8], repl: &[u8]) {
    let mut i = 0;
    while i < s.len() {
        if s[i..].starts_with(end) {
            out.extend_from_slice(repl);
            i += 1;
        } else {
            out.push(s[i]);
            i += 1;
        }
    }
}

fn append_dsml_attr_escaped(out: &mut Vec<u8>, s: &str) {
    for &c in s.as_bytes() {
        match c {
            b'&' => out.extend_from_slice(b"&amp;"),
            b'<' => out.extend_from_slice(b"&lt;"),
            b'>' => out.extend_from_slice(b"&gt;"),
            b'"' => out.extend_from_slice(b"&quot;"),
            c => out.push(c),
        }
    }
}

fn append_dsml_parameter_text(out: &mut Vec<u8>, s: &str) {
    append_sentinel_escape(
        out,
        s.as_bytes(),
        "</｜DSML｜parameter>".as_bytes(),
        b"&lt;",
    );
}

fn append_dsml_json_literal(out: &mut Vec<u8>, s: &str) {
    append_sentinel_escape(
        out,
        s.as_bytes(),
        "</｜DSML｜parameter>".as_bytes(),
        b"\\u003c",
    );
}

fn append_dsml_arg(out: &mut Vec<u8>, arg: &crate::json::JsonArg) {
    put(out, "<｜DSML｜parameter name=\"");
    append_dsml_attr_escaped(out, &arg.key);
    put(out, "\" string=\"");
    put(out, if arg.is_string { "true" } else { "false" });
    put(out, "\">");
    if arg.is_string {
        append_dsml_parameter_text(out, &arg.value);
    } else {
        append_dsml_json_literal(out, &arg.value);
    }
    put(out, "</｜DSML｜parameter>\n");
}

fn append_dsml_arguments_from_json(
    out: &mut Vec<u8>,
    json: &str,
    order: Option<&ToolSchemaOrder>,
) -> bool {
    let Some(mut args) = json_args_parse(json) else {
        return false;
    };
    if let Some(order) = order {
        for prop in &order.prop {
            if let Some(idx) = json_args_find_unused(&args, prop) {
                append_dsml_arg(out, &args[idx]);
                args[idx].used = true;
            }
        }
    }
    for arg in &args {
        if !arg.used {
            append_dsml_arg(out, arg);
        }
    }
    true
}

fn append_dsml_tools_prompt_text(out: &mut Vec<u8>, tool_schemas: &str, tool_required: bool) {
    if tool_schemas.is_empty() {
        return;
    }
    put(
        out,
        "## Tools\n\n\
You have access to a set of tools to help answer the user question. \
You can invoke tools by writing a \"<｜DSML｜tool_calls>\" block like the following:\n\n\
<｜DSML｜tool_calls>\n\
<｜DSML｜invoke name=\"$TOOL_NAME\">\n\
<｜DSML｜parameter name=\"$PARAMETER_NAME\" string=\"true|false\">$PARAMETER_VALUE</｜DSML｜parameter>\n\
...\n\
</｜DSML｜invoke>\n\
<｜DSML｜invoke name=\"$TOOL_NAME2\">\n\
...\n\
</｜DSML｜invoke>\n\
</｜DSML｜tool_calls>\n\n\
String parameters should be specified as raw text and set `string=\"true\"`. \
Preserve characters such as `>`, `&`, and `&&` exactly; never replace normal string characters with XML or HTML entity escapes. \
Only if a string value itself contains the exact closing parameter tag `</｜DSML｜parameter>`, write that tag as `&lt;/｜DSML｜parameter>` inside the value. \
For all other types (numbers, booleans, arrays, objects), pass the value in JSON format and set `string=\"false\"`.\n\n\
If thinking_mode is enabled (triggered by <think>), you MUST output your complete reasoning inside <think>...</think> BEFORE any tool calls or final response.\n\n\
Otherwise, output directly after </think> with tool calls or final response.\n\n\
### Available Tool Schemas\n\n",
    );
    put(out, tool_schemas);
    put(
        out,
        "\n\nYou MUST strictly follow the above defined tool name and parameter schemas to invoke tool calls. \
Use the exact parameter names from the schemas.",
    );
    if tool_required {
        put(
            out,
            "\n\n### Required Tool Use\n\n\
You MUST call at least one available tool in this turn. \
Do not end the turn with only reasoning or a final answer. \
After any reasoning, emit a valid <｜DSML｜tool_calls> block.",
        );
    }
}

fn append_dsml_tool_calls_text(out: &mut Vec<u8>, m: &ChatMsg) {
    if m.calls.is_empty() {
        return;
    }
    if !m.raw_dsml.is_empty() {
        put(out, &m.raw_dsml);
        return;
    }
    put(out, "\n\n<｜DSML｜tool_calls>\n");
    for tc in &m.calls {
        put(out, "<｜DSML｜invoke name=\"");
        append_dsml_attr_escaped(out, &tc.name);
        put(out, "\">\n");
        if !append_dsml_arguments_from_json(out, &tc.arguments, None) {
            put(
                out,
                "<｜DSML｜parameter name=\"arguments\" string=\"true\">",
            );
            append_dsml_parameter_text(out, &tc.arguments);
            put(out, "</｜DSML｜parameter>\n");
        }
        put(out, "</｜DSML｜invoke>\n");
    }
    put(out, "</｜DSML｜tool_calls>");
}

fn append_solar_tool_schema_blocks(out: &mut Vec<u8>, tool_schemas: &str) {
    let mut p = Json::new(tool_schemas);
    while p.peek().is_some() {
        p.ws();
        if p.peek().is_none() {
            break;
        }
        let before = p.i;
        if let Some(schema) = json_raw_value(&mut p) {
            put(out, SOLAR_TOOL_START);
            put(out, &schema);
            put(out, SOLAR_TOOL_END);
            out.push(b'\n');
        } else {
            put(out, SOLAR_TOOL_START);
            out.extend_from_slice(&tool_schemas.as_bytes()[before..]);
            put(out, SOLAR_TOOL_END);
            out.push(b'\n');
            return;
        }
    }
}

fn append_solar_tools_prompt_text(out: &mut Vec<u8>, tool_schemas: &str) {
    if tool_schemas.is_empty() {
        return;
    }
    put(
        out,
        "## Tools\n\
- You may invoke one or more tools to assist with the user's query.\n\n\
### Available Tools\n",
    );
    append_solar_tool_schema_blocks(out, tool_schemas);
    put(out, "\n### Tool Call Instruction\n");
    put(
        out,
        "- If using a tool, any reasoning must strictly precede the call. Do not append any text after the tool call.\n\
- If no tool is required, answer directly from your knowledge without ever mentioning the availability or absence of tools.\n\
- Each tool call MUST use this following format: ",
    );
    put(out, SOLAR_TOOL_CALLS);
    put(out, "{example-tool-name}\n");
    put(out, SOLAR_TOOL_ARG_START);
    put(out, "{example-key-name-1}");
    put(out, SOLAR_TOOL_ARG_VALUE);
    put(out, "{example-value-1}");
    put(out, SOLAR_TOOL_ARG_END);
    put(out, "\n");
    put(out, SOLAR_TOOL_ARG_START);
    put(out, "{example-key-name-2}");
    put(out, SOLAR_TOOL_ARG_VALUE);
    put(out, "{example-value-2}");
    put(out, SOLAR_TOOL_ARG_END);
    put(out, "\n");
    put(out, SOLAR_TOOL_CALL_END);
    put(out, "\n");
}

fn append_solar_arg(out: &mut Vec<u8>, arg: &crate::json::JsonArg) {
    put(out, SOLAR_TOOL_ARG_START);
    put(out, &arg.key);
    put(out, SOLAR_TOOL_ARG_VALUE);
    put(out, &arg.value);
    put(out, SOLAR_TOOL_ARG_END);
    out.push(b'\n');
}

fn append_solar_arguments_from_json(
    out: &mut Vec<u8>,
    json: &str,
    order: Option<&ToolSchemaOrder>,
) -> bool {
    let Some(mut args) = json_args_parse(json) else {
        return false;
    };
    if let Some(order) = order {
        for prop in &order.prop {
            if let Some(idx) = json_args_find_unused(&args, prop) {
                append_solar_arg(out, &args[idx]);
                args[idx].used = true;
            }
        }
    }
    for arg in &args {
        if !arg.used {
            append_solar_arg(out, arg);
        }
    }
    true
}

fn append_solar_tool_calls_text(out: &mut Vec<u8>, m: &ChatMsg, orders: &[ToolSchemaOrder]) {
    if m.calls.is_empty() {
        return;
    }
    if !m.raw_dsml.is_empty() {
        put(out, &m.raw_dsml);
        return;
    }
    for (i, tc) in m.calls.iter().enumerate() {
        if i > 0 {
            out.push(b'\n');
        }
        put(out, SOLAR_TOOL_CALLS);
        put(out, &tc.name);
        out.push(b'\n');
        let order = tool_schema_orders_find(orders, &tc.name);
        if !append_solar_arguments_from_json(out, &tc.arguments, order) {
            put(out, SOLAR_TOOL_ARG_START);
            put(out, "arguments");
            put(out, SOLAR_TOOL_ARG_VALUE);
            put(
                out,
                if tc.arguments.is_empty() {
                    "{}"
                } else {
                    &tc.arguments
                },
            );
            put(out, SOLAR_TOOL_ARG_END);
            out.push(b'\n');
        }
        put(out, SOLAR_TOOL_CALL_END);
    }
}

fn append_qwen_arg(out: &mut Vec<u8>, arg: &crate::json::JsonArg) {
    put(out, "<parameter=");
    put(out, &arg.key);
    put(out, ">\n");
    put(out, &arg.value);
    put(out, "\n</parameter>\n");
}

fn append_qwen_arguments_from_json(
    out: &mut Vec<u8>,
    json: &str,
    order: Option<&ToolSchemaOrder>,
) -> bool {
    let Some(mut args) = json_args_parse(json) else {
        return false;
    };
    if let Some(order) = order {
        for prop in &order.prop {
            if let Some(idx) = json_args_find_unused(&args, prop) {
                append_qwen_arg(out, &args[idx]);
                args[idx].used = true;
            }
        }
    }
    for arg in &args {
        if !arg.used {
            append_qwen_arg(out, arg);
        }
    }
    true
}

fn append_qwen_tool_calls_text(out: &mut Vec<u8>, m: &ChatMsg, orders: &[ToolSchemaOrder]) {
    if m.calls.is_empty() {
        return;
    }
    if !m.raw_dsml.is_empty() {
        put(out, &m.raw_dsml);
        return;
    }
    for (i, tc) in m.calls.iter().enumerate() {
        if i > 0 {
            out.push(b'\n');
        }
        put(out, QWEN_TOOL_CALL_START);
        put(out, "\n<function=");
        put(out, &tc.name);
        put(out, ">\n");
        let order = tool_schema_orders_find(orders, &tc.name);
        if !append_qwen_arguments_from_json(out, &tc.arguments, order) {
            put(out, "<parameter=arguments>\n");
            put(
                out,
                if tc.arguments.is_empty() {
                    "{}"
                } else {
                    &tc.arguments
                },
            );
            put(out, "\n</parameter>\n");
        }
        put(out, "</function>\n");
        put(out, QWEN_TOOL_CALL_END);
    }
}

fn append_glm_tag_body(out: &mut Vec<u8>, text: &[u8], end: &str) {
    append_sentinel_escape(out, text, end.as_bytes(), b"&lt;");
}

fn append_glm_arg(out: &mut Vec<u8>, arg: &crate::json::JsonArg) {
    put(out, "<arg_key>");
    append_glm_tag_body(out, arg.key.as_bytes(), "</arg_key>");
    put(out, "</arg_key><arg_value>");
    append_glm_tag_body(out, arg.value.as_bytes(), "</arg_value>");
    put(out, "</arg_value>");
}

fn append_glm_arguments_from_json(
    out: &mut Vec<u8>,
    json: &str,
    order: Option<&ToolSchemaOrder>,
) -> bool {
    let Some(mut args) = json_args_parse(json) else {
        return false;
    };
    if let Some(order) = order {
        for prop in &order.prop {
            if let Some(idx) = json_args_find_unused(&args, prop) {
                append_glm_arg(out, &args[idx]);
                args[idx].used = true;
            }
        }
    }
    for arg in &args {
        if !arg.used {
            append_glm_arg(out, arg);
        }
    }
    true
}

fn append_glm_tool_calls_text(out: &mut Vec<u8>, m: &ChatMsg, orders: &[ToolSchemaOrder]) {
    if m.calls.is_empty() {
        return;
    }
    if !m.raw_dsml.is_empty() {
        put(out, &m.raw_dsml);
        return;
    }
    out.push(b'\n');
    for tc in &m.calls {
        put(out, GLM_TOOL_CALL_START);
        put(out, &tc.name);
        let order = tool_schema_orders_find(orders, &tc.name);
        if !append_glm_arguments_from_json(out, &tc.arguments, order) {
            put(out, "<arg_key>arguments</arg_key><arg_value>");
            append_glm_tag_body(out, tc.arguments.as_bytes(), "</arg_value>");
            put(out, "</arg_value>");
        }
        put(out, GLM_TOOL_CALL_END);
    }
    out.push(b'\n');
}

fn append_glm_tool_schema_json(out: &mut Vec<u8>, schema: &str) -> Option<bool> {
    let args = json_args_parse(schema)?;
    if args
        .iter()
        .any(|arg| arg.key == "defer_loading" && !arg.is_string && arg.value == "true")
    {
        return Some(false);
    }
    out.push(b'{');
    let mut wrote = false;
    for arg in args
        .iter()
        .filter(|arg| arg.key != "defer_loading" && arg.key != "strict")
    {
        if wrote {
            put(out, ", ");
        }
        out.extend(json_escape_bytes(arg.key.as_bytes()));
        put(out, ": ");
        if arg.is_string {
            out.extend(json_escape_bytes(arg.value.as_bytes()));
        } else {
            put(out, &arg.value);
        }
        wrote = true;
    }
    out.push(b'}');
    Some(true)
}

fn append_glm_tools_prompt_text(out: &mut Vec<u8>, tool_schemas: &str) {
    if tool_schemas.is_empty() {
        return;
    }
    put(
        out,
        "\n# Tools\n\n\
You may call one or more functions to assist with the user query.\n\n\
You are provided with function signatures within <tools></tools> XML tags:\n\
<tools>\n\n",
    );
    let mut p = Json::new(tool_schemas);
    let mut any = false;
    loop {
        p.ws();
        if p.peek().is_none() {
            break;
        }
        let Some(schema) = json_raw_value(&mut p) else {
            break;
        };
        match append_glm_tool_schema_json(out, &schema) {
            Some(true) => {
                put(out, "\n\n\n");
                any = true;
            }
            Some(false) => {}
            None => {
                put(out, &schema);
                put(out, "\n\n\n");
                any = true;
            }
        }
    }
    if !any {
        out.push(b'\n');
    }
    put(
        out,
        "</tools>\n\n\
For each function call, output the function name and arguments within the following XML format:\n\
<tool_call>{function-name}<arg_key>{arg-key-1}</arg_key>\
<arg_value>{arg-value-1}</arg_value><arg_key>{arg-key-2}</arg_key>\
<arg_value>{arg-value-2}</arg_value>...</tool_call>",
    );
}

fn append_glm_message_content(out: &mut Vec<u8>, m: &ChatMsg) {
    if m.parts.is_empty() {
        put(out, &m.content);
        return;
    }
    for part in &m.parts {
        match part {
            ChatPart::Text(text) => put(out, text),
            ChatPart::Image(_) => {
                put(out, GLM_VISION_START);
                put(out, GLM_IMAGE);
                put(out, GLM_VISION_END);
            }
        }
    }
}

fn append_glm_tool_result_message(out: &mut Vec<u8>, m: &ChatMsg) {
    let mut views = Vec::new();
    collect_tool_result_message(m, &mut views);
    for view in views {
        put(out, "<tool_response>");
        append_glm_tag_body(out, view.text, "</tool_response>");
        put(out, "</tool_response>");
    }
}

fn append_glm_assistant_prefix(out: &mut Vec<u8>, m: &ChatMsg, preserve_reasoning: bool) {
    if m.content.starts_with("<think>") || m.content.starts_with("</think>") {
        return;
    }
    put(out, "<think>");
    if preserve_reasoning {
        put(out, &m.reasoning);
    }
    put(out, "</think>");
}

pub fn render_glm_chat_ex(
    msgs: &[ChatMsg],
    tool_schemas: &str,
    tool_orders: &[ToolSchemaOrder],
    think_mode: ThinkMode,
) -> Result<Vec<u8>, RenderError> {
    let think = think_mode_enabled(think_mode);
    let tool_context = chat_history_uses_tool_context(msgs, tool_schemas);
    let last_user_idx = msgs
        .iter()
        .rposition(|m| role_is_user_like(&m.role))
        .map(|i| i as i32)
        .unwrap_or(-1);
    let mut out = GLM_BOS.as_bytes().to_vec();
    if think {
        put(&mut out, "<|system|>");
        put(
            &mut out,
            match think_mode {
                ThinkMode::High => "Reasoning Effort: High",
                ThinkMode::Max => "Reasoning Effort: Max",
                ThinkMode::Low => "Reasoning Effort: Max",
                ThinkMode::None => "",
            },
        );
    }
    if !tool_schemas.is_empty() {
        let mut tools = Vec::new();
        append_glm_tools_prompt_text(&mut tools, tool_schemas);
        if !tools.is_empty() {
            put(&mut out, "<|system|>");
            out.extend(tools);
        }
    }
    for m in msgs.iter().filter(|m| role_is_system(&m.role)) {
        put(&mut out, "<|system|>");
        put(&mut out, &m.content);
    }

    let mut pending_assistant = false;
    let mut observation_open = false;
    for (i, m) in msgs.iter().enumerate() {
        if role_is_system(&m.role) {
            observation_open = false;
        } else if chat_msg_is_model_tool_result(m) {
            if !observation_open {
                put(&mut out, "<|observation|>");
            }
            append_glm_tool_result_message(&mut out, m);
            observation_open = true;
            pending_assistant = true;
        } else if m.role == "user" {
            observation_open = false;
            put(&mut out, "<|user|>");
            append_glm_message_content(&mut out, m);
            pending_assistant = true;
        } else if m.role == "assistant" {
            observation_open = false;
            put(&mut out, "<|assistant|>");
            append_glm_assistant_prefix(
                &mut out,
                m,
                think && (tool_context || i as i32 > last_user_idx),
            );
            put_trimmed(&mut out, &m.content);
            append_glm_tool_calls_text(&mut out, m, tool_orders);
            pending_assistant = false;
        }
    }
    if pending_assistant {
        put(&mut out, "<|assistant|>");
        put(&mut out, if think { "<think>" } else { "<think></think>" });
    }
    Ok(out)
}

fn append_motif3_tool_calls_text(out: &mut Vec<u8>, m: &ChatMsg, include_ids: bool) {
    if m.calls.is_empty() {
        return;
    }
    if !m.raw_tool_text.is_empty() {
        put(out, &m.raw_tool_text);
        return;
    }
    for tc in &m.calls {
        put(out, "\n<tool_call>{\"name\": ");
        out.extend(json_escape_bytes(tc.name.as_bytes()));
        put(out, ", \"arguments\": ");
        if tc.arguments.is_empty() {
            put(out, "null");
        } else {
            put(out, &tc.arguments);
        }
        if include_ids && !tc.id.is_empty() {
            put(out, ", \"id\": ");
            out.extend(json_escape_bytes(tc.id.as_bytes()));
        }
        put(out, "}</tool_call>");
    }
}

fn append_exaone_tool_calls_text(out: &mut Vec<u8>, m: &ChatMsg, content_before: bool) {
    if m.calls.is_empty() {
        return;
    }
    if !m.raw_tool_text.is_empty() {
        let raw = m.raw_tool_text.as_bytes();
        let mut start = 0;
        while start < raw.len() && is_c_space(raw[start]) {
            start += 1;
        }
        if content_before {
            out.push(b'\n');
        }
        out.extend_from_slice(&raw[start..]);
        return;
    }
    for (i, tc) in m.calls.iter().enumerate() {
        if content_before || i > 0 {
            out.push(b'\n');
        }
        put(out, "<tool_call>{\"name\": ");
        out.extend(json_escape_bytes(tc.name.as_bytes()));
        put(out, ", \"arguments\": ");
        if tc.arguments.is_empty() {
            put(out, "null");
        } else {
            put(out, &tc.arguments);
        }
        put(out, "}</tool_call>");
    }
}

fn append_exaone_tools_declaration(out: &mut Vec<u8>, tool_schemas: &str) {
    if tool_schemas.is_empty() {
        return;
    }
    put(out, "<|tool_declare|>\n# Tools\n");
    let mut p = Json::new(tool_schemas);
    while p.peek().is_some() {
        p.ws();
        if p.peek().is_none() {
            break;
        }
        let rest = p.i;
        if let Some(raw) = json_raw_value(&mut p) {
            let schema = json_minify_raw_value(&raw);
            put(out, "<tool>");
            put(out, &schema);
            put(out, "</tool>\n");
        } else {
            put(out, "<tool>");
            out.extend_from_slice(&tool_schemas.as_bytes()[rest..]);
            put(out, "</tool>\n");
            break;
        }
    }
    put(out, "<|endofturn|>\n");
}

fn append_motif3_tools_system_text(
    out: &mut Vec<u8>,
    tool_schemas: &str,
    tool_orders: &[ToolSchemaOrder],
) {
    put(
        out,
        "# Tools\n\n\
You may call one or more functions to assist with the user query.\n\n\
You are provided with function signatures within <tools></tools> XML tags:\n\n\
<tools>",
    );
    let bytes = tool_schemas.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        out.push(b'\n');
        if let Some(rel) = bytes[i..].iter().position(|&c| c == b'\n') {
            out.extend_from_slice(&bytes[i..i + rel]);
            i += rel + 1;
        } else {
            out.extend_from_slice(&bytes[i..]);
            break;
        }
    }
    put(
        out,
        "\n</tools>\n\nFor each function call, output in JSON within <tool_call> tags:\n",
    );
    for order in tool_orders {
        put(out, "\n<tool_call>{\"name\": ");
        out.extend(json_escape_bytes(order.name.as_bytes()));
        put(out, ", \"arguments\": {");
        for (j, prop) in order.prop.iter().enumerate() {
            if j > 0 {
                put(out, ", ");
            }
            out.extend(json_escape_bytes(prop.as_bytes()));
            put(out, ": <");
            put(out, prop);
            out.push(b'>');
        }
        put(out, "}}</tool_call>");
    }
}

fn dots3_pyspace_json(out: &mut Vec<u8>, raw: &str) {
    let bytes = raw.as_bytes();
    let mut i = 0;
    let mut in_string = false;
    while i < bytes.len() {
        let c = bytes[i];
        if in_string {
            out.push(c);
            if c == b'\\' && i + 1 < bytes.len() {
                out.push(bytes[i + 1]);
                i += 2;
                continue;
            }
            if c == b'"' {
                in_string = false;
            }
            i += 1;
            continue;
        }
        if is_c_space(c) {
            i += 1;
            continue;
        }
        out.push(c);
        if c == b'"' {
            in_string = true;
        }
        if c == b',' || c == b':' {
            out.push(b' ');
        }
        i += 1;
    }
}

fn append_dots3_tool_call_text(out: &mut Vec<u8>, tc: &ToolCall) {
    put(out, "\n<dots_function_call>\n<invoke name=\"");
    put(out, &tc.name);
    put(out, "\">");
    let mut p = Json::new(&tc.arguments);
    p.ws();
    if p.bump() == Some(b'{') {
        p.ws();
        while p.peek().is_some() && p.peek() != Some(b'}') {
            let Some(key) = crate::json::json_string(&mut p) else {
                break;
            };
            p.ws();
            if p.bump() != Some(b':') {
                break;
            }
            p.ws();
            let is_string = p.peek() == Some(b'"');
            if is_string {
                let Some(plain) = crate::json::json_string(&mut p) else {
                    break;
                };
                put(out, "\n<parameter name=\"");
                put(out, &key);
                put(out, "\">\n");
                put(out, &plain);
                put(out, "\n</parameter>");
            } else if let Some(value) = json_raw_value(&mut p) {
                put(out, "\n<parameter name=\"");
                put(out, &key);
                put(out, "\">\n");
                dots3_pyspace_json(out, &value);
                put(out, "\n</parameter>");
            } else {
                break;
            }
            p.ws();
            if p.peek() == Some(b',') {
                p.i += 1;
            }
            p.ws();
        }
    }
    put(out, "\n</invoke>\n</dots_function_call>");
}

fn append_dots3_tool_calls_text(out: &mut Vec<u8>, m: &ChatMsg) {
    for tc in &m.calls {
        append_dots3_tool_call_text(out, tc);
    }
}

fn append_dots3_tools_system_text(out: &mut Vec<u8>, tool_schemas: &str) {
    put(
        out,
        "\n\n# Tools\n\n\
You may call one or more functions to assist with the user query.\n\n\
You are provided with function signatures within <tools></tools> \
XML tags:\n<tools>",
    );
    let bytes = tool_schemas.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        let rel = bytes[i..].iter().position(|&c| c == b'\n');
        let n = rel.unwrap_or(bytes.len() - i);
        let line = &tool_schemas[i..i + n];
        put(out, "\n{\"type\": \"function\", \"function\": ");
        dots3_pyspace_json(out, line);
        out.push(b'}');
        i += n;
        if i < bytes.len() && bytes[i] == b'\n' {
            i += 1;
        }
    }
    put(
        out,
        "\n</tools>\n\n\
When making tool calls, use XML format to invoke tools and pass \
parameters:\n\n\
<dots_function_call>\n\
<invoke name=\"tool-name-1\">\n\
<parameter name=\"param-key-1\">\n\
param-value-1\n\
</parameter>\n\
<parameter name=\"param-key-2\">\n\
param-value-2\n\
</parameter>\n\
...\n\
</invoke>\n\
</dots_function_call>",
    );
}

/// Protect `</tool_result>` inside tool output. C writes `&lt;` and advances
/// one byte, so a match becomes `&lt;/tool_result>`.
pub fn append_tool_result_text(out: &mut Vec<u8>, s: &[u8]) {
    let end = b"</tool_result>";
    let mut i = 0;
    while i < s.len() {
        if s[i..].starts_with(end) {
            out.extend_from_slice(b"&lt;");
            i += 1;
        } else {
            out.push(s[i]);
            i += 1;
        }
    }
}

/// Byte-for-byte `render_dsml_chat_prompt_text_choice`.
pub fn render_dsml_chat(
    msgs: &[ChatMsg],
    tool_schemas: &str,
    think_mode: ThinkMode,
) -> Result<Vec<u8>, RenderError> {
    render_dsml_chat_choice(msgs, tool_schemas, think_mode, ToolChoice::Auto)
}

pub fn render_dsml_chat_choice(
    msgs: &[ChatMsg],
    tool_schemas: &str,
    think_mode: ThinkMode,
    tool_choice: ToolChoice,
) -> Result<Vec<u8>, RenderError> {
    let think = think_mode_enabled(think_mode);
    let tool_context = chat_history_uses_tool_context(msgs, tool_schemas);
    let mut last_user_idx = -1i32;
    let mut system = Vec::new();
    if !tool_schemas.is_empty() {
        append_dsml_tools_prompt_text(
            &mut system,
            tool_schemas,
            tool_choice == ToolChoice::Required,
        );
    }
    for m in msgs {
        if !role_is_system(&m.role) {
            continue;
        }
        if !system.is_empty() {
            put(&mut system, "\n\n");
        }
        put(&mut system, &m.content);
    }
    for (i, m) in msgs.iter().enumerate() {
        if role_is_user_like(&m.role) {
            last_user_idx = i as i32;
        }
    }

    let mut out = Vec::new();
    put(&mut out, DSML_BOS);
    put(&mut out, think_effort_prefix(think_mode));
    out.extend_from_slice(&system);

    let mut pending_assistant = false;
    let mut pending_tool_result = false;
    for (i, m) in msgs.iter().enumerate() {
        if role_is_system(&m.role) {
            continue;
        } else if m.role == "user" {
            put(&mut out, DSML_USER);
            put(&mut out, &m.content);
            pending_assistant = true;
            pending_tool_result = false;
        } else if m.role == "tool" || m.role == "function" {
            if !pending_tool_result {
                put(&mut out, DSML_USER);
            }
            put(&mut out, "<tool_result>");
            append_tool_result_text(&mut out, m.content.as_bytes());
            put(&mut out, "</tool_result>");
            pending_assistant = true;
            pending_tool_result = true;
        } else if m.role == "assistant" {
            if pending_assistant {
                put(&mut out, DSML_ASSISTANT);
                if think {
                    if tool_context || (i as i32) > last_user_idx {
                        put(&mut out, "<think>");
                        put(&mut out, &m.reasoning);
                        put(&mut out, "</think>");
                    } else {
                        put(&mut out, "</think>");
                    }
                } else {
                    put(&mut out, "</think>");
                }
            }
            put(&mut out, &m.content);
            append_dsml_tool_calls_text(&mut out, m);
            put(&mut out, DSML_EOS);
            pending_assistant = false;
            pending_tool_result = false;
        }
    }

    if pending_assistant {
        put(&mut out, DSML_ASSISTANT);
        put(&mut out, if think { "<think>" } else { "</think>" });
    }
    Ok(out)
}

fn is_c_space(c: u8) -> bool {
    matches!(c, b' ' | b'\t' | b'\n' | b'\r' | 0x0b | 0x0c)
}

fn put_trimmed(out: &mut Vec<u8>, text: &str) -> bool {
    let b = text.as_bytes();
    let mut start = 0;
    while start < b.len() && is_c_space(b[start]) {
        start += 1;
    }
    let mut end = b.len();
    while end > start && is_c_space(b[end - 1]) {
        end -= 1;
    }
    out.extend_from_slice(&b[start..end]);
    end > start
}

fn chat_msg_is_model_tool_result(m: &ChatMsg) -> bool {
    m.role == "tool"
        || m.role == "function"
        || (m.role == "user" && (!m.tool_call_id.is_empty() || !m.tool_call_ids.is_empty()))
}

fn tool_result_id_at(m: &ChatMsg, index: usize) -> &str {
    if index < m.tool_call_ids.len() {
        &m.tool_call_ids[index]
    } else if index == 0 {
        &m.tool_call_id
    } else {
        ""
    }
}

struct ToolResultView<'a> {
    text: &'a [u8],
    id: &'a str,
    used: bool,
}

fn collect_tool_result_message<'a>(m: &'a ChatMsg, views: &mut Vec<ToolResultView<'a>>) {
    if m.role == "user" && (!m.tool_call_id.is_empty() || !m.tool_call_ids.is_empty()) {
        let before = views.len();
        let bytes = m.content.as_bytes();
        let mut i = 0;
        let mut exact = true;
        let mut result_index = 0usize;
        while i < bytes.len() {
            while i < bytes.len() && is_c_space(bytes[i]) {
                i += 1;
            }
            if i >= bytes.len() {
                break;
            }
            let open = b"<tool_result>";
            let close = b"</tool_result>";
            if !bytes[i..].starts_with(open) {
                exact = false;
                break;
            }
            i += open.len();
            if let Some(rel) = bytes[i..].windows(close.len()).position(|w| w == close) {
                views.push(ToolResultView {
                    text: &bytes[i..i + rel],
                    id: tool_result_id_at(m, result_index),
                    used: false,
                });
                result_index += 1;
                i += rel + close.len();
            } else {
                exact = false;
                break;
            }
        }
        if exact && views.len() > before {
            return;
        }
        views.truncate(before);
    }
    views.push(ToolResultView {
        text: m.content.as_bytes(),
        id: tool_result_id_at(m, 0),
        used: false,
    });
}

/// Official Motif-3 path (`render_motif3_chat_prompt_text`).
pub fn render_motif3_chat(
    msgs: &[ChatMsg],
    tool_schemas: &str,
    think_mode: ThinkMode,
) -> Result<Vec<u8>, RenderError> {
    render_motif3_chat_ex(msgs, tool_schemas, &[], think_mode)
}

pub fn render_motif3_chat_ex(
    msgs: &[ChatMsg],
    tool_schemas: &str,
    tool_orders: &[ToolSchemaOrder],
    think_mode: ThinkMode,
) -> Result<Vec<u8>, RenderError> {
    let think = think_mode_enabled(think_mode);
    let have_tools = !tool_schemas.is_empty();
    let first_system = msgs.first().is_some_and(|m| role_is_system(&m.role));
    let last_assistant = msgs.iter().rposition(|m| m.role == "assistant");

    let mut out = Vec::new();
    put(&mut out, "<|beginoftext|>");
    if have_tools {
        put(&mut out, "<|startofturn|><|system|>");
        append_motif3_tools_system_text(&mut out, tool_schemas, tool_orders);
        if first_system {
            put(&mut out, "\n\n");
            put(&mut out, &msgs[0].content);
        }
        put(&mut out, "<|endofturn|>");
    } else if first_system {
        put(&mut out, "<|startofturn|><|system|>");
        put(&mut out, &msgs[0].content);
        put(&mut out, "<|endofturn|>");
    }
    for (i, m) in msgs.iter().enumerate() {
        if i == 0 && first_system {
            continue;
        } else if role_is_system(&m.role) {
            put(&mut out, "<|startofturn|><|system|>");
            put(&mut out, &m.content);
            put(&mut out, "<|endofturn|>");
        } else if m.role == "user" {
            put(&mut out, "<|startofturn|><|user|>");
            put(&mut out, &m.content);
            put(&mut out, "<|endofturn|>");
        } else if m.role == "assistant" {
            put(&mut out, "<|startofturn|><|assistant|>");
            if !m.reasoning.is_empty() && (have_tools || last_assistant == Some(i)) {
                put(&mut out, "<think>");
                put_trimmed(&mut out, &m.reasoning);
                put(&mut out, "</think>");
            }
            put_trimmed(&mut out, &m.content);
            append_motif3_tool_calls_text(&mut out, m, true);
            put(&mut out, "<|endofturn|>");
        } else if m.role == "tool" || m.role == "function" {
            let group_start =
                i == 0 || (msgs[i - 1].role != "tool" && msgs[i - 1].role != "function");
            let group_end = i + 1 == msgs.len()
                || (msgs[i + 1].role != "tool" && msgs[i + 1].role != "function");
            if group_start {
                put(&mut out, "<|startofturn|><|tool|>");
            }
            put(&mut out, "<tool_response>{\"tool_call_id\": ");
            out.extend(json_escape_bytes(m.tool_call_id.as_bytes()));
            put(&mut out, ", \"content\": ");
            out.extend(json_escape_bytes(m.content.as_bytes()));
            put(&mut out, "}</tool_response>");
            if group_end {
                put(&mut out, "<|endofturn|>");
            }
        }
    }
    put(&mut out, "<|startofturn|><|assistant|><think>");
    if !think {
        put(&mut out, "</think>");
    }
    Ok(out)
}

fn dots3_reason_from_content(content: &str) -> (String, String) {
    if let Some(close) = content.find("</think>") {
        let head = &content[..close];
        let open = head.rfind("<think>");
        let mut rb = if let Some(o) = open { o + 7 } else { 0 };
        let mut re = close;
        let b = content.as_bytes();
        while re > rb && b[re - 1] == b'\n' {
            re -= 1;
        }
        while rb < re && b[rb] == b'\n' {
            rb += 1;
        }
        let mut cb = close + 8;
        while cb < content.len() && b[cb] == b'\n' {
            cb += 1;
        }
        (content[rb..re].to_string(), content[cb..].to_string())
    } else {
        (String::new(), content.to_string())
    }
}

fn trim_ws_range(s: &str) -> &str {
    let b = s.as_bytes();
    let mut start = 0;
    while start < b.len() && is_c_space(b[start]) {
        start += 1;
    }
    let mut end = b.len();
    while end > start && is_c_space(b[end - 1]) {
        end -= 1;
    }
    &s[start..end]
}

/// Official dots3-note path (`render_dots3_chat_prompt_text`).
pub fn render_dots3_chat(
    msgs: &[ChatMsg],
    tool_schemas: &str,
    think_mode: ThinkMode,
) -> Result<Vec<u8>, RenderError> {
    let think = think_mode_enabled(think_mode);
    let have_tools = !tool_schemas.is_empty();
    let first_system = msgs.first().is_some_and(|m| role_is_system(&m.role));
    let mut out = Vec::new();
    put(&mut out, "<|system|>");
    if first_system {
        put(&mut out, &msgs[0].content);
    } else {
        put(&mut out, "You are a helpful assistant.");
    }
    if have_tools {
        append_dots3_tools_system_text(&mut out, tool_schemas);
    }
    put(&mut out, "<|endofsystem|>");
    for (i, m) in msgs.iter().enumerate() {
        if i == 0 && first_system {
            continue;
        }
        let is_user = m.role == "user";
        if is_user || role_is_system(&m.role) {
            put(&mut out, "<|user|>");
            put(&mut out, &m.content);
            if is_user && !think && !m.content.ends_with("<no_think>") {
                put(&mut out, "<no_think>");
            }
            put(&mut out, "<|endofuser|>");
        } else if m.role == "assistant" {
            let (reason, content) = if m.reasoning.is_empty() {
                dots3_reason_from_content(&m.content)
            } else {
                (m.reasoning.clone(), m.content.clone())
            };
            let trimmed = trim_ws_range(&reason);
            put(&mut out, "<|assistant|>");
            if !think {
                put(&mut out, "<think>\n\n</think>\n\n");
                put(&mut out, &content);
            } else if !trimmed.is_empty() {
                put(&mut out, "<think>\n");
                put(&mut out, trimmed);
                put(&mut out, "\n</think>\n\n");
                put(&mut out, &content);
            } else {
                put(&mut out, &content);
            }
            append_dots3_tool_calls_text(&mut out, m);
            put(&mut out, "<|endofassistant|>");
        } else if m.role == "tool" || m.role == "function" {
            let group_start =
                i == 0 || (msgs[i - 1].role != "tool" && msgs[i - 1].role != "function");
            let group_end = i + 1 == msgs.len()
                || (msgs[i + 1].role != "tool" && msgs[i + 1].role != "function");
            if group_start {
                put(&mut out, "<|user|>");
            }
            put(&mut out, "\n<dots_function_response>\n");
            put(&mut out, &m.content);
            put(&mut out, "\n</dots_function_response>");
            if group_end {
                put(&mut out, "<|endofuser|>");
            }
        }
    }
    put(&mut out, "<|assistant|>");
    if !think {
        put(&mut out, "<think>\n\n</think>\n\n");
    }
    Ok(out)
}

/// K-EXAONE path (`render_exaone_chat_prompt_text`).
pub fn render_exaone_chat(
    msgs: &[ChatMsg],
    tool_schemas: &str,
    think_mode: ThinkMode,
) -> Result<Vec<u8>, RenderError> {
    let mut last_user_idx = -1i32;
    for (i, m) in msgs.iter().enumerate() {
        if m.role == "user" && !chat_msg_is_model_tool_result(m) {
            last_user_idx = i as i32;
        }
    }
    let mut out = Vec::new();
    append_exaone_tools_declaration(&mut out, tool_schemas);
    let mut i = 0;
    while i < msgs.len() {
        let m = &msgs[i];
        if chat_msg_is_model_tool_result(m) {
            let mut views = Vec::new();
            while i < msgs.len() && chat_msg_is_model_tool_result(&msgs[i]) {
                collect_tool_result_message(&msgs[i], &mut views);
                i += 1;
            }
            put(&mut out, "<|tool|>\n");
            for (k, v) in views.iter().enumerate() {
                if k > 0 {
                    out.push(b'\n');
                }
                put(&mut out, "<tool_result>");
                out.extend_from_slice(v.text);
                put(&mut out, "</tool_result>");
            }
            put(&mut out, "<|endofturn|>\n");
            continue;
        } else if role_is_system(&m.role) {
            put(&mut out, "<|system|>\n");
            put(&mut out, &m.content);
            put(&mut out, "<|endofturn|>\n");
        } else if m.role == "user" {
            put(&mut out, "<|user|>\n");
            put(&mut out, &m.content);
            put(&mut out, "<|endofturn|>\n");
        } else if m.role == "assistant" {
            put(&mut out, "<|assistant|>\n<think>\n");
            if !m.reasoning.is_empty() && (i as i32) > last_user_idx {
                put_trimmed(&mut out, &m.reasoning);
            }
            put(&mut out, "\n</think>\n\n");
            let have_content = put_trimmed(&mut out, &m.content);
            append_exaone_tool_calls_text(&mut out, m, have_content);
            put(&mut out, "<|endofturn|>\n");
        }
        i += 1;
    }
    put(&mut out, "<|assistant|>\n<think>\n");
    if !think_mode_enabled(think_mode) {
        put(&mut out, "\n</think>\n\n");
    }
    Ok(out)
}

pub(crate) fn solar_role_open(out: &mut Vec<u8>, role: &str) {
    put(out, SOLAR_IM_START);
    put(out, role);
    put(out, SOLAR_IM_CONTENT);
}

pub(crate) fn append_solar_tool_response_text(out: &mut Vec<u8>, s: &[u8]) {
    let end = SOLAR_TOOL_RESPONSE_END.as_bytes();
    let mut i = 0;
    while i < s.len() {
        if s[i..].starts_with(end) {
            out.extend_from_slice(b"&lt;");
            i += 1;
        } else {
            out.push(s[i]);
            i += 1;
        }
    }
}

/// Solar-Open2 path (`render_solar_chat_prompt_text_choice`).
pub fn render_solar_chat(
    msgs: &[ChatMsg],
    tool_schemas: &str,
    think_mode: ThinkMode,
) -> Result<Vec<u8>, RenderError> {
    render_solar_chat_ex(msgs, tool_schemas, &[], think_mode)
}

pub fn render_solar_chat_ex(
    msgs: &[ChatMsg],
    tool_schemas: &str,
    tool_orders: &[ToolSchemaOrder],
    think_mode: ThinkMode,
) -> Result<Vec<u8>, RenderError> {
    let mut last_user_idx = -1i32;
    for (i, m) in msgs.iter().enumerate() {
        if m.role == "user" && !chat_msg_is_model_tool_result(m) {
            last_user_idx = i as i32;
        }
    }
    let mut system = Vec::new();
    let mut have_system = false;
    for m in msgs {
        if !role_is_system(&m.role) {
            continue;
        }
        if !have_system {
            put(&mut system, "## System Prompt");
            have_system = true;
        }
        put(&mut system, "\n\n");
        put(&mut system, &m.content);
    }
    if !tool_schemas.is_empty() {
        if !system.is_empty() {
            put(&mut system, "\n\n");
        }
        append_solar_tools_prompt_text(&mut system, tool_schemas);
    }
    let mut out = Vec::new();
    if !system.is_empty() {
        solar_role_open(&mut out, "system");
        out.extend_from_slice(&system);
        put(&mut out, SOLAR_IM_END);
        out.push(b'\n');
    }
    let mut i = 0;
    while i < msgs.len() {
        let m = &msgs[i];
        if role_is_system(&m.role) {
            i += 1;
            continue;
        } else if m.role == "user" && !chat_msg_is_model_tool_result(m) {
            solar_role_open(&mut out, "user");
            put(&mut out, &m.content);
            put(&mut out, SOLAR_IM_END);
            out.push(b'\n');
            i += 1;
        } else if chat_msg_is_model_tool_result(m) {
            let prior = if i > 0 { Some(&msgs[i - 1]) } else { None };
            let mut views = Vec::new();
            while i < msgs.len() && chat_msg_is_model_tool_result(&msgs[i]) {
                collect_tool_result_message(&msgs[i], &mut views);
                i += 1;
            }
            solar_role_open(&mut out, "tool");
            let mut first = true;
            if let Some(prior) = prior {
                if prior.role == "assistant" {
                    for tc in &prior.calls {
                        if tc.id.is_empty() {
                            continue;
                        }
                        if let Some(v) = views.iter_mut().find(|v| !v.used && v.id == tc.id) {
                            if !first {
                                out.push(b'\n');
                            }
                            first = false;
                            put(&mut out, SOLAR_TOOL_RESPONSE_START);
                            append_solar_tool_response_text(&mut out, v.text);
                            put(&mut out, SOLAR_TOOL_RESPONSE_END);
                            v.used = true;
                        }
                    }
                }
            }
            for v in &views {
                if v.used {
                    continue;
                }
                if !first {
                    out.push(b'\n');
                }
                first = false;
                put(&mut out, SOLAR_TOOL_RESPONSE_START);
                append_solar_tool_response_text(&mut out, v.text);
                put(&mut out, SOLAR_TOOL_RESPONSE_END);
            }
            put(&mut out, "\n");
            put(&mut out, SOLAR_IM_END);
            out.push(b'\n');
        } else if m.role == "assistant" {
            solar_role_open(&mut out, "assistant");
            put(&mut out, SOLAR_THINK_START);
            if !m.reasoning.is_empty() && (i as i32) > last_user_idx {
                put(&mut out, &m.reasoning);
            }
            put(&mut out, SOLAR_THINK_END);
            put(&mut out, &m.content);
            append_solar_tool_calls_text(&mut out, m, tool_orders);
            if !m.calls.is_empty() {
                out.push(b'\n');
            }
            put(&mut out, SOLAR_IM_END);
            out.push(b'\n');
            i += 1;
        } else {
            i += 1;
        }
    }
    solar_role_open(&mut out, "assistant");
    put(&mut out, SOLAR_THINK_START);
    if !think_mode_enabled(think_mode) {
        put(&mut out, SOLAR_THINK_END);
    }
    Ok(out)
}

fn qwen_reasoning_instruction(mode: ThinkMode) -> &'static str {
    match mode {
        ThinkMode::Low => concat!(
            "Reasoning effort is set to low. Keep your thinking brief and focused, ",
            "moving directly to the conclusion without unnecessary elaboration."
        ),
        ThinkMode::High | ThinkMode::Max => concat!(
            "Reasoning effort is set to xhigh. Please think carefully through the task, ",
            "validate key assumptions, consider plausible alternatives, and prioritize ",
            "correctness, consistency, and clarity in the final answer."
        ),
        ThinkMode::None => "",
    }
}

fn append_qwen_tools_prompt_text(out: &mut Vec<u8>, tool_schemas: &str) {
    put(
        out,
        "# Tools\n\nYou have access to the following functions:\n\n<tools>",
    );
    let mut p = Json::new(tool_schemas);
    while p.peek().is_some() {
        p.ws();
        if p.peek().is_none() {
            break;
        }
        let Some(schema) = json_raw_value(&mut p) else {
            break;
        };
        put(out, "\n{\"type\": \"function\", \"function\": ");
        dots3_pyspace_json(out, &schema);
        out.push(b'}');
    }
    put(
        out,
        concat!(
            "\n</tools>",
            "\n\nIf you choose to call a function ONLY reply in the following format with NO suffix:",
            "\n\n<tool_call>\n<function=example_function_name>",
            "\n<parameter=example_parameter_1>\nvalue_1\n</parameter>",
            "\n<parameter=example_parameter_2>\nThis is the value for the second parameter",
            "\nthat can span\nmultiple lines\n</parameter>\n</function>\n</tool_call>",
            "\n\n<IMPORTANT>\nReminder:",
            "\n- Function calls MUST follow the specified format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags",
            "\n- Required parameters MUST be specified",
            "\n- You may provide optional reasoning for your function call in natural language BEFORE the function call, but NOT after",
            "\n- If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls",
            "\n</IMPORTANT>"
        ),
    );
}

pub(crate) fn append_qwen_tool_response_text(out: &mut Vec<u8>, s: &[u8]) {
    let end = QWEN_TOOL_RESPONSE_END.as_bytes();
    let mut i = 0;
    while i < s.len() {
        if s[i..].starts_with(end) {
            out.extend_from_slice(b"&lt;");
            i += 1;
        } else {
            out.push(s[i]);
            i += 1;
        }
    }
}

pub fn render_qwen_chat_ex(
    msgs: &[ChatMsg],
    tool_schemas: &str,
    tool_orders: &[ToolSchemaOrder],
    think_mode: ThinkMode,
) -> Result<Vec<u8>, RenderError> {
    let mut system = Vec::new();
    for m in msgs.iter().filter(|m| role_is_system(&m.role)) {
        if !system.is_empty() {
            put(&mut system, "\n\n");
        }
        put_trimmed(&mut system, &m.content);
    }

    let instruction = qwen_reasoning_instruction(think_mode);
    let mut out = Vec::new();
    if !tool_schemas.is_empty() {
        put(&mut out, QWEN_IM_START);
        put(&mut out, "system\n");
        if !instruction.is_empty() {
            put(&mut out, instruction);
            put(&mut out, "\n\n");
        }
        append_qwen_tools_prompt_text(&mut out, tool_schemas);
        if !system.is_empty() {
            put(&mut out, "\n\n");
            out.extend_from_slice(&system);
        }
        put(&mut out, QWEN_IM_END);
        out.push(b'\n');
    } else if !system.is_empty() || !instruction.is_empty() {
        put(&mut out, QWEN_IM_START);
        put(&mut out, "system\n");
        if !instruction.is_empty() {
            put(&mut out, instruction);
            if !system.is_empty() {
                put(&mut out, "\n\n");
            }
        }
        out.extend_from_slice(&system);
        put(&mut out, QWEN_IM_END);
        out.push(b'\n');
    }

    let mut i = 0;
    while i < msgs.len() {
        let m = &msgs[i];
        if role_is_system(&m.role) {
            i += 1;
            continue;
        }
        if m.role == "user" && !chat_msg_is_model_tool_result(m) {
            put(&mut out, QWEN_IM_START);
            put(&mut out, "user\n");
            if m.parts.is_empty() {
                put_trimmed(&mut out, &m.content);
            } else {
                let mut content = String::new();
                for part in &m.parts {
                    match part {
                        ChatPart::Text(text) => content.push_str(text),
                        ChatPart::Image(_) => {
                            content.push_str(QWEN_VISION_START);
                            content.push_str(QWEN_IMAGE_PAD);
                            content.push_str(QWEN_VISION_END);
                        }
                    }
                }
                put_trimmed(&mut out, &content);
            }
            put(&mut out, QWEN_IM_END);
            out.push(b'\n');
            i += 1;
            continue;
        }
        if chat_msg_is_model_tool_result(m) {
            let mut views = Vec::new();
            while i < msgs.len() && chat_msg_is_model_tool_result(&msgs[i]) {
                collect_tool_result_message(&msgs[i], &mut views);
                i += 1;
            }
            if !views.is_empty() {
                put(&mut out, QWEN_IM_START);
                put(&mut out, "user");
                for v in views {
                    put(&mut out, "\n");
                    put(&mut out, QWEN_TOOL_RESPONSE_START);
                    put(&mut out, "\n");
                    append_qwen_tool_response_text(&mut out, v.text);
                    put(&mut out, "\n");
                    put(&mut out, QWEN_TOOL_RESPONSE_END);
                }
                put(&mut out, QWEN_IM_END);
                out.push(b'\n');
            }
            continue;
        }
        if m.role == "assistant" {
            put(&mut out, QWEN_IM_START);
            put(&mut out, "assistant\n<think>\n");
            put_trimmed(&mut out, &m.reasoning);
            put(&mut out, "\n</think>\n\n");
            let content_start = out.len();
            put_trimmed(&mut out, &m.content);
            if !m.calls.is_empty() && out.len() > content_start {
                put(&mut out, "\n\n");
            }
            append_qwen_tool_calls_text(&mut out, m, tool_orders);
            put(&mut out, QWEN_IM_END);
            out.push(b'\n');
        }
        i += 1;
    }

    put(&mut out, QWEN_IM_START);
    put(&mut out, "assistant\n<think>\n");
    if !think_mode_enabled(think_mode) {
        put(&mut out, "\n</think>\n\n");
    }
    Ok(out)
}

pub fn render_chat(
    syntax: ModelSyntax,
    msgs: &[ChatMsg],
    tool_schemas: &str,
    think_mode: ThinkMode,
) -> Result<Vec<u8>, RenderError> {
    render_chat_choice(
        syntax,
        msgs,
        tool_schemas,
        &[],
        think_mode,
        ToolChoice::Auto,
    )
}

pub fn render_chat_choice(
    syntax: ModelSyntax,
    msgs: &[ChatMsg],
    tool_schemas: &str,
    tool_orders: &[ToolSchemaOrder],
    think_mode: ThinkMode,
    tool_choice: ToolChoice,
) -> Result<Vec<u8>, RenderError> {
    match syntax {
        ModelSyntax::Motif3 => render_motif3_chat_ex(msgs, tool_schemas, tool_orders, think_mode),
        ModelSyntax::Exaone => render_exaone_chat(msgs, tool_schemas, think_mode),
        ModelSyntax::Dots3 => render_dots3_chat(msgs, tool_schemas, think_mode),
        ModelSyntax::SolarOpen2 => {
            render_solar_chat_ex(msgs, tool_schemas, tool_orders, think_mode)
        }
        ModelSyntax::Qwen4Exp => render_qwen_chat_ex(msgs, tool_schemas, tool_orders, think_mode),
        ModelSyntax::Glm53 => render_glm_chat_ex(msgs, tool_schemas, tool_orders, think_mode),
        ModelSyntax::K2Horizon => render_k2_chat(msgs, tool_schemas, think_mode),
        ModelSyntax::DeepSeek => {
            render_dsml_chat_choice(msgs, tool_schemas, think_mode, tool_choice)
        }
    }
}

/// Render the suffix appended to an exact live bank frontier for an
/// Anthropic/Responses tool-result-only continuation.
pub fn render_live_tool_tail(
    syntax: ModelSyntax,
    api: Api,
    msgs: &[ChatMsg],
    think_mode: ThinkMode,
) -> Result<Vec<u8>, RenderError> {
    let start = match api {
        Api::Responses => {
            let mut start = msgs.len();
            while start > 0 && matches!(msgs[start - 1].role.as_str(), "tool" | "function") {
                start -= 1;
            }
            (start < msgs.len()).then_some(start)
        }
        Api::Anthropic => {
            let mut end = msgs.len();
            while end > 0 && role_is_system(&msgs[end - 1].role) {
                end -= 1;
            }
            let mut start = end;
            while start > 0 {
                let msg = &msgs[start - 1];
                if msg.role != "user"
                    || (msg.tool_call_id.is_empty() && msg.tool_call_ids.is_empty())
                {
                    break;
                }
                start -= 1;
            }
            (start < end).then_some(start)
        }
        Api::Openai => None,
    };
    let Some(start) = start else {
        return Err(RenderError("live tool-result suffix is unavailable"));
    };
    let tail = &msgs[start..];
    let mut out = Vec::new();
    match syntax {
        ModelSyntax::Glm53 => {
            let think = think_mode_enabled(think_mode);
            let mut pending_assistant = false;
            let mut observation_open = false;
            for m in tail {
                if role_is_system(&m.role) {
                    observation_open = false;
                } else if chat_msg_is_model_tool_result(m) {
                    if !observation_open {
                        put(&mut out, "<|observation|>");
                    }
                    append_glm_tool_result_message(&mut out, m);
                    observation_open = true;
                    pending_assistant = true;
                } else if m.role == "user" {
                    observation_open = false;
                    put(&mut out, "<|user|>");
                    append_glm_message_content(&mut out, m);
                    pending_assistant = true;
                } else if m.role == "assistant" {
                    observation_open = false;
                    put(&mut out, "<|assistant|>");
                    append_glm_assistant_prefix(&mut out, m, think);
                    put_trimmed(&mut out, &m.content);
                    append_glm_tool_calls_text(&mut out, m, &[]);
                    pending_assistant = false;
                }
            }
            if pending_assistant {
                put(&mut out, "<|assistant|>");
                put(&mut out, if think { "<think>" } else { "<think></think>" });
            }
        }
        ModelSyntax::DeepSeek => {
            put(&mut out, DSML_EOS);
            let tail: Vec<_> = tail
                .iter()
                .filter(|msg| !role_is_system(&msg.role))
                .cloned()
                .collect();
            let rendered = render_dsml_chat_choice(&tail, "", think_mode, ToolChoice::Auto)?;
            out.extend_from_slice(
                rendered
                    .strip_prefix(DSML_BOS.as_bytes())
                    .unwrap_or(&rendered),
            );
        }
        ModelSyntax::SolarOpen2 => {
            put(&mut out, SOLAR_IM_END);
            out.push(b'\n');
            out.extend(render_solar_chat_ex(tail, "", &[], think_mode)?);
        }
        ModelSyntax::Motif3 => {
            put(&mut out, "<|endofturn|>");
            let rendered = render_motif3_chat_ex(tail, "", &[], think_mode)?;
            out.extend_from_slice(
                rendered
                    .strip_prefix(b"<|beginoftext|>")
                    .unwrap_or(&rendered),
            );
        }
        ModelSyntax::Exaone => {
            put(&mut out, "<|endofturn|>\n");
            out.extend(render_exaone_chat(tail, "", think_mode)?);
        }
        ModelSyntax::Dots3 => {
            put(&mut out, "<|endofassistant|>");
            let mut rendered = render_dots3_chat(tail, "", think_mode)?;
            let body = rendered
                .windows(b"<|endofsystem|>".len())
                .position(|w| w == b"<|endofsystem|>")
                .map(|at| rendered.split_off(at + b"<|endofsystem|>".len()))
                .unwrap_or_default();
            out.extend(body);
        }
        ModelSyntax::Qwen4Exp => {
            put(&mut out, QWEN_IM_END);
            out.push(b'\n');
            out.extend(render_qwen_chat_ex(tail, "", &[], think_mode)?);
        }
        ModelSyntax::K2Horizon => {
            put(&mut out, K2_IM_END);
            let rendered = render_k2_chat(tail, "", think_mode)?;
            out.extend_from_slice(
                rendered
                    .strip_prefix(K2_BOS.as_bytes())
                    .unwrap_or(&rendered),
            );
        }
    }
    Ok(out)
}

fn append_k2_tools_system(out: &mut Vec<u8>, tool_schemas: &str, system: &str) {
    put(out, K2_IM_START);
    put(out, "system\n# Tools\nYou may call one or more tools to assist with the user query.\n\nAvailable tools are:\n\n<ifm|tools>\n");
    put(out, tool_schemas);
    put(out, "\n</ifm|tools>\n\nWhen calling tools, you MUST follow the tool-call format below:\n\nWrap all tool calls in a single <ifm|tool_calls></ifm|tool_calls> block. For each call, emit one JSON object with the function name and arguments on the same line inside <ifm|tool_call></ifm|tool_call> tags:\n\n<ifm|tool_calls>\n<ifm|tool_call>{\"name\": <function-name>, \"arguments\": <args-json-object>}</ifm|tool_call>\n</ifm|tool_calls>");
    if !system.is_empty() {
        put(out, "\n\n");
        put(out, system);
    }
    put(out, K2_IM_END);
}

fn append_k2_tool_calls(out: &mut Vec<u8>, m: &ChatMsg) {
    if m.calls.is_empty() {
        return;
    }
    if !m.raw_dsml.is_empty() {
        put(out, &m.raw_dsml);
        return;
    }
    put(out, K2_TOOL_CALLS_START);
    for tc in &m.calls {
        put(out, "\n");
        put(out, K2_TOOL_CALL_START);
        put(out, "{\"name\": ");
        out.extend(json_escape_bytes(tc.name.as_bytes()));
        put(out, ", \"arguments\": ");
        put(
            out,
            &json_minify_raw_value(if tc.arguments.is_empty() {
                "null"
            } else {
                &tc.arguments
            }),
        );
        put(out, "}");
        put(out, K2_TOOL_CALL_END);
    }
    put(out, "\n");
    put(out, K2_TOOL_CALLS_END);
}

/// Official IFM chat protocol using its supported JSON tool dialect.
pub fn render_k2_chat(
    msgs: &[ChatMsg],
    tool_schemas: &str,
    think_mode: ThinkMode,
) -> Result<Vec<u8>, RenderError> {
    let mut out = Vec::new();
    put(&mut out, K2_BOS);
    let first_system = msgs.first().filter(|m| role_is_system(&m.role));
    if !tool_schemas.is_empty() {
        append_k2_tools_system(
            &mut out,
            tool_schemas,
            first_system.map(|m| m.content.as_str()).unwrap_or(""),
        );
    }
    for (i, m) in msgs.iter().enumerate() {
        if role_is_system(&m.role) {
            if i == 0 && !tool_schemas.is_empty() {
                continue;
            }
            put(&mut out, K2_IM_START);
            put(&mut out, "system\n");
            put(&mut out, &m.content);
            put(&mut out, K2_IM_END);
            continue;
        }
        if m.role == "user" && !chat_msg_is_model_tool_result(m) {
            put(&mut out, K2_IM_START);
            put(&mut out, "user\n");
            put(&mut out, &m.content);
            put(&mut out, K2_IM_END);
            continue;
        }
        if chat_msg_is_model_tool_result(m) {
            put(&mut out, K2_IM_START);
            put(&mut out, "tool\n");
            put(&mut out, &m.content);
            put(&mut out, K2_IM_END);
            continue;
        }
        if m.role == "assistant" {
            put(&mut out, K2_IM_START);
            put(&mut out, "assistant\n");
            put(&mut out, K2_THINK_START);
            put(&mut out, "\n");
            put(&mut out, &m.reasoning);
            put(&mut out, K2_THINK_END);
            put(&mut out, &m.content);
            append_k2_tool_calls(&mut out, m);
            put(&mut out, K2_IM_END);
        }
    }
    put(&mut out, K2_IM_START);
    put(&mut out, "assistant\n");
    put(&mut out, K2_THINK_START);
    put(&mut out, "\n");
    if !think_mode_enabled(think_mode) {
        put(&mut out, K2_THINK_END);
        put(&mut out, "\n");
    }
    Ok(out)
}
