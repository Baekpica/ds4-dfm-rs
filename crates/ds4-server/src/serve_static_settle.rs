use super::{settle_static_finish, StaticRow};
use crate::generate::{chat_format_for_syntax, responses_ids, stream_req_from_parsed};
use crate::parse::ParsedRequest;
use crate::render::syntax_for_model_id;
use crate::route::{decode_budget, think_mode_enabled, Api, ReqKind};
use crate::stream::{anthropic_final_response, final_response, responses_final_response};
use crate::tools::parse_generated_for_response;

pub struct StaticSettle<'a> {
    pub parsed: &'a ParsedRequest,
    pub job_id: &'a str,
    pub created: i64,
    pub cors: bool,
    pub default_tokens: i32,
    pub model_id: i32,
    pub prompt_n: i32,
    pub row: &'a StaticRow,
}

pub fn detok_static_row(
    row: &StaticRow,
    budget: i32,
    is_stop: impl Fn(i32) -> bool,
    text_of: impl Fn(i32) -> Vec<u8>,
) -> (Vec<u8>, &'static str, i32) {
    let (finish, n) = settle_static_finish(row.finish, row.tokens.len(), budget);
    let mut text = Vec::new();
    let mut emitted = 0i32;
    for &token in row.tokens.iter().take(n) {
        if is_stop(token) {
            break;
        }
        text.extend(text_of(token));
        emitted += 1;
    }
    (text, finish, emitted)
}

pub fn write_static_completion(
    spec: StaticSettle<'_>,
    is_stop: impl Fn(i32) -> bool,
    text_of: impl Fn(i32) -> Vec<u8>,
) -> Vec<u8> {
    let budget = decode_budget(
        spec.parsed.max_tokens_set,
        spec.parsed.max_tokens,
        spec.default_tokens,
    );
    let (text, finish, emitted) = detok_static_row(spec.row, budget, is_stop, text_of);
    let req = stream_req_from_parsed(spec.parsed, spec.model_id);
    match spec.parsed.kind {
        ReqKind::Completion => final_response(
            &req,
            spec.job_id,
            &text,
            None,
            finish,
            spec.prompt_n,
            emitted,
            spec.created,
            spec.cors,
            &[],
            &[],
        ),
        ReqKind::Chat => write_static_chat(spec, &req, &text, finish, emitted),
    }
}

fn write_static_chat(
    spec: StaticSettle<'_>,
    req: &crate::stream::StreamReq,
    text: &[u8],
    finish: &str,
    emitted: i32,
) -> Vec<u8> {
    let syntax = syntax_for_model_id(spec.model_id);
    let format = chat_format_for_syntax(syntax);
    let (parsed_gen, finish) = parse_generated_for_response(
        syntax,
        text,
        false,
        false,
        think_mode_enabled(spec.parsed.think_mode),
        format,
        &spec.parsed.tool_orders,
        finish,
    );
    match spec.parsed.api {
        Api::Anthropic => anthropic_final_response(
            req,
            spec.job_id,
            &parsed_gen.content,
            Some(&parsed_gen.reasoning),
            finish,
            None,
            spec.prompt_n,
            emitted,
            spec.cors,
            &parsed_gen.calls,
        ),
        Api::Responses => {
            let (rid, rsid, mid) = responses_ids(spec.job_id);
            responses_final_response(
                req,
                &parsed_gen.content,
                Some(&parsed_gen.reasoning),
                finish,
                spec.prompt_n,
                emitted,
                0,
                spec.created,
                spec.cors,
                &rid,
                &rsid,
                &mid,
                &parsed_gen.calls,
            )
        }
        Api::Openai => final_response(
            req,
            spec.job_id,
            &parsed_gen.content,
            Some(&parsed_gen.reasoning),
            finish,
            spec.prompt_n,
            emitted,
            spec.created,
            spec.cors,
            &parsed_gen.calls,
            &[],
        ),
    }
}
