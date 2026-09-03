//! Server host. Phase 7 ports surfaces by feature; no API redesign.

pub mod admit;
pub mod cont;
pub mod dist_cli;
pub mod dsml;
pub mod error;
pub mod format;
pub mod generate;
pub mod http;
pub mod json;
pub mod kv_cli;
pub mod metrics;
pub mod models;
pub mod parse;
pub mod render;
pub mod retry;
pub mod route;
pub mod serve;
#[cfg(test)]
mod serve_503_body_test;
pub mod serve_cont;
#[cfg(any(feature = "native", test))]
mod serve_cont_evict;
mod serve_cont_prefill;
#[cfg(test)]
mod serve_cont_prefill_test;
mod serve_cont_roll;
mod serve_serial_fit;
mod serve_serial_reclaim;
#[cfg(test)]
mod serve_serial_reclaim_test;
pub mod serve_static;
pub mod stream;
#[cfg(any(feature = "native", test))]
mod tool_memory;
pub mod tool_stream;
pub mod tools;
pub mod worker;
#[cfg(feature = "native")]
pub mod worker_run;

pub use admit::{
    enqueue, enqueue_release, enqueue_shed_error, mint_job_id, next_job_id, preparse_shed,
    queue_unlink_head, AdmitState, EnqVerdict, SHED_CLIENTS, SHED_CONT_HOLD, SHED_NAMES,
    SHED_QUEUE_AGE, SHED_QUEUE_BYTES, SHED_QUEUE_DEPTH, SHED_REASONS, SHED_SLOW_READER,
};
pub use cont::{
    dump_script, live_tool_result_ids, place_bank_continuation, BankContConflict, ContOwner,
    ContRecord, ContRegistry, ContState, CONT_GRACE_S, CONT_HOLD_SHED_S, CONT_PIN_DEADLINE_S,
    CONT_REGISTRY_MAX_DEFAULT, CONT_TTL_S,
};
pub use dist_cli::DistArgs;
pub use dsml::{
    agent_turn_reasoning_cap, dsml_decode_state_for_text, dump_script as dsml_dump_script,
    DsmlDecodeState, DsmlDecodeTracker, SampleOverride, SamplePolicy, SAMPLE_OVERRIDE_GREEDY,
    SAMPLE_OVERRIDE_NONE,
};
pub use error::{
    anthropic_error_body, anthropic_error_type, http_head, http_reason, http_response_bytes,
    openai_error_body, openai_error_type, retry_after_header, wire_error_body,
    wire_http_error_bytes,
};
pub use format::{
    output_format_type_supported, parse_output_config_effort, parse_output_config_format,
    parse_output_format_value, parse_reasoning_effort_value, parse_responses_text_value,
};
#[cfg(feature = "native")]
pub use generate::NativeDecode;
pub use generate::{
    generate_and_write, generation_blocked, render_prompt, stop_list_find_from, DecodeIo,
    GenerateError, GenerateOutcome, ScriptedDecode, ScriptedStep,
};
pub use http::{
    chunked_enabled, content_length, header_accepts_json, header_chunked, header_end,
    parse_surface_for_path, read_http_request, shed_surface_for_path, HttpRequest,
};
pub use json::{
    json_args_parse, json_bool, json_content, json_escape, json_escape_bytes, json_int,
    json_minify_raw_value, json_number, json_raw_value, json_skip_value, json_string, Json,
    JsonArg,
};
pub use metrics::{
    dump_memgov_names, gov_modes_from_env, render_memgov_metrics, render_metrics,
    render_metrics_fragment, render_metrics_prefix, render_metrics_runtime, render_stats_json,
    render_stats_json_ex, render_stats_memgov_json, MemgovSnap, ReconcileSnap, RouteMetrics,
    RuntimeMetrics, GOV_CMP_NAMES, GOV_CONSUMER_NAMES, GOV_STATUS_NAMES, MEM_CLASS_NAMES,
    MEM_DOMAIN_NAMES, RECLAIM_STATUS_NAMES, REJECT_REASON_NAMES, REJLANE_NAMES, ROUTE_LANE_NAMES,
    ROUTE_SURFACE_NAMES, THINK_MODE_NAMES,
};
pub use models::{
    append_model_json_values, json_models_array_dup, model_alias_disables_thinking,
    model_alias_enables_thinking, model_alias_known, model_id_from_gguf_path, model_id_known,
    model_one_json, models_list_json,
};
pub use parse::{
    default_temperature, parse_anthropic_request, parse_chat_request, parse_completion_request,
    parse_request, parse_responses_request, ChatMsg, ChatPart, ImageMime, ParseEnv, ParsedRequest,
    RequestImage, ToolCall, ToolChoice, ToolSchemaOrder, DEFAULT_MIN_P, DEFAULT_TEMPERATURE,
    DEFAULT_TOP_P,
};
pub use render::{
    append_tool_result_text, render_chat, render_chat_choice, render_dots3_chat, render_dsml_chat,
    render_dsml_chat_choice, render_exaone_chat, render_live_tool_tail, render_motif3_chat,
    render_motif3_chat_ex, render_qwen_chat_ex, render_solar_chat, render_solar_chat_ex,
    role_is_system, role_is_user_like, syntax_for_model_id, think_effort_prefix, tool_start_marker,
    ModelSyntax, RenderError, DSML_ASSISTANT, DSML_BOS, DSML_EOS, DSML_USER, THINK_HIGH_PREFIX,
    THINK_MAX_PREFIX,
};
pub use retry::dump_script as retry_dump_script;
pub use route::{
    compute_needs, decode_budget, parse_reasoning_effort_name, route_decide, think_mode_enabled,
    think_mode_from_enabled, wire_surface_for, Api, NeedInput, ReqKind, RouteDecision, RouteEnv,
    ThinkMode, WireSurface, LANE_CONTINUOUS, LANE_NONE, LANE_SERIAL, LANE_STATIC,
    NEED_BANK_FRONTIER, NEED_CONTINUATION_PUBLISH, NEED_CORRECTIVE_RECOVERY, NEED_DURABLE_RESPONSE,
    NEED_IMAGE, NEED_LIVE_FRONTIER, NEED_PER_ROW_SAMPLING, NEED_PREFILL_ONLY, NEED_STOP_SCAN,
    NEED_STREAMING, NEED_THINKING, NEED_TOKEN_IDS, NEED_TOOL_SCAN, REASON_COALESCE_OFF,
    REASON_CONT, REASON_CONT_BANK, REASON_CONT_UNAVAILABLE, REASON_NAMES,
    REASON_NEED_CONTINUATION_PUBLISH, REASON_NEED_CORRECTIVE_RECOVERY, REASON_NEED_DURABLE,
    REASON_NEED_LIVE_FRONTIER, REASON_NEED_PREFILL_ONLY, REASON_STATIC_NO_CONT,
    REASON_STATIC_PROMPT_BOUNDS, REASON_SURFACE, REASON_TOKEN_IDS_PROJECTION,
    REASON_TOOLS_COMPLETION,
};
pub use serve::{
    accept_loop, accept_loop_with_engine, accept_loop_with_engine_cont, handle_client,
    handle_client_inner, listen, ServerConfig, ServerInner,
};
#[cfg(feature = "native")]
pub use serve_cont::ContLane;
pub use serve_cont::{cont_prompt_tokens, ContExec, ContStep, ContStepper};
pub use stream::{
    anthropic_final_response, anthropic_sse_finish_live, anthropic_sse_start_live,
    anthropic_sse_stream_update, append_tool_call_deltas_json, append_tool_calls_json,
    final_response, openai_sse_finish_live, openai_sse_stream_update, openai_stream_start,
    project_anthropic_thinking, project_openai_chat_thinking, project_openai_chat_utf8,
    project_openai_completion, project_responses_thinking, responses_final_response,
    responses_sse_created, responses_sse_finish_live, responses_sse_stream_update,
    responses_stream_init, sse_chunk, sse_done, sse_headers, think_end, think_start, unix_now,
    utf8_stream_safe_len, utf8_trim_tail, ChatFormat, ReqTimings, StreamReq, Writer, CREATED_TEST,
    TAPE_PLAIN, TAPE_THINKING, TAPE_UTF8, TEST_MSG_ID, TEST_RESP_ID, TEST_RS_ID,
};
pub use tool_stream::{dump_script as tool_stream_dump_script, DsmlToolState, DsmlToolStream};
pub use tools::{
    assign_tool_ids, parse_generated_for_model_id, parse_generated_for_response,
    parse_generated_message, ParsedGenerated, SemAccum, SemFeed,
};
pub use worker::{server_launch, ServerLaunch, WORKER_REQUIRES_MODEL};
#[cfg(feature = "native")]
pub use worker_run::run_assembled_worker;
