//! v0.6.3 schema-format refusal (ds4-on-spark#10).

use crate::json::{json_skip_value, json_string, Json};
use crate::route::{parse_reasoning_effort_name, RequestedEffort, ThinkMode};

pub fn output_format_type_supported(field: &str, typ: &str) -> Result<(), String> {
    if typ == "text" {
        return Ok(());
    }
    if typ == "json_object" || typ == "json_schema" {
        return Err(format!(
            "{field} type '{typ}' is not supported: structured output is \
             unsupported; omit {field} or use type \"text\""
        ));
    }
    Err(format!("{field} type '{typ}' is not supported"))
}

pub fn parse_output_format_value(p: &mut Json<'_>, field: &str) -> Result<(), String> {
    p.ws();
    if p.lit("null") {
        return Ok(());
    }
    if p.peek() == Some(b'"') {
        let typ = json_string(p).ok_or_else(|| String::new())?;
        return output_format_type_supported(field, &typ);
    }
    if p.bump() != Some(b'{') {
        return Err(String::new());
    }
    p.ws();
    let mut typ: Option<String> = None;
    while p.peek().is_some() && p.peek() != Some(b'}') {
        let key = json_string(p).ok_or_else(|| String::new())?;
        p.ws();
        if p.bump() != Some(b':') {
            return Err(String::new());
        }
        if key == "type" {
            p.ws();
            typ = Some(json_string(p).ok_or_else(|| String::new())?);
        } else if !json_skip_value(p) {
            return Err(String::new());
        }
        p.ws();
        if p.peek() == Some(b',') {
            p.i += 1;
        }
        p.ws();
    }
    if p.bump() != Some(b'}') {
        return Err(String::new());
    }
    if let Some(t) = typ {
        output_format_type_supported(field, &t)?;
    }
    Ok(())
}

pub fn parse_responses_text_value(p: &mut Json<'_>) -> Result<(), String> {
    p.ws();
    if p.lit("null") {
        return Ok(());
    }
    if p.peek() != Some(b'{') {
        if json_skip_value(p) {
            return Ok(());
        }
        return Err(String::new());
    }
    p.i += 1;
    p.ws();
    while p.peek().is_some() && p.peek() != Some(b'}') {
        let key = json_string(p).ok_or_else(|| String::new())?;
        p.ws();
        if p.bump() != Some(b':') {
            return Err(String::new());
        }
        if key == "format" {
            parse_output_format_value(p, "text.format")?;
        } else if !json_skip_value(p) {
            return Err(String::new());
        }
        p.ws();
        if p.peek() == Some(b',') {
            p.i += 1;
        }
        p.ws();
    }
    if p.bump() != Some(b'}') {
        return Err(String::new());
    }
    Ok(())
}

/// The `reasoning_effort` value: a name from the shared vocabulary, or the
/// oracle's numeric form -- an integer in [1,100] renders as itself
/// (encoding.py:456-473).  A number turns thinking on and takes the engine's
/// collapsed HIGH mode: only the head line carries the number.
pub fn parse_reasoning_effort_value(p: &mut Json<'_>) -> Result<Option<RequestedEffort>, String> {
    p.ws();
    if p.lit("null") {
        return Ok(None);
    }
    if let Some(budget) = json_effort_budget(p) {
        return Ok(Some(RequestedEffort {
            mode: ThinkMode::High,
            budget: Some(budget),
        }));
    }
    let name = json_string(p).ok_or_else(|| String::new())?;
    parse_reasoning_effort_name(&name)
        .map(RequestedEffort::named)
        .map(Some)
        .ok_or_else(|| String::new())
}

/// A bare integer token in [1,100]: the oracle asserts `type(effort) is int`
/// (encoding.py:464), so a fraction or an exponent is not a budget and falls
/// through to the refusal.  An integer outside the range is refused there too.
fn json_effort_budget(p: &mut Json<'_>) -> Option<u8> {
    p.ws();
    let start = p.i;
    let mut v: u32 = 0;
    while let Some(c) = p.peek() {
        if !c.is_ascii_digit() {
            break;
        }
        v = (v * 10 + (c - b'0') as u32).min(u32::from(u8::MAX));
        p.i += 1;
    }
    if p.i == start || matches!(p.peek(), Some(b'.' | b'e' | b'E')) {
        p.i = start;
        return None;
    }
    (1..=100).contains(&v).then_some(v as u8)
}

pub fn parse_output_config_effort(p: &mut Json<'_>) -> Result<Option<RequestedEffort>, String> {
    p.ws();
    if p.lit("null") {
        return Ok(None);
    }
    if p.peek() != Some(b'{') {
        if json_skip_value(p) {
            return Ok(None);
        }
        return Err(String::new());
    }
    p.i += 1;
    p.ws();
    let mut effort = None;
    while p.peek().is_some() && p.peek() != Some(b'}') {
        let key = json_string(p).ok_or_else(|| String::new())?;
        p.ws();
        if p.bump() != Some(b':') {
            return Err(String::new());
        }
        if key == "effort" {
            effort = parse_reasoning_effort_value(p)?;
        } else if key == "format" {
            parse_output_format_value(p, "output_config.format")?;
        } else if !json_skip_value(p) {
            return Err(String::new());
        }
        p.ws();
        if p.peek() == Some(b',') {
            p.i += 1;
        }
        p.ws();
    }
    if p.bump() != Some(b'}') {
        return Err(String::new());
    }
    Ok(effort)
}

pub fn parse_output_config_format(p: &mut Json<'_>) -> Result<(), String> {
    parse_output_config_effort(p).map(|_| ())
}
