//! v0.6.3 schema-format refusal (ds4-on-spark#10).

use crate::json::{json_skip_value, json_string, Json};
use crate::route::{parse_reasoning_effort_name, ThinkMode};

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

pub fn parse_reasoning_effort_value(p: &mut Json<'_>) -> Result<Option<ThinkMode>, String> {
    p.ws();
    if p.lit("null") {
        return Ok(None);
    }
    let name = json_string(p).ok_or_else(|| String::new())?;
    parse_reasoning_effort_name(&name)
        .map(Some)
        .ok_or_else(|| String::new())
}

pub fn parse_output_config_effort(p: &mut Json<'_>) -> Result<Option<ThinkMode>, String> {
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
