//! Transformers' `tojson`: Python JSON spacing, ordering and number spelling.
//! This compatibility code is shared by every model; templates stay unchanged.

use minijinja::value::{Kwargs, Value};
use minijinja::{Error, ErrorKind};
use serde_json::Value as Json;
use std::fmt::Write;

struct Options {
    indent: Option<String>,
    item_sep: String,
    key_sep: String,
    ensure_ascii: bool,
    sort_keys: bool,
}

fn invalid(message: impl std::fmt::Display) -> Error {
    Error::new(ErrorKind::InvalidOperation, format!("tojson: {message}"))
}

impl Options {
    fn parse(kwargs: Kwargs) -> Result<Self, Error> {
        let indent: Option<Value> = kwargs.get("indent")?;
        let indent = match indent {
            None => None,
            Some(value) if value.is_none() => None,
            Some(value) => {
                if let Some(text) = value.as_str() {
                    Some(text.to_owned())
                } else {
                    let size = value.as_i64().ok_or_else(|| invalid("invalid indent"))?;
                    // Bound allocation for erroneous artifact data.
                    if size > 4096 {
                        return Err(invalid("indent exceeds 4096"));
                    }
                    Some(" ".repeat(size.max(0) as usize))
                }
            }
        };
        let separators: Option<Value> = kwargs.get("separators")?;
        let (item_sep, key_sep) = match separators {
            Some(value) if !value.is_none() => {
                let parts = value.try_iter()?.collect::<Vec<_>>();
                if parts.len() != 2 {
                    return Err(invalid("separators needs two strings"));
                }
                let item = parts[0]
                    .as_str()
                    .ok_or_else(|| invalid("invalid separator"))?;
                let key = parts[1]
                    .as_str()
                    .ok_or_else(|| invalid("invalid separator"))?;
                (item.to_owned(), key.to_owned())
            }
            _ => (
                if indent.is_some() { "," } else { ", " }.into(),
                ": ".into(),
            ),
        };
        let ensure_ascii = kwargs.get::<Option<bool>>("ensure_ascii")?.unwrap_or(false);
        let sort_keys = kwargs.get::<Option<bool>>("sort_keys")?.unwrap_or(false);
        kwargs.assert_all_used()?;
        Ok(Self {
            indent,
            item_sep,
            key_sep,
            ensure_ascii,
            sort_keys,
        })
    }

    fn line(&self, out: &mut String, depth: usize) {
        if let Some(indent) = &self.indent {
            out.push('\n');
            for _ in 0..depth {
                out.push_str(indent);
            }
        }
    }

    fn string(&self, text: &str, out: &mut String) {
        // serde supplies control/quote escaping. Python additionally escapes
        // DEL and non-ASCII as UTF-16 code units when ensure_ascii is enabled.
        let escaped = serde_json::to_string(text).expect("string serialization");
        if !self.ensure_ascii {
            out.push_str(&escaped);
            return;
        }
        for ch in escaped.chars() {
            if ch < '\u{7f}' {
                out.push(ch);
                continue;
            }
            for unit in ch.encode_utf16(&mut [0; 2]) {
                write!(out, "\\u{unit:04x}").expect("string write");
            }
        }
    }

    fn write(&self, value: &Json, out: &mut String, depth: usize) {
        match value {
            Json::Null => out.push_str("null"),
            Json::Bool(value) => out.push_str(if *value { "true" } else { "false" }),
            Json::String(value) => self.string(value, out),
            Json::Number(value) if value.is_f64() => {
                // Rust's shortest round-trip Debug uses Python's notation
                // boundaries. Python pads the exponent and always signs it.
                let number = format!("{:?}", value.as_f64().expect("f64"));
                if let Some((mantissa, exponent)) = number.split_once('e') {
                    let exponent: i32 = exponent.parse().expect("float exponent");
                    write!(out, "{mantissa}e{exponent:+03}").expect("string write");
                } else {
                    out.push_str(&number);
                }
            }
            Json::Number(value) => write!(out, "{value}").expect("string write"),
            Json::Array(items) => {
                out.push('[');
                for (i, item) in items.iter().enumerate() {
                    if i != 0 {
                        out.push_str(&self.item_sep);
                    }
                    self.line(out, depth + 1);
                    self.write(item, out, depth + 1);
                }
                if !items.is_empty() {
                    self.line(out, depth);
                }
                out.push(']');
            }
            Json::Object(items) => {
                out.push('{');
                let mut items = items.iter().collect::<Vec<_>>();
                if self.sort_keys {
                    items.sort_by(|a, b| a.0.cmp(b.0));
                }
                for (i, (key, value)) in items.iter().enumerate() {
                    if i != 0 {
                        out.push_str(&self.item_sep);
                    }
                    self.line(out, depth + 1);
                    self.string(key, out);
                    out.push_str(&self.key_sep);
                    self.write(value, out, depth + 1);
                }
                if !items.is_empty() {
                    self.line(out, depth);
                }
                out.push('}');
            }
        }
    }
}

pub(crate) fn tojson_filter(value: Value, kwargs: Kwargs) -> Result<String, Error> {
    if value.is_undefined() {
        return Err(invalid("undefined value"));
    }
    let options = Options::parse(kwargs)?;
    let value = serde_json::to_value(&value).map_err(invalid)?;
    let mut out = String::new();
    options.write(&value, &mut out, 0);
    Ok(out)
}
