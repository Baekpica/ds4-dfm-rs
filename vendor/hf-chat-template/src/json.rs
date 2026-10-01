//! Transformers' `tojson`: Python JSON spacing, ordering and number spelling.
//! This compatibility code is shared by every model; templates stay unchanged.

use minijinja::value::{DynObject, Kwargs, Object, ObjectRepr, Value, ValueKind};
use minijinja::{Error, ErrorKind};
use serde::ser::{Error as _, SerializeMap, SerializeSeq};
use serde::{Serialize, Serializer};
use serde_json::Value as Json;
use std::cmp::Ordering;
use std::fmt::Write;
use std::sync::Arc;

#[derive(Debug)]
struct WideInteger(serde_json::Number);

impl Object for WideInteger {
    fn repr(self: &Arc<Self>) -> ObjectRepr {
        ObjectRepr::Plain
    }

    fn render(self: &Arc<Self>, out: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(out, "{}", self.0)
    }

    fn custom_cmp(self: &Arc<Self>, other: &DynObject) -> Option<Ordering> {
        let other = other.downcast_ref::<Self>()?;
        let left = self.0.to_string();
        let right = other.0.to_string();
        let left_negative = left.starts_with('-');
        let right_negative = right.starts_with('-');
        if left_negative != right_negative {
            return Some(right_negative.cmp(&left_negative));
        }
        // JSON integer lexemes have no leading zeroes. Length then digits
        // orders magnitudes exactly without converting through a float.
        let order = left.len().cmp(&right.len()).then_with(|| left.cmp(&right));
        Some(if left_negative {
            order.reverse()
        } else {
            order
        })
    }
}

pub(crate) fn is_wide_integer(value: &Value) -> bool {
    value.downcast_object_ref::<WideInteger>().is_some()
}

/// serde_json arbitrary-precision numbers serialize as private structs.
/// Convert JSON explicitly so templates still receive native scalar values.
pub(crate) fn from_json(value: &Json) -> Result<Value, Error> {
    Ok(match value {
        Json::Null => Value::from(()),
        Json::Bool(value) => Value::from(*value),
        Json::String(value) => Value::from(value.clone()),
        Json::Array(items) => items
            .iter()
            .map(from_json)
            .collect::<Result<Vec<_>, _>>()?
            .into_iter()
            .collect(),
        Json::Object(items) => items
            .iter()
            .map(|(key, value)| Ok((key.clone(), from_json(value)?)))
            .collect::<Result<Vec<_>, Error>>()?
            .into_iter()
            .collect(),
        Json::Number(value) => {
            let text = value.to_string();
            if let Some(value) = value.as_i64() {
                Value::from(value)
            } else if let Some(value) = value.as_u64() {
                Value::from(value)
            } else if let Ok(value) = text.parse::<i128>() {
                Value::from(value)
            } else if text.contains(['.', 'e', 'E']) {
                Value::from(
                    value
                        .as_f64()
                        .filter(|v| v.is_finite())
                        .ok_or_else(|| invalid("float exceeds Jinja range"))?,
                )
            } else {
                // MiniJinja coerces U128 arithmetic to i128. Keep every
                // integer outside signed i128 opaque so arithmetic rejects
                // instead of wrapping, while display and JSON remain exact.
                Value::from_object(WideInteger(value.clone()))
            }
        }
    })
}

struct JsonValue<'a>(&'a Value);

impl Serialize for JsonValue<'_> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        if let Some(integer) = self.0.downcast_object_ref::<WideInteger>() {
            return integer.0.serialize(serializer);
        }
        match self.0.kind() {
            ValueKind::Seq | ValueKind::Iterable => {
                let mut seq = serializer.serialize_seq(self.0.len())?;
                for value in self.0.try_iter().map_err(S::Error::custom)? {
                    seq.serialize_element(&JsonValue(&value))?;
                }
                seq.end()
            }
            ValueKind::Map => {
                let mut map = serializer.serialize_map(self.0.len())?;
                for key in self.0.try_iter().map_err(S::Error::custom)? {
                    let value = self.0.get_item(&key).map_err(S::Error::custom)?;
                    map.serialize_entry(&key, &JsonValue(&value))?;
                }
                map.end()
            }
            _ => self.0.serialize(serializer),
        }
    }
}

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
                out.push_str(&float_repr(value.as_f64().expect("f64")));
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

pub(crate) fn float_repr(value: f64) -> String {
    let number = format!("{value:?}");
    if let Some((mantissa, exponent)) = number.split_once('e') {
        let exponent: i32 = exponent.parse().expect("float exponent");
        return format!("{mantissa}e{exponent:+03}");
    }
    number
}

pub(crate) fn tojson_filter(value: Value, kwargs: Kwargs) -> Result<String, Error> {
    if value.is_undefined() {
        return Err(invalid("undefined value"));
    }
    let options = Options::parse(kwargs)?;
    let value = serde_json::to_value(JsonValue(&value)).map_err(invalid)?;
    let mut out = String::new();
    options.write(&value, &mut out, 0);
    Ok(out)
}
