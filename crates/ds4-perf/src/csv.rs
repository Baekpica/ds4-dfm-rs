use std::io::{self, BufRead};

// Read records incrementally: GPU traces can contain millions of launches.
// Quotes, escaped quotes and embedded newlines are legal in Nsight names.
pub fn records(
    reader: impl BufRead,
    mut visit: impl FnMut(Vec<String>) -> Result<(), String>,
) -> Result<(), String> {
    let mut field = Vec::new();
    let mut row = Vec::new();
    let mut quoted = false;
    let mut closed = false;
    for byte in reader.bytes() {
        let byte = byte.map_err(|e| e.to_string())?;
        if quoted {
            if byte == b'"' {
                quoted = false;
                closed = true;
            } else {
                field.push(byte);
            }
            continue;
        }
        match byte {
            b'"' if closed => {
                field.push(b'"');
                quoted = true;
                closed = false;
            }
            b'"' if field.is_empty() => quoted = true,
            b',' | b'\n' => {
                row.push(take_field(&mut field)?);
                closed = false;
                if byte == b'\n' {
                    visit(std::mem::take(&mut row))?;
                }
            }
            b'\r' => {}
            b' ' | b'\t' if closed => {}
            _ if closed => return Err("characters after closing CSV quote".into()),
            _ => field.push(byte),
        }
    }
    if quoted {
        return Err("unterminated CSV quote".into());
    }
    if !field.is_empty() || !row.is_empty() || closed {
        row.push(take_field(&mut field)?);
        visit(row)?;
    }
    Ok(())
}

fn take_field(bytes: &mut Vec<u8>) -> Result<String, String> {
    String::from_utf8(std::mem::take(bytes)).map_err(|e| format!("CSV is not UTF-8: {e}"))
}

pub fn field(value: &str) -> String {
    if value.contains([',', '"', '\r', '\n']) {
        format!("\"{}\"", value.replace('"', "\"\""))
    } else {
        value.into()
    }
}

pub fn key(value: &str) -> String {
    value
        .trim()
        .trim_start_matches('\u{feff}')
        .to_ascii_lowercase()
}

pub fn column(header: &[String], names: &[&str]) -> Option<usize> {
    header.iter().position(|s| names.contains(&key(s).as_str()))
}

pub fn value<'a>(row: &'a [String], header: &[String], names: &[&str]) -> Option<&'a str> {
    row.get(column(header, names)?)
        .map(|s| s.trim())
        .filter(|s| !s.is_empty())
}

pub fn number(value: &str) -> Result<f64, String> {
    value
        .trim()
        .parse::<f64>()
        .ok()
        .filter(|v| v.is_finite() && *v >= 0.0)
        .ok_or_else(|| format!("invalid nonnegative number: {value}"))
}

pub fn time(row: &[String], header: &[String], name: &str) -> Option<f64> {
    for (unit, factor) in [
        ("ns", 1.0),
        ("us", 1_000.0),
        ("ms", 1_000_000.0),
        ("s", 1_000_000_000.0),
    ] {
        let col = format!("{} ({unit})", name.to_ascii_lowercase());
        if let Some(value) = value(row, header, &[&col]) {
            return number(value)
                .ok()
                .map(|v| v * factor)
                .filter(|v| v.is_finite());
        }
    }
    None
}

pub fn open(path: &std::path::Path) -> io::Result<io::BufReader<std::fs::File>> {
    std::fs::File::open(path).map(io::BufReader::new)
}

pub fn table(
    reader: impl BufRead,
    marker: &str,
    mut visit: impl FnMut(&[String], &[String]) -> Result<(), String>,
) -> Result<(), String> {
    let mut header = Vec::new();
    records(reader, |row| {
        if column(&row, &[marker]).is_some() {
            header = row;
            return Ok(());
        }
        if header.is_empty() || row.iter().all(|s| s.trim().is_empty()) {
            return Ok(());
        }
        if row.len() != header.len() {
            return Err("unexpected CSV record width after header".into());
        }
        visit(&header, &row)
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quoted_templates_and_newlines() {
        let input = "Name,Count\r\n\"foo<int, 128, bar<float>>\",2\r\n\"line\n\"\"two\"\"\",3";
        let mut rows = Vec::new();
        records(input.as_bytes(), |row| {
            rows.push(row);
            Ok(())
        })
        .unwrap();
        assert_eq!(rows[1], ["foo<int, 128, bar<float>>", "2"]);
        assert_eq!(rows[2], ["line\n\"two\"", "3"]);
        assert!(records(b"\"bad".as_slice(), |_| Ok(())).is_err());
        for invalid in ["NaN", "inf", "-1"] {
            assert!(number(invalid).is_err());
        }
    }
}
