use super::approval::{Approval, DENIED_EDIT};
use super::web_tools::{io_detail, read_bytes};

mod obs;

pub(crate) const EDIT: &str = "edit";
const UPTO: &[u8] = b"[upto]";

pub(super) struct LineSpan {
    start: usize,
    content_end: usize,
    end: usize,
}

pub(crate) fn edit_result(path: Option<&str>, old: Option<&str>, new: Option<&str>) -> Vec<u8> {
    edit_result_with(path, old, new, &mut Approval::NonInteractive)
}

pub(crate) fn edit_result_with(
    path: Option<&str>,
    old: Option<&str>,
    new: Option<&str>,
    approval: &mut Approval<'_>,
) -> Vec<u8> {
    let Some(path) = path.filter(|path| !path.is_empty()) else {
        return b"Tool error: edit requires path\n".to_vec();
    };
    let Some(old) = old.filter(|old| !old.is_empty()) else {
        return b"Tool error: edit requires non-empty old text\n".to_vec();
    };
    let Some(new) = new else {
        return b"Tool error: edit requires new text\n".to_vec();
    };
    let data = match read_bytes(path) {
        Ok(data) => data,
        Err(error) => return wrap_error(&error),
    };
    if let Err(error) = find_old_span(&data, old.as_bytes()) {
        return wrap_error(error.as_bytes());
    }
    if !approval.allow_edit(path) {
        return DENIED_EDIT.to_vec();
    }
    // Approval can wait while an editor saves; apply only to the current bytes.
    let data = match read_bytes(path) {
        Ok(data) => data,
        Err(error) => return wrap_error(&error),
    };
    let (offset, remove_len, anchored) = match find_old_span(&data, old.as_bytes()) {
        Ok(found) => found,
        Err(error) => return wrap_error(error.as_bytes()),
    };
    apply_splice(path, &data, offset, remove_len, new.as_bytes(), anchored)
}

fn wrap_error(detail: &[u8]) -> Vec<u8> {
    let mut out = b"Tool error: ".to_vec();
    out.extend_from_slice(detail);
    out.push(b'\n');
    out
}

fn find_bytes(hay: &[u8], needle: &[u8]) -> Option<usize> {
    if needle.is_empty() {
        return Some(0);
    }
    hay.windows(needle.len())
        .position(|window| window == needle)
}

fn find_unique(data: &[u8], needle: &[u8], label: &str) -> Result<usize, String> {
    if needle.is_empty() {
        return Err(format!("{label} anchor is empty"));
    }
    let Some(first) = find_bytes(data, needle) else {
        return Err(format!("{label} anchor not found"));
    };
    let rest = first.saturating_add(1);
    if rest <= data.len() && find_bytes(&data[rest..], needle).is_some() {
        return Err(format!("{label} anchor is not unique"));
    }
    Ok(first)
}

fn find_unique_after(
    data: &[u8],
    start: usize,
    needle: &[u8],
    label: &str,
) -> Result<usize, String> {
    if needle.is_empty() {
        return Err(format!("{label} anchor is empty"));
    }
    if start > data.len() {
        return Err(format!("{label} search starts outside file"));
    }
    let Some(rel) = find_bytes(&data[start..], needle) else {
        return Err(format!("{label} anchor not found after old head"));
    };
    let first = start + rel;
    let rest = first.saturating_add(1);
    if rest <= data.len() && find_bytes(&data[rest..], needle).is_some() {
        return Err(format!("{label} anchor is not unique after old head"));
    }
    Ok(first)
}

fn find_old_span(data: &[u8], old: &[u8]) -> Result<(usize, usize, bool), String> {
    match find_bytes(old, UPTO) {
        None => {
            let at = find_unique(data, old, "old text")?;
            Ok((at, old.len(), false))
        }
        Some(upto) => {
            if find_bytes(&old[upto + UPTO.len()..], UPTO).is_some() {
                return Err("old text contains more than one [upto] marker".into());
            }
            let head = &old[..upto];
            let tail = &old[upto + UPTO.len()..];
            if !tail.iter().any(|byte| !byte.is_ascii_whitespace()) {
                return Err("old text after [upto] must include a unique tail anchor".into());
            }
            let head_at = find_unique(data, head, "old head")?;
            let tail_at = find_unique_after(data, head_at + head.len(), tail, "old tail")?;
            Ok((head_at, tail_at - head_at + tail.len(), true))
        }
    }
}

pub(super) fn split_lines(data: &[u8]) -> Vec<LineSpan> {
    let mut spans = Vec::new();
    let mut pos = 0;
    while pos < data.len() {
        let start = pos;
        while pos < data.len() && !matches!(data[pos], b'\n' | b'\r') {
            pos += 1;
        }
        let content_end = pos;
        if pos < data.len() {
            if data[pos] == b'\r' && data.get(pos + 1) == Some(&b'\n') {
                pos += 2;
            } else {
                pos += 1;
            }
        }
        spans.push(LineSpan {
            start,
            content_end,
            end: pos,
        });
    }
    spans
}

fn line_for_offset(spans: &[LineSpan], offset: usize) -> i32 {
    if spans.is_empty() {
        return 1;
    }
    for (index, span) in spans.iter().enumerate() {
        if offset < span.end {
            return (index + 1) as i32;
        }
    }
    spans.len() as i32
}

fn write_bytes(path: &str, data: &[u8]) -> Result<(), String> {
    super::file_write::replace(path.as_ref(), data)
        .map_err(|(stage, error)| format!("{stage} {path}: {}", io_detail(&error)))
}

fn apply_splice(
    path: &str,
    data: &[u8],
    offset: usize,
    remove_len: usize,
    insert: &[u8],
    anchored: bool,
) -> Vec<u8> {
    let mut out =
        Vec::with_capacity(offset + insert.len() + data.len().saturating_sub(offset + remove_len));
    out.extend_from_slice(&data[..offset]);
    out.extend_from_slice(insert);
    out.extend_from_slice(&data[offset + remove_len..]);
    if let Err(error) = write_bytes(path, &out) {
        return wrap_error(error.as_bytes());
    }
    let old_spans = split_lines(data);
    let new_spans = split_lines(&out);
    let kind = if anchored {
        "anchored old/new replacement"
    } else {
        "old/new replacement"
    };
    if old_spans.is_empty() {
        return format!("Edited {path} using {kind}\n").into_bytes();
    }
    let old_last = if remove_len > 0 {
        offset + remove_len - 1
    } else {
        offset
    }
    .min(data.len().saturating_sub(1));
    let start_line = line_for_offset(&old_spans, offset);
    let end_line = line_for_offset(&old_spans, old_last);
    let delta = new_spans.len() as i32 - old_spans.len() as i32;
    obs::format_edit_result(path, start_line, end_line, delta, &out, kind)
}
