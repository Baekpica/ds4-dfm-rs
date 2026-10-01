//! Buffer each XML call until complete; emit ordinary text around calls.
use std::ops::Range;

use crate::parse::{ToolCall, ToolSchemaOrder};
use crate::render::{IQUEST_TOOL_CALL_END, IQUEST_TOOL_CALL_START};
use crate::tools::iquest::call_body;

use super::utf8_stream_safe_len;

#[derive(Debug)]
pub(super) enum Fragment {
    Text(Range<usize>),
    Call(i32, ToolCall),
}

#[derive(Debug, Default)]
pub(super) struct Channels {
    pub(super) pos: usize,
    pub(super) calls: i32,
    pub(super) text: Vec<u8>,
}

impl Channels {
    pub(super) fn advance(
        &mut self,
        raw: &[u8],
        final_: bool,
        orders: &[ToolSchemaOrder],
    ) -> Option<Fragment> {
        if self.pos >= raw.len() {
            return None;
        }
        let start = IQUEST_TOOL_CALL_START.as_bytes();
        let end = IQUEST_TOOL_CALL_END.as_bytes();
        let tail = &raw[self.pos..];
        let marker = tail.windows(start.len()).position(|x| x == start);
        let limit = if let Some(rel) = marker {
            if rel == 0 {
                let first = self.pos + start.len();
                if let Some(rel) = raw[first..].windows(end.len()).position(|x| x == end) {
                    let close = first + rel;
                    let next = close + end.len();
                    if let Some(call) = call_body(&raw[first..close], orders) {
                        self.pos = next;
                        let index = self.calls;
                        self.calls += 1;
                        return Some(Fragment::Call(index, call));
                    }
                    next
                } else if final_ {
                    raw.len()
                } else {
                    return None;
                }
            } else {
                self.pos + rel
            }
        } else if final_ {
            raw.len()
        } else {
            let keep = (1..start.len().min(tail.len() + 1))
                .rev()
                .find(|&n| tail.ends_with(&start[..n]))
                .unwrap_or(0);
            raw.len() - keep
        };
        let limit = utf8_stream_safe_len(raw, self.pos, limit, final_);
        if limit == self.pos {
            return None;
        }
        let span = self.pos..limit;
        self.text.extend_from_slice(&raw[span.clone()]);
        self.pos = limit;
        Some(Fragment::Text(span))
    }
}
