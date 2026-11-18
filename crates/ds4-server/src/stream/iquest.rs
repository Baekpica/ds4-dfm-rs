//! Buffer each XML call until complete; emit ordinary text around calls.
use std::ops::Range;

use crate::parse::{ToolCall, ToolSchemaOrder};
use crate::render::{IQUEST_TOOL_CALL_END, IQUEST_TOOL_CALL_START};
use crate::tools::iquest::call_body;

use super::{utf8_stream_safe_len, ChatFormat};
use crate::tools::hy4;
use std::collections::VecDeque;

#[derive(Debug)]
pub(super) enum Fragment {
    Text(Range<usize>),
    Call(i32, ToolCall),
}

#[derive(Debug, Default)]
pub(super) struct Channels {
    hy4: bool,
    pending: VecDeque<ToolCall>,
    pub(super) pos: usize,
    pub(super) calls: i32,
    pub(super) text: Vec<u8>,
}

impl Channels {
    pub(super) fn for_tools(format: ChatFormat, enabled: bool) -> Option<Self> {
        (enabled && matches!(format, ChatFormat::IQuestQ1 | ChatFormat::Hy4)).then(|| Self {
            hy4: format == ChatFormat::Hy4,
            ..Default::default()
        })
    }

    fn next_call(&mut self) -> Option<Fragment> {
        let call = self.pending.pop_front()?;
        let index = self.calls;
        self.calls += 1;
        Some(Fragment::Call(index, call))
    }

    pub(super) fn advance(
        &mut self,
        raw: &[u8],
        final_: bool,
        orders: &[ToolSchemaOrder],
    ) -> Option<Fragment> {
        if let Some(call) = self.next_call() {
            return Some(call);
        }
        if self.pos >= raw.len() {
            return None;
        }
        let (start, end) = if self.hy4 {
            (hy4::START, hy4::END)
        } else {
            (IQUEST_TOOL_CALL_START, IQUEST_TOOL_CALL_END)
        };
        let (start, end) = (start.as_bytes(), end.as_bytes());
        let tail = &raw[self.pos..];
        let marker = tail.windows(start.len()).position(|x| x == start);
        let limit = if let Some(rel) = marker {
            if rel == 0 {
                let first = self.pos + start.len();
                if let Some(rel) = raw[first..].windows(end.len()).position(|x| x == end) {
                    let close = first + rel;
                    let next = close + end.len();
                    let body = &raw[first..close];
                    if self.hy4 {
                        if let Some(calls) = hy4::group(body, orders) {
                            self.pending.extend(calls);
                        }
                    } else if let Some(call) = call_body(body, orders) {
                        self.pending.push_back(call);
                    }
                    if let Some(call) = self.next_call() {
                        self.pos = next;
                        return Some(call);
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
