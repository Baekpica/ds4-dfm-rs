//! Incremental channel framing over the accumulator's append-only raw bytes.

use std::ops::Range;

use crate::render::inkling::{END, EOS, INVOKE, MODEL, TEXT, THINK};

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(super) enum Channel {
    #[default]
    Header,
    Text,
    Thinking,
    Tool,
    Done,
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum Flush {
    More,
    Final,
}

pub(super) struct Fragment {
    pub(super) channel: Channel,
    pub(super) body: Range<usize>,
    pub(super) closed: bool,
}

#[derive(Debug, Default)]
pub(super) struct Channels {
    pos: usize,
    channel: Channel,
    text: Vec<Range<usize>>,
    thinking: Vec<Range<usize>>,
}

impl Channels {
    pub(super) fn advance(&mut self, raw: &[u8], flush: Flush) -> Option<Fragment> {
        while self.pos < raw.len() && self.channel != Channel::Done {
            if self.channel == Channel::Header {
                if raw[self.pos..].starts_with(MODEL.as_bytes()) {
                    self.pos += MODEL.len();
                }
                if raw[self.pos..].starts_with(EOS.as_bytes()) {
                    self.channel = Channel::Done;
                    return None;
                }
                let (at, marker, channel) = [
                    (TEXT, Channel::Text),
                    (THINK, Channel::Thinking),
                    (INVOKE, Channel::Tool),
                ]
                .into_iter()
                .filter_map(|(marker, channel)| {
                    raw[self.pos..]
                        .windows(marker.len())
                        .position(|s| s == marker.as_bytes())
                        .map(|at| (self.pos + at, marker, channel))
                })
                .min_by_key(|(at, _, _)| *at)?;
                if channel != Channel::Tool && at != self.pos {
                    self.channel = Channel::Done;
                    return None;
                }
                self.pos = at + marker.len();
                self.channel = channel;
                continue;
            }
            let end = raw[self.pos..]
                .windows(END.len())
                .position(|s| s == END.as_bytes())
                .map(|at| self.pos + at);
            let mut limit = end.unwrap_or(raw.len());
            if end.is_none() && flush == Flush::More {
                // Hold only a suffix that could complete the end marker.
                for held in (1..END.len().min(raw.len() - self.pos + 1)).rev() {
                    if raw.ends_with(&END.as_bytes()[..held]) {
                        limit -= held;
                        break;
                    }
                }
            }
            limit = super::utf8_stream_safe_len(raw, self.pos, limit, flush == Flush::Final);
            let body = self.pos..limit;
            let channel = self.channel;
            if body.is_empty() && end.is_none() && flush == Flush::More {
                return None;
            }
            let ranges = match channel {
                Channel::Text => Some(&mut self.text),
                Channel::Thinking => Some(&mut self.thinking),
                _ => None,
            };
            if let Some(ranges) = ranges.filter(|_| !body.is_empty()) {
                if let Some(last) = ranges.last_mut().filter(|r| r.end == body.start) {
                    last.end = body.end;
                } else {
                    ranges.push(body.clone());
                }
            }
            self.pos = limit;
            if let Some(end) = end {
                self.pos = end + END.len();
                self.channel = Channel::Header;
            } else if flush == Flush::Final {
                self.channel = Channel::Done;
            }
            return Some(Fragment {
                channel,
                body,
                closed: end.is_some(),
            });
        }
        None
    }

    pub(super) fn text(&self, raw: &[u8], channel: Channel) -> Vec<u8> {
        let ranges = if channel == Channel::Thinking {
            &self.thinking
        } else {
            &self.text
        };
        ranges
            .iter()
            .flat_map(|r| &raw[r.start.min(raw.len())..r.end.min(raw.len())])
            .copied()
            .collect()
    }
}
