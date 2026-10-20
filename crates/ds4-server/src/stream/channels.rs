//! Incremental channel framing over the accumulator's append-only raw bytes,
//! for the channel-framed families: Inkling and Beam's Harmony.

use std::ops::Range;

use super::ChatFormat;
use crate::render::inkling;
use crate::tools::beam;

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

/// A family's message framing. Only a tool header may follow text (its
/// recipient). A closer in `ends` reopens a header (a closer equal to `lead`
/// stays to lead it); one in `last` ends the generation.
#[derive(Debug)]
pub(super) struct Grammar {
    // Skipped before a header: Inkling's model role, Beam's assistant start.
    lead: &'static str,
    // At a header: the generation is over.
    done: &'static [&'static str],
    headers: [(&'static str, Channel); 3],
    ends: &'static [&'static str],
    last: &'static [&'static str],
}

static INKLING: Grammar = Grammar {
    lead: inkling::MODEL,
    done: &[inkling::EOS],
    headers: [
        (inkling::TEXT, Channel::Text),
        (inkling::THINK, Channel::Thinking),
        (inkling::INVOKE, Channel::Tool),
    ],
    ends: &[inkling::END],
    last: &[],
};

// The template also closes analysis at a bare `<|start|>assistant`.
static BEAM: Grammar = Grammar {
    lead: beam::ASSISTANT,
    done: &[beam::RETURN, beam::CALL],
    headers: [
        (beam::FINAL, Channel::Text),
        (beam::ANALYSIS, Channel::Thinking),
        (beam::TOOL, Channel::Tool),
    ],
    ends: &[beam::END, beam::ASSISTANT],
    last: &[beam::RETURN, beam::CALL],
};

#[derive(Debug)]
pub(super) struct Channels {
    grammar: &'static Grammar,
    pos: usize,
    channel: Channel,
    text: Vec<Range<usize>>,
    thinking: Vec<Range<usize>>,
}

fn find(hay: &[u8], needle: &str) -> Option<usize> {
    hay.windows(needle.len())
        .position(|s| s == needle.as_bytes())
}

impl Channels {
    pub(super) fn for_format(format: ChatFormat) -> Option<Self> {
        let grammar = match format {
            ChatFormat::Inkling => &INKLING,
            ChatFormat::Beam => &BEAM,
            _ => return None,
        };
        Some(Self {
            grammar,
            pos: 0,
            channel: Channel::Header,
            text: Vec::new(),
            thinking: Vec::new(),
        })
    }

    pub(super) fn advance(&mut self, raw: &[u8], flush: Flush) -> Option<Fragment> {
        let g = self.grammar;
        while self.pos < raw.len() && self.channel != Channel::Done {
            if self.channel == Channel::Header {
                if raw[self.pos..].starts_with(g.lead.as_bytes()) {
                    self.pos += g.lead.len();
                }
                if g.done
                    .iter()
                    .any(|d| raw[self.pos..].starts_with(d.as_bytes()))
                {
                    self.channel = Channel::Done;
                    return None;
                }
                let (at, marker, channel) = g
                    .headers
                    .iter()
                    .filter_map(|&(marker, channel)| {
                        find(&raw[self.pos..], marker).map(|at| (self.pos + at, marker, channel))
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
            let closers = || g.ends.iter().chain(g.last);
            let end = closers()
                .filter_map(|&m| find(&raw[self.pos..], m).map(|at| (self.pos + at, m)))
                .min_by_key(|&(at, _)| at);
            let mut limit = end.map_or(raw.len(), |(at, _)| at);
            if end.is_none() && flush == Flush::More {
                // Hold only a suffix that could complete a closer.
                limit -= closers()
                    .filter_map(|m| {
                        (1..m.len().min(raw.len() - self.pos + 1))
                            .rev()
                            .find(|&held| raw.ends_with(&m.as_bytes()[..held]))
                    })
                    .max()
                    .unwrap_or(0);
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
            if let Some((at, marker)) = end {
                self.pos = if marker == g.lead {
                    at
                } else {
                    at + marker.len()
                };
                self.channel = if g.last.contains(&marker) {
                    Channel::Done
                } else {
                    Channel::Header
                };
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

#[cfg(test)]
mod tests {
    use super::*;

    fn frames(raw: &[u8], split: usize) -> (Vec<u8>, Vec<u8>, usize) {
        let mut ch = Channels::for_format(ChatFormat::Beam).unwrap();
        let (mut thinking, mut text, mut tools) = (Vec::new(), Vec::new(), 0);
        for (cut, flush) in [(split, Flush::More), (raw.len(), Flush::Final)] {
            while let Some(f) = ch.advance(&raw[..cut], flush) {
                match f.channel {
                    Channel::Thinking => thinking.extend_from_slice(&raw[f.body]),
                    Channel::Text => text.extend_from_slice(&raw[f.body]),
                    Channel::Tool => tools += usize::from(f.closed),
                    _ => {}
                }
            }
        }
        (thinking, text, tools)
    }

    #[test]
    fn beam_frames_every_split() {
        let raw = b"<|channel|>analysis<|message|>plan it<|end|><|start|>assistant<|channel|>final<|message|>answer";
        for split in 0..=raw.len() {
            assert_eq!(
                frames(raw, split),
                (b"plan it".to_vec(), b"answer".to_vec(), 0),
                "split {split}"
            );
        }
        let raw = "<|channel|>analysis<|message|>p<|end|><|start|>assistant to=f<|channel|>analysis \
                   <|constrain|>xml<|message|><rfl:parameter name=\"a\">\n1\n</rfl:parameter><|call|>";
        for split in 0..=raw.len() {
            assert_eq!(
                frames(raw.as_bytes(), split),
                (b"p".to_vec(), Vec::new(), 1),
                "split {split}"
            );
        }
    }
}
