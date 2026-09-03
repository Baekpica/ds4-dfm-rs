use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::Arc;

use ds4_kv::{read_trailer, Store as KvStore, EXT_TOOL_MAP};

use crate::parse::{ChatMsg, ToolCall};
use crate::render::{MOTIF_TOOL_CALLS, SOLAR_TOOL_CALLS, SOLAR_TOOL_CALL_END};
use crate::tools::{
    release_tool_id, reserve_tool_id, DSML_TOOL_CALLS_END, DSML_TOOL_CALLS_END_SHORT,
    DSML_TOOL_CALLS_START, DSML_TOOL_CALLS_START_SHORT,
};

const KTM_HEADER: &[u8; 4] = b"KTM\x01";
const DEFAULT_MAX_IDS: usize = 100_000;
const MAX_ID_BYTES: usize = 256;
const MAX_DSML_BYTES: usize = 512 * 1024 * 1024;

fn wire_lengths(id: &[u8], dsml: &[u8]) -> Option<(u32, u32)> {
    if id.is_empty() || dsml.is_empty() || id.contains(&0) || dsml.contains(&0) {
        return None;
    }
    Some((
        u32::try_from(id.len()).ok()?,
        u32::try_from(dsml.len()).ok()?,
    ))
}

fn encode_ktm_bounded(entries: &[(&[u8], &[u8])], max_bytes: usize) -> Option<Vec<u8>> {
    let mut count = 0u32;
    let mut bytes = 8u64;
    for &(id, dsml) in entries {
        let Some((id_len, dsml_len)) = wire_lengths(id, dsml) else {
            continue;
        };
        count = count.checked_add(1)?;
        bytes = bytes
            .checked_add(8)?
            .checked_add(u64::from(id_len))?
            .checked_add(u64::from(dsml_len))?;
    }
    if count == 0 {
        return Some(Vec::new());
    }
    if bytes > u64::try_from(max_bytes).unwrap_or(u64::MAX) {
        return None;
    }

    let mut out = Vec::new();
    out.try_reserve_exact(usize::try_from(bytes).ok()?).ok()?;
    out.extend_from_slice(KTM_HEADER);
    out.extend_from_slice(&count.to_le_bytes());
    for &(id, dsml) in entries {
        let Some((id_len, dsml_len)) = wire_lengths(id, dsml) else {
            continue;
        };
        out.extend_from_slice(&id_len.to_le_bytes());
        out.extend_from_slice(&dsml_len.to_le_bytes());
        out.extend_from_slice(id);
        out.extend_from_slice(dsml);
    }
    Some(out)
}

#[cfg(test)]
pub(crate) fn encode_ktm(entries: &[(&[u8], &[u8])]) -> Option<Vec<u8>> {
    encode_ktm_bounded(entries, usize::MAX)
}

pub(crate) fn decode_ktm<F>(bytes: &[u8], max_ids: usize, mut accept: F) -> usize
where
    F: FnMut(&[u8], &[u8]) -> bool,
{
    if bytes.len() < 8 || &bytes[..4] != KTM_HEADER {
        return 0;
    }
    let count = u32::from_le_bytes(bytes[4..8].try_into().unwrap());
    let max_ids = if max_ids == 0 {
        DEFAULT_MAX_IDS
    } else {
        max_ids
    };
    if u64::from(count) > u64::try_from(max_ids).unwrap_or(u64::MAX).saturating_mul(4) {
        return 0;
    }

    let mut loaded = 0;
    let mut pos = 8usize;
    for _ in 0..count {
        let Some(lens_end) = pos.checked_add(8).filter(|&end| end <= bytes.len()) else {
            return loaded;
        };
        let id_len = u32::from_le_bytes(bytes[pos..pos + 4].try_into().unwrap()) as usize;
        let dsml_len = u32::from_le_bytes(bytes[pos + 4..lens_end].try_into().unwrap()) as usize;
        if id_len == 0 || id_len > MAX_ID_BYTES || dsml_len == 0 || dsml_len > MAX_DSML_BYTES {
            return loaded;
        }
        pos = lens_end;
        let Some(id_end) = pos.checked_add(id_len).filter(|&end| end <= bytes.len()) else {
            return loaded;
        };
        let Some(dsml_end) = id_end
            .checked_add(dsml_len)
            .filter(|&end| end <= bytes.len())
        else {
            return loaded;
        };
        let id = &bytes[pos..id_end];
        let id = &id[..id.iter().position(|&byte| byte == 0).unwrap_or(id.len())];
        let dsml = &bytes[id_end..dsml_end];
        let dsml = &dsml[..dsml
            .iter()
            .position(|&byte| byte == 0)
            .unwrap_or(dsml.len())];
        if accept(id, dsml) {
            loaded += 1;
        }
        pos = dsml_end;
    }
    loaded
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Source {
    Disk,
    Ram,
}

struct Entry {
    block: Arc<str>,
    source: Source,
    stamp: u64,
}

pub(crate) struct ToolMemory {
    by_id: HashMap<String, Entry>,
    blocks: HashMap<Arc<str>, VecDeque<String>>,
    clock: u64,
    bytes: usize,
    max_ids: usize,
    max_bytes: usize,
}

impl Default for ToolMemory {
    fn default() -> Self {
        Self::new(DEFAULT_MAX_IDS, MAX_DSML_BYTES)
    }
}

impl Drop for ToolMemory {
    fn drop(&mut self) {
        for id in self.by_id.keys() {
            release_tool_id(id);
        }
    }
}

impl ToolMemory {
    pub(crate) fn new(max_ids: usize, max_bytes: usize) -> Self {
        Self {
            by_id: HashMap::new(),
            blocks: HashMap::new(),
            clock: 0,
            bytes: 0,
            max_ids: if max_ids == 0 {
                DEFAULT_MAX_IDS
            } else {
                max_ids
            },
            max_bytes: if max_bytes == 0 {
                MAX_DSML_BYTES
            } else {
                max_bytes
            },
        }
    }

    #[cfg(test)]
    pub(crate) fn contains_id(&self, id: &str) -> bool {
        self.by_id.contains_key(id)
    }

    pub(crate) fn put(&mut self, id: &str, dsml: &str, source: Source) {
        if id.is_empty()
            || id.len() > MAX_ID_BYTES
            || dsml.is_empty()
            || dsml.len() > MAX_DSML_BYTES
            || id.as_bytes().contains(&0)
            || dsml.as_bytes().contains(&0)
        {
            return;
        }

        self.clock = self.clock.wrapping_add(1);
        if let Some(entry) = self.by_id.get_mut(id) {
            if entry.block.as_ref() == dsml {
                entry.stamp = self.clock;
                if source == Source::Ram {
                    entry.source = Source::Ram;
                }
                self.prune();
                return;
            }
        }
        self.remove(id);

        let block = self
            .blocks
            .get_key_value(dsml)
            .map(|(block, _)| Arc::clone(block))
            .unwrap_or_else(|| {
                let block: Arc<str> = Arc::from(dsml);
                self.bytes = self.bytes.saturating_add(block.len() + 1);
                self.blocks.insert(Arc::clone(&block), VecDeque::new());
                block
            });
        self.blocks
            .get_mut(block.as_ref())
            .unwrap()
            .push_front(id.to_string());
        self.bytes = self.bytes.saturating_add(id.len() + 1);
        self.by_id.insert(
            id.to_string(),
            Entry {
                block,
                source,
                stamp: self.clock,
            },
        );
        reserve_tool_id(id);
        self.prune();
    }

    pub(crate) fn remember(&mut self, calls: &[ToolCall], raw_dsml: &str) -> usize {
        if raw_dsml.is_empty() {
            return 0;
        }
        let mut remembered = 0;
        for call in calls {
            self.put(&call.id, raw_dsml, Source::Ram);
            if self
                .by_id
                .get(&call.id)
                .map(|entry| entry.block.as_ref() == raw_dsml)
                .unwrap_or(false)
            {
                remembered += 1;
            }
        }
        remembered
    }

    pub(crate) fn attach(&mut self, messages: &mut [ChatMsg]) -> usize {
        let mut attached = 0;
        for message in messages {
            if message.calls.is_empty()
                || !message.raw_dsml.is_empty()
                || !message.raw_tool_text.is_empty()
            {
                continue;
            }
            let mut matched: Option<Arc<str>> = None;
            let mut exact = true;
            for call in &message.calls {
                let Some(block) = self.lookup(&call.id) else {
                    exact = false;
                    continue;
                };
                if let Some(current) = &matched {
                    if !Arc::ptr_eq(current, &block) {
                        exact = false;
                    }
                } else {
                    matched = Some(block);
                }
            }
            if exact {
                if let Some(block) = matched {
                    message.raw_dsml = block.to_string();
                    message.raw_tool_text = block.to_string();
                    attached += 1;
                }
            }
        }
        attached
    }

    pub(crate) fn checkpoint(&self, text: &[u8]) -> Option<Vec<u8>> {
        let mut entries = Vec::new();
        let mut seen: HashSet<Arc<str>> = HashSet::new();
        let mut pos = 0;
        while let Some((start, end)) = next_tool_block(text, pos) {
            if let Ok(raw) = std::str::from_utf8(&text[start..end]) {
                if let Some((block, ids)) = self.blocks.get_key_value(raw) {
                    if seen.insert(Arc::clone(block)) {
                        for id in ids {
                            entries.push((id.as_bytes(), block.as_bytes()));
                        }
                    }
                }
            }
            pos = end;
        }
        // A continuous bank deliberately omits its last sampled token, which
        // can cut the current tool block inside the closing marker. Complete
        // blocks use the hash lookup above; only this bounded LRU tail needs a
        // scan, and equal-prefix ambiguity fails closed.
        let mut tail = None;
        let mut tail_len = 0;
        let mut ambiguous = false;
        for (block, ids) in &self.blocks {
            if seen.contains(block) {
                continue;
            }
            let matched = tool_block_tail_prefix_len(text, block.as_bytes());
            if matched > tail_len {
                tail = Some((block, ids));
                tail_len = matched;
                ambiguous = false;
            } else if matched != 0 && matched == tail_len {
                ambiguous = true;
            }
        }
        if !ambiguous {
            if let Some((block, ids)) = tail {
                for id in ids {
                    entries.push((id.as_bytes(), block.as_bytes()));
                }
            }
        }
        encode_ktm_bounded(&entries, self.max_bytes)
    }

    pub(crate) fn load_trailer(&mut self, trailer: &[u8], wanted: &HashSet<String>) -> usize {
        let max_ids = self.max_ids;
        decode_ktm(trailer, max_ids, |id, dsml| {
            let Ok(id) = std::str::from_utf8(id) else {
                return false;
            };
            if !wanted.contains(id) {
                return false;
            }
            let Ok(dsml) = std::str::from_utf8(dsml) else {
                return false;
            };
            self.put(id, dsml, Source::Disk);
            true
        })
    }

    pub(crate) fn wanted_ids(messages: &[ChatMsg]) -> HashSet<String> {
        let mut wanted = HashSet::new();
        for message in messages {
            if !message.tool_call_id.is_empty() {
                wanted.insert(message.tool_call_id.clone());
            }
            wanted.extend(
                message
                    .tool_call_ids
                    .iter()
                    .filter(|id| !id.is_empty())
                    .cloned(),
            );
            wanted.extend(
                message
                    .calls
                    .iter()
                    .map(|call| &call.id)
                    .filter(|id| !id.is_empty())
                    .cloned(),
            );
        }
        wanted
    }

    pub(crate) fn restore_store(
        &mut self,
        store: &KvStore,
        model_id: u8,
        messages: &[ChatMsg],
    ) -> usize {
        let mut missing = Self::wanted_ids(messages);
        missing.retain(|id| !self.by_id.contains_key(id));
        if missing.is_empty() {
            return 0;
        }
        let paths: Vec<_> = store
            .entries()
            .iter()
            .filter(|entry| {
                entry.header.model_id == model_id && entry.header.ext_flags & EXT_TOOL_MAP != 0
            })
            .map(|entry| entry.path.clone())
            .collect();
        let mut loaded = 0;
        for path in paths {
            let Ok((header, trailer)) = read_trailer(&path, MAX_DSML_BYTES as u64) else {
                continue;
            };
            if header.model_id == model_id && header.ext_flags & EXT_TOOL_MAP != 0 {
                loaded += self.load_trailer(&trailer, &missing);
                missing.retain(|id| !self.by_id.contains_key(id));
                if missing.is_empty() {
                    break;
                }
            }
        }
        loaded
    }

    fn lookup(&mut self, id: &str) -> Option<Arc<str>> {
        self.clock = self.clock.wrapping_add(1);
        let entry = self.by_id.get_mut(id)?;
        entry.stamp = self.clock;
        Some(Arc::clone(&entry.block))
    }

    fn prune(&mut self) {
        // ponytail: tool calls are rare; replace the O(n) oldest scan only if profiling says so.
        while self.by_id.len() > self.max_ids || self.bytes > self.max_bytes {
            let Some(oldest) = self
                .by_id
                .iter()
                .min_by_key(|(_, entry)| entry.stamp)
                .map(|(id, _)| id.clone())
            else {
                break;
            };
            self.remove(&oldest);
        }
    }

    fn remove(&mut self, id: &str) {
        let Some(entry) = self.by_id.remove(id) else {
            return;
        };
        release_tool_id(id);
        self.bytes = self.bytes.saturating_sub(id.len() + 1);
        let empty = if let Some(ids) = self.blocks.get_mut(entry.block.as_ref()) {
            ids.retain(|entry_id| entry_id != id);
            ids.is_empty()
        } else {
            false
        };
        if empty {
            self.blocks.remove(entry.block.as_ref());
            self.bytes = self.bytes.saturating_sub(entry.block.len() + 1);
        }
    }
}

fn find_bytes(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    haystack
        .windows(needle.len())
        .position(|window| window == needle)
}

fn tool_block_forms() -> [(&'static [u8], &'static [u8], Option<&'static [u8]>); 9] {
    [
        (
            b"\n\n<\xef\xbd\x9cDSML\xef\xbd\x9ctool_calls>",
            DSML_TOOL_CALLS_END.as_bytes(),
            None,
        ),
        (
            DSML_TOOL_CALLS_START.as_bytes(),
            DSML_TOOL_CALLS_END.as_bytes(),
            None,
        ),
        (
            b"\n\n<DSML\xef\xbd\x9ctool_calls>",
            DSML_TOOL_CALLS_END_SHORT.as_bytes(),
            None,
        ),
        (
            DSML_TOOL_CALLS_START_SHORT.as_bytes(),
            DSML_TOOL_CALLS_END_SHORT.as_bytes(),
            None,
        ),
        (b"\n\n<tool_calls>", b"</tool_calls>", None),
        (b"<tool_calls>", b"</tool_calls>", None),
        (
            SOLAR_TOOL_CALLS.as_bytes(),
            SOLAR_TOOL_CALL_END.as_bytes(),
            Some(SOLAR_TOOL_CALLS.as_bytes()),
        ),
        (
            b"\n<tool_call>",
            b"</tool_call>",
            Some(MOTIF_TOOL_CALLS.as_bytes()),
        ),
        (
            MOTIF_TOOL_CALLS.as_bytes(),
            b"</tool_call>",
            Some(MOTIF_TOOL_CALLS.as_bytes()),
        ),
    ]
}

fn tool_block_tail_prefix_len(text: &[u8], block: &[u8]) -> usize {
    if text.is_empty() || block.is_empty() {
        return 0;
    }
    let mut best = 0;
    for (start_marker, _, _) in tool_block_forms() {
        let mut block_from = 0;
        while block_from < block.len() {
            let Some(block_rel) = find_bytes(&block[block_from..], start_marker) else {
                break;
            };
            let block_marker = block_from + block_rel;
            let min_candidate = text.len().saturating_sub(block.len());
            let mut text_from = min_candidate.saturating_add(block_marker).min(text.len());
            while text_from < text.len() {
                let Some(text_rel) = find_bytes(&text[text_from..], start_marker) else {
                    break;
                };
                let text_marker = text_from + text_rel;
                if text_marker >= block_marker {
                    let candidate = text_marker - block_marker;
                    let suffix = &text[candidate..];
                    if suffix.len() <= block.len()
                        && suffix.len() >= block_marker + start_marker.len()
                        && block.starts_with(suffix)
                    {
                        best = best.max(suffix.len());
                    }
                }
                text_from = text_marker + 1;
            }
            block_from = block_marker + 1;
        }
    }
    best
}

fn next_tool_block(text: &[u8], from: usize) -> Option<(usize, usize)> {
    let mut best: Option<(usize, usize, Option<&[u8]>, &[u8])> = None;
    for (start_marker, end_marker, repeat_start) in tool_block_forms() {
        let Some(start_rel) = find_bytes(&text[from..], start_marker) else {
            continue;
        };
        let start = from + start_rel;
        let body = start + start_marker.len();
        let Some(end_rel) = find_bytes(&text[body..], end_marker) else {
            continue;
        };
        let end = body + end_rel + end_marker.len();
        if best.map(|current| start < current.0).unwrap_or(true) {
            best = Some((start, end, repeat_start, end_marker));
        }
    }
    let (start, mut end, repeat_start, repeat_end) = best?;
    if let Some(repeat_start) = repeat_start {
        loop {
            let next = text[end..]
                .iter()
                .position(|byte| !byte.is_ascii_whitespace())
                .map(|offset| end + offset)
                .unwrap_or(text.len());
            if !text[next..].starts_with(repeat_start) {
                break;
            }
            let body = next + repeat_start.len();
            let Some(end_rel) = find_bytes(&text[body..], repeat_end) else {
                break;
            };
            end = body + end_rel + repeat_end.len();
        }
    }
    Some((start, end))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::{HashMap, HashSet};
    use std::path::PathBuf;
    use std::process::Command;

    fn hex(bytes: &[u8]) -> String {
        bytes.iter().map(|byte| format!("{byte:02x}")).collect()
    }

    fn oracle() -> PathBuf {
        std::env::var("DS4_KV_C_ORACLE")
            .map(PathBuf::from)
            .unwrap_or_else(|_| {
                PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/parity/kv_c_oracle")
            })
    }

    #[test]
    fn ktm_wire_uses_binary_version_and_little_endian_lengths() {
        let got = encode_ktm(&[(b"call_a", b"<tool>")]).unwrap();
        let mut expected = b"KTM\x01\x01\0\0\0\x06\0\0\0\x06\0\0\0".to_vec();
        expected.extend_from_slice(b"call_a<tool>");
        assert_eq!(got, expected);
        assert_eq!(encode_ktm(&[]).unwrap(), Vec::<u8>::new());
    }

    #[test]
    fn ktm_codec_matches_c_oracle() {
        let entries = [
            (b"call_a".as_slice(), b"<tool>one</tool>".as_slice()),
            (b"call_b".as_slice(), b"<tool>two</tool>".as_slice()),
        ];
        let rust = encode_ktm(&entries).unwrap();
        let output = Command::new(oracle())
            .args([
                "ktm-encode",
                &hex(entries[0].0),
                &hex(entries[0].1),
                &hex(entries[1].0),
                &hex(entries[1].1),
            ])
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        assert_eq!(String::from_utf8(output.stdout).unwrap().trim(), hex(&rust));

        let output = Command::new(oracle())
            .args(["ktm-decode", &hex(&rust), "100000"])
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        assert_eq!(
            String::from_utf8(output.stdout).unwrap(),
            format!(
                "{}:{}\n{}:{}\nloaded=2\n",
                hex(entries[0].0),
                hex(entries[0].1),
                hex(entries[1].0),
                hex(entries[1].1)
            )
        );
    }

    #[test]
    fn ktm_decode_keeps_valid_prefix_and_filters_wanted() {
        let mut bytes = encode_ktm(&[(b"keep", b"one"), (b"want", b"two")]).unwrap();
        bytes.pop();
        let mut got = Vec::new();
        assert_eq!(
            decode_ktm(&bytes, 100_000, |id, dsml| {
                got.push((id.to_vec(), dsml.to_vec()));
                true
            }),
            1
        );
        assert_eq!(got, vec![(b"keep".to_vec(), b"one".to_vec())]);

        let bytes = encode_ktm(&[(b"skip", b"one"), (b"want", b"two")]).unwrap();
        let wanted = HashSet::from([b"want".as_slice()]);
        let mut got = Vec::new();
        assert_eq!(
            decode_ktm(&bytes, 100_000, |id, dsml| {
                if !wanted.contains(id) {
                    return false;
                }
                got.push((id.to_vec(), dsml.to_vec()));
                true
            }),
            1
        );
        assert_eq!(got, vec![(b"want".to_vec(), b"two".to_vec())]);
    }

    #[test]
    fn ktm_decode_rejects_bad_header_count_and_lengths() {
        assert_eq!(decode_ktm(b"KTM1\0\0\0\0", 100_000, |_, _| true), 0);

        let mut over_count = b"KTM\x01".to_vec();
        over_count.extend_from_slice(&400_001u32.to_le_bytes());
        assert_eq!(decode_ktm(&over_count, 0, |_, _| true), 0);

        let mut zero_id = b"KTM\x01\x01\0\0\0".to_vec();
        zero_id.extend_from_slice(&0u32.to_le_bytes());
        zero_id.extend_from_slice(&1u32.to_le_bytes());
        zero_id.push(b'x');
        assert_eq!(decode_ktm(&zero_id, 100_000, |_, _| true), 0);

        let mut over_id = b"KTM\x01\x01\0\0\0".to_vec();
        over_id.extend_from_slice(&257u32.to_le_bytes());
        over_id.extend_from_slice(&1u32.to_le_bytes());
        assert_eq!(decode_ktm(&over_id, 100_000, |_, _| true), 0);

        let mut over_dsml = b"KTM\x01\x01\0\0\0".to_vec();
        over_dsml.extend_from_slice(&1u32.to_le_bytes());
        over_dsml.extend_from_slice(&(512u32 * 1024 * 1024 + 1).to_le_bytes());
        assert_eq!(decode_ktm(&over_dsml, 100_000, |_, _| true), 0);
    }

    #[test]
    fn ktm_decode_preserves_order_for_duplicate_last_wins() {
        let bytes = encode_ktm(&[(b"same", b"old"), (b"same", b"new")]).unwrap();
        let mut by_id = HashMap::new();
        assert_eq!(
            decode_ktm(&bytes, 100_000, |id, dsml| {
                by_id.insert(id.to_vec(), dsml.to_vec());
                true
            }),
            2
        );
        assert_eq!(
            by_id.get(b"same".as_slice()).map(Vec::as_slice),
            Some(b"new".as_slice())
        );
    }

    #[test]
    fn ktm_decode_matches_c_string_nul_truncation() {
        let bytes = b"KTM\x01\x01\0\0\0\x03\0\0\0\x03\0\0\0a\0bx\0y";
        let mut got = None;
        assert_eq!(
            decode_ktm(bytes, 100_000, |id, dsml| {
                got = Some((id.to_vec(), dsml.to_vec()));
                true
            }),
            1
        );
        assert_eq!(got, Some((b"a".to_vec(), b"x".to_vec())));

        let output = Command::new(oracle())
            .args(["ktm-decode", &hex(bytes), "100000"])
            .output()
            .unwrap();
        assert!(output.status.success());
        assert_eq!(
            String::from_utf8(output.stdout).unwrap(),
            "61:78\nloaded=1\n"
        );
    }

    fn dsml(name: &str) -> String {
        format!(
            "\n\n{DSML_TOOL_CALLS_START}\n<｜DSML｜invoke name=\"{name}\">\n</｜DSML｜invoke>\n{DSML_TOOL_CALLS_END}"
        )
    }

    fn calls(ids: &[&str]) -> Vec<ToolCall> {
        ids.iter()
            .map(|id| ToolCall {
                id: (*id).into(),
                ..Default::default()
            })
            .collect()
    }

    #[test]
    fn memory_shares_blocks_preserves_id_order_and_upgrades_source() {
        let raw = dsml("bash");
        let mut memory = ToolMemory::new(10, 4096);
        memory.put("call_a", &raw, Source::Disk);
        memory.put("call_b", &raw, Source::Disk);
        let shared_bytes = memory.bytes;
        assert_eq!(memory.remember(&calls(&["call_a"]), &raw), 1);
        assert_eq!(memory.bytes, shared_bytes);
        assert_eq!(memory.blocks.len(), 1);
        assert_eq!(memory.by_id["call_a"].source, Source::Ram);

        let trailer = memory.checkpoint(raw.as_bytes()).unwrap();
        let mut ids = Vec::new();
        assert_eq!(
            decode_ktm(&trailer, 10, |id, _| {
                ids.push(String::from_utf8(id.to_vec()).unwrap());
                true
            }),
            2
        );
        assert_eq!(ids, ["call_b", "call_a"]);
    }

    #[test]
    fn attach_is_all_or_nothing_for_each_assistant_block() {
        let first = dsml("first");
        let second = dsml("second");
        let mut memory = ToolMemory::new(10, 4096);
        memory.put("call_a", &first, Source::Ram);
        memory.put("call_b", &first, Source::Ram);
        memory.put("call_other", &second, Source::Ram);

        let mut messages = vec![
            ChatMsg {
                calls: calls(&["call_a", "call_b"]),
                ..Default::default()
            },
            ChatMsg {
                calls: calls(&["call_a", "missing"]),
                ..Default::default()
            },
            ChatMsg {
                calls: calls(&["call_a", "call_other"]),
                ..Default::default()
            },
        ];
        assert_eq!(memory.attach(&mut messages), 1);
        assert_eq!(messages[0].raw_dsml, first);
        assert!(messages[1].raw_dsml.is_empty());
        assert!(messages[2].raw_dsml.is_empty());
    }

    #[test]
    fn checkpoint_filters_complete_blocks_and_deduplicates_solar_span() {
        let exact = dsml("exact");
        let absent = dsml("absent");
        let solar = format!(
            "{SOLAR_TOOL_CALLS}one{SOLAR_TOOL_CALL_END} \n {SOLAR_TOOL_CALLS}two{SOLAR_TOOL_CALL_END}"
        );
        let mut memory = ToolMemory::new(10, 8192);
        memory.put("call_exact", &exact, Source::Ram);
        memory.put("call_absent", &absent, Source::Ram);
        memory.put("call_solar", &solar, Source::Ram);

        let text = format!("prefix{exact} middle {exact} tail {solar}");
        let trailer = memory.checkpoint(text.as_bytes()).unwrap();
        let mut got = Vec::new();
        assert_eq!(
            decode_ktm(&trailer, 10, |id, dsml| {
                got.push((id.to_vec(), dsml.to_vec()));
                true
            }),
            2
        );
        assert_eq!(got[0], (b"call_exact".to_vec(), exact.into_bytes()));
        assert_eq!(got[1], (b"call_solar".to_vec(), solar.into_bytes()));
    }

    #[test]
    fn checkpoint_keeps_a_tool_turn_cut_inside_its_close_marker() {
        let raw = dsml("bank-cut");
        let mut memory = ToolMemory::new(10, 4096);
        memory.put("call_cut", &raw, Source::Ram);
        let cut = raw.len() - 5;
        let text = format!("prompt{}", &raw[..cut]);

        let trailer = memory.checkpoint(text.as_bytes()).unwrap();
        let mut got = Vec::new();
        assert_eq!(
            decode_ktm(&trailer, 10, |id, dsml| {
                got.push((id.to_vec(), dsml.to_vec()));
                true
            }),
            1
        );
        assert_eq!(got, vec![(b"call_cut".to_vec(), raw.into_bytes())]);
    }

    #[test]
    fn checkpoint_restores_motif_exact_tool_text_from_a_partial_bank_key() {
        let raw =
            "\n<tool_call>{\"name\":\"pair_values\",\"arguments\":{\"a\":1,\"b\":2}}</tool_call>";
        let mut source = ToolMemory::new(10, 4096);
        source.put("call_motif", raw, Source::Ram);
        let mut text = b"prompt".to_vec();
        text.extend_from_slice(&raw.as_bytes()[..raw.len() - 5]);
        let trailer = source.checkpoint(&text).unwrap();

        let mut restored = ToolMemory::new(10, 4096);
        let wanted = HashSet::from(["call_motif".to_string()]);
        assert_eq!(restored.load_trailer(&trailer, &wanted), 1);
        let mut messages = vec![ChatMsg {
            role: "assistant".into(),
            calls: calls(&["call_motif"]),
            ..Default::default()
        }];
        assert_eq!(restored.attach(&mut messages), 1);
        assert_eq!(messages[0].raw_tool_text, raw);
    }

    #[test]
    fn memory_prunes_lru_and_counts_shared_dsml_once() {
        let raw = dsml("shared");
        let budget = raw.len() + 1 + ("call_a".len() + 1) * 2;
        let mut memory = ToolMemory::new(2, budget);
        memory.put("call_a", &raw, Source::Ram);
        memory.put("call_b", &raw, Source::Ram);
        assert_eq!(memory.blocks.len(), 1);
        assert_eq!(memory.bytes, budget);
        assert!(memory.lookup("call_a").is_some());
        memory.put("call_c", &raw, Source::Ram);
        assert!(memory.contains_id("call_a"));
        assert!(!memory.contains_id("call_b"));
        assert!(memory.contains_id("call_c"));
        assert_eq!(memory.bytes, budget);
    }

    #[test]
    fn checkpoint_rejects_shared_block_expansion_over_memory_budget() {
        let raw = dsml("shared-expansion");
        let budget = raw.len() + 1 + ("call_a".len() + 1) * 3;
        let mut memory = ToolMemory::new(3, budget);
        memory.put("call_a", &raw, Source::Ram);
        memory.put("call_b", &raw, Source::Ram);
        memory.put("call_c", &raw, Source::Ram);

        assert_eq!(memory.by_id.len(), 3);
        assert!(memory.checkpoint(raw.as_bytes()).is_none());
    }

    #[test]
    fn trailer_load_filters_wanted_ids() {
        let trailer = encode_ktm(&[(b"skip", b"one"), (b"want", b"two")]).unwrap();
        let mut memory = ToolMemory::new(10, 4096);
        assert_eq!(
            memory.load_trailer(&trailer, &HashSet::from(["want".to_string()])),
            1
        );
        assert!(!memory.contains_id("skip"));
        assert!(memory.contains_id("want"));
    }

    #[test]
    fn store_restore_filters_model_and_requested_ids_before_attach() {
        use ds4_kv::{Header, Options, Reason, Record, Store, EXT_TOOL_MAP};
        use std::fs;

        let dir =
            std::env::temp_dir().join(format!("ds4-tool-memory-restore-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        let mut store = Store::open(&dir, 16, true, Options::default()).unwrap();
        let raw = dsml("restore");
        let trailer = encode_ktm(&[
            (b"call_want", raw.as_bytes()),
            (b"call_skip", b"not requested"),
        ])
        .unwrap();
        store
            .write(Record {
                header: Header {
                    quant_bits: 2,
                    reason: Reason::Evict,
                    ext_flags: EXT_TOOL_MAP,
                    model_id: 2,
                    tokens: 1,
                    hits: 0,
                    ctx_size: 4096,
                    created_at: 1,
                    last_used: 1,
                    payload_bytes: 0,
                    text_bytes: 0,
                },
                text: raw.as_bytes().to_vec(),
                payload: b"opaque".to_vec(),
                trailer,
            })
            .unwrap();
        let mut messages = vec![ChatMsg {
            calls: calls(&["call_want"]),
            ..Default::default()
        }];
        assert_eq!(ToolMemory::wanted_ids(&messages).len(), 1);
        let mut memory = ToolMemory::new(10, 4096);
        assert_eq!(memory.restore_store(&store, 1, &messages), 0);
        assert_eq!(memory.restore_store(&store, 2, &messages), 1);
        assert!(!memory.contains_id("call_skip"));
        assert_eq!(memory.attach(&mut messages), 1);
        assert_eq!(messages[0].raw_dsml, raw);

        let fresh = dsml("fresh");
        memory.put("call_want", &fresh, Source::Ram);
        assert_eq!(memory.restore_store(&store, 2, &messages), 0);
        let mut replay = vec![ChatMsg {
            calls: calls(&["call_want"]),
            ..Default::default()
        }];
        assert_eq!(memory.attach(&mut replay), 1);
        assert_eq!(replay[0].raw_dsml, fresh);
        let _ = fs::remove_dir_all(dir);
    }
}
