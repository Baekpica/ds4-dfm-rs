//! KVC disk store. File format, SHA key, eviction, and prefix lookup
//! match `ds4_kvstore.c` at `v0.6.3-dfm`. Payload bytes stay opaque.

mod extensions;
mod format;
mod host;
mod identity;
mod policy;
mod sha1;
mod store;

pub use extensions::{
    bank_persist_ext_flags, decode_ktm, encode_ktm, BankThinkingExtensions, ExtensionRecord,
    KTM_MAGIC, KTM_VERSION,
};
pub use format::{
    decode_file, encode_file, fill_header, key_kind, parse_header, path_for_sha, read_envelope,
    read_path, read_trailer, sha_hex_name, text_sha_hex, write_path, Envelope, FormatError, Header,
    Reason, Record, EXT_BANK_REPLAY_V1, EXT_IMAGE_PIXELS_V2, EXT_RESPONSES_VISIBLE,
    EXT_SESSION_TITLE, EXT_THINKING_VISIBLE, EXT_TOOL_MAP, FIXED_HEADER, PAYLOAD_ABI, VERSION,
};
pub use host::{bank_checkpoint_due_from_host, continued_store_target_from_host, HostKvView};
pub use policy::{
    bank_checkpoint_due, chat_anchor_pos, continued_store_target, eviction_score, file_size_fits,
    store_len, EvictionContext, Options, ScoreEntry,
};
pub use sha1::sha1_hex;
pub use store::{Entry, PayloadGuard, PayloadTemp, PrefixAnswer, Store};
