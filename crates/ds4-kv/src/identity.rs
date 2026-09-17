//! Fixed identity footer after opaque payload and optional KTM records.
//! Reading identity never scans payload bytes or allocates from file lengths.

use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;

pub(crate) const FLAG: u8 = 1 << 6;
pub(crate) const BYTES: usize = 40;
const MAGIC: &[u8; 4] = b"KVI\x01";
const END: &[u8; 4] = b"KVI!";

pub(crate) fn append(trailer: &mut Vec<u8>, identity: &[u8; 32]) {
    trailer.extend_from_slice(MAGIC);
    trailer.extend_from_slice(identity);
    trailer.extend_from_slice(END);
}

pub(crate) fn read(path: &Path, file_size: u64, trailer_bytes: u64, flags: u8) -> Option<[u8; 32]> {
    if flags & FLAG == 0 || trailer_bytes < BYTES as u64 || file_size < BYTES as u64 {
        return None;
    }
    let mut file = File::open(path).ok()?;
    if file.metadata().ok()?.len() != file_size {
        return None;
    }
    file.seek(SeekFrom::Start(file_size - BYTES as u64)).ok()?;
    let mut footer = [0; BYTES];
    file.read_exact(&mut footer).ok()?;
    decode(flags, &footer)
}

pub(crate) fn decode(flags: u8, trailer: &[u8]) -> Option<[u8; 32]> {
    if flags & FLAG == 0 || trailer.len() < BYTES {
        return None;
    }
    let footer = &trailer[trailer.len() - BYTES..];
    if &footer[..4] != MAGIC || &footer[36..] != END {
        return None;
    }
    footer[4..36].try_into().ok()
}
