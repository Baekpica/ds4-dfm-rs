//! Blocking frame I/O. Same layout as `dist_write_frame_header` / `dist_read_frame_header`.

use std::io::{self, Read, Write};

use crate::codec::{decode_frame_header, encode_frame_header, CodecError, FRAME_HEADER_BYTES};

pub fn write_frame<W: Write>(w: &mut W, typ: u32, payload: &[u8]) -> io::Result<()> {
    let bytes = u32::try_from(payload.len())
        .map_err(|_| io::Error::new(io::ErrorKind::InvalidInput, "distributed frame too large"))?;
    w.write_all(&encode_frame_header(typ, bytes))?;
    w.write_all(payload)?;
    Ok(())
}

pub fn read_frame<R: Read>(r: &mut R) -> io::Result<(u32, Vec<u8>)> {
    FrameReader::default().read(r)
}

/// One unbuffered reader per connection; timeouts retain the consumed prefix.
#[derive(Default)]
pub(crate) struct FrameReader {
    header: [u8; FRAME_HEADER_BYTES],
    header_read: usize,
    typ: Option<u32>,
    payload: Vec<u8>,
    payload_read: usize,
}

impl FrameReader {
    pub(crate) fn read(&mut self, r: &mut impl Read) -> io::Result<(u32, Vec<u8>)> {
        if self.typ.is_none() {
            read_part(r, &mut self.header, &mut self.header_read)?;
            let h = decode_frame_header(&self.header).map_err(|e| match e {
                CodecError::BadMagic(m) => io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("bad frame magic 0x{m:08x}"),
                ),
                other => io::Error::new(io::ErrorKind::InvalidData, other.to_string()),
            })?;
            self.payload
                .try_reserve_exact(h.bytes as usize)
                .map_err(|_| {
                    io::Error::new(io::ErrorKind::OutOfMemory, "distributed frame too large")
                })?;
            self.payload.resize(h.bytes as usize, 0);
            self.typ = Some(h.typ);
        }
        read_part(r, &mut self.payload, &mut self.payload_read)?;
        self.header_read = 0;
        self.payload_read = 0;
        Ok((
            self.typ.take().expect("frame header"),
            std::mem::take(&mut self.payload),
        ))
    }
}

fn read_part(r: &mut impl Read, bytes: &mut [u8], read: &mut usize) -> io::Result<()> {
    while *read < bytes.len() {
        match r.read(&mut bytes[*read..]) {
            Ok(0) => return Err(io::ErrorKind::UnexpectedEof.into()),
            Ok(n) => *read += n,
            Err(e) if e.kind() == io::ErrorKind::Interrupted => continue,
            Err(e) => return Err(e),
        }
    }
    Ok(())
}
