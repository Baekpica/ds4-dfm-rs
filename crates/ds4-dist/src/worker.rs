//! Worker control loop: HELLO, WORK parse/validate, slice eval, RESULT.

use std::collections::HashMap;
use std::io::{Read, Write};
use std::net::TcpStream;
use std::sync::{Arc, Mutex};

use crate::activation::{bits_or_default, wire_bytes};
use crate::codec::{
    decode_hello_payload, encode_hello_payload, u64_from_halves, CodecError, Hello, Work,
    MSG_ERROR, MSG_HELLO, MSG_SNAPSHOT_LOAD_BEGIN, MSG_SNAPSHOT_SAVE_REQ, MSG_WORK, RESULT_ACK,
    RESULT_HIDDEN_STATE, RESULT_LOGITS, ROUTE_F_OUTPUT_LOGITS, ROUTE_RETURN_UPSTREAM,
    WORK_FIXED_BYTES, WORK_F_ACK_ONLY, WORK_F_INPUT_HC, WORK_F_OUTPUT_LOGITS, WORK_F_RESET_SESSION,
    WORK_F_VALID_MASK,
};
use crate::exec::{SliceExec, WorkRequest};
use crate::forward::ERR_FORWARD_HIDDEN;
use crate::hash::{token_hash_update_span, TOKEN_HASH_INIT};
use crate::hops::ForwarderPool;
use crate::native_snapshot::{dispatch_worker_snapshot, MemorySnapshotStore, SnapshotStore};
use crate::relay::{forward_work_blocking, local_work_telemetry, now_sec, usec_since};
use crate::route::{decode_route_blob, validate_route_blob};
use crate::transport::{read_frame, write_frame};
use crate::work::{
    decode_work_body, encode_logits_payload, encode_result_frame, error_result_frame,
    ok_result_hdr, ResultBody, WorkBody,
};
use crate::worker_snapshot::WorkerSnapshotIdentity;

pub fn send_hello<W: Write>(w: &mut W, hello: &Hello, model_name: &str) -> std::io::Result<()> {
    let payload = encode_hello_payload(hello, model_name)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidInput, e.to_string()))?;
    write_frame(w, MSG_HELLO, &payload)
}

pub fn recv_hello<R: Read>(r: &mut R) -> std::io::Result<(Hello, String)> {
    let (typ, body) = read_frame(r)?;
    if typ != MSG_HELLO {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("expected HELLO frame, got type {typ}"),
        ));
    }
    decode_hello_payload(&body)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e.to_string()))
}

pub struct Worker<E: SliceExec, S: SnapshotStore = MemorySnapshotStore> {
    pub exec: E,
    sessions: HashMap<u64, u64>,
    store: S,
    hops: ForwarderPool,
}

impl<E: SliceExec> Worker<E> {
    pub fn new(exec: E) -> Self {
        Self::with_store(exec, MemorySnapshotStore::new())
    }
}

impl<E: SliceExec, S: SnapshotStore> Worker<E, S> {
    pub fn with_store(exec: E, store: S) -> Self {
        Self {
            exec,
            sessions: HashMap::new(),
            store,
            hops: ForwarderPool::new(),
        }
    }

    pub fn store(&self) -> &S {
        &self.store
    }

    pub fn store_mut(&mut self) -> &mut S {
        &mut self.store
    }

    pub fn session_count(&self) -> usize {
        self.sessions.len()
    }

    pub fn clear_sessions(&mut self) -> u32 {
        let n = u32::try_from(self.sessions.len()).unwrap_or(u32::MAX);
        self.sessions.clear();
        n
    }

    pub(crate) fn apply_snapshot<Stream: Read + Write>(
        &mut self,
        stream: &mut Stream,
        typ: u32,
        body: &[u8],
    ) -> std::io::Result<()> {
        match dispatch_worker_snapshot(
            stream,
            typ,
            body,
            self.snapshot_identity(),
            self.exec.vocab(),
            &mut self.store,
        ) {
            Ok(Some((session_id, token_hash))) => {
                self.sessions.insert(session_id, token_hash);
            }
            Ok(None) => {}
            Err(_) => {}
        }
        Ok(())
    }

    fn snapshot_identity(&self) -> WorkerSnapshotIdentity {
        WorkerSnapshotIdentity {
            model_id: self.exec.model_id(),
            layer_start: self.exec.layer_start(),
            layer_end: self.exec.layer_end(),
            ctx_size: self.exec.ctx_size(),
        }
    }

    pub fn hello(&self, listen_port: u32, ctx_size: u32, model_name: &str) -> (Hello, String) {
        (
            Hello {
                model_id: self.exec.model_id(),
                quant_bits: 0,
                layer_start: self.exec.layer_start(),
                layer_end: self.exec.layer_end(),
                has_output: if self.exec.has_output() { 1 } else { 0 },
                has_hidden: 1,
                ctx_size,
                n_layers: self.exec.n_layers(),
                listen_port,
                model_name_len: 0,
            },
            model_name.to_string(),
        )
    }

    pub(crate) fn bind_hops(&mut self, upstream: Arc<Mutex<TcpStream>>) {
        self.hops.bind(upstream);
    }

    pub(crate) fn shutdown_hops(&mut self) {
        self.hops.shutdown();
    }

    pub fn serve(&mut self, stream: &mut TcpStream) -> std::io::Result<()> {
        if let Ok(clone) = stream.try_clone() {
            self.bind_hops(Arc::new(Mutex::new(clone)));
        }
        let rc = self.drive(stream);
        self.shutdown_hops();
        rc
    }

    pub fn serve_once(&mut self, stream: &mut TcpStream) -> std::io::Result<()> {
        self.drive_one(stream)
    }

    fn drive<Stream: Read + Write>(&mut self, stream: &mut Stream) -> std::io::Result<()> {
        loop {
            match self.drive_one(stream) {
                Ok(()) => {}
                Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => return Ok(()),
                Err(e) => return Err(e),
            }
        }
    }

    fn drive_one<Stream: Read + Write>(&mut self, stream: &mut Stream) -> std::io::Result<()> {
        let (typ, body) = read_frame(stream)?;
        self.handle_frame(stream, typ, &body)
    }

    pub(crate) fn handle_frame<Stream: Read + Write>(
        &mut self,
        stream: &mut Stream,
        typ: u32,
        body: &[u8],
    ) -> std::io::Result<()> {
        if typ == MSG_ERROR {
            return Err(std::io::Error::new(
                std::io::ErrorKind::Other,
                format!("coordinator error: {}", String::from_utf8_lossy(body)),
            ));
        }
        match typ {
            MSG_WORK => {
                let reply = self.process_work(body, stream)?;
                if let Some(frame) = reply {
                    stream.write_all(&frame)?;
                }
                Ok(())
            }
            MSG_SNAPSHOT_SAVE_REQ | MSG_SNAPSHOT_LOAD_BEGIN => {
                self.apply_snapshot(stream, typ, body)
            }
            _ => {
                write_frame(stream, MSG_ERROR, b"unsupported distributed worker frame")?;
                Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!("rejected unsupported frame type {typ}"),
                ))
            }
        }
    }

    pub(crate) fn process_work<W: Write>(
        &mut self,
        payload: &[u8],
        stream: &mut W,
    ) -> std::io::Result<Option<Vec<u8>>> {
        if payload.len() < WORK_FIXED_BYTES {
            return Ok(Some(error_result_frame(
                0,
                "truncated distributed WORK frame",
            )));
        }
        let request_id = match Work::decode(payload) {
            Ok(w) => u64_from_halves(w.request_hi, w.request_lo),
            Err(_) => 0,
        };
        let body = match decode_work_body(payload) {
            Ok(b) => b,
            Err(e) => {
                let msg = match e {
                    CodecError::Invalid(s) => s,
                    _ => "truncated distributed WORK frame",
                };
                return Ok(Some(error_result_frame(request_id, msg)));
            }
        };
        match self.handle_decoded(body, stream) {
            Ok(frame) => Ok(frame),
            Err(msg) => Ok(Some(error_result_frame(request_id, &msg))),
        }
    }

    fn handle_decoded<W: Write>(
        &mut self,
        body: WorkBody,
        stream: &mut W,
    ) -> Result<Option<Vec<u8>>, String> {
        let work = body.work;
        let request_id = u64_from_halves(work.request_hi, work.request_lo);
        let session_id = u64_from_halves(work.session_hi, work.session_lo);
        let prefix_hash = u64_from_halves(work.prefix_hash_hi, work.prefix_hash_lo);
        let result_hash = u64_from_halves(work.result_hash_hi, work.result_hash_lo);

        let fail = |m: String| -> Result<Option<Vec<u8>>, String> {
            Ok(Some(error_result_frame(request_id, &m)))
        };

        if work.route_count == 0 {
            return fail("WORK frame is missing distributed route".into());
        }
        if work.route_index >= work.route_count {
            return fail("invalid distributed WORK route metadata".into());
        }
        if work.model_id != self.exec.model_id() {
            return fail(format!(
                "model id mismatch: work={} worker={}",
                work.model_id,
                self.exec.model_id()
            ));
        }
        if work.layer_start != self.exec.layer_start() || work.layer_end != self.exec.layer_end() {
            return fail(format!(
                "worker is assigned layers {}:{} but request asked for {}:{}",
                self.exec.layer_start(),
                self.exec.layer_end(),
                work.layer_start,
                work.layer_end
            ));
        }
        if (work.flags & !WORK_F_VALID_MASK) != 0 {
            return fail("invalid distributed WORK flags".into());
        }
        if body.tokens.is_empty() {
            return fail("WORK frame has no tokens".into());
        }
        let ctx = self.exec.ctx_size();
        if work.pos0 > ctx || work.n_tokens > ctx.saturating_sub(work.pos0) {
            return fail("WORK token span exceeds worker context".into());
        }
        let output_logits = (work.flags & WORK_F_OUTPUT_LOGITS) != 0;
        let input_hc_present = (work.flags & WORK_F_INPUT_HC) != 0;
        let ack_only = (work.flags & WORK_F_ACK_ONLY) != 0;
        if input_hc_present && work.layer_start == 0 {
            return fail("layer 0 WORK must not provide input hidden-state".into());
        }
        if !input_hc_present && work.layer_start != 0 {
            return fail("nonzero layer WORK requires input hidden-state".into());
        }
        if output_logits && !self.exec.has_output() {
            return fail("worker was not assigned the output head".into());
        }
        if output_logits && work.layer_end + 1 != self.exec.n_layers() {
            return fail("WORK logits require final transformer layer".into());
        }
        for &t in &body.tokens {
            if t < 0 || (t as u32) >= self.exec.vocab() {
                return fail("WORK token id is outside the model vocabulary".into());
            }
        }
        if token_hash_update_span(prefix_hash, &body.tokens) != result_hash {
            return fail("WORK token prefix hash metadata mismatch".into());
        }

        let hc_values = self.exec.hidden_values();
        let expected_hc = (work.n_tokens as u64).saturating_mul(hc_values) as usize;
        if input_hc_present && body.input_hc.len() != expected_hc {
            return fail("input hidden-state size does not match token span".into());
        }
        if !input_hc_present && !body.input_hc.is_empty() {
            return fail("WORK frame has hidden bytes without input flag".into());
        }

        if let Err(e) =
            validate_route_blob(&body.route_blob, work.route_count, self.exec.n_layers())
        {
            return fail(e.to_string());
        }
        let (entries, ret) =
            decode_route_blob(&body.route_blob, work.route_count).map_err(|e| e.to_string())?;
        let current = entries
            .get(work.route_index as usize)
            .ok_or_else(|| "invalid route entry index".to_string())?;
        if current.layer_start != work.layer_start || current.layer_end != work.layer_end {
            return fail("WORK layer range does not match route entry".into());
        }
        let route_out = (current.flags & ROUTE_F_OUTPUT_LOGITS) != 0;
        if route_out != output_logits {
            return fail("WORK logits flag does not match route entry".into());
        }
        let has_next = (work.route_index + 1) < work.route_count;
        if has_next && output_logits {
            return fail("non-final route entry requested logits".into());
        }
        if !has_next && ret.kind != ROUTE_RETURN_UPSTREAM {
            return fail("unsupported final result destination".into());
        }

        let reset = (work.flags & WORK_F_RESET_SESSION) != 0;
        if reset {
            self.sessions.insert(session_id, TOKEN_HASH_INIT);
        }
        let sess_hash = *self.sessions.entry(session_id).or_insert(TOKEN_HASH_INIT);
        if sess_hash != prefix_hash {
            return fail("worker KV prefix hash mismatch".into());
        }

        let final_ack_only = ack_only && !has_next;
        let local_output = output_logits && !has_next && !final_ack_only;
        let produce_hidden = !local_output && !final_ack_only;
        let eval_t0 = now_sec();
        let out = match self.exec.eval(&WorkRequest {
            session_id,
            request_id,
            tokens: body.tokens.clone(),
            pos0: work.pos0,
            layer_start: work.layer_start,
            layer_end: work.layer_end,
            reset,
            produce_hidden,
            produce_logits: local_output,
            input_hc: body.input_hc.clone(),
        }) {
            Ok(v) => v,
            Err(e) => return fail(e),
        };
        let eval_usec = usec_since(eval_t0, now_sec());
        self.sessions.insert(session_id, result_hash);

        if has_next {
            let next = entries[(work.route_index + 1) as usize].clone();
            let mut fwd = body.clone();
            fwd.work.layer_start = next.layer_start;
            fwd.work.layer_end = next.layer_end;
            fwd.work.route_index = work.route_index + 1;
            fwd.work.flags |= WORK_F_INPUT_HC;
            if (next.flags & ROUTE_F_OUTPUT_LOGITS) != 0 {
                fwd.work.flags |= WORK_F_OUTPUT_LOGITS;
            } else {
                fwd.work.flags &= !WORK_F_OUTPUT_LOGITS;
            }
            let hidden = out.hidden.unwrap_or_default();
            let output_bytes =
                match wire_bytes(bits_or_default(work.input_hc_bits), hidden.len() as u64) {
                    Some(n) => n,
                    None => return fail(ERR_FORWARD_HIDDEN.into()),
                };
            let telemetry = local_work_telemetry(&work, eval_usec, output_bytes);
            fwd.input_hc = hidden;
            let hop = if self.hops.is_bound() {
                self.hops.forward(&next, &fwd, telemetry)
            } else {
                forward_work_blocking(&next, &fwd, stream, telemetry)
            };
            if let Err(msg) = hop {
                return fail(msg);
            }
            return Ok(None);
        }

        let (kind, payload, bits) = if final_ack_only {
            (RESULT_ACK, Vec::new(), 0u32)
        } else if local_output {
            let logits = out
                .logits
                .ok_or_else(|| "slice exec returned no logits".to_string())?;
            (RESULT_LOGITS, encode_logits_payload(&logits), 32)
        } else {
            let hidden = out
                .hidden
                .ok_or_else(|| "slice exec returned no hidden-state".to_string())?;
            let bits = bits_or_default(work.input_hc_bits);
            let wire = crate::activation::encode_activation(&hidden, bits)
                .ok_or_else(|| "invalid output hidden-state size".to_string())?;
            (RESULT_HIDDEN_STATE, wire, bits)
        };
        let mut hdr = ok_result_hdr(request_id, result_hash, kind, bits);
        hdr.payload_bytes = payload.len() as u32;
        let telemetry = local_work_telemetry(&work, eval_usec, payload.len() as u32);
        let frame = encode_result_frame(&ResultBody {
            hdr,
            telemetry: vec![telemetry],
            payload,
        })
        .map_err(|e| e.to_string())?;
        Ok(Some(frame))
    }
}
