//! Session-backed assemble + reconnect for `--role worker`.

use std::cell::RefCell;
use std::io::{Read, Seek, SeekFrom, Write};
use std::rc::Rc;

use ds4_core::{LayerPayloadLoad, LayerSliceEval, Model, ModelFamily, Session, Shape};
use ds4_dist::{
    assemble_worker, copy_chunked, create_temp, open_data_listener, print_worker_listen_banner,
    slice_meta, token_hash_prefix, worker_listen_port, worker_plan, Options, SliceExec, SliceMeta,
    SnapshotLoad, SnapshotSave, SnapshotStore, TempShard, WorkOutput, WorkRequest,
    WorkerListenBanner, LOAD_PREFIX, SAVE_PREFIX,
};

fn hidden_values(shape: &Shape) -> u64 {
    match shape.family {
        ModelFamily::SolarOpen2 | ModelFamily::ExaoneMoe | ModelFamily::Qwen4Exp => {
            u64::from(shape.n_embd)
        }
        ModelFamily::DeepSeek4
        | ModelFamily::Motif3
        | ModelFamily::Dots3Note
        | ModelFamily::Glm53 => u64::from(shape.n_hc) * u64::from(shape.n_embd),
    }
}

struct SessionSliceExec<'m> {
    session: Rc<RefCell<Session<'m>>>,
    meta: SliceMeta,
}

impl SliceExec for SessionSliceExec<'_> {
    fn model_id(&self) -> u32 {
        self.meta.model_id
    }
    fn n_layers(&self) -> u32 {
        self.meta.n_layers
    }
    fn vocab(&self) -> u32 {
        self.meta.vocab
    }
    fn ctx_size(&self) -> u32 {
        self.meta.ctx_size
    }
    fn hidden_values(&self) -> u64 {
        self.meta.hidden_values
    }
    fn has_output(&self) -> bool {
        self.meta.has_output
    }
    fn layer_start(&self) -> u32 {
        self.meta.layer_start
    }
    fn layer_end(&self) -> u32 {
        self.meta.layer_end
    }

    fn eval(&mut self, req: &WorkRequest) -> Result<WorkOutput, String> {
        if req.reset {
            self.session
                .borrow_mut()
                .layer_slice_reset()
                .map_err(|error| error.to_string())?;
        }
        let n = req.tokens.len();
        let mut output_hc = req
            .produce_hidden
            .then(|| vec![0.0; (n as u64).saturating_mul(self.meta.hidden_values) as usize]);
        let mut logits = req
            .produce_logits
            .then(|| vec![0.0; n.saturating_mul(self.meta.vocab as usize)]);
        self.session
            .borrow_mut()
            .eval_layer_slice(LayerSliceEval {
                tokens: &req.tokens,
                pos0: req.pos0,
                layer_start: req.layer_start,
                layer_end: req.layer_end,
                input_hc: (!req.input_hc.is_empty()).then_some(req.input_hc.as_slice()),
                output_hc: output_hc.as_deref_mut(),
                logits: logits.as_deref_mut(),
            })
            .map_err(|error| error.to_string())?;
        Ok(WorkOutput {
            hidden: output_hc,
            logits,
        })
    }
}

struct SessionSnapshotStore<'m> {
    session: Rc<RefCell<Session<'m>>>,
    layer_start: u32,
    layer_end: u32,
}

impl SnapshotStore for SessionSnapshotStore<'_> {
    type SaveReader = TempShard;

    fn save(&mut self, req: SnapshotSave) -> Result<(TempShard, u64), String> {
        let live = self.session.borrow();
        if live.host().tokens().len() as u32 != req.token_count {
            return Err("worker snapshot token count mismatch".into());
        }
        if token_hash_prefix(live.host().tokens()) != req.token_hash {
            return Err("worker snapshot token hash mismatch".into());
        }
        drop(live);
        let mut tmp = create_temp(SAVE_PREFIX)
            .map_err(|_| "failed to create worker snapshot temp file".to_string())?;
        self.session
            .borrow()
            .save_layer_payload(tmp.path(), self.layer_start, self.layer_end)
            .map_err(|_| "failed to save worker KV shard".to_string())?;
        tmp.rewind()
            .map_err(|_| "failed to rewind worker KV shard".to_string())?;
        let payload_bytes = tmp
            .seek(SeekFrom::End(0))
            .map_err(|_| "failed to measure worker KV shard".to_string())?;
        tmp.rewind()
            .map_err(|_| "failed to rewind worker KV shard".to_string())?;
        Ok((tmp, payload_bytes))
    }

    fn load(&mut self, req: SnapshotLoad<'_>, payload: &mut dyn Read) -> Result<(), String> {
        let mut tmp = create_temp(LOAD_PREFIX)
            .map_err(|_| "failed to restore worker KV shard".to_string())?;
        copy_chunked(payload, &mut tmp, req.payload_bytes)
            .map_err(|_| "failed to restore worker KV shard".to_string())?;
        tmp.flush()
            .map_err(|_| "failed to restore worker KV shard".to_string())?;
        self.session
            .borrow_mut()
            .load_layer_payload(LayerPayloadLoad {
                path: tmp.path(),
                payload_bytes: req.payload_bytes,
                tokens: req.tokens,
                layer_start: self.layer_start,
                layer_end: self.layer_end,
            })
            .map_err(|_| "failed to restore worker KV shard".to_string())?;
        Ok(())
    }
}

pub fn run_assembled_worker(model: &Model, ctx: i32, dist: &Options) -> Result<i32, String> {
    let session = Rc::new(RefCell::new(model.session(ctx).map_err(|e| e.to_string())?));
    let (listener, bound) = open_data_listener(
        dist.listen_host.as_deref(),
        worker_listen_port(dist.listen_port),
    )
    .map_err(|e| e.to_string())?;
    let shape = &model.bind_plan().shape;
    let vocab = u32::try_from(model.vocab().n_vocab().max(0)).unwrap_or(0);
    let meta = slice_meta(
        u32::try_from(model.model_id()).unwrap_or(0),
        shape.n_layer,
        vocab,
        u32::try_from(ctx).unwrap_or(0),
        hidden_values(shape),
        &dist.layers,
    );
    let plan = worker_plan(
        &meta,
        u32::try_from(model.routed_quant_bits().max(0)).unwrap_or(0),
        u32::from(bound),
        shape.name,
    );
    let exec = SessionSliceExec {
        session: Rc::clone(&session),
        meta,
    };
    let store = SessionSnapshotStore {
        session,
        layer_start: meta.layer_start,
        layer_end: meta.layer_end,
    };
    let mut assembled = assemble_worker(exec, store, plan.hello, plan.model_name);
    let host = dist
        .coordinator_host
        .as_deref()
        .ok_or("--role worker requires --coordinator HOST PORT")?;
    print_worker_listen_banner(&WorkerListenBanner {
        layer_start: assembled.hello.layer_start,
        has_output: assembled.hello.has_output != 0,
        layer_end: assembled.hello.layer_end,
        model_id: model.model_id(),
        listen_host: dist.listen_host.as_deref(),
        listen_port: assembled.hello.listen_port,
        coordinator_host: host,
        coordinator_port: dist.coordinator_port,
    });
    let port = u16::try_from(dist.coordinator_port)
        .map_err(|_| "--role worker requires --coordinator HOST PORT")?;
    ds4_dist::reconnect_local(
        &mut assembled.worker,
        ds4_dist::LocalReconnect {
            connect: || ds4_dist::connect_endpoint(host, port),
            hello: &assembled.hello,
            model_name: &assembled.model_name,
            sleep: ds4_dist::sleep_reconnect,
            should_stop: || false,
            listener: Some(&listener),
        },
    )
    .map_err(|e| e.to_string())?;
    Ok(0)
}
