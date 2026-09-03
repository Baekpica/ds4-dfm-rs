//! Worker `SliceExec` over a live `Session` layer-slice eval.

use std::cell::RefCell;
use std::rc::Rc;

use ds4_core::{LayerSliceEval, Model, ModelFamily, Session, Shape};
use ds4_dist::{resolved_layer_end, Layers, SliceExec, WorkOutput, WorkRequest};

pub fn hidden_values(shape: &Shape) -> u64 {
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

pub fn hidden_buffer_len(hidden_values: u64, n_tokens: usize) -> usize {
    (n_tokens as u64).saturating_mul(hidden_values) as usize
}

pub fn logits_buffer_len(vocab: u32, n_tokens: usize) -> usize {
    n_tokens.saturating_mul(vocab as usize)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SliceMeta {
    pub model_id: u32,
    pub n_layers: u32,
    pub vocab: u32,
    pub ctx_size: u32,
    pub hidden_values: u64,
    pub has_output: bool,
    pub layer_start: u32,
    pub layer_end: u32,
}

pub fn slice_meta(model_id: u32, shape: &Shape, ctx_size: u32, layers: &Layers) -> SliceMeta {
    let n_layers = shape.n_layer;
    SliceMeta {
        model_id,
        n_layers,
        vocab: shape.n_vocab,
        ctx_size,
        hidden_values: hidden_values(shape),
        has_output: layers.has_output,
        layer_start: layers.start,
        layer_end: resolved_layer_end(layers, n_layers),
    }
}

pub struct SessionSliceExec<'m> {
    session: Rc<RefCell<Session<'m>>>,
    meta: SliceMeta,
}

impl<'m> SessionSliceExec<'m> {
    pub fn new(session: Session<'m>, model: &Model, ctx_size: u32, layers: &Layers) -> Self {
        Self::from_shared(Rc::new(RefCell::new(session)), model, ctx_size, layers)
    }

    pub fn from_shared(
        session: Rc<RefCell<Session<'m>>>,
        model: &Model,
        ctx_size: u32,
        layers: &Layers,
    ) -> Self {
        let shape = &model.bind_plan().shape;
        let vocab = u32::try_from(model.vocab().n_vocab().max(0)).unwrap_or(0);
        let mut meta = slice_meta(model.model_id() as u32, shape, ctx_size, layers);
        meta.vocab = vocab;
        Self { session, meta }
    }

    pub fn from_meta(session: Rc<RefCell<Session<'m>>>, meta: SliceMeta) -> Self {
        Self { session, meta }
    }

    pub fn session(&self) -> &Rc<RefCell<Session<'m>>> {
        &self.session
    }

    pub fn meta(&self) -> SliceMeta {
        self.meta
    }
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
        let mut output_hc = if req.produce_hidden {
            Some(vec![
                0.0;
                hidden_buffer_len(
                    self.meta.hidden_values,
                    req.tokens.len()
                )
            ])
        } else {
            None
        };
        let mut logits = if req.produce_logits {
            Some(vec![
                0.0;
                logits_buffer_len(self.meta.vocab, req.tokens.len())
            ])
        } else {
            None
        };
        self.session
            .borrow_mut()
            .eval_layer_slice(LayerSliceEval {
                tokens: &req.tokens,
                pos0: req.pos0,
                layer_start: req.layer_start,
                layer_end: req.layer_end,
                input_hc: if req.input_hc.is_empty() {
                    None
                } else {
                    Some(req.input_hc.as_slice())
                },
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

#[cfg(test)]
mod tests {
    use super::*;
    use ds4_core::{
        SHAPE_FLASH, SHAPE_KEXAONE_236B, SHAPE_QWEN38_FLASH_NEXT, SHAPE_SOLAR_OPEN2_250B,
    };

    #[test]
    fn hidden_values_match_c_family_split() {
        assert_eq!(
            hidden_values(&SHAPE_FLASH),
            u64::from(SHAPE_FLASH.n_hc) * u64::from(SHAPE_FLASH.n_embd)
        );
        assert_eq!(
            hidden_values(&SHAPE_SOLAR_OPEN2_250B),
            u64::from(SHAPE_SOLAR_OPEN2_250B.n_embd)
        );
        assert_eq!(
            hidden_values(&SHAPE_KEXAONE_236B),
            u64::from(SHAPE_KEXAONE_236B.n_embd)
        );
        assert_eq!(
            hidden_values(&SHAPE_QWEN38_FLASH_NEXT),
            u64::from(SHAPE_QWEN38_FLASH_NEXT.n_embd)
        );
    }

    #[test]
    fn slice_meta_resolves_output_layer_end() {
        let layers = Layers {
            start: 20,
            end: 20,
            has_output: true,
            set: true,
        };
        let meta = slice_meta(7, &SHAPE_FLASH, 4096, &layers);
        assert_eq!(meta.model_id, 7);
        assert_eq!(meta.n_layers, 43);
        assert_eq!(meta.layer_start, 20);
        assert_eq!(meta.layer_end, 42);
        assert!(meta.has_output);
        assert_eq!(
            hidden_buffer_len(meta.hidden_values, 2),
            2 * hidden_values(&SHAPE_FLASH) as usize
        );
        assert_eq!(
            logits_buffer_len(meta.vocab, 2),
            2 * SHAPE_FLASH.n_vocab as usize
        );
    }
}
