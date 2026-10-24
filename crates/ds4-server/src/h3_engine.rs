//! In-process renderer for the OpenAI Videos host.
//!
//! The renderer trait is unchanged; only the implementation differs. The
//! engine keeps the prompt embedding and the prepared DiT across requests
//! (`ds4_core::minimax_h3::engine`, mirroring `h3.h:180-184`), and it owns a
//! thread because the device handles are `!Send`, which also makes admission
//! one render at a time.

use std::path::{Path, PathBuf};

use ds4_core::minimax_h3::engine::{EngineHandle, RenderRequest};
use ds4_core::minimax_h3::video_api::RenderSpec;

use crate::h3_video::{RenderError, Renderer, RendererState};

pub struct EngineRenderer {
    engine: EngineHandle,
    steps: usize,
    reuse: usize,
    seed: u64,
}

impl EngineRenderer {
    /// Open the checkpoint on the engine thread. The request fields the
    /// OpenAI surface does not carry (steps, reuse, seed) are chosen here at
    /// startup, never by a caller.
    pub fn new(
        model_dir: PathBuf,
        steps: usize,
        reuse: usize,
        seed: u64,
    ) -> Result<EngineRenderer, String> {
        let engine = EngineHandle::spawn(model_dir).map_err(|error| error.to_string())?;

        Ok(EngineRenderer {
            engine,
            steps,
            reuse,
            seed,
        })
    }
}

impl Renderer for EngineRenderer {
    fn state(&self) -> RendererState {
        RendererState::Ready
    }

    fn render(&self, spec: &RenderSpec, output: &Path) -> Result<(), RenderError> {
        let request = RenderRequest {
            prompt: spec.prompt.clone(),
            width: spec.size.width as i32,
            height: spec.size.height as i32,
            frames: spec.frames as i32,
            steps: self.steps,
            reuse: self.reuse,
            seed: self.seed,
            output: output.to_path_buf(),
        };
        let report = self
            .engine
            .render(request)
            .map_err(|error| RenderError::new("render_failed", error.to_string()))?;

        eprintln!("h3-engine: {} {}", output.display(), report.summary());

        Ok(())
    }
}
