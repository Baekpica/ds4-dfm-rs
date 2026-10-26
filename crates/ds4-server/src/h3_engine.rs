//! In-process renderer for the h3 local v1 host.
//!
//! The renderer trait is unchanged; only the implementation differs. The
//! engine keeps the prompt embedding and the prepared DiT across requests
//! (`ds4_core::minimax_h3::engine`, mirroring `h3.h:180-184`), and it owns a
//! thread because the device handles are `!Send`, which also makes admission
//! one render at a time.

use std::path::{Path, PathBuf};

use ds4_core::minimax_h3::engine::{EngineHandle, RenderRequest};
use ds4_core::minimax_h3::local_api::{Plan, ReferenceKind};
use ds4_core::minimax_h3::reference;

use crate::h3_video::{MediaFile, RenderError, Renderer, RendererState};

pub struct EngineRenderer {
    engine: EngineHandle,
    steps: usize,
    reuse: usize,
}

impl EngineRenderer {
    /// Open the checkpoint on the engine thread. The settings the local body
    /// does not carry (steps, reuse) are chosen here at startup, never by a
    /// caller; the seed and the target ride the request.
    pub fn new(model_dir: PathBuf, steps: usize, reuse: usize) -> Result<EngineRenderer, String> {
        let engine = EngineHandle::spawn(model_dir).map_err(|error| error.to_string())?;

        Ok(EngineRenderer {
            engine,
            steps,
            reuse,
        })
    }
}

impl Renderer for EngineRenderer {
    fn state(&self) -> RendererState {
        RendererState::Ready
    }

    fn render(&self, spec: &Plan, output: &Path) -> Result<(), RenderError> {
        // The engine reads pixels from a path, as the reference does
        // (`h3.c:1231-1257`); the decoded wire bytes live beside the output
        // for the duration of the render.
        let first = MediaFile::write(spec.first_frame.as_ref(), output, "first")?;
        let last = MediaFile::write(spec.last_frame.as_ref(), output, "last")?;
        let files = spec
            .references
            .iter()
            .enumerate()
            .map(|(index, reference)| MediaFile::write_reference(reference, output, index))
            .collect::<Result<Vec<_>, _>>()?;
        let references = files
            .iter()
            .zip(&spec.references)
            .map(|(file, media)| reference::Reference {
                kind: match media.kind {
                    ReferenceKind::Image => reference::Kind::Image,
                    ReferenceKind::Video => reference::Kind::Video,
                    ReferenceKind::Audio => reference::Kind::Audio,
                },
                path: file.path().to_path_buf(),
                audio_path: None,
                include_embedded_audio: media.include_audio,
            })
            .collect();
        let request = RenderRequest {
            prompt: spec.prompt.clone(),
            // The local contract has one canvas: the file is the model canvas.
            width: spec.size.width as i32,
            height: spec.size.height as i32,
            frames: spec.frames as i32,
            steps: self.steps,
            reuse: self.reuse,
            seed: spec.seed,
            first_frame: first.as_ref().map(|file| file.path().to_path_buf()),
            last_frame: last.as_ref().map(|file| file.path().to_path_buf()),
            references,
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
