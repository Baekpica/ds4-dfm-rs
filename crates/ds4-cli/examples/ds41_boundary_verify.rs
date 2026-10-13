//! Live DSpark regression: short raw prompts must complete like plain decode.
//! Open one weight owner; each case creates fresh request state.
//!
//! make tests/ds41_boundary_verify
//! tests/ds41_boundary_verify MODEL ENGRAM_DIR [--no-engram]

use ds4_core::{Backend, Model, V41RunOptions};
use std::path::Path;

const PROMPT_COUNTS: [usize; 4] = [1, 2, 5, 6];
const OUTPUT_COUNTS: [i32; 2] = [1, 16];
const VERIFY_K: i32 = 5;

#[derive(Clone, Copy, Debug)]
enum Mode {
    Plain,
    Draft,
}

fn generate(
    model: &Model,
    path: &Path,
    prompt: &[i32],
    n_predict: i32,
    engram_dir: Option<&str>,
    mode: Mode,
) -> Result<Vec<i32>, String> {
    let mut opts = V41RunOptions {
        engram_dir,
        no_engram: engram_dir.is_none(),
        dspark: Some(match mode {
            Mode::Plain => 0,
            Mode::Draft => 2,
        }),
        graph: Some(true),
        verify_k: VERIFY_K,
        emit_trace: true,
        kv_reuse: Some(false),
        ..V41RunOptions::default()
    };
    let mut ids = Vec::new();
    let mut emit = |token| {
        ids.push(token);
        true
    };
    let mut progress = |_: &str, _: i32, _: i32| true;
    model
        .v41_generate(path, prompt, n_predict, &mut opts, &mut emit, &mut progress)
        .map_err(|error| {
            format!(
                "{mode:?}: prompt={} output={n_predict}: {error}",
                prompt.len()
            )
        })?;
    Ok(ids)
}

fn run() -> Result<(), String> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.len() < 2 || args.len() > 3 {
        return Err("usage: ds41_boundary_verify MODEL ENGRAM_DIR [--no-engram]".into());
    }
    let engram_dir = match args.get(2).map(String::as_str) {
        None => Some(args[1].as_str()),
        Some("--no-engram") => None,
        Some(arg) => return Err(format!("unknown argument: {arg}")),
    };
    let path = Path::new(&args[0]);
    let model = Model::open(&args[0], Backend::Cuda, 0, true).map_err(|e| e.to_string())?;
    if model.v41_ctx().is_none() {
        return Err("this gate requires a DeepSeek V4.1 artifact".into());
    }
    let seed = model
        .tokenize_text("The capital of France is Paris. Continue with a short explanation.")
        .map_err(|e| e.to_string())?;
    if seed.len() < *PROMPT_COUNTS.last().unwrap() {
        return Err("seed text does not provide six raw tokens".into());
    }

    for count in PROMPT_COUNTS {
        let prompt = &seed.as_slice()[..count];
        for n_predict in OUTPUT_COUNTS {
            let plain = generate(&model, path, prompt, n_predict, engram_dir, Mode::Plain)?;
            let draft = generate(&model, path, prompt, n_predict, engram_dir, Mode::Draft)?;
            if plain.is_empty() || plain != draft {
                return Err(format!(
                    "prompt={count} output={n_predict}: plain={plain:?} draft={draft:?}"
                ));
            }
            // A successful fallback must not make the speculative comparison vacuous.
            let (rounds, offered, _) = ds4_core::v41_last_spec_stats();
            if n_predict > 1 && (rounds < 1 || offered < 1) {
                return Err(format!("prompt={count}: DSpark performed no verification"));
            }
            println!(
                "PASS prompt={count} output={n_predict} emitted={} spec_rounds={rounds}",
                draft.len()
            );
        }
    }
    Ok(())
}

fn main() -> std::process::ExitCode {
    match run() {
        Ok(()) => std::process::ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("ds41 boundary gate: FAIL: {error}");
            std::process::ExitCode::FAILURE
        }
    }
}
