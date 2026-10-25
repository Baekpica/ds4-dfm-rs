//! The portable byte-level BPE tokenizer, against the reference's own fixture.
//!
//! Reference: `/data/h3.c` at `eae5d54`, `tests/test_tokenizer_portable.c:36-48`
//! and the fixture it is written against, `tests/tokenizer_portable_fixture.json`.
//! The fixture is vendored at `tests/h3/fixtures/` byte for byte and pinned by
//! sha256 below, so a changed fixture fails before any assertion runs.
//!
//! Assertions, in the reference's order:
//!
//! - `"A A"` encodes to `{1, 3}`: the space is its own merge (`Ġ`), and
//!   decoding returns `"A A"`.
//! - `"e"` + U+0301 is normalized to NFC (`Ã©`, id 4) and decodes to `"é"`.
//! - `"<special>A"` matches the added token before normalization, then `"A"`:
//!   `{10, 1}`, decoding to the same text.
//! - An empty prompt with the pad flag becomes `{H3_PAD_TOKEN_ID}`.
//!
//! Model-free and host-only: run with `make tests/h3-test_tokenizer_portable`.
//! The optional argument overrides the fixture path, as the reference's
//! `argv[1]` does; the pin applies to whatever file is loaded.

use std::path::Path;
use std::process::ExitCode;

use ds4_core::minimax_h3::tokenizer::{EmptyPrompt, Tokenizer, PAD_TOKEN_ID};
use sha2::{Digest, Sha256};

const DEFAULT_FIXTURE: &str = "tests/h3/fixtures/tokenizer_portable_fixture.json";

/// sha256 of `/data/h3.c/tests/tokenizer_portable_fixture.json` at `eae5d54`.
const FIXTURE_SHA256: &str = "4f300dcd98c564e1e04b0131a1a50f5b5f30b467cb48a45c6ce6745240a47e73";

fn main() -> ExitCode {
    let path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| DEFAULT_FIXTURE.to_string());

    match run(Path::new(&path)) {
        Ok(()) => {
            println!("ok: portable ICU byte-level BPE tokenizer");
            ExitCode::SUCCESS
        }
        Err(message) => {
            eprintln!("FAIL tests/h3/test_tokenizer_portable.rs: {message}");
            ExitCode::from(1)
        }
    }
}

fn run(path: &Path) -> Result<(), String> {
    check_pin(path)?;
    let tokenizer = Tokenizer::load(path).map_err(|error| error.token())?;

    // "A A": the space survives as its own symbol.
    check(&tokenizer, "A A", &[1, 3], "A A")?;

    // "e" + combining acute: NFC composes it to the fixture's "é".
    check(&tokenizer, "e\u{301}", &[4], "é")?;

    // An added token is matched in the raw text before normalization.
    check(&tokenizer, "<special>A", &[10, 1], "<special>A")?;

    let ids = tokenizer
        .encode("", EmptyPrompt::Pad)
        .map_err(|error| error.token())?;

    if ids != [PAD_TOKEN_ID] {
        return Err(format!(
            "empty prompt encoded to {ids:?}, expected {{{PAD_TOKEN_ID}}}"
        ));
    }

    Ok(())
}

fn check(tokenizer: &Tokenizer, text: &str, expected: &[u32], decoded: &str) -> Result<(), String> {
    let ids = tokenizer
        .encode(text, EmptyPrompt::Nothing)
        .map_err(|error| error.token())?;

    if ids != expected {
        return Err(format!(
            "{text:?} encoded to {ids:?}, expected {expected:?}"
        ));
    }

    let back = tokenizer.decode(&ids).map_err(|error| error.token())?;

    if back != decoded {
        return Err(format!("{ids:?} decoded to {back:?}, expected {decoded:?}"));
    }

    Ok(())
}

/// A golden that is not pinned is not evidence: refuse a substituted fixture.
fn check_pin(path: &Path) -> Result<(), String> {
    let bytes =
        std::fs::read(path).map_err(|error| format!("cannot read {}: {error}", path.display()))?;
    let digest: [u8; 32] = Sha256::digest(&bytes).into();
    let hex: String = digest.iter().map(|byte| format!("{byte:02x}")).collect();

    if hex != FIXTURE_SHA256 {
        return Err(format!(
            "{} has sha256 {hex}, expected {FIXTURE_SHA256}",
            path.display()
        ));
    }

    Ok(())
}
