//! Reject impossible draft widths before artifact validation or native open.
use ds4_core::{Backend, Model, ModelOpenOption};
use ds4_sys::{
    ds4_bridge_bind_plan, ds4_bridge_distributed_options, ds4_bridge_model,
    ds4_bridge_model_open_options,
};
use std::ffi::c_char;
use std::path::PathBuf;

const GGUF_VERSION: u32 = 3;
const GGUF_STRING: u32 = 8;
const GGUF_ALIGNMENT: usize = 32;

// Metadata-only fixtures must fail before any native work is reached.
#[no_mangle]
extern "C" fn ds4_bridge_bind_plan_check(
    _: *const ds4_bridge_bind_plan,
    _: *mut c_char,
    _: usize,
) -> i32 {
    unreachable!("metadata-only fixture reached native binding")
}

#[no_mangle]
extern "C" fn ds4_bridge_model_open(
    _: *mut *mut ds4_bridge_model,
    _: *const ds4_bridge_model_open_options,
    _: *mut c_char,
    _: usize,
) -> i32 {
    unreachable!("metadata-only fixture reached native open")
}

#[no_mangle]
extern "C" fn ds4_bridge_model_open_distributed(
    _: *mut *mut ds4_bridge_model,
    _: *const ds4_bridge_model_open_options,
    _: *const ds4_bridge_distributed_options,
    _: *mut c_char,
    _: usize,
) -> i32 {
    unreachable!("metadata-only fixture reached native distributed open")
}

#[no_mangle]
extern "C" fn ds4_bridge_model_free(_: *mut ds4_bridge_model) {
    unreachable!("metadata-only fixture created a native model")
}

struct Fixture(PathBuf);

impl Fixture {
    fn new(arch: &str) -> Self {
        let path = std::env::temp_dir().join(format!(
            "ds4-iquest-open-{}-{arch}.gguf",
            std::process::id()
        ));
        let mut bytes = Vec::from(*b"GGUF");
        bytes.extend_from_slice(&GGUF_VERSION.to_le_bytes());
        bytes.extend_from_slice(&0u64.to_le_bytes());
        bytes.extend_from_slice(&1u64.to_le_bytes());
        for (index, value) in ["general.architecture", arch].into_iter().enumerate() {
            if index == 1 {
                bytes.extend_from_slice(&GGUF_STRING.to_le_bytes());
            }
            bytes.extend_from_slice(&(value.len() as u64).to_le_bytes());
            bytes.extend_from_slice(value.as_bytes());
        }
        bytes.resize(bytes.len().next_multiple_of(GGUF_ALIGNMENT), 0);
        std::fs::write(&path, bytes).unwrap();
        Self(path)
    }

    fn error(&self, options: &[ModelOpenOption]) -> String {
        match Model::open_configured(
            self.0.to_str().unwrap(),
            Backend::Cuda,
            0,
            true,
            None,
            options,
        ) {
            Ok(_) => panic!("metadata-only fixture must not load"),
            Err(error) => error.to_string(),
        }
    }
}

impl Drop for Fixture {
    fn drop(&mut self) {
        std::fs::remove_file(&self.0).unwrap();
    }
}

#[test]
fn unsupported_drafts_fail_early() {
    let fixture = Fixture::new("iquest_q1");
    for width in (8..=17).chain([i32::MAX]) {
        let message = fixture.error(&[ModelOpenOption::MtpDraftTokens(width)]);
        assert!(
            message.contains("IQuest-Q1 accepts at most seven recursive draft tokens"),
            "width {width}: {message}"
        );
    }
    // Valid widths and the omitted native default still reach artifact checks.
    for options in [
        vec![],
        vec![ModelOpenOption::MtpDraftTokens(1)],
        vec![ModelOpenOption::MtpDraftTokens(7)],
    ] {
        assert!(fixture.error(&options).contains("validate failed:"));
    }
    assert!(fixture
        .error(&[ModelOpenOption::MtpDraftTokens(0)])
        .contains("mtp draft tokens must be positive"));
}

#[test]
fn other_family_keeps_draft_rules() {
    let fixture = Fixture::new("solar-open2");
    let message = fixture.error(&[ModelOpenOption::MtpDraftTokens(i32::MAX)]);
    assert!(message.contains("validate failed:"), "{message}");
}
