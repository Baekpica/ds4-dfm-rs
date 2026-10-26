//! The CUDA device identity the h3 host reports (`native/h3/h3_device_gpu.h`).
//!
//! Mirrors the reference probe `h3_device_probe`
//! (`/data/h3.c` at `eae5d54`, `h3_device_cuda.cu:17-50`) and the struct
//! `h3_device_info` (`h3.h:130-141`). The probe is additive: a new `h3dev_`
//! name, no existing family's device query is shared or changed.
//!
//! Field mapping, in the reference's units: `name` and `architecture` are the
//! `char[128]` strings (`h3.h:131-132`), the three memory fields are bytes
//! (`h3.h:133-135`), and `apple_gpu_family`/`metal4` are Metal-only fields the
//! CUDA probe leaves zero (`h3_device_cuda.cu:23`). `recommended_working_set`
//! and `max_buffer_length` are both `totalGlobalMem` (`:46-47`), not two
//! different limits.

use std::ffi::CStr;
use std::os::raw::{c_char, c_int};

/// `h3_device_info.name` and `.architecture` are `char[128]` (`h3.h:131-132`).
const NAME_SIZE: usize = 128;
/// The `error[256]` every reference entry point takes.
const ERROR_SIZE: usize = 256;

/// `h3_device_info` (`h3.h:130-141`) with the C layout intact.
#[repr(C)]
struct RawDeviceInfo {
    name: [c_char; NAME_SIZE],
    architecture: [c_char; NAME_SIZE],
    physical_memory: u64,
    recommended_working_set: u64,
    max_buffer_length: u64,
    apple_gpu_family: c_int,
    metal4: c_int,
    unified_memory: c_int,
}

unsafe extern "C" {
    fn h3dev_probe(info: *mut RawDeviceInfo, error: *mut c_char, error_size: usize) -> c_int;
}

impl RawDeviceInfo {
    fn zeroed() -> RawDeviceInfo {
        RawDeviceInfo {
            name: [0; NAME_SIZE],
            architecture: [0; NAME_SIZE],
            physical_memory: 0,
            recommended_working_set: 0,
            max_buffer_length: 0,
            apple_gpu_family: 0,
            metal4: 0,
            unified_memory: 0,
        }
    }
}

/// The reference's `h3_device_info`, in CUDA terms.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeviceInfo {
    pub name: String,
    pub architecture: String,
    /// Whole-machine RAM in bytes (`h3_device_cuda.cu:41-45`); 0 off Linux.
    pub physical_memory: u64,
    /// `totalGlobalMem` (`:46`).
    pub recommended_working_set: u64,
    /// `totalGlobalMem` (`:47`), the reference's second field of the same
    /// figure.
    pub max_buffer_length: u64,
    /// Metal-only (`h3.h:135`); 0 on every CUDA device.
    pub apple_gpu_family: i32,
    /// Metal 4 support (`h3.h:136`); never set by the CUDA probe.
    pub metal4: bool,
    /// `unifiedAddressing` (`h3_device_cuda.cu:48`).
    pub unified_memory: bool,
}

/// Fill the reference's device record, or return its own error text.
pub fn probe() -> Result<DeviceInfo, String> {
    let mut raw = RawDeviceInfo::zeroed();
    let mut error = [0 as c_char; ERROR_SIZE];

    let probed = unsafe { h3dev_probe(&mut raw, error.as_mut_ptr(), ERROR_SIZE) };
    if probed == 0 {
        return Err(message(&error));
    }

    Ok(DeviceInfo {
        name: field(&raw.name),
        architecture: field(&raw.architecture),
        physical_memory: raw.physical_memory,
        recommended_working_set: raw.recommended_working_set,
        max_buffer_length: raw.max_buffer_length,
        apple_gpu_family: raw.apple_gpu_family,
        metal4: raw.metal4 != 0,
        unified_memory: raw.unified_memory != 0,
    })
}

/// One `snprintf("%.127s")` field: the native side always terminates it.
fn field(raw: &[c_char; NAME_SIZE]) -> String {
    let text = unsafe { CStr::from_ptr(raw.as_ptr()) };
    text.to_string_lossy().into_owned()
}

/// The C string in `error`, or a placeholder if the native side wrote none.
fn message(error: &[c_char; ERROR_SIZE]) -> String {
    let end = error
        .iter()
        .position(|byte| *byte == 0)
        .unwrap_or(ERROR_SIZE);
    let bytes: Vec<u8> = error[..end].iter().map(|byte| *byte as u8).collect();
    let text = String::from_utf8_lossy(&bytes).into_owned();

    if text.is_empty() {
        "native h3 device probe failed without a message".to_string()
    } else {
        text
    }
}
