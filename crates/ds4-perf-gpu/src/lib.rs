// This helper owns profiler CUDA calls; it is never an inference dependency.
#![cfg_attr(not(feature = "cuda"), forbid(unsafe_code))]

#[cfg(feature = "cuda")]
pub mod device;

#[cfg(feature = "cuda")]
pub mod calibration;

#[cfg(feature = "cuda")]
mod collector;
