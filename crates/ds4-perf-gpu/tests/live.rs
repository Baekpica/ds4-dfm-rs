#![cfg(feature = "cuda")]

#[test]
#[ignore = "requires an idle CUDA device and NVRTC"]
fn measured_envelope() {
    let result = ds4_perf_gpu::calibration::run(0).unwrap();
    for samples in [&result.copy_gb_s, &result.fp32_gflop_s, &result.launch_us] {
        assert_eq!(samples.len(), result.repeats as usize);
        assert!(samples
            .iter()
            .all(|value| value.is_finite() && *value > 0.0));
    }
    assert!(result.transfer_bytes > result.gpu.l2_bytes as u64);
}
