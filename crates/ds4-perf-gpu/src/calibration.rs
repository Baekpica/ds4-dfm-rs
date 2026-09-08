use cudarc::{
    driver::{CudaContext, LaunchConfig, PushKernelArg},
    nvrtc::{compile_ptx_with_opts, CompileOptions},
};
use ds4_perf::machine::Calibration;
use std::time::Instant;

const WARMUPS: u32 = 2;
const REPEATS: u32 = 7;
const FMA_ITERATIONS: i32 = 4096;
const LAUNCHES: u32 = 256;
const SOURCE: &str = r#"
extern "C" __global__ void copy_words(float *out, const float *input, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = input[i];
}
extern "C" __global__ void fma_envelope(float *out, int iterations) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    float a=0.1f,b=0.2f,c=0.3f,d=0.4f,e=0.5f,f=0.6f,g=0.7f,h=0.8f;
    for (int j=0; j<iterations; ++j) {
        a=fmaf(a,0.9999f,0.0001f); b=fmaf(b,0.9999f,0.0001f);
        c=fmaf(c,0.9999f,0.0001f); d=fmaf(d,0.9999f,0.0001f);
        e=fmaf(e,0.9999f,0.0001f); f=fmaf(f,0.9999f,0.0001f);
        g=fmaf(g,0.9999f,0.0001f); h=fmaf(h,0.9999f,0.0001f);
    }
    out[i]=a+b+c+d+e+f+g+h;
}
extern "C" __global__ void empty_launch() {}
"#;

pub fn run(ordinal: usize) -> Result<Calibration, String> {
    measure(ordinal).map_err(|e| e.to_string())
}

fn measure(ordinal: usize) -> Result<Calibration, Box<dyn std::error::Error>> {
    let gpu = crate::device::inspect(ordinal)?;
    let ctx = CudaContext::new(ordinal)?;
    let stream = ctx.default_stream();
    // Larger than L2; bounded to 512 MiB total so this is not a memory stress test.
    let bytes = (64 * 1024 * 1024usize).max(gpu.l2_bytes.max(0) as usize * 2);
    if bytes > 256 * 1024 * 1024 || gpu.free_memory_bytes < (bytes * 4) as u64 {
        return Err("insufficient bounded calibration headroom".into());
    }
    let n = (bytes / 4) as i32;
    let source = stream.clone_htod(&vec![1.0f32; n as usize])?;
    let mut target = stream.alloc_zeros::<f32>(n as usize)?;
    let workers = gpu.multiprocessors as u32 * 256 * 8;
    let mut compute = stream.alloc_zeros::<f32>(workers as usize)?;
    let ptx = compile_ptx_with_opts(
        SOURCE,
        CompileOptions {
            options: vec![format!(
                "--gpu-architecture=compute_{}{}",
                gpu.compute_major, gpu.compute_minor
            )],
            ..Default::default()
        },
    )?;
    let ptx_sha256 = ds4_perf::artifact::hash_bytes(ptx.to_src().as_bytes());
    let module = ctx.load_module(ptx)?;
    let copy = module.load_function("copy_words")?;
    let fma = module.load_function("fma_envelope")?;
    let empty = module.load_function("empty_launch")?;
    let mut copy_gb_s = Vec::new();
    let mut fp32_gflop_s = Vec::new();
    let mut launch_us = Vec::new();
    let mut copy_times = Vec::new();
    let mut compute_times = Vec::new();
    let mut launch_times = Vec::new();
    for repetition in 0..WARMUPS + REPEATS {
        let start =
            stream.record_event(Some(cudarc::driver::sys::CUevent_flags::CU_EVENT_DEFAULT))?;
        // All pointers are cudarc-owned slices on this stream; n bounds the kernel.
        unsafe {
            stream
                .launch_builder(&copy)
                .arg(&mut target)
                .arg(&source)
                .arg(&n)
                .launch(LaunchConfig::for_num_elems(n as u32))?;
        }
        let end =
            stream.record_event(Some(cudarc::driver::sys::CUevent_flags::CU_EVENT_DEFAULT))?;
        end.synchronize()?;
        let copy_ms = start.elapsed_ms(&end)? as f64;

        let start =
            stream.record_event(Some(cudarc::driver::sys::CUevent_flags::CU_EVENT_DEFAULT))?;
        // Exactly workers threads write an allocation of workers f32 elements.
        unsafe {
            stream
                .launch_builder(&fma)
                .arg(&mut compute)
                .arg(&FMA_ITERATIONS)
                .launch(LaunchConfig {
                    grid_dim: (workers / 256, 1, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                })?;
        }
        let end =
            stream.record_event(Some(cudarc::driver::sys::CUevent_flags::CU_EVENT_DEFAULT))?;
        end.synchronize()?;
        let compute_ms = start.elapsed_ms(&end)? as f64;

        let host_start = Instant::now();
        for _ in 0..LAUNCHES {
            // This kernel has no arguments or memory accesses.
            unsafe {
                stream.launch_builder(&empty).launch(LaunchConfig {
                    grid_dim: (1, 1, 1),
                    block_dim: (1, 1, 1),
                    shared_mem_bytes: 0,
                })?;
            }
        }
        stream.synchronize()?;
        let launch_ms = host_start.elapsed().as_secs_f64() * 1000.0;
        let latency = launch_ms * 1000.0 / LAUNCHES as f64;
        if repetition >= WARMUPS {
            copy_gb_s.push(2.0 * bytes as f64 / (copy_ms * 1e6));
            fp32_gflop_s.push(workers as f64 * FMA_ITERATIONS as f64 * 16.0 / (compute_ms * 1e6));
            launch_us.push(latency);
            copy_times.push(copy_ms);
            compute_times.push(compute_ms);
            launch_times.push(launch_ms);
        }
    }
    if stream.clone_dtoh(&target)?.iter().any(|v| *v != 1.0) {
        return Err("copy calibration validation failed".into());
    }
    let compute_result = stream.clone_dtoh(&compute)?;
    if !valid_compute(&compute_result) {
        return Err("FMA calibration validation failed".into());
    }
    if [&copy_gb_s, &fp32_gflop_s, &launch_us]
        .iter()
        .any(|values| values.iter().any(|v| !v.is_finite() || *v <= 0.0))
    {
        return Err("invalid calibration timing".into());
    }
    Ok(Calibration { ptx_sha256, method:"cuda-event-copy-fp32-fma-v1".into(),source_sha256:ds4_perf::artifact::hash_bytes(SOURCE.as_bytes()),
        copy_ms:copy_times,compute_ms:compute_times,launch_batch_ms:launch_times,
        compute_threads:workers,compute_block_threads:256,fma_iterations:FMA_ITERATIONS as u32,launch_count:LAUNCHES,
        gpu, transfer_bytes: bytes as u64, warmups: WARMUPS, repeats: REPEATS, copy_gb_s, fp32_gflop_s, launch_us,
        notes: vec!["Copy counts read plus write bytes, uses CUDA event time, and exceeds L2 capacity.".into(),
                    "FP32 measures eight independent SIMT FMA chains; it is not Tensor Core peak throughput.".into(),
                    "Launch latency is host wall time for 256 empty kernels plus completion synchronization, per kernel.".into(),
                    "Observed rates depend on current clocks, temperature and other processes; this is a measured envelope, not a hardware guarantee.".into()] })
}

fn valid_compute(values: &[f32]) -> bool {
    let mut expected = [0.1f32, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8];
    for _ in 0..FMA_ITERATIONS {
        for value in &mut expected {
            *value = value.mul_add(0.9999, 0.0001);
        }
    }
    let expected: f32 = expected.iter().sum();
    !values.is_empty()
        && values
            .iter()
            .all(|v| v.is_finite() && (*v - expected).abs() <= 0.00002)
}

#[cfg(test)]
mod tests {
    #[test]
    fn rejects_short_fma_work() {
        let mut one_step = [0.1f32, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8];
        for value in &mut one_step {
            *value = value.mul_add(0.9999, 0.0001);
        }
        assert!(!super::valid_compute(&[one_step.iter().sum()]));
    }
}
