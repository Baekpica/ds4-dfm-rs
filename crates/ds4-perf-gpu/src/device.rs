use cudarc::driver::{sys::CUdevice_attribute::*, CudaContext};
use ds4_perf::machine::Gpu;

pub fn inspect(ordinal: usize) -> Result<Gpu, String> {
    let ctx = CudaContext::new(ordinal).map_err(|e| e.to_string())?;
    let attr = |key| ctx.attribute(key).map_err(|e| e.to_string());
    let (free, total) = ctx.mem_get_info().map_err(|e| e.to_string())?;
    let uuid = ctx.uuid().map_err(|e| e.to_string())?;
    let mut driver_version = 0;
    // CUDA writes one integer to this live local, through cudarc's bindings.
    unsafe { cudarc::driver::sys::cuDriverGetVersion(&mut driver_version) }
        .result()
        .map_err(|e| e.to_string())?;
    let mut unavailable = std::collections::BTreeMap::new();
    let mut optional = |name: &str, key| match attr(key) {
        Ok(value) if value > 0 => Some(value),
        other => {
            unavailable.insert(name.into(), format!("{other:?}"));
            None
        }
    };
    let memory_bus_bits = optional(
        "memory_bus_bits",
        CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH,
    );
    let memory_clock_khz = optional("memory_clock_khz", CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE);
    let clock_khz = optional("clock_khz", CU_DEVICE_ATTRIBUTE_CLOCK_RATE);
    Ok(Gpu {
        ordinal,
        name: ctx.name().map_err(|e| e.to_string())?,
        uuid: uuid.bytes.iter().map(|b| format!("{:02x}", *b)).collect(),
        compute_major: attr(CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR)?,
        compute_minor: attr(CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR)?,
        driver_version,
        total_memory_bytes: total as u64,
        free_memory_bytes: free as u64,
        multiprocessors: attr(CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)?,
        warp_size: attr(CU_DEVICE_ATTRIBUTE_WARP_SIZE)?,
        max_threads_per_block: attr(CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK)?,
        max_threads_per_sm: attr(CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR)?,
        max_blocks_per_sm: attr(CU_DEVICE_ATTRIBUTE_MAX_BLOCKS_PER_MULTIPROCESSOR)?,
        registers_per_sm: attr(CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_MULTIPROCESSOR)?,
        shared_bytes_per_sm: attr(CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR)?,
        shared_bytes_per_block: attr(CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN)?,
        l2_bytes: attr(CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE)?,
        memory_bus_bits,
        memory_clock_khz,
        clock_khz,
        unavailable,
    })
}
