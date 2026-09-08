// CUPTI injection follows NVIDIA's cupti_trace_injection sample. NVTX injection
// points directly to libcupti; all application annotations remain NVIDIA nvtx.
use cudarc::{
    cupti::{
        result::activity,
        sys::{self, CUpti_ActivityKind::*},
    },
    driver,
};
use ds4_perf::trace::Event;
use std::{
    alloc::{alloc, dealloc, Layout},
    ffi::CStr,
    fs::OpenOptions,
    io::{BufWriter, Write},
    sync::{
        atomic::{AtomicU64, Ordering},
        Mutex, OnceLock,
    },
};

const BUFFER_BYTES: usize = 8 * 1024 * 1024;
const BUFFER_ALIGN: usize = 8;
static OUTPUT: OnceLock<Mutex<BufWriter<std::fs::File>>> = OnceLock::new();
static INITIALIZED: OnceLock<Result<(), String>> = OnceLock::new();
static ERRORS: AtomicU64 = AtomicU64::new(0);
static DROPPED: AtomicU64 = AtomicU64::new(0);

fn emit(event: Event) {
    let Some(output) = OUTPUT.get() else {
        return;
    };
    let Ok(mut output) = output.lock() else {
        ERRORS.fetch_add(1, Ordering::Relaxed);
        return;
    };
    if serde_json::to_writer(&mut *output, &event).is_err() || output.write_all(b"\n").is_err() {
        ERRORS.fetch_add(1, Ordering::Relaxed);
    }
}

fn error(message: String) {
    ERRORS.fetch_add(1, Ordering::Relaxed);
    emit(Event::Error { message });
}

// These callbacks are called by CUPTI with its documented output pointers.
extern "C" fn request_buffer(buffer: *mut *mut u8, size: *mut usize, records: *mut usize) {
    let layout = Layout::from_size_align(BUFFER_BYTES, BUFFER_ALIGN).expect("constant layout");
    unsafe {
        *buffer = alloc(layout);
        *size = if (*buffer).is_null() { 0 } else { BUFFER_BYTES };
        *records = 0;
        if (*buffer).is_null() {
            ERRORS.fetch_add(1, Ordering::Relaxed);
        }
    }
}

extern "C" fn complete_buffer(
    context: driver::sys::CUcontext,
    stream: u32,
    buffer: *mut u8,
    _size: usize,
    valid: usize,
) {
    // No panic may cross CUPTI's C boundary. Record failures for the parent.
    let result = std::panic::catch_unwind(|| unsafe { consume(buffer, valid) });
    match result {
        Ok(Ok(())) => {}
        Ok(Err(message)) => error(message),
        Err(_) => error("activity callback panicked".into()),
    }
    let mut dropped = 0;
    // CUPTI owns context lifetime throughout this callback.
    match unsafe { activity::get_num_dropped_records(context, stream, &mut dropped) } {
        Ok(()) => {
            DROPPED.fetch_add(dropped as u64, Ordering::Relaxed);
        }
        Err(e) => error(e.to_string()),
    }
    if !buffer.is_null() {
        // This is exactly the allocation made by request_buffer.
        unsafe {
            dealloc(
                buffer,
                Layout::from_size_align(BUFFER_BYTES, BUFFER_ALIGN).expect("constant layout"),
            );
        }
    }
}

unsafe fn string(pointer: *const std::ffi::c_char) -> String {
    if pointer.is_null() {
        return String::new();
    }
    // CUPTI guarantees activity strings remain valid until this callback returns.
    unsafe { CStr::from_ptr(pointer) }
        .to_string_lossy()
        .into_owned()
}

unsafe fn consume(buffer: *mut u8, valid: usize) -> Result<(), String> {
    if valid == 0 {
        return Ok(());
    }
    if buffer.is_null() || valid > BUFFER_BYTES {
        return Err("invalid CUPTI buffer length".into());
    }
    let mut record = std::ptr::null_mut();
    loop {
        match unsafe { activity::get_next_record(buffer, valid, &mut record) } {
            Ok(()) => {}
            Err(cudarc::cupti::result::CuptiError(
                sys::CUptiResult::CUPTI_ERROR_MAX_LIMIT_REACHED,
            )) => return Ok(()),
            Err(error) => return Err(error.to_string()),
        }
        if record.is_null() {
            return Err("null activity record".into());
        }
        // Runtime API version is checked before enable. Cast only the documented
        // activity kind to its CUDA 13.3 record; copy values before buffer release.
        match unsafe { (*record).kind } {
            CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL => {
                let k = unsafe { &*record.cast::<sys::CUpti_ActivityKernel12>() };
                if [k.gridX, k.gridY, k.gridZ, k.blockX, k.blockY, k.blockZ]
                    .iter()
                    .any(|v| *v < 0)
                {
                    return Err("negative launch dimension".into());
                }
                emit(Event::Kernel {
                    name: unsafe { string(k.name) },
                    start: k.start,
                    end: k.end,
                    device: k.deviceId,
                    context: k.contextId,
                    stream: k.streamId,
                    correlation: k.correlationId,
                    grid: [k.gridX as u32, k.gridY as u32, k.gridZ as u32],
                    block: [k.blockX as u32, k.blockY as u32, k.blockZ as u32],
                    registers: k.registersPerThread as u32,
                    shared_bytes: k.staticSharedMemory.max(0) as u64
                        + k.dynamicSharedMemory.max(0) as u64,
                    graph_id: k.graphId,
                });
            }
            CUPTI_ACTIVITY_KIND_MARKER => {
                let m = unsafe { &*record.cast::<sys::CUpti_ActivityMarker2>() };
                if !matches!(
                    m.objectKind,
                    sys::CUpti_ActivityObjectKind::CUPTI_ACTIVITY_OBJECT_THREAD
                        | sys::CUpti_ActivityObjectKind::CUPTI_ACTIVITY_OBJECT_PROCESS
                ) {
                    continue;
                }
                let id = unsafe { m.objectId.pt };
                emit(Event::Marker {
                    name: unsafe { string(m.name) },
                    domain: unsafe { string(m.domain) },
                    timestamp: m.timestamp,
                    id: m.id,
                    flags: m.flags as u32,
                    pid: id.processId,
                    tid: id.threadId,
                });
            }
            CUPTI_ACTIVITY_KIND_MEMCPY => {
                let m = unsafe { &*record.cast::<sys::CUpti_ActivityMemcpy6>() };
                emit(Event::Memop {
                    operation: "memcpy".into(),
                    start: m.start,
                    end: m.end,
                    bytes: m.bytes,
                    device: m.deviceId,
                    context: m.contextId,
                    stream: m.streamId,
                });
            }
            CUPTI_ACTIVITY_KIND_MEMSET => {
                let m = unsafe { &*record.cast::<sys::CUpti_ActivityMemset4>() };
                emit(Event::Memop {
                    operation: "memset".into(),
                    start: m.start,
                    end: m.end,
                    bytes: m.bytes,
                    device: m.deviceId,
                    context: m.contextId,
                    stream: m.streamId,
                });
            }
            _ => {}
        }
    }
}

extern "C" fn finish() {
    let _ = std::panic::catch_unwind(|| {
        if let Err(e) =
            activity::flush_all(sys::CUpti_ActivityFlag::CUPTI_ACTIVITY_FLAG_FLUSH_FORCED as u32)
        {
            error(e.to_string());
        }
        emit(Event::End {
            dropped: DROPPED.load(Ordering::Relaxed),
            errors: ERRORS.load(Ordering::Relaxed),
        });
        if let Some(output) = OUTPUT.get() {
            if let Ok(mut output) = output.lock() {
                let _ = output.flush();
                let _ = output.get_ref().sync_all();
            }
        }
    });
}

fn initialize() -> Result<(), String> {
    let path =
        std::env::var_os("DS4_PERF_CUPTI_OUTPUT").ok_or("DS4_PERF_CUPTI_OUTPUT is required")?;
    let file = OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(path)
        .map_err(|e| e.to_string())?;
    OUTPUT
        .set(Mutex::new(BufWriter::new(file)))
        .map_err(|_| "collector already initialized")?;
    let mut version = 0;
    // cudarc supplies the binding; the output pointer is a live u32 local.
    unsafe { sys::cuptiGetVersion(&mut version) }
        .result()
        .map_err(|e| e.to_string())?;
    emit(Event::Start {
        schema_version: 1,
        api_version: version,
        pid: std::process::id(),
    });
    if version != sys::CUPTI_API_VERSION {
        return Err(format!(
            "CUPTI record ABI mismatch: runtime {version}, bindings {}",
            sys::CUPTI_API_VERSION
        ));
    }
    activity::register_callbacks(Some(request_buffer), Some(complete_buffer))
        .map_err(|e| e.to_string())?;
    for kind in [
        CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL,
        CUPTI_ACTIVITY_KIND_MEMCPY,
        CUPTI_ACTIVITY_KIND_MEMSET,
        CUPTI_ACTIVITY_KIND_MARKER,
    ] {
        activity::enable(kind).map_err(|e| e.to_string())?;
    }
    // Linux atexit is the documented CUPTI injection sample flush point.
    if unsafe { libc::atexit(finish) } != 0 {
        return Err("cannot register CUPTI exit flush".into());
    }
    Ok(())
}

#[unsafe(no_mangle)]
pub extern "C" fn InitializeInjection() -> i32 {
    let result = std::panic::catch_unwind(|| INITIALIZED.get_or_init(initialize));
    match result {
        Ok(Ok(())) => 1,
        Ok(Err(message)) => {
            error(message.clone());
            flush_output();
            0
        }
        Err(_) => {
            error("CUPTI initialization panicked".into());
            flush_output();
            0
        }
    }
}

fn flush_output() {
    if let Some(output) = OUTPUT.get() {
        if let Ok(mut output) = output.lock() {
            if output.flush().is_err() || output.get_ref().sync_all().is_err() {
                eprintln!("ds4-perf CUPTI: could not persist collector output");
            }
        }
    }
}
