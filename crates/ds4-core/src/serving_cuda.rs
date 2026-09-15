//! Context-free CUDA device identity and memory topology for pre-open quotes.

pub(super) struct Device {
    pub(super) uuid: String,
    pub(super) integrated: bool,
}

#[cfg(target_os = "linux")]
pub(super) fn device() -> Option<Device> {
    use libloading::Library;
    use std::ffi::c_int;
    use std::sync::OnceLock;
    type Init = unsafe extern "C" fn(u32) -> c_int;
    type Get = unsafe extern "C" fn(*mut c_int, c_int) -> c_int;
    type Uuid = unsafe extern "C" fn(*mut [u8; 16], c_int) -> c_int;
    type Attribute = unsafe extern "C" fn(*mut c_int, c_int, c_int) -> c_int;
    static DRIVER: OnceLock<Option<Library>> = OnceLock::new();
    // SAFETY: load the system CUDA driver using its documented C ABI. Keep
    // it loaded for the process lifetime; the native engine shares the driver.
    let lib = DRIVER
        .get_or_init(|| unsafe { Library::new("libcuda.so.1").ok() })
        .as_ref()?;
    // SAFETY: signatures match cuda.h. All output pointers address initialized
    // storage of the documented size. Device queries create no CUDA context.
    let (uuid, integrated) = unsafe {
        let init = lib.get::<Init>(b"cuInit\0").ok()?;
        let get = lib.get::<Get>(b"cuDeviceGet\0").ok()?;
        let uuid_fn = lib.get::<Uuid>(b"cuDeviceGetUuid_v2\0").ok()?;
        let attr = lib.get::<Attribute>(b"cuDeviceGetAttribute\0").ok()?;
        let mut dev = 0;
        let mut uuid = [0u8; 16];
        let mut integrated = 0;
        // CU_DEVICE_ATTRIBUTE_INTEGRATED = 18. CUDA applies its enumeration
        // order and CUDA_VISIBLE_DEVICES before resolving ordinal zero.
        if init(0) != 0
            || get(&mut dev, 0) != 0
            || uuid_fn(&mut uuid, dev) != 0
            || attr(&mut integrated, 18, dev) != 0
        {
            return None;
        }
        (uuid, integrated != 0)
    };
    let mut id = String::from("GPU-");
    use std::fmt::Write;
    for (i, byte) in uuid.iter().enumerate() {
        if matches!(i, 4 | 6 | 8 | 10) {
            id.push('-');
        }
        write!(id, "{byte:02x}").ok()?;
    }
    Some(Device {
        uuid: id,
        integrated,
    })
}

#[cfg(not(target_os = "linux"))]
pub(super) fn device() -> Option<Device> {
    None
}
