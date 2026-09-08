use serde::{Deserialize, Serialize};

/// Owned activity records: no CUPTI pointers survive a completed buffer callback.
#[derive(Debug, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum Event {
    Start {
        schema_version: u32,
        api_version: u32,
        pid: u32,
    },
    End {
        dropped: u64,
        errors: u64,
    },
    Error {
        message: String,
    },
    Kernel {
        name: String,
        start: u64,
        end: u64,
        device: u32,
        context: u32,
        stream: u32,
        correlation: u32,
        grid: [u32; 3],
        block: [u32; 3],
        registers: u32,
        shared_bytes: u64,
        graph_id: u32,
    },
    Marker {
        name: String,
        domain: String,
        timestamp: u64,
        id: u32,
        flags: u32,
        pid: u32,
        tid: u32,
    },
    Memop {
        operation: String,
        start: u64,
        end: u64,
        bytes: u64,
        device: u32,
        context: u32,
        stream: u32,
    },
}
