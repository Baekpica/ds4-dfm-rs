use super::{BashJob, BashTable};
use crate::agent::web_tools::io_detail;
use std::ffi::CString;
use std::fs::File;
use std::os::fd::FromRawFd;
use std::os::unix::process::CommandExt;
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::sync::mpsc;
use std::thread;
use std::time::Instant;

unsafe extern "C" {
    fn mkstemp(template: *mut std::ffi::c_char) -> i32;
}

pub(super) fn spawn_job(
    table: &mut BashTable,
    command: &str,
    timeout: i32,
) -> Result<usize, String> {
    let (path, file) = create_output_file()?;
    let stdout = file.try_clone().map_err(|error| {
        format!(
            "failed to create temporary output file: {}",
            io_detail(&error)
        )
    })?;
    let stderr = file.try_clone().map_err(|error| {
        format!(
            "failed to create temporary output file: {}",
            io_detail(&error)
        )
    })?;
    drop(file);
    let mut command_process = Command::new("/bin/sh");
    command_process
        .arg("-c")
        .arg(command)
        .stdin(Stdio::inherit())
        .stdout(Stdio::from(stdout))
        .stderr(Stdio::from(stderr))
        .process_group(0);
    let (ready_tx, ready) = mpsc::sync_channel(1);
    let (control, control_rx) = mpsc::channel();
    let (done_tx, done) = mpsc::channel();
    let watcher = thread::Builder::new()
        .name("ds4-bash-watch".into())
        .spawn(move || match command_process.spawn() {
            Ok(child) => {
                let _ = ready_tx.send(Ok(child.id() as i32));
                let _ = done_tx.send(super::watch::run(child, timeout, control_rx));
            }
            Err(error) => {
                let _ = ready_tx.send(Err(format!("failed to fork: {}", io_detail(&error))));
            }
        })
        .map_err(|error| {
            let _ = std::fs::remove_file(&path);
            format!("failed to start job watcher: {}", io_detail(&error))
        })?;
    let pid = match ready
        .recv()
        .unwrap_or_else(|_| Err("job watcher stopped during startup".into()))
    {
        Ok(pid) => pid,
        Err(error) => {
            let _ = watcher.join();
            let _ = std::fs::remove_file(&path);
            return Err(error);
        }
    };
    // Retire the thread's stack even when no later tool call observes this job.
    drop(watcher);
    if table.next_id <= 0 {
        table.next_id = 1;
    }
    let id = table.next_id;
    table.next_id += 1;
    table.jobs.insert(
        0,
        BashJob {
            id,
            pid,
            path,
            control,
            done,
            start: Instant::now(),
            timeout_sec: f64::from(timeout),
            observed_once: false,
            exit_status: -1,
            running: true,
            timed_out: false,
        },
    );
    Ok(0)
}

fn create_output_file() -> Result<(PathBuf, File), String> {
    let mut tmpl = CString::new("/tmp/ds4_agent_output_XXXXXX")
        .map_err(|_| "failed to create temporary output file: invalid template".to_string())?
        .into_bytes_with_nul();
    // SAFETY: [Category 8 — FFI] `tmpl` is an exclusive CString buffer ending
    // in `XXXXXX\0`; `mkstemp` rewrites those six bytes in place and does not
    // retain the pointer.
    let fd = unsafe { mkstemp(tmpl.as_mut_ptr().cast()) };
    if fd < 0 {
        return Err(format!(
            "failed to create temporary output file: {}",
            io_detail(&std::io::Error::last_os_error())
        ));
    }
    let path = CString::from_vec_with_nul(tmpl)
        .map_err(|_| "failed to create temporary output file: invalid path".to_string())?
        .into_string()
        .map_err(|_| "failed to create temporary output file: invalid path".to_string())?;
    // SAFETY: [Category 13 — Library / unsafe contract] `fd` is a live exclusive
    // descriptor from `mkstemp` (fd >= 0 checked). `File` takes ownership; no
    // other close path exists.
    Ok((PathBuf::from(path), unsafe { File::from_raw_fd(fd) }))
}
