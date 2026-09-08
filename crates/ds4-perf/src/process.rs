use std::{
    fs,
    path::Path,
    process::{Child, Command, ExitStatus},
    time::{Duration, Instant},
};

static INTERRUPTED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

pub fn signals() -> Result<(), String> {
    ctrlc::set_handler(|| INTERRUPTED.store(true, std::sync::atomic::Ordering::Relaxed))
        .map_err(|e| e.to_string())
}

#[derive(Clone, Copy, Debug)]
pub struct Limits {
    pub timeout: Duration,
    pub bytes: u64,
}

impl Default for Limits {
    fn default() -> Self {
        Self {
            timeout: Duration::from_secs(1800),
            bytes: 2 * 1024 * 1024 * 1024,
        }
    }
}

pub fn bytes(path: &Path) -> Result<u64, String> {
    let mut total = 0u64;
    for entry in fs::read_dir(path).map_err(|e| e.to_string())? {
        let entry = entry.map_err(|e| e.to_string())?;
        let metadata = entry.metadata().map_err(|e| e.to_string())?;
        let size = if entry.file_type().map_err(|e| e.to_string())?.is_dir() {
            bytes(&entry.path())?
        } else {
            metadata.len()
        };
        total = total.checked_add(size).ok_or("output size overflow")?;
    }
    Ok(total)
}

fn stop(child: &mut Child) {
    #[cfg(unix)]
    {
        use nix::{
            sys::signal::{killpg, Signal},
            unistd::Pid,
        };
        let group = Pid::from_raw(child.id() as i32);
        let _ = killpg(group, Signal::SIGTERM);
        std::thread::sleep(Duration::from_millis(200));
        let _ = killpg(group, Signal::SIGKILL);
    }
    #[cfg(not(unix))]
    {
        let _ = child.kill();
    }
    let end = Instant::now() + Duration::from_secs(2);
    while Instant::now() < end {
        if child.try_wait().is_ok_and(|s| s.is_some()) {
            break;
        }
        std::thread::sleep(Duration::from_millis(25));
    }
}

pub fn run(command: &mut Command, out: &Path, limits: Limits) -> Result<ExitStatus, String> {
    if INTERRUPTED.load(std::sync::atomic::Ordering::Relaxed) {
        return Err("interrupted".into());
    }
    #[cfg(unix)]
    {
        use std::os::unix::process::CommandExt;
        command.process_group(0);
    }
    let mut child = command.spawn().map_err(|e| e.to_string())?;
    let start = Instant::now();
    let result = (|| loop {
        if INTERRUPTED.load(std::sync::atomic::Ordering::Relaxed) {
            return Err("interrupted".into());
        }
        if bytes(out)? > limits.bytes {
            return Err("profiler output budget exceeded".into());
        }
        if let Some(status) = child.try_wait().map_err(|e| e.to_string())? {
            #[cfg(unix)]
            if nix::sys::signal::killpg(nix::unistd::Pid::from_raw(child.id() as i32), None).is_ok()
            {
                return Err("process exited with live descendants in its group".into());
            }
            return Ok(status);
        }
        if start.elapsed() >= limits.timeout {
            return Err(format!(
                "process {} exceeded {} second deadline",
                child.id(),
                limits.timeout.as_secs()
            ));
        }
        std::thread::sleep(Duration::from_millis(25));
    })();
    if result.is_err() {
        stop(&mut child);
    }
    result
}

#[cfg(all(test, unix))]
mod tests {
    use super::*;
    #[test]
    fn rejects_orphaned_descendants() {
        let root = std::env::temp_dir().join(format!("ds4-orphan-{}", std::process::id()));
        fs::create_dir(&root).unwrap();
        let mut command = Command::new("sh");
        command.args(["-c", "sleep 2 &"]);
        let result = run(&mut command, &root, Limits::default());
        fs::remove_dir_all(root).unwrap();
        assert!(result.is_err_and(|e| e.contains("descendants")));
    }

    #[test]
    fn terminates_hanging_group() {
        let root = std::env::temp_dir().join(format!("ds4-deadline-{}", std::process::id()));
        fs::create_dir(&root).unwrap();
        let mut command = Command::new("sh");
        command.args(["-c", "sleep 30 & wait"]);
        let start = Instant::now();
        let error = run(
            &mut command,
            &root,
            Limits {
                timeout: Duration::from_millis(100),
                bytes: 1024,
            },
        )
        .unwrap_err();
        assert!(error.contains("deadline"));
        assert!(start.elapsed() < Duration::from_secs(5));
        fs::remove_dir_all(root).unwrap();
    }
}

/// Include generated reports and copies in the final output accounting.
pub fn check_bytes(out: &Path, limits: Limits) -> Result<(), String> {
    if bytes(out)? > limits.bytes {
        return Err("profiler output budget exceeded".into());
    }
    Ok(())
}
