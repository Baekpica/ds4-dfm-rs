use super::{SIGKILL, SIGTERM};
use std::io;
use std::os::unix::process::ExitStatusExt;
use std::process::Child;
use std::sync::mpsc::{Receiver, RecvTimeoutError};
use std::time::{Duration, Instant};

pub(super) struct JobExit {
    pub status: i32,
    pub timed_out: bool,
}

fn signal_group(pid: u32, signal: i32) {
    // SAFETY: [Category 8 — FFI] the watcher owns this unreaped group leader.
    unsafe {
        libc::kill(-(pid as i32), signal);
    }
}

fn exited(pid: u32) -> io::Result<bool> {
    // SAFETY: [Category 8 — FFI] waitid initializes siginfo; WNOWAIT keeps the
    // leader's PID reserved until the remaining group is killed and Child reaped.
    unsafe {
        let mut info: libc::siginfo_t = std::mem::zeroed();
        if libc::waitid(
            libc::P_PID,
            pid,
            &mut info,
            libc::WEXITED | libc::WNOHANG | libc::WNOWAIT,
        ) != 0
        {
            return Err(io::Error::last_os_error());
        }
        Ok(info.si_pid() != 0)
    }
}

pub(super) fn run(mut child: Child, timeout: i32, control: Receiver<i32>) -> JobExit {
    let deadline = Instant::now() + Duration::from_secs(timeout as u64);
    let mut terminate_at = None;
    let mut timed_out = false;
    loop {
        match exited(child.id()) {
            Ok(true) => break,
            Ok(false) => {}
            Err(error) if error.kind() == io::ErrorKind::Interrupted => continue,
            Err(error) if error.raw_os_error() == Some(libc::ECHILD) => {
                return JobExit {
                    status: -1,
                    timed_out,
                };
            }
            Err(_) => break,
        }
        let now = Instant::now();
        if now >= deadline {
            timed_out = true;
            break;
        }
        if terminate_at.is_some_and(|at| now >= at) {
            break;
        }
        let wait = deadline
            .saturating_duration_since(now)
            .min(Duration::from_millis(20));
        match control.recv_timeout(wait) {
            Ok(SIGTERM) => {
                signal_group(child.id(), SIGTERM);
                terminate_at.get_or_insert(now + Duration::from_secs(1));
            }
            Ok(_) | Err(RecvTimeoutError::Disconnected) => break,
            Err(RecvTimeoutError::Timeout) => {}
        }
    }
    signal_group(child.id(), SIGKILL);
    let status = child.wait().ok().map_or(-1, |status| {
        status
            .code()
            .unwrap_or_else(|| status.signal().map_or(-1, |sig| 128 + sig))
    });
    JobExit { status, timed_out }
}
