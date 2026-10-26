mod obs;
mod spawn;
mod watch;

use super::search::scan::parse_int_default as parse_int;
use obs::observation;
use spawn::spawn_job;
use std::ffi::CString;
use std::path::PathBuf;
use std::sync::mpsc::{Receiver, Sender, TryRecvError};
use std::time::Instant;

pub(crate) const BASH: &str = "bash";
pub(crate) const BASH_STATUS: &str = "bash_status";
pub(crate) const BASH_STOP: &str = "bash_stop";

const DEFAULT_TIMEOUT: i32 = 3600;
const MAX_TIMEOUT: f64 = 24.0 * 3600.0;
const DEFAULT_REFRESH: i32 = 60;
const MAX_REFRESH: i32 = 3600;

unsafe extern "C" {
    fn strtod(nptr: *const std::ffi::c_char, endptr: *mut *mut std::ffi::c_char) -> f64;
}

const SIGTERM: i32 = 15;
const SIGKILL: i32 = 9;

pub(crate) struct BashTable {
    pub(super) next_id: i32,
    pub(super) jobs: Vec<BashJob>,
}

impl Default for BashTable {
    fn default() -> Self {
        Self {
            next_id: 1,
            jobs: Vec::new(),
        }
    }
}

pub(crate) struct BashJob {
    pub(super) id: i32,
    pub(super) pid: i32,
    pub(super) path: PathBuf,
    control: Sender<i32>,
    done: Receiver<watch::JobExit>,
    pub(super) start: Instant,
    pub(super) timeout_sec: f64,
    pub(super) observed_once: bool,
    pub(super) exit_status: i32,
    pub(super) running: bool,
    pub(super) timed_out: bool,
}

impl Drop for BashJob {
    fn drop(&mut self) {
        self.stop(SIGKILL);
    }
}

impl BashJob {
    fn stop(&mut self, signal: i32) {
        if !self.running {
            return;
        }
        let _ = self.control.send(signal);
        if let Ok(done) = self.done.recv() {
            self.exit_status = done.status;
            self.timed_out = done.timed_out;
        }
        self.running = false;
    }

    fn poll(&mut self) {
        if !self.running {
            return;
        }
        match self.done.try_recv() {
            Ok(done) => {
                self.exit_status = done.status;
                self.timed_out = done.timed_out;
                self.running = false;
            }
            Err(TryRecvError::Disconnected) => self.running = false,
            Err(TryRecvError::Empty) => {}
        }
    }
}

pub(crate) fn parse_timeout(value: Option<&str>) -> i32 {
    let Some(value) = value.filter(|value| !value.is_empty()) else {
        return DEFAULT_TIMEOUT;
    };
    let Ok(c_value) = CString::new(value) else {
        return DEFAULT_TIMEOUT;
    };
    let mut end = std::ptr::null_mut();
    // SAFETY: [Category 8 — FFI] `c_value` is a live CString; `end` is a
    // stack `*mut c_char` written by libc. `strtod` does not retain either
    // pointer. Matches C `strtod` timeout parse.
    let parsed = unsafe { strtod(c_value.as_ptr(), &mut end) };
    if end == c_value.as_ptr().cast_mut() || !parsed.is_finite() || parsed <= 0.0 {
        return DEFAULT_TIMEOUT;
    }
    parsed.clamp(1.0, MAX_TIMEOUT) as i32
}

pub(crate) fn bash_result(table: &mut BashTable, name: &str, call: &BashArgs<'_>) -> Vec<u8> {
    match name {
        BASH => start_bash(table, call),
        BASH_STATUS | BASH_STOP => follow_bash(table, name, call),
        _ => b"Tool error: unknown tool\n".to_vec(),
    }
}

pub(crate) struct BashArgs<'a> {
    pub command: Option<&'a str>,
    pub timeout_sec: Option<&'a str>,
    pub refresh_sec: Option<&'a str>,
    pub job: Option<&'a str>,
    pub pid: Option<&'a str>,
}

fn start_bash(table: &mut BashTable, call: &BashArgs<'_>) -> Vec<u8> {
    let Some(command) = call.command.filter(|command| !command.is_empty()) else {
        return b"Tool error: bash requires command\n".to_vec();
    };
    let timeout = parse_timeout(call.timeout_sec);
    let refresh = parse_int(call.refresh_sec, DEFAULT_REFRESH, 1, MAX_REFRESH);
    match spawn_job(table, command, timeout) {
        Ok(index) => finish_job(table, index, true, refresh, false),
        Err(error) => format!("Tool error: bash failed to start: {error}\n").into_bytes(),
    }
}

fn follow_bash(table: &mut BashTable, name: &str, call: &BashArgs<'_>) -> Vec<u8> {
    let job_id = parse_int(call.job, 0, 0, i32::MAX);
    let pid = parse_int(call.pid, 0, 0, i32::MAX);
    let refresh = parse_int(call.refresh_sec, DEFAULT_REFRESH, 1, MAX_REFRESH);
    let Some(index) = find_job(table, job_id, pid) else {
        return format!("Tool error: bash job not found: job={job_id} pid={pid}\n").into_bytes();
    };
    finish_job(table, index, name == BASH_STOP, refresh, name == BASH_STOP)
}

fn find_job(table: &BashTable, id: i32, pid: i32) -> Option<usize> {
    table
        .jobs
        .iter()
        .position(|job| (id > 0 && job.id == id) || (id <= 0 && pid > 0 && job.pid == pid))
}

fn finish_job(
    table: &mut BashTable,
    index: usize,
    wait: bool,
    refresh: i32,
    stop: bool,
) -> Vec<u8> {
    if stop {
        table.jobs[index].stop(SIGTERM);
    }
    if wait || stop {
        refresh_for(&mut table.jobs[index], refresh);
    } else {
        table.jobs[index].poll();
    }
    let out = observation(&mut table.jobs[index]);
    if !table.jobs[index].running {
        table.jobs.remove(index);
    }
    out
}

fn refresh_for(job: &mut BashJob, refresh: i32) {
    let start = Instant::now();
    let limit = f64::from(refresh);
    while job.running && start.elapsed().as_secs_f64() < limit {
        job.poll();
        if !job.running {
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(20));
    }
    job.poll();
}

#[cfg(all(test, target_os = "linux"))]
mod tests {
    use super::*;
    use std::time::Duration;

    fn assert_descendant_stopped(pid: i32) {
        let deadline = Instant::now() + Duration::from_secs(1);
        loop {
            let state = std::fs::read_to_string(format!("/proc/{pid}/stat")).unwrap_or_default();
            if state.is_empty() || state.split_whitespace().nth(2) == Some("Z") {
                return;
            }
            assert!(
                Instant::now() < deadline,
                "descendant still running: {state}"
            );
            std::thread::sleep(Duration::from_millis(10));
        }
    }

    #[test]
    fn audit_bash_timeout_without_observation() {
        let mut table = BashTable::default();
        spawn_job(&mut table, "sleep 30 & echo $!; wait", 1).unwrap();
        let pid = table.jobs[0].pid;
        let path = table.jobs[0].path.clone();
        let deadline = Instant::now() + Duration::from_secs(3);
        while std::path::Path::new(&format!("/proc/{pid}")).exists() && Instant::now() < deadline {
            std::thread::sleep(Duration::from_millis(20));
        }
        assert!(
            !std::path::Path::new(&format!("/proc/{pid}")).exists(),
            "deadline must reap without a status query"
        );
        let child: i32 = std::fs::read_to_string(&path)
            .unwrap()
            .trim()
            .parse()
            .unwrap();
        assert_descendant_stopped(child);
        table.jobs[0].poll();
        assert!(table.jobs[0].timed_out);
        assert!(!table.jobs[0].running);
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn bash_normal_stop_and_drop_reap_the_owned_group() {
        let threads = || std::fs::read_dir("/proc/self/task").unwrap().count();
        let baseline = threads();
        for action in ["exit", "stop", "drop"] {
            let mut table = BashTable::default();
            let command = if action == "exit" {
                "sleep 30 & echo $!; exit 7"
            } else {
                "trap '' TERM; sleep 30 & echo $!; wait"
            };
            spawn_job(&mut table, command, 30).unwrap();
            let pid = table.jobs[0].pid;
            let path = table.jobs[0].path.clone();
            let deadline = Instant::now() + Duration::from_secs(3);
            while std::fs::metadata(&path).unwrap().len() == 0 && Instant::now() < deadline {
                std::thread::sleep(Duration::from_millis(20));
            }
            if action == "stop" {
                table.jobs[0].stop(SIGTERM);
            }
            if action == "drop" {
                table.jobs.clear();
            }
            while std::path::Path::new(&format!("/proc/{pid}")).exists()
                && Instant::now() < deadline
            {
                std::thread::sleep(Duration::from_millis(20));
            }
            assert!(
                !std::path::Path::new(&format!("/proc/{pid}")).exists(),
                "{action}"
            );
            if let Some(job) = table.jobs.first_mut() {
                job.poll();
                assert!(!job.running && !job.timed_out);
                assert_eq!(job.exit_status, if action == "exit" { 7 } else { 137 });
            }
            let child: i32 = std::fs::read_to_string(&path)
                .unwrap()
                .trim()
                .parse()
                .unwrap();
            assert_descendant_stopped(child);
            drop(table);
            while threads() > baseline && Instant::now() < deadline {
                std::thread::sleep(Duration::from_millis(20));
            }
            assert_eq!(threads(), baseline, "watcher retired on {action}");
            std::fs::remove_file(path).unwrap();
        }
    }
}
