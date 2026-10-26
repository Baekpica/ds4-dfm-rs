use super::BashJob;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};

const HEAD_BYTES: usize = 8 * 1024;
const HEAD_LINES: i32 = 100;
const TAIL_BYTES: usize = 32 * 1024;
const PROGRESS_TAIL_LINES: i32 = 4;
const FINAL_TAIL_LINES: i32 = 20;

pub(super) fn observation(job: &mut BashJob) -> Vec<u8> {
    job.poll();
    let first = !job.observed_once;
    let (bytes, display_lines, last_nl) = file_stats(&job.path);
    let elapsed = job.start.elapsed().as_secs_f64();
    let mut out = if job.running {
        format!(
            "bash job={} pid={} status=running elapsed_sec={elapsed:.1} timeout_sec={:.0}\n",
            job.id, job.pid, job.timeout_sec
        )
    } else {
        format!(
            "bash job={} pid={} status=done elapsed_sec={elapsed:.1} timed_out={}\nexit_status={}\n",
            job.id,
            job.pid,
            i32::from(job.timed_out),
            job.exit_status
        )
    }
    .into_bytes();

    if bytes == 0 {
        out.extend_from_slice(b"<output>\n</output>\n");
    } else if first {
        let (head, shown, byte_limited) = read_head(&job.path);
        let truncated = byte_limited || display_lines > shown;
        if !job.running && !truncated {
            out.extend_from_slice(b"<output>\n");
            out.extend_from_slice(&head);
            if !head.is_empty() && !head.ends_with(b"\n") {
                out.push(b'\n');
            }
            out.extend_from_slice(b"</output>\n");
        } else {
            append_path_head(&mut out, &job.path, bytes, display_lines, &head);
        }
    } else {
        let tail_lines = if job.running {
            PROGRESS_TAIL_LINES
        } else {
            FINAL_TAIL_LINES
        };
        let tail = read_tail(&job.path, tail_lines);
        let path = job.path.display();
        out.extend_from_slice(
            format!("output_path={path} ({bytes} bytes, {display_lines} lines)\n").as_bytes(),
        );
        out.extend_from_slice(format!("<tail -{tail_lines} {path}>\n").as_bytes());
        out.extend_from_slice(&tail);
        if !tail.is_empty() && !tail.ends_with(b"\n") {
            out.push(b'\n');
        }
        out.extend_from_slice(b"</tail>\n");
    }
    if job.running {
        out.extend_from_slice(
            format!(
                "\nUse bash_status job={} to get info before refresh time; use bash_stop job={} to stop execution\n",
                job.id, job.id
            )
            .as_bytes(),
        );
    }
    job.observed_once = true;
    let _ = (last_nl,);
    out
}

fn file_stats(path: &std::path::Path) -> (usize, i32, bool) {
    let Ok(mut file) = File::open(path) else {
        return (0, 0, true);
    };
    let mut buf = [0u8; 8192];
    let mut bytes = 0;
    let mut newlines = 0i32;
    let mut last_nl = true;
    loop {
        match file.read(&mut buf) {
            Ok(0) => break,
            Ok(n) => {
                bytes += n;
                newlines = newlines
                    .saturating_add(buf[..n].iter().filter(|&&b| b == b'\n').count() as i32);
                last_nl = buf[n - 1] == b'\n';
            }
            Err(e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
            Err(_) => return (0, 0, true),
        }
    }
    (bytes, newlines.saturating_add(i32::from(!last_nl)), last_nl)
}

fn read_head(path: &std::path::Path) -> (Vec<u8>, i32, bool) {
    let Ok(mut file) = File::open(path) else {
        return (b"<failed to reopen output file>\n".to_vec(), 0, false);
    };
    let mut out = Vec::new();
    let mut buf = [0u8; 1];
    let mut lines = 0i32;
    while lines < HEAD_LINES && out.len() < HEAD_BYTES {
        match file.read(&mut buf) {
            Ok(0) => break,
            Ok(_) => {
                out.push(buf[0]);
                if buf[0] == b'\n' {
                    lines += 1;
                }
            }
            Err(_) => break,
        }
    }
    let byte_limited = out.len() >= HEAD_BYTES;
    let shown = lines + i32::from(!out.is_empty() && !out.ends_with(b"\n"));
    (out, shown, byte_limited)
}

fn read_tail(path: &std::path::Path, max_lines: i32) -> Vec<u8> {
    let read_window = || -> std::io::Result<Vec<u8>> {
        let mut file = File::open(path)?;
        let end = file.seek(SeekFrom::End(0))?;
        file.seek(SeekFrom::Start(end.saturating_sub(TAIL_BYTES as u64)))?;
        let mut data = Vec::new();
        file.take(TAIL_BYTES as u64).read_to_end(&mut data)?;
        Ok(data)
    };
    let Ok(data) = read_window() else {
        return b"<failed to reopen output file>\n".to_vec();
    };
    if data.is_empty() {
        return Vec::new();
    }
    let mut newlines = 0;
    let mut cut = 0;
    for (index, byte) in data.iter().enumerate().rev() {
        if *byte == b'\n' {
            newlines += 1;
            if newlines > max_lines {
                cut = index + 1;
                break;
            }
        }
    }
    data[cut..].to_vec()
}

fn append_path_head(
    out: &mut Vec<u8>,
    path: &std::path::Path,
    bytes: usize,
    lines: i32,
    head: &[u8],
) {
    let shown = path.display();
    out.extend_from_slice(
        format!("output_path={shown} ({bytes} bytes, {lines} lines)\n").as_bytes(),
    );
    out.extend_from_slice(format!("<head -{HEAD_LINES} {shown}>\n").as_bytes());
    out.extend_from_slice(head);
    if !head.is_empty() && !head.ends_with(b"\n") {
        out.push(b'\n');
    }
    out.extend_from_slice(b"</head>\n");
}

#[cfg(all(test, target_os = "linux"))]
mod tests {
    use super::*;
    use std::io::{Seek, SeekFrom, Write};

    #[test]
    fn audit_bash_large_log_bounded_memory() {
        if std::env::var_os("DS4_AUDIT_BASH_LOG_CHILD").is_none() {
            let status = std::process::Command::new(std::env::current_exe().unwrap())
                .args([
                    "--exact",
                    "agent::bash::obs::tests::audit_bash_large_log_bounded_memory",
                    "--test-threads=1",
                ])
                .env("DS4_AUDIT_BASH_LOG_CHILD", "1")
                .status()
                .unwrap();
            assert!(status.success());
            return;
        }
        let path = std::env::temp_dir().join(format!("ds4-audit-log-{}", std::process::id()));
        let size = 128 * 1024 * 1024;
        let mut file = File::create(&path).unwrap();
        file.set_len(size).unwrap();
        file.seek(SeekFrom::End(-12)).unwrap();
        file.write_all(b"tail1\ntail2\n").unwrap();
        drop(file);
        let pages: u64 = std::fs::read_to_string("/proc/self/statm")
            .unwrap()
            .split_whitespace()
            .next()
            .unwrap()
            .parse()
            .unwrap();
        // SAFETY: limit changes only this subprocess, after the sparse file exists.
        unsafe {
            let mut limit: libc::rlimit = std::mem::zeroed();
            assert_eq!(libc::getrlimit(libc::RLIMIT_AS, &mut limit), 0);
            limit.rlim_cur = pages * libc::sysconf(libc::_SC_PAGESIZE) as u64 + 32 * 1024 * 1024;
            assert_eq!(libc::setrlimit(libc::RLIMIT_AS, &limit), 0);
        }
        assert_eq!(file_stats(&path), (size as usize, 2, true));
        assert_eq!(read_tail(&path, 1), b"tail2\n");
        std::fs::remove_file(path).unwrap();
    }
}
