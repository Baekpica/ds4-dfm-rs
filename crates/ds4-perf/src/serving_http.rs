//! Curl owns HTTP framing. The observer timestamps complete SSE data events;
//! role-only events cannot masquerade as the first generated output.
use crate::{process, runner};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::{
    fs::{self, File},
    io::{BufRead, BufReader},
    path::{Path, PathBuf},
    process::{Command, Stdio},
    sync::{
        atomic::{AtomicBool, Ordering},
        mpsc,
    },
    time::{Duration, Instant},
};

const POLL_MS: u64 = 1;

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Event {
    pub elapsed_ms: f64,
    pub data: String,
}

pub struct Http {
    curl: PathBuf,
    origin: String,
    root: PathBuf,
    limits: process::Limits,
}

impl Http {
    pub fn new(origin: &str, root: &Path, limits: process::Limits) -> Result<Self, String> {
        let authority = origin
            .strip_prefix("http://")
            .ok_or("serving measurements require a local http:// origin")?
            .trim_end_matches('/');
        let (host, port) = authority
            .rsplit_once(':')
            .ok_or("URL requires an explicit port")?;
        if !["127.0.0.1", "localhost", "[::1]"].contains(&host)
            || port.parse::<u16>().ok().filter(|n| *n != 0).is_none()
        {
            return Err("serving measurements require a loopback host and valid port".into());
        }
        let host = if host == "localhost" {
            "127.0.0.1"
        } else {
            host
        };
        let curl = runner::resolve(std::ffi::OsStr::new("curl")).ok_or("curl is required")?;
        Ok(Self {
            curl,
            origin: format!("http://{host}:{port}"),
            root: root.into(),
            limits,
        })
    }

    fn command(&self, path: &str, dir: &Path, name: &str) -> Result<Command, String> {
        let mut command = Command::new(&self.curl);
        // Disable curlrc/proxy behavior so localhost collection is reproducible.
        command
            .args([
                "-q",
                "--silent",
                "--show-error",
                "--no-buffer",
                "--noproxy",
                "*",
                "--proto",
                "=http",
                "--fail-with-body",
                "--max-time",
            ])
            .arg(self.limits.timeout.as_secs_f64().to_string())
            .arg("--dump-header")
            .arg(dir.join(format!("{name}.headers")))
            .arg(format!("{}{path}", self.origin))
            .stderr(Stdio::from(
                File::create(dir.join(format!("{name}.stderr"))).map_err(|e| e.to_string())?,
            ));
        Ok(command)
    }

    fn execute(&self, command: &mut Command, dir: &Path, name: &str) -> Result<(), String> {
        let status = process::run(command, &self.root, self.limits)?;
        if !status.success() {
            return Err(format!(
                "{name}: curl failed: {status}; see {}",
                dir.join(format!("{name}.stderr")).display()
            ));
        }
        let headers =
            fs::read_to_string(dir.join(format!("{name}.headers"))).map_err(|e| e.to_string())?;
        let status = headers.lines().rfind(|line| line.starts_with("HTTP/"));
        if status.and_then(|line| line.split_whitespace().nth(1)) != Some("200") {
            return Err(format!("{name}: expected HTTP 200"));
        }
        Ok(())
    }

    pub fn stats(&self, dir: &Path, name: &str) -> Result<Value, String> {
        let file = dir.join(format!("{name}.json"));
        let mut command = self.command("/v1/stats", dir, name)?;
        command.stdout(Stdio::from(File::create(&file).map_err(|e| e.to_string())?));
        self.execute(&mut command, dir, name)?;
        serde_json::from_slice(&fs::read(file).map_err(|e| e.to_string())?)
            .map_err(|e| e.to_string())
    }

    pub fn stream(&self, dir: &Path) -> Result<Vec<Event>, String> {
        self.stream_events(dir, Instant::now(), None)
    }

    pub fn stream_events(
        &self,
        dir: &Path,
        start: Instant,
        updates: Option<mpsc::Sender<Event>>,
    ) -> Result<Vec<Event>, String> {
        let file = dir.join("response.sse");
        let mut command = self.command("/v1/chat/completions", dir, "response")?;
        command
            .args([
                "--header",
                "Content-Type: application/json",
                "--data-binary",
            ])
            .arg(format!("@{}", dir.join("request.json").display()))
            .stdout(Stdio::from(File::create(&file).map_err(|e| e.to_string())?));
        let input = File::open(file).map_err(|e| e.to_string())?;
        let finished = AtomicBool::new(false);
        let (result, observed) = std::thread::scope(|scope| {
            let observer = scope
                .spawn(|| observe(input, start, &finished, self.limits.bytes, updates.as_ref()));
            let result = self.execute(&mut command, dir, "response");
            finished.store(true, Ordering::Release);
            (result, observer.join().map_err(|_| "SSE observer panicked"))
        });
        let events = observed??;
        fs::write(
            dir.join("events.json"),
            serde_json::to_vec_pretty(&events).map_err(|e| e.to_string())?,
        )
        .map_err(|e| e.to_string())?;
        result?;
        Ok(events)
    }
}

fn observe(
    input: File,
    start: Instant,
    finished: &AtomicBool,
    limit: u64,
    updates: Option<&mpsc::Sender<Event>>,
) -> Result<Vec<Event>, String> {
    let mut reader = BufReader::new(input);
    let mut line = Vec::new();
    let mut data = Vec::new();
    let mut events = Vec::new();
    let mut bytes = 0u64;
    loop {
        let n = reader
            .read_until(b'\n', &mut line)
            .map_err(|e| e.to_string())?;
        bytes += n as u64;
        if bytes > limit {
            return Err("SSE output budget exceeded".into());
        }
        if n == 0 {
            if finished.load(Ordering::Acquire) {
                if !line.is_empty() || !data.is_empty() {
                    return Err("truncated SSE event".into());
                }
                return Ok(events);
            }
            std::thread::sleep(Duration::from_millis(POLL_MS));
            continue;
        }
        if !line.ends_with(b"\n") {
            continue;
        }
        let complete = std::str::from_utf8(&line)
            .map_err(|e| e.to_string())?
            .trim_end_matches(['\r', '\n']);
        if complete.is_empty() && !data.is_empty() {
            let event = Event {
                elapsed_ms: start.elapsed().as_secs_f64() * 1000.0,
                data: data.join("\n"),
            };
            if let Some(updates) = updates {
                let _ = updates.send(event.clone());
            }
            events.push(event);
            data.clear();
        } else if let Some(value) = complete.strip_prefix("data:") {
            data.push(value.strip_prefix(' ').unwrap_or(value).to_owned());
        }
        line.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn fragmented_utf8_event() {
        let path = std::env::temp_dir().join(format!("ds4-sse-{}", std::process::id()));
        let mut writer = File::create(&path).unwrap();
        let input = File::open(&path).unwrap();
        let finished = AtomicBool::new(false);
        let data = "data: 안녕\r\n\r\n".as_bytes();
        writer.write_all(&data[..7]).unwrap();
        let events = std::thread::scope(|scope| {
            let observer = scope.spawn(|| observe(input, Instant::now(), &finished, 1024, None));
            std::thread::sleep(Duration::from_millis(10));
            writer.write_all(&data[7..]).unwrap();
            finished.store(true, Ordering::Release);
            observer.join().unwrap().unwrap()
        });
        fs::remove_file(path).unwrap();
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].data, "안녕");
    }

    #[test]
    fn rejects_nonlocal_origin() {
        for origin in [
            "https://localhost:8002",
            "http://example.com:8002",
            "http://127.0.0.1:0",
            "http://localhost:8002/path",
        ] {
            assert!(Http::new(origin, Path::new("."), process::Limits::default()).is_err());
        }
    }
}
