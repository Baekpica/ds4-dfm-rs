use super::web_tools::{handle_round_with_cursor, Browser, ReadCursor};
use super::write::{write_result, WRITE};
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

static TEMP_ID: AtomicU64 = AtomicU64::new(0);

struct NoWeb;

impl Browser for NoWeb {
    fn google_search(&mut self, _query: &str) -> Result<String, String> {
        Err("unused".into())
    }

    fn visit_page(&mut self, _url: &str) -> Result<String, String> {
        Err("unused".into())
    }
}

fn oracle() -> PathBuf {
    if let Ok(path) = std::env::var("DS4_AGENT_C_ORACLE") {
        return PathBuf::from(path);
    }
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/parity/agent_c_oracle")
}

fn decode_oracle_hex(stdout: &[u8]) -> Vec<u8> {
    std::str::from_utf8(stdout)
        .expect("oracle hex UTF-8")
        .trim()
        .as_bytes()
        .chunks_exact(2)
        .map(|pair| u8::from_str_radix(std::str::from_utf8(pair).unwrap(), 16).unwrap())
        .collect()
}

fn c_write(path: Option<&str>, content: Option<&str>) -> Vec<u8> {
    let oracle = oracle();
    assert!(
        oracle.exists(),
        "build C oracle: make tests/parity/agent_c_oracle"
    );
    let mut command = std::process::Command::new(oracle);
    command.arg("write");
    if let Some(path) = path {
        command.arg(path);
        if let Some(content) = content {
            command.arg(content);
        }
    }
    let output = command.output().expect("run C agent oracle");
    assert!(
        output.status.success(),
        "C agent oracle failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    decode_oracle_hex(&output.stdout)
}

fn write_call(path: Option<&str>, content: Option<&str>) -> Vec<u8> {
    let mut body = String::from("<｜DSML｜invoke name=\"write\">\n");
    if let Some(path) = path {
        body.push_str(&format!(
            "<｜DSML｜parameter name=\"path\" string=\"true\">{path}</｜DSML｜parameter>\n"
        ));
    }
    if let Some(content) = content {
        body.push_str(&format!(
            "<｜DSML｜parameter name=\"content\" string=\"true\">{content}</｜DSML｜parameter>\n"
        ));
    }
    body.push_str("</｜DSML｜invoke>\n");
    format!("<｜DSML｜tool_calls>\n{body}</｜DSML｜tool_calls>").into_bytes()
}

fn observation(result: &[u8]) -> Vec<u8> {
    let mut out = format!("Tool result 1 ({WRITE}):\n").into_bytes();
    out.extend_from_slice(result);
    if !result.is_empty() && !result.ends_with(b"\n") {
        out.push(b'\n');
    }
    out
}

fn temp_path(name: &str) -> PathBuf {
    PathBuf::from(format!(
        "/tmp/ds4_agent_write_{}_{}_{name}",
        std::process::id(),
        TEMP_ID.fetch_add(1, Ordering::Relaxed)
    ))
}

#[test]
fn write_missing_path_and_content_match_c() {
    assert_eq!(write_result(None, Some("x")), c_write(None, None));
    assert_eq!(write_result(Some(""), Some("x")), c_write(Some(""), None));
    assert_eq!(
        write_result(Some("/tmp/x"), None),
        c_write(Some("/tmp/x"), None)
    );
}

#[test]
fn write_creates_overwrites_and_allows_empty_content() {
    let path = temp_path("file.txt");
    let path_text = path.to_str().expect("UTF-8 write path");
    let mut web = NoWeb;
    let mut cursor = ReadCursor::default();

    let created = handle_round_with_cursor(
        &write_call(Some(path_text), Some("hello")),
        &mut web,
        &mut cursor,
    )
    .expect("write create");
    assert_eq!(
        created.observation,
        observation(&c_write(Some(path_text), Some("hello")))
    );
    assert_eq!(std::fs::read(&path).unwrap(), b"hello");

    let overwritten = handle_round_with_cursor(
        &write_call(Some(path_text), Some("z")),
        &mut web,
        &mut cursor,
    )
    .expect("write overwrite");
    assert_eq!(
        overwritten.observation,
        observation(&c_write(Some(path_text), Some("z")))
    );
    assert_eq!(std::fs::read(&path).unwrap(), b"z");

    let empty = handle_round_with_cursor(
        &write_call(Some(path_text), Some("")),
        &mut web,
        &mut cursor,
    )
    .expect("write empty");
    assert_eq!(
        empty.observation,
        observation(&c_write(Some(path_text), Some("")))
    );
    assert_eq!(std::fs::read(&path).unwrap(), b"");
    std::fs::remove_file(path).unwrap();
}

#[test]
fn write_open_error_matches_c_strerror() {
    let missing_parent = temp_path("no-such-dir").join("file.txt");
    let path_text = missing_parent.to_str().expect("UTF-8 write path");
    let mut web = NoWeb;
    let mut cursor = ReadCursor::default();
    let failed = handle_round_with_cursor(
        &write_call(Some(path_text), Some("hello")),
        &mut web,
        &mut cursor,
    )
    .expect("write open error");
    assert_eq!(
        failed.observation,
        observation(&c_write(Some(path_text), Some("hello")))
    );
    assert!(failed
        .observation
        .starts_with(b"Tool result 1 (write):\nTool error: open for write failed: "));
}

#[test]
fn audit_write_failure() {
    const CHILD: &str = "DS4_TEST_WRITE_LIMIT";
    if std::env::var_os(CHILD).is_none() {
        let status = std::process::Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                "agent::write_parity::audit_write_failure",
                "--nocapture",
            ])
            .env(CHILD, "1")
            .status()
            .unwrap();
        assert!(status.success());
        return;
    }
    let dir = temp_path("failure");
    std::fs::create_dir(&dir).unwrap();
    let path = dir.join("file");
    std::fs::write(&path, b"before").unwrap();
    // Limit only this subprocess; the kernel fails the write after eight bytes.
    let mut old = libc::rlimit {
        rlim_cur: 0,
        rlim_max: 0,
    };
    unsafe {
        assert_eq!(libc::getrlimit(libc::RLIMIT_FSIZE, &mut old), 0);
        libc::signal(libc::SIGXFSZ, libc::SIG_IGN);
        let limit = libc::rlimit {
            rlim_cur: 8,
            rlim_max: old.rlim_max,
        };
        assert_eq!(libc::setrlimit(libc::RLIMIT_FSIZE, &limit), 0);
    }
    let result = write_result(path.to_str(), Some("0123456789abcdef"));
    unsafe {
        assert_eq!(libc::setrlimit(libc::RLIMIT_FSIZE, &old), 0);
    }
    let saved = std::fs::read(&path).unwrap();
    let entries = std::fs::read_dir(&dir).unwrap().count();
    std::fs::remove_dir_all(dir).unwrap();
    assert!(result.starts_with(b"Tool error:"));
    assert_eq!(saved, b"before");
    assert_eq!(entries, 1, "failed writes must clean their staging file");
}

#[test]
fn audit_hardlink_write() {
    let path = temp_path("linked");
    let alias = path.with_extension("alias");
    std::fs::write(&path, b"before").unwrap();
    std::fs::hard_link(&path, &alias).unwrap();
    let result = write_result(path.to_str(), Some("after"));
    let edited = super::edit::edit_result(path.to_str(), Some("before"), Some("after"));
    let saved = std::fs::read(&path).unwrap();
    let aliased = std::fs::read(&alias).unwrap();
    std::fs::remove_file(path).unwrap();
    std::fs::remove_file(alias).unwrap();
    assert!(result.starts_with(b"Tool error:"));
    assert!(edited.starts_with(b"Tool error:"));
    assert_eq!(saved, b"before");
    assert_eq!(aliased, b"before");
}

#[test]
fn audit_symlink_write() {
    use std::os::unix::fs::{symlink, PermissionsExt};
    let dir = temp_path("symlink");
    std::fs::create_dir(&dir).unwrap();
    let target = dir.join("target");
    let link = dir.join("link");
    symlink("target", &link).unwrap();
    // A dangling link must create its target, not replace the symlink.
    assert!(write_result(link.to_str(), Some("before")).starts_with(b"Wrote"));
    std::fs::set_permissions(&target, std::fs::Permissions::from_mode(0o640)).unwrap();
    assert!(
        super::edit::edit_result(link.to_str(), Some("before"), Some("after"))
            .starts_with(b"Edited")
    );
    assert_eq!(std::fs::read(&target).unwrap(), b"after");
    assert_eq!(std::fs::read_link(&link).unwrap(), PathBuf::from("target"));
    assert_eq!(
        std::fs::metadata(&target).unwrap().permissions().mode() & 0o777,
        0o640
    );
    assert_eq!(std::fs::read_dir(&dir).unwrap().count(), 2);
    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
#[ignore = "run under strace -f -e inject=fsync:error=EIO (or renameat:error=EIO)"]
fn audit_commit_failure() {
    let dir = temp_path("commit-failure");
    std::fs::create_dir(&dir).unwrap();
    let path = dir.join("file");
    std::fs::write(&path, b"before").unwrap();
    let result = write_result(path.to_str(), Some("after"));
    let saved = std::fs::read(&path).unwrap();
    let entries = std::fs::read_dir(&dir).unwrap().count();
    std::fs::remove_dir_all(dir).unwrap();
    assert!(result.starts_with(b"Tool error:"));
    assert_eq!(saved, b"before");
    assert_eq!(entries, 1);
}
