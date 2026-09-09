//! Interactive CLI commands matching `ds4_cli.c` `print_repl_help` / `run_repl`.

use std::sync::atomic::{AtomicBool, Ordering};

static INTERRUPTED: AtomicBool = AtomicBool::new(false);

extern "C" fn sigint_handler(_: libc::c_int) {
    INTERRUPTED.store(true, Ordering::SeqCst);
}

pub fn interrupt_requested() -> bool {
    INTERRUPTED.load(Ordering::SeqCst)
}

pub fn interrupt_clear() {
    INTERRUPTED.store(false, Ordering::SeqCst);
}

pub struct InterruptGuard {
    old: Option<libc::sigaction>,
}

impl InterruptGuard {
    pub fn install() -> Self {
        interrupt_clear();
        // SAFETY: [Category 4 — Uninitialized Memory] `sigaction` is a C POD;
        // all-zero is a valid bit pattern. `old` is written by `sigaction`
        // before any field is read; `sa` is filled (handler, mask) before install.
        let mut old = unsafe { std::mem::zeroed::<libc::sigaction>() };
        let mut sa = unsafe { std::mem::zeroed::<libc::sigaction>() };
        sa.sa_sigaction = sigint_handler as *const () as usize;
        // SAFETY: [Category 8 — FFI] `sigemptyset` only writes the caller-owned
        // mask; the pointer is not retained.
        unsafe {
            libc::sigemptyset(&mut sa.sa_mask);
        }
        // SAFETY: [Category 8 — FFI] installs a process-wide SIGINT handler and
        // copies the previous action into `old` so Drop can restore it. `sa` is
        // fully initialized; both pointers are stack-owned for the call only.
        let rc = unsafe { libc::sigaction(libc::SIGINT, &sa, &mut old) };
        Self {
            old: (rc == 0).then_some(old),
        }
    }
}

impl Drop for InterruptGuard {
    fn drop(&mut self) {
        if let Some(old) = self.old.take() {
            // SAFETY: [Category 8 — FFI] restores the previous SIGINT action
            // captured at install. `old` is a complete `sigaction`; NULL oact
            // is allowed by POSIX.
            unsafe {
                libc::sigaction(libc::SIGINT, &old, std::ptr::null_mut());
            }
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InterruptEnd {
    Rollback,
    KeepEos,
}

pub fn interrupt_end(interrupted: bool, generated: i32) -> InterruptEnd {
    if interrupted && generated == 0 {
        InterruptEnd::Rollback
    } else {
        InterruptEnd::KeepEos
    }
}

unsafe extern "C" {
    fn linenoise(prompt: *const libc::c_char) -> *mut libc::c_char;
    fn linenoiseFree(ptr: *mut libc::c_void);
    fn linenoiseSetMultiLine(ml: libc::c_int);
    fn linenoiseHistorySetMaxLen(len: libc::c_int) -> libc::c_int;
    fn linenoiseHistoryLoad(filename: *const libc::c_char) -> libc::c_int;
    fn linenoiseHistoryAdd(line: *const libc::c_char) -> libc::c_int;
    fn linenoiseHistorySave(filename: *const libc::c_char) -> libc::c_int;
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ReplRead {
    Line(String),
    Eof,
    Interrupted,
}

pub fn stdin_is_tty() -> bool {
    // SAFETY: [Category 8 — FFI] `STDIN_FILENO` is the process stdin; `isatty`
    // only inspects the descriptor and does not retain it.
    unsafe { libc::isatty(libc::STDIN_FILENO) == 1 }
}

pub fn init_linenoise_history(path: &std::path::Path) {
    let Some(c_path) = path.to_str().and_then(|p| std::ffi::CString::new(p).ok()) else {
        return;
    };
    // SAFETY: [Category 8 — FFI] linenoise config/load take a NUL-terminated
    // path borrowed for the call. The C library copies what it needs; `c_path`
    // outlives the three calls.
    unsafe {
        linenoiseSetMultiLine(1);
        linenoiseHistorySetMaxLen(HISTORY_MAX_LEN as libc::c_int);
        linenoiseHistoryLoad(c_path.as_ptr());
    }
}

pub fn remember_linenoise_line(path: &std::path::Path, line: &str) {
    let Ok(c_line) = std::ffi::CString::new(line) else {
        return;
    };
    let Some(c_path) = path.to_str().and_then(|p| std::ffi::CString::new(p).ok()) else {
        return;
    };
    // SAFETY: [Category 8 — FFI] both pointers are live CStrings borrowed for
    // the call. linenoise copies the line into its history; it does not retain
    // the path pointer after return.
    unsafe {
        linenoiseHistoryAdd(c_line.as_ptr());
        linenoiseHistorySave(c_path.as_ptr());
    }
}

pub fn read_repl_line(prompt: &str) -> Result<ReplRead, String> {
    if stdin_is_tty() {
        return read_linenoise_line(prompt);
    }
    read_plain_line()
}

fn read_linenoise_line(prompt: &str) -> Result<ReplRead, String> {
    let c_prompt = std::ffi::CString::new(prompt).map_err(|_| "prompt contains NUL".to_string())?;
    // SAFETY: [Category 8 — FFI] prompt is a live CString; linenoise does not
    // retain it. A non-NULL return is a heap C string owned until `linenoiseFree`.
    let ptr = unsafe { linenoise(c_prompt.as_ptr()) };
    if ptr.is_null() {
        if interrupt_requested()
            || std::io::Error::last_os_error().raw_os_error() == Some(libc::EAGAIN)
        {
            return Ok(ReplRead::Interrupted);
        }
        return Ok(ReplRead::Eof);
    }
    // SAFETY: [Category 8 — FFI] `ptr` is non-null (checked) and NUL-terminated
    // by linenoise; we copy to an owned String before free.
    let line = unsafe { std::ffi::CStr::from_ptr(ptr) }
        .to_string_lossy()
        .into_owned();
    // SAFETY: [Category 12 — Double free] exclusive ownership of the linenoise
    // allocation; this is the only free, after the CStr copy.
    unsafe { linenoiseFree(ptr.cast()) };
    Ok(ReplRead::Line(line))
}

fn read_plain_line() -> Result<ReplRead, String> {
    use std::io::{self, BufRead, Write};
    let mut stdout = io::stdout();
    write!(stdout, "ds4> ").map_err(|e| e.to_string())?;
    stdout.flush().map_err(|e| e.to_string())?;
    let mut line = String::new();
    match io::stdin().lock().read_line(&mut line) {
        Ok(0) if interrupt_requested() => Ok(ReplRead::Interrupted),
        Ok(0) => Ok(ReplRead::Eof),
        Ok(_) => Ok(ReplRead::Line(line)),
        Err(error) if error.kind() == io::ErrorKind::Interrupted => Ok(ReplRead::Interrupted),
        Err(error) => Err(error.to_string()),
    }
}

pub const HISTORY_MAX_LEN: usize = 512;

pub fn history_file_path() -> std::path::PathBuf {
    let home = std::env::var_os("HOME").filter(|home| !home.is_empty());
    std::path::PathBuf::from(home.unwrap_or_else(|| ".".into())).join(".ds4_history")
}

pub struct History {
    path: std::path::PathBuf,
    lines: Vec<String>,
}

impl History {
    pub fn load(path: std::path::PathBuf) -> Self {
        let mut hist = Self {
            path,
            lines: Vec::new(),
        };
        let Ok(bytes) = std::fs::read(&hist.path) else {
            return hist;
        };
        for raw in bytes.split(|byte| *byte == b'\n') {
            if raw.is_empty() {
                continue;
            }
            let mut entry = raw.to_vec();
            for byte in &mut entry {
                if *byte == b'\r' {
                    *byte = b'\n';
                }
            }
            hist.add(String::from_utf8_lossy(&entry).into_owned());
        }
        hist
    }

    pub fn add(&mut self, line: String) -> bool {
        if line.is_empty() {
            return false;
        }
        if self.lines.last().is_some_and(|last| last == &line) {
            return false;
        }
        if self.lines.len() == HISTORY_MAX_LEN {
            self.lines.remove(0);
        }
        self.lines.push(line);
        true
    }

    pub fn save(&self) -> std::io::Result<()> {
        use std::io::Write;
        use std::os::unix::fs::OpenOptionsExt;

        let mut file = std::fs::OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)
            .mode(0o600)
            .open(&self.path)?;
        for line in &self.lines {
            for &byte in line.as_bytes() {
                file.write_all(&[if byte == b'\n' { b'\r' } else { byte }])?;
            }
            file.write_all(b"\n")?;
        }
        file.flush()
    }

    pub fn lines(&self) -> &[String] {
        &self.lines
    }
}

pub const REPL_HELP: &str = "\
Commands:
  /help          Show this help.
  /think         Use normal thinking mode.
  /think-max     Use Think Max only when context is at least 393216 tokens.
  /nothink       Disable thinking mode.
  /ctx N         Set context size for following prompts.
  /power N       Set GPU duty cycle percentage, 1..100.
  /read FILE     Read a prompt from FILE and run it.
  /quit, /exit   Leave the prompt.
  Ctrl+C         Stop generation and return to the prompt.
";

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ReplLine {
    Help,
    Think,
    ThinkMax,
    NoThink,
    Ctx(i32),
    Power(Option<i32>),
    Read(String),
    Quit,
    Prompt(String),
    Empty,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ReplError {
    UnknownCommand(String),
    CtxNeedsPositive,
    PowerRange,
    ReadNeedsPath,
}

impl ReplError {
    pub fn message(&self) -> String {
        match self {
            ReplError::UnknownCommand(cmd) => {
                format!("ds4: unknown command: {cmd}\nds4: type /help for commands\n")
            }
            ReplError::CtxNeedsPositive => "ds4: /ctx needs a positive integer\n".into(),
            ReplError::PowerRange => "ds4: /power must be between 1 and 100\n".into(),
            ReplError::ReadNeedsPath => "ds4: /read needs a file path\n".into(),
        }
    }
}

pub fn trim_inplace(s: &str) -> &str {
    s.trim()
}

pub fn parse_repl_line(line: &str) -> Result<ReplLine, ReplError> {
    let cmd = trim_inplace(line);
    if cmd.is_empty() {
        return Ok(ReplLine::Empty);
    }
    if cmd == "/help" {
        return Ok(ReplLine::Help);
    }
    if cmd == "/think" {
        return Ok(ReplLine::Think);
    }
    if cmd == "/think-max" {
        return Ok(ReplLine::ThinkMax);
    }
    if cmd == "/nothink" {
        return Ok(ReplLine::NoThink);
    }
    if cmd == "/quit" || cmd == "/exit" {
        return Ok(ReplLine::Quit);
    }
    if let Some(rest) = strip_command(cmd, "/power") {
        if rest.is_empty() {
            return Ok(ReplLine::Power(None));
        }
        return parse_power_percent(rest).map(|v| ReplLine::Power(Some(v)));
    }
    if let Some(rest) = strip_command(cmd, "/ctx") {
        if rest.is_empty() {
            return Err(ReplError::CtxNeedsPositive);
        }
        return parse_positive_int(rest)
            .ok_or(ReplError::CtxNeedsPositive)
            .map(ReplLine::Ctx);
    }
    if let Some(rest) = strip_command(cmd, "/read") {
        if rest.is_empty() {
            return Err(ReplError::ReadNeedsPath);
        }
        return Ok(ReplLine::Read(rest.to_string()));
    }
    if cmd.starts_with('/') {
        return Err(ReplError::UnknownCommand(cmd.to_string()));
    }
    Ok(ReplLine::Prompt(cmd.to_string()))
}

fn strip_command<'a>(cmd: &'a str, name: &str) -> Option<&'a str> {
    let rest = cmd.strip_prefix(name)?;
    if rest.is_empty() {
        return Some("");
    }
    if rest.starts_with(|c: char| c.is_ascii_whitespace()) {
        return Some(rest.trim());
    }
    None
}

fn parse_positive_int(s: &str) -> Option<i32> {
    let v = s.parse::<i32>().ok()?;
    (v > 0).then_some(v)
}

fn parse_power_percent(s: &str) -> Result<i32, ReplError> {
    let v = s.parse::<i32>().map_err(|_| ReplError::PowerRange)?;
    if (1..=100).contains(&v) {
        Ok(v)
    } else {
        Err(ReplError::PowerRange)
    }
}

pub fn think_mode_message(mode: ThinkRepl) -> &'static str {
    match mode {
        ThinkRepl::High => "Thinking mode: high.",
        ThinkRepl::Max => "Thinking mode: max.",
        ThinkRepl::HighBelowCtx => "Thinking mode: high (ctx below 393216).",
        ThinkRepl::None => "Thinking mode: none.",
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ThinkRepl {
    High,
    Max,
    HighBelowCtx,
    None,
}

pub const THINK_MAX_MIN_CTX: i32 = 393216;

pub fn think_max_active(ctx: i32) -> bool {
    ctx >= THINK_MAX_MIN_CTX
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReplThinkCmd {
    None,
    Low,
    Max,
}

pub struct ReplChat {
    pub(crate) conversation: Option<crate::conversation::Conversation>,
    pub transcript: ds4_core::TokenBuffer,
    pub prefix_tokens: usize,
    pub think: ReplThinkCmd,
    pub ctx: i32,
}

impl ReplChat {
    pub fn new(nothink: bool, ctx: i32) -> Self {
        Self {
            conversation: None,
            transcript: ds4_core::TokenBuffer::new(),
            prefix_tokens: 0,
            think: if nothink {
                ReplThinkCmd::None
            } else {
                ReplThinkCmd::Low
            },
            ctx,
        }
    }

    pub fn effective_think(&self) -> ds4_core::ChatThinkMode {
        match self.think {
            ReplThinkCmd::None => ds4_core::ChatThinkMode::None,
            ReplThinkCmd::Low => ds4_core::ChatThinkMode::Low,
            ReplThinkCmd::Max if think_max_active(self.ctx) => ds4_core::ChatThinkMode::High,
            ReplThinkCmd::Max => ds4_core::ChatThinkMode::Low,
        }
    }

    pub fn thinking_enabled(&self) -> bool {
        self.effective_think() != ds4_core::ChatThinkMode::None
    }

    pub fn think_message(&self) -> &'static str {
        think_mode_message(match self.think {
            ReplThinkCmd::None => ThinkRepl::None,
            ReplThinkCmd::Low => ThinkRepl::High,
            ReplThinkCmd::Max if think_max_active(self.ctx) => ThinkRepl::Max,
            ReplThinkCmd::Max => ThinkRepl::HighBelowCtx,
        })
    }

    pub fn wants_effort_prefix(&self) -> bool {
        matches!(self.effective_think(), ds4_core::ChatThinkMode::High)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn help_text_matches_c() {
        assert!(REPL_HELP.contains("/help          Show this help."));
        assert!(REPL_HELP.contains("/quit, /exit   Leave the prompt."));
        assert!(REPL_HELP.contains("Ctrl+C         Stop generation and return to the prompt."));
    }

    #[test]
    fn parses_commands() {
        assert_eq!(parse_repl_line("  /help  ").unwrap(), ReplLine::Help);
        assert_eq!(parse_repl_line("/think").unwrap(), ReplLine::Think);
        assert_eq!(parse_repl_line("/think-max").unwrap(), ReplLine::ThinkMax);
        assert_eq!(parse_repl_line("/nothink").unwrap(), ReplLine::NoThink);
        assert_eq!(parse_repl_line("/quit").unwrap(), ReplLine::Quit);
        assert_eq!(parse_repl_line("/exit").unwrap(), ReplLine::Quit);
        assert_eq!(parse_repl_line("/ctx 8192").unwrap(), ReplLine::Ctx(8192));
        assert_eq!(parse_repl_line("/power").unwrap(), ReplLine::Power(None));
        assert_eq!(
            parse_repl_line("/power 50").unwrap(),
            ReplLine::Power(Some(50))
        );
        assert_eq!(
            parse_repl_line("/read notes.txt").unwrap(),
            ReplLine::Read("notes.txt".into())
        );
        assert_eq!(
            parse_repl_line("hello there").unwrap(),
            ReplLine::Prompt("hello there".into())
        );
        assert_eq!(parse_repl_line("   ").unwrap(), ReplLine::Empty);
    }

    #[test]
    fn unknown_and_invalid_match_c() {
        assert_eq!(
            parse_repl_line("/nope").unwrap_err().message(),
            "ds4: unknown command: /nope\nds4: type /help for commands\n"
        );
        assert_eq!(
            parse_repl_line("/ctx").unwrap_err().message(),
            "ds4: /ctx needs a positive integer\n"
        );
        assert_eq!(
            parse_repl_line("/power 0").unwrap_err().message(),
            "ds4: /power must be between 1 and 100\n"
        );
        assert_eq!(
            parse_repl_line("/read").unwrap_err().message(),
            "ds4: /read needs a file path\n"
        );
    }

    #[test]
    fn think_messages_match_c() {
        assert_eq!(think_mode_message(ThinkRepl::High), "Thinking mode: high.");
        assert_eq!(think_mode_message(ThinkRepl::Max), "Thinking mode: max.");
        assert_eq!(
            think_mode_message(ThinkRepl::HighBelowCtx),
            "Thinking mode: high (ctx below 393216)."
        );
        assert_eq!(think_mode_message(ThinkRepl::None), "Thinking mode: none.");
        assert!(!think_max_active(32768));
        assert!(think_max_active(393216));
    }

    #[test]
    fn repl_chat_think_gates_match_c() {
        let low = ReplChat::new(false, 32768);
        assert_eq!(low.effective_think(), ds4_core::ChatThinkMode::Low);
        assert!(low.thinking_enabled());
        assert!(!low.wants_effort_prefix());
        assert_eq!(low.think_message(), "Thinking mode: high.");

        let none = ReplChat::new(true, 32768);
        assert_eq!(none.effective_think(), ds4_core::ChatThinkMode::None);
        assert!(!none.thinking_enabled());
        assert_eq!(none.think_message(), "Thinking mode: none.");

        let mut max = ReplChat::new(false, 32768);
        max.think = ReplThinkCmd::Max;
        assert_eq!(max.effective_think(), ds4_core::ChatThinkMode::Low);
        assert!(!max.wants_effort_prefix());
        assert_eq!(
            max.think_message(),
            "Thinking mode: high (ctx below 393216)."
        );

        max.ctx = THINK_MAX_MIN_CTX;
        assert_eq!(max.effective_think(), ds4_core::ChatThinkMode::High);
        assert!(max.wants_effort_prefix());
        assert_eq!(max.think_message(), "Thinking mode: max.");
    }

    #[test]
    fn interrupt_end_matches_c() {
        assert_eq!(interrupt_end(true, 0), InterruptEnd::Rollback);
        assert_eq!(interrupt_end(true, 1), InterruptEnd::KeepEos);
        assert_eq!(interrupt_end(false, 0), InterruptEnd::KeepEos);
        assert_eq!(interrupt_end(false, 4), InterruptEnd::KeepEos);
    }

    #[test]
    fn history_path_add_and_cr_roundtrip_match_c() {
        let old = std::env::var_os("HOME");
        std::env::set_var("HOME", "/tmp/ds4-home");
        assert_eq!(
            history_file_path(),
            std::path::PathBuf::from("/tmp/ds4-home/.ds4_history")
        );
        std::env::set_var("HOME", "");
        assert_eq!(
            history_file_path(),
            std::path::PathBuf::from("./.ds4_history")
        );
        match old {
            Some(home) => std::env::set_var("HOME", home),
            None => std::env::remove_var("HOME"),
        }

        let path = std::env::temp_dir().join(format!(
            "ds4-history-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0)
        ));
        let mut hist = History::load(path.clone());
        assert!(hist.add("hello".into()));
        assert!(!hist.add("hello".into()));
        assert!(hist.add("a\nb".into()));
        hist.save().unwrap();
        let raw = std::fs::read(&path).unwrap();
        assert_eq!(raw, b"hello\na\rb\n");
        let loaded = History::load(path.clone());
        assert_eq!(loaded.lines(), &["hello".to_string(), "a\nb".to_string()]);
        let _ = std::fs::remove_file(path);
    }
}
