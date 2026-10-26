//! Interactive write/edit approval matching C `agent_prompt_yes_no`.
//! `--non-interactive` never asks; interactive never auto-allows.

use std::io::{self, BufRead, Write};

pub(crate) const DENIED_WRITE: &[u8] = b"Tool error: user denied write\n";
pub(crate) const DENIED_EDIT: &[u8] = b"Tool error: user denied edit\n";

pub(crate) trait Ask {
    fn yes_no(&mut self, prompt: &str) -> bool;
}

pub(crate) enum Approval<'a> {
    NonInteractive,
    Interactive(&'a mut dyn Ask),
}

impl Approval<'_> {
    pub(crate) fn allow_write(&mut self, path: &str) -> bool {
        self.allow(&write_prompt(path))
    }

    pub(crate) fn allow_edit(&mut self, path: &str) -> bool {
        self.allow(&edit_prompt(path))
    }

    fn allow(&mut self, prompt: &str) -> bool {
        match self {
            Self::NonInteractive => true,
            Self::Interactive(ask) => ask.yes_no(prompt),
        }
    }
}

pub(crate) struct StdinAsk;

impl Ask for StdinAsk {
    fn yes_no(&mut self, prompt: &str) -> bool {
        let stdin = io::stdin();
        let mut stdout = io::stdout();
        prompt_yes_no(prompt, &mut stdin.lock(), &mut stdout)
    }
}

fn write_prompt(path: &str) -> String {
    format!("Write {path}? (y/n) ")
}

fn edit_prompt(path: &str) -> String {
    format!("Edit {path}? (y/n) ")
}

/// C `agent_prompt_yes_no`: skip space/tab, `y`/`Y` allow, `n`/`N` deny, else re-prompt.
pub(crate) fn parse_yes_no_line(line: &str) -> Option<bool> {
    let bytes = line.as_bytes();
    let mut i = 0;
    while i < bytes.len() && (bytes[i] == b' ' || bytes[i] == b'\t') {
        i += 1;
    }
    match bytes.get(i) {
        Some(b'y' | b'Y') => Some(true),
        Some(b'n' | b'N') => Some(false),
        Some(_) | None => None,
    }
}

fn prompt_yes_no(prompt: &str, reader: &mut impl BufRead, writer: &mut impl Write) -> bool {
    loop {
        if write!(writer, "{prompt}").is_err() || writer.flush().is_err() {
            return false;
        }
        let mut line = String::new();
        match reader.read_line(&mut line) {
            Ok(0) => return false,
            Ok(_) => {
                if let Some(answer) = parse_yes_no_line(&line) {
                    return answer;
                }
            }
            Err(_) => return false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::edit::edit_result_with;
    use super::super::write::write_result_with;
    use super::{
        parse_yes_no_line, prompt_yes_no, Approval, Ask, StdinAsk, DENIED_EDIT, DENIED_WRITE,
    };
    use std::io::Cursor;
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicU64, Ordering};

    static TEMP_ID: AtomicU64 = AtomicU64::new(0);

    struct ScriptedAsk {
        answers: Vec<bool>,
        asked: usize,
    }

    impl Ask for ScriptedAsk {
        fn yes_no(&mut self, _prompt: &str) -> bool {
            self.asked += 1;
            if self.answers.is_empty() {
                return false;
            }
            self.answers.remove(0)
        }
    }

    fn temp_path(name: &str) -> PathBuf {
        PathBuf::from(format!(
            "/tmp/ds4_agent_approval_{}_{}_{name}",
            std::process::id(),
            TEMP_ID.fetch_add(1, Ordering::Relaxed)
        ))
    }

    #[test]
    fn stdin_ask_is_interactive_asker() {
        fn needs_ask<T: Ask>(_: T) {}
        needs_ask(StdinAsk);
    }

    #[test]
    fn parse_yes_no_line_matches_c() {
        assert_eq!(parse_yes_no_line("y\n"), Some(true));
        assert_eq!(parse_yes_no_line("Y"), Some(true));
        assert_eq!(parse_yes_no_line(" n\n"), Some(false));
        assert_eq!(parse_yes_no_line("\tN"), Some(false));
        assert_eq!(parse_yes_no_line("yes"), Some(true));
        assert_eq!(parse_yes_no_line("no"), Some(false));
        assert_eq!(parse_yes_no_line(""), None);
        assert_eq!(parse_yes_no_line("maybe\n"), None);
    }

    #[test]
    fn prompt_yes_no_denies_on_eof_and_allows_after_retry() {
        let mut out = Vec::new();
        assert!(!prompt_yes_no("P ", &mut Cursor::new(""), &mut out));

        let mut out = Vec::new();
        assert!(prompt_yes_no(
            "P ",
            &mut Cursor::new("maybe\ny\n"),
            &mut out
        ));
        assert_eq!(out, b"P P ");
    }

    #[test]
    fn write_is_blocked_when_interactive_deny() {
        let path = temp_path("deny.txt");
        std::fs::write(&path, b"old").unwrap();
        let path_text = path.to_str().expect("utf8");
        let mut ask = ScriptedAsk {
            answers: vec![false],
            asked: 0,
        };
        let out = write_result_with(
            Some(path_text),
            Some("new"),
            &mut Approval::Interactive(&mut ask),
        );
        assert_eq!(out, DENIED_WRITE);
        assert_eq!(ask.asked, 1);
        assert_eq!(std::fs::read(&path).unwrap(), b"old");
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn write_overwrites_when_non_interactive() {
        let path = temp_path("overwrite.txt");
        std::fs::write(&path, b"old").unwrap();
        let path_text = path.to_str().expect("utf8");
        let out = write_result_with(Some(path_text), Some("z"), &mut Approval::NonInteractive);
        assert_eq!(out, format!("Wrote 1 bytes to {path_text}\n").into_bytes());
        assert_eq!(std::fs::read(&path).unwrap(), b"z");
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn write_proceeds_when_interactive_allow() {
        let path = temp_path("allow.txt");
        let path_text = path.to_str().expect("utf8");
        let mut ask = ScriptedAsk {
            answers: vec![true],
            asked: 0,
        };
        let out = write_result_with(
            Some(path_text),
            Some("hello"),
            &mut Approval::Interactive(&mut ask),
        );
        assert_eq!(out, format!("Wrote 5 bytes to {path_text}\n").into_bytes());
        assert_eq!(ask.asked, 1);
        assert_eq!(std::fs::read(&path).unwrap(), b"hello");
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn write_missing_args_do_not_ask_when_interactive() {
        let mut ask = ScriptedAsk {
            answers: vec![true],
            asked: 0,
        };
        let out = write_result_with(None, Some("x"), &mut Approval::Interactive(&mut ask));
        assert_eq!(out, b"Tool error: write requires path\n");
        assert_eq!(ask.asked, 0);
    }

    #[test]
    fn edit_is_blocked_when_interactive_deny() {
        let path = temp_path("edit-deny.txt");
        std::fs::write(&path, b"alpha\nkeep\n").unwrap();
        let path_text = path.to_str().expect("utf8");
        let mut ask = ScriptedAsk {
            answers: vec![false],
            asked: 0,
        };
        let out = edit_result_with(
            Some(path_text),
            Some("alpha"),
            Some("beta"),
            &mut Approval::Interactive(&mut ask),
        );
        assert_eq!(out, DENIED_EDIT);
        assert_eq!(ask.asked, 1);
        assert_eq!(std::fs::read(&path).unwrap(), b"alpha\nkeep\n");
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn edit_unique_replace_when_non_interactive() {
        let path = temp_path("edit-ni.txt");
        std::fs::write(&path, b"alpha\nkeep\n").unwrap();
        let path_text = path.to_str().expect("utf8");
        let out = edit_result_with(
            Some(path_text),
            Some("alpha"),
            Some("beta"),
            &mut Approval::NonInteractive,
        );
        assert!(out.starts_with(format!("Edited {path_text} using old/new replacement").as_bytes()));
        assert_eq!(std::fs::read(&path).unwrap(), b"beta\nkeep\n");
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn edit_not_unique_does_not_ask_when_interactive() {
        let path = temp_path("dup.txt");
        std::fs::write(&path, b"aaaa").unwrap();
        let path_text = path.to_str().expect("utf8");
        let mut ask = ScriptedAsk {
            answers: vec![true],
            asked: 0,
        };
        let out = edit_result_with(
            Some(path_text),
            Some("aaa"),
            Some("y"),
            &mut Approval::Interactive(&mut ask),
        );
        assert!(out.starts_with(b"Tool error: "));
        assert_eq!(ask.asked, 0);
        assert_eq!(std::fs::read(&path).unwrap(), b"aaaa");
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn audit_edit_approval_reread() {
        struct SaveDuringAsk<'a> {
            path: &'a std::path::Path,
            text: &'a str,
        }
        impl Ask for SaveDuringAsk<'_> {
            fn yes_no(&mut self, _: &str) -> bool {
                std::fs::write(self.path, self.text).unwrap();
                true
            }
        }
        for (text, expected, success) in [
            ("alpha\nnew tail\n", "beta\nnew tail\n", true),
            ("gone\n", "gone\n", false),
            ("alpha alpha\n", "alpha alpha\n", false),
        ] {
            let path = temp_path("concurrent-edit");
            std::fs::write(&path, b"alpha\nkeep\n").unwrap();
            let mut ask = SaveDuringAsk { path: &path, text };
            let result = edit_result_with(
                path.to_str(),
                Some("alpha"),
                Some("beta"),
                &mut Approval::Interactive(&mut ask),
            );
            let saved = std::fs::read_to_string(&path).unwrap();
            std::fs::remove_file(path).unwrap();
            assert_eq!(saved, expected);
            assert_eq!(result.starts_with(b"Edited"), success);
        }
    }
}
