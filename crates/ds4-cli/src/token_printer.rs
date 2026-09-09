//! CLI output: preserve C thinking formatting and project Inkling channel IDs.

use std::io::{self, IsTerminal, Write};

use ds4_core::ModelFamily;

const THINK_OPEN: &[u8] = b"<think>";
const THINK_CLOSE: &[u8] = b"</think>";
const INKLING_MODEL: i32 = 200001;
const INKLING_TEXT: i32 = 200004;
const INKLING_THINK: i32 = 200008;
const INKLING_END: i32 = 200010;

/// C `token_printer_set_grey`: SGR 90 bright-black.
const GREY: &[u8] = b"\x1b[90m";
/// C `token_printer_reset_color`.
const RESET: &[u8] = b"\x1b[0m";

pub(crate) struct TokenPrinter {
    family: ModelFamily,
    format_thinking: bool,
    in_think: bool,
    color_open: bool,
    use_color: bool,
    pending: Vec<u8>,
    last_output_newline: bool,
}

impl TokenPrinter {
    pub(crate) fn new(family: ModelFamily, format_thinking: bool) -> Self {
        let mut printer = Self::with_color(format_thinking, io::stdout().is_terminal());
        printer.family = family;
        if family == ModelFamily::Inkling {
            printer.in_think = false;
        }
        printer
    }

    pub(crate) fn with_color(format_thinking: bool, use_color: bool) -> Self {
        Self {
            family: ModelFamily::DeepSeek4,
            format_thinking,
            in_think: format_thinking,
            color_open: false,
            use_color,
            pending: Vec::new(),
            last_output_newline: true,
        }
    }

    pub(crate) fn write_token<W: Write>(
        &mut self,
        out: &mut W,
        token: i32,
        text: &[u8],
    ) -> io::Result<()> {
        if self.family != ModelFamily::Inkling {
            return self.write_text(out, text);
        }
        // Use special IDs so ordinary text that spells a marker stays literal.
        // Inkling emits these boundaries even when thinking effort is zero.
        match token {
            INKLING_MODEL => return Ok(()),
            INKLING_THINK => {
                self.in_think = self.format_thinking;
                return Ok(());
            }
            INKLING_TEXT | INKLING_END => {
                self.in_think = false;
                self.reset_color(out)?;
                if token == INKLING_END && !self.last_output_newline {
                    out.write_all(b"\n")?;
                    self.last_output_newline = true;
                }
                return Ok(());
            }
            _ => {}
        }
        for &byte in text {
            self.write_char(out, byte)?;
        }
        Ok(())
    }

    pub(crate) fn write_text<W: Write>(&mut self, out: &mut W, text: &[u8]) -> io::Result<()> {
        if !self.format_thinking {
            out.write_all(text)?;
            if let Some(last) = text.last() {
                self.last_output_newline = *last == b'\n';
            }
            return Ok(());
        }
        self.process(out, text, false)
    }

    pub(crate) fn finish<W: Write>(&mut self, out: &mut W) -> io::Result<()> {
        if self.format_thinking {
            self.process(out, &[], true)?;
            self.reset_color(out)?;
        }
        if !self.last_output_newline {
            out.write_all(b"\n")?;
            self.last_output_newline = true;
        }
        out.flush()
    }

    fn process<W: Write>(&mut self, out: &mut W, text: &[u8], finish: bool) -> io::Result<()> {
        let mut bytes = std::mem::take(&mut self.pending);
        bytes.extend_from_slice(text);
        let mut i = 0;
        while i < bytes.len() {
            let rem = &bytes[i..];
            if rem.starts_with(THINK_OPEN) {
                self.in_think = true;
                i += THINK_OPEN.len();
                continue;
            }
            if rem.starts_with(THINK_CLOSE) {
                self.in_think = false;
                self.reset_color(out)?;
                if !self.last_output_newline {
                    out.write_all(b"\n")?;
                    self.last_output_newline = true;
                }
                i += THINK_CLOSE.len();
                continue;
            }
            if !finish
                && rem[0] == b'<'
                && ((rem.len() < THINK_OPEN.len() && THINK_OPEN.starts_with(rem))
                    || (rem.len() < THINK_CLOSE.len() && THINK_CLOSE.starts_with(rem)))
            {
                self.pending.extend_from_slice(rem);
                break;
            }
            self.write_char(out, rem[0])?;
            i += 1;
        }
        Ok(())
    }

    fn write_char<W: Write>(&mut self, out: &mut W, c: u8) -> io::Result<()> {
        if self.in_think {
            self.set_grey(out)?;
        }
        out.write_all(&[c])?;
        self.last_output_newline = c == b'\n';
        Ok(())
    }

    fn set_grey<W: Write>(&mut self, out: &mut W) -> io::Result<()> {
        if self.use_color && !self.color_open {
            out.write_all(GREY)?;
            self.color_open = true;
        }
        Ok(())
    }

    fn reset_color<W: Write>(&mut self, out: &mut W) -> io::Result<()> {
        if self.use_color && self.color_open {
            out.write_all(RESET)?;
            self.color_open = false;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    enum Style {
        Plain,
        Thinking,
        Color,
    }

    fn inkling_output(style: Style, pieces: &[(i32, &[u8])]) -> Vec<u8> {
        let mut printer = TokenPrinter::new(ModelFamily::Inkling, !matches!(style, Style::Plain));
        printer.use_color = matches!(style, Style::Color);
        let mut output = Vec::new();
        for &(id, text) in pieces {
            printer.write_token(&mut output, id, text).unwrap();
        }
        printer.finish(&mut output).unwrap();
        output
    }

    #[test]
    fn inkling_text_boundaries() {
        let pieces: &[(i32, &[u8])] = &[
            (200004, b"<|content_text|>"),
            (19, b"4"),
            (200010, b"<|end_message|>"),
        ];
        for style in [Style::Plain, Style::Thinking] {
            assert_eq!(inkling_output(style, pieces), b"4\n");
        }
    }

    #[test]
    fn inkling_thinking_transition() {
        let pieces: &[(i32, &[u8])] = &[
            (200008, b"<|content_thinking|>"),
            (100, b"Let me check."),
            (200010, b"<|end_message|>"),
            (200001, b"<|message_model|>"),
            (200004, b"<|content_text|>"),
            (19, b"4"),
            (200010, b"<|end_message|>"),
        ];
        assert_eq!(
            inkling_output(Style::Thinking, pieces),
            b"Let me check.\n4\n"
        );
        assert_eq!(
            inkling_output(Style::Color, pieces),
            b"\x1b[90mLet me check.\x1b[0m\n4\n"
        );
    }

    #[test]
    fn inkling_literal_markers() {
        let text = b"Use <think> and <|content_text|> literally.";
        assert_eq!(
            inkling_output(Style::Thinking, &[(100, text)]),
            [text.as_slice(), b"\n"].concat()
        );
    }

    #[test]
    fn inkling_empty_thinking() {
        assert_eq!(
            inkling_output(
                Style::Color,
                &[
                    (200008, b"<|content_thinking|>"),
                    (200010, b"<|end_message|>"),
                    (200004, b"<|content_text|>"),
                    (19, b"4"),
                ]
            ),
            b"4\n"
        );
    }
}
