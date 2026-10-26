use super::approval::{Approval, DENIED_WRITE};
use super::web_tools::io_detail;

pub(crate) const WRITE: &str = "write";

pub(crate) fn write_result(path: Option<&str>, content: Option<&str>) -> Vec<u8> {
    write_result_with(path, content, &mut Approval::NonInteractive)
}

pub(crate) fn write_result_with(
    path: Option<&str>,
    content: Option<&str>,
    approval: &mut Approval<'_>,
) -> Vec<u8> {
    let Some(path) = path.filter(|path| !path.is_empty()) else {
        return b"Tool error: write requires path\n".to_vec();
    };
    let Some(content) = content else {
        return b"Tool error: write requires content\n".to_vec();
    };
    if !approval.allow_write(path) {
        return DENIED_WRITE.to_vec();
    }
    if let Err((stage, error)) = super::file_write::replace(path.as_ref(), content.as_bytes()) {
        let stage = if stage == "open" {
            "open for write"
        } else {
            "write"
        };
        return format!("Tool error: {stage} failed: {}\n", io_detail(&error)).into_bytes();
    }
    format!("Wrote {} bytes to {path}\n", content.len()).into_bytes()
}
