use ds4_core::chat_template::{RenderClock, Template};
use serde_json::json;
use std::process::Command;

const CHILD_RESULT: &str = "DS4_CHAT_CLOCK_RESULT";
const SOURCE: &str = "{{ strftime_now('%Y-%m-%d %H:%M:%S') }}";

// Each child starts with its own TZ; no test changes the parent environment.
const ORACLE: &str = r#"
import datetime
import json
import os
import pathlib
import subprocess
import sys
import tempfile

with tempfile.TemporaryDirectory(prefix='ds4-chat-clock-') as directory:
    result = pathlib.Path(directory) / 'result.json'
    environment = dict(os.environ, DS4_CHAT_CLOCK_RESULT=str(result))
    before = datetime.datetime.now().replace(microsecond=0)
    child = subprocess.run(
        [sys.argv[1], '--exact', 'model_clock_uses_local_time', '--nocapture'],
        env=environment, capture_output=True, text=True,
    )
    assert child.returncode == 0, child.stdout + child.stderr
    after = datetime.datetime.now().replace(microsecond=0)
    rendered = json.loads(result.read_text())
    actual = datetime.datetime.strptime(rendered['system'], '%Y-%m-%d %H:%M:%S')
    assert before <= actual <= after, (
        f"TZ={os.environ['TZ']}: rendered {actual}, expected local time "
        f"between {before} and {after}"
    )
    assert rendered['fixed'] == '1970-01-01 00:00:00', rendered
"#;

#[test]
fn model_clock_uses_local_time() {
    if let Some(result) = std::env::var_os(CHILD_RESULT) {
        render_child(std::path::Path::new(&result));
        return;
    }

    let executable = std::env::current_exe().unwrap();
    let mut failures = Vec::new();
    for timezone in ["Etc/GMT-9", "Etc/GMT+7"] {
        let output = Command::new("python3")
            .args(["-c", ORACLE])
            .arg(&executable)
            .env("TZ", timezone)
            .env_remove(CHILD_RESULT)
            .output()
            .unwrap();
        if !output.status.success() {
            failures.push(String::from_utf8_lossy(&output.stderr).into_owned());
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

fn render_child(result: &std::path::Path) {
    let directory = result.parent().unwrap();
    let model = directory.join("model.gguf");
    // A header without tensors exercises asset loading without model weights.
    let mut header = b"GGUF".to_vec();
    header.extend(3u32.to_le_bytes());
    header.extend(0u64.to_le_bytes());
    header.extend(0u64.to_le_bytes());
    header.resize(32, 0);
    std::fs::write(&model, header).unwrap();
    std::fs::write(directory.join("chat_template.jinja"), SOURCE).unwrap();

    let template = Template::from_model(&model).unwrap().unwrap();
    let system = template.render(&json!({})).unwrap();
    let fixed = Template::compile(SOURCE, RenderClock::Fixed(0))
        .unwrap()
        .render(&json!({}))
        .unwrap();
    std::fs::write(
        result,
        serde_json::to_vec(&json!({"system": system, "fixed": fixed})).unwrap(),
    )
    .unwrap();
}
