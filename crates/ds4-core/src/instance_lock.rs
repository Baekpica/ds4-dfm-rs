//! One model owner per lock file. A model maps tens of GiB, so a stray
//! second process is worse than a startup error. Native used to take this
//! lock inside engine open and `exit(2)` when it was held, which ended the
//! Rust host without an error; the host takes it now and native skips it.
//! `File::try_lock` is `flock`, so deprecated C hosts still exclude us.

use std::fs::{File, OpenOptions, TryLockError};
use std::io::{Read, Write};
use std::os::unix::fs::OpenOptionsExt;
use std::path::{Path, PathBuf};

use crate::Error;

const LOCK_PATH_ENV: &str = "DS4_LOCK_FILE";
const DEFAULT_LOCK_PATH: &str = "/tmp/ds4.lock";
/// The C hosts' exit status for a lock failure.
const LOCK_ERROR_CODE: i32 = 2;

/// Holds the lock until dropped; the file names the owner's pid.
pub(crate) fn acquire() -> Result<File, Error> {
    let path = std::env::var_os(LOCK_PATH_ENV)
        .filter(|path| !path.is_empty())
        .map_or_else(|| PathBuf::from(DEFAULT_LOCK_PATH), PathBuf::from);
    acquire_at(&path)
}

fn acquire_at(path: &Path) -> Result<File, Error> {
    let fail = |what: &str, error: std::io::Error| Error {
        code: LOCK_ERROR_CODE,
        message: format!("failed to {what} lock file {}: {error}", path.display()),
    };
    let mut file = OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(false)
        .mode(0o600)
        .open(path)
        .map_err(|e| fail("open", e))?;

    match file.try_lock() {
        Ok(()) => {}
        Err(TryLockError::WouldBlock) => return Err(held(&mut file)),
        Err(TryLockError::Error(e)) => return Err(fail("lock", e)),
    }

    file.set_len(0).map_err(|e| fail("truncate", e))?;
    writeln!(file, "{}", std::process::id()).map_err(|e| fail("write", e))?;
    Ok(file)
}

/// Names the holder from the pid it wrote.
fn held(file: &mut File) -> Error {
    let mut text = String::new();
    let owner = file
        .read_to_string(&mut text)
        .ok()
        .and_then(|_| text.trim().parse::<u32>().ok());
    let message = match owner {
        Some(pid) if pid == std::process::id() => "a model is already open in this process".into(),
        Some(pid) => {
            format!("another ds4 process is already running (pid {pid}); refusing to start")
        }
        None => "another ds4 process is already running; refusing to start".into(),
    };
    Error {
        code: LOCK_ERROR_CODE,
        message,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn held_lock_is_an_error() {
        let dir = std::env::temp_dir().join(format!("ds4-lock-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("ds4.lock");

        let first = acquire_at(&path).unwrap();
        let again = acquire_at(&path).unwrap_err();
        assert_eq!(again.code, LOCK_ERROR_CODE);
        assert!(again.message.contains("this process"), "{}", again.message);

        // C hosts lock with flock(2); they must see the lock as held.
        let status = std::process::Command::new("flock")
            .args(["-n", path.to_str().unwrap(), "true"])
            .status();
        if let Ok(status) = status {
            assert!(!status.success());
        }

        drop(first);
        drop(acquire_at(&path).unwrap());
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
