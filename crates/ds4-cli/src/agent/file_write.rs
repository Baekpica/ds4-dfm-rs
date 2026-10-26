use std::fs::{self, OpenOptions};
use std::io::{self, Write};
use std::os::fd::AsRawFd;
use std::os::unix::fs::{MetadataExt, OpenOptionsExt};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

static TEMP_ID: AtomicU64 = AtomicU64::new(0);

#[cfg(target_os = "linux")]
fn copy_access_acl(source: &fs::File, target: &fs::File) -> io::Result<()> {
    let name = c"system.posix_acl_access";
    let mut value = vec![0u8; 65536]; // Linux XATTR_SIZE_MAX.
    let size = unsafe {
        libc::fgetxattr(
            source.as_raw_fd(),
            name.as_ptr(),
            value.as_mut_ptr().cast(),
            value.len(),
        )
    };
    if size >= 0 {
        let rc = unsafe {
            libc::fsetxattr(
                target.as_raw_fd(),
                name.as_ptr(),
                value.as_ptr().cast(),
                size as usize,
                0,
            )
        };
        return if rc == 0 {
            Ok(())
        } else {
            Err(io::Error::last_os_error())
        };
    }
    let error = io::Error::last_os_error();
    if !matches!(error.raw_os_error(), Some(libc::ENODATA | libc::EOPNOTSUPP)) {
        return Err(error);
    }
    // Staging may inherit an ACL that the original file did not have.
    let rc = unsafe { libc::fremovexattr(target.as_raw_fd(), name.as_ptr()) };
    if rc != 0 {
        let error = io::Error::last_os_error();
        if !matches!(error.raw_os_error(), Some(libc::ENODATA | libc::EOPNOTSUPP)) {
            return Err(error);
        }
    }
    Ok(())
}

fn target(path: &Path) -> io::Result<PathBuf> {
    let mut path = path.to_path_buf();
    // Follow even dangling final symlinks, as File::create did.
    for _ in 0..40 {
        match fs::symlink_metadata(&path) {
            Ok(meta) if meta.file_type().is_symlink() => {
                let link = fs::read_link(&path)?;
                path = path.parent().unwrap_or(Path::new(".")).join(link);
            }
            Ok(_) => return fs::canonicalize(path),
            Err(error) if error.kind() != io::ErrorKind::NotFound => return Err(error),
            Err(_) => {
                let parent = path
                    .parent()
                    .filter(|p| !p.as_os_str().is_empty())
                    .unwrap_or(Path::new("."));
                return Ok(fs::canonicalize(parent)?.join(
                    path.file_name()
                        .ok_or_else(|| io::Error::from(io::ErrorKind::InvalidInput))?,
                ));
            }
        }
    }
    Err(io::Error::from_raw_os_error(libc::ELOOP))
}

pub(super) fn replace(path: &Path, data: &[u8]) -> Result<(), (&'static str, io::Error)> {
    let open = |error| ("open", error);
    let path = target(path).map_err(open)?;
    if fs::symlink_metadata(&path).is_ok_and(|meta| !meta.is_file()) {
        return Err(open(io::Error::other(
            "refusing to replace a non-regular file",
        )));
    }
    let original = match OpenOptions::new()
        .write(true)
        .custom_flags(libc::O_NOFOLLOW | libc::O_NONBLOCK)
        .open(&path)
    {
        Ok(file) => Some(file),
        Err(error) if error.kind() == io::ErrorKind::NotFound => None,
        Err(error) => return Err(open(error)),
    };
    let meta = original
        .as_ref()
        .map(|file| file.metadata())
        .transpose()
        .map_err(open)?;
    if meta.as_ref().is_some_and(|m| !m.is_file() || m.nlink() > 1) {
        return Err(open(io::Error::other(
            "refusing to replace a non-regular or hardlinked file",
        )));
    }
    let parent = path.parent().expect("resolved file has a parent");
    let (tmp, mut file) = loop {
        let id = TEMP_ID.fetch_add(1, Ordering::Relaxed);
        let tmp = parent.join(format!(".ds4-write-{}-{id}.tmp", std::process::id()));
        match OpenOptions::new()
            .write(true)
            .create_new(true)
            .mode(if meta.is_some() { 0o600 } else { 0o666 })
            .open(&tmp)
        {
            Ok(file) => break (tmp, file),
            Err(error) if error.kind() == io::ErrorKind::AlreadyExists => continue,
            Err(error) => return Err(open(error)),
        }
    };
    let result = (|| {
        file.write_all(data)?;
        file.flush()?;
        if let Some(meta) = &meta {
            // Refuse the replacement if ownership cannot be preserved.
            let rc = unsafe { libc::fchown(file.as_raw_fd(), meta.uid(), meta.gid()) };
            if rc != 0 {
                return Err(io::Error::last_os_error());
            }
            file.set_permissions(meta.permissions())?;
            #[cfg(target_os = "linux")]
            copy_access_acl(original.as_ref().expect("original metadata"), &file)?;
            let current = fs::symlink_metadata(&path)?;
            if current.dev() != meta.dev()
                || current.ino() != meta.ino()
                || current.nlink() != 1
                || current.len() != meta.len()
                || current.mtime() != meta.mtime()
                || current.mtime_nsec() != meta.mtime_nsec()
                || current.ctime() != meta.ctime()
                || current.ctime_nsec() != meta.ctime_nsec()
            {
                return Err(io::Error::other("file changed while preparing replacement"));
            }
        } else if path.try_exists()? {
            return Err(io::Error::new(
                io::ErrorKind::AlreadyExists,
                "file appeared while preparing replacement",
            ));
        }
        file.sync_all()?;
        fs::rename(&tmp, &path)
    })();
    if result.is_err() {
        let _ = fs::remove_file(tmp);
    }
    result.map_err(|error| ("write", error))
}
