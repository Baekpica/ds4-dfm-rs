//! Worker receive prefetch: `DS4_DIST_WORKER_PREFETCH_DEPTH` and the C job queue.

use std::collections::VecDeque;
use std::io::{self, Write};
use std::net::TcpStream;
use std::sync::{Arc, Condvar, Mutex};
use std::thread;

use crate::codec::{MSG_ERROR, MSG_SNAPSHOT_LOAD_BEGIN, MSG_SNAPSHOT_SAVE_REQ, MSG_WORK};
use crate::exec::SliceExec;
use crate::native_snapshot::SnapshotStore;
use crate::transport::{read_frame, write_frame};
use crate::worker::Worker;

pub const PREFETCH_DEPTH_DEFAULT: u32 = 2;
pub const PREFETCH_DEPTH_MIN: u32 = 1;
pub const PREFETCH_DEPTH_MAX: u32 = 8;
pub const ERR_OOM_QUEUE: &str = "out of memory queueing distributed WORK";
pub const ERR_OOM_READ: &str = "out of memory reading distributed WORK frame";

pub fn prefetch_depth_from(env: Option<&str>) -> u32 {
    let Some(raw) = env else {
        return PREFETCH_DEPTH_DEFAULT;
    };
    if raw.is_empty() {
        return PREFETCH_DEPTH_DEFAULT;
    }
    match parse_strtol(raw) {
        Some(v) if v >= i64::from(PREFETCH_DEPTH_MIN) && v <= i64::from(PREFETCH_DEPTH_MAX) => {
            v as u32
        }
        _ => PREFETCH_DEPTH_DEFAULT,
    }
}

pub fn prefetch_depth() -> u32 {
    prefetch_depth_from(
        std::env::var("DS4_DIST_WORKER_PREFETCH_DEPTH")
            .ok()
            .as_deref(),
    )
}

pub fn prefetch_disabled_from(env: Option<&std::ffi::OsStr>) -> bool {
    env.is_some()
}

pub fn prefetch_disabled() -> bool {
    prefetch_disabled_from(std::env::var_os("DS4_DIST_DISABLE_WORKER_PREFETCH").as_deref())
}

pub fn prefetch_enabled_message(depth: u32) -> String {
    format!("ds4: distributed worker: receive prefetch depth {depth} enabled")
}

fn parse_strtol(s: &str) -> Option<i64> {
    let s = s.trim_start_matches(|c: char| c.is_ascii_whitespace());
    if s.is_empty() {
        return None;
    }
    let (neg, digits) = match s.as_bytes()[0] {
        b'+' => (false, &s[1..]),
        b'-' => (true, &s[1..]),
        _ => (false, s),
    };
    if digits.is_empty() || !digits.as_bytes()[0].is_ascii_digit() {
        return None;
    }
    if !digits.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    let v = digits.parse::<i64>().ok()?;
    Some(if neg { -v } else { v })
}

struct QueueInner<T> {
    depth: u32,
    queued: u32,
    jobs: VecDeque<T>,
    closed: bool,
    canceled: bool,
}

pub struct JobQueue<T = Vec<u8>> {
    inner: Mutex<QueueInner<T>>,
    not_empty: Condvar,
    not_full: Condvar,
}

impl<T> JobQueue<T> {
    pub fn new(depth: u32) -> Self {
        let depth = depth.max(1);
        Self {
            inner: Mutex::new(QueueInner {
                depth,
                queued: 0,
                jobs: VecDeque::new(),
                closed: false,
                canceled: false,
            }),
            not_empty: Condvar::new(),
            not_full: Condvar::new(),
        }
    }

    pub fn depth(&self) -> u32 {
        self.inner.lock().expect("prefetch queue").depth
    }

    pub fn queued(&self) -> u32 {
        self.inner.lock().expect("prefetch queue").queued
    }

    pub fn enqueue(&self, job: T) -> bool {
        let mut g = self.inner.lock().expect("prefetch queue");
        while !g.closed && !g.canceled && g.queued >= g.depth {
            g = self.not_full.wait(g).expect("prefetch queue");
        }
        if g.closed || g.canceled {
            return false;
        }
        g.jobs.push_back(job);
        g.queued += 1;
        self.not_empty.notify_one();
        true
    }

    pub fn wait_until_queued(&self, n: u32) {
        let mut g = self.inner.lock().expect("prefetch queue");
        while g.queued < n && !g.closed && !g.canceled {
            g = self.not_empty.wait(g).expect("prefetch queue");
        }
    }

    fn take_job(g: &mut QueueInner<T>, not_full: &Condvar) -> Option<T> {
        if g.canceled || g.jobs.is_empty() {
            return None;
        }
        let job = g.jobs.pop_front();
        g.queued = g.queued.saturating_sub(1);
        not_full.notify_one();
        job
    }

    pub fn try_pop(&self) -> Option<T> {
        let mut g = self.inner.lock().expect("prefetch queue");
        Self::take_job(&mut g, &self.not_full)
    }

    pub fn pop(&self) -> Option<T> {
        let mut g = self.inner.lock().expect("prefetch queue");
        while g.jobs.is_empty() && !g.closed && !g.canceled {
            g = self.not_empty.wait(g).expect("prefetch queue");
        }
        Self::take_job(&mut g, &self.not_full)
    }

    pub(crate) fn drained(&self) -> bool {
        let g = self.inner.lock().expect("prefetch queue");
        g.closed && g.jobs.is_empty()
    }

    pub fn finish(&self) {
        let mut g = self.inner.lock().expect("prefetch queue");
        g.closed = true;
        self.not_empty.notify_all();
        self.not_full.notify_all();
    }

    pub fn cancel(&self) {
        let mut g = self.inner.lock().expect("prefetch queue");
        g.closed = true;
        g.canceled = true;
        g.jobs.clear();
        g.queued = 0;
        self.not_empty.notify_all();
        self.not_full.notify_all();
    }
}

pub(crate) struct MutexWrite<'a>(pub &'a Mutex<TcpStream>);

impl Write for MutexWrite<'_> {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        self.0.lock().expect("worker write").write(buf)
    }

    fn flush(&mut self) -> io::Result<()> {
        self.0.lock().expect("worker write").flush()
    }
}

impl<E: SliceExec + Send, S: SnapshotStore + Send> Worker<E, S> {
    pub fn serve_prefetch(&mut self, stream: &mut TcpStream) -> io::Result<()> {
        let depth = prefetch_depth();
        let queue = Arc::new(JobQueue::<Vec<u8>>::new(depth));
        let worker = Mutex::new(&mut *self);
        let writer = Arc::new(Mutex::new(stream.try_clone()?));
        let mut reader = stream.try_clone()?;
        worker
            .lock()
            .expect("worker hops")
            .bind_hops(Arc::clone(&writer));
        eprintln!("{}", prefetch_enabled_message(depth));

        let rc = thread::scope(|scope| {
            let eval = scope.spawn(|| -> io::Result<()> {
                while let Some(payload) = queue.pop() {
                    let reply = {
                        let mut w = worker.lock().expect("worker eval");
                        w.process_work(&payload, &mut MutexWrite(writer.as_ref()))?
                    };
                    if let Some(frame) = reply {
                        writer.lock().expect("worker write").write_all(&frame)?;
                    }
                }
                Ok(())
            });

            let mut loop_err: Option<io::Error> = None;
            loop {
                let (typ, body) = match read_frame(&mut reader) {
                    Ok(v) => v,
                    Err(e) if e.kind() == io::ErrorKind::UnexpectedEof => break,
                    Err(e) => {
                        loop_err = Some(e);
                        break;
                    }
                };
                if typ == MSG_ERROR {
                    loop_err = Some(io::Error::new(
                        io::ErrorKind::Other,
                        format!("coordinator error: {}", String::from_utf8_lossy(&body)),
                    ));
                    break;
                }
                if typ == MSG_WORK {
                    if !queue.enqueue(body) {
                        loop_err = Some(io::Error::new(
                            io::ErrorKind::Interrupted,
                            "prefetch queue canceled",
                        ));
                        break;
                    }
                    continue;
                }
                if typ == MSG_SNAPSHOT_SAVE_REQ || typ == MSG_SNAPSHOT_LOAD_BEGIN {
                    let mut w = worker.lock().expect("worker snapshot");
                    let mut out = writer.lock().expect("worker snapshot");
                    w.apply_snapshot(&mut *out, typ, &body)?;
                    continue;
                }
                {
                    let mut out = writer.lock().expect("worker write");
                    write_frame(
                        &mut *out,
                        MSG_ERROR,
                        b"unsupported distributed worker frame",
                    )?;
                }
                loop_err = Some(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("rejected unsupported frame type {typ}"),
                ));
                break;
            }

            if loop_err.is_some() {
                queue.cancel();
            } else {
                queue.finish();
            }
            let eval_rc = eval.join().expect("prefetch eval");
            match (loop_err, eval_rc) {
                (Some(e), _) => Err(e),
                (None, Err(e)) => Err(e),
                (None, Ok(())) => Ok(()),
            }
        });
        self.shutdown_hops();
        rc
    }
}
