//! !Send receive-prefetch: queue on a helper thread, eval on the session thread.

use std::collections::HashMap;
use std::io::{self, ErrorKind, Write};
use std::net::{Shutdown, TcpStream};
use std::sync::mpsc::{self, Receiver};
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};

use crate::codec::{MSG_ERROR, MSG_SNAPSHOT_LOAD_BEGIN, MSG_SNAPSHOT_SAVE_REQ, MSG_WORK};
use crate::exec::SliceExec;
use crate::native_snapshot::SnapshotStore;
use crate::prefetch::{prefetch_depth, prefetch_enabled_message, JobQueue, MutexWrite};
use crate::serve_local::{drive_ready, Drive, IDLE};
use crate::transport::{write_frame, FrameReader};
use crate::worker::Worker;

pub enum PrefetchJob {
    Work(Vec<u8>),
    Snapshot { typ: u32, body: Vec<u8> },
}

pub fn local_prefetch_enabled_from(env: Option<&std::ffi::OsStr>) -> bool {
    !crate::prefetch::prefetch_disabled_from(env)
}

pub fn local_prefetch_enabled() -> bool {
    local_prefetch_enabled_from(std::env::var_os("DS4_DIST_DISABLE_WORKER_PREFETCH").as_deref())
}

fn recv_prefetch(
    mut reader: TcpStream,
    queue: &JobQueue<PrefetchJob>,
    writer: &Mutex<TcpStream>,
    resume: Receiver<()>,
    mut on_job: impl FnMut(),
) -> Option<io::Error> {
    let mut frames = FrameReader::default();
    let err = loop {
        match frames.read(&mut reader) {
            Ok((typ, body)) if typ == MSG_WORK => {
                if !queue.enqueue(PrefetchJob::Work(body)) {
                    break Some(io::Error::new(
                        ErrorKind::Interrupted,
                        "prefetch queue canceled",
                    ));
                }
                on_job();
            }
            Ok((typ, body)) if typ == MSG_SNAPSHOT_SAVE_REQ || typ == MSG_SNAPSHOT_LOAD_BEGIN => {
                if !queue.enqueue(PrefetchJob::Snapshot { typ, body }) {
                    break Some(io::Error::new(
                        ErrorKind::Interrupted,
                        "prefetch queue canceled",
                    ));
                }
                on_job();
                // LOAD chunks belong to apply_snapshot, which runs on the session thread.
                if typ == MSG_SNAPSHOT_LOAD_BEGIN && resume.recv().is_err() {
                    break None;
                }
            }
            Ok((typ, body)) if typ == MSG_ERROR => {
                break Some(io::Error::new(
                    ErrorKind::Other,
                    format!("coordinator error: {}", String::from_utf8_lossy(&body)),
                ));
            }
            Ok((typ, _)) => {
                let _ = write_frame(
                    &mut *writer.lock().expect("worker write"),
                    MSG_ERROR,
                    b"unsupported distributed worker frame",
                );
                break Some(io::Error::new(
                    ErrorKind::InvalidData,
                    format!("rejected unsupported frame type {typ}"),
                ));
            }
            Err(e) if e.kind() == ErrorKind::UnexpectedEof => break None,
            Err(e)
                if matches!(
                    e.kind(),
                    ErrorKind::TimedOut | ErrorKind::WouldBlock | ErrorKind::Interrupted
                ) =>
            {
                continue;
            }
            Err(e) => break Some(e),
        }
    };
    if err.is_some() {
        queue.cancel();
    } else {
        queue.finish();
    }
    on_job();
    err
}

fn apply_prefetch_job<E, S>(
    worker: &mut Worker<E, S>,
    writer: &Arc<Mutex<TcpStream>>,
    resume: &mpsc::Sender<()>,
    job: PrefetchJob,
) -> io::Result<()>
where
    E: SliceExec,
    S: SnapshotStore,
{
    match job {
        PrefetchJob::Work(payload) => {
            worker.bind_hops(Arc::clone(writer));
            let reply = worker.process_work(&payload, &mut MutexWrite(writer))?;
            if let Some(frame) = reply {
                writer.lock().expect("worker write").write_all(&frame)?;
            }
            Ok(())
        }
        PrefetchJob::Snapshot { typ, body } => {
            let mut out = writer.lock().expect("worker snapshot");
            let result = worker.apply_snapshot(&mut *out, typ, &body);
            if typ == MSG_SNAPSHOT_LOAD_BEGIN {
                let _ = resume.send(());
            }
            result
        }
    }
}

pub fn serve_prefetch_local_with<E, S>(
    worker: &mut Worker<E, S>,
    stream: &mut TcpStream,
    queue: &JobQueue<PrefetchJob>,
) -> io::Result<()>
where
    E: SliceExec,
    S: SnapshotStore,
{
    let writer = Arc::new(Mutex::new(stream.try_clone()?));
    let reader = stream.try_clone()?;
    let (resume_tx, resume_rx) = mpsc::channel();
    worker.bind_hops(Arc::clone(&writer));
    eprintln!("{}", prefetch_enabled_message(queue.depth()));
    let rc = thread::scope(|scope| {
        let recv = scope.spawn(|| recv_prefetch(reader, queue, &writer, resume_rx, || {}));
        let mut eval_err = None;
        while let Some(job) = queue.pop() {
            if let Err(e) = apply_prefetch_job(worker, &writer, &resume_tx, job) {
                eval_err = Some(e);
                queue.cancel();
                break;
            }
        }
        drop(resume_tx);
        let _ = writer
            .lock()
            .expect("worker write")
            .shutdown(Shutdown::Both);
        match (recv.join().expect("prefetch recv"), eval_err) {
            (Some(e), _) => Err(e),
            (None, Some(e)) => Err(e),
            (None, None) => Ok(()),
        }
    });
    worker.shutdown_hops();
    rc
}

struct PrefetchConn {
    queue: Arc<JobQueue<PrefetchJob>>,
    writer: Arc<Mutex<TcpStream>>,
    recv: Option<JoinHandle<()>>,
    resume: Option<mpsc::Sender<()>>,
}

impl Drop for PrefetchConn {
    fn drop(&mut self) {
        self.queue.cancel();
        self.resume.take();
        let _ = self
            .writer
            .lock()
            .expect("worker write")
            .shutdown(Shutdown::Both);
        if let Some(handle) = self.recv.take() {
            let _ = handle.join();
        }
    }
}

fn start_prefetch_recv(
    stream: TcpStream,
    wake: mpsc::Sender<usize>,
    idx: usize,
) -> io::Result<PrefetchConn> {
    let _ = stream.set_nodelay(true);
    let writer = Arc::new(Mutex::new(stream.try_clone()?));
    let queue = Arc::new(JobQueue::<PrefetchJob>::new(prefetch_depth()));
    let (resume_tx, resume_rx) = mpsc::channel();
    let q = Arc::clone(&queue);
    let w = Arc::clone(&writer);
    eprintln!("{}", prefetch_enabled_message(queue.depth()));
    let recv = thread::Builder::new()
        .name("ds4-dist-prefetch-recv".into())
        .spawn(move || {
            let _ = recv_prefetch(stream, &q, &w, resume_rx, || {
                let _ = wake.send(idx);
            });
        })?;
    Ok(PrefetchConn {
        queue,
        writer,
        recv: Some(recv),
        resume: Some(resume_tx),
    })
}

fn drain_conn<E, S>(worker: &mut Worker<E, S>, conn: &PrefetchConn) -> io::Result<()>
where
    E: SliceExec,
    S: SnapshotStore,
{
    while let Some(job) = conn.queue.try_pop() {
        apply_prefetch_job(
            worker,
            &conn.writer,
            conn.resume.as_ref().expect("live receiver"),
            job,
        )?;
    }
    Ok(())
}

pub(crate) fn serve_coordinator_and_hops_prefetch<E, S, Stop>(
    worker: &mut Worker<E, S>,
    coordinator: &mut TcpStream,
    hops: Option<&Receiver<TcpStream>>,
    should_stop: &mut Stop,
) -> io::Result<()>
where
    E: SliceExec,
    S: SnapshotStore,
    Stop: FnMut() -> bool,
{
    coordinator.set_read_timeout(Some(IDLE))?;
    let (wake_tx, wake_rx) = mpsc::channel();
    let mut conns = HashMap::new();
    let mut next_id = 0usize;
    let mut frames = FrameReader::default();
    let rc = 'run: loop {
        if should_stop() {
            break Ok(());
        }
        if let Some(rx) = hops {
            while let Ok(hop) = rx.try_recv() {
                let idx = next_id;
                next_id = match next_id.checked_add(1) {
                    Some(id) => id,
                    None => break 'run Err(io::Error::other("connection id exhausted")),
                };
                match start_prefetch_recv(hop, wake_tx.clone(), idx) {
                    Ok(conn) => {
                        conns.insert(idx, conn);
                    }
                    Err(e) => break 'run Err(e),
                }
            }
        }
        while let Ok(idx) = wake_rx.try_recv() {
            if let Some(conn) = conns.get(&idx) {
                if let Err(e) = drain_conn(worker, conn) {
                    break 'run Err(e);
                }
                if conn.queue.drained() {
                    conns.remove(&idx);
                }
            }
        }
        match drive_ready(worker, coordinator, &mut frames) {
            Drive::Continue => {}
            Drive::Closed => break Ok(()),
            Drive::Failed(e) => break Err(e),
        }
    };
    drop(conns);
    worker.shutdown_hops();
    rc
}
