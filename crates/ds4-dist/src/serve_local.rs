//! Same-thread coordinator + hop mux. Accept thread only accepts.

use std::io::{self, ErrorKind};
use std::net::{SocketAddr, TcpListener, TcpStream};
use std::sync::mpsc::{self, Receiver};
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};
use std::time::Duration;

use crate::coordinator::accept_data_client;
use crate::exec::SliceExec;
use crate::native_snapshot::SnapshotStore;
use crate::transport::FrameReader;
use crate::worker::Worker;

pub(crate) const IDLE: Duration = Duration::from_millis(25);

pub(crate) struct HopAccept {
    rx: Option<Receiver<TcpStream>>,
    wakeup: SocketAddr,
    handle: Option<JoinHandle<()>>,
}

impl HopAccept {
    pub(crate) fn receiver(&self) -> Option<&Receiver<TcpStream>> {
        self.rx.as_ref()
    }
}

impl Drop for HopAccept {
    fn drop(&mut self) {
        self.rx.take();
        let _ = TcpStream::connect(self.wakeup);
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
    }
}

pub(crate) fn spawn_hop_accept(listener: &TcpListener) -> io::Result<HopAccept> {
    let listener = listener.try_clone()?;
    let wakeup = listener.local_addr()?;
    let (tx, rx) = mpsc::channel();
    let handle = thread::Builder::new()
        .name("ds4-dist-hop-accept".into())
        .spawn(move || loop {
            match accept_data_client(&listener) {
                Ok(stream) => {
                    if tx.send(stream).is_err() {
                        break;
                    }
                }
                Err(_) => break,
            }
        })?;
    Ok(HopAccept {
        rx: Some(rx),
        wakeup,
        handle: Some(handle),
    })
}

pub(crate) enum Drive {
    Continue,
    Closed,
    Failed(io::Error),
}

pub(crate) fn drive_ready<E, S>(
    worker: &mut Worker<E, S>,
    stream: &mut TcpStream,
    frames: &mut FrameReader,
) -> Drive
where
    E: SliceExec,
    S: SnapshotStore,
{
    let result = frames.read(stream).and_then(|(typ, body)| {
        if typ == crate::codec::MSG_SNAPSHOT_LOAD_BEGIN {
            // IDLE polls idle connections; a snapshot owns its stream until completion.
            let timeout = stream.read_timeout()?;
            stream.set_read_timeout(None)?;
            let result = worker.handle_frame(stream, typ, &body);
            stream.set_read_timeout(timeout)?;
            result
        } else {
            worker.handle_frame(stream, typ, &body)
        }
    });
    match result {
        Ok(()) => Drive::Continue,
        Err(e) => match e.kind() {
            ErrorKind::TimedOut | ErrorKind::WouldBlock | ErrorKind::Interrupted => Drive::Continue,
            ErrorKind::UnexpectedEof | ErrorKind::InvalidData => Drive::Closed,
            _ => Drive::Failed(e),
        },
    }
}

pub(crate) fn serve_coordinator_and_hops<E, S, Stop>(
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
    let mut accepted = Vec::new();
    let mut frames = FrameReader::default();
    let rc = loop {
        if should_stop() {
            break Ok(());
        }
        if let Some(rx) = hops {
            while let Ok(hop) = rx.try_recv() {
                let _ = hop.set_nodelay(true);
                let _ = hop.set_read_timeout(Some(IDLE));
                accepted.push((hop, FrameReader::default()));
            }
        }
        let mut i = 0;
        while i < accepted.len() {
            if let Ok(clone) = accepted[i].0.try_clone() {
                worker.bind_hops(Arc::new(Mutex::new(clone)));
            }
            let (stream, reader) = &mut accepted[i];
            match drive_ready(worker, stream, reader) {
                Drive::Continue => i += 1,
                Drive::Closed | Drive::Failed(_) => {
                    accepted.swap_remove(i);
                }
            }
        }
        match drive_ready(worker, coordinator, &mut frames) {
            Drive::Continue => {}
            Drive::Closed => break Ok(()),
            Drive::Failed(e) => break Err(e),
        }
    };
    worker.shutdown_hops();
    rc
}
