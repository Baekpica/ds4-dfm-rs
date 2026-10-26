//! Session worker entry and compatibility helpers.

use std::cell::RefCell;
use std::rc::Rc;

use crate::session_exec::{SessionSliceExec, SliceMeta};
use crate::session_snapshot::SessionSnapshotStore;
use ds4_core::Session;

pub use ds4_dist::{
    print_worker_listen_banner, worker_listen_banner, worker_listen_port, worker_plan,
    WorkerListenBanner, WorkerPlan,
};
pub use ds4_server::run_assembled_worker;

pub const WORKER_USE_PREFETCH: bool = false;

pub type AssembledWorker<'m> =
    ds4_dist::AssembledWorker<SessionSliceExec<'m>, SessionSnapshotStore<'m>>;

pub fn assemble_worker<'m>(
    session: Rc<RefCell<Session<'m>>>,
    meta: SliceMeta,
    quant_bits: u32,
    listen_port: u32,
    model_name: &str,
) -> AssembledWorker<'m> {
    let plan = worker_plan(&meta, quant_bits, listen_port, model_name);
    let exec = SessionSliceExec::from_meta(Rc::clone(&session), meta);
    let store = SessionSnapshotStore::from_shared(session, meta.layer_start, meta.layer_end);
    ds4_dist::assemble_worker(exec, store, plan.hello, plan.model_name)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::session_exec::slice_meta;
    use ds4_core::SHAPE_FLASH;
    use ds4_dist::{Layers, Worker};

    #[test]
    fn worker_plan_uses_bound_data_listen_port() {
        let layers = Layers {
            start: 20,
            end: 20,
            has_output: true,
            set: true,
        };
        let meta = slice_meta(7, &SHAPE_FLASH, 4096, &layers);
        let (_listener, port) = ds4_dist::open_data_listener(Some("127.0.0.1"), 0).unwrap();
        let plan = worker_plan(&meta, 2, u32::from(port), SHAPE_FLASH.name);
        assert_eq!(plan.hello.listen_port, u32::from(port));
        assert_ne!(plan.hello.listen_port, 0);
    }

    #[test]
    fn assemble_worker_compiles_against_reconnect_local() {
        type SessionWorker<'m> = Worker<SessionSliceExec<'m>, SessionSnapshotStore<'m>>;
        fn bind<'m, C, Sl, St>(
            worker: &mut SessionWorker<'m>,
            spec: ds4_dist::LocalReconnect<'_, C, Sl, St>,
        ) -> std::io::Result<()>
        where
            C: FnMut() -> std::io::Result<std::net::TcpStream>,
            Sl: FnMut(),
            St: FnMut() -> bool,
        {
            ds4_dist::reconnect_local(worker, spec)
        }
        let _ = bind::<fn() -> std::io::Result<std::net::TcpStream>, fn(), fn() -> bool>;
    }

    fn assert_not_send<T: ?Sized>() {
        struct Probe<U: ?Sized>(std::marker::PhantomData<U>);
        trait OnlyIfNotSend<A> {
            fn ok() {}
        }
        impl<U: ?Sized> OnlyIfNotSend<()> for Probe<U> {}
        impl<U: ?Sized + Send> OnlyIfNotSend<bool> for Probe<U> {}
        let _ = <Probe<T> as OnlyIfNotSend<_>>::ok;
    }

    #[test]
    fn session_slice_exec_cannot_bind_send_reconnect_with() {
        // Given: worker types wrap !Send Session (Rc + Session::_not_send)
        // When: the Send reconnect path requires E: Send + S: Send
        // Then: those types do not implement Send, and the worker entry keeps prefetch off
        assert_not_send::<SessionSliceExec<'static>>();
        assert_not_send::<SessionSnapshotStore<'static>>();
        assert_not_send::<Session<'static>>();
        assert!(!WORKER_USE_PREFETCH);
    }

    #[test]
    fn worker_plan_matches_hello_and_layer_range() {
        let layers = Layers {
            start: 20,
            end: 20,
            has_output: true,
            set: true,
        };
        let meta = slice_meta(7, &SHAPE_FLASH, 4096, &layers);
        let plan = worker_plan(&meta, 2, 7100, SHAPE_FLASH.name);
        assert_eq!(plan.hello.model_id, 7);
        assert_eq!(plan.hello.quant_bits, 2);
        assert_eq!(plan.hello.layer_start, meta.layer_start);
        assert_eq!(plan.hello.layer_end, meta.layer_end);
        assert_eq!(plan.hello.has_output, 1);
        assert_eq!(plan.hello.has_hidden, 1);
        assert_eq!(plan.hello.listen_port, 7100);
        assert_eq!(plan.model_name, SHAPE_FLASH.name);
        assert_eq!(plan.hello.model_name_len, SHAPE_FLASH.name.len() as u32);
    }

    #[test]
    fn worker_listen_banner_contains_data_listen_and_coordinator() {
        // Given: known worker meta (C layers 20:output, model_id 7)
        let layers = Layers {
            start: 20,
            end: 20,
            has_output: true,
            set: true,
        };
        let meta = slice_meta(7, &SHAPE_FLASH, 4096, &layers);
        let plan = worker_plan(&meta, 2, 7100, SHAPE_FLASH.name);

        // When: format the C boot stderr line
        let line = worker_listen_banner(&WorkerListenBanner {
            layer_start: plan.hello.layer_start,
            has_output: plan.hello.has_output != 0,
            layer_end: plan.hello.layer_end,
            model_id: i32::try_from(plan.hello.model_id).expect("model_id fits i32"),
            listen_host: Some("127.0.0.1"),
            listen_port: plan.hello.listen_port,
            coordinator_host: "10.0.0.1",
            coordinator_port: 1234,
        });

        // Then: C tokens and full fprintf shape
        assert!(
            line.contains("data_listen="),
            "banner must contain data_listen=: {line}"
        );
        assert!(
            line.contains("connecting to coordinator"),
            "banner must contain connecting to coordinator: {line}"
        );
        assert_eq!(
            line,
            "ds4: distributed worker: layers 20:output model_id=7 data_listen=127.0.0.1:7100 connecting to coordinator 10.0.0.1:1234"
        );
        assert_eq!(
            ds4_dist::connected_message("10.0.0.1", "1234"),
            "ds4: distributed worker: connected to coordinator 10.0.0.1:1234"
        );
        assert_eq!(
            ds4_dist::disconnected_message(false),
            "ds4: distributed worker: coordinator disconnected; reconnecting"
        );
    }

    #[test]
    fn worker_listen_banner_prints_star_when_host_missing() {
        // Given: no listen host (C listen_host == NULL)
        let spec = WorkerListenBanner {
            layer_start: 20,
            has_output: false,
            layer_end: 30,
            model_id: 7,
            listen_host: None,
            listen_port: 7100,
            coordinator_host: "10.0.0.1",
            coordinator_port: 1234,
        };

        // When: format the C boot stderr line
        let line = worker_listen_banner(&spec);

        // Then: missing host is `*` like C
        assert!(
            line.contains("data_listen=*:7100"),
            "missing host must print *: {line}"
        );
        assert!(
            line.contains("connecting to coordinator"),
            "banner must contain connecting to coordinator: {line}"
        );
        assert_eq!(
            line,
            "ds4: distributed worker: layers 20:30 model_id=7 data_listen=*:7100 connecting to coordinator 10.0.0.1:1234"
        );
    }
}
