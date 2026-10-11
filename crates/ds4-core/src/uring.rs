//! io_uring channel for the engram row reads — the Rust mirror of
//! `core_v41_ering.c` (the C engine at /data/YoungAi, commit 3946dbc).
//!
//! Why a ring at all (`core_v41_ering.c:1-10`): a decode step reads 48 rows
//! of 264 B from the 203 GB table; the disk's own measure is 200 us for one
//! 4 KB O_DIRECT read but ~0.25 ms for 48 in flight, while the engine's
//! thread-pool round cost 2.1 ms (48 threads woken together, one lock).  One
//! submit fills the queue, the kernel does the concurrency, one wait drains
//! it — a round is about one disk latency.
//!
//! The shape being mirrored, piece by piece:
//!
//! - `v41_ering_open` (`core_v41_ering.c:39-69`): the raw `io_uring_setup`
//!   syscall (no liburing), SQ/CQ/SQE mmaps, one 2-block bounce slot per
//!   queue entry, and a free-slot stack so a slot belongs to exactly one
//!   in-flight request (`:82-88` — the strict-batch version it replaced was
//!   the slot-reuse race that corrupted rows out of completion order).
//! - `v41_ering_push` (`:89-112`): fill SQEs until the submission queue is
//!   full or the slots run out, then one `io_uring_enter`.
//! - `v41_ering_wait` (`:122-144`): GETEVENTS loop that re-fills the queue
//!   from returned slots while draining completions (the queue stays full);
//!   a direct read's bytes are memcpy'd out of the bounce slot on completion
//!   and the slot returns to the stack only after they are home.
//! - O_DIRECT alignment (`:96-108`): a 264-byte row at an arbitrary offset is
//!   read as its aligned superset (at most 2 blocks) into the bounce slot.
//!
//! Only Linux has io_uring; elsewhere `Ring::open` returns None and the
//! caller keeps the serial row reads.  (The engine's fallback is its thread
//! pool, `core_v41_ering.c:147-153`; the port has no pool, so its fallback is
//! the pre-unit-P serial loop — a named deviation, non-Linux only.)

use std::os::raw::c_long;
use std::ptr;
use std::sync::atomic::{AtomicU32, Ordering};

/// `V41_ERING_BLK` (`core_v41_ering.c:20`): the O_DIRECT alignment block.
const ERING_BLK: usize = 4096;
/// The engine's queue depth (`core_v41_engram.c:231`: `v41_ering_open(512u)`).
pub const ERING_CAP: u32 = 512;

/// `v41_ering_req` (`core_v41.h:260`): one read.  `direct` = the fd is
/// O_DIRECT, so the read goes through a bounce slot and is copied out.
#[derive(Clone, Copy)]
pub struct Req {
    pub fd: i32,
    pub direct: bool,
    pub len: u32,
    pub off: u64,
    pub dst: *mut u8,
}

#[cfg(target_os = "linux")]
mod imp {
    use super::*;

    // ---- linux/io_uring.h ABI (the header the engine includes directly,
    // core_v41_ering.c:16; the structs are stable uapi) ----

    const IORING_OP_READ: u8 = 22;
    const IORING_ENTER_GETEVENTS: u32 = 1;
    const IORING_OFF_SQ_RING: i64 = 0;
    const IORING_OFF_CQ_RING: i64 = 0x8000000;
    const IORING_OFF_SQES: i64 = 0x10000000;
    const IORING_FEAT_SINGLE_MMAP: u32 = 1;

    const PROT_READ: i32 = 1;
    const PROT_WRITE: i32 = 2;
    const MAP_SHARED: i32 = 1;
    const MAP_POPULATE: i32 = 0x8000;

    #[repr(C)]
    #[derive(Default)]
    struct SqringOffsets {
        head: u32,
        tail: u32,
        ring_mask: u32,
        ring_entries: u32,
        flags: u32,
        dropped: u32,
        array: u32,
        resv1: u32,
        resv2: u64,
    }

    #[repr(C)]
    #[derive(Default)]
    struct CqringOffsets {
        head: u32,
        tail: u32,
        ring_mask: u32,
        ring_entries: u32,
        overflow: u32,
        cqes: u32,
        flags: u32,
        resv1: u32,
        resv2: u64,
    }

    #[repr(C)]
    #[derive(Default)]
    struct IoUringParams {
        sq_entries: u32,
        cq_entries: u32,
        flags: u32,
        sq_thread_cpu: u32,
        sq_thread_idle: u32,
        features: u32,
        wq_fd: u32,
        resv: [u32; 3],
        sq_off: SqringOffsets,
        cq_off: CqringOffsets,
    }

    /// `struct io_uring_sqe` (64 bytes).  Only the fields the ring fills are
    /// named; the rest is zeroed exactly like the engine's `memset(s, 0)`.
    #[repr(C)]
    #[derive(Default)]
    struct Sqe {
        opcode: u8,
        flags: u8,
        ioprio: u16,
        fd: i32,
        off: u64,
        addr: u64,
        len: u32,
        rw_flags: u32,
        user_data: u64,
        buf_index: u16,
        personality: u16,
        splice_fd_in: i32,
        pad2: [u64; 2],
    }

    /// `struct io_uring_cqe`.
    #[repr(C)]
    #[derive(Default)]
    struct Cqe {
        user_data: u64,
        res: i32,
        flags: u32,
    }

    extern "C" {
        fn syscall(num: c_long, ...) -> c_long;
        fn mmap(
            addr: *mut std::ffi::c_void,
            len: usize,
            prot: i32,
            flags: i32,
            fd: i32,
            off: i64,
        ) -> *mut std::ffi::c_void;
        fn munmap(addr: *mut std::ffi::c_void, len: usize) -> i32;
        fn close(fd: i32) -> i32;
    }

    fn mmap_fd(fd: i32, len: usize, off: i64) -> Option<*mut u8> {
        let p = unsafe {
            mmap(
                ptr::null_mut(),
                len,
                PROT_READ | PROT_WRITE,
                MAP_SHARED | MAP_POPULATE,
                fd,
                off,
            )
        };
        if p as isize == -1 {
            return None;
        }
        Some(p as *mut u8)
    }

    pub struct Ring {
        fd: i32,
        cap: u32,
        sq_map: *mut u8,
        sq_map_sz: usize,
        cq_map: *mut u8,
        cq_map_sz: usize,
        single_map: bool,
        sqes: *mut Sqe,
        sqes_sz: usize,
        sq_head: *const AtomicU32,
        sq_tail: *const AtomicU32,
        sq_mask: u32,
        sq_array: *mut u32,
        cq_head: *const AtomicU32,
        cq_tail: *const AtomicU32,
        cq_mask: u32,
        cqes: *mut Cqe,
        bounce: *mut u8,
        bounce_sz: usize,
        free_slot: Vec<u32>,
        nfree: usize,
        /// The round's requests, resident like the engine's `J->reqs`
        /// (`core_v41_engram.c:232`); filled by the caller, consumed by
        /// `push`/`wait` through `submitted`.
        reqs: Vec<Req>,
        submitted: u32,
        done: u32,
        inflight: u32,
        err: bool,
    }

    // The ring is a single-owner object (the engine's ejob is too): raw
    // pointers into its own mmaps, used from the thread that runs the round.
    // P2: the owner is the feed's `FeedCore` mutex — the calling thread runs
    // the small (decode) rounds and the fetch worker runs the oversize
    // (prefill) rounds (`core_v41_engram.c:128-150`), never both at once.
    // `Send` is the claim that the mutex provides exactly that exclusivity.
    unsafe impl Send for Ring {}

    impl Ring {
        /// `v41_ering_open` (`core_v41_ering.c:39-69`).  Any failure returns
        /// None — the caller falls back; the engine refuses the same way.
        pub fn open(cap: u32) -> Option<Ring> {
            let mut p = IoUringParams::default();
            let fd = unsafe {
                syscall(
                    libc::SYS_io_uring_setup as c_long,
                    cap as c_long,
                    &mut p as *mut IoUringParams as c_long,
                )
            } as i32;
            if fd < 0 {
                return None;
            }
            let cap = p.sq_entries;
            let mut sq_map_sz = p.sq_off.array as usize + p.sq_entries as usize * 4;
            let cq_map_sz = p.cq_off.cqes as usize + p.cq_entries as usize * 16;
            let single_map = p.features & IORING_FEAT_SINGLE_MMAP != 0;
            if single_map && cq_map_sz > sq_map_sz {
                sq_map_sz = cq_map_sz;
            }
            let Some(sq_map) = mmap_fd(fd, sq_map_sz, IORING_OFF_SQ_RING) else {
                unsafe { close(fd) };
                return None;
            };
            let (cq_map, cq_map_sz) = if single_map {
                (sq_map, sq_map_sz)
            } else {
                let Some(m) = mmap_fd(fd, cq_map_sz, IORING_OFF_CQ_RING) else {
                    unsafe {
                        munmap(sq_map as *mut _, sq_map_sz);
                        close(fd);
                    }
                    return None;
                };
                (m, cq_map_sz)
            };
            let sqes_sz = p.sq_entries as usize * std::mem::size_of::<Sqe>();
            let Some(sqes) = mmap_fd(fd, sqes_sz, IORING_OFF_SQES) else {
                unsafe {
                    if !single_map {
                        munmap(cq_map as *mut _, cq_map_sz);
                    }
                    munmap(sq_map as *mut _, sq_map_sz);
                    close(fd);
                }
                return None;
            };

            let (sq_head, sq_tail, sq_mask, sq_array) = unsafe {
                (
                    sq_map.add(p.sq_off.head as usize) as *const AtomicU32,
                    sq_map.add(p.sq_off.tail as usize) as *const AtomicU32,
                    *(sq_map.add(p.sq_off.ring_mask as usize) as *const u32),
                    sq_map.add(p.sq_off.array as usize) as *mut u32,
                )
            };
            let (cq_head, cq_tail, cq_mask, cqes) = unsafe {
                (
                    cq_map.add(p.cq_off.head as usize) as *const AtomicU32,
                    cq_map.add(p.cq_off.tail as usize) as *const AtomicU32,
                    *(cq_map.add(p.cq_off.ring_mask as usize) as *const u32),
                    cq_map.add(p.cq_off.cqes as usize) as *mut Cqe,
                )
            };

            // `posix_memalign(V41_ERING_BLK, cap * 2 * V41_ERING_BLK)`
            // (core_v41_ering.c:64): one bounce block pair per slot.
            let bounce_sz = cap as usize * 2 * ERING_BLK;
            let layout = std::alloc::Layout::from_size_align(bounce_sz, ERING_BLK).ok()?;
            let bounce = unsafe { std::alloc::alloc(layout) };
            if bounce.is_null() {
                unsafe {
                    munmap(sqes as *mut _, sqes_sz);
                    if !single_map {
                        munmap(cq_map as *mut _, cq_map_sz);
                    }
                    munmap(sq_map as *mut _, sq_map_sz);
                    close(fd);
                }
                return None;
            }

            let mut free_slot: Vec<u32> = (0..cap).map(|i| cap - 1 - i).collect(); // :66, stack top = slot 0
            free_slot.shrink_to_fit();
            let nfree = cap as usize;
            Some(Ring {
                fd,
                cap,
                sq_map,
                sq_map_sz,
                cq_map,
                cq_map_sz,
                single_map,
                sqes: sqes as *mut Sqe,
                sqes_sz,
                sq_head,
                sq_tail,
                sq_mask,
                sq_array,
                cq_head,
                cq_tail,
                cq_mask,
                cqes,
                bounce,
                bounce_sz,
                free_slot,
                nfree,
                reqs: Vec::new(),
                submitted: 0,
                done: 0,
                inflight: 0,
                err: false,
            })
        }

        pub fn cap(&self) -> u32 {
            self.cap
        }

        /// The resident request buffer (`J->reqs`); fill, then `submit`.
        pub fn reqs_mut(&mut self) -> &mut Vec<Req> {
            &mut self.reqs
        }

        /// `v41_ering_submit` (`core_v41_ering.c:114-118`): a submit while a
        /// previous round is still in flight is a caller-order bug, refused.
        pub fn submit(&mut self) -> bool {
            if self.inflight != 0 {
                return false;
            }
            self.submitted = 0;
            self.done = 0;
            self.err = false;
            self.push()
        }

        /// `v41_ering_push` (`core_v41_ering.c:89-112`).
        fn push(&mut self) -> bool {
            let sq_tail = unsafe { &*self.sq_tail };
            let mut tail = sq_tail.load(Ordering::Relaxed);
            let mut pushed = 0u32;
            while (self.submitted as usize) < self.reqs.len() && self.nfree > 0 {
                // The submission ring has cap slots; in-flight <= cap and each
                // in-flight request held one, which the kernel read before
                // `enter` returned — checked against head, not inferred (:92).
                let head = unsafe { (*self.sq_head).load(Ordering::Acquire) };
                if tail.wrapping_sub(head) >= self.cap {
                    break;
                }
                let i = self.submitted as usize;
                self.nfree -= 1;
                let slot = self.free_slot[self.nfree];
                let q = self.reqs[i];
                let s = unsafe { &mut *self.sqes.add((tail & self.sq_mask) as usize) };
                *s = Sqe::default();
                s.opcode = IORING_OP_READ;
                s.fd = q.fd;
                s.user_data = ((slot as u64) << 32) | i as u64; // :98, the completion finds slot + request
                if q.direct {
                    let a = q.off & !(ERING_BLK as u64 - 1);
                    let span = q.off + q.len as u64 - a;
                    s.addr = (unsafe { self.bounce.add(slot as usize * 2 * ERING_BLK) }) as u64;
                    s.len = span.div_ceil(ERING_BLK as u64) as u32 * ERING_BLK as u32;
                    s.off = a;
                } else {
                    s.addr = q.dst as u64;
                    s.len = q.len;
                    s.off = q.off;
                }
                unsafe { *self.sq_array.add((tail & self.sq_mask) as usize) = tail & self.sq_mask };
                tail = tail.wrapping_add(1);
                pushed += 1;
                self.submitted += 1;
                self.inflight += 1;
            }
            if pushed == 0 {
                return true;
            }
            unsafe { (*self.sq_tail).store(tail, Ordering::Release) };
            let rc = unsafe {
                syscall(
                    libc::SYS_io_uring_enter as c_long,
                    self.fd as c_long,
                    pushed as c_long,
                    0,
                    0,
                    ptr::null::<std::ffi::c_void>(),
                    0,
                )
            };
            rc >= 0
        }

        /// `v41_ering_wait` (`core_v41_ering.c:122-144`): drain completions,
        /// keep the queue full, copy direct reads out of their slots.
        pub fn wait(&mut self) -> bool {
            while self.done < self.reqs.len() as u32 {
                if self.submitted < self.reqs.len() as u32 && self.nfree > 0 && !self.push() {
                    self.err = true;
                    break;
                }
                if self.inflight == 0 {
                    // Nothing in flight and nothing pushable = the slot
                    // accounting is broken; never spin forever (:126).
                    self.err = true;
                    break;
                }
                let rc = unsafe {
                    syscall(
                        libc::SYS_io_uring_enter as c_long,
                        self.fd as c_long,
                        0,
                        1,
                        IORING_ENTER_GETEVENTS,
                        ptr::null::<std::ffi::c_void>(),
                        0,
                    )
                };
                if rc < 0 {
                    if std::io::Error::last_os_error().kind() == std::io::ErrorKind::Interrupted {
                        continue;
                    }
                    self.err = true;
                    break;
                }
                let cq_head = unsafe { &*self.cq_head };
                let cq_tail = unsafe { &*self.cq_tail };
                let mut head = cq_head.load(Ordering::Relaxed);
                let tail = cq_tail.load(Ordering::Acquire);
                while head != tail {
                    let c = unsafe { &*self.cqes.add((head & self.cq_mask) as usize) };
                    let i = (c.user_data & 0xffff_ffff) as usize;
                    let slot = (c.user_data >> 32) as usize;
                    if i >= self.reqs.len() || slot >= self.cap as usize {
                        self.err = true; // not this round's request; never happens (:131)
                        head = head.wrapping_add(1);
                        continue;
                    }
                    let q = self.reqs[i];
                    if q.direct {
                        let a = q.off & !(ERING_BLK as u64 - 1);
                        let need = q.off + q.len as u64 - a;
                        if c.res < 0 || (c.res as u64) < need {
                            self.err = true;
                        } else {
                            unsafe {
                                ptr::copy_nonoverlapping(
                                    self.bounce.add(slot * 2 * ERING_BLK + (q.off - a) as usize),
                                    q.dst,
                                    q.len as usize,
                                );
                            }
                        }
                    } else if c.res != q.len as i32 {
                        self.err = true;
                    }
                    self.free_slot[self.nfree] = slot as u32; // the row is home; the slot returns
                    self.nfree += 1;
                    head = head.wrapping_add(1);
                    self.done += 1;
                    self.inflight -= 1;
                }
                cq_head.store(head, Ordering::Release);
            }
            !self.err
        }
    }

    impl Drop for Ring {
        fn drop(&mut self) {
            unsafe {
                munmap(self.sqes as *mut _, self.sqes_sz);
                if !self.single_map {
                    munmap(self.cq_map as *mut _, self.cq_map_sz);
                }
                munmap(self.sq_map as *mut _, self.sq_map_sz);
                close(self.fd);
                let layout =
                    std::alloc::Layout::from_size_align(self.bounce_sz, ERING_BLK).unwrap();
                std::alloc::dealloc(self.bounce, layout);
            }
        }
    }
}

#[cfg(not(target_os = "linux"))]
mod imp {
    use super::*;

    /// Non-Linux: no io_uring; the caller keeps the serial row reads.
    pub struct Ring;

    impl Ring {
        pub fn open(_cap: u32) -> Option<Ring> {
            None
        }
        pub fn cap(&self) -> u32 {
            0
        }
        pub fn reqs_mut(&mut self) -> &mut Vec<Req> {
            unreachable!()
        }
        pub fn submit(&mut self) -> bool {
            false
        }
        pub fn wait(&mut self) -> bool {
            false
        }
    }
}

pub use imp::Ring;

#[cfg(all(test, target_os = "linux"))]
mod tests {
    use super::*;
    use std::os::unix::fs::FileExt;
    use std::os::unix::io::AsRawFd;

    /// A deterministic byte pattern; the file is 1 MiB so offsets can wander.
    fn pattern(len: usize) -> Vec<u8> {
        let mut v = Vec::with_capacity(len);
        let mut x: u32 = 0x9e37_79b9;
        for _ in 0..len {
            x = x.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            v.push((x >> 24) as u8);
        }
        v
    }

    fn lcg(state: &mut u64) -> u64 {
        *state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        *state >> 33
    }

    /// The ring must deliver exactly the bytes a plain read delivers, for
    /// unaligned offsets and lengths (the O_DIRECT aligned-superset path) and
    /// for more requests than the queue depth (the push/drain refill loop).
    #[test]
    fn ring_reads_match_plain_reads() {
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("target/uring-test");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join(format!("pattern-{}.bin", std::process::id()));
        let data = pattern(1 << 20);
        std::fs::write(&path, &data).unwrap();

        let plain = std::fs::File::open(&path).unwrap();
        let mut direct = None;
        {
            use std::os::unix::fs::OpenOptionsExt;
            // The aarch64/x86_64 O_DIRECT values the engram shard open uses
            // (engram.rs:216-226); the file may live on a filesystem that
            // refuses it, in which case the direct leg is skipped loudly.
            #[cfg(target_arch = "aarch64")]
            const O_DIRECT: i32 = 0o200000;
            #[cfg(not(target_arch = "aarch64"))]
            const O_DIRECT: i32 = 0o40000;
            if let Ok(f) = std::fs::OpenOptions::new()
                .read(true)
                .custom_flags(O_DIRECT)
                .open(&path)
            {
                direct = Some(f);
            } else {
                eprintln!("uring test: O_DIRECT refused here; direct leg skipped");
            }
        }

        let cap = 8u32;
        let mut ring = Ring::open(cap).expect("io_uring unavailable");
        assert_eq!(ring.cap(), cap);

        for direct_leg in [false, true] {
            let Some(file) = (if direct_leg {
                direct.as_ref()
            } else {
                Some(&plain)
            }) else {
                continue;
            };
            let fd = file.as_raw_fd();

            // 100 requests over the 1 MiB file: unaligned offsets, row-like
            // and odd lengths, far more than the 8-deep queue.
            let mut state = 0x1234_5678_9abc_def0u64 ^ direct_leg as u64;
            let mut offs: Vec<u64> = Vec::new();
            let mut lens: Vec<u32> = Vec::new();
            for _ in 0..100 {
                offs.push(lcg(&mut state) % (data.len() as u64 - 8192));
                lens.push(1 + (lcg(&mut state) % 4096) as u32);
            }
            let mut want: Vec<Vec<u8>> = Vec::new();
            let mut got: Vec<Vec<u8>> = Vec::new();
            for i in 0..100 {
                let mut expect = vec![0u8; lens[i] as usize];
                plain.read_exact_at(&mut expect, offs[i]).unwrap();
                want.push(expect);
                got.push(vec![0u8; lens[i] as usize]);
            }
            {
                let reqs = ring.reqs_mut();
                reqs.clear();
                for i in 0..100 {
                    reqs.push(Req {
                        fd,
                        direct: direct_leg,
                        len: lens[i],
                        off: offs[i],
                        dst: got[i].as_mut_ptr(),
                    });
                }
            }
            assert!(ring.submit(), "submit failed (direct={direct_leg})");
            assert!(ring.wait(), "wait failed (direct={direct_leg})");
            for (i, expect) in want.iter().enumerate() {
                assert_eq!(&got[i], expect, "request {i} (direct={direct_leg})");
            }
        }

        // A submit while the previous round is in flight is refused
        // (core_v41_ering.c:115): queue one small round, then try a second.
        let mut buf = vec![0u8; 16];
        {
            let reqs = ring.reqs_mut();
            reqs.clear();
            reqs.push(Req {
                fd: plain.as_raw_fd(),
                direct: false,
                len: 16,
                off: 0,
                dst: buf.as_mut_ptr(),
            });
        }
        assert!(ring.submit());
        assert!(
            !ring.submit(),
            "a second submit with a round in flight must be refused"
        );
        assert!(ring.wait());
        assert_eq!(buf[..], data[..16]);

        let _ = std::fs::remove_file(&path);
    }
}
