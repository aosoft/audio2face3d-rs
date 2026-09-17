#![cfg(any(feature = "mock", feature = "native"))]
mod settings;
mod support;
use audio2face3d::client::{types::*, *};
use std::{
    alloc::{GlobalAlloc, Layout, System},
    sync::atomic::{AtomicUsize, Ordering},
    time::{Duration, Instant},
};
use support::wait;
struct Measured;
static LIVE: AtomicUsize = AtomicUsize::new(0);
static PEAK: AtomicUsize = AtomicUsize::new(0);
static CALLS: AtomicUsize = AtomicUsize::new(0);
fn allocated(n: usize) {
    CALLS.fetch_add(1, Ordering::Relaxed);
    let live = LIVE.fetch_add(n, Ordering::Relaxed) + n;
    PEAK.fetch_max(live, Ordering::Relaxed);
}
// SAFETY: All operations delegate to System with unchanged pointers/layouts; counters do not allocate.
unsafe impl GlobalAlloc for Measured {
    // SAFETY: The caller supplies the valid layout forwarded unchanged to System.
    unsafe fn alloc(&self, l: Layout) -> *mut u8 {
        // SAFETY: Forward the caller-provided valid layout to System.
        let p = unsafe { System.alloc(l) };
        if !p.is_null() {
            allocated(l.size());
        }
        p
    }
    // SAFETY: The caller owns the allocation described by p/l, forwarded unchanged.
    unsafe fn dealloc(&self, p: *mut u8, l: Layout) {
        // SAFETY: Forward the caller-owned allocation and its original layout.
        unsafe { System.dealloc(p, l) };
        LIVE.fetch_sub(l.size(), Ordering::Relaxed);
    }
    // SAFETY: The valid caller layout is forwarded unchanged.
    unsafe fn alloc_zeroed(&self, l: Layout) -> *mut u8 {
        // SAFETY: Forward the caller-provided valid layout to System.
        let p = unsafe { System.alloc_zeroed(l) };
        if !p.is_null() {
            allocated(l.size());
        }
        p
    }
    // SAFETY: The original allocation and new size follow GlobalAlloc requirements.
    unsafe fn realloc(&self, p: *mut u8, l: Layout, n: usize) -> *mut u8 {
        // SAFETY: Pointer, original layout and new size satisfy GlobalAlloc requirements.
        let out = unsafe { System.realloc(p, l, n) };
        if !out.is_null() {
            LIVE.fetch_sub(l.size(), Ordering::Relaxed);
            allocated(n);
        }
        out
    }
}
#[global_allocator]
static ALLOCATOR: Measured = Measured;
fn begin() -> (usize, usize) {
    let live = LIVE.load(Ordering::Relaxed);
    PEAK.store(live, Ordering::Relaxed);
    (live, CALLS.load(Ordering::Relaxed))
}
fn report(name: &str, start: (usize, usize)) {
    println!(
        "MEMORY {name} peak_extra_requested_bytes={} retained_extra_requested_bytes={} allocations={}",
        PEAK.load(Ordering::Relaxed).saturating_sub(start.0),
        LIVE.load(Ordering::Relaxed).saturating_sub(start.0),
        CALLS.load(Ordering::Relaxed) - start.1
    );
}
fn config() -> DirectConfig {
    DirectConfig {
        max_queued: 32,
        limits: Limits {
            max_requests: 33,
            max_buffered_bytes: 8 * 1024 * 1024,
            input_queue_items: 2,
            output_queue_items: 2,
            max_input_chunk_bytes: 128 * 1024,
            input_queue_bytes: 256 * 1024,
            ..Default::default()
        },
        ..Default::default()
    }
}
fn profile(mut make: impl FnMut() -> Client, remote: bool) {
    let client = make();
    let baseline = begin();
    let mut held = vec![];
    for _ in 0..33 {
        let (mut input, output, control) = client.start(RequestOptions::default()).unwrap().split();
        for _ in 0..2 {
            wait(input.send(InputChunk::new(
                PcmBuffer::from_vec(vec![0; 65536]).unwrap(),
                vec![],
            )))
            .unwrap();
        }
        held.push((input, output, control));
    }
    assert!(client.start(RequestOptions::default()).is_err());
    let peak = PEAK.load(Ordering::Relaxed).saturating_sub(baseline.0);
    assert!(
        peak < if remote {
            32 * 1024 * 1024
        } else {
            10 * 1024 * 1024
        },
        "unbounded waiting storage: {peak}"
    );
    report("33_requests_two_64KiB_chunks", baseline);
    for (input, output, control) in held {
        control.cancel();
        drop(input);
        drop(output);
        assert!(wait(control.closed()).is_err());
    }
    wait(client.shutdown()).unwrap();
    drop(client);
    report("after_waiting_shutdown", baseline);
    let client = make();
    let baseline = begin();
    let start = Instant::now();
    let (mut input, mut output, control) = client.start(RequestOptions::default()).unwrap().split();
    let chunks = std::env::var("A2F_PROFILE_CHUNKS")
        .map(|v| v.parse::<usize>().unwrap())
        .unwrap_or(64);
    let bytes = chunks * 65536;
    std::thread::scope(|scope| {
        let sending = scope.spawn(move || {
            for _ in 0..chunks {
                wait(input.send(InputChunk::new(
                    PcmBuffer::from_vec(vec![0; 65536]).unwrap(),
                    vec![],
                )))
                .unwrap();
            }
            wait(input.finish()).unwrap();
        });
        let mut received = 0;
        let mut frames = 0;
        while let Some(event) = wait(output.recv()).unwrap() {
            match event {
                OutputEvent::Audio(a) => {
                    assert!(a.pcm().as_bytes().iter().all(|&v| v == 0));
                    received += a.pcm().as_bytes().len();
                }
                OutputEvent::Curves(_) => {
                    frames += 1;
                    std::thread::sleep(Duration::from_micros(200));
                }
                _ => {}
            }
        }
        sending.join().unwrap();
        assert_eq!(received, bytes);
        println!(
            "STREAM bytes={received} frames={frames} elapsed_ms={}",
            start.elapsed().as_millis()
        );
    });
    wait(control.closed()).unwrap();
    let peak = PEAK.load(Ordering::Relaxed).saturating_sub(baseline.0);
    assert!(
        peak < if remote {
            16 * 1024 * 1024
        } else {
            2 * 1024 * 1024
        },
        "clip retained instead of streaming: {peak}"
    );
    report("slow_receiver", baseline);
    wait(client.shutdown()).unwrap();
    drop(client);
    drop(output);
    drop(control);
    report("after_long_shutdown", baseline);
}

#[test]
#[ignore = "manual allocation/holding profile; run alone with --nocapture"]
fn waiting_requests_and_slow_long_stream() {
    profile(|| wait(Client::direct(config())).unwrap(), false);
}
#[cfg(feature = "mock")]
#[test]
#[ignore = "manual loopback allocation profile; includes client, server and transport"]
fn server_waiting_requests_and_slow_stream() {
    let before_runtime = LIVE.load(Ordering::Relaxed);
    let runtime = tokio::runtime::Runtime::new().unwrap();
    let listener = runtime
        .block_on(tokio::net::TcpListener::bind("127.0.0.1:0"))
        .unwrap();
    let url = format!("http://{}", listener.local_addr().unwrap());
    let (tx, rx) = tokio::sync::oneshot::channel();
    let server = runtime.spawn(audio2face3d_server::server::serve(
        settings::config(["test"]),
        listener,
        async {
            let _ = rx.await;
        },
    ));
    profile(
        || {
            let mut settings = ServerConfig::new(&url);
            settings.runtime = Some(runtime.handle().clone());
            settings.limits = config().limits;
            wait(Client::server(settings)).unwrap()
        },
        true,
    );
    tx.send(()).unwrap();
    runtime.block_on(server).unwrap().unwrap();
    drop(runtime);
    println!(
        "RUNTIME_STOP retained_extra_requested_bytes={}",
        LIVE.load(Ordering::Relaxed).saturating_sub(before_runtime)
    );
}
