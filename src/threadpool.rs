//! Persistent thread pool for parallel matmul dispatch.
//!
//! Workers sleep on a condvar between dispatches. The dispatcher stores
//! closures as raw pointers and blocks until all workers complete.
//!
//! Note: `run()` is not safe for concurrent calls from multiple threads.
//! This matches usage: `forward()` takes `&mut self`.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::thread::{self, JoinHandle};

/// A no-op closure used as a harmless placeholder fat-pointer for unused funcs slots.
/// Workers only dereference funcs[i] when n_groups >= i+1, so this is never called.
static NOOP_FN: &(dyn Fn(usize, usize) + Send + Sync) = &|_, _| {};

struct WorkState {
    /// Up to 3 function pointers for split dispatch.
    funcs: [*const dyn Fn(usize, usize); 3],
    /// Cumulative thread boundaries: group 0 = [0, bounds[0]),
    /// group 1 = [bounds[0], bounds[1]), group 2 = [bounds[1], bounds[2]).
    bounds: [usize; 3],
    n_groups: usize,
    /// Incremented each dispatch so workers can detect new work.
    generation: u64,
    /// Set to true to shut down all workers.
    shutdown: bool,
}

unsafe impl Send for WorkState {}
unsafe impl Sync for WorkState {}

pub struct ThreadPool {
    shared: Arc<(Mutex<WorkState>, Condvar)>,
    done: Arc<AtomicUsize>,
    done_signal: Arc<(Mutex<bool>, Condvar)>,
    workers: Vec<JoinHandle<()>>,
    n_threads: usize,
}

impl ThreadPool {
    pub fn new() -> Self {
        let n_threads = thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1);

        // We need valid fat pointers as placeholders; use a no-op closure.
        // Safety: placeholder slots beyond n_groups are never dereferenced.
        let noop: &dyn Fn(usize, usize) = &|_, _| {};
        let noop_ptr = noop as *const dyn Fn(usize, usize);
        let shared = Arc::new((
            Mutex::new(WorkState {
                funcs: [noop_ptr, noop_ptr, noop_ptr],
                bounds: [0, 0, 0],
                n_groups: 0,
                generation: 0,
                shutdown: false,
            }),
            Condvar::new(),
        ));
        let done = Arc::new(AtomicUsize::new(0));
        let done_signal = Arc::new((Mutex::new(false), Condvar::new()));

        let mut workers = Vec::with_capacity(n_threads);
        for tid in 0..n_threads {
            let shared = Arc::clone(&shared);
            let done = Arc::clone(&done);
            let done_signal = Arc::clone(&done_signal);
            let handle = thread::spawn(move || {
                let mut last_gen: u64 = 0;
                loop {
                    let (funcs, bounds, n_groups);
                    {
                        let (lock, cvar) = &*shared;
                        let mut state = lock.lock().unwrap();
                        while state.generation == last_gen && !state.shutdown {
                            state = cvar.wait(state).unwrap();
                        }
                        if state.shutdown {
                            return;
                        }
                        last_gen = state.generation;
                        funcs = state.funcs;
                        bounds = state.bounds;
                        n_groups = state.n_groups;
                    }
                    let n_active_total = bounds[n_groups - 1];
                    if tid < n_active_total {
                        if tid < bounds[0] {
                            let f = unsafe { &*funcs[0] };
                            f(tid, bounds[0]);
                        } else if n_groups >= 2 && tid < bounds[1] {
                            let f = unsafe { &*funcs[1] };
                            f(tid - bounds[0], bounds[1] - bounds[0]);
                        } else if n_groups >= 3 && tid < bounds[2] {
                            let f = unsafe { &*funcs[2] };
                            f(tid - bounds[1], bounds[2] - bounds[1]);
                        }
                    }
                    if done.fetch_sub(1, Ordering::AcqRel) == 1 {
                        let (lock, cvar) = &*done_signal;
                        let mut finished = lock.lock().unwrap();
                        *finished = true;
                        cvar.notify_one();
                    }
                }
            });
            workers.push(handle);
        }

        ThreadPool { shared, done, done_signal, workers, n_threads }
    }

    pub fn thread_count(&self) -> usize {
        self.n_threads
    }

    fn dispatch(
        &self,
        funcs: [*const dyn Fn(usize, usize); 3],
        bounds: [usize; 3],
        n_groups: usize,
    ) {
        // Reset done counter — all n_threads must check in (even if idle)
        self.done.store(self.n_threads, Ordering::Release);
        {
            let mut finished = self.done_signal.0.lock().unwrap();
            *finished = false;
        }

        {
            let (lock, cvar) = &*self.shared;
            let mut state = lock.lock().unwrap();
            state.funcs = funcs;
            state.bounds = bounds;
            state.n_groups = n_groups;
            state.generation += 1;
            cvar.notify_all();
        }

        // Wait for all threads to finish
        {
            let (lock, cvar) = &*self.done_signal;
            let mut finished = lock.lock().unwrap();
            while !*finished {
                finished = cvar.wait(finished).unwrap();
            }
        }
    }

    pub fn run(&self, n: usize, f: impl Fn(usize, usize) + Send + Sync) {
        debug_assert!(n <= self.n_threads, "n ({n}) > pool size ({})", self.n_threads);
        if n == 0 { return; }

        // Safety: we block in dispatch() until all workers finish, so the closure's
        // stack frame is guaranteed alive for the full duration. We erase
        // the lifetime to satisfy the WorkState fat-pointer field.
        let func_ref: &dyn Fn(usize, usize) = &f;
        let func_ref: &dyn Fn(usize, usize) = unsafe { std::mem::transmute(func_ref) };
        self.dispatch(
            [func_ref as *const _, NOOP_FN as *const _, NOOP_FN as *const _],
            [n, 0, 0],
            1,
        );
    }

    pub fn run_split3(
        &self,
        n1: usize, f1: impl Fn(usize, usize) + Send + Sync,
        n2: usize, f2: impl Fn(usize, usize) + Send + Sync,
        n3: usize, f3: impl Fn(usize, usize) + Send + Sync,
    ) {
        debug_assert!(
            n1 + n2 + n3 <= self.n_threads,
            "split3 {} + {} + {} > pool {}", n1, n2, n3, self.n_threads
        );
        if n1 + n2 + n3 == 0 { return; }
        let r1: &dyn Fn(usize, usize) = &f1;
        let r2: &dyn Fn(usize, usize) = &f2;
        let r3: &dyn Fn(usize, usize) = &f3;
        // Safety: dispatch() blocks until all workers finish.
        let r1: &dyn Fn(usize, usize) = unsafe { std::mem::transmute(r1) };
        let r2: &dyn Fn(usize, usize) = unsafe { std::mem::transmute(r2) };
        let r3: &dyn Fn(usize, usize) = unsafe { std::mem::transmute(r3) };
        self.dispatch(
            [r1 as *const _, r2 as *const _, r3 as *const _],
            [n1, n1 + n2, n1 + n2 + n3],
            3,
        );
    }
}

impl Drop for ThreadPool {
    fn drop(&mut self) {
        {
            let (lock, cvar) = &*self.shared;
            let mut state = lock.lock().unwrap();
            state.shutdown = true;
            cvar.notify_all();
        }
        for handle in self.workers.drain(..) {
            let _ = handle.join();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    #[test]
    fn test_run_all_threads_execute() {
        let pool = ThreadPool::new();
        let count = AtomicUsize::new(0);
        pool.run(pool.thread_count(), |_tid, _n| {
            count.fetch_add(1, Ordering::Relaxed);
        });
        assert_eq!(count.load(Ordering::Relaxed), pool.thread_count());
    }

    #[test]
    fn test_run_subset_of_threads() {
        let pool = ThreadPool::new();
        let n = pool.thread_count().min(4);
        let count = AtomicUsize::new(0);
        pool.run(n, |_tid, _n| {
            count.fetch_add(1, Ordering::Relaxed);
        });
        assert_eq!(count.load(Ordering::Relaxed), n);
    }

    #[test]
    fn test_run_writes_to_slices() {
        let pool = ThreadPool::new();
        let n = pool.thread_count();
        let mut data = vec![0u32; n];
        let ptr = data.as_mut_ptr() as usize;
        pool.run(n, |tid, _n| {
            unsafe { *(ptr as *mut u32).add(tid) = tid as u32 + 1; }
        });
        for i in 0..n {
            assert_eq!(data[i], i as u32 + 1);
        }
    }

    #[test]
    fn test_run_multiple_dispatches() {
        let pool = ThreadPool::new();
        let count = AtomicUsize::new(0);
        for _ in 0..100 {
            pool.run(pool.thread_count(), |_tid, _n| {
                count.fetch_add(1, Ordering::Relaxed);
            });
        }
        assert_eq!(count.load(Ordering::Relaxed), pool.thread_count() * 100);
    }

    #[test]
    fn test_run_split3() {
        let pool = ThreadPool::new();
        let n = pool.thread_count();
        let n1 = n / 2;
        let n2 = n / 4;
        let n3 = n - n1 - n2;
        let c1 = AtomicUsize::new(0);
        let c2 = AtomicUsize::new(0);
        let c3 = AtomicUsize::new(0);
        pool.run_split3(
            n1, |_tid, _n| { c1.fetch_add(1, Ordering::Relaxed); },
            n2, |_tid, _n| { c2.fetch_add(1, Ordering::Relaxed); },
            n3, |_tid, _n| { c3.fetch_add(1, Ordering::Relaxed); },
        );
        assert_eq!(c1.load(Ordering::Relaxed), n1);
        assert_eq!(c2.load(Ordering::Relaxed), n2);
        assert_eq!(c3.load(Ordering::Relaxed), n3);
    }

}
