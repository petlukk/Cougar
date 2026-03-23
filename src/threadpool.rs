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

struct WorkState {
    /// Raw pointer to the closure. Transmuted from &dyn Fn(usize, usize).
    /// Safety: dispatcher blocks until all workers complete, so the closure's
    /// stack frame is alive for the entire execution.
    func: *const dyn Fn(usize, usize),
    /// How many threads should execute this batch.
    n_active: usize,
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

        // We need a valid fat pointer as a placeholder; use a no-op closure.
        // Safety: this placeholder is never called (generation starts at 0,
        // workers wait for generation > 0 before executing).
        let noop: &dyn Fn(usize, usize) = &|_, _| {};
        let shared = Arc::new((
            Mutex::new(WorkState {
                func: noop as *const dyn Fn(usize, usize),
                n_active: 0,
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
                    let (func_ptr, n_active);
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
                        func_ptr = state.func;
                        n_active = state.n_active;
                    }
                    if tid < n_active {
                        let f = unsafe { &*func_ptr };
                        f(tid, n_active);
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

    pub fn run(&self, n: usize, f: impl Fn(usize, usize) + Send + Sync) {
        debug_assert!(n <= self.n_threads, "n ({n}) > pool size ({})", self.n_threads);
        if n == 0 { return; }

        // Reset done counter — all n_threads must check in (even if idle)
        self.done.store(self.n_threads, Ordering::Release);
        {
            let mut finished = self.done_signal.0.lock().unwrap();
            *finished = false;
        }

        // Store closure as raw pointer and dispatch.
        // Safety: we block below until all workers finish, so the closure's
        // stack frame is guaranteed alive for the full duration. We erase
        // the lifetime to satisfy the WorkState fat-pointer field.
        let func_ref: &dyn Fn(usize, usize) = &f;
        let func_ref: &dyn Fn(usize, usize) =
            unsafe { std::mem::transmute(func_ref) };
        {
            let (lock, cvar) = &*self.shared;
            let mut state = lock.lock().unwrap();
            state.func = func_ref as *const dyn Fn(usize, usize);
            state.n_active = n;
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
        let count = AtomicUsize::new(0);
        pool.run(4, |_tid, _n| {
            count.fetch_add(1, Ordering::Relaxed);
        });
        assert_eq!(count.load(Ordering::Relaxed), 4);
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
}
