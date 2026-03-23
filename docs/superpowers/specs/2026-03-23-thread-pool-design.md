# Thread Pool Design — Persistent Worker Threads for Cougar

**Date:** 2026-03-23
**Goal:** Replace per-matmul `std::thread::scope` with a persistent thread pool to eliminate ~2000+ thread create/teardown cycles per token. Target: 10 tok/s → 12-13 tok/s.

## Problem

Every matmul call in `matmul.rs` creates fresh threads via `std::thread::scope` and tears them down on completion. Critically, `ternary_matmul_qkv` and `ternary_matmul_parallel_pair` have **nested thread spawning**: they spawn 2-3 coordinator threads, each of which calls `ternary_matmul_mt_n`, which itself spawns N sub-threads via its own `thread::scope`. Per layer this is roughly 69 thread spawns, times 30 layers = ~2000+ OS thread lifecycle events per decoded token.

## Design

### ThreadPool struct (`src/threadpool.rs`)

A pool of N worker threads (N = `std::thread::available_parallelism()`) created once at startup. Workers sleep on a shared condvar between dispatches. Zero external dependencies.

### API

```rust
pub struct ThreadPool { /* workers, shared state */ }

impl ThreadPool {
    /// Spawn N workers (N = available_parallelism).
    pub fn new() -> Self;

    /// Run `f` on the first `n` workers. Blocks until all complete.
    /// f(thread_id, n_threads) — thread_id is 0..n.
    /// Panics if n > thread_count (debug_assert).
    pub fn run(&self, n: usize, f: impl Fn(usize, usize) + Send + Sync);

    /// Run f1 on threads 0..n1, f2 on threads n1..n1+n2 concurrently.
    /// Blocks until both complete. Thread counts are dynamic, not hardcoded.
    pub fn run_split2(&self, n1: usize, f1: ..., n2: usize, f2: ...);

    /// Run f1/f2/f3 on disjoint thread ranges concurrently.
    /// Thread counts computed dynamically from pool size (e.g. q=N/2, k=N/4, v=N/4).
    pub fn run_split3(&self, n1: usize, f1: ..., n2: usize, f2: ..., n3: usize, f3: ...);

    /// Number of threads in the pool.
    pub fn thread_count(&self) -> usize;
}
```

### Internals

- **Shared state:** `Arc<Mutex<WorkState>>` + `Condvar` for waking workers.
- **Work descriptor:** Up to 3 function pointer slots (for split dispatch). Each slot stores a raw pointer to the caller's closure plus the thread ID range it applies to.
- **Unsafe lifetime erasure:** The closures passed to `run`/`run_split*` are non-`'static` (they borrow stack data). Workers are persistent and outlive any single `run()` call. To bridge this, the dispatcher transmutes closures to raw `*const dyn Fn(usize, usize)` pointers stored in shared state. **Safety invariant:** the dispatcher blocks until all workers complete, so the closure's stack frame is alive for the entire execution. This is the same pattern used by rayon and crossbeam internally. Each `run*` method contains a single `unsafe` block for this transmute.
- **Worker loop:** lock → wait on condvar → wake → read work → unlock → execute → atomically decrement done counter → if last, notify dispatcher condvar → loop.
- **Completion:** `AtomicUsize` done counter. Dispatcher waits on a second condvar, notified by the last finishing worker.
- **Shutdown:** `Drop` impl sets shutdown flag, notifies all workers, joins all threads.
- **Overflow:** `debug_assert!(n <= self.thread_count())` in `run()`, and `debug_assert!(n1 + n2 + n3 <= self.thread_count())` in split variants.

### Flattening nested thread spawning

Currently `ternary_matmul_qkv` and `ternary_matmul_parallel_pair` spawn coordinator threads that each call `ternary_matmul_mt_n`, which spawns its own threads — two levels of `thread::scope`. The pool collapses this to a single level:

- `ternary_matmul_qkv` becomes `pool.run_split3(q_threads, ..., k_threads, ..., v_threads, ...)` where each closure contains the per-thread work currently inside `ternary_matmul_mt_n`'s spawned closures (chunk calculation, i2_dot_i8 loop, scaling). No more delegation to `ternary_matmul_mt_n` from these functions.
- `ternary_matmul_parallel_pair` becomes `pool.run_split2(half, ..., total-half, ...)` with the same inlining.
- `ternary_matmul_mt_n` retains its own `pool.run(n, ...)` path for direct callers (output proj, down proj).

### Function mapping

| Function | Current | New |
|----------|---------|-----|
| `ternary_matmul_mt_n` | `thread::scope`, N threads | `pool.run(n, ...)` |
| `ternary_matmul_mt` | wrapper calling `ternary_matmul_mt_n` | thin wrapper, passes `pool.thread_count()` to `ternary_matmul_mt_n` |
| `i8_output_matmul_mt` | sequential quantize then `thread::scope` | sequential quantize on caller thread, then `pool.run(N, ...)` for dot-product loop |
| `f32_matmul_mt` | `thread::scope`, N threads | `pool.run(N, ...)` |
| `f16_matmul_mt` | `thread::scope`, N threads | `pool.run(N, ...)` |
| `ternary_matmul_qkv` | `thread::scope` → 3 coordinators → nested `thread::scope` | `pool.run_split3(q, ..., k, ..., v, ...)` — single level, inlined work |
| `ternary_matmul_parallel_pair` | `thread::scope` → 2 coordinators → nested `thread::scope` | `pool.run_split2(half, ..., rest, ...)` — single level, inlined work |

### Integration

- `InferenceState` gains a `pool: ThreadPool` field, initialized once in `new()`.
- `pool` declared before buffer fields so `Drop` joins workers before freeing buffers (though in practice `run()` is synchronous so no in-flight work at drop time).
- `forward()` passes `&self.pool` to matmul functions.
- All matmul functions take `&ThreadPool` instead of spawning threads.
- Thread split ratios computed dynamically from `pool.thread_count()`, not hardcoded.

### What doesn't change

- FFI kernel calls — identical pointer math.
- Eä kernel files — untouched.
- Forward loop structure — same operation order.
- Thread count — same N, just persistent.
- `embed_f16_lookup` — sequential, no threading.

### Expected impact

- Eliminate ~2000+ thread create/teardown cycles per token (was previously underestimated as 240 — nested spawning in QKV and parallel_pair multiplies the real count)
- Save 5-15ms per token
- 10 tok/s → ~12-13 tok/s
