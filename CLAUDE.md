# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

Cougar is a BitNet b1.58 inference engine: Rust + [Eä](https://github.com/petlukk/eacompute) SIMD kernels. Zero crate dependencies. Single binary with embedded kernels.

## Hard Rules

1. **No file exceeds 500 lines.** Split before you hit the limit.
2. **Every feature proven by end-to-end test.** If it's not tested, it doesn't exist.
3. **No fake functions.** No `todo!()`, no stubs, no silent fallbacks. If hardware doesn't support it, the compiler errors.
4. **No premature features.** Don't build what isn't needed yet.
5. **Delete, don't comment.** Dead code gets removed.
6. **No LUT shortcuts.** Pure compute — prove Eä kernels beat LUT-based runners with raw SIMD math.

## Build & Test

```bash
# Build kernels (requires eacompute compiler)
EA=/path/to/eacompute/target/release/ea make kernels

# Build binary (kernels embedded — no LD_LIBRARY_PATH needed)
cargo build --release

# Run all Rust tests (70 tests)
cargo test

# Run C kernel tests (102 assertions across 13 harnesses)
EA=/path/to/eacompute/target/release/ea make test

# Run a single Rust test
cargo test test_name -- --nocapture

# Run the binary
./target/release/cougar --model ~/.cougar/models/ggml-model-i2_s.gguf --prompt "hello"
./target/release/cougar --interactive
./target/release/cougar --serve --port 8080
```

## Architecture

**Inference pipeline** (`forward.rs`): For each of 30 layers: RMSNorm → quantize f32→i8 → ternary matmul (QKV) → RoPE → fused online softmax attention → O projection → residual → FFN (fused gate+up → squared ReLU → down) → residual. Output: i8 quantized embedding projection → logits → sample.

**Kernel embedding** (`build.rs` + `embed.rs`): `build.rs` generates `include_bytes!` for all 7 .so files + content hash. At runtime, `embed.rs` extracts to `~/.cougar/lib/v{VERSION}-{HASH}/` and loads via raw `dlopen`/`dlsym`. `ffi.rs` wraps each kernel function pointer as `pub unsafe fn`.

**Thread pool** (`threadpool.rs`): Persistent condvar-based workers. `run(n, f)` for uniform dispatch, `run_split3(n1,f1,n2,f2,n3,f3)` for concurrent QKV. Falls back to sequential when <3 threads available.

**Tokenizer** (`tokenizer.rs`): BPE from GGUF metadata. GPT-2 byte encoding reversed at load time via `gpt2_unicode_to_byte()` so vocab stores raw bytes. Encode/decode work with raw bytes naturally.

**Three modes**: `--prompt` (single shot), `--interactive` (REPL streaming to stdout), `--serve` (HTTP + SSE web chat UI).

## Key Patterns

- **FFI calls**: Always go through `ffi.rs` wrappers which delegate to `embed::k()` function pointer table. Never use `#[link]` directly.
- **Matmul dispatch**: `ternary_matmul_mt_n()` is the workhorse — splits rows across threads, each calls `i2_dot_i8_4row` kernel. `ternary_matmul_fused_pair()` uses the dual kernel for gate+up.
- **Zero allocation in hot loops**: Matmul uses stack `[0i32; 4]` arrays, not `Vec`. The `on_token` callback is `FnMut` (inlined, zero-cost).
- **Profiling**: `prof!` macro wraps timed sections, skips `Instant::now()` when not profiling (pos != 1).

## Eä Kernels

8 kernels in `kernels/`, compiled to .so via eacompute. ILP-optimized:
- `bitnet_i2s.ea`: Ternary matmul (single-row dual-acc, 4-row group-first interleaving, 4-row dual gate/up interleaving)
- `bitnet_i8dot.ea`: i8×u8 dot product (2x unrolled, dual accumulators)
- `bitnet_fused_attn.ea`: Online softmax attention (no scores buffer)
- `bitnet_quant.ea`, `bitnet_rmsnorm.ea`, `bitnet_activate.ea`, `bitnet_vecadd.ea`

## Commit Convention

Commit as Peter Lukka (peter.lukka@gmail.com). Push to git@github.com:petlukk/Cougar.git.
Prefix: `feat:`, `fix:`, `perf:`, `test:`, `docs:`, `refactor:`, `cleanup:`.
