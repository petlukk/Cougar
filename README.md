```
        /\_/\
       ( o.o )  ╔═══════════════════════════════════╗
        >╥╥<    ║  C O U G A R                      ║
       /|   |\  ║  LLM inference engine              ║
      (_|   |_) ╚═══════════════════════════════════╝
```

# Cougar

Llama.cpp is a beast, but even beasts have predators.

A ~7,100-line LLM inference engine written in Rust + [Ea](https://github.com/petlukk/eacompute) SIMD kernels. No llama.cpp. No dependencies. Single binary with embedded kernels, interactive REPL, and web chat UI.

**Supported models:** BitNet b1.58 (I2_S) and Llama-family Q4_K_M (Llama 3, Mistral, Qwen)

## Performance

16 threads, x86-64 AVX2, native Linux:

### BitNet b1.58 2B-4T

| | Cougar | BitNet.cpp |
|---|---|---|
| Decode | **19.3 tok/s** | 14.8 tok/s |
| | **+31% faster** | |

### Llama 3.2 3B Instruct Q4_K_M

| | Cougar | llama.cpp |
|---|---|---|
| Decode | **8.3 tok/s** | 8.4 tok/s |
| Prefill (6 tok) | 14.0 tok/s | 27.8 tok/s |
| Prefill (21 tok) | 17.7 tok/s | 32.7 tok/s |

Decode within 1% of llama.cpp. Prefill gap is true kernel-level GEMM batching (planned).

## Quick start

### 1. Build

```bash
# Build kernels (needs eacompute compiler)
EA=/path/to/ea make kernels

# Build binary (kernels embedded — no LD_LIBRARY_PATH needed)
cargo build --release

# Add to PATH (do this once)
ln -s $(pwd)/target/release/cougar ~/.local/bin/cougar
```

### 2. Download a model

```bash
mkdir -p ~/.cougar/models

# BitNet b1.58 2B-4T (fast, 1.7 GB)
curl -L "https://huggingface.co/microsoft/bitnet-b1.58-2B-4T-gguf/resolve/main/ggml-model-i2_s.gguf" \
  -o ~/.cougar/models/ggml-model-i2_s.gguf

# Llama 3.2 3B Instruct Q4_K_M (smarter, 1.9 GB)
curl -L "https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF/resolve/main/Llama-3.2-3B-Instruct-Q4_K_M.gguf" \
  -o ~/.cougar/models/Llama-3.2-3B-Instruct-Q4_K_M.gguf
```

### 3. Run

```bash
# Chat with Llama in the browser
cougar --model llama --serve

# Chat with BitNet in the terminal
cougar --model bitnet --interactive

# Single prompt
cougar --model llama --prompt "tell me a joke"

# No --model needed if a model exists in ~/.cougar/models/
cougar --prompt "Hello"
```

`--model llama` and `--model bitnet` are shorthands for the default paths. You can also pass any GGUF file path directly.

## Architecture

```
cougar/
  kernels/     10 Ea SIMD kernels (.ea -> .so, embedded in binary)
  src/         16 Rust modules (76 tests)
  tests/       3 C kernel test harnesses (29 tests)
  build.rs     kernel embedding + ABI hash
```

**7,144 LOC total** (3,075 source + 4,069 tests)

### Inference pipeline

Two forward paths dispatch based on GGUF weight type:

**BitNet (I2_S):** RMSNorm -> i8 quantize -> ternary matmul (i2 x i8 via maddubs) -> squared ReLU -> i8 output projection. Fused gate+up dual kernel shares activation loads.

**Llama Q4_K_M (mixed Q4_K + Q6_K):** RMSNorm -> Q8_K quantize -> Q4_K/Q6_K matmul (nibble x i8 via maddubs, 6-bit scale unpacking) -> SiLU (fused inline) -> GEMM-style batched prefill with L1 weight reuse.

Both paths use a persistent condvar-based thread pool with QKV `run_split3` concurrent dispatch.

### Kernels

10 Ea kernels, 1,587 lines:

| Kernel | Lines | What |
|--------|------:|------|
| `q4k_dot.ea` | 342 | Q4_K dot product: 1-row, 4-row, 4-row dual |
| `q6k_dot.ea` | 256 | Q6_K dot product: 1-row, 4-row |
| `bitnet_i2s.ea` | 242 | Ternary matmul: 1-row, 4-row, 4-row dual (x86) |
| `bitnet_i2s_arm.ea` | 145 | Ternary matmul (ARM NEON) |
| `bitnet_fused_attn.ea` | 120 | Single-pass online softmax attention |
| `bitnet_i8dot.ea` | 106 | i8 x u8 dot for quantized output |
| `bitnet_quant.ea` | 105 | f32 -> i8 quantization + activation sum |
| `q4k_quant.ea` | 88 | f32 -> Q8_K quantization + bsums |
| `bitnet_rmsnorm.ea` | 54 | RMS normalization |
| `bitnet_activate.ea` | 32 | Squared ReLU x up (fused) |
| `bitnet_vecadd.ea` | 17 | Residual vector add |

### Key optimizations

- **Persistent thread pool** -- condvar-based, zero per-dispatch allocation
- **QKV run_split3** -- Q/K/V projections concurrent in single dispatch
- **Fused gate+up+SiLU** -- vertical fusion eliminates intermediate buffers
- **Dual 4-row kernels** -- gate+up share activation loads at register level
- **GEMM-style prefill** -- weight rows loaded once, reused across all prompt tokens
- **Q6_K mixed dispatch** -- per-tensor Q4_K/Q6_K detection for Q4_K_M models
- **Tied embedding fallback** -- handles models without separate output.weight

## Tests

105 tests total, zero warnings:

| Suite | Tests |
|---|---|
| Rust (`cargo test`) | 76 |
| C kernel (Q6K dot) | 15 |
| C kernel (Q4K dot) | 7 |
| C kernel (Q8K quant) | 7 |

## CLI

```
cougar --model llama --serve                     # web chat UI
cougar --model bitnet --interactive              # terminal chat
cougar --model llama --prompt "tell me a joke"   # single prompt
cougar --model /path/to/model.gguf --serve       # custom model

Options:
  --model <llama|bitnet|path.gguf>   Model shorthand or file path
  --prompt <text>         Generate from a single prompt
  --interactive           Interactive REPL (stdin/stdout)
  --serve                 Web chat UI with SSE streaming
  --max-tokens N          Maximum tokens to generate (default: 128)
  --temperature T         Sampling temperature, 0 = greedy (default: 0)
  --repetition-penalty F  Penalize repeated tokens (default: 1.1)
  --max-seq-len N         Maximum sequence length (default: 2048)
  --port N                Server port (default: 8080)
```

## Building

Requires:
- [eacompute](https://github.com/petlukk/eacompute) compiler (`ea` binary)
- Rust 1.63+
- x86-64 with AVX2 + FMA (or ARM with NEON for BitNet)

```bash
# Build eacompute first
cd ~/projects/eacompute && cargo build --release --features=llvm

# Build cougar
cd ~/projects/cougar
make kernels                    # compile .ea -> .so
cargo build --release           # kernels embedded in binary
cargo test                      # 76 Rust tests
```

Ea kernels are compiled to `.so`, embedded via `include_bytes!` at build time, extracted to `~/.cougar/lib/` on first run, and loaded via `dlopen`. No `LD_LIBRARY_PATH` needed.
