# eabitnet — Standalone BitNet Runner with Eä Kernels

## What

Minimal standalone binary that runs BitNet b1.58 2B-4T text generation using Eä
SIMD kernels for the entire inference pipeline. No llama.cpp. No eaclaw. Fewer
lines and faster than Microsoft's BitNet.cpp.

## Goal

~1720 lines total (~840 Eä + ~880 Rust). Microsoft's `ggml-bitnet-mad.cpp` alone
is 1056 lines of hand-written intrinsics, on top of the entire llama.cpp framework.

## Model

Microsoft BitNet b1.58 2B-4T:
- 24 layers, 2560 hidden dim, 32 attention heads, 80 head dim
- 6912 FFN intermediate, 32K vocab
- RoPE positional encoding, squared ReLU activation, RMSNorm
- I2_S ternary weights (~1.19 GB GGUF)
- No GQA (32 KV heads = 32 attention heads, 1:1)

## Architecture

```
eabitnet/
├── kernels/                    # Eä source (existing + new)
│   ├── bitnet_i2s.ea           # ternary matmul x86 (DONE, 145 lines)
│   ├── bitnet_i2s_arm.ea       # ternary matmul ARM (DONE, 145 lines)
│   ├── bitnet_quant.ea         # f32→i8 quant (DONE, 100 lines)
│   ├── bitnet_lut.ea           # LUT matmul (DONE, 235 lines, not used)
│   ├── bitnet_rmsnorm.ea       # NEW: sum-of-squares + scale (~40 lines)
│   ├── bitnet_softmax.ea       # NEW: exp + normalize (~50 lines)
│   ├── bitnet_rope.ea          # NEW: sin/cos positional encoding (~40 lines)
│   ├── bitnet_attention.ea     # NEW: batched QK dot + V accumulate (~60 lines)
│   └── bitnet_activate.ea      # NEW: squared ReLU fused (~25 lines)
├── src/                        # Rust glue
│   ├── main.rs                 # CLI entry (~50 lines)
│   ├── gguf.rs                 # GGUF parser (~200 lines)
│   ├── tokenizer.rs            # BPE encode/decode (~200 lines)
│   ├── model.rs                # weight loading + model struct (~150 lines)
│   ├── forward.rs              # transformer forward pass (~200 lines)
│   └── ffi.rs                  # Eä kernel FFI declarations (~80 lines)
├── build_kernels.sh            # compile .ea → .so
├── Cargo.toml
└── tests/                      # kernel tests (existing) + integration
```

## Eä Kernels

### Existing (DONE, 40/40 tests passing)

**bitnet_i2s.ea** (x86, ~24 Gop/s)
- `i2_dot_i8(weights: *u8, activations: *i8, n: i32) -> i32`
- `i2_dot_i8_4row(w0-3: *u8, activations: *i8, scores: *mut i32, n: i32)`
- Ternary dot product via `maddubs_i32`. Weight format: 2-bit packed {0,1,2},
  4 groups of 32 per 128-weight block (QK=128).

**bitnet_i2s_arm.ea** (aarch64, ~28 Gop/s on Pi 5)
- Same API, ARM NEON via `vdot_i32`.

**bitnet_quant.ea** (x86 only)
- `quant_f32_i8(src: *f32, dst: *mut i8, out_scale: *mut f32, n: i32)`
- `pack_ternary_row(ternary: *u8, dst: *mut u8, n: i32)`
- Per-tensor absmax quantization. Caller allocates dst with 12 extra bytes.

### New Kernels

**bitnet_rmsnorm.ea** (~40 lines)
- `rmsnorm_f32(x: *f32, weight: *f32, out: *mut f32, n: i32, eps: f32)`
- Pass 1: sum of squares via quad f32x8 accumulators (reduction/best_kernel pattern)
- Pass 2: `out[i] = x[i] * weight[i] / sqrt(sumsq/n + eps)`

**bitnet_softmax.ea** (~50 lines)
- `softmax_f32(x: *f32, out: *mut f32, n: i32)`
- Pass 1: max via f32x8 dual accumulators
- Pass 2: `exp(x[i] - max)` via 4th-order polynomial, accumulate sum
- Pass 3: divide by sum
- Fast exp approximation — sufficient for softmax (errors cancel in ratio)

**bitnet_rope.ea** (~40 lines)
- `rope_f32(q: *mut f32, k: *mut f32, freqs: *f32, head_dim: i32, n_heads: i32)`
- In-place rotation of (q[2i], q[2i+1]) and (k[2i], k[2i+1]) by (cos θ, sin θ)
- Frequency table precomputed in Rust, passed as pointer

**bitnet_attention.ea** (~60 lines)
- `attn_scores_f32(q: *f32, k_cache: *f32, out: *mut f32, head_dim: i32, seq_len: i32, scale: f32)`
  - Batched Q·K dot products, scaled by 1/√d. Pattern from batch_dot/best_kernel.
- `attn_weighted_sum_f32(scores: *f32, v_cache: *f32, out: *mut f32, head_dim: i32, seq_len: i32)`
  - Weighted V accumulation: `out = Σ score[t] * V[t]`. Pattern from astro_stack accumulate.

**bitnet_activate.ea** (~25 lines)
- `squared_relu_mul_f32(gate: *f32, up: *f32, out: *mut f32, n: i32)`
- Fused: `out[i] = (gate[i] * |gate[i]|) * up[i]`
- Branchless abs via `select`, single f32x8 pass

## Transformer Forward Pass

One token through the model:

```
token_id
  → embedding lookup (Rust: index into f32 table)
  → for each of 24 layers:
      ├── rmsnorm_f32(x, attn_norm_weight)
      ├── quant_f32_i8(normed) → i8 activations + scale
      ├── Q = i2_dot_i8_4row(x_i8, W_q)    # 32 heads × 80 dim = 2560
      ├── K = i2_dot_i8_4row(x_i8, W_k)    # 32 heads × 80 dim = 2560
      ├── V = i2_dot_i8_4row(x_i8, W_v)    # 32 heads × 80 dim = 2560
      ├── rope_f32(Q, K, freqs)
      ├── KV cache: append K, V at position
      ├── for each of 32 heads:
      │     ├── attn_scores_f32(Q[h], K_cache[h], scores, 80, seq_len, 1/√80)
      │     ├── softmax_f32(scores, scores, seq_len)
      │     └── attn_weighted_sum_f32(scores, V_cache[h], attn_out[h], 80, seq_len)
      ├── quant_f32_i8(attn_out)
      ├── O = i2_dot_i8_4row(attn_out_i8, W_o)
      ├── x = x + O                          # Rust: vector add
      ├── rmsnorm_f32(x, ffn_norm_weight)
      ├── quant_f32_i8(normed)
      ├── gate = i2_dot_i8_4row(x_i8, W_gate)  # 2560 → 6912
      ├── up   = i2_dot_i8_4row(x_i8, W_up)    # 2560 → 6912
      ├── squared_relu_mul_f32(gate, up, hidden)
      ├── quant_f32_i8(hidden)
      ├── down = i2_dot_i8_4row(hidden_i8, W_down)  # 6912 → 2560
      └── x = x + down                       # Rust: vector add
  → rmsnorm_f32(x, final_norm_weight)
  → logits = i2_dot_i8_4row(x_i8, W_vocab)   # 2560 → 32000
  → sample(logits)                             # Rust: temperature + top-p
```

Each linear layer requires a quant step before it (BitNet quantizes activations
to i8 before ternary matmul). 7 linear layers per transformer layer + 1 final
= 169 quant + matmul calls per token.

## Rust Components

**gguf.rs** (~200 lines)
- Parse GGUF v3 header, metadata KV pairs, tensor info
- Memory-map tensor data (mmap, no copy)
- Extract model hyperparams + vocab from metadata
- Return `GgufFile` with metadata + tensor offset map

**tokenizer.rs** (~200 lines)
- BPE encode/decode parsed from GGUF vocab metadata
- Byte-level BPE (LLaMA tokenizer format)
- No external dependencies

**model.rs** (~150 lines)
- `BitNetModel` struct: hyperparams + pointers into mmap'd weight tensors
- `load(path)` — parse GGUF, validate shapes, store weight pointers
- No copying — I2_S weights stay in mmap, ready for Eä kernels

**forward.rs** (~200 lines)
- `generate(model, tokens, max_tokens) -> Vec<u32>`
- Allocates scratch buffers once: hidden states, QKV, attention scores, KV cache
- KV cache: `[n_layers][max_seq_len][n_heads][head_dim]` plain f32
- Temperature + top-p sampling

**ffi.rs** (~80 lines)
- `extern "C"` declarations for all Eä kernels
- Debug-mode slice length assertions

**main.rs** (~50 lines)
- `eabitnet --model path.gguf --prompt "text" [--max-tokens N] [--temperature T]`

## Ternary Offset Correction

I2_S kernels return raw dot where weight ∈ {0,1,2} not {-1,0,+1}.
Correction: `true_dot = raw_dot - sum(activations)`.

This is applied in `forward.rs` after each `i2_dot_i8` call. The activation sum
is computed once per quant step (a single `sum_f32x8` call on the pre-quant f32
values, or accumulated during quant).

## I2_S Weight Format

Matches Microsoft's GGUF format:
- Block size: QK=128 (128 ternary weights per block)
- Block layout: 32 bytes = 4 groups of 8 bytes, each group packs 32 weights at 2 bits
- Encoding: `0b00 = 0 (maps to -1)`, `0b01 = 1 (maps to 0)`, `0b10 = 2 (maps to +1)`
- Per-tensor f32 scale (stored separately in GGUF metadata)

## Build

```bash
# Compile Eä kernels to shared libraries
./build_kernels.sh

# Build Rust binary (links against kernel .so files)
cargo build --release

# Run
./target/release/eabitnet --model bitnet-b1.58-2B-4T-i2s.gguf --prompt "Hello"
```

## Out of Scope

- ARM quant kernel (x86-only for now, ARM port is future work)
- eakv / KV cache compression (plain f32 buffer, ~240MB at 2048 ctx)
- Batch inference / parallel prefill (one token at a time)
- Chat template (raw text in/out)
- Model download (user provides GGUF path)
- Cross-compilation setup (x86 first, ARM later)
- eaclaw integration (separate future project)
