# eabitnet — eaclaw with BitNet via Eä kernels

## What

Standalone binary that runs eaclaw's full agent stack (tools, WhatsApp, ShellGuard,
eakv KV cache) on top of Microsoft's BitNet b1.58 2B-4T model, using Eä SIMD kernels
for ternary inference. Targets Raspberry Pi 5 (ARM aarch64, 8GB RAM).

eaclaw remains untouched as a git submodule. eabitnet adds BitNet support on top.

## Architecture

```
eabitnet (standalone binary)
  ├── eaclaw (git submodule, untouched)
  │     ├── llama.cpp submodule
  │     ├── eakv (KV cache compression)
  │     └── full agent (tools, WhatsApp, ShellGuard, etc.)
  ├── patches/
  │     └── i2s_support.patch  (adds GGML_TYPE_I2_S to llama.cpp)
  ├── csrc/
  │     └── eabitnet_bridge.c  (registers Eä kernels as I2_S type traits)
  ├── kernels/ (pre-compiled .o from eabitnet kernel repo)
  │     ├── bitnet_i2s.o       (x86, maddubs_i32)
  │     ├── bitnet_i2s_arm.o   (aarch64, vdot_i32)
  │     └── bitnet_quant.o     (x86, quant_f32_i8 + pack_ternary_row)
  ├── build.rs (applies patch, compiles bridge, links kernels)
  └── src/main.rs (reuses eaclaw's agent, swaps LLM provider)
```

## Model

Microsoft BitNet b1.58 2B-4T:
- 2 billion parameters, trained on 4 trillion tokens
- GGUF file: `ggml-model-i2_s.gguf` (~1.19 GB)
- Weight format: I2_S (2-bit ternary packed, 4 weights per byte)
- Tokenizer: standard LLaMA (handled by llama.cpp)
- Source: `huggingface.co/microsoft/bitnet-b1.58-2B-4T-gguf`

## Kernels

No LUT kernel — i2s is faster for i8 activations.

| Platform | Kernel | Intrinsic | Performance |
|----------|--------|-----------|-------------|
| x86_64 | `bitnet_i2s.ea` | maddubs_i32 | ~24 Gop/s |
| aarch64 | `bitnet_i2s_arm.ea` | vdot_i32 | ~28 Gop/s (Pi 5) |
| both | `bitnet_quant.ea` | narrow_f32x4_i8 | — |

## I2_S patch (minimal additions to llama.cpp)

Applied at build time to eaclaw's llama.cpp submodule. Adds:

1. **`GGML_TYPE_I2_S` enum value** in `ggml.h`
2. **I2_S block structure** — 32 bytes = 128 ternary weights, 4 groups of 32,
   matching the existing `bitnet_i2s.ea` layout (QK=128)
3. **GGUF type mapping** — so `ggml-model-i2_s.gguf` loads correctly
4. **Type traits entry** — block size, type size, from_float, vec_dot pointers
5. **BitNet model architecture handler** — squared ReLU activation, ternary
   quantization-aware layers

The patch does NOT modify any existing llama.cpp code paths — it only adds new
type definitions and a model architecture handler.

## eabitnet_bridge.c

Registers eabitnet Eä kernels as the compute functions for I2_S type traits:

```c
// Called once at startup before model load
void eabitnet_register_kernels(void) {
    ggml_type_traits[GGML_TYPE_I2_S].vec_dot      = eabitnet_i2_dot_i8;
    ggml_type_traits[GGML_TYPE_I2_S].vec_dot_type  = GGML_TYPE_I8;
    ggml_type_traits[GGML_TYPE_I2_S].from_float    = eabitnet_quant_f32_i8;
    ggml_type_traits[GGML_TYPE_I2_S].block_size    = 128;
    ggml_type_traits[GGML_TYPE_I2_S].type_size     = 32;
}
```

This hooks into ggml's existing MUL_MAT dispatch — no patching of the compute
path. When ggml encounters an I2_S weight tensor, it calls our vec_dot function.

The bridge also wraps `i2_dot_i8_4row` for the multi-row path that ggml uses
for better cache utilization.

## Build flow

1. `build.rs` checks out eaclaw's llama.cpp submodule
2. Applies `patches/i2s_support.patch` to the llama.cpp source
3. Builds llama.cpp via CMake (same flags as eaclaw)
4. Compiles `csrc/eabitnet_bridge.c` with cc crate
5. Links pre-compiled Eä kernel `.o` files from `kernels/`
6. Links eaclaw crates as dependencies (agent, tools, WhatsApp, etc.)
7. Builds `eabitnet` binary

## Runtime

```
eabitnet --model bitnet-2b
```

- First run: downloads `ggml-model-i2_s.gguf` from HuggingFace to `~/.eabitnet/models/`
- Loads model via patched llama.cpp (I2_S recognized)
- Registers eabitnet kernels via bridge
- Starts eaclaw agent with BitNet as the LLM backend
- All eaclaw features available: tools, WhatsApp, ShellGuard, eakv KV cache

## Testing

- **Unit:** eabitnet kernel tests (40/40 already passing)
- **Integration:** load GGUF, run short generation, verify coherent output
- **Cross-platform:** build on x86 (dev) and aarch64 (Pi 5)
- **Regression:** eaclaw's existing tests unaffected (submodule untouched)

## Out of scope

- LUT kernel (dropped — i2s is faster for i8 activations)
- Multi-model switching (eabitnet always runs BitNet)
- GPU backends (CPU-only, Pi 5 target)
- Custom tokenizer (BitNet uses standard LLaMA tokenizer)
- Modifying eaclaw (stays untouched as submodule)
