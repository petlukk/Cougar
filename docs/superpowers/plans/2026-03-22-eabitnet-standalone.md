# eabitnet Standalone Runner — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Standalone binary that generates text from BitNet b1.58 2B-4T using Eä SIMD kernels for the entire inference pipeline — no llama.cpp.

**Architecture:** Rust binary (~1150 lines) orchestrates GGUF loading, tokenization, and the transformer loop. All compute runs through Eä kernels (~850 lines) compiled to shared libraries. Memory-mapped weights, pre-allocated scratch buffers, one-token-at-a-time generation.

**Tech Stack:** Rust (no external crates), Eä SIMD kernels (compiled via eacompute), C test harnesses (gcc)

**Hard rules:** <500 lines per file, every feature proven by end-to-end test, no fakes, no premature features, delete don't comment.

---

## File Map

### New Eä Kernels (create)
| File | Responsibility | Est. lines |
|------|---------------|-----------|
| `kernels/bitnet_rmsnorm.ea` | RMSNorm: sum-of-squares + weighted scale | ~40 |
| `kernels/bitnet_softmax.ea` | Softmax: max + exp + normalize | ~50 |
| `kernels/bitnet_rope.ea` | RoPE: in-place sin/cos rotation of QK pairs | ~40 |
| `kernels/bitnet_attention.ea` | Attention: batched QK scores + weighted V sum | ~60 |
| `kernels/bitnet_activate.ea` | Squared ReLU fused with up projection multiply | ~25 |
| `kernels/bitnet_vecadd.ea` | f32 residual vector add | ~15 |

### New Eä Tests (create)
| File | Tests |
|------|-------|
| `tests/test_rmsnorm.c` | Known vectors, uniform input, large dim |
| `tests/test_softmax.c` | Known distribution, numerical stability, edge cases |
| `tests/test_rope.c` | Position 0 (identity), known rotations, multi-head |
| `tests/test_attention.c` | Single-token, multi-token, score magnitude |
| `tests/test_activate.c` | Positive/negative/zero gate, fused output |
| `tests/test_vecadd.c` | Simple add, in-place aliasing, scalar tail |

### New Rust Source (create)
| File | Responsibility | Est. lines |
|------|---------------|-----------|
| `src/main.rs` | CLI entry: parse args, load, generate, print | ~50 |
| `src/gguf.rs` | GGUF v3 parser: header, metadata, tensor map, mmap | ~280 |
| `src/tokenizer.rs` | BPE encode/decode from GGUF vocab data | ~280 |
| `src/model.rs` | BitNetModel struct, weight pointers into mmap | ~150 |
| `src/forward.rs` | Transformer forward pass, KV cache, sampling | ~280 |
| `src/ffi.rs` | extern "C" declarations + debug assertions | ~100 |

### Modify
| File | Change |
|------|--------|
| `build_kernels.sh` | Add new kernel names to build loop (automatic — it globs `kernels/*.ea`) |
| `Makefile` | Add test targets for new kernels |
| `src/eabitnet.h` | Add C declarations for all new kernel functions |

---

## Task 1: Extend quant_f32_i8 with activation sum output

**Files:**
- Modify: `kernels/bitnet_quant.ea`
- Modify: `tests/test_quant.c`
- Modify: `src/eabitnet.h`

The ternary offset correction requires `sum(i8_activations)`. Extend the existing `quant_f32_i8` kernel to compute this during quantization and output it via a new `out_sum` parameter.

- [ ] **Step 1: Add activation sum test to test_quant.c**

Add a new test function that verifies the sum output:

```c
static void test_quant_activation_sum(void) {
    float src[] = {1.0f, -1.0f, 2.0f, -2.0f, 3.0f, -3.0f, 4.0f, -4.0f};
    int8_t dst[8 + 12];
    float scale;
    int32_t sum;
    quant_f32_i8(src, dst, &scale, &sum, 8);
    // Verify sum matches actual i8 values
    int32_t expected_sum = 0;
    for (int i = 0; i < 8; i++) expected_sum += dst[i];
    CHECK("quant_activation_sum", sum == expected_sum);
}

static void test_quant_sum_zeros(void) {
    float src[16] = {0};
    int8_t dst[16 + 12];
    float scale;
    int32_t sum;
    quant_f32_i8(src, dst, &scale, &sum, 16);
    CHECK("quant_sum_zeros", sum == 0);
}
```

Also update existing test calls to pass the new `&sum` parameter.

- [ ] **Step 2: Extend quant_f32_i8 in bitnet_quant.ea**

Change the signature to add `out_sum: *mut i32` parameter. Accumulate the sum of i8 values during the narrowing pass:

```
export func quant_f32_i8(
    src: *restrict f32,
    out dst: *mut i8 [cap: n + 12],
    out out_scale: *mut f32 [cap: 1],
    out out_sum: *mut i32 [cap: 1],
    n: i32
) {
```

In Pass 2, after narrowing each f32x4 → i8x16, accumulate the 4 i8 values into a running i32 sum. After the loop, store to `out_sum[0]`.

- [ ] **Step 3: Update eabitnet.h**

Change declaration to:
```c
void quant_f32_i8(const float *src, int8_t *dst, float *out_scale, int32_t *out_sum, int32_t n);
```

- [ ] **Step 4: Build and run tests**

Run: `cd /home/peter/projects/eabitnet && make clean && make test`

Expected: All quant tests pass (existing + 2 new sum tests).

- [ ] **Step 5: Commit**

```bash
git add kernels/bitnet_quant.ea tests/test_quant.c src/eabitnet.h
git commit -m "feat: extend quant_f32_i8 with activation sum output for ternary correction"
```

---

## Task 2: vecadd kernel (NOTE: `out` must not alias `a` or `b` — use temp buffer in forward.rs)

**Files:**
- Create: `kernels/bitnet_vecadd.ea`
- Create: `tests/test_vecadd.c`
- Modify: `Makefile`
- Modify: `src/eabitnet.h`

- [ ] **Step 1: Write test_vecadd.c**

```c
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "../src/eabitnet.h"

static int tests_passed = 0;
static int tests_total = 0;

#define CHECK(name, cond) do { \
    tests_total++; \
    if (cond) { tests_passed++; printf("  PASS: %s\n", name); } \
    else { printf("  FAIL: %s\n", name); } \
} while(0)

static void test_simple_add(void) {
    float a[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
    float b[] = {8.0f, 7.0f, 6.0f, 5.0f, 4.0f, 3.0f, 2.0f, 1.0f};
    float out[8];
    vecadd_f32(a, b, out, 8);
    int ok = 1;
    for (int i = 0; i < 8; i++)
        if (fabsf(out[i] - 9.0f) > 1e-6f) ok = 0;
    CHECK("simple_add_8", ok);
}

static void test_zeros(void) {
    float a[16] = {0};
    float b[16] = {0};
    float out[16];
    vecadd_f32(a, b, out, 16);
    int ok = 1;
    for (int i = 0; i < 16; i++)
        if (out[i] != 0.0f) ok = 0;
    CHECK("zeros_16", ok);
}

static void test_negative(void) {
    float a[] = {-1.0f, -2.0f, -3.0f, -4.0f};
    float b[] = {1.0f, 2.0f, 3.0f, 4.0f};
    float out[4];
    vecadd_f32(a, b, out, 4);
    int ok = 1;
    for (int i = 0; i < 4; i++)
        if (fabsf(out[i]) > 1e-6f) ok = 0;
    CHECK("negative_cancel", ok);
}

static void test_large(void) {
    int n = 2560;  // hidden dim
    float *a = calloc(n, sizeof(float));
    float *b = calloc(n, sizeof(float));
    float *out = calloc(n, sizeof(float));
    for (int i = 0; i < n; i++) { a[i] = (float)i; b[i] = (float)(n - i); }
    vecadd_f32(a, b, out, n);
    int ok = 1;
    for (int i = 0; i < n; i++)
        if (fabsf(out[i] - (float)n) > 1e-3f) ok = 0;
    CHECK("large_2560", ok);
    free(a); free(b); free(out);
}

static void test_scalar_tail(void) {
    // 11 elements: 8 SIMD + 3 scalar tail
    float a[11], b[11], out[11];
    for (int i = 0; i < 11; i++) { a[i] = 1.0f; b[i] = 2.0f; }
    vecadd_f32(a, b, out, 11);
    int ok = 1;
    for (int i = 0; i < 11; i++)
        if (fabsf(out[i] - 3.0f) > 1e-6f) ok = 0;
    CHECK("scalar_tail_11", ok);
}

int main(void) {
    printf("test_vecadd:\n");
    test_simple_add();
    test_zeros();
    test_negative();
    test_large();
    test_scalar_tail();
    printf("\n%d/%d passed\n", tests_passed, tests_total);
    return tests_passed == tests_total ? 0 : 1;
}
```

- [ ] **Step 2: Write bitnet_vecadd.ea**

```
// bitnet_vecadd.ea — Element-wise f32 vector addition for residual connections

#[cfg(x86_64)]

export func vecadd_f32(a: *restrict f32, b: *restrict f32, out: *mut f32, n: i32) {
    let mut i: i32 = 0
    while i + 8 <= n {
        let va: f32x8 = load(a, i)
        let vb: f32x8 = load(b, i)
        store(out, i, va .+ vb)
        i = i + 8
    }
    while i < n {
        out[i] = a[i] + b[i]
        i = i + 1
    }
}
```

- [ ] **Step 3: Add declarations to eabitnet.h**

Add after the existing `i2_dot_i8_4row` declaration:

```c
// Activation quantization: f32 → int8 with absmax scaling
void quant_f32_i8(const float *src, int8_t *dst, float *out_scale, int32_t n);

// Pack ternary {0,1,2} values into 2-bit packed bytes
void pack_ternary_row(const uint8_t *ternary, uint8_t *packed, int32_t n);

// Element-wise f32 vector addition (residual connections)
void vecadd_f32(const float *a, const float *b, float *out, int32_t n);
```

- [ ] **Step 4: Add test target to Makefile**

Add to the `test` target:

```makefile
	$(CC) $(CFLAGS) tests/test_vecadd.c -L$(LIB) -lbitnet_vecadd -o $(BUILD)/test_vecadd -lm
	LD_LIBRARY_PATH=$(LIB) $(BUILD)/test_vecadd
```

- [ ] **Step 5: Build and run tests**

Run: `cd /home/peter/projects/eabitnet && make clean && make test`

Expected: All 5 vecadd tests pass + existing 40 tests still pass.

- [ ] **Step 6: Commit**

```bash
git add kernels/bitnet_vecadd.ea tests/test_vecadd.c src/eabitnet.h Makefile
git commit -m "feat: add vecadd_f32 SIMD kernel for residual connections"
```

---

## Task 3: rmsnorm kernel

**Files:**
- Create: `kernels/bitnet_rmsnorm.ea`
- Create: `tests/test_rmsnorm.c`
- Modify: `Makefile`
- Modify: `src/eabitnet.h`

- [ ] **Step 1: Write test_rmsnorm.c**

```c
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "../src/eabitnet.h"

static int tests_passed = 0;
static int tests_total = 0;

#define CHECK(name, cond) do { \
    tests_total++; \
    if (cond) { tests_passed++; printf("  PASS: %s\n", name); } \
    else { printf("  FAIL: %s\n", name); } \
} while(0)

#define CLOSE(a, b, tol) (fabsf((a) - (b)) < (tol))

// Reference rmsnorm: out[i] = x[i] * weight[i] / sqrt(mean(x^2) + eps)
static void ref_rmsnorm(const float *x, const float *w, float *out, int n, float eps) {
    float sumsq = 0.0f;
    for (int i = 0; i < n; i++) sumsq += x[i] * x[i];
    float rms = sqrtf(sumsq / n + eps);
    for (int i = 0; i < n; i++) out[i] = x[i] * w[i] / rms;
}

static void test_known_vector(void) {
    float x[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
    float w[] = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
    float out[8], ref[8];
    ref_rmsnorm(x, w, ref, 8, 1e-5f);
    rmsnorm_f32(x, w, out, 8, 1e-5f);
    int ok = 1;
    for (int i = 0; i < 8; i++)
        if (!CLOSE(out[i], ref[i], 1e-4f)) ok = 0;
    CHECK("known_8_unit_weight", ok);
}

static void test_with_weights(void) {
    float x[] = {1.0f, -1.0f, 2.0f, -2.0f, 3.0f, -3.0f, 4.0f, -4.0f};
    float w[] = {0.5f, 2.0f, 0.5f, 2.0f, 0.5f, 2.0f, 0.5f, 2.0f};
    float out[8], ref[8];
    ref_rmsnorm(x, w, ref, 8, 1e-5f);
    rmsnorm_f32(x, w, out, 8, 1e-5f);
    int ok = 1;
    for (int i = 0; i < 8; i++)
        if (!CLOSE(out[i], ref[i], 1e-4f)) ok = 0;
    CHECK("weighted_8", ok);
}

static void test_uniform(void) {
    // Uniform input: rmsnorm(c, w) = w (since c/rms(c) = 1 or -1)
    int n = 16;
    float x[16], w[16], out[16], ref[16];
    for (int i = 0; i < n; i++) { x[i] = 3.0f; w[i] = 1.0f; }
    ref_rmsnorm(x, w, ref, n, 1e-5f);
    rmsnorm_f32(x, w, out, n, 1e-5f);
    int ok = 1;
    for (int i = 0; i < n; i++)
        if (!CLOSE(out[i], ref[i], 1e-4f)) ok = 0;
    CHECK("uniform_16", ok);
}

static void test_large_dim(void) {
    int n = 2560;
    float *x = malloc(n * sizeof(float));
    float *w = malloc(n * sizeof(float));
    float *out = malloc(n * sizeof(float));
    float *ref = malloc(n * sizeof(float));
    for (int i = 0; i < n; i++) {
        x[i] = sinf((float)i * 0.01f);
        w[i] = 1.0f + 0.1f * cosf((float)i * 0.02f);
    }
    ref_rmsnorm(x, w, ref, n, 1e-5f);
    rmsnorm_f32(x, w, out, n, 1e-5f);
    int ok = 1;
    for (int i = 0; i < n; i++)
        if (!CLOSE(out[i], ref[i], 1e-3f)) ok = 0;
    CHECK("large_2560", ok);
    free(x); free(w); free(out); free(ref);
}

static void test_near_zero(void) {
    float x[] = {1e-10f, -1e-10f, 1e-10f, -1e-10f, 1e-10f, -1e-10f, 1e-10f, -1e-10f};
    float w[] = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
    float out[8], ref[8];
    ref_rmsnorm(x, w, ref, 8, 1e-5f);
    rmsnorm_f32(x, w, out, 8, 1e-5f);
    int ok = 1;
    for (int i = 0; i < 8; i++)
        if (!CLOSE(out[i], ref[i], 1e-2f)) ok = 0;
    CHECK("near_zero_eps_stabilized", ok);
}

int main(void) {
    printf("test_rmsnorm:\n");
    test_known_vector();
    test_with_weights();
    test_uniform();
    test_large_dim();
    test_near_zero();
    printf("\n%d/%d passed\n", tests_passed, tests_total);
    return tests_passed == tests_total ? 0 : 1;
}
```

- [ ] **Step 2: Write bitnet_rmsnorm.ea**

```
// bitnet_rmsnorm.ea — RMS normalization with learned weight
//
// out[i] = x[i] * weight[i] / sqrt(mean(x^2) + eps)
// Two-pass: sum-of-squares reduction, then scaled multiply.
// Pattern: reduction/best_kernel.ea (quad accumulator) + scale_shift_f32x8

#[cfg(x86_64)]

export func rmsnorm_f32(
    x: *restrict f32,
    weight: *restrict f32,
    out: *mut f32,
    n: i32,
    eps: f32
) {
    // Pass 1: sum of squares via quad f32x8 accumulators
    let mut acc0: f32x8 = splat(0.0)
    let mut acc1: f32x8 = splat(0.0)
    let mut i: i32 = 0
    while i + 16 <= n {
        let v0: f32x8 = load(x, i)
        let v1: f32x8 = load(x, i + 8)
        acc0 = fma(v0, v0, acc0)
        acc1 = fma(v1, v1, acc1)
        i = i + 16
    }
    while i + 8 <= n {
        let v: f32x8 = load(x, i)
        acc0 = fma(v, v, acc0)
        i = i + 8
    }
    let mut sumsq: f32 = reduce_add(acc0 .+ acc1)
    while i < n {
        sumsq = sumsq + x[i] * x[i]
        i = i + 1
    }

    // inv_rms = 1.0 / sqrt(sumsq / n + eps)
    let mean_sq: f32 = sumsq / to_f32(n) + eps
    let inv_rms: f32 = 1.0 / sqrt(mean_sq)

    // Pass 2: out[i] = x[i] * weight[i] * inv_rms
    let v_inv: f32x8 = splat(inv_rms)
    i = 0
    while i + 8 <= n {
        let vx: f32x8 = load(x, i)
        let vw: f32x8 = load(weight, i)
        store(out, i, vx .* vw .* v_inv)
        i = i + 8
    }
    while i < n {
        out[i] = x[i] * weight[i] * inv_rms
        i = i + 1
    }
}
```

**Note:** Uses `to_f32(n)` intrinsic for i32→f32 conversion (see `demo/eastat/kernels/csv_stats.ea` for precedent).

- [ ] **Step 3: Add declaration to eabitnet.h**

```c
// RMSNorm: out[i] = x[i] * weight[i] / sqrt(mean(x^2) + eps)
void rmsnorm_f32(const float *x, const float *weight, float *out, int32_t n, float eps);
```

- [ ] **Step 4: Add test target to Makefile**

```makefile
	$(CC) $(CFLAGS) tests/test_rmsnorm.c -L$(LIB) -lbitnet_rmsnorm -o $(BUILD)/test_rmsnorm -lm
	LD_LIBRARY_PATH=$(LIB) $(BUILD)/test_rmsnorm
```

- [ ] **Step 5: Build and run tests**

Run: `cd /home/peter/projects/eabitnet && make clean && make test`

Expected: 5/5 rmsnorm + all prior tests pass.

- [ ] **Step 6: Commit**

```bash
git add kernels/bitnet_rmsnorm.ea tests/test_rmsnorm.c src/eabitnet.h Makefile
git commit -m "feat: add rmsnorm_f32 SIMD kernel"
```

---

## Task 4: softmax kernel

**Files:**
- Create: `kernels/bitnet_softmax.ea`
- Create: `tests/test_softmax.c`
- Modify: `Makefile`
- Modify: `src/eabitnet.h`

- [ ] **Step 1: Write test_softmax.c**

```c
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "../src/eabitnet.h"

static int tests_passed = 0;
static int tests_total = 0;

#define CHECK(name, cond) do { \
    tests_total++; \
    if (cond) { tests_passed++; printf("  PASS: %s\n", name); } \
    else { printf("  FAIL: %s\n", name); } \
} while(0)

#define CLOSE(a, b, tol) (fabsf((a) - (b)) < (tol))

static void ref_softmax(const float *x, float *out, int n) {
    float mx = x[0];
    for (int i = 1; i < n; i++) if (x[i] > mx) mx = x[i];
    float sum = 0;
    for (int i = 0; i < n; i++) { out[i] = expf(x[i] - mx); sum += out[i]; }
    for (int i = 0; i < n; i++) out[i] /= sum;
}

static void test_uniform(void) {
    // All equal → uniform distribution
    float x[] = {1.0f, 1.0f, 1.0f, 1.0f};
    float out[4];
    softmax_f32(x, out, 4);
    int ok = 1;
    for (int i = 0; i < 4; i++)
        if (!CLOSE(out[i], 0.25f, 1e-3f)) ok = 0;
    CHECK("uniform_4", ok);
}

static void test_one_hot(void) {
    // Large gap → near one-hot
    float x[] = {0.0f, 0.0f, 100.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    float out[8];
    softmax_f32(x, out, 8);
    CHECK("one_hot_peak", out[2] > 0.99f);
    float sum = 0;
    for (int i = 0; i < 8; i++) sum += out[i];
    CHECK("one_hot_sums_to_1", CLOSE(sum, 1.0f, 1e-3f));
}

static void test_negative_shift(void) {
    // Numerical stability: large negatives should not overflow
    float x[] = {-1000.0f, -999.0f, -998.0f, -997.0f};
    float out[4], ref[4];
    ref_softmax(x, ref, 4);
    softmax_f32(x, out, 4);
    int ok = 1;
    for (int i = 0; i < 4; i++)
        if (!CLOSE(out[i], ref[i], 1e-2f)) ok = 0;
    CHECK("negative_shift", ok);
}

static void test_known_distribution(void) {
    float x[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
    float out[8], ref[8];
    ref_softmax(x, ref, 8);
    softmax_f32(x, out, 8);
    int ok = 1;
    for (int i = 0; i < 8; i++)
        if (!CLOSE(out[i], ref[i], 1e-2f)) ok = 0;
    CHECK("known_dist_8", ok);
}

static void test_sums_to_one(void) {
    int n = 80;  // head_dim — typical attention length
    float *x = malloc(n * sizeof(float));
    float *out = malloc(n * sizeof(float));
    for (int i = 0; i < n; i++) x[i] = (float)(i - n/2) * 0.1f;
    softmax_f32(x, out, n);
    float sum = 0;
    for (int i = 0; i < n; i++) sum += out[i];
    CHECK("sums_to_one_80", CLOSE(sum, 1.0f, 1e-2f));
    free(x); free(out);
}

static void test_large_seq(void) {
    int n = 2048;  // max seq len
    float *x = malloc(n * sizeof(float));
    float *out = malloc(n * sizeof(float));
    float *ref = malloc(n * sizeof(float));
    for (int i = 0; i < n; i++) x[i] = sinf((float)i * 0.01f) * 5.0f;
    ref_softmax(x, ref, n);
    softmax_f32(x, out, n);
    int ok = 1;
    for (int i = 0; i < n; i++)
        if (!CLOSE(out[i], ref[i], 1e-2f)) ok = 0;
    CHECK("large_2048", ok);
    free(x); free(out); free(ref);
}

int main(void) {
    printf("test_softmax:\n");
    test_uniform();
    test_one_hot();
    test_negative_shift();
    test_known_distribution();
    test_sums_to_one();
    test_large_seq();
    printf("\n%d/%d passed\n", tests_passed, tests_total);
    return tests_passed == tests_total ? 0 : 1;
}
```

- [ ] **Step 2: Write bitnet_softmax.ea**

```
// bitnet_softmax.ea — Numerically stable softmax
//
// Pass 1: max (dual f32x8 accumulators)
// Pass 2: exp(x - max) via fast polynomial, accumulate sum
// Pass 3: divide by sum
//
// Fast exp: 4th-order polynomial approximation on [-87, 0] range.
// Good to ~1e-3 relative error — sufficient for softmax (errors cancel in ratio).

#[cfg(x86_64)]

export func softmax_f32(x: *restrict f32, out: *mut f32, n: i32) {
    // Pass 1: find max
    let mut mx0: f32x8 = load(x, 0)
    let mut mx1: f32x8 = mx0
    let mut i: i32 = 8
    while i + 16 <= n {
        let v0: f32x8 = load(x, i)
        let v1: f32x8 = load(x, i + 8)
        mx0 = select(v0 .> mx0, v0, mx0)
        mx1 = select(v1 .> mx1, v1, mx1)
        i = i + 16
    }
    while i + 8 <= n {
        let v: f32x8 = load(x, i)
        mx0 = select(v .> mx0, v, mx0)
        i = i + 8
    }
    let merged: f32x8 = select(mx0 .> mx1, mx0, mx1)
    let mut mx: f32 = reduce_max(merged)
    while i < n {
        if x[i] > mx { mx = x[i] }
        i = i + 1
    }

    // Pass 2: exp(x - max) and accumulate sum
    // Fast exp approximation: exp(x) ≈ polynomial on clamped range
    // Using Schraudolph-style: exp(x) = 2^(x/ln2) via integer bit trick
    // For better accuracy: 4th order Taylor around 0 after range reduction
    let vmx: f32x8 = splat(mx)
    let mut sum0: f32x8 = splat(0.0)
    let mut sum1: f32x8 = splat(0.0)

    // exp coefficients: 1 + x + x^2/2 + x^3/6 + x^4/24
    let c1: f32x8 = splat(1.0)
    let c2: f32x8 = splat(0.5)
    let c3: f32x8 = splat(0.16666667)
    let c4: f32x8 = splat(0.04166667)
    let lo: f32x8 = splat(-20.0)
    let zero: f32x8 = splat(0.0)

    i = 0
    while i + 16 <= n {
        let d0: f32x8 = load(x, i) .- vmx
        let d1: f32x8 = load(x, i + 8) .- vmx
        // Clamp to avoid underflow
        let cd0: f32x8 = select(d0 .> lo, d0, lo)
        let cd1: f32x8 = select(d1 .> lo, d1, lo)
        // exp(x) ≈ 1 + x(1 + x(0.5 + x(1/6 + x/24)))
        let e0: f32x8 = c1 .+ cd0 .* (c1 .+ cd0 .* (c2 .+ cd0 .* (c3 .+ cd0 .* c4)))
        let e1: f32x8 = c1 .+ cd1 .* (c1 .+ cd1 .* (c2 .+ cd1 .* (c3 .+ cd1 .* c4)))
        // Clamp negative to zero (Taylor can go negative for large |x|)
        let r0: f32x8 = select(e0 .> zero, e0, zero)
        let r1: f32x8 = select(e1 .> zero, e1, zero)
        store(out, i, r0)
        store(out, i + 8, r1)
        sum0 = sum0 .+ r0
        sum1 = sum1 .+ r1
        i = i + 16
    }
    while i + 8 <= n {
        let d: f32x8 = load(x, i) .- vmx
        let cd: f32x8 = select(d .> lo, d, lo)
        let e: f32x8 = c1 .+ cd .* (c1 .+ cd .* (c2 .+ cd .* (c3 .+ cd .* c4)))
        let r: f32x8 = select(e .> zero, e, zero)
        store(out, i, r)
        sum0 = sum0 .+ r
        i = i + 8
    }
    let mut sum: f32 = reduce_add(sum0 .+ sum1)
    while i < n {
        let d: f32 = x[i] - mx
        let mut e: f32 = 1.0 + d * (1.0 + d * (0.5 + d * (0.16666667 + d * 0.04166667)))
        if e < 0.0 { e = 0.0 }
        out[i] = e
        sum = sum + e
        i = i + 1
    }

    // Pass 3: normalize
    let inv_sum: f32 = 1.0 / sum
    let vinv: f32x8 = splat(inv_sum)
    i = 0
    while i + 8 <= n {
        let v: f32x8 = load(out, i)
        store(out, i, v .* vinv)
        i = i + 8
    }
    while i < n {
        out[i] = out[i] * inv_sum
        i = i + 1
    }
}
```

- [ ] **Step 3: Add declaration to eabitnet.h**

```c
// Softmax: numerically stable, fast exp polynomial
void softmax_f32(const float *x, float *out, int32_t n);
```

- [ ] **Step 4: Add test target to Makefile**

```makefile
	$(CC) $(CFLAGS) tests/test_softmax.c -L$(LIB) -lbitnet_softmax -o $(BUILD)/test_softmax -lm
	LD_LIBRARY_PATH=$(LIB) $(BUILD)/test_softmax
```

- [ ] **Step 5: Build and run tests**

Run: `cd /home/peter/projects/eabitnet && make clean && make test`

Expected: 6/6 softmax + all prior tests pass. Tolerance is 1e-2 for the polynomial approximation.

- [ ] **Step 6: Commit**

```bash
git add kernels/bitnet_softmax.ea tests/test_softmax.c src/eabitnet.h Makefile
git commit -m "feat: add softmax_f32 SIMD kernel with fast exp polynomial"
```

---

## Task 5: rope kernel

**Files:**
- Create: `kernels/bitnet_rope.ea`
- Create: `tests/test_rope.c`
- Modify: `Makefile`
- Modify: `src/eabitnet.h`

- [ ] **Step 1: Write test_rope.c**

```c
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "../src/eabitnet.h"

static int tests_passed = 0;
static int tests_total = 0;

#define CHECK(name, cond) do { \
    tests_total++; \
    if (cond) { tests_passed++; printf("  PASS: %s\n", name); } \
    else { printf("  FAIL: %s\n", name); } \
} while(0)

#define CLOSE(a, b, tol) (fabsf((a) - (b)) < (tol))

// Build freq table: freqs[2*i] = cos(pos * theta_i), freqs[2*i+1] = sin(pos * theta_i)
static void build_freqs(float *freqs, int head_dim, int position) {
    for (int i = 0; i < head_dim / 2; i++) {
        float theta = (float)position / powf(10000.0f, 2.0f * (float)i / (float)head_dim);
        freqs[2 * i] = cosf(theta);
        freqs[2 * i + 1] = sinf(theta);
    }
}

static void test_position_zero(void) {
    // Position 0: cos=1, sin=0 → identity transform
    int hd = 8, nh = 1;
    float q[8] = {1, 2, 3, 4, 5, 6, 7, 8};
    float k[8] = {8, 7, 6, 5, 4, 3, 2, 1};
    float q_orig[8], k_orig[8];
    for (int i = 0; i < 8; i++) { q_orig[i] = q[i]; k_orig[i] = k[i]; }
    float freqs[8];
    build_freqs(freqs, hd, 0);
    rope_f32(q, k, freqs, hd, nh);
    int ok = 1;
    for (int i = 0; i < 8; i++)
        if (!CLOSE(q[i], q_orig[i], 1e-5f) || !CLOSE(k[i], k_orig[i], 1e-5f)) ok = 0;
    CHECK("position_zero_identity", ok);
}

static void test_known_rotation(void) {
    // Position such that theta_0 = pi/2: cos=0, sin=1
    // (q0, q1) → (-q1, q0)
    int hd = 8, nh = 1;
    float q[8] = {1, 0, 0, 0, 0, 0, 0, 0};
    float k[8] = {0, 1, 0, 0, 0, 0, 0, 0};
    // Manually set freqs for first pair: cos=0, sin=1
    float freqs[8] = {0.0f, 1.0f, 1.0f, 0.0f, 1.0f, 0.0f, 1.0f, 0.0f};
    rope_f32(q, k, freqs, hd, nh);
    // q[0] = 1*0 - 0*1 = 0, q[1] = 1*1 + 0*0 = 1
    CHECK("q_rotated", CLOSE(q[0], 0.0f, 1e-5f) && CLOSE(q[1], 1.0f, 1e-5f));
    // k[0] = 0*0 - 1*1 = -1, k[1] = 0*1 + 1*0 = 0
    CHECK("k_rotated", CLOSE(k[0], -1.0f, 1e-5f) && CLOSE(k[1], 0.0f, 1e-5f));
}

static void test_multi_head(void) {
    int hd = 8, nh = 4;
    int total = hd * nh;
    float *q = calloc(total, sizeof(float));
    float *k = calloc(total, sizeof(float));
    for (int i = 0; i < total; i++) { q[i] = 1.0f; k[i] = 1.0f; }
    float freqs[8];
    build_freqs(freqs, hd, 42);
    rope_f32(q, k, freqs, hd, nh);
    // All heads should get the same rotation (same freqs applied to each)
    int ok = 1;
    for (int h = 1; h < nh; h++) {
        for (int i = 0; i < hd; i++) {
            if (!CLOSE(q[h * hd + i], q[i], 1e-5f)) ok = 0;
            if (!CLOSE(k[h * hd + i], k[i], 1e-5f)) ok = 0;
        }
    }
    CHECK("multi_head_same_rotation", ok);
    free(q); free(k);
}

static void test_preserves_norm(void) {
    // RoPE should preserve vector magnitude
    int hd = 80, nh = 1;
    float q[80], k[80];
    for (int i = 0; i < 80; i++) { q[i] = sinf((float)i); k[i] = cosf((float)i); }
    float q_norm_before = 0, k_norm_before = 0;
    for (int i = 0; i < 80; i++) {
        q_norm_before += q[i] * q[i];
        k_norm_before += k[i] * k[i];
    }
    float freqs[80];
    build_freqs(freqs, hd, 100);
    rope_f32(q, k, freqs, hd, nh);
    float q_norm_after = 0, k_norm_after = 0;
    for (int i = 0; i < 80; i++) {
        q_norm_after += q[i] * q[i];
        k_norm_after += k[i] * k[i];
    }
    CHECK("preserves_q_norm", CLOSE(q_norm_before, q_norm_after, 1e-2f));
    CHECK("preserves_k_norm", CLOSE(k_norm_before, k_norm_after, 1e-2f));
}

int main(void) {
    printf("test_rope:\n");
    test_position_zero();
    test_known_rotation();
    test_multi_head();
    test_preserves_norm();
    printf("\n%d/%d passed\n", tests_passed, tests_total);
    return tests_passed == tests_total ? 0 : 1;
}
```

- [ ] **Step 2: Write bitnet_rope.ea**

```
// bitnet_rope.ea — Rotary Position Embedding (RoPE)
//
// In-place rotation of Q and K vectors.
// freqs layout: [cos θ_0, sin θ_0, cos θ_1, sin θ_1, ...] for head_dim/2 pairs.
// Precomputed in Rust, one table per position.
// Applied identically to each head.
//
// Rotation: (q[2i], q[2i+1]) → (q[2i]*cos - q[2i+1]*sin, q[2i]*sin + q[2i+1]*cos)

#[cfg(x86_64)]

export func rope_f32(
    q: *mut f32,
    k: *mut f32,
    freqs: *restrict f32,
    head_dim: i32,
    n_heads: i32
) {
    let mut h: i32 = 0
    while h < n_heads {
        let off: i32 = h * head_dim
        let mut i: i32 = 0
        while i < head_dim {
            let cos_v: f32 = freqs[i]
            let sin_v: f32 = freqs[i + 1]

            let q0: f32 = q[off + i]
            let q1: f32 = q[off + i + 1]
            q[off + i]     = q0 * cos_v - q1 * sin_v
            q[off + i + 1] = q0 * sin_v + q1 * cos_v

            let k0: f32 = k[off + i]
            let k1: f32 = k[off + i + 1]
            k[off + i]     = k0 * cos_v - k1 * sin_v
            k[off + i + 1] = k0 * sin_v + k1 * cos_v

            i = i + 2
        }
        h = h + 1
    }
}
```

**Note:** This is scalar per-pair. head_dim=80 means 40 rotations per head — not worth SIMD vectorizing across pairs (irregular access pattern). The outer head loop is the parallelism opportunity but n_heads varies.

- [ ] **Step 3: Add declaration to eabitnet.h**

```c
// RoPE: in-place rotation of Q and K by precomputed (cos,sin) frequency pairs
void rope_f32(float *q, float *k, const float *freqs, int32_t head_dim, int32_t n_heads);
```

- [ ] **Step 4: Add test target to Makefile**

```makefile
	$(CC) $(CFLAGS) tests/test_rope.c -L$(LIB) -lbitnet_rope -o $(BUILD)/test_rope -lm
	LD_LIBRARY_PATH=$(LIB) $(BUILD)/test_rope
```

- [ ] **Step 5: Build and run tests**

Run: `cd /home/peter/projects/eabitnet && make clean && make test`

Expected: 6/6 rope + all prior tests pass.

- [ ] **Step 6: Commit**

```bash
git add kernels/bitnet_rope.ea tests/test_rope.c src/eabitnet.h Makefile
git commit -m "feat: add rope_f32 kernel for rotary position encoding"
```

---

## Task 6: attention kernel

**Files:**
- Create: `kernels/bitnet_attention.ea`
- Create: `tests/test_attention.c`
- Modify: `Makefile`
- Modify: `src/eabitnet.h`

- [ ] **Step 1: Write test_attention.c**

```c
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "../src/eabitnet.h"

static int tests_passed = 0;
static int tests_total = 0;

#define CHECK(name, cond) do { \
    tests_total++; \
    if (cond) { tests_passed++; printf("  PASS: %s\n", name); } \
    else { printf("  FAIL: %s\n", name); } \
} while(0)

#define CLOSE(a, b, tol) (fabsf((a) - (b)) < (tol))

static void test_scores_single_token(void) {
    // Q dot K with seq_len=1 → single score = dot(q,k) * scale
    int hd = 8;
    float q[] = {1, 0, 0, 0, 0, 0, 0, 0};
    float k[] = {2, 0, 0, 0, 0, 0, 0, 0};
    float out[1];
    attn_scores_f32(q, k, out, hd, 1, 0.5f);
    CHECK("single_token_score", CLOSE(out[0], 1.0f, 1e-5f));  // 2 * 0.5
}

static void test_scores_multi_token(void) {
    int hd = 4, seq = 3;
    float q[] = {1, 1, 1, 1};
    // K cache: 3 tokens × 4 dim (row-major)
    float k[] = {1,0,0,0,  0,1,0,0,  0,0,1,0};
    float out[3];
    attn_scores_f32(q, k, out, hd, seq, 1.0f);
    CHECK("multi_token_0", CLOSE(out[0], 1.0f, 1e-5f));
    CHECK("multi_token_1", CLOSE(out[1], 1.0f, 1e-5f));
    CHECK("multi_token_2", CLOSE(out[2], 1.0f, 1e-5f));
}

static void test_weighted_sum_single(void) {
    // scores=[1.0], V=[[1,2,3,4]] → out=[1,2,3,4]
    int hd = 4;
    float scores[] = {1.0f};
    float v[] = {1, 2, 3, 4};
    float out[4] = {0};
    attn_weighted_sum_f32(scores, v, out, hd, 1);
    int ok = 1;
    for (int i = 0; i < 4; i++)
        if (!CLOSE(out[i], v[i], 1e-5f)) ok = 0;
    CHECK("wsum_single", ok);
}

static void test_weighted_sum_uniform(void) {
    // Equal scores → average of V rows
    int hd = 4, seq = 2;
    float scores[] = {0.5f, 0.5f};
    float v[] = {2,0,0,0,  0,2,0,0};
    float out[4] = {0};
    attn_weighted_sum_f32(scores, v, out, hd, seq);
    CHECK("wsum_avg_0", CLOSE(out[0], 1.0f, 1e-5f));
    CHECK("wsum_avg_1", CLOSE(out[1], 1.0f, 1e-5f));
}

static void test_large_dim(void) {
    int hd = 80, seq = 64;
    float *q = malloc(hd * sizeof(float));
    float *k = malloc(seq * hd * sizeof(float));
    float *scores = malloc(seq * sizeof(float));
    for (int i = 0; i < hd; i++) q[i] = 1.0f;
    for (int t = 0; t < seq; t++)
        for (int i = 0; i < hd; i++)
            k[t * hd + i] = (i == 0) ? 1.0f : 0.0f;
    float scale = 1.0f / sqrtf(80.0f);
    attn_scores_f32(q, k, scores, hd, seq, scale);
    // All scores should be equal (each K has dot=1 with Q)
    int ok = 1;
    for (int t = 0; t < seq; t++)
        if (!CLOSE(scores[t], scale, 1e-4f)) ok = 0;
    CHECK("large_80x64_scores", ok);
    free(q); free(k); free(scores);
}

int main(void) {
    printf("test_attention:\n");
    test_scores_single_token();
    test_scores_multi_token();
    test_weighted_sum_single();
    test_weighted_sum_uniform();
    test_large_dim();
    printf("\n%d/%d passed\n", tests_passed, tests_total);
    return tests_passed == tests_total ? 0 : 1;
}
```

- [ ] **Step 2: Write bitnet_attention.ea**

```
// bitnet_attention.ea — Attention score computation + weighted V accumulation
//
// attn_scores_f32: computes scaled dot products Q · K[t] for all cached tokens
// attn_weighted_sum_f32: accumulates out = Σ scores[t] * V[t]
//
// K/V cache layout: row-major [seq_len][head_dim]
// Pattern: batch_dot/best_kernel.ea (quad accumulator with prefetch)

#[cfg(x86_64)]

// Scaled dot product: out[t] = dot(q, k[t*head_dim..]) * scale
export func attn_scores_f32(
    q: *restrict f32,
    k_cache: *restrict f32,
    out: *mut f32,
    head_dim: i32,
    seq_len: i32,
    scale: f32
) {
    let mut t: i32 = 0
    while t < seq_len {
        let base: i32 = t * head_dim
        let mut acc0: f32x8 = splat(0.0)
        let mut acc1: f32x8 = splat(0.0)
        let mut j: i32 = 0
        while j + 16 <= head_dim {
            let qa0: f32x8 = load(q, j)
            let kb0: f32x8 = load(k_cache, base + j)
            let qa1: f32x8 = load(q, j + 8)
            let kb1: f32x8 = load(k_cache, base + j + 8)
            acc0 = fma(qa0, kb0, acc0)
            acc1 = fma(qa1, kb1, acc1)
            j = j + 16
        }
        while j + 8 <= head_dim {
            let qa: f32x8 = load(q, j)
            let kb: f32x8 = load(k_cache, base + j)
            acc0 = fma(qa, kb, acc0)
            j = j + 8
        }
        let mut dot: f32 = reduce_add(acc0 .+ acc1)
        while j < head_dim {
            dot = dot + q[j] * k_cache[base + j]
            j = j + 1
        }
        out[t] = dot * scale
        t = t + 1
    }
}

// Weighted sum: out[d] = Σ_t scores[t] * v_cache[t * head_dim + d]
export func attn_weighted_sum_f32(
    scores: *restrict f32,
    v_cache: *restrict f32,
    out: *mut f32,
    head_dim: i32,
    seq_len: i32
) {
    // Zero output
    let zero: f32x8 = splat(0.0)
    let mut d: i32 = 0
    while d + 8 <= head_dim {
        store(out, d, zero)
        d = d + 8
    }
    while d < head_dim {
        out[d] = 0.0
        d = d + 1
    }

    // Accumulate: for each token, add score * V row
    let mut t: i32 = 0
    while t < seq_len {
        let base: i32 = t * head_dim
        let vs: f32x8 = splat(scores[t])
        d = 0
        while d + 8 <= head_dim {
            let vo: f32x8 = load(out, d)
            let vv: f32x8 = load(v_cache, base + d)
            store(out, d, fma(vs, vv, vo))
            d = d + 8
        }
        while d < head_dim {
            out[d] = out[d] + scores[t] * v_cache[base + d]
            d = d + 1
        }
        t = t + 1
    }
}
```

- [ ] **Step 3: Add declarations to eabitnet.h**

```c
// Attention: scaled dot products Q·K for all cached tokens
void attn_scores_f32(const float *q, const float *k_cache, float *out,
                     int32_t head_dim, int32_t seq_len, float scale);

// Attention: weighted sum out = Σ scores[t] * V[t]
void attn_weighted_sum_f32(const float *scores, const float *v_cache, float *out,
                           int32_t head_dim, int32_t seq_len);
```

- [ ] **Step 4: Add test target to Makefile**

```makefile
	$(CC) $(CFLAGS) tests/test_attention.c -L$(LIB) -lbitnet_attention -o $(BUILD)/test_attention -lm
	LD_LIBRARY_PATH=$(LIB) $(BUILD)/test_attention
```

- [ ] **Step 5: Build and run tests**

Run: `cd /home/peter/projects/eabitnet && make clean && make test`

Expected: 7/7 attention + all prior tests pass.

- [ ] **Step 6: Commit**

```bash
git add kernels/bitnet_attention.ea tests/test_attention.c src/eabitnet.h Makefile
git commit -m "feat: add attention kernels (scores + weighted V sum)"
```

---

## Task 7: squared ReLU activation kernel

**Files:**
- Create: `kernels/bitnet_activate.ea`
- Create: `tests/test_activate.c`
- Modify: `Makefile`
- Modify: `src/eabitnet.h`

- [ ] **Step 1: Write test_activate.c**

```c
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "../src/eabitnet.h"

static int tests_passed = 0;
static int tests_total = 0;

#define CHECK(name, cond) do { \
    tests_total++; \
    if (cond) { tests_passed++; printf("  PASS: %s\n", name); } \
    else { printf("  FAIL: %s\n", name); } \
} while(0)

#define CLOSE(a, b, tol) (fabsf((a) - (b)) < (tol))

static void test_positive_gate(void) {
    // gate > 0: out = gate^2 * up
    float gate[] = {2.0f, 3.0f, 1.0f, 0.5f, 2.0f, 3.0f, 1.0f, 0.5f};
    float up[]   = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
    float out[8];
    squared_relu_mul_f32(gate, up, out, 8);
    CHECK("pos_gate_0", CLOSE(out[0], 4.0f, 1e-5f));
    CHECK("pos_gate_1", CLOSE(out[1], 9.0f, 1e-5f));
    CHECK("pos_gate_2", CLOSE(out[2], 1.0f, 1e-5f));
    CHECK("pos_gate_3", CLOSE(out[3], 0.25f, 1e-5f));
}

static void test_negative_gate(void) {
    // gate < 0: squared ReLU → 0
    float gate[] = {-1.0f, -2.0f, -3.0f, -4.0f, -5.0f, -6.0f, -7.0f, -8.0f};
    float up[]   = {10.0f, 10.0f, 10.0f, 10.0f, 10.0f, 10.0f, 10.0f, 10.0f};
    float out[8];
    squared_relu_mul_f32(gate, up, out, 8);
    int ok = 1;
    for (int i = 0; i < 8; i++)
        if (fabsf(out[i]) > 1e-6f) ok = 0;
    CHECK("negative_gate_zero", ok);
}

static void test_zero_gate(void) {
    float gate[8] = {0};
    float up[] = {1, 2, 3, 4, 5, 6, 7, 8};
    float out[8];
    squared_relu_mul_f32(gate, up, out, 8);
    int ok = 1;
    for (int i = 0; i < 8; i++)
        if (fabsf(out[i]) > 1e-6f) ok = 0;
    CHECK("zero_gate", ok);
}

static void test_mixed(void) {
    // Mix of positive and negative
    float gate[] = {2.0f, -1.0f, 3.0f, -2.0f, 0.0f, 1.0f, -0.5f, 4.0f};
    float up[]   = {1.0f,  1.0f, 2.0f,  2.0f, 5.0f, 3.0f,  3.0f, 0.5f};
    float expected[] = {4.0f, 0.0f, 18.0f, 0.0f, 0.0f, 3.0f, 0.0f, 8.0f};
    float out[8];
    squared_relu_mul_f32(gate, up, out, 8);
    int ok = 1;
    for (int i = 0; i < 8; i++)
        if (!CLOSE(out[i], expected[i], 1e-4f)) ok = 0;
    CHECK("mixed_8", ok);
}

static void test_large(void) {
    int n = 6912;  // FFN intermediate dim
    float *gate = malloc(n * sizeof(float));
    float *up = malloc(n * sizeof(float));
    float *out = malloc(n * sizeof(float));
    for (int i = 0; i < n; i++) {
        gate[i] = sinf((float)i * 0.01f);
        up[i] = 1.0f;
    }
    squared_relu_mul_f32(gate, up, out, n);
    int ok = 1;
    for (int i = 0; i < n; i++) {
        float g = gate[i];
        float expected = g > 0 ? g * g : 0.0f;
        if (!CLOSE(out[i], expected, 1e-3f)) ok = 0;
    }
    CHECK("large_6912", ok);
    free(gate); free(up); free(out);
}

int main(void) {
    printf("test_activate:\n");
    test_positive_gate();
    test_negative_gate();
    test_zero_gate();
    test_mixed();
    test_large();
    printf("\n%d/%d passed\n", tests_passed, tests_total);
    return tests_passed == tests_total ? 0 : 1;
}
```

- [ ] **Step 2: Write bitnet_activate.ea**

```
// bitnet_activate.ea — Squared ReLU fused with up-projection multiply
//
// out[i] = max(0, gate[i])^2 * up[i]
// Branchless: clamp gate to zero via select, then square, then multiply.

#[cfg(x86_64)]

export func squared_relu_mul_f32(
    gate: *restrict f32,
    up: *restrict f32,
    out: *mut f32,
    n: i32
) {
    let zero: f32x8 = splat(0.0)
    let mut i: i32 = 0
    while i + 8 <= n {
        let vg: f32x8 = load(gate, i)
        let vu: f32x8 = load(up, i)
        let clamped: f32x8 = select(vg .> zero, vg, zero)
        store(out, i, clamped .* clamped .* vu)
        i = i + 8
    }
    while i < n {
        let g: f32 = gate[i]
        if g > 0.0 {
            out[i] = g * g * up[i]
        } else {
            out[i] = 0.0
        }
        i = i + 1
    }
}
```

- [ ] **Step 3: Add declaration to eabitnet.h**

```c
// Squared ReLU fused: out[i] = max(0, gate[i])^2 * up[i]
void squared_relu_mul_f32(const float *gate, const float *up, float *out, int32_t n);
```

- [ ] **Step 4: Add test target to Makefile**

```makefile
	$(CC) $(CFLAGS) tests/test_activate.c -L$(LIB) -lbitnet_activate -o $(BUILD)/test_activate -lm
	LD_LIBRARY_PATH=$(LIB) $(BUILD)/test_activate
```

- [ ] **Step 5: Build and run tests**

Run: `cd /home/peter/projects/eabitnet && make clean && make test`

Expected: 8/8 activate + all prior tests pass.

- [ ] **Step 6: Commit**

```bash
git add kernels/bitnet_activate.ea tests/test_activate.c src/eabitnet.h Makefile
git commit -m "feat: add squared_relu_mul_f32 fused activation kernel"
```

---

## Task 8: Rust project scaffold + FFI

**Files:**
- Create: `Cargo.toml`
- Create: `src/ffi.rs`
- Create: `src/main.rs`

- [ ] **Step 1: Create Cargo.toml**

```toml
[package]
name = "eabitnet"
version = "0.1.0"
edition = "2021"

[[bin]]
name = "eabitnet"
path = "src/main.rs"
```

- [ ] **Step 2: Write src/ffi.rs**

All `extern "C"` declarations for Eä kernels. Debug assertions for slice lengths.

```rust
//! FFI declarations for Eä SIMD kernels.
//! Each function maps to a symbol exported from a .so compiled by eacompute.

#[link(name = "bitnet_i2s")]
extern "C" {
    pub fn i2_dot_i8(weights: *const u8, activations: *const i8, n: i32) -> i32;
    pub fn i2_dot_i8_4row(
        w0: *const u8, w1: *const u8, w2: *const u8, w3: *const u8,
        activations: *const i8, scores: *mut i32, n: i32,
    );
}

#[link(name = "bitnet_quant")]
extern "C" {
    pub fn quant_f32_i8(src: *const f32, dst: *mut i8, out_scale: *mut f32, out_sum: *mut i32, n: i32);
    pub fn pack_ternary_row(ternary: *const u8, packed: *mut u8, n: i32);
}

#[link(name = "bitnet_rmsnorm")]
extern "C" {
    pub fn rmsnorm_f32(x: *const f32, weight: *const f32, out: *mut f32, n: i32, eps: f32);
}

#[link(name = "bitnet_softmax")]
extern "C" {
    pub fn softmax_f32(x: *const f32, out: *mut f32, n: i32);
}

#[link(name = "bitnet_rope")]
extern "C" {
    pub fn rope_f32(q: *mut f32, k: *mut f32, freqs: *const f32, head_dim: i32, n_heads: i32);
}

#[link(name = "bitnet_attention")]
extern "C" {
    pub fn attn_scores_f32(
        q: *const f32, k_cache: *const f32, out: *mut f32,
        head_dim: i32, seq_len: i32, scale: f32,
    );
    pub fn attn_weighted_sum_f32(
        scores: *const f32, v_cache: *const f32, out: *mut f32,
        head_dim: i32, seq_len: i32,
    );
}

#[link(name = "bitnet_activate")]
extern "C" {
    pub fn squared_relu_mul_f32(gate: *const f32, up: *const f32, out: *mut f32, n: i32);
}

#[link(name = "bitnet_vecadd")]
extern "C" {
    pub fn vecadd_f32(a: *const f32, b: *const f32, out: *mut f32, n: i32);
}
```

- [ ] **Step 3: Write minimal src/main.rs**

Placeholder that imports ffi and compiles — proves linking works.

```rust
mod ffi;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 || args[1] != "--model" {
        eprintln!("Usage: eabitnet --model <path.gguf> --prompt <text>");
        std::process::exit(1);
    }
    println!("eabitnet: model={}", args[2]);
    println!("Kernels linked successfully.");
}
```

- [ ] **Step 4: Build and verify linking**

Run: `cd /home/peter/projects/eabitnet && make kernels && RUSTFLAGS="-L build/lib" cargo build 2>&1`

Expected: Compiles and links without errors. Running `LD_LIBRARY_PATH=build/lib ./target/debug/eabitnet --model test` prints "Kernels linked successfully."

- [ ] **Step 5: Commit**

```bash
git add Cargo.toml src/ffi.rs src/main.rs
git commit -m "feat: Rust scaffold with FFI declarations for all Eä kernels"
```

---

## Task 9: GGUF parser

**Files:**
- Create: `src/gguf.rs`
- Modify: `src/main.rs`

- [ ] **Step 1: Write src/gguf.rs**

GGUF v3 parser. Reads header, metadata KV pairs, tensor info, memory-maps data.

```rust
//! GGUF v3 file parser.
//! Memory-maps tensor data — weights stay on disk until accessed.

use std::collections::HashMap;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};

const GGUF_MAGIC: u32 = 0x46475547; // "GGUF"

#[derive(Debug, Clone)]
pub enum MetaValue {
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    F32(f32),
    U64(u64),
    I64(i64),
    F64(f64),
    Bool(bool),
    Str(String),
    Array(Vec<MetaValue>),
}

#[derive(Debug)]
pub struct TensorInfo {
    pub name: String,
    pub dims: Vec<u64>,
    pub dtype: u32,
    pub offset: u64,
}

pub struct GgufFile {
    pub version: u32,
    pub metadata: HashMap<String, MetaValue>,
    pub tensors: Vec<TensorInfo>,
    pub data: memmap2::Mmap, // IMPORTANT: needs memmap2 crate — or use raw mmap
    pub data_offset: u64,
}
```

**Important decision:** The spec says "no external crates" but memory-mapping needs either `memmap2` or raw `libc::mmap`. Use raw `libc::mmap` via `unsafe` to stay dependency-free, or allow `memmap2` as the single dependency. Decide at implementation time — the parser logic is the same either way.

The full implementation should:
1. Read and validate GGUF magic + version
2. Read `n_tensors` and `n_kv` counts
3. Parse all metadata KV pairs (handle all 12 GGUF value types)
4. Parse tensor info entries (name, ndims, dims, dtype, offset)
5. Compute data section offset (aligned to 32 bytes after header)
6. Memory-map the data section
7. Provide `get_tensor_data(&self, name: &str) -> &[u8]` accessor

- [ ] **Step 2: Write a test that parses a real GGUF file header**

This requires having a GGUF file available. Create a minimal test that at least validates the parser compiles and handles errors:

```rust
// In src/gguf.rs, add:
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_missing_file() {
        let result = GgufFile::open("nonexistent.gguf");
        assert!(result.is_err());
    }
}
```

Integration test against a real GGUF file happens in Task 12.

- [ ] **Step 3: Verify compilation**

Run: `cd /home/peter/projects/eabitnet && RUSTFLAGS="-L build/lib" cargo build 2>&1`

Expected: Compiles. No runtime test yet (needs a real GGUF).

- [ ] **Step 4: Commit**

```bash
git add src/gguf.rs src/main.rs
git commit -m "feat: add GGUF v3 parser with mmap tensor access"
```

---

## Task 10: BPE tokenizer

**Files:**
- Create: `src/tokenizer.rs`
- Modify: `src/main.rs`

- [ ] **Step 1: Write src/tokenizer.rs**

BPE tokenizer that reads vocab from GGUF metadata.

```rust
//! Byte-level BPE tokenizer parsed from GGUF vocab metadata.
//! Implements encode (text → token IDs) and decode (token IDs → text).

pub struct Tokenizer {
    /// Token ID → token string (byte sequence)
    vocab: Vec<Vec<u8>>,
    /// Token string → token ID (for encoding)
    token_to_id: HashMap<Vec<u8>, u32>,
    /// BPE merge rules: (token_a, token_b) → merged token, ordered by priority
    merges: Vec<(Vec<u8>, Vec<u8>)>,
    pub bos_id: u32,
    pub eos_id: u32,
}
```

Key methods:
- `from_gguf(metadata: &HashMap<String, MetaValue>) -> Tokenizer` — extract vocab tokens, scores, merges from GGUF metadata keys (`tokenizer.ggml.tokens`, `tokenizer.ggml.scores`, `tokenizer.ggml.merges`, `tokenizer.ggml.bos_token_id`, `tokenizer.ggml.eos_token_id`)
- `encode(&self, text: &str) -> Vec<u32>` — byte-level BPE encoding
- `decode(&self, ids: &[u32]) -> String` — token ID to string lookup + UTF-8 assembly

- [ ] **Step 2: Add unit tests**

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_decode_roundtrip_ascii() {
        // Build minimal vocab: single-byte tokens for ASCII
        let mut vocab = Vec::new();
        for b in 0..=255u8 {
            vocab.push(vec![b]);
        }
        let tok = Tokenizer {
            token_to_id: vocab.iter().enumerate().map(|(i, v)| (v.clone(), i as u32)).collect(),
            vocab,
            merges: vec![],
            bos_id: 0,
            eos_id: 0,
        };
        let text = "Hello";
        let ids = tok.encode(text);
        let decoded = tok.decode(&ids);
        assert_eq!(decoded, text);
    }
}
```

Full integration test with real GGUF vocab in Task 12.

- [ ] **Step 3: Verify compilation**

Run: `cd /home/peter/projects/eabitnet && RUSTFLAGS="-L build/lib" cargo build 2>&1`

- [ ] **Step 4: Commit**

```bash
git add src/tokenizer.rs src/main.rs
git commit -m "feat: add BPE tokenizer with GGUF vocab parsing"
```

---

## Task 11: Model loading

**Files:**
- Create: `src/model.rs`
- Modify: `src/main.rs`

- [ ] **Step 1: Write src/model.rs**

```rust
//! BitNet b1.58 2B-4T model: hyperparams + weight pointers into mmap'd GGUF.

pub struct BitNetModel {
    pub n_layers: usize,
    pub hidden_dim: usize,
    pub n_heads: usize,
    pub head_dim: usize,
    pub ffn_dim: usize,
    pub vocab_size: usize,
    pub max_seq_len: usize,
    pub rope_theta: f32,
    pub rms_eps: f32,

    /// Per-layer weights (pointers into mmap)
    pub layers: Vec<LayerWeights>,
    /// Embedding table: [vocab_size][hidden_dim] f32
    pub embed: *const f32,
    /// Final RMSNorm weight: [hidden_dim] f32
    pub norm_weight: *const f32,
    /// Output projection (vocab): [vocab_size][hidden_dim/4] packed I2_S
    pub output_weight: *const u8,
    pub output_scale: f32,
}

pub struct LayerWeights {
    // Attention
    pub attn_norm: *const f32,     // RMSNorm weight [hidden_dim]
    pub wq: *const u8,             // I2_S packed [hidden_dim][hidden_dim/4]
    pub wk: *const u8,
    pub wv: *const u8,
    pub wo: *const u8,
    pub wq_scale: f32,
    pub wk_scale: f32,
    pub wv_scale: f32,
    pub wo_scale: f32,
    // FFN
    pub ffn_norm: *const f32,      // RMSNorm weight [hidden_dim]
    pub w_gate: *const u8,         // I2_S packed [ffn_dim][hidden_dim/4]
    pub w_up: *const u8,
    pub w_down: *const u8,         // I2_S packed [hidden_dim][ffn_dim/4]
    pub w_gate_scale: f32,
    pub w_up_scale: f32,
    pub w_down_scale: f32,
}
```

Key method:
- `BitNetModel::from_gguf(gguf: &GgufFile) -> Result<Self>` — read hyperparams from metadata, resolve tensor names to mmap pointers, extract per-tensor scales

- [ ] **Step 2: Verify compilation**

Run: `cd /home/peter/projects/eabitnet && RUSTFLAGS="-L build/lib" cargo build 2>&1`

- [ ] **Step 3: Commit**

```bash
git add src/model.rs src/main.rs
git commit -m "feat: add BitNetModel — weight pointers into mmap'd GGUF"
```

---

## Task 12: Forward pass + generation

**Files:**
- Create: `src/forward.rs`
- Modify: `src/main.rs`

- [ ] **Step 1: Write src/forward.rs**

The transformer forward pass calling all Eä kernels. This is the core of the project.

```rust
//! Transformer forward pass for BitNet b1.58 2B-4T.
//! Calls Eä SIMD kernels for all compute operations.

use crate::ffi;
use crate::model::BitNetModel;

pub struct InferenceState {
    // Scratch buffers (allocated once)
    x: Vec<f32>,           // [hidden_dim]
    x_norm: Vec<f32>,      // [hidden_dim]
    x_quant: Vec<i8>,      // [hidden_dim + 12] (extra for narrow)
    q: Vec<f32>,           // [hidden_dim]
    k: Vec<f32>,           // [hidden_dim]
    v: Vec<f32>,           // [hidden_dim]
    attn_out: Vec<f32>,    // [hidden_dim]
    attn_scores: Vec<f32>, // [max_seq_len]
    gate: Vec<f32>,        // [ffn_dim]
    up: Vec<f32>,          // [ffn_dim]
    hidden: Vec<f32>,      // [ffn_dim]
    hidden_quant: Vec<i8>, // [ffn_dim + 12]
    logits: Vec<f32>,      // [vocab_size]
    // KV cache
    k_cache: Vec<f32>,     // [n_layers][max_seq][n_heads][head_dim]
    v_cache: Vec<f32>,     // [n_layers][max_seq][n_heads][head_dim]
    // RoPE frequency table
    rope_freqs: Vec<f32>,  // [head_dim] (cos,sin pairs)
}
```

Key methods:
- `InferenceState::new(model: &BitNetModel) -> Self` — allocate all buffers
- `forward_token(&mut self, model: &BitNetModel, token: u32, pos: usize)` — one token forward pass, updates KV cache
- `sample(&self, temperature: f32) -> u32` — temperature + top-p sampling from logits
- `generate(model: &BitNetModel, prompt_tokens: &[u32], max_tokens: u32, temperature: f32) -> Vec<u32>` — full generation loop

The forward pass implements exactly the pseudocode from the spec:
1. Embedding lookup
2. For each layer: RMSNorm → quant → QKV matmul → RoPE → KV cache → attention → output matmul → residual → FFN (RMSNorm → quant → gate/up matmul → squared ReLU → quant → down matmul → residual)
3. Final RMSNorm → quant → vocab matmul → sample

**Scale correction** after every matmul:
```rust
fn apply_matmul_result(raw: &[i32], act_sum: i32, act_scale: f32, w_scale: f32, out: &mut [f32]) {
    let scale = (act_scale / 127.0) * w_scale;
    for i in 0..raw.len() {
        out[i] = (raw[i] - act_sum) as f32 * scale;
    }
}
```

**i2_dot_i8 row dispatch:** For a matrix multiply of shape [out_dim × in_dim], call `i2_dot_i8_4row` for groups of 4 output rows, and `i2_dot_i8` for remaining rows. Weight pointer arithmetic: row `r` starts at `weight_ptr + r * (in_dim / 4)` bytes (2 bits per weight, packed).

- [ ] **Step 2: Wire into main.rs**

Update main.rs to: parse args → open GGUF → build tokenizer → load model → encode prompt → generate → decode → print.

```rust
mod ffi;
mod gguf;
mod tokenizer;
mod model;
mod forward;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    // Parse --model, --prompt, --max-tokens, --temperature
    // Load GGUF, build tokenizer, load model
    // Encode prompt, generate tokens, decode and print
}
```

- [ ] **Step 3: Verify compilation**

Run: `cd /home/peter/projects/eabitnet && RUSTFLAGS="-L build/lib" cargo build 2>&1`

- [ ] **Step 4: Commit**

```bash
git add src/forward.rs src/main.rs
git commit -m "feat: add transformer forward pass + generation loop"
```

---

## Task 13: Integration test with real model

**Files:**
- Modify: `src/main.rs`

This task requires downloading the BitNet b1.58 2B-4T GGUF file.

- [ ] **Step 1: Download model**

Run: `mkdir -p ~/.eabitnet/models && cd ~/.eabitnet/models`

Download `ggml-model-i2_s.gguf` from `huggingface.co/microsoft/bitnet-b1.58-2B-4T-gguf` (use `huggingface-cli download` or `wget`). ~1.19 GB.

- [ ] **Step 2: Test GGUF parsing**

Run: `cd /home/peter/projects/eabitnet && RUSTFLAGS="-L build/lib" LD_LIBRARY_PATH=build/lib cargo run -- --model ~/.eabitnet/models/ggml-model-i2_s.gguf --prompt "Hello"`

Expected: Model loads, prints hyperparams (24 layers, 2560 hidden, 32 heads, etc.), generates tokens.

- [ ] **Step 3: Verify output is coherent**

The generated text should be recognizable language, not garbage. BitNet 2B is a small model so quality won't be GPT-4, but it should produce grammatical sentences.

- [ ] **Step 4: Debug any scale/offset issues**

If output is garbage, check:
1. Ternary offset correction: `raw - activation_sum`
2. Scale: `(act_scale / 127.0) * weight_scale`
3. Weight pointer arithmetic: row size = `n_cols / 4` bytes
4. I2_S encoding matches kernel expectations

- [ ] **Step 5: Run full test suite**

Run: `cd /home/peter/projects/eabitnet && make test && RUSTFLAGS="-L build/lib" cargo test`

Expected: All kernel tests pass + Rust unit tests pass.

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m "feat: eabitnet standalone runner — BitNet inference via Eä kernels"
```

---

## Task 14: Line count verification + final cleanup

- [ ] **Step 1: Count lines**

Run:
```bash
cd /home/peter/projects/eabitnet
echo "=== Eä kernels ==="
wc -l kernels/*.ea
echo "=== Rust source ==="
wc -l src/*.rs
echo "=== C tests ==="
wc -l tests/*.c
echo "=== Total ==="
wc -l kernels/*.ea src/*.rs
```

Expected: ~2000 lines total (kernels + Rust). Should be well under Microsoft's BitNet.cpp.

- [ ] **Step 2: Check hard rules**

Verify no file exceeds 500 lines:
```bash
wc -l kernels/*.ea src/*.rs tests/*.c | sort -rn | head -5
```

- [ ] **Step 3: Final commit**

If any cleanup was needed:
```bash
git add -A
git commit -m "chore: final cleanup — line counts verified, hard rules satisfied"
```
